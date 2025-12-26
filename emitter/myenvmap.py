# from https://github.com/gerwang/diff-sdf/blob/main/python/emitters/envmap.py

import io
from typing import List, Tuple

import drjit as dr
import mitsuba as mi
import numpy as np


def dr_sphdir(theta, phi):
    sin_theta, cos_theta = dr.sincos(theta)
    sin_phi, cos_phi = dr.sincos(phi)

    return mi.Vector3f(
        cos_phi * sin_theta,
        sin_phi * sin_theta,
        cos_theta
    )

def indent(name: str):
    lines = name.split('\n')
    lines[1:] = [f'  {x}' for x in lines[1:]]
    return '\n'.join(lines)

class MyEnvironmentMapEmitter(mi.Emitter):
    PLUGIN_NAME = "myenvmap"

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # Until `set_scene` is called, we have no information
        #   about the scene and default to the unit bounding sphere.
        self.m_bsphere: mi.ScalarBoundingSphere3f = mi.ScalarBoundingSphere3f(mi.ScalarPoint3f(0.), 1.)

        self.m_filename = ''
        if props.has_property('bitmap'):
            # Creates a Bitmap texture directly from an existing Bitmap object
            if props.has_property('filename'):
                raise ValueError("Cannot specify both \"bitmap\" and \"filename\".")
            bitmap = props.get('bitmap')
        else:
            fs = mi.Thread.thread().file_resolver()
            file_path = fs.resolve(props.get('filename'))
            self.m_filename = str(file_path.filename())
            bitmap = mi.Bitmap(file_path)

        if bitmap.width() < 2 or bitmap.height() < 3:
            raise ValueError('{}: the environment map resolution must be at least '
                             '2x3 pixels'.format('<Bitmap>' if len(self.m_filename) == 0 else self.m_filename))

        pixel_format = mi.Bitmap.PixelFormat.RGB
        if mi.is_spectral:
            pixel_format = mi.Bitmap.PixelFormat.RGBA
        bitmap = bitmap.convert(pixel_format, mi.Struct.Type.Float32, srgb_gamma=False)

        # Allocate a larger image including an extra column to
        #            account for the periodic boundary
        res = mi.ScalarVector2u(bitmap.width() + 1, bitmap.height())
        bitmap_2 = mi.Bitmap(pixel_format=bitmap.pixel_format(),
                             component_format=bitmap.component_format(), size=res)

        # Luminance image used for importance sampling
        luminance = np.empty(np.asarray(res)[::-1], dtype=np.float32)
        # in_arr = np.asarray(bitmap)
        # out_arr = np.asarray(bitmap_2)
        # lum_arr = luminance
        # Convert bitmap to a C-contiguous float32 array with shape (channels, N)
        in_arr = np.asarray(bitmap)
        flat = in_arr.reshape(-1, in_arr.shape[-1]).astype(np.float32)
        # mitsuba.Vector3f expects shape (3, *), i.e. (channels, N), C-order
        flat = np.ascontiguousarray(flat.T)
        rgb = mi.Vector3f(flat)
        lum = mi.luminance(rgb)

        theta_scale = 1. / (bitmap.size().y - 1) * dr.pi

        #         MIS Compensation: Optimizing Sampling Techniques in Multiple
        #            Importance Sampling" Ondrej Karlik, Martin Sik, Petr Vivoda, Tomas
        #            Skrivan, and Jaroslav Krivanek. SIGGRAPH Asia 2019 */
        luminance_offset = 0.
        if props.get('mis_compensation', False):
            min_lum = dr.min(lum)[0]
            lum_accum_d = dr.sum(lum)[0]
            luminance_offset = lum_accum_d / dr.prod(bitmap.size())
            #             Be wary of constant environment maps: average and minimum
            #                should be sufficiently different
            if luminance_offset - min_lum <= 0.01 * luminance_offset:
                luminance_offset = 0.  # disable

        sin_theta_np = np.sin(np.arange(bitmap.size().y) * theta_scale)
        coeff = rgb
        lum_shifted = dr.maximum(lum - luminance_offset, 0.0)
        luminance[:, :-1] = np.asarray(lum_shifted).reshape(bitmap.height(), bitmap.width()) * sin_theta_np[:, None]
        luminance[:, -1] = luminance[:, 0]
        bitmap_2_ref = np.asarray(bitmap_2).reshape(bitmap.height(), bitmap.width() + 1, -1)
        bitmap_2_ref[:, :-1] = np.asarray(coeff).reshape(bitmap.height(), bitmap.width(), -1)
        bitmap_2_ref[:, -1] = bitmap_2_ref[:, 0]

        self.m_data = mi.TensorXf(np.asarray(bitmap_2))

        self.m_scale = props.get('scale', 1.)
        self.m_warp = mi.Hierarchical2D0(luminance)
        self.m_d65 = mi.Texture.D65(1.0)
        self.m_flags = mi.EmitterFlags.Infinite | mi.EmitterFlags.SpatiallyVarying
        self.m_to_world = mi.Transform4f()

    def world_transform(self) -> mi.Transform4f:
        return self.m_to_world @ super().world_transform()

    def traverse(self, callback: mi.TraversalCallback) -> None:
        super().traverse(callback)
        callback.put('scale', self.m_scale, flags=mi.ParamFlags.Differentiable)
        callback.put('data', self.m_data, flags=mi.ParamFlags.Differentiable | mi.ParamFlags.Discontinuous)
        callback.put('to_world', self.m_to_world, flags=mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys: List[str] = []) -> None:
        if len(keys) == 0 or 'data' in keys:
            res = mi.ScalarVector2u(self.m_data.shape[1], self.m_data.shape[0])
            if dr.is_jit_v(mi.Float):
                row_index = dr.arange(mi.UInt32, res.y) * res.x
                v0 = dr.gather(dr.cuda.ad.Array3f, self.m_data.array, row_index)
                v1 = dr.gather(dr.cuda.ad.Array3f, self.m_data.array, row_index + (res.x - 1))
                v01 = .5 * (v0 + v1)
                dr.scatter(self.m_data.array, v01, row_index)
                dr.scatter(self.m_data.array, v01, row_index + (res.x - 1))

            data = dr.migrate(self.m_data.array, dr.AllocType.Host)  # convert to cpu

            if dr.is_jit_v(mi.Float):
                dr.sync_thread()

            luminance = np.empty((res.y, res.x), dtype=mi.ScalarFloat)

            # dat_array = data
            # lum_array = luminance

            # pixel_width = 4 if mi.is_spectral else 3

            theta_scale = 1. / (res.y - 1) * dr.pi
            sin_theta_np = np.sin(np.arange(res.y) * theta_scale)
            v01 = .5 * (data[:, 0] + data[:, -1])
            data[:, 0] = data[:, -1] = v01
            lum = mi.luminance(mi.Color3f(data.reshape(-1, data.shape[-1])))
            luminance[:] = np.asarray(lum).reshape(luminance.shape) * sin_theta_np[:, None]

            self.m_warp = mi.Hierarchical2D0(luminance)
        super().parameters_changed(keys)

    def set_scene(self, scene: mi.Scene) -> None:
        if scene.bbox().valid():
            self.m_bsphere = mi.ScalarBoundingSphere3f(scene.bbox().bounding_sphere())
            self.m_bsphere.radius = dr.maximum(mi.math.RayEpsilon,
                                               self.m_bsphere.radius * (1. + mi.math.RayEpsilon))
        else:
            self.m_bsphere.center = 0.
            self.m_bsphere.radius = mi.math.RayEpsilon

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        v = self.world_transform().inverse().transform_affine(-si.wi)

        uv = mi.Point2f(dr.atan2(v.x, -v.z) * dr.inv_two_pi,
                        dr.safe_acos(v.y) * dr.inv_pi)
        return mi.depolarizer(self.eval_spectrum(uv, si.wavelengths, active))

    def sample_ray(self, time: float, sample1: float, sample2: mi.Point2f, sample3: mi.Point2f,
                   active: bool = True) -> Tuple[mi.Ray3f, mi.Color3f]:
        # 1. Sample spatial component
        offset = mi.warp.square_to_uniform_disk_concentric(sample2)

        # 2. Sample directional component
        uv, pdf = self.m_warp.sample(sample3, active=active)
        uv.x += .5 / (self.m_data.shape[1] - 1)

        active &= pdf > 0.

        theta = uv.y * dr.pi
        phi = uv.x * dr.two_pi

        d = dr_sphdir(theta, phi)
        d = mi.Vector3f(d.y, d.z, -d.x)

        inv_sin_theta = 1. / (dr.sin(theta) + 1e-10)
        pdf *= inv_sin_theta * dr.inv_two_pi * dr.inv_pi

        # Unlike \ref sample_direction, ray goes from the envmap toward the scene
        d_global = self.world_transform().transform_affine(-d)

        # Compute ray origin
        perpendicular_offset = mi.Frame3f(d).to_world(mi.Vector3f(offset.x, offset.y, 0))
        origin = self.m_bsphere.center + (perpendicular_offset - d_global) * self.m_bsphere.radius

        # 3. Sample spectral component (weight accounts for radiance)
        si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si.t = 0.
        si.time = time
        si.p = origin
        si.uv = uv
        wavelengths, weight = self.sample_wavelengths(si, sample1, active)

        r2 = dr.sqr(self.m_bsphere.radius)
        ray = mi.Ray3f(origin, d_global, time, wavelengths)
        weight *= dr.pi * r2 / pdf

        return ray, weight & active

    def sample_wavelengths(self, si, sample: float, active: bool = True) -> Tuple[mi.Color0f, mi.Color3f]:
        raise NotImplementedError('Seems that sample_shifted not exported')
        wavelengths, weight = self.m_d65.sample_spectrum(
            si, mi.math.sample_shifted(sample), active,
        )  # Seems that sample_shifted not exported

        return wavelengths, weight * self.eval_spectrum(si.uv, wavelengths, active, include_whitepoint=False)

    def eval_spectrum(self, uv: mi.Point2f, wavelengths: mi.Color0f,
                      active, include_whitepoint=True):
        res = mi.ScalarVector2u(self.m_data.shape[1], self.m_data.shape[0])
        uv.x -= .5 / (res.x - 1)
        uv -= dr.floor(uv)
        uv *= mi.Vector2f(res - 1)

        pos = dr.minimum(mi.Point2u(uv), res - 2)

        w1 = uv - mi.Point2f(pos)
        w0 = 1. - w1

        width = res.x
        index = dr.fma(pos.y, width, pos.x)

        v00 = dr.gather(mi.Color3f, self.m_data.array, index, active)
        v10 = dr.gather(mi.Color3f, self.m_data.array, index + 1, active)
        v01 = dr.gather(mi.Color3f, self.m_data.array, index + width, active)
        v11 = dr.gather(mi.Color3f, self.m_data.array, index + width + 1, active)

        v0 = dr.fma(w0.x, v00, w1.x * v10)
        v1 = dr.fma(w0.x, v01, w1.x * v11)
        v = dr.fma(w0.y, v0, w1.y * v1)

        return v * self.m_scale

    def sample_direction(self, it: mi.SurfaceInteraction3f, sample: mi.Point2f, active: bool = True):
        uv, pdf = self.m_warp.sample(sample, active=active)
        uv.x += .5 / (self.m_data.shape[1] - 1)
        active &= pdf > 0.

        theta = uv.y * dr.pi
        phi = uv.x * dr.two_pi

        d = dr_sphdir(theta, phi)
        d = mi.Vector3f(d.y, d.z, -d.x)  # TODO

        # Needed when the reference point is on the sensor, which is not part of the bbox
        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2. * radius

        inv_sin_theta = 1. / (1e-10 + dr.sin(theta))

        d = self.world_transform().transform_affine(d)

        ds = mi.DirectionSample3f()
        ds.p = it.p + d * dist
        ds.n = -d
        ds.uv = uv
        ds.time = it.time
        ds.pdf = dr.select(
            active,
            pdf * inv_sin_theta * (1. / (2. * dr.sqr(dr.pi))),
            0.
        )
        ds.delta = False
        ds.emitter = mi.EmitterPtr(self)
        ds.d = d
        ds.dist = dist

        weight = mi.depolarizer(self.eval_spectrum(uv, it.wavelengths, active)) / ds.pdf

        return ds, weight & active

    def mock_sample_direction(self, it: mi.SurfaceInteraction3f,
                              d: mi.Vector3f, pdf: mi.Float, active: bool = True):
        active &= pdf > 0.

        # Needed when the reference point is on the sensor, which is not part of the bbox
        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2. * radius

        ds = mi.DirectionSample3f()
        ds.p = it.p + d * dist
        ds.n = -d
        ds.uv = 0.
        ds.time = it.time
        ds.pdf = pdf
        ds.delta = False
        ds.emitter = mi.EmitterPtr(self)
        ds.d = d
        ds.dist = dist

        return ds

    def pdf_direction(self, it: mi.Interaction3f, ds: mi.DirectionSample3f, active: bool = True) -> mi.Float:
        d = self.world_transform().inverse().transform_affine(ds.d)
        uv = mi.Point2f(dr.atan2(d.x, -d.z) * dr.inv_two_pi,
                        dr.safe_acos(d.y) * dr.inv_pi)
        uv.x -= .5 / (self.m_data.shape[1] - 1)  # TODO
        uv -= dr.floor(uv)

        inv_sin_theta = 1. / (1e-10 + dr.safe_sqrt(dr.sqr(d.x) + dr.sqr(d.z)))

        return self.m_warp.eval(uv, active=active) * inv_sin_theta * (1. / (2. * dr.sqr(dr.pi)))

    def eval_direction(self, it, ds, active: bool = True) -> mi.Color3f:
        return self.eval_spectrum(ds.uv, it.wavelengths, active)

    def sample_position(self, ref: float, ds: mi.Point2f, active: bool = True):
        if dr.is_jit_v(mi.Float):
            # /* Do not throw an exception in JIT-compiled variants. This
            #                function might be invoked by DrJit's virtual function call
            #                recording mechanism despite not influencing any actual
            #                calculation. */
            return dr.zeros(mi.PositionSample3f), np.nan
        else:
            raise NotImplementedError('sample_position')

    def bbox(self):
        return mi.ScalarBoundingBox3f()

    def to_string(self):
        res = mi.ScalarVector2u(self.m_data.shape[1], self.m_data.shape[0])
        oss = io.StringIO()
        oss.write(f'MyEnvironmentMapEmitter[\n')
        if self.m_filename != '':
            oss.write(f'  filename = "{self.m_filename}",\n')
        oss.write(f'  res = "{res}",\n'
                  f'  bsphere = {indent(str(self.m_bsphere))},\n'
                  f']')
        return oss.getvalue()

mi.register_emitter('myenvmap', lambda props:  MyEnvironmentMapEmitter(props))