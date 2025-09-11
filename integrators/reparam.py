import gc
import drjit as dr
import mitsuba as mi

PI = dr.pi
EPS = 1e-8

class ReparamIntegrator(mi.SamplingIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 4)
        self.hide_emitters = props.get('hide_emitters', False)
    
    def prepare(self, sensor, seed, spp, aovs=[]):
        film = sensor.film()
        sampler = sensor.sampler().clone()
        if spp != 0:
            sampler.set_sample_count(spp)
        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)
        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()
        wavefront_size = dr.prod(film_size) * spp

        wavefront_size_limit = 0xffffffff if dr.is_jit_v(mi.Float) else 0x40000000
        if wavefront_size > wavefront_size_limit:
            raise Exception(f"Wavefront {wavefront_size} exceeds {wavefront_size_limit}")
        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)
        return sampler, spp
    
    def sample_rays(self, scene, sensor, sampler, reparam=None):
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray

        When a reparameterization function is provided via the 'reparam'
        argument, it will be applied to the returned image-space position (i.e.
        the sample positions will be moving). The other two return values
        remain detached.
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2u()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(mi.Int(-film_size[0]), pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += film.crop_offset()

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        ray, weight = sensor.sample_ray_differential(time=wavelength_sample, sample1=sampler.next_1d(),
                                                     sample2=pos_adjusted,sample3=aperture_sample)
        det = mi.Float(1.0)
        if reparam is not None:
            assert not rfilter.is_box_filter()
            assert film.sample_border()

            with dr.resume_grad():
                reparam_d, det = reparam(ray=ray, depth=mi.UInt32(0))

                # Create a fake interaction along the sampled ray and use it to the
                # position with derivative tracking
                it = dr.zeros(mi.Interaction3f)
                it.p = ray.o + reparam_d
                ds, _ = sensor.sample_direction(it, aperture_sample)
                # Return a reparameterized image position
                pos_f = ds.uv + film.crop_offset()

        return ray, weight, pos_f, det

    @dr.syntax
    def ray_test(self, scene, sampler, ray, active, threshold = 0.4):
        #Stochastic Ray Tracing of Transparent 3D Gaussians, 3.3 section
        active=mi.Mask(active)
        original_active = mi.Mask(active)
        ray = mi.Ray3f(dr.detach(ray))
    
        T = mi.Float(1.0)
        while dr.hint(active, label=f"Shadow ray test"):
            si_cur = scene.ray_intersect(ray)
            active &= si_cur.is_valid() & si_cur.shape.is_ellipsoids()
                        
            transmission = self.eval_transmission(si_cur, ray, active)
            alpha = 1.0 - transmission # opacity as a probability
            T[active] *= transmission

            random_value = sampler.next_1d()
            active &= random_value < (1 - alpha)
            ray.o[active] = si_cur.p + ray.d * 1e-4

        occluded = T < threshold

        return occluded & original_active

    #-------------------- 3DGS --------------------
    def eval_transmission(self, si, ray, active):
        """
        Evaluate the transmission model on intersected volumetric primitives
        """
        def gather_ellipsoids_props(self, prim_index, active):
            if self is not None and self.is_ellipsoids():
                si = dr.zeros(mi.SurfaceInteraction3f)
                si.prim_index = prim_index
                data = self.eval_attribute_x("ellipsoid", si, active)
                center  = mi.Point3f([data[i] for i in range(3)])
                scale   = mi.Vector3f([data[i + 3] for i in range(3)])
                quat    = mi.Quaternion4f([data[i + 6] for i in range(4)])
                rot     = dr.quat_to_matrix(quat, size=3)
                return center, scale, rot
            else:
                return mi.Point3f(0), mi.Vector3f(0), mi.Matrix3f(0)

        center, scale, rot = dr.dispatch(si.shape, gather_ellipsoids_props, si.prim_index, active)

        opacity = si.shape.eval_attribute_1("opacities", si, active)

        # Gaussian splatting transmittance model
        # Find the peak location along the ray, from "3D Gaussian Ray Tracing"
        o = rot.T * (ray.o - center) / scale
        d = rot.T * ray.d / scale
        t_peak = -dr.dot(o, d) / dr.dot(d, d)
        p_peak = ray(t_peak)

        # Gaussian kernel evaluation
        p = rot.T * (p_peak - center)
        density = dr.exp(-0.5 * (p.x**2 / scale.x**2 + p.y**2 / scale.y**2 + p.z**2 / scale.z**2))

        return 1.0 - dr.minimum(opacity * density, 0.9999)

    def eval_bsdf_component(self, si, ray, active):
        def eval(shape, si, ray, active):
            if shape is not None and shape.is_ellipsoids():
                normals = shape.eval_attribute_3("normals", si, active)
                normals = dr.normalize(mi.Vector3f(normals))

                albedos = shape.eval_attribute_3("albedos", si, active)
                albedos = dr.maximum(albedos, 0.0)

                roughnesses = shape.eval_attribute_1("roughnesses", si, active)
                roughnesses = dr.clamp(roughnesses, 0.0, 1.0)

                metallics = shape.eval_attribute_1("metallics", si, active)
                metallics = dr.clamp(metallics, 0.0, 1.0)

                return normals, albedos, roughnesses, metallics
            else:
                return mi.Vector3f(0.0), mi.Color3f(0.0), mi.Float(0.0), mi.Float(0.0)

        return dr.dispatch(si.shape, eval, si, ray, active)

    #-------------------- BSDF --------------------
    def fresnel_schlick(self, F0, cosTheta):
        # F0: Color3f, cosTheta: scalar or array in [0,1]
        # Schlick approximation: F = F0 + (1-F0)*(1-cosTheta)^5
        c = dr.clamp(1.0 - cosTheta, 0.0, 1.0)
        c5 = dr.power(c, 5.0)
        return F0 + (1.0 - F0) * c5

    def ggx_D(self, N, H, roughness):
        # roughness is scalar in [0,1], we use alpha = roughness^2
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        NdotH = dr.clamp(dr.dot(N, H), 0.0, 1.0)
        NdotH2 = NdotH * NdotH
        denom = NdotH2 * (alpha2 - 1.0) + 1.0
        D = alpha2 / (PI * denom * denom + EPS)
        return D

    def smith_G1(self, N, V, alpha):
        # G1 for GGX (Heitz) ; alpha is roughness^2
        NdotV = dr.clamp(dr.dot(N, V), 0.0, 1.0)
        # avoid division by 0
        a = alpha
        # common stable form:
        tmp = dr.sqrt(a * a + (1.0 - a * a) * (NdotV * NdotV))
        G1 = 2.0 * NdotV / (NdotV + tmp + EPS)
        return G1

    def ggx_G(self, N, V, L, roughness):
        alpha = roughness * roughness
        G1V = self.smith_G1(N, V, alpha)
        G1L = self.smith_G1(N, L, alpha)
        return G1V * G1L

    def eval_bsdf(self, albedo, roughness, metallic, N, V, L, H):
        albedo = dr.clamp(albedo, 0.0, 1.0)
        roughness = dr.clamp(roughness, 0.1, 0.9)
        metallic = dr.clamp(metallic, 0.1, 0.9)
        
        NdotL = dr.clamp(dr.dot(N, L), 0.0, 1.0)
        NdotV = dr.clamp(dr.dot(N, V), 0.0, 1.0)
        NdotH = dr.clamp(dr.dot(N, H), 0.0, 1.0)
        VdotH = dr.clamp(dr.dot(V, H), 0.0, 1.0)

        # --- Fresnel base F0 mix (dielectric 0.04 vs albedo for metal) ---
        F0_dielectric = mi.Color3f(0.04)
        # ensure types broadcast correctly: metallic may be scalar or per-item
        F0 = dr.lerp(F0_dielectric, albedo, metallic)  # Color3f

        # Fresnel term (Schlick)
        F = self.fresnel_schlick(F0, VdotH)  # Color3f

        # --- D, G ---
        D = self.ggx_D(N, H, roughness)  # scalar per item
        G = self.ggx_G(N, V, L, roughness)  # scalar per item

        # Specular numerator (Color) = D * G * F
        spec_num = F * D * G  # Color3f (broadcast D,G)

        denom = 4.0 * (NdotV * NdotL + 1e-8)  # scalar
        specular = spec_num / denom  # Color3f

        F_avg = (F[0] + F[1] + F[2]) / 3.0
        diffuse = (1.0 - metallic) * (albedo / PI) * (1.0 - F_avg)

        bsdf_val = specular + diffuse

        return bsdf_val
    
