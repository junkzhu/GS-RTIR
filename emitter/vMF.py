# modified from https://github.com/gerwang/diff-sdf/blob/main/python/emitters/vMF.py

from typing import List

import drjit as dr
import mitsuba as mi
from .SGEmitter import SGEmitter

class VonMisesFisherEmitter(SGEmitter):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.position = mi.Point3f(0.)
        self.weight = mi.Float(1.)
        self.std = mi.Float(1.)
        self.weight_warp = mi.DiscreteDistribution(self.weight)
        self.inv_var = 1.0 / dr.sqr(self.std)

    def parameters_changed(self, keys: List[str] = []) -> None:
        if len(keys) == 0 or 'weight' in keys:
            self.weight_warp = mi.DiscreteDistribution(self.weight)
        if len(keys) == 0 or 'std' in keys:
            self.inv_var = 1.0 / dr.sqr(self.std)
        super().parameters_changed(keys)

    def sample_direction(self, it: mi.SurfaceInteraction3f, sample: mi.Point2f, active: bool = True):
        p = self.world_transform().inverse().transform_affine(it.p)
        index, sample_x_re = self.weight_warp.sample_reuse(sample.x, active=active)
        sample.x = sample_x_re
        sel_pos = dr.gather(mi.Point3f, self.position, index, active=active)
        sel_inv_var = dr.gather(mi.Float, self.inv_var, index, active=active)
        lobe = sel_pos - p
        lobe_norm = dr.norm(lobe)
        mu = lobe / lobe_norm
        frame = mi.Frame3f(mu)
        kappa = lobe_norm * sel_inv_var
        d = mi.warp.square_to_von_mises_fisher(sample, kappa)
        d = frame.to_world(d)
        d = self.world_transform().transform_affine(d)
        d = dr.normalize(d)

        # Needed when the reference point is on the sensor, which is not part of the bbox
        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2. * radius

        ds = mi.DirectionSample3f()
        ds.p = it.p + d * dist
        ds.n = -d
        ds.uv = 0.
        ds.time = it.time
        ds.delta = False
        ds.emitter = mi.EmitterPtr(self)
        ds.d = d
        ds.dist = dist
        ds.pdf = self.pdf_direction(it, ds, active=active)

        weight = mi.Spectrum(0.)
        return ds, weight & active

    def traverse(self, callback: mi.TraversalCallback) -> None:
        super().traverse(callback)
        callback.put('position', self.position, flags=mi.ParamFlags.Differentiable)
        callback.put('weight', self.weight, flags=mi.ParamFlags.Differentiable)
        callback.put('std', self.std, flags=mi.ParamFlags.Differentiable)

    def pdf_direction(self, it: mi.Interaction3f, ds: mi.DirectionSample3f, active: bool = True) -> mi.Float:
        d = self.world_transform().inverse().transform_affine(ds.d)
        d = dr.normalize(d)
        p = self.world_transform().inverse().transform_affine(it.p)
        i = mi.Int(0)
        pdf_sum = mi.Float(0.)
        while i < self.weight_warp.size():
            lobe = dr.gather(mi.Point3f, self.position, i) - p
            lobe_norm = dr.norm(lobe)
            mu = lobe / lobe_norm
            # TODO handle the case when lobe_norm equals 0
            frame = mi.Frame3f(mu)
            kappa = lobe_norm * dr.gather(mi.Float, self.inv_var, i)
            pdf_sum += mi.warp.square_to_von_mises_fisher_pdf(
                frame.to_local(d), kappa) * self.weight_warp.eval_pmf_normalized(i, active=active)
            i += 1
        return pdf_sum & active