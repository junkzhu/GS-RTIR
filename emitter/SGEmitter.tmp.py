import io
from typing import Optional

import drjit as dr
import mitsuba as mi
from .sgenvmap_util import fibonacci_sphere, expm1, indent

class SGEmitter(mi.Emitter):
    def __init__(self, props: mi.Properties):
        """
        What do I need:
            load_config: str, file path of nerf model
            bounded_radius: float, the bounding sphere of bounded scene, centered at (0, 0, 0)
            ..., some options for experiments
        Args:
            props:
        """
        super().__init__(props)

        self.m_bsphere: mi.ScalarBoundingSphere3f = mi.ScalarBoundingSphere3f(mi.ScalarPoint3f(0.), 1.)
        
        fib_arr = fibonacci_sphere( {{ num_sgs }} )
        {% for i in range(num_sgs) %}
        {% if sg_init.shape == (num_sgs, 7) %}
        self.lobe_{{ i }} = mi.Vector3f([{{ sg_init[i, 0] }}, {{ sg_init[i, 1] }}, {{ sg_init[i, 2] }}])
        self.lambda_{{ i }} = mi.Float({{ sg_init[i, 3] }})
        self.mu_{{ i }} = mi.Vector3f([{{ sg_init[i, 4] }}, {{ sg_init[i, 5] }}, {{ sg_init[i, 6] }}])
        {% else %}
        self.lobe_{{ i }} = mi.Vector3f(fib_arr[{{ i }}])
        self.lambda_{{ i }} = mi.Float({% if i % 2 == 0 %} 1 {% else %} 20 {% endif %})
        self.mu_{{ i }} = mi.Vector3f([0.1, 0.1, 0.1])
        {% endif %}
        {% endfor %}
        
        self._update_weight_warp()
        
        self.m_flags = mi.EmitterFlags.Infinite | mi.EmitterFlags.SpatiallyVarying
        self.m_to_world = mi.Transform4f()

    def set_scene(self, scene: mi.Scene) -> None:
        if scene.bbox().valid():
            self.m_bsphere = mi.ScalarBoundingSphere3f(scene.bbox().bounding_sphere())
            self.m_bsphere.radius = dr.maximum(mi.math.RayEpsilon,
                                               self.m_bsphere.radius * (1. + mi.math.RayEpsilon))
        else:
            self.m_bsphere.center = 0.
            self.m_bsphere.radius = mi.math.RayEpsilon
    
    def _update_weight_warp(self) -> None:
        """Update the discrete distribution for SG weights"""
        {% for i in range(num_sgs) %}
        self.weight_{{ i }} = dr.norm(expm1(self.mu_{{ i }}))
        {% endfor %}
        weight_array = dr.zeros(mi.Float, {{ num_sgs }})
        {% for i in range(num_sgs) %}
        dr.scatter(weight_array, self.weight_{{ i }}, {{ i }})
        {% endfor %}
        self.weight_warp = mi.DiscreteDistribution(weight_array)
    
    def parameters_changed(self, keys=None) -> None:
        if keys is None:
            keys = []
        if len(keys) == 0 or any('mu' in k for k in keys):
            self._update_weight_warp()
        super().parameters_changed(keys)

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        return self.eval_sg(si.p, -si.wi, active)

    def eval_sg_ray(self, o: mi.Point3f, v: mi.Vector3f, near: mi.Float, active: bool = True) -> mi.Color3f:
        res = mi.Color3f(0.0)
        {% for i in range(num_sgs) %}
        res += dr.exp(self.lambda_{{ i }} * (dr.dot(v, dr.normalize(self.lobe_{{ i }})) - 1)) * expm1(self.mu_{{ i }})
        {% endfor %}
        
        return res

    def eval_sg(self, o: mi.Point3f, v: mi.Vector3f, active: bool = True) -> mi.Color3f:
        world2cam = self.world_transform().inverse()
        tmp_ray = mi.Ray3f(o, v)
        tmp_ray = world2cam @ tmp_ray
        o, v = tmp_ray.o, tmp_ray.d
        return self.eval_sg_ray(o, v, 0.0, active)
    
    def eval_direction(self, it, ds, active: bool = True) -> mi.Color3f:
        return self.eval_sg(it.p, -ds.d, active)

    def sample_direction(self, it: mi.SurfaceInteraction3f, sample: mi.Point2f, active: bool = True):
        # Sample a SG based on the weights, then sample a direction from the selected vMF
        index, sample_x_re = self.weight_warp.sample_reuse(sample.x, active=active)
        sample.x = sample_x_re

        {% for i in range(num_sgs) %}
        lobe_{{ i }} = dr.select(index == {{ i }}, self.lobe_{{ i }}, mi.Vector3f(0.))
        lambda_{{ i }} = dr.select(index == {{ i }}, self.lambda_{{ i }}, mi.Float(0.))
        {% endfor %}
        
        sel_lobe = mi.Vector3f(0.)
        sel_lambda = mi.Float(0.)
        {% for i in range(num_sgs) %}
        sel_lobe += lobe_{{ i }}
        sel_lambda += lambda_{{ i }}
        {% endfor %}
        
        # vMF: μ = normalized(lobe), κ = lambda
        mu = dr.normalize(sel_lobe)
        kappa = sel_lambda
        frame = mi.Frame3f(mu)
        
        d_local = mi.warp.square_to_von_mises_fisher(sample, kappa)
        d = frame.to_world(d_local)
        d = self.world_transform().transform_affine(d)
        d = dr.normalize(d)
        
        # Calculate the bounding sphere radius and distance
        radius = dr.maximum(self.m_bsphere.radius, dr.norm(it.p - self.m_bsphere.center))
        dist = 2. * radius
        
        # Fill in DirectionSample3f
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
    
    def pdf_direction(self, it: mi.Interaction3f, ds: mi.DirectionSample3f, active: bool = True) -> mi.Float:
        # Convert direction to local space
        d = self.world_transform().inverse().transform_affine(ds.d)
        d = dr.normalize(d)
        
        i = mi.Int(0)
        pdf_sum = mi.Float(0.)
        while i < {{ num_sgs }}:
            {% for j in range(num_sgs) %}
            {% if j == 0 %}
            lobe_i = dr.select(i == {{ j }}, self.lobe_{{ j }}, mi.Vector3f(0.))
            lambda_i = dr.select(i == {{ j }}, self.lambda_{{ j }}, mi.Float(0.))
            {% else %}
            lobe_i = dr.select(i == {{ j }}, self.lobe_{{ j }}, lobe_i)
            lambda_i = dr.select(i == {{ j }}, self.lambda_{{ j }}, lambda_i)
            {% endif %}
            {% endfor %}
            
            mu = dr.normalize(lobe_i)
            kappa = lambda_i
            frame = mi.Frame3f(mu)
            pdf_i = mi.warp.square_to_von_mises_fisher_pdf(frame.to_local(d), kappa)
            
            weight_i = self.weight_warp.eval_pmf_normalized(i, active=active)
            
            pdf_sum += pdf_i * weight_i
            i += 1
        
        return pdf_sum & active

    def to_string(self):
        res = mi.ScalarVector2u(self.m_data.shape[1], self.m_data.shape[0])
        oss = io.StringIO()
        oss.write(f'SGEmitter[\n')
        if self.m_filename != '':
            oss.write(f'  filename = "{self.m_filename}",\n')
        oss.write(f'  res = "{res}",\n'
                  f'  bsphere = {indent(str(self.m_bsphere))},\n')
        oss.write(f']')
        return oss.getvalue()

    def traverse(self, callback: mi.TraversalCallback) -> None:
        super().traverse(callback)
        {% for i in range(num_sgs) %}
        callback.put('lgtSGslobe_{{ i }}', self.lobe_{{ i }}, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_{{ i }}', self.lambda_{{ i }}, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_{{ i }}', self.mu_{{ i }}, flags=mi.ParamFlags.Differentiable)
        {% endfor %}
