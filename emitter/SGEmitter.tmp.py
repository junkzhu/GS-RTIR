import io
from typing import Optional

import drjit as dr
import mitsuba as mi
from .myenvmap import MyEnvironmentMapEmitter, indent
from .sgenvmap_util import fibonacci_sphere

class SGEmitter(MyEnvironmentMapEmitter):
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

        self.scene: Optional[mi.Scene] = None
        padding_size = float(props.get('padding_size', 0.0))
        self.bounding_box = mi.BoundingBox3f(mi.Point3f(0.0 - padding_size), mi.Point3f(1.0 + padding_size))
        fib_arr = fibonacci_sphere( {{ num_sgs }} )
        {% for i in range(num_sgs) %}
        {% if sg_init.shape == (num_sgs, 7) %}
        self.lobe_{{ i }} = mi.Vector3f([{{ sg_init[i, 0] }}, {{ sg_init[i, 1] }}, {{ sg_init[i, 2] }}])
        self.lambda_{{ i }} = mi.Float({{ sg_init[i, 3] }})
        self.mu_{{ i }} = mi.Vector3f([{{ sg_init[i, 4] }}, {{ sg_init[i, 5] }}, {{ sg_init[i, 6] }}])
        {% else %}
        self.lobe_{{ i }} = mi.Vector3f(fib_arr[{{ i }}])
        self.lambda_{{ i }} = mi.Float(5.0)
        self.mu_{{ i }} = mi.Vector3f([1.0, 1.0, 1.0])
        {% endif %}
        {% endfor %}

    def set_scene(self, scene: mi.Scene) -> None:
        super().set_scene(scene)
        self.scene = scene

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        return self.eval_sg(si.p, -si.wi, active)

    def eval_sg_ray(self, o: mi.Point3f, v: mi.Vector3f, near: mi.Float, active: bool = True) -> mi.Color3f:
        res = mi.Color3f(0.0)
        {% for i in range(num_sgs) %}
        res += dr.exp(self.lambda_{{ i }} * (dr.dot(v, dr.normalize(self.lobe_{{ i }})) - 1)) * self.mu_{{ i }}
        {% endfor %}

        return res

    def eval_sg(self, o: mi.Point3f, v: mi.Vector3f, active: bool = True) -> mi.Color3f:
        world2cam = self.world_transform().inverse()
        tmp_ray = mi.Ray3f(o, v)
        tmp_ray = world2cam @ tmp_ray
        o, v = tmp_ray.o, tmp_ray.d
        return self.eval_sg_ray(o, v, 0.0, active)

    def sample_direction(self, it: mi.SurfaceInteraction3f, sample: mi.Point2f, active: bool = True):
        ds, _ = super().sample_direction(it, sample, active)
        return ds, mi.Spectrum(0.) & active

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
