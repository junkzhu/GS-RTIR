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
        fib_arr = fibonacci_sphere( 8 )
            
        self.lobe_0 = mi.Vector3f(fib_arr[0])
        self.lambda_0 = mi.Float(20.0)
        self.mu_0 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_1 = mi.Vector3f(fib_arr[1])
        self.lambda_1 = mi.Float(20.0)
        self.mu_1 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_2 = mi.Vector3f(fib_arr[2])
        self.lambda_2 = mi.Float(20.0)
        self.mu_2 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_3 = mi.Vector3f(fib_arr[3])
        self.lambda_3 = mi.Float(20.0)
        self.mu_3 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_4 = mi.Vector3f(fib_arr[4])
        self.lambda_4 = mi.Float(20.0)
        self.mu_4 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_5 = mi.Vector3f(fib_arr[5])
        self.lambda_5 = mi.Float(20.0)
        self.mu_5 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_6 = mi.Vector3f(fib_arr[6])
        self.lambda_6 = mi.Float(20.0)
        self.mu_6 = mi.Vector3f([0.5, 0.5, 0.5])
        
        self.lobe_7 = mi.Vector3f(fib_arr[7])
        self.lambda_7 = mi.Float(20.0)
        self.mu_7 = mi.Vector3f([0.5, 0.5, 0.5])
        
    def set_scene(self, scene: mi.Scene) -> None:
        super().set_scene(scene)
        self.scene = scene

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        return self.eval_sg(si.p, -si.wi, active)

    def eval_sg_ray(self, o: mi.Point3f, v: mi.Vector3f, near: mi.Float, active: bool = True) -> mi.Color3f:
        res = mi.Color3f(0.0)
        
        res += dr.exp(self.lambda_0 * (dr.dot(v, dr.normalize(self.lobe_0)) - 1)) * self.mu_0    
        res += dr.exp(self.lambda_1 * (dr.dot(v, dr.normalize(self.lobe_1)) - 1)) * self.mu_1
        res += dr.exp(self.lambda_2 * (dr.dot(v, dr.normalize(self.lobe_2)) - 1)) * self.mu_2
        res += dr.exp(self.lambda_3 * (dr.dot(v, dr.normalize(self.lobe_3)) - 1)) * self.mu_3
        res += dr.exp(self.lambda_4 * (dr.dot(v, dr.normalize(self.lobe_4)) - 1)) * self.mu_4
        res += dr.exp(self.lambda_5 * (dr.dot(v, dr.normalize(self.lobe_5)) - 1)) * self.mu_5
        res += dr.exp(self.lambda_6 * (dr.dot(v, dr.normalize(self.lobe_6)) - 1)) * self.mu_6
        res += dr.exp(self.lambda_7 * (dr.dot(v, dr.normalize(self.lobe_7)) - 1)) * self.mu_7
        
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
        
        callback.put('lgtSGslobe_0', self.lobe_0, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_0', self.lambda_0, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_0', self.mu_0, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_1', self.lobe_1, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_1', self.lambda_1, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_1', self.mu_1, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_2', self.lobe_2, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_2', self.lambda_2, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_2', self.mu_2, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_3', self.lobe_3, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_3', self.lambda_3, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_3', self.mu_3, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_4', self.lobe_4, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_4', self.lambda_4, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_4', self.mu_4, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_5', self.lobe_5, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_5', self.lambda_5, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_5', self.mu_5, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_6', self.lobe_6, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_6', self.lambda_6, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_6', self.mu_6, flags=mi.ParamFlags.Differentiable)
        
        callback.put('lgtSGslobe_7', self.lobe_7, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGslambda_7', self.lambda_7, flags=mi.ParamFlags.Differentiable)
        callback.put('lgtSGsmu_7', self.mu_7, flags=mi.ParamFlags.Differentiable)