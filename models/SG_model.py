import sys
import importlib
import mitsuba as mi
import numpy as np
from jinja2 import Template
        
class SGModel:
    """
    Performs inverse rendering to optimize SG parameters using Mitsuba's AD framework.
    """
    def __init__(self, num_sgs=16, sg_init=None, euler_deg=(0.0, 0.0, 0.0)):  
        def _rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
            rx = np.deg2rad(rx_deg)
            ry = np.deg2rad(ry_deg)
            rz = np.deg2rad(rz_deg)
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            return r_z @ r_y @ r_x

        if sg_init is not None:
            if sg_init.ndim != 2 or sg_init.shape[1] != 7:
                raise ValueError("The 'sg_init' array must have shape (num_sgs, 7).")

            rot_mat = _rotation_matrix_xyz(*euler_deg)
            sg_init[:, 0:3] = sg_init[:, 0:3] @ rot_mat.T

        with open('emitter/SGEmitter.tmp.py', 'r') as f:
            template = Template(f.read())
            rendered_code = template.render(num_sgs=num_sgs, sg_init=sg_init)
            with open('emitter/SGEmitter.py', 'w') as f:
                f.write(rendered_code)

        # reload SGEmitter module
        module_name = "emitter.SGEmitter"
        if module_name in sys.modules:
            del sys.modules[module_name]

        SGEmitter = importlib.import_module(module_name)
        
        importlib.reload(SGEmitter)

        SGEmitterClass = SGEmitter.SGEmitter
        mi.register_emitter('SGEmitter', lambda props: SGEmitterClass(props))