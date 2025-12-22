import sys
import importlib
import mitsuba as mi
from jinja2 import Template
        
class SGModel:
    """
    Performs inverse rendering to optimize SG parameters using Mitsuba's AD framework.
    """
    def __init__(self, num_sgs=16, base_color_init=[0.5, 0.5, 0.5], sg_init=None):    

        with open('emitter/SGEmitter.tmp.py', 'r') as f:
            template = Template(f.read())
            rendered_code = template.render(num_sgs=num_sgs, base_color_init=base_color_init, sg_init=sg_init)
            with open('emitter/SGEmitter.py', 'w') as f:
                f.write(rendered_code)

        # reload SGEmitter module
        module_name = "emitter.SGEmitter"
        if module_name in sys.modules:
            del sys.modules[module_name]

        SGEmitter = importlib.import_module(module_name)
        
        vMF_module = importlib.import_module("emitter.vMF")
        importlib.reload(vMF_module)

        VonMisesFisherEmitter = vMF_module.VonMisesFisherEmitter

        mi.register_emitter('vMF', lambda props: VonMisesFisherEmitter(props))