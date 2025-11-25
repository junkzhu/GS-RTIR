import mitsuba as mi
from jinja2 import Template
from emitter.vMF import VonMisesFisherEmitter
        
class SGModel:
    """
    Performs inverse rendering to optimize SG parameters using Mitsuba's AD framework.
    """
    def __init__(self, num_sgs=16, sg_init=None):    
        self.num_sgs = num_sgs

        with open('emitter/SGEmitter.tmp.py', 'r') as f:
            template = Template(f.read())
            rendered_code = template.render(num_sgs=num_sgs, sg_init=sg_init)
            with open('emitter/SGEmitter.py', 'w') as f:
                f.write(rendered_code)

        mi.register_emitter('vMF', lambda props: VonMisesFisherEmitter(props))