import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *
from utils import *
from models import *
from integrators import *
from losses import *

def make_sensor(cam_pos, look_at, up, fov=45.0, resx=1600, resy=600):
    """Create a camera sensor"""
    film = mi.load_dict({
        'type': 'hdrfilm',
        'width': resx,
        'height': resy,
        'pixel_format': 'rgb',
        'pixel_filter': {'type': 'tent'},
        'sample_border': True
    })

    sensor = mi.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': mi.ScalarTransform4f.look_at(origin=cam_pos, target=look_at, up=up),
        'film': film
    })

    return sensor

def set_scene_basic_config(envmap_path):
    """Set up the teaser scene configuration"""
    return {
        'type': 'scene',
        'integrator': {
            'type': 'gsprim_prb',
            'max_depth': 3,
            'pt_rate': 1.0,
            'gaussian_max_depth': 128,
            'hide_emitters': False,
            'use_mis': True,
            'selfocc_offset_max': 0.1,
            'geometry_threshold': 0.2,
            'separate_direct_indirect': True,
            'optimize_mesh': True
        },
        
        # BSDF Material
        "white": {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': [0.885809, 0.698859, 0.666422]
            },
            'metallic': 0.0,
            'roughness': 1.0,
            'specular': 0.0
        },

        "green": {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': [0.105421, 0.37798, 0.076425]
            },
            'metallic': 0.0,
            'roughness': 1.0,
            'specular': 0.0
        },

        "red": {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': [0.570068, 0.0430135, 0.0443706]
            },
            'metallic': 0.0,
            'roughness': 1.0,
            'specular': 0.0
        },

        "gold": {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': [1.000, 0.766, 0.336]
            },
            'metallic': 0.8,
            'roughness': 0.3,
            'specular': 0.0
        },

        # Light
        'emitter': {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': envmap_path,
            'to_world': mi.ScalarTransform4f.rotate([0, 1, 0], -180) @ mi.ScalarTransform4f.rotate([1, 0, 0], 0)
        },

        "light": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_luminaire.obj",
            "to_world": mi.ScalarTransform4f.translate([0.0, -0.001, 0]),
            "bsdf": {
                "type": "ref",
                "id": "white"
            },
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [9, 6.5, 3]
                }
            }
        },

        "light2": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_luminaire.obj",
            "to_world": mi.ScalarTransform4f.rotate([0, 0, 1], 90).translate([0.0, -0.001, 0]),
            "bsdf": {
                "type": "ref",
                "id": "white"
            },
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [9, 6.5, 3]
                }
            }
        },


        # Meshs
        "floor": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_floor.obj",
            "bsdf": {
                "type": "ref",
                "id": "white"
            }
        },

        "ceiling": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_ceiling.obj",
            "bsdf": {
                "type": "ref",
                "id": "white"
            }
        },

        "back": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_back_with_window.obj",
            "bsdf": {
                "type": "ref",
                "id": "white"
            }
        },

        "greenwall": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_greenwall.obj",
            "bsdf": {
                "type": "ref",
                "id": "green"
            }
        },

        "redwall": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/cbox_redwall.obj",
            "bsdf": {
                "type": "ref",
                "id": "red"
            }
        },

        "dragon": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/dragon.obj",
            "to_world": mi.ScalarTransform4f.scale([0.005, 0.005, 0.005]).translate([100, -200, 0]).rotate([0, 1 ,0], -30),
            "bsdf": {
                "type": "ref",
                "id": "gold"
            }
        }

    }

def add_gaussian_config(scene_config, gaussians_attributes, name_prefix='gaussian'):
    """Add Gaussian models to the scene configuration"""
    is_multiple = isinstance(gaussians_attributes, list)
    
    if is_multiple:
        for i, attrs in enumerate(gaussians_attributes):
            gaussian_name = f'{name_prefix}{i+1}'
            scene_config[gaussian_name] = {
                'type': 'ellipsoidsmesh',
                'centers': attrs['centers'],
                'scales': attrs['scales'],
                'quaternions': attrs['quats'],
                'opacities': attrs['sigmats'],
                'sh_coeffs': attrs['features'],
                'normals': attrs['normals'],
                'albedos': attrs['albedos'],
                'roughnesses': attrs['roughnesses'],
                'metallics': attrs['metallics']
            }
    else:
        scene_config['gaussians'] = {
            'type': 'ellipsoidsmesh',
            'centers': gaussians_attributes['centers'],
            'scales': gaussians_attributes['scales'],
            'quaternions': gaussians_attributes['quats'],
            'opacities': gaussians_attributes['sigmats'],
            'sh_coeffs': gaussians_attributes['features'],
            'normals': gaussians_attributes['normals'],
            'albedos': gaussians_attributes['albedos'],
            'roughnesses': gaussians_attributes['roughnesses'],
            'metallics': gaussians_attributes['metallics']
        }
    
    return scene_config

def get_model_configs():
    """Get the 3D model configurations"""
    return [
        # TensoIR models
        {
            'ply_path': './demos/scenes/TensoIR/armadillo.ply',
            'rotation': [-90.0, 210.0, 0.0],
            'scale': [0.65, 0.65, 0.65],
            'translation': [-0.45, -0.25, -0.5]
        },

        {
            'ply_path': './demos/scenes/TensoIR/ficus.ply',
            'rotation': [-90.0, 100.0, 0.0],
            'scale': [0.4, 0.4, 0.4],
            'translation': [-0.05, -0.3, -0.1]
        },

        {
            'ply_path': './demos/scenes/Synthetic4Relight/air_baloons.ply',
            'rotation': [-90.0, 30.0, 0.0],
            'scale': [0.3, 0.3, 0.3],
            'translation': [0.4, 0.3, 0.0]
        },
    ]

def set_scene_config(envmap_path):
    """Main function to render the GT scene"""
    
    # Get 3D model configurations
    model_configs = get_model_configs()
    
    # Initialize ellipsoids factory
    ellipsoidsfactory = EllipsoidsFactory()
    
    # Load and transform all Gaussian models
    all_gaussians_attributes = []
    with time_measure("Loading multiple gaussians"):
        for config in model_configs:
            # Load Gaussian model
            gaussians = GaussianModel()
            gaussians.restore_from_ply(config['ply_path'], False)
            
            # Apply transformations
            if config['translation'] is not None:
                gaussians.translate(torch.tensor(config['translation'], dtype=torch.float32))
            
            if config['scale'] is not None:
                gaussians.scale(torch.tensor(config['scale'], dtype=torch.float32))
            
            if config['rotation'] is not None:
                # Convert Euler angles to rotation matrix
                rx, ry, rz = config['rotation']
                rx_rad, ry_rad, rz_rad = torch.deg2rad(torch.tensor([rx, ry, rz], dtype=torch.float32))
                
                R_x = torch.tensor([[1, 0, 0], [0, torch.cos(rx_rad), -torch.sin(rx_rad)], [0, torch.sin(rx_rad), torch.cos(rx_rad)]], dtype=torch.float32)
                R_y = torch.tensor([[torch.cos(ry_rad), 0, torch.sin(ry_rad)], [0, 1, 0], [-torch.sin(ry_rad), 0, torch.cos(ry_rad)]], dtype=torch.float32)
                R_z = torch.tensor([[torch.cos(rz_rad), -torch.sin(rz_rad), 0], [torch.sin(rz_rad), torch.cos(rz_rad), 0], [0, 0, 1]], dtype=torch.float32)
                
                rotmat = R_z @ R_y @ R_x
                gaussians.rotate(rotmat)
            
            # Convert to ellipsoid attributes
            gaussians_attrs = ellipsoidsfactory.load_gaussian(gaussians=gaussians)
            all_gaussians_attributes.append(gaussians_attrs)
    
    # Create and configure scene
    scene_config = set_scene_basic_config(envmap_path)
    scene_config = add_gaussian_config(scene_config, all_gaussians_attributes)
    
    # Load scene
    scene_dict = mi.load_dict(scene_config)
    
    return scene_dict