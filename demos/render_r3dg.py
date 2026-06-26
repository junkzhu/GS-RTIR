import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import torch
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *
from utils import *
from models import *
from integrators import *
from datasets import *
import argparse

from tqdm import trange

# Output directory configuration
OUTPUT_DIR = join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'R3DG')
ensure_dir(OUTPUT_DIR)
ensure_dir(join(OUTPUT_DIR,'./rgb/'))
ensure_dir(join(OUTPUT_DIR,'./albedo/'))
ensure_dir(join(OUTPUT_DIR,'./roughness/'))
ensure_dir(join(OUTPUT_DIR,'./metallic/'))
ensure_dir(join(OUTPUT_DIR,'./direct_light/'))
ensure_dir(join(OUTPUT_DIR,'./indirect_light/'))
ensure_dir(join(OUTPUT_DIR,'./normal/'))
ensure_dir(join(OUTPUT_DIR,'./sh/'))

def make_sensor(cam_pos, look_at, up=(0, 0, 1), fov=45.0, resx=1600, resy=600):
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

def set_teaser_scene_config(envmap_path, render_splatting_result):
    """Set up the teaser scene configuration"""
    return {
        'type': 'scene',
        'integrator': {
            'type': 'gsprim_prb',
            'max_depth': 3,
            'pt_rate': 0.0 if render_splatting_result else 1.0,
            'gaussian_max_depth': 128,
            'hide_emitters': True if render_splatting_result else False,
            'use_mis': True,
            'selfocc_offset_max': 0.5,
            'selfocc_mode': 'normal',
            'selfocc_offset_fixed': 0.1,
            'geometry_threshold': 0.2,
            'separate_direct_indirect': True,
        },
        'rectangle': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([5, 5, 1]).translate([0, 0, -0.34]),
            'material': {
                'type': 'principled',
                'base_color': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
                'metallic': 1.0,
                'roughness': 0.05,
                'specular': 0.5
            }
        },
        'emitter': {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': envmap_path,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 180) @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        },




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

def render_define_scene(idx, scene_dict, render_spp, sensor):
    """Render the scene with the given sensor"""
    
    # Render scene
    integrator = mi.load_dict({
        'type': 'gsprim_prb',
        'max_depth': 3,
        'pt_rate': 0.0 if render_splatting_result else 1.0,
        'gaussian_max_depth': 128,
        'hide_emitters': True if render_splatting_result else False,
        'use_mis': True,
        'selfocc_offset_max': 0.5,
        'selfocc_mode': 'normal',
        'selfocc_offset_fixed': 0.1,
        'geometry_threshold': 0.2,
        'separate_direct_indirect': True,
    })

    film = sensor.film()
    film.clear()
    integrator.prepare_film(sensor=sensor, aovs=integrator.aovs())

    integrator.render_process(scene_dict, sensor=sensor, spp=render_spp, seed=idx)

    buffer = film.develop()

    dr.eval(buffer)

    rgb, aovs = unpack_buffer(buffer)
    
    if not render_splatting_result:
        mi.util.write_bitmap(join(OUTPUT_DIR,'./rgb/', f'{idx}.png'), rgb)
        mi.util.write_bitmap(join(OUTPUT_DIR, './albedo/', f'{idx}.png'), aovs['albedo'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './roughness/', f'{idx}.png'), aovs['roughness'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './metallic/', f'{idx}.png'), aovs['metallic'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './direct_light/', f'{idx}.png'), aovs['direct_light'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './indirect_light/', f'{idx}.png'), aovs['indirect_light'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './normal/', f'{idx}.png'), aovs['normal'])
    else:
        mi.util.write_bitmap(join(OUTPUT_DIR, './sh/', f'{idx}.png'), rgb)

    del buffer, rgb, aovs
    gc.collect()

def get_model_configs():
    """Get the 3D model configurations"""
    return [
        # TensoIR models

        {
            'ply_path': './demos/scenes/R3DG_models/jugs_rtir.ply',
            'rotation': [0, 0, 90],
            'scale': [1.5, 1.5, 1.5],
            'translation': [2.0, 2.0, -0.1]
        },

        {
            'ply_path': './demos/scenes/R3DG_models/hotdog_rtir.ply',
            'rotation': [0, 0, 120],
            'scale': [1.0, 1.0, 1.0],
            'translation': [-2.0, 2.0, -0.17]
        },

        {
            'ply_path': './demos/scenes/Synthetic4Relight/jugs.ply',
            'rotation': [0, 0, 90],
            'scale': [1.5, 1.5, 1.5],
            'translation': [2.0, -2.0, -0.1]
        },

        {
            'ply_path': './demos/scenes/TensoIR/hotdog.ply',
            'rotation': [0, 0, 120],
            'scale': [1.0, 1.0, 1.0],
            'translation': [-2.0, -2.0, -0.17]
        },
    ]

def render_lego(idx, render_spp, envmap_path, sensor, render_splatting_result):
    """Main function to render the teaser scene"""
    
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
    scene_config = set_teaser_scene_config(envmap_path, render_splatting_result)
    scene_config = add_gaussian_config(scene_config, all_gaussians_attributes)
    
    # Load scene
    scene_dict = mi.load_dict(scene_config)
    
    # Render the scene
    render_define_scene(idx, scene_dict, render_spp, sensor)

if __name__ == "__main__":
    # Configuration parameters
    render_spp = 2000
    envmap_path = '/path/to/datasets/TensoIR/Environment_Maps/high_res_envmaps_1k/sunset.hdr'
    render_splatting_result = False
    
   # with time_measure("Loading dataset"):
        #sensors = load_sensors('/path/to/datasets/TensoIR/lego/', 1600, 800)

    sensor = make_sensor(
        cam_pos=(-20, 0.0, 15.0),
        look_at=(0.0, 0.0, -1.0),
        up=(0, 0, 1),
        fov=25,
        resx=1200,
        resy=800
    )

    #for idx in trange(len(sensors), desc="Rendering", unit="spp"):
    render_lego(0, render_spp, envmap_path, sensor, render_splatting_result)
