import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *
from utils import *
from models import *
from integrators import *

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

def set_teaser_scene_config(envmap_path):
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
        },
        'rectangle': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([5, 8, 1]).translate([0, 0, 0]),
            'material': {
                'type': 'principled',
                'base_color': {'type': 'rgb', 'value': [0.05, 0.05, 0.05]},
                'metallic': 1.0,
                'roughness': 0.05,
                'specular': 0.5
            }
        },
        'emitter': {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': envmap_path,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 0) @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        },
        # 'sphere_area_light': {
        #     'type': 'sphere',
        #     'center': [0, 0, 1],
        #     'radius': 0.3,
        #     'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': 10.0}}
        # }
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

def render_define_scene(scene_dict, render_spp, sensor, output_dir):
    """Render the scene with the given sensor"""
    
    # Render scene
    img, aovs = mi.render(scene_dict, sensor=sensor, spp=render_spp)
    
    # Save RGB render
    rgb_bmp = mi.Bitmap(img)
    mi.util.write_bitmap(join(output_dir, 'rgb.png'), rgb_bmp)
    
    # Save AOVs if available
    if 'albedo' in aovs:
        albedo_bmp = mi.Bitmap(aovs['albedo'][:, :, :3])
        mi.util.write_bitmap(join(output_dir, 'albedo.png'), albedo_bmp)
    
    if 'roughness' in aovs:
        roughness_bmp = mi.Bitmap(aovs['roughness'][:, :, :1])
        mi.util.write_bitmap(join(output_dir, 'roughness.png'), roughness_bmp)
    
    if 'metallic' in aovs:
        metallic_bmp = mi.Bitmap(aovs['metallic'][:, :, :1])
        mi.util.write_bitmap(join(output_dir, 'metallic.png'), metallic_bmp)
    
    if 'normal' in aovs:
        normal_img = aovs['normal'][:, :, :3]
        normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)
        normal_bmp = mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img + 1) / 2, 0)))
        mi.util.write_bitmap(join(output_dir, 'normal.png'), normal_bmp)
    
    if 'direct_light' in aovs:
        direct_light_bmp = mi.Bitmap(aovs['direct_light'][:, :, :3])
        mi.util.write_bitmap(join(output_dir, 'direct_light.png'), direct_light_bmp)
    
    if 'indirect_light' in aovs:
        indirect_light_bmp = mi.Bitmap(aovs['indirect_light'][:, :, :3])
        mi.util.write_bitmap(join(output_dir, 'indirect_light.png'), indirect_light_bmp)

def get_model_configs():
    """Get the 3D model configurations"""
    return [
        # TensoIR models
        {
            'ply_path': './demos/scenes/TensoIR/lego.ply',
            'rotation': [0, 0, -30],
            'scale': None,
            'translation': [2.0, -6.0, 0.35]
        },
        {
            'ply_path': './demos/scenes/TensoIR/armadillo.ply',
            'rotation': [0, 0, 90],
            'scale': None,
            'translation': [2.0, -3.5, 0.63]
        },
        {
            'ply_path': './demos/scenes/TensoIR/ficus.ply',
            'rotation': [0, 0, 180],
            'scale': [0.8, 0.8, 0.8],
            'translation': [2.0, -2, 0.7]
        },
        {
            'ply_path': './demos/scenes/TensoIR/hotdog.ply',
            'rotation': [0, 0, 140],
            'scale': [0.8, 0.8, 0.8],
            'translation': [-0.1, -4.8, 0.12]
        },
        # Synthetic4Relight models
        {
            'ply_path': './demos/scenes/Synthetic4Relight/air_baloons.ply',
            'rotation': [0, 0, 150],
            'scale': [0.5, 0.5, 0.5],
            'translation': [3.0, 6.0, 1.0]
        },
        {
            'ply_path': './demos/scenes/Synthetic4Relight/chair.ply',
            'rotation': [0, 0, -60],
            'scale': [0.8, 0.8, 0.8],
            'translation': [2.0, 3.0, 0.4]
        },
        {
            'ply_path': './demos/scenes/Synthetic4Relight/jugs.ply',
            'rotation': [0, 0, -90],
            'scale': [0.8, 0.8, 0.8],
            'translation': [2.0, 0, 0.05]
        }
    ]

def render_teaser(render_spp, envmap_path, sensor, output_dir):
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
    scene_config = set_teaser_scene_config(envmap_path)
    scene_config = add_gaussian_config(scene_config, all_gaussians_attributes)
    
    # Load scene
    scene_dict = mi.load_dict(scene_config)
    
    # Render the scene
    render_define_scene(scene_dict, render_spp, sensor, output_dir)

if __name__ == "__main__":
    # Configuration parameters
    render_spp = 32
    envmap_path = './demos/scenes/envmaps/bloem_train_track_cloudy_4k.exr'
    
    # Output directory configuration
    OUTPUT_DIR = join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'teaser')
    ensure_dir(OUTPUT_DIR)
    
    # Camera parameters
    sensor = make_sensor(
        cam_pos=(-25, 0.0, 12.5),
        look_at=(0.0, 0.0, 0.0),
        up=(0, 0, 1),
        fov=40,
        resx=1600,
        resy=600
    )

    render_teaser(render_spp, envmap_path, sensor, OUTPUT_DIR)