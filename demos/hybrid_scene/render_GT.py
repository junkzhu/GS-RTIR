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

from common_config import *

# Output directory configuration
OUTPUT_DIR = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'hybrid_scene', 'GT')
ensure_dir(OUTPUT_DIR)

def render_define_scene(scene_dict, render_spp, sensor):
    """Render the scene with the given sensor"""
    
    # Render scene
    img, aovs = mi.render(scene_dict, sensor=sensor, spp=render_spp)
    
    # Save RGB render
    rgb_bmp = mi.Bitmap(img)
    mi.util.write_bitmap(join(OUTPUT_DIR, 'rgb.png'), rgb_bmp)
    
    # Save AOVs if available
    if 'albedo' in aovs:
        albedo_bmp = mi.Bitmap(aovs['albedo'][:, :, :3])
        mi.util.write_bitmap(join(OUTPUT_DIR, 'albedo.png'), albedo_bmp)
    
    if 'roughness' in aovs:
        roughness_bmp = mi.Bitmap(aovs['roughness'][:, :, :1])
        mi.util.write_bitmap(join(OUTPUT_DIR, 'roughness.png'), roughness_bmp)
    
    if 'metallic' in aovs:
        metallic_bmp = mi.Bitmap(aovs['metallic'][:, :, :1])
        mi.util.write_bitmap(join(OUTPUT_DIR, 'metallic.png'), metallic_bmp)
    
    if 'normal' in aovs:
        normal_img = aovs['normal'][:, :, :3]
        normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)
        normal_bmp = mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img + 1) / 2, 0)))
        mi.util.write_bitmap(join(OUTPUT_DIR, 'normal.png'), normal_bmp)
    
    if 'direct_light' in aovs:
        direct_light_bmp = mi.Bitmap(aovs['direct_light'][:, :, :3])
        mi.util.write_bitmap(join(OUTPUT_DIR, 'direct_light.png'), direct_light_bmp)
    
    if 'indirect_light' in aovs:
        indirect_light_bmp = mi.Bitmap(aovs['indirect_light'][:, :, :3])
        mi.util.write_bitmap(join(OUTPUT_DIR, 'indirect_light.png'), indirect_light_bmp)

def render_GT(render_spp, envmap_path, sensor):
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
    
    # Render the scene
    render_define_scene(scene_dict, render_spp, sensor)

if __name__ == "__main__":
    # Configuration parameters
    render_spp = 2000
    envmap_path = './demos/scenes/envmaps/pretoria_gardens_4k.exr'
    
    # Camera parameters
    sensor = make_sensor(
        cam_pos=(0.0, 0.0, 4.2),
        look_at=(0.0, 0.0, 0.0),
        up=(0, 1, 0),
        fov=40,
        resx=800,
        resy=800
    )

    render_GT(render_spp, envmap_path, sensor)