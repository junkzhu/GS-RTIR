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

from tqdm import trange

# Output directory configuration
OUTPUT_DIR = join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'teaser')
ensure_dir(OUTPUT_DIR)
ensure_dir(join(OUTPUT_DIR,'./rgb/'))
ensure_dir(join(OUTPUT_DIR,'./albedo/'))
ensure_dir(join(OUTPUT_DIR,'./roughness/'))
ensure_dir(join(OUTPUT_DIR,'./metallic/'))
ensure_dir(join(OUTPUT_DIR,'./direct_light/'))
ensure_dir(join(OUTPUT_DIR,'./indirect_light/'))
ensure_dir(join(OUTPUT_DIR,'./final/'))
ensure_dir(join(OUTPUT_DIR,'./normal/'))

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
            'selfocc_offset_max': 0.5,
            'selfocc_mode': 'normal',
            'selfocc_offset_fixed': 0.1,
            'geometry_threshold': 0.2,
            'separate_direct_indirect': True,
        },
        # 'rectangle': {
        #     'type': 'rectangle',
        #     'to_world': mi.ScalarTransform4f().scale([5, 8, 1]).translate([0, 0, 0]),
        #     'material': {
        #         'type': 'principled',
        #         'base_color': {'type': 'rgb', 'value': [0.05, 0.05, 0.05]},
        #         'metallic': 1.0,
        #         'roughness': 0.05,
        #         'specular': 0.5
        #     }
        # },
        'emitter': {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': envmap_path,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 180) @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        },
        # 'sphere_area_light': {
        #     'type': 'sphere',
        #     'center': [0, 0, 1],
        #     'radius': 0.3,
        #     'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': 10.0}}
        # }

        # box
        "matellic_gray": {
            'type': 'principled',
            'base_color': {
                'type': 'rgb',
                'value': [0.78, 0.78, 0.78]
            },
            'metallic': 0.8,
            'roughness': 0.01,
            'specular': 0.0
        },

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

        "orange": {
            "type": "principled",
            "base_color": {
                "type": "rgb",
                "value": [0.90, 0.45, 0.10]
            },
            "metallic": 0.0,
            "roughness": 1.0,
            "specular": 0.0
        },


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

        "teaser_box": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/teaser_box.obj",
            "to_world": (
                mi.ScalarTransform4f()
                .scale([0.8, 0.6, 0.6])
                .rotate([1, 0, 0], 90)
                .rotate([0, 1, 0], 180)
                .translate([0, 0, 0])
            ),
            "bsdf": {"type": "ref", "id": "white"}
        },

        "metallic_box": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/box.obj",
            "to_world": (
                mi.ScalarTransform4f()
                .translate([2.75, 0.0, -2.5])
                .scale([2.0, 2.0, 0.8])
                .rotate([0, 0, 1], 15)
            ),
            "bsdf": {"type": "ref", "id": "matellic_gray"}
        },

        # "orange_box": {
        #     "type": "obj",
        #     "filename": "./demos/scenes/meshes/box.obj",
        #     "to_world": (
        #         mi.ScalarTransform4f()
        #         .translate([1.0, -1.5, -2.3])
        #         .scale([2.0, 2.0, 2.0])
        #         .rotate([0, 0, 0], 60)
        #     ),
        #     "bsdf": {"type": "ref", "id": "orange"}
        # },

        "area_light_box": {
            "type": "obj",
            "filename": "./demos/scenes/meshes/box.obj",
            "to_world": (
                mi.ScalarTransform4f()
                .translate([0.1, -3.3, -2.0])
                .scale([0.3, 0.3, 0.5])
                .rotate([0, 0, 1], 60)
            ),
            "bsdf": {"type": "ref", "id": "white"},
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [9, 6.5, 3]
                }
            }
        },

        # "dragon": {
        #     "type": "obj",
        #     "filename": "./demos/scenes/meshes/dragon.obj",
        #     "to_world": mi.ScalarTransform4f.scale([0.015, 0.015, 0.015]).translate([100, -200, -160]).rotate([1, 0 ,0], 90).rotate([0, 1 ,0], -120),
        #     "bsdf": {
        #         "type": "ref",
        #         "id": "white"
        #     }
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

def render_define_scene(scene_dict, render_spp, sensor):
    """Render the scene with the given sensor"""
    
    # Render scene
    integrator = mi.load_dict({
        'type': 'gsprim_prb',
        'max_depth': 3,
        'pt_rate': 1.0,
        'gaussian_max_depth': 128,
        'hide_emitters': False,
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

    for i in trange(render_spp, desc="Rendering", unit="spp"):
        integrator.render_process(scene_dict, sensor=sensor, spp=1, seed=i)

        buffer = film.develop()

        dr.eval(buffer)

        rgb, aovs = unpack_buffer(buffer)
        
        mi.util.write_bitmap(join(OUTPUT_DIR,'./rgb/', f'{i}.png'), rgb)
        mi.util.write_bitmap(join(OUTPUT_DIR, './albedo/', f'{i}.png'), aovs['albedo'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './roughness/', f'{i}.png'), aovs['roughness'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './metallic/', f'{i}.png'), aovs['metallic'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './direct_light/', f'{i}.png'), aovs['direct_light'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './indirect_light/', f'{i}.png'), aovs['indirect_light'])
        mi.util.write_bitmap(join(OUTPUT_DIR, './normal/', f'{i}.png'), aovs['normal'])

        del buffer, rgb, aovs
        gc.collect()

    buffer = film.develop()
    
    rgb, aovs = unpack_buffer(buffer)
    albedo = aovs['albedo']
    roughness = aovs['roughness']
    metallic = aovs['metallic']
    direct = aovs['direct_light']
    indirect = aovs['indirect_light']
    normal = aovs['normal']

    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'rgb.png'), rgb)
    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'albedo.png'), albedo)
    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'roughness.png'), roughness)
    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'metallic.png'), metallic)
    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'direct.png'), direct)
    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'indirect.png'), indirect)
    mi.util.write_bitmap(join(OUTPUT_DIR, './final/',f'normal.png'), normal)


def get_model_configs():
    """Get the 3D model configurations"""
    return [
        # TensoIR models
        {
            'ply_path': './demos/scenes/TensoIR/lego.ply',
            'rotation': [0, 0, -10],
            'scale': [0.7, 0.7, 0.7],
            'translation': [4, -0.1, -1.52]
        },
        {
            'ply_path': './demos/scenes/TensoIR/armadillo.ply',
            'rotation': [0, 0, 50],
            'scale': [1.8, 1.8, 1.8],
            'translation': [2.5, -3.5, -1]
        },
        {
            'ply_path': './demos/scenes/TensoIR/ficus.ply',
            'rotation': [0, 0, 180],
            'scale': [1.2, 1.2, 1.2],
            'translation': [3.0, 3.4, -1.15]
        },
        {
            'ply_path': './demos/scenes/TensoIR/hotdog.ply',
            'rotation': [0, 0, 90],
            'scale': [0.5, 0.5, 0.5],
            'translation': [-1.5, -3.3, -2.44]
        },
        # Synthetic4Relight models
        {
            'ply_path': './demos/scenes/Synthetic4Relight/air_baloons.ply',
            'rotation': [0, 0, 180],
            'scale': [0.7, 0.7, 0.7],
            'translation': [7.0, -2.3, 0.0]
        },
        {
            'ply_path': './demos/scenes/Synthetic4Relight/chair.ply',
            'rotation': [0, 0, -120],
            'scale': [0.6, 0.6, 0.6],
            'translation': [4.0, 1.5, -1.5]
        },
        {
            'ply_path': './demos/scenes/Synthetic4Relight/jugs.ply',
            'rotation': [0, 0, 70],
            'scale': [1.0, 1.3, 1.0],
            'translation': [-0.3, -1.5, -2.4]
        },

        {
            'ply_path': './demos/scenes/RT4Relight/bear.ply',
            'rotation': [0, 0, 100],
            'scale': [0.6, 0.6, 0.6],
            'translation': [1.5, 2.7, -2.2]
        },

        {
            'ply_path': './demos/scenes/RT4Relight/bread.ply',
            'rotation': [0, 0, 30],
            'scale': [0.6, 0.6, 0.6],
            'translation': [2.2, 4.5, -2.12]
        },

        {
            'ply_path': './demos/scenes/Metallic/hollow.ply',
            'rotation': [0, -30, 0],
            'scale': [0.4, 0.4, 0.4],
            'translation': [2.5, -1.5, -1.3]
        },

        # {
        #     'ply_path': './demos/scenes/Metallic/horse.ply',
        #     'rotation': [0, 0, -90],
        #     'scale': [0.8, 0.8, 0.8],
        #     'translation': [2.0, 2.0, 0.05]
        # }
    ]

def render_teaser(render_spp, envmap_path, sensor):
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
    render_define_scene(scene_dict, render_spp, sensor)

if __name__ == "__main__":
    # Configuration parameters
    render_spp = 999999
    envmap_path = './demos/scenes/envmaps/bloem_train_track_cloudy_4k.exr'
    
    # Camera parameters
    sensor = make_sensor(
        cam_pos=(-15, 6.0, 2.0),
        look_at=(0.5, 0.0, -0.5),
        up=(0, 0, 1),
        fov=30,
        resx=1600,
        resy=800
    )

    render_teaser(render_spp, envmap_path, sensor)