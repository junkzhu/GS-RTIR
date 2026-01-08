import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tqdm
import numpy as np
import os
from os.path import join
from pathlib import Path

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *

from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

def load_scene_config(gaussians_attributes):
    global OPTIMIZE_PARAMS
    scene_config = {
        'type': 'scene',
        'integrator': {
            'type': 'hybrid_rt',
            'max_depth': 2,
            'pt_rate': 1.0,
            'gaussian_max_depth': 128,
            'hide_emitters': True,
            'use_mis': True,
            'selfocc_offset_max': 0.1,
            'geometry_threshold': 0.2,
            'separate_direct_indirect': True,
        },
        'gaussians': {
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
        },
        'rectangle': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, -0.5]), # .scale([2, 2, 1])
            'material': {
                'type': 'diffuse'
                }
            }
        # 'sphere': {
        #     'type': 'sphere',
        #     'center': [0, 2, 0],
        #     'radius': 0.5,
        #     'bsdf': {
        #         'type': 'diffuse'
        #     }
        # }
    }

    scene_config['emitter'] = {
        'type': 'envmap',
        'id': 'EnvironmentMapEmitter',
        'filename': args.envmap_path, #TODO:修改成--envmap_path
        'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                    mi.ScalarTransform4f.rotate([1, 0, 0], 90)
    }
    
    return scene_config

def render_scene(dataset, scene_dict):    
    # Ensure render directory exists
    ensure_dir(OUTPUT_HYBRID_DIR)

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Rendering the scene")

    for idx, sensor in pbar:
        img, aovs = mi.render(scene_dict, sensor=sensor, spp=32) #img is linear space #TODO

        #aovs
        albedo_img = aovs['albedo'][:, :, :3]
        roughness_img = aovs['roughness'][:, :, :1]
        #metallic_img = aovs['metallic'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]

        normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)

        albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
        roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
        #metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
        normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), dataset.target_res) 

        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_albedo' + ('.png')), albedo_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_roughness' + ('.png')), roughness_bmp)
        #write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_metallic' + ('.png')), metallic_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_normal' + ('.png')), normal_bmp)

        if args.separate_direct_indirect:
            direct_light_img = aovs['direct_light'][:, :, :3]
            indirect_light_img = aovs['indirect_light'][:, :, :3]

            direct_light_bmp = resize_img(mi.Bitmap(direct_light_img),dataset.target_res)
            indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img),dataset.target_res)

            write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_direct_light' + ('.png')), direct_light_bmp)
            write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_indirect_light' + ('.png')), indirect_light_bmp)

        rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}' + ('.png')), rgb_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_ref' + ('.png')), dataset.ref_images[idx][sensor.film().crop_size()[0]])

if __name__ == "__main__":
    with time_measure("Loading dataset"):
        dataset = Dataset('/home/zjk/datasets/TensoIR/armadillo', RENDER_UPSAMPLE_ITER, "train")

    with time_measure("Initializing gaussians"):
        gaussians = GaussianModel()
        gaussians.restore_from_ply('/home/zjk/code/GS-RTIR/outputs/TensoIR/armadillo#0/ply/iter_799_rescaled.ply', False) #TODO

    with time_measure("Loading gaussian to ellipsoids factory"):
        ellipsoidsfactory = EllipsoidsFactory()
        gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    with time_measure("Loading scene config"):
        scene_config = load_scene_config(gaussians_attributes)
        scene_dict = mi.load_dict(scene_config)
        params = mi.traverse(scene_dict)

    render_scene(dataset, scene_dict)