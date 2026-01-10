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

def load_scene_config(gaussians_attributes, envmap_init_path, optimize_envmap):
    global OPTIMIZE_PARAMS
    scene_config = {
        'type': 'scene',
        'integrator': {
            'type': args.integrator_type,
            'max_depth': args.max_bounce_num,
            'pt_rate': args.spp_pt_rate,
            'gaussian_max_depth': 128,
            'hide_emitters': args.hide_emitter,
            'use_mis': args.use_mis,
            'selfocc_offset_max': args.selfocc_offset_max,
            'geometry_threshold': args.geometry_threshold,
            'separate_direct_indirect': args.separate_direct_indirect,
        },
        'shape': {
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
    }

    if optimize_envmap:
        if args.spherical_gaussian:
            # register SG envmap
            SGModel(
                num_sgs = args.num_sgs,
                sg_init = np.load(envmap_init_path)
            )
            
            scene_config['envmap'] = {
                'type': 'vMF',
                'filename': '/home/zjk/datasets/TensoIR/Environment_Maps/high_res_envmaps_1k/sunset.hdr',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.lgtSGs*']
            OPTIMIZE_PARAMS += ['envmap.position'] + ['envmap.weight'] + ['envmap.std']
        
        else:
            scene_config['envmap'] = {
                'type': 'envmap',
                'filename': '/home/zjk/datasets/TensoIR/Environment_Maps/high_res_envmaps_1k/sunset.hdr',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.data']

    else:
        scene_config['emitter'] = {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': args.envmap_path,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        }
    
    return scene_config

def render_relight_images(gaussians, dataset, scene_dict, params):
    # update albedo
    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)
    params['shape.albedos'] = dr.ravel(gaussians_attributes['albedos'])

    envmaps = load_hdr_paths(args.envmap_root)
    
    # Ensure render directories exist
    ensure_dir(OUTPUT_RENDER_DIR)
    ensure_dir(OUTPUT_RELIGHT_DIR)
    
    for envmap in envmaps:
        #create folder
        envmap_name = Path(envmap).stem
        envmap_dir = os.path.realpath(os.path.join(OUTPUT_RELIGHT_DIR, f'./{envmap_name}'))
        ensure_dir(envmap_dir)

        #change envmap
        envmap_bmp = mi.Bitmap(envmap)
        bitmap = mi.TensorXf(np.array(envmap_bmp, dtype=np.float32))[..., :3]
        params['EnvironmentMapEmitter.data'] = bitmap
        params.update()

        relight_pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc=f"Relighting {envmap_name}")
        for idx, sensor in relight_pbar:
            if idx % args.stride != 0:
                continue
            
            # Check if the image already exists
            output_path = join(envmap_dir, f'{idx:02d}.png')
            if os.path.exists(output_path):
                continue  # Skip if image already exists
            
            img, aovs = mi.render(scene_dict, params=params, sensor=sensor, spp=args.render_spp) #img is linear space
            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            write_bitmap(output_path, rgb_bmp)

            if args.separate_direct_indirect:
                direct_light_img = aovs['direct_light'][:, :, :3]
                indirect_light_img = aovs['indirect_light'][:, :, :3]

                direct_light_bmp = resize_img(mi.Bitmap(direct_light_img),dataset.target_res)
                indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img),dataset.target_res)

                write_bitmap(join(envmap_dir, f'{idx:02d}_direct_light' + ('.png')), direct_light_bmp)
                write_bitmap(join(envmap_dir, f'{idx:02d}_indirect_light' + ('.png')), indirect_light_bmp)

def render_materials(dataset, scene_dict):
    albedo_list, ref_albedo_list = [], []
    
    # Ensure render directory exists
    ensure_dir(OUTPUT_RENDER_DIR)

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Rendering views")

    for idx, sensor in pbar:
        if idx % args.stride != 0:
            continue

        img, aovs = mi.render(scene_dict, sensor=sensor, spp=args.render_spp) #img is linear space

        #aovs
        albedo_img = aovs['albedo'][:, :, :3]
        roughness_img = aovs['roughness'][:, :, :1]
        metallic_img = aovs['metallic'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]

        normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)

        albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
        roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
        metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
        normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), dataset.target_res) 


        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_albedo_ori' + ('.png')), albedo_bmp)
        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_albedo_ref' + ('.png')), dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]])

        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_roughness' + ('.png')), roughness_bmp)
        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_roughness_ref' + ('.png')), dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]])

        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_metallic' + ('.png')), metallic_bmp)
        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_normal' + ('.png')), normal_bmp)

        if args.separate_direct_indirect:
            direct_light_img = aovs['direct_light'][:, :, :3]
            indirect_light_img = aovs['indirect_light'][:, :, :3]

            direct_light_bmp = resize_img(mi.Bitmap(direct_light_img),dataset.target_res)
            indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img),dataset.target_res)

            write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_direct_light' + ('.png')), direct_light_bmp)
            write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_indirect_light' + ('.png')), indirect_light_bmp)

        albedo_list.append(albedo_bmp)
        ref_albedo_list.append(dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]])

        rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}' + ('.png')), rgb_bmp)
        write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_ref' + ('.png')), dataset.ref_images[idx][sensor.film().crop_size()[0]])

    single_channel_ratio, three_channel_ratio = compute_rescale_ratio(ref_albedo_list, albedo_list)
    print("Albedo scale:", three_channel_ratio)

    for idx, albedo_img in enumerate(albedo_list):
        real_idx = idx * args.stride

        albedo_bmp = albedo_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
        albedo_img = three_channel_ratio * mi.TensorXf(albedo_bmp)
        albedo_bmp = mi.Bitmap(albedo_img)

        write_bitmap(join(OUTPUT_RENDER_DIR, f'{real_idx:02d}_albedo' + ('.png')), albedo_bmp)

    return three_channel_ratio

if __name__ == "__main__":
    with time_measure("Loading dataset"):
        dataset = Dataset(args.dataset_path, RENDER_UPSAMPLE_ITER, "test")

    with time_measure("Initializing gaussians"):
        gaussians = GaussianModel()
        gaussians.restore_from_ply(args.ply_path, False)

    with time_measure("Loading gaussian to ellipsoids factory"):
        ellipsoidsfactory = EllipsoidsFactory()
        gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    with time_measure("Loading scene config"):
        scene_config = load_scene_config(gaussians_attributes, args.envmap_init_path, args.envmap_optimization)
        scene_dict = mi.load_dict(scene_config)
        params = mi.traverse(scene_dict)

    #---------------render materials-----------------
    three_channel_ratio = render_materials(dataset, scene_dict)

    #---------------rescale albedo-----------------
    gaussians.rescale_albedo(three_channel_ratio)

    save_path = args.ply_path.replace('.ply', '_rescaled.ply')
    gaussians.save_ply(save_path)

    #---------------relight-----------------
    if args.relight:
        with time_measure("Relighting: Initializing gaussians"):
            gaussians = GaussianModel()
            args.ply_path = args.ply_path.replace(".ply", "_rescaled.ply")
            gaussians.restore_from_ply(args.ply_path, False)

        with time_measure("Relighting: Loading gaussian to ellipsoids factory"):
            ellipsoidsfactory = EllipsoidsFactory()
            gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

        with time_measure("Relighting: Loading scene config"):
            scene_config = load_scene_config(gaussians_attributes, args.envmap_init_path, False)
            scene_dict = mi.load_dict(scene_config)
            params = mi.traverse(scene_dict)

        render_relight_images(gaussians, dataset, scene_dict, params)