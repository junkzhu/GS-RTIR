import torch
import tqdm
import numpy as np
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

def load_scene_config(envmap_init_path):
    global OPTIMIZE_PARAMS
    scene_config = {
        'type': 'scene',
        'integrator': {
            'type': INTEGRATOR,
            'max_depth': MAX_BOUNCE_NUM,
            'pt_rate': SPP_PT_RATE,
            'gaussian_max_depth': 128,
            'hide_emitters': HIDE_EMITTER,
            'use_mis': USE_MIS
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

    if OPTIMIZE_ENVMAP:
        if SPHERICAL_GAUSSIAN:
            # register SG envmap
            SGModel(
                num_sgs = NUM_SGS,
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
                'filename': 'D:/dataset/Environment_Maps/high_res_envmaps_1k/init_1280_640.exr',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.data']

    else:
        scene_config['emitter'] = {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': ENVMAP_PATH,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        }
    
    return scene_config

def render_relight_images(gaussians, dataset, scene_dict, params):
    # update albedo
    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)
    params['shape.albedos'] = dr.ravel(gaussians_attributes['albedos'])

    envmaps = load_hdr_paths(ENVMAP_ROOT)
    
    for envmap in envmaps:
        #create folder
        envmap_name = Path(envmap).stem
        envmap_dir = os.path.realpath(os.path.join(OUTPUT_RELIGHT_DIR, f'./{envmap_name}'))
        os.makedirs(envmap_dir, exist_ok=True)

        #change envmap
        bitmap = mi.TensorXf(mi.Bitmap(envmap))
        params['EnvironmentMapEmitter.data'] = bitmap

        relight_pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc=f"Relighting {envmap_name}")
        for idx, sensor in relight_pbar:
            img, _ = mi.render(scene_dict, params=params, sensor=sensor, spp=RENDER_SPP)
            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            mi.util.write_bitmap(join(envmap_dir, f'{idx:02d}' + ('.png')), rgb_bmp)

def render_materials(dataset, scene_dict, integrator):
    albedo_list, ref_albedo_list = [], []

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Rendering views")

    for idx, sensor in pbar:
        img, aovs = mi.render(scene_dict, sensor=sensor, spp=RENDER_SPP, integrator=integrator)
        
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

        #mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_albedo' + ('.png')), albedo_bmp)
        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_roughness' + ('.png')), roughness_bmp)
        #mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_metallic' + ('.png')), metallic_bmp)
        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_normal' + ('.png')), normal_bmp)

        albedo_list.append(albedo_bmp)
        ref_albedo_list.append(dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]])

        rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}' + ('.png')), rgb_bmp)

    single_channel_ratio, three_channel_ratio = compute_rescale_ratio(ref_albedo_list, albedo_list)
    print("Albedo scale:", three_channel_ratio)

    for idx, albedo_img in enumerate(albedo_list):
        albedo_bmp = albedo_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
        albedo_img = three_channel_ratio * mi.TensorXf(albedo_bmp)
        albedo_bmp = mi.Bitmap(albedo_img)

        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_albedo' + ('.png')), albedo_bmp)

    return three_channel_ratio

if __name__ == "__main__":

    dataset = Dataset(DATASET_PATH, RENDER_UPSAMPLE_ITER, "test")

    gaussians = GaussianModel()
    gaussians.restore_from_ply(PLY_PATH, False)

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    scene_dict = load_scene_config(ENVMAP_INIT_PATH)
    params = mi.traverse(scene_dict)

    aov_integrator = mi.load_dict({
        'type': 'gsprim_prb',
        'max_depth': 1,
        'gaussian_max_depth': 128,
        'hide_emitters': True,
        'use_mis': False
    })

    #---------------render materials-----------------
    three_channel_ratio = render_materials(dataset, scene_dict, aov_integrator)

    #---------------rescale albedo-----------------
    gaussians.rescale_albedo(three_channel_ratio)

    save_path = PLY_PATH.replace('.ply', '_rescaled.ply')
    gaussians.save_ply(save_path)

    #---------------relight-----------------
    if RELIGHT:
        render_relight_images(gaussians, dataset, scene_dict, params)