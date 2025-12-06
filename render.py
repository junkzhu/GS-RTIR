import torch
import tqdm
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *

from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

if __name__ == "__main__":

    dataset = Dataset(DATASET_PATH, RENDER_UPSAMPLE_ITER, "test")

    gaussians = GaussianModel()
    gaussians.restore_from_ply(PLY_PATH, False)

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    scene_dict = mi.load_dict({
        'type': 'scene',
        'emitter': { 
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': ENVMAP_PATH,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        },
        'integrator': {
            'type': INTEGRATOR,
            'max_depth': MAX_BOUNCE_NUM,
            'gaussian_max_depth': 128,
            'hide_emitters': True,
            'use_mis': True
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
    })

    albedo_list, ref_albedo_list = [], []

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Rendering views")

    for idx, sensor in pbar:
        img, aovs = mi.render(scene_dict, sensor=sensor, spp=RENDER_SPP)

        ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
        ref_albedo = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
        ref_roughness = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]
        
        #aovs
        albedo_img = aovs['albedo'][:, :, :3]
        roughness_img = aovs['roughness'][:, :, :1]
        metallic_img = aovs['metallic'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]

        normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)
        normal_mask_flat = np.reshape(normal_mask, (-1,1)).squeeze()

        rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
        rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)
        albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
        roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
        #metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
        normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), dataset.target_res) 

        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_rgb' + ('.png')), rgb_bmp)
        #mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_albedo' + ('.png')), albedo_bmp)
        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_roughness' + ('.png')), roughness_bmp)
        #mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_metallic' + ('.png')), metallic_bmp)
        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_normal' + ('.png')), normal_bmp)

        albedo_list.append(albedo_bmp)
        ref_albedo_list.append(dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]])

    single_channel_ratio, three_channel_ratio = compute_rescale_ratio(ref_albedo_list, albedo_list)
    
    print("albedo_scale:", three_channel_ratio)

    for idx, albedo_img in enumerate(albedo_list):
        albedo_img = three_channel_ratio * albedo_img
        albedo_bmp = mi.Bitmap(albedo_img)

        mi.util.write_bitmap(join(OUTPUT_RENDER_DIR, f'{idx:02d}_albedo' + ('.png')), albedo_bmp)

    #---------------rescale albedo-----------------
    gaussians.rescale_albedo(three_channel_ratio)

    save_path = PLY_PATH.replace('.ply', '_rescaled.ply')
    gaussians.save_ply(save_path)