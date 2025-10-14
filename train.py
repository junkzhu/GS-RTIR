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
    dataset = Dataset(DATASET_PATH)

    gaussians = GaussianModel()
    gaussians.restore_from_ply(PLY_PATH) #TODO:补充checkpoint读取

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    scene_dict = mi.load_dict({
        'type': 'scene',
        #TODO:换成可以优化的自定义光源
        'emitter': { 
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': ENVMAP_PATH,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        },
        'integrator': {
            'type': INTEGRATOR,
            'max_depth': 4,
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

    params = mi.traverse(scene_dict)
    params.keep(OPTIMIZE_PARAMS) 
    for _, param in params.items():
        dr.enable_grad(param)
    opt = mi.ad.Adam(lr=0.01, params=params)
    #opt.set_learning_rate({'shape.opacities':0.001})
    seed = 0

    pbar = tqdm.tqdm(range(NITER))
    for i in pbar:
        loss = mi.Float(0.0)
        
        for idx, sensor in dataset.get_sensor_iterator(i):
            img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=SPP * PRIMAL_SPP_MULT, spp_grad=SPP,
                                  seed=seed, seed_grad=seed+1+len(dataset.sensors))
            
            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            ref_albedo_img = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
            ref_roughness_img = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]][:, :, :1]
            ref_normal_img = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]

            #aovs
            albedo_img = aovs['albedo'][:, :, :3]
            roughness_img = aovs['roughness'][:, :, :1]
            metallic_img = aovs['metallic'][:, :, :1]
            depth_img = aovs['depth'][:, :, :1]
            normal_img = aovs['normal'][:, :, :3]

            view_loss = l1(img, ref_img) / dataset.batch_size
            albedo_loss = l1(albedo_img, ref_albedo_img) / dataset.batch_size
            roughness_loss = l1(roughness_img, ref_roughness_img) / dataset.batch_size

            #loss follow GS-IR
            view_tv_loss = TV(dr.concat([albedo_img, roughness_img, metallic_img], axis=2), ref_img) / dataset.batch_size
            lamb_loss = dr.mean(1.0-roughness_img) + dr.mean(metallic_img)

            #convert depth to fake_normal
            fake_normal_img = convert_depth_to_normal(depth_img, sensor)
            normal_loss = lnormal(normal_img, ref_normal_img) / dataset.batch_size
            normal_tv_loss = TV(ref_img, normal_img) / dataset.batch_size

            total_loss = view_loss + view_tv_loss + 0.001 * lamb_loss #+ normal_loss + normal_tv_loss + albedo_loss + roughness_loss

            dr.backward(total_loss)

            loss += total_loss

            #----------------------------------save the results----------------------------------
            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)
            albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
            roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
            #metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
            depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), dataset.target_res)

            normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)
            normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, normal_img, 0))), dataset.target_res) 

            mi.util.write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
            mi.util.write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_albedo' + ('.png')), albedo_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_roughness' + ('.png')), roughness_bmp)
            #mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_metallic' + ('.png')), metallic_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_depth' + ('.png')), depth_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_normal' + ('.png')), normal_bmp)            

        opt.step()
        params.update(opt)

        dataset.update_sensors(i)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        if (i+1) % 100 == 0:
            gaussians.restore_from_params(params)
            save_path = f"{OUTPUT_PLY_DIR}/iter_{i:03d}.ply"
            gaussians.save_ply(save_path)
            print(f"[Iter {i}] Saved PLY to {save_path}")