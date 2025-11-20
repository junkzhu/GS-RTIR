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
from emitter import *
from datasets import *
from losses import *

if __name__ == "__main__":
    dataset = Dataset(DATASET_PATH)

    gaussians = GaussianModel()
    gaussians.restore_from_ply(PLY_PATH, RESET_ATTRIBUTE) #TODO:补充checkpoint读取

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    # original envmap
    envmap = mi.Bitmap(ENVMAP_PATH)
    envmap = np.array(envmap)
    mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'ref' + ('.exr')), envmap)


    #---------------------------------- config ----------------------------------
    scene_config = {
        'type': 'scene',
        'integrator': {
            'type': INTEGRATOR,
            'max_depth': MAX_BOUNCE_NUM,
            'pt_rate': SPP_PT_RATE,
            'gaussian_max_depth': 128,
            'hide_emitters': True,
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
        scene_config['envmap'] = {
            'type': 'vMF',
            'filename': 'D:/dataset/Environment_Maps/high_res_envmaps_1k/sunset.hdr',
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        }
        OPTIMIZE_PARAMS += ['envmap.lgtSGs*']
    else:
        scene_config['emitter'] = {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': ENVMAP_PATH,
            'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], 90)
        }

    scene_dict = mi.load_dict(scene_config)

    params = mi.traverse(scene_dict)
    params.keep(OPTIMIZE_PARAMS)
    for _, param in params.items():
        dr.enable_grad(param)
    opt = mi.ad.Adam(lr=0.01, params=params)
    opt.set_learning_rate({'shape.normals':0.001})
    seed = 0

    loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list = [], [], [], [], []

    pbar = tqdm.tqdm(range(NITER))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        albedo_psnr = mi.Float(0.0)
        roughness_mse = mi.Float(0.0)
        normal_mae = mi.Float(0.0)
        
        for idx, sensor in dataset.get_sensor_iterator(i):
            img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=SPP * PRIMAL_SPP_MULT, spp_grad=SPP,
                                  seed=seed, seed_grad=seed + 1 + len(dataset.sensors))
            
            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            ref_albedo = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
            ref_roughness = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]
            ref_normal = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]
            
            albedo_priors_img = dataset.albedo_priors_images[idx][sensor.film().crop_size()[0]]
            roughness_priors_img = dataset.roughness_priors_images[idx][sensor.film().crop_size()[0]]
            normal_priors_img = dataset.normal_priors_images[idx][sensor.film().crop_size()[0]]

            #aovs
            albedo_img = aovs['albedo'][:, :, :3]
            roughness_img = aovs['roughness'][:, :, :1]
            metallic_img = aovs['metallic'][:, :, :1]
            depth_img = aovs['depth'][:, :, :1]
            normal_img = aovs['normal'][:, :, :3]
            
            normal_mask = np.any(ref_normal != 0, axis=2, keepdims=True)
            normal_mask_flat = np.reshape(normal_mask, (-1,1)).squeeze()

            view_loss = l1(img, ref_img) / dataset.batch_size

            #priors loss
            albedo_priors_loss = l2(albedo_img, albedo_priors_img) / dataset.batch_size
            roughness_priors_loss = l2(roughness_img, roughness_priors_img) / dataset.batch_size
            normal_priors_loss = l2(normal_img, normal_priors_img) / dataset.batch_size
            priors_loss = albedo_priors_loss + roughness_priors_loss + normal_priors_loss

            #loss follow GS-IR
            view_tv_loss = TV(dr.concat([albedo_img, roughness_img, metallic_img], axis=2), ref_img) / dataset.batch_size
            lamb_loss = dr.mean(1.0-roughness_img) / dataset.batch_size

            #convert depth to fake_normal
            fake_normal_img = convert_depth_to_normal(depth_img, sensor)
            normal_loss = lnormal_sqr(normal_img, fake_normal_img, normal_mask_flat) / dataset.batch_size
            
            #tv loss
            albedo_tv_loss = TV(albedo_priors_img, albedo_img) / dataset.batch_size
            roughness_tv_loss = TV(roughness_priors_img, roughness_img) / dataset.batch_size
            normal_tv_loss = TV(normal_priors_img, normal_img) / dataset.batch_size
            tv_loss = view_tv_loss + albedo_tv_loss + roughness_tv_loss + normal_tv_loss
  
            #total loss
            total_loss = view_loss + 0.1 * normal_loss + 0.05 * priors_loss + 0.01 * tv_loss + 0.001 * lamb_loss

            dr.backward(total_loss)

            loss += total_loss

            #----------------------------------save the results----------------------------------
            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)
            albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
            roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
            #metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
            depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), dataset.target_res)
            normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), dataset.target_res) 

            mi.util.write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
            mi.util.write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_albedo' + ('.png')), albedo_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_roughness' + ('.png')), roughness_bmp)
            #mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_metallic' + ('.png')), metallic_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_depth' + ('.png')), depth_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_normal' + ('.png')), normal_bmp)            

            rgb_psnr += lpsnr(ref_img, img) / dataset.batch_size
            albedo_psnr += lpsnr(ref_albedo, albedo_img) / dataset.batch_size
            roughness_mse += l2(ref_roughness, roughness_img) / dataset.batch_size
            normal_mae += lmae(ref_normal, normal_img, normal_mask.squeeze()) / dataset.batch_size
            
            loss_list.append(np.asarray(total_loss))
            normal_MAE_list.append(np.asarray(normal_mae))
            rgb_PSNR_list.append(np.asarray(rgb_psnr))
            albedo_PSNR_list.append(np.asarray(albedo_psnr))
            roughness_MSE_list.append(np.asarray(roughness_mse))

        opt.step()
        params.update(opt)

        dataset.update_sensors(i)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        pbar.set_postfix({'rgb': rgb_psnr, 'albedo': albedo_psnr, 'roughness': roughness_mse, 'normal': normal_mae})


        # save envmap
        if OPTIMIZE_ENVMAP:
            envmap_img = render_envmap_bitmap(params)
            mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'{i:04d}' + ('.png')), envmap_img)


        if (i+1) in dataset.render_upsample_iter:
            plot_loss(loss_list, label='Total Loss', output_file=join(OUTPUT_DIR, 'total_loss.png'))
            plot_loss(rgb_PSNR_list, label = "RGB PSNR", output_file=join(OUTPUT_DIR, 'rgb_psnr.png'))
            plot_loss(albedo_PSNR_list, label='Albedo PSNR', output_file=join(OUTPUT_DIR, 'albedo_psnr.png'))
            plot_loss(roughness_MSE_list, label='Roughness MSE', output_file=join(OUTPUT_DIR, 'roughness_mse.png'))
            plot_loss(normal_MAE_list, label='Normal MAE', output_file=join(OUTPUT_DIR, 'normal_mae.png'))

            gaussians.restore_from_params(params)
            save_path = f"{OUTPUT_PLY_DIR}/iter_{i:03d}.ply"
            gaussians.save_ply(save_path)
            print(f"[Iter {i}] Saved PLY to {save_path}")