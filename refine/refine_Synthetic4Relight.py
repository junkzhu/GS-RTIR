import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tqdm
import numpy as np
from os.path import join
from omegaconf import OmegaConf

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *

import optimizers

from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

opacities_scale = 2.0

if __name__ == "__main__":
    refine_conf = OmegaConf.load('configs/refine.yaml')
    dataset = Dataset(DATASET_PATH, REFINE_UPSAMPLE_ITER)

    gaussians = GaussianModel()
    if PLY_PATH.endswith(".ply"):
        gaussians.restore_from_ply(PLY_PATH, RESET_ATTRIBUTE)
    elif PLY_PATH.endswith(".pt"):
        gaussians.restore_from_ckpt(PLY_PATH)
    else:
        raise ValueError(f"Unsupported file type: {PLY_PATH}")

    gsstrategy = GSStrategyModel('configs/gs.yaml')
    gsstrategy.sensors_normal = dataset.sensors_normal
    gsstrategy.sensors_intrinsic = dataset.sensors_intrinsic

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    scene_dict = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'volprim_refine',
            'max_depth': 128
        },
        'shape': {
            'type': 'ellipsoidsmesh',
            'centers': gaussians_attributes['centers'],
            'scales': gaussians_attributes['scales'],
            'quaternions': gaussians_attributes['quats'],
            'opacities': gaussians_attributes['sigmats'],
            'sh_coeffs': gaussians_attributes['features'],

            'normals': gaussians_attributes['normals']
        }
    })

    params = mi.traverse(scene_dict)
    params.keep(REFINE_PARAMS) 
    for _, param in params.items():
        dr.enable_grad(param)
    
    #clear opacity
    n_ellipsoids = params['shape.opacities'].shape[0]
    params['shape.opacities'] = params['shape.opacities'] * opacities_scale
    #params['shape.opacities'] = dr.full(mi.Float, 1.0, n_ellipsoids)

    #clear sh & normal
    m_sh_coeffs = params['shape.sh_coeffs'].shape[0] // n_ellipsoids
    m_normals = params['shape.normals'].shape[0] // n_ellipsoids
    #params['shape.sh_coeffs'] = dr.full(mi.Float, 0.0, n_ellipsoids * m_sh_coeffs)
    params['shape.normals'] = dr.full(mi.Float, 0.1, n_ellipsoids * m_normals)
    
    opt = optimizers.BoundedAdam()
    ellipsoids = Ellipsoid.unravel(params['shape.data'])
    opt['centers'] = ellipsoids.center
    opt['scales']  = ellipsoids.scale
    opt['quats']   = mi.Vector4f(ellipsoids.quat)
    opt['opacities'] = params['shape.opacities']
    opt['sh_coeffs'] = params['shape.sh_coeffs']
    opt['normals'] = params['shape.normals']

    opt.set_learning_rate({
        'centers':   refine_conf.optimizer.params.centers_lr,
        'scales':    refine_conf.optimizer.params.scales_lr,
        'quats':     refine_conf.optimizer.params.quats_lr,
        'opacities': refine_conf.optimizer.params.opacities_lr,
        'sh_coeffs': refine_conf.optimizer.params.sh_coeffs_lr,
        'normals': refine_conf.optimizer.params.normals_lr
    })

    opt.set_bounds('scales',    lower=1e-6, upper=1e2)
    opt.set_bounds('opacities', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('sh_coeffs', lower=-1, upper=1)
    opt.set_bounds('normals', lower=-1, upper=1)

    def update_params(opt):
        params['shape.data'] = Ellipsoid.ravel(opt['centers'], opt['scales'], mi.Quaternion4f(opt['quats']))
        params['shape.opacities'] = opt['opacities']
        params['shape.sh_coeffs'] = opt['sh_coeffs']
        params['shape.normals'] = opt['normals']
        params.update()

    update_params(opt)

    seed = 0

    psnr_list = []
    mae_list = []

    pbar = tqdm.tqdm(range(refine_conf.optimizer.iterations))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        normal_mae = mi.Float(0.0)
        
        gsstrategy.lr_schedule(opt, i, refine_conf.optimizer.iterations, refine_conf.optimizer.scheduler.min_factor)

        for idx, sensor in dataset.get_sensor_iterator(i):
            img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=8, spp_grad=1,
                                  seed=seed, seed_grad=seed+1+len(dataset.sensors))
            
            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            
            normal_priors_img = dataset.normal_priors_images[idx][sensor.film().crop_size()[0]]

            #aovs
            depth_img = aovs['depth'][:, :, :1]
            normal_img = aovs['normal'][:, :, :3]
            normal_mask = np.any(normal_priors_img != 0, axis=2, keepdims=True)
            normal_mask_flat = np.reshape(normal_mask, (-1,1)).squeeze()
            
            view_loss = l1(ref_img, img) / dataset.batch_size

            # normal priors
            normal_priors_loss = l2(normal_priors_img, normal_img) / dataset.batch_size

            # convert depth to fake_normal
            fake_normal_img = convert_depth_to_normal(depth_img, sensor)
            normal_loss = lnormal_sqr(fake_normal_img, normal_img, normal_mask_flat) / dataset.batch_size
            normal_tv_loss = TV(ref_img, normal_img) / dataset.batch_size

            fake_normal_loss = lnormal_sqr(normal_priors_img, fake_normal_img, normal_mask_flat) / dataset.batch_size

            # encourage opacity to be 0 or 1
            opacity_loss = opacity_entropy_loss(opt['opacities'])

            total_loss = view_loss + 0.1 * normal_loss + normal_priors_loss + normal_tv_loss + fake_normal_loss + 0.1 * opacity_loss

            dr.backward(total_loss)
            
            loss += total_loss

            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)
            depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), dataset.target_res)
            normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), dataset.target_res)
            fake_normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (fake_normal_img+1)/2, 0))), dataset.target_res)

            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_depth' + ('.png')), depth_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_normal' + ('.png')), normal_bmp)      
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_fake_normal' + ('.png')), fake_normal_bmp)      

            rgb_psnr += lpsnr(ref_img, img) / dataset.batch_size
            normal_mae += lmae(normal_priors_img, normal_img, normal_mask.squeeze()) / dataset.batch_size

        #grad restraint
        # grad = dr.grad(params['shape.data'])
        # grad_clamped = dr.clip(grad, -1e-4, 1e-4)
        # dr.set_grad(params['shape.data'], grad_clamped)

        gsstrategy.update_grad_norm(opt=opt)

        opt.step()

        gsstrategy.update_gs(opt=opt, step=i)
        update_params(opt)

        psnr_list.append(rgb_psnr)
        mae_list.append(normal_mae)

        #scale value restraint
        data = params['shape.data']
        data_np = np.array(data)
        N = data_np.shape[0] // 10
        data_np = data_np.reshape(N, 10)
        data_np[:, 3:6] = np.clip(data_np[:, 3:6], 1e-3, 0.05)
        params['shape.data'] = dr.cuda.ad.Float(data_np.reshape(-1))

        dataset.update_sensors(i)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        pbar.set_postfix({'rgb': rgb_psnr, 'normal': normal_mae})

        if (i > 0) and (i % 50 == 0): 
            gaussians.restore_from_params(params)
            gaussians.save_ply(REFINE_PATH)

    plot_loss(psnr_list, label='PSNR', output_file=join(OUTPUT_REFINE_DIR, 'psnr.png'))
    plot_loss(mae_list, label='MAE', output_file=join(OUTPUT_REFINE_DIR, 'mae.png'))

    #save ply
    gaussians.restore_from_params(params)
    gaussians.save_ply(REFINE_PATH)
