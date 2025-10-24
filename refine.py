import torch
import tqdm
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *

import optimizers

from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

if __name__ == "__main__":
    dataset = Dataset(DATASET_PATH, REFINE_UPSAMPLE_ITER)

    gaussians = GaussianModel()
    gaussians.restore_from_ply(PLY_PATH)

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
    params['shape.opacities'] = dr.full(mi.Float, 0.01, n_ellipsoids)

    #clear sh & normal
    m_sh_coeffs = params['shape.sh_coeffs'].shape[0] // n_ellipsoids
    m_normals = params['shape.normals'].shape[0] // n_ellipsoids
    params['shape.sh_coeffs'] = dr.full(mi.Float, 0.0, n_ellipsoids * m_sh_coeffs)
    params['shape.normals'] = dr.full(mi.Float, 0.1, n_ellipsoids * m_normals)
    
    opt = optimizers.BoundedAdam(lr=0.0001, params=params)
    opt.set_learning_rate({'shape.opacities':0.02})
    opt.set_learning_rate({'shape.sh_coeffs':0.02})
    opt.set_learning_rate({'shape.normals':0.02})

    opt.set_bounds('shape.scales',    lower=1e-6)
    opt.set_bounds('shape.opacities', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('shape.sh_coeffs', lower=-1, upper=1)
    opt.set_bounds('shape.normals', lower=-1, upper=1)

    seed = 0

    pbar = tqdm.tqdm(range(REFINE_NITER))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        normal_mae = mi.Float(0.0)
        
        for idx, sensor in dataset.get_sensor_iterator(i):
            img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=1, spp_grad=1,
                                  seed=seed, seed_grad=seed+1+len(dataset.sensors))
            
            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            ref_normal = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]
            
            normal_priors_img = dataset.normal_priors_images[idx][sensor.film().crop_size()[0]]

            #aovs
            depth_img = aovs['depth'][:, :, :1]
            normal_img = aovs['normal'][:, :, :3]
            
            view_loss = l1(ref_img, img) / dataset.batch_size

            # normal priors
            normal_priors_loss = l2(normal_priors_img, normal_img) / dataset.batch_size

            # convert depth to fake_normal
            fake_normal_img = convert_depth_to_normal(depth_img, sensor)
            normal_loss = lnormal(normal_img, fake_normal_img) / dataset.batch_size
            normal_tv_loss = TV(ref_img, normal_img) / dataset.batch_size

            # encourage opacity to be 0 or 1
            opacity_loss = opacity_entropy_loss(params['shape.opacities'])

            total_loss = view_loss + normal_priors_loss + normal_loss + normal_tv_loss + 0.1 * opacity_loss

            dr.backward(total_loss)
            
            loss += total_loss

            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)
            depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), dataset.target_res)
            normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)
            normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, normal_img, 0))), dataset.target_res) 

            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_depth' + ('.png')), depth_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_normal' + ('.png')), normal_bmp)            

            rgb_psnr += lpsnr(ref_img, img) / dataset.batch_size
            normal_mae += lmae(ref_normal, normal_img) / dataset.batch_size

        opt.step()
        params.update(opt)

        dataset.update_sensors(i)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        pbar.set_postfix({'rgb': rgb_psnr, 'normal': normal_mae})

    #save ply
    gaussians.restore_from_params(params)
    gaussians.save_ply(REFINE_PATH)
