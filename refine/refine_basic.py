import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    dataset = Dataset(args.dataset_path, REFINE_UPSAMPLE_ITER)

    gaussians = GaussianModel()
    gaussians.restore_from_ply(args.ply_path, args.reset_attribute)

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    scene_dict = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'volprim_rf',
            'max_depth': 128
        },
        'shape': {
            'type': 'ellipsoidsmesh',
            'centers': gaussians_attributes['centers'],
            'scales': gaussians_attributes['scales'],
            'quaternions': gaussians_attributes['quats'],
            'opacities': gaussians_attributes['sigmats'],
            'sh_coeffs': gaussians_attributes['features']
        }
    })

    params = mi.traverse(scene_dict)

    params.keep(REFINE_PARAMS) 
    for _, param in params.items():
        dr.enable_grad(param)
    
    #clear opacity
    n_ellipsoids = params['shape.opacities'].shape[0]
    #params['shape.opacities'] = dr.full(mi.Float, 1.0, n_ellipsoids)

    #clear
    m_sh_coeffs = params['shape.sh_coeffs'].shape[0] // n_ellipsoids
    #params['shape.sh_coeffs'] = dr.full(mi.Float, 0.0, n_ellipsoids * m_sh_coeffs)

    opt = optimizers.BoundedAdam(lr=0.0001, params=params)
    opt.set_learning_rate({'shape.sh_coeffs':0.002})

    opt.set_bounds('shape.opacities', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('shape.sh_coeffs', lower=-1, upper=1)

    seed = 0

    psnr_list = []

    pbar = tqdm.tqdm(range(args.refine_niter))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        
        for idx, sensor in dataset.get_sensor_iterator():
            
            img = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=32, spp_grad=1,
                                  seed=seed, seed_grad=seed+1+len(dataset.sensors))
            
            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            
            view_loss = l1(ref_img, img) / dataset.batch_size

            total_loss = view_loss # + normal_loss #+ normal_priors_loss + normal_loss + normal_tv_loss + 0.1 * opacity_loss

            dr.backward(total_loss)

            loss += total_loss

            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)

            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
            mi.util.write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)        

            rgb_psnr += lpsnr(ref_img, img) / dataset.batch_size

        opt.step()
        params.update(opt)

        psnr_list.append(rgb_psnr)

        dataset.update_sensors(i)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        pbar.set_postfix({'rgb': rgb_psnr})

    plot_loss(psnr_list, label='PSNR', output_file=join(OUTPUT_REFINE_DIR, 'psnr.png'))

    #save ply
    gaussians.restore_from_params(params)
    gaussians.save_ply(args.refine_path)
