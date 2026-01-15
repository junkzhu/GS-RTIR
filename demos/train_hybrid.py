import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tqdm
import numpy as np
from os.path import join
from omegaconf import OmegaConf

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from scipy.spatial import cKDTree

import optimizers
from constants import *

from utils import *
from models import *
from integrators import *
from emitter import *
from datasets import *
from losses import *

from hybrid_render import set_hybrid_scene, sensor_for_hybrid
from datasets.dataset_readers import load_mipmaps

def update_sensors(sensor, i, res):
    if i == 199:
        set_sensor_res(sensor, [400, 400])
        res = 400
    if i == 399:
        set_sensor_res(sensor, [800, 800])
        res = 800
    return res

def set_and_reset_optimize_params(params):
    optimize_params = []

    optimize_params += ['greenwall*'] + ['redwall*']

    # Mesh材质参数
    optimize_params += ['white.base_color.value'] + ['white.roughness.value'] + ['white.metallic.value']
    optimize_params += ['green.base_color.value'] + ['green.roughness.value'] + ['green.metallic.value']
    optimize_params += ['red.base_color.value'] + ['red.roughness.value'] + ['red.metallic.value']
    optimize_params += ['gold.base_color.value'] + ['gold.roughness.value'] + ['gold.metallic.value']

    # GS材质参数
    optimize_params += ['gaussian1.albedos'] + ['gaussian1.roughnesses'] + ['gaussian1.metallics']

    # 光源参数
    #optimize_params += ['light.emitter.radiance.value']

    # 初始化材质参数
    params['white.base_color.value'] = mi.Color3f(0.5)  
    params['green.base_color.value'] = mi.Color3f(0.5)  
    params['red.base_color.value'] = mi.Color3f(0.5)  
    params['gold.base_color.value'] = mi.Color3f(0.5)  
    
    params['white.roughness.value'] = mi.Float(1.0)  
    params['green.roughness.value'] = mi.Float(1.0)  
    params['red.roughness.value'] = mi.Float(1.0)  
    params['gold.roughness.value'] = mi.Float(1.0)  

    params['white.metallic.value'] = mi.Float(0.0)  
    params['green.metallic.value'] = mi.Float(0.0)  
    params['red.metallic.value'] = mi.Float(0.0)  
    params['gold.metallic.value'] = mi.Float(0.0)  

    params['gaussian1.albedos'] = dr.ones_like(params['gaussian1.albedos']) * 0.5 
    params['gaussian1.roughnesses'] = dr.ones_like(params['gaussian1.roughnesses'])
    params['gaussian1.metallics'] = dr.zeros_like(params['gaussian1.metallics'])

    #params['light.emitter.radiance.value'] = mi.Color3f(0.5)

    return optimize_params


def register_optimizer(params):
    opt = optimizers.BoundedAdam()

    opt['gaussian1.albedos'] = params['gaussian1.albedos']
    opt['gaussian1.roughnesses'] = params['gaussian1.roughnesses']
    opt['gaussian1.metallics'] = params['gaussian1.metallics']
    
    opt['white.base_color.value'] = params['white.base_color.value']
    opt['green.base_color.value'] = params['green.base_color.value']
    opt['red.base_color.value'] = params['red.base_color.value']
    opt['gold.base_color.value'] = params['gold.base_color.value']

    opt['white.roughness.value'] = params['white.roughness.value']
    opt['green.roughness.value'] = params['green.roughness.value']
    opt['red.roughness.value'] = params['red.roughness.value']
    opt['gold.roughness.value'] = params['gold.roughness.value']

    opt['white.metallic.value'] = params['white.metallic.value']
    opt['green.metallic.value'] = params['green.metallic.value']
    opt['red.metallic.value'] = params['red.metallic.value']
    opt['gold.metallic.value'] = params['gold.metallic.value']

    #opt['light.emitter.radiance.value'] = params['light.emitter.radiance.value']
    
    opt.set_learning_rate(0.02)
    
    opt.set_bounds('gaussian1.albedos', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gaussian1.roughnesses', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gaussian1.metallics', lower=0, upper=1-1e-6)

    opt.set_bounds('white.base_color.value', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('green.base_color.value', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('red.base_color.value', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gold.base_color.value', lower=1e-6, upper=1-1e-6)

    opt.set_bounds('white.roughness.value', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('green.roughness.value', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('red.roughness.value', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gold.roughness.value', lower=1e-6, upper=1-1e-6)

    opt.set_bounds('white.metallic.value', lower=0, upper=1-1e-6)
    opt.set_bounds('green.metallic.value', lower=0, upper=1-1e-6)
    opt.set_bounds('red.metallic.value', lower=0, upper=1-1e-6)
    opt.set_bounds('gold.metallic.value', lower=0, upper=1-1e-6)

    #opt.set_bounds('light.emitter.radiance.value', lower=0)

    return opt

def update_params(opt, params):
    params['gaussian1.albedos'] = opt['gaussian1.albedos']
    params['gaussian1.roughnesses'] = opt['gaussian1.roughnesses']
    params['gaussian1.metallics'] = opt['gaussian1.metallics']
    
    params['white.base_color.value'] = opt['white.base_color.value']
    params['green.base_color.value'] = opt['green.base_color.value']
    params['red.base_color.value'] = opt['red.base_color.value']
    params['gold.base_color.value'] = opt['gold.base_color.value']

    params['white.roughness.value'] = opt['white.roughness.value']
    params['green.roughness.value'] = opt['green.roughness.value']
    params['red.roughness.value'] = opt['red.roughness.value']
    params['gold.roughness.value'] = opt['gold.roughness.value']

    params['white.metallic.value'] = opt['white.metallic.value']
    params['green.metallic.value'] = opt['green.metallic.value']
    params['red.metallic.value'] = opt['red.metallic.value']
    params['gold.metallic.value'] = opt['gold.metallic.value']

    #params['light.emitter.radiance.value'] = opt['light.emitter.radiance.value']

    params.update()

def setup_scene_and_optimizer(scene_config):
    """Setup scene configuration and optimizer"""

    # Setup parameters for optimization
    scene_dict = mi.load_dict(scene_config)
    params = mi.traverse(scene_dict)

    optimize_params = set_and_reset_optimize_params(params)
    params.keep(optimize_params)

    for _, param in params.items():
        dr.enable_grad(param)

    # Register optimizer
    opt = register_optimizer(params)
    update_params(opt, params)

    return scene_dict, params, opt

def compute_losses(img, aovs, ref_img, sensor, opt, i, prior_albedo, prior_roughness, prior_metallic):
    """Compute all losses for the current iteration"""
    # Extract aovs
    albedo_img = aovs['albedo'][:, :, :3]
    roughness_img = aovs['roughness'][:, :, :1]
    metallic_img = aovs['metallic'][:, :, :1]
    depth_img = aovs['depth'][:, :, :1]
    normal_img = aovs['normal'][:, :, :3]
    
    # Get prior images
    albedo_priors_img = prior_albedo
    roughness_priors_img = prior_roughness
    metallic_priors_img = prior_metallic

    # Compute view loss
    view_loss = l1(img, ref_img, convert_to_srgb=True)

    # Compute priors losses
    albedo_priors_loss = l2(albedo_img, albedo_priors_img)
    roughness_priors_loss = l2(roughness_img, roughness_priors_img)
    metallic_priors_loss = l2(metallic_img, metallic_priors_img)
    priors_loss = albedo_priors_loss + roughness_priors_loss + metallic_priors_loss 

    # Compute TV losses
    rgb_tv_loss = TV(ref_img, img)
    material_tv_loss = TV(ref_img, dr.concat([albedo_img, roughness_img, metallic_img], axis=2))
    tv_loss = rgb_tv_loss + material_tv_loss
    
    # Compute lambda loss
    lamb_loss = dr.mean(1.0-roughness_img)
    
    # Compute total loss with warmup weights
    total_loss = mi.TensorXf([0.0])
    if i < 200:
        total_loss += view_loss + 0.001 * lamb_loss + 0.01 * tv_loss + priors_loss
    else:
        total_loss += view_loss + 0.001 * lamb_loss + 0.01 * tv_loss + 0.05 * priors_loss

    return total_loss, img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, ref_img, sensor

def save_render_results(i, img, ref_img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, aovs):
    """Save rendering results for the current iteration"""
    rgb_bmp = resize_img(mi.Bitmap(img), [800, 800])
    rgb_ref_bmp = resize_img(mi.Bitmap(ref_img), [800, 800])
    albedo_bmp = resize_img(mi.Bitmap(albedo_img), [800, 800])
    roughness_bmp = resize_img(mi.Bitmap(roughness_img), [800, 800])
    metallic_bmp = resize_img(mi.Bitmap(metallic_img), [800, 800])

    depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), [800, 800])
    
    write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}' + ('.png')), rgb_bmp)
    write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_ref' + ('.png')), rgb_ref_bmp)
    write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_albedo' + ('.png')), albedo_bmp)
    write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_roughness' + ('.png')), roughness_bmp)
    write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_metallic' + ('.png')), metallic_bmp)
    write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_depth' + ('.png')), depth_bmp)

    # Save direct/indirect light if available
    if args.separate_direct_indirect and 'direct_light' in aovs and 'indirect_light' in aovs:
        direct_light_img = aovs['direct_light'][:, :, :3]
        indirect_light_img = aovs['indirect_light'][:, :, :3]

        direct_light_bmp = resize_img(mi.Bitmap(direct_light_img), [800, 800])
        indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img), [800, 800])
        
        write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_direct_light' + ('.png')), direct_light_bmp)
        write_bitmap(join(OUTPUT_HYBRID_OPT_DIR, f'opt-{i:04d}_indirect_light' + ('.png')), indirect_light_bmp)

def compute_metrics(img, ref_img, ref_albedo, ref_roughness, aovs, sensor):
    """Compute metrics (PSNR, MSE, MAE) for the current iteration"""
    rgb_psnr = lpsnr(ref_img, img, convert_to_srgb=True)
    albedo_psnr = None
    roughness_mse = None
    normal_mae = None

    if args.dataset_type in ["TensoIR", "RTIR"]:
        albedo_img = aovs['albedo'][:, :, :3]
        roughness_img = aovs['roughness'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]
    
        albedo_psnr = lpsnr(ref_albedo, albedo_img) 
        roughness_mse = l2(ref_roughness, roughness_img) 
        #normal_mae = lmae(ref_normal, normal_img, normal_mask.squeeze()) / dataset.batch_size
    
    return rgb_psnr, albedo_psnr, roughness_mse, normal_mae

def save_iteration_results(i, params, loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, gaussians):
    if i in [200, 400, 600, 800]:
        plot_loss(loss_list, label='Total Loss', output_file=join(OUTPUT_HYBRID_OPT_DIR, 'total_loss.png'))
        plot_loss(rgb_PSNR_list, label = "RGB PSNR", output_file=join(OUTPUT_HYBRID_OPT_DIR, 'rgb_psnr.png'))
        
        if args.dataset_type in ["TensoIR", "RTIR"]:
            plot_loss(albedo_PSNR_list, label='Albedo PSNR', output_file=join(OUTPUT_HYBRID_OPT_DIR, 'albedo_psnr.png'))
            plot_loss(roughness_MSE_list, label='Roughness MSE', output_file=join(OUTPUT_HYBRID_OPT_DIR, 'roughness_mse.png'))

        gaussians.restore_from_params(params)
        save_path = f"{OUTPUT_HYBRID_OPT_DIR}/iter_{i:03d}.ply"
        gaussians.save_ply(save_path)
        print(f"[Iter {i}] Saved PLY to {save_path}")

def train_loop(sensor, gaussians, scene_dict, params, opt, ref_image, ref_albedo, ref_roughness, prior_albedo, prior_roughness, prior_metallic):
    """Main training loop"""
    seed = 0
    loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list = [], [], [], [], []

    res = 200
    set_sensor_res(sensor, [res, res])

    pbar = tqdm.tqdm(range(800))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        albedo_psnr = mi.Float(0.0)
        roughness_mse = mi.Float(0.0)
        normal_mae = mi.Float(0.0)
        
        # Render scene
        img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                spp=args.training_spp * args.primal_spp_mult, spp_grad=args.training_spp,
                                seed=seed, seed_grad=seed + 1 + i)
        
        dr.eval(img)
        dr.eval(aovs)

        seed += 1 + i

        ref_img = ref_image[res]
        ref_albedo_img = ref_albedo[res]
        ref_roughness_img = ref_roughness[res]
        
        prior_albedo_img = prior_albedo[res]
        prior_roughness_img = prior_roughness[res]
        prior_metallic_img = prior_metallic[res]
         
        # Compute all losses
        total_loss, img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, ref_img, sensor = compute_losses(
            img, aovs, ref_img, sensor, opt, i, prior_albedo_img, prior_roughness_img, prior_metallic_img
        )

        dr.backward(total_loss)

        loss += total_loss

        # Save rendering results
        save_render_results(i, img, ref_img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, aovs)

        # Compute metrics
        rgb_psnr_iter, albedo_psnr_iter, roughness_mse_iter, normal_mae_iter = compute_metrics(
            img, ref_img, ref_albedo_img, ref_roughness_img, aovs, sensor
        )
            
        rgb_psnr += rgb_psnr_iter
        albedo_psnr += albedo_psnr_iter
        roughness_mse += roughness_mse_iter
        #     normal_mae += normal_mae_iter
            
        # Update loss lists
        loss_list.append(np.asarray(total_loss))
        rgb_PSNR_list.append(np.asarray(rgb_psnr))
        albedo_PSNR_list.append(np.asarray(albedo_psnr))
        roughness_MSE_list.append(np.asarray(roughness_mse))

        # Update parameters
        opt.step()
        update_params(opt, params)
        res = update_sensors(sensor, i, res)

        # Update progress bar
        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        if args.dataset_type in ["TensoIR", "RTIR"]:
            pbar.set_postfix({'rgb': rgb_psnr, 'albedo': albedo_psnr, 'roughness': roughness_mse})
        else:
            pbar.set_postfix({'rgb': rgb_psnr})

        # Save iteration results
        #save_iteration_results(i, params, loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, gaussians)

if __name__ == "__main__":
    ensure_dir(OUTPUT_HYBRID_OPT_DIR)

    ref_image = load_mipmaps('/home/zjk/code/GS-RTIR/outputs/demos#0/hybrid_results/rgb.png', bsrgb2linear = True)
    ref_albedo = load_mipmaps('/home/zjk/code/GS-RTIR/outputs/demos#0/hybrid_results/albedo.png', bsrgb2linear = False)
    ref_roughness = load_mipmaps('/home/zjk/code/GS-RTIR/outputs/demos#0/hybrid_results/roughness.png', bsrgb2linear = False)

    prior_albedo = load_mipmaps('/home/zjk/code/GS-RTIR/outputs/demos#0/hybrid_results/albedo_sunset.png', bsrgb2linear = True)
    prior_roughness = load_mipmaps('/home/zjk/code/GS-RTIR/outputs/demos#0/hybrid_results/roughness_sunset.png', bsrgb2linear = False)
    prior_metallic = load_mipmaps('/home/zjk/code/GS-RTIR/outputs/demos#0/hybrid_results/metallic_sunset.png', bsrgb2linear = False)

    #config the scene
    scene_config, gaussians = set_hybrid_scene('/home/zjk/datasets/bloem_train_track_cloudy_4k.exr')
    
    # Set integrator
    scene_config['integrator'] = {
        'type': 'gsprim_prb',
        'max_depth': 3,
        'pt_rate': 1.0,
        'gaussian_max_depth': 128,
        'hide_emitters': False,
        'use_mis': True,
        'selfocc_offset_max': 0.1,
        'geometry_threshold': 0.2,
        'separate_direct_indirect': True,
        'optimize_mesh': True,
    }

    sensor = sensor_for_hybrid()

    # Setup scene and optimizer
    scene_dict, params, opt = setup_scene_and_optimizer(scene_config)

    # Run training loop
    train_loop(sensor, gaussians, scene_dict, params, opt, ref_image, ref_albedo, ref_roughness, prior_albedo, prior_roughness, prior_metallic)