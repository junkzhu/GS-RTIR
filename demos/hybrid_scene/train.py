import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *
from utils import *
from models import *
from integrators import *
from losses import *
import optimizers
from datasets.dataset_readers import load_mipmaps

from common_config import *

# Output directory configuration
OUTPUT_DIR = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'hybrid_scene', 'opt')
ensure_dir(OUTPUT_DIR)

def set_optimize_params(params):
    """Reset optimize params"""
    
    optimize_params = []

    optimize_params += ['greenwall*'] + ['redwall*']

    optimize_params += ['white.base_color.value'] + ['white.roughness.value'] + ['white.metallic.value']
    optimize_params += ['green.base_color.value'] + ['green.roughness.value'] + ['green.metallic.value']
    optimize_params += ['red.base_color.value'] + ['red.roughness.value'] + ['red.metallic.value']
    optimize_params += ['gold.base_color.value'] + ['gold.roughness.value'] + ['gold.metallic.value']

    optimize_params += ['gaussian1.albedos'] + ['gaussian1.roughnesses'] + ['gaussian1.metallics']
    optimize_params += ['gaussian2.albedos'] + ['gaussian2.roughnesses'] + ['gaussian2.metallics']
    optimize_params += ['gaussian3.albedos'] + ['gaussian3.roughnesses'] + ['gaussian3.metallics']

    #optimize_params += ['light.emitter.radiance.value']

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

    params['gaussian2.albedos'] = dr.ones_like(params['gaussian2.albedos']) * 0.5 
    params['gaussian2.roughnesses'] = dr.ones_like(params['gaussian2.roughnesses'])
    params['gaussian2.metallics'] = dr.zeros_like(params['gaussian2.metallics'])

    params['gaussian3.albedos'] = dr.ones_like(params['gaussian3.albedos']) * 0.5 
    params['gaussian3.roughnesses'] = dr.ones_like(params['gaussian3.roughnesses'])
    params['gaussian3.metallics'] = dr.zeros_like(params['gaussian3.metallics'])

    #params['light.emitter.radiance.value'] = mi.Color3f(0.5)

    params.keep(optimize_params)

def register_optimizer(params):
    opt = optimizers.BoundedAdam()

    opt['gaussian1.albedos'] = params['gaussian1.albedos']
    opt['gaussian1.roughnesses'] = params['gaussian1.roughnesses']
    opt['gaussian1.metallics'] = params['gaussian1.metallics']
    
    opt['gaussian2.albedos'] = params['gaussian2.albedos']
    opt['gaussian2.roughnesses'] = params['gaussian2.roughnesses']
    opt['gaussian2.metallics'] = params['gaussian2.metallics']
    
    opt['gaussian3.albedos'] = params['gaussian3.albedos']
    opt['gaussian3.roughnesses'] = params['gaussian3.roughnesses']
    opt['gaussian3.metallics'] = params['gaussian3.metallics']

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

    opt.set_bounds('gaussian2.albedos', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gaussian2.roughnesses', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gaussian2.metallics', lower=0, upper=1-1e-6)

    opt.set_bounds('gaussian3.albedos', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gaussian3.roughnesses', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('gaussian3.metallics', lower=0, upper=1-1e-6)

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

    params['gaussian2.albedos'] = opt['gaussian2.albedos']
    params['gaussian2.roughnesses'] = opt['gaussian2.roughnesses']
    params['gaussian2.metallics'] = opt['gaussian2.metallics']

    params['gaussian3.albedos'] = opt['gaussian3.albedos']
    params['gaussian3.roughnesses'] = opt['gaussian3.roughnesses']
    params['gaussian3.metallics'] = opt['gaussian3.metallics']
    
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

def compute_losses(img, aovs, ref_img):
    """Compute all losses for the current iteration"""
    # Extract aovs
    albedo_img = aovs['albedo'][:, :, :3]
    roughness_img = aovs['roughness'][:, :, :1]
    metallic_img = aovs['metallic'][:, :, :1]

    # Compute view loss
    view_loss = l1(img, ref_img, convert_to_srgb=True)

    # Compute TV losses
    rgb_tv_loss = TV(ref_img, img)
    material_tv_loss = TV(ref_img, dr.concat([albedo_img, roughness_img, metallic_img], axis=2))
    tv_loss = rgb_tv_loss + material_tv_loss
    
    # Compute lambda loss
    lamb_loss = dr.mean(1.0-roughness_img)
    
    # Compute total loss with warmup weights
    total_loss = mi.TensorXf([0.0])
    total_loss += view_loss + 0.001 * lamb_loss + 0.01 * tv_loss

    return total_loss

def save_render_results(i, img, ref_img, aovs):
    rgb_bmp = resize_img(mi.Bitmap(img), [800, 800])
    rgb_ref_bmp = resize_img(mi.Bitmap(ref_img), [800, 800])
    albedo_bmp = resize_img(mi.Bitmap(aovs['albedo'][:, :, :3]), [800, 800])
    roughness_bmp = resize_img(mi.Bitmap(aovs['roughness'][:, :, :1]), [800, 800])
    metallic_bmp = resize_img(mi.Bitmap(aovs['metallic'][:, :, :1]), [800, 800])

    write_bitmap(join(OUTPUT_DIR, f'opt-{i:04d}' + ('.png')), rgb_bmp)
    write_bitmap(join(OUTPUT_DIR, f'opt-{i:04d}_ref' + ('.png')), rgb_ref_bmp)
    write_bitmap(join(OUTPUT_DIR, f'opt-{i:04d}_albedo' + ('.png')), albedo_bmp)
    write_bitmap(join(OUTPUT_DIR, f'opt-{i:04d}_roughness' + ('.png')), roughness_bmp)
    write_bitmap(join(OUTPUT_DIR, f'opt-{i:04d}_metallic' + ('.png')), metallic_bmp)

def compute_metrics(img, ref_img, ref_albedo_img, ref_roughness_img, ref_metallic_img, aovs):
    """Compute metrics for the current iteration"""
    # Extract aovs
    albedo_img = aovs['albedo'][:, :, :3]
    roughness_img = aovs['roughness'][:, :, :1]
    metallic_img = aovs['metallic'][:, :, :1]

    rgb_psnr = lpsnr(ref_img, img, convert_to_srgb=True)
    albedo_psnr = lpsnr(ref_albedo_img, albedo_img) 
    roughness_mse = l2(ref_roughness_img, roughness_img)
    metallic_mse = l2(ref_metallic_img, metallic_img)

    return rgb_psnr, albedo_psnr, roughness_mse, metallic_mse

def update_sensors(sensor, i, res):
    if i == 199:
        set_sensor_res(sensor, [400, 400])
        res = 400
    if i == 399:
        set_sensor_res(sensor, [800, 800])
        res = 800
    return res

def train_loop(sensor, scene_dict, params, opt, ref_images, training_spp):
    seed = 0
    loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, metallic_MSE_list = [], [], [], [], []

    res = 200
    set_sensor_res(sensor, [res, res])

    pbar = tqdm.tqdm(range(800))
    for i in pbar:
        loss, rgb_psnr, albedo_psnr, roughness_mse, metallic_mse = mi.Float(0.0), mi.Float(0.0), mi.Float(0.0), mi.Float(0.0), mi.Float(0.0)
        img, aovs = mi.render(scene_dict, sensor=sensor, params=params, spp=training_spp * 4, spp_grad=training_spp, seed=seed, seed_grad=seed + 1)

        dr.eval(img)
        dr.eval(aovs)

        seed += 1 + i

        ref_img, ref_albedo_img, ref_roughness_img, ref_metallic_img = ref_images['rgb'][res], ref_images['albedo'][res], ref_images['roughness'][res], ref_images['metallic'][res]
        total_loss= compute_losses(img, aovs, ref_img)

        dr.backward(total_loss)

        loss += total_loss
        
        # Save render results
        if i % 5 == 0:
            save_render_results(i, img, ref_img, aovs)

        # Compute metrics
        rgb_psnr_iter, albedo_psnr_iter, roughness_mse_iter, metallic_mse_iter = compute_metrics(
            img, ref_img, ref_albedo_img, ref_roughness_img, ref_metallic_img, aovs
        )

        rgb_psnr += rgb_psnr_iter
        albedo_psnr += albedo_psnr_iter
        roughness_mse += roughness_mse_iter
        metallic_mse += metallic_mse_iter
        
        loss_list.append(np.asarray(total_loss))
        rgb_PSNR_list.append(np.asarray(rgb_psnr))
        albedo_PSNR_list.append(np.asarray(albedo_psnr))
        roughness_MSE_list.append(np.asarray(roughness_mse))
        metallic_MSE_list.append(np.asarray(metallic_mse))

        opt.step()
        update_params(opt, params)
        res = update_sensors(sensor, i, res)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)
        pbar.set_postfix({'rgb': rgb_psnr, 'albedo': albedo_psnr, 'roughness': roughness_mse, 'metallic': metallic_mse})


if __name__ == "__main__":
    # Configuration parameters
    render_spp = 32
    training_spp = 16
    envmap_path = './demos/scenes/envmaps/pretoria_gardens_4k.exr'
        
    # Load Images
    ref_image = load_mipmaps('./demos/outputs/hybrid_scene/GT/rgb.png', bsrgb2linear = True)
    ref_albedo = load_mipmaps('./demos/outputs/hybrid_scene/GT/albedo.png', bsrgb2linear = False)
    ref_roughness = load_mipmaps('./demos/outputs/hybrid_scene/GT/roughness.png', bsrgb2linear = False)
    ref_metallic = load_mipmaps('./demos/outputs/hybrid_scene/GT/metallic.png', bsrgb2linear = False)

    ref_images = {
        'rgb': ref_image,
        'albedo': ref_albedo,
        'roughness': ref_roughness,
        'metallic': ref_metallic,
    }

    # Camera parameters
    sensor = make_sensor(
        cam_pos=(0.0, 0.0, 4.2),
        look_at=(0.0, 0.0, 0.0),
        up=(0, 1, 0),
        fov=40,
        resx=800,
        resy=800
    )

    # Setup scene
    scene_dict = set_scene_config(envmap_path)

    # set optimize params
    params = mi.traverse(scene_dict)
    set_optimize_params(params)

    for _, param in params.items():
        dr.enable_grad(param)

    # Setup optimizer
    opt = register_optimizer(params)
    update_params(opt, params)

    # Training loop
    train_loop(sensor, scene_dict, params, opt, ref_images, training_spp)

    
