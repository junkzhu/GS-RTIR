import torch
import tqdm
import numpy as np
from os.path import join
from omegaconf import OmegaConf

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')


import optimizers
from constants import *

from utils import *
from models import *
from integrators import *
from emitter import *
from datasets import *
from losses import *

def load_scene_config():
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
            'selfocc_mode': args.selfocc_mode,
            'selfocc_offset_fixed': args.selfocc_offset_fixed,
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

    if args.envmap_optimization:
        if args.spherical_gaussian:
            # register SG envmap
            SGModel(
                num_sgs = args.num_sgs,
                #sg_init = np.load("output/final_optimized_sgs.npy")
            )
            
            scene_config['envmap'] = {
                'type': 'vMF',
                'filename': './emitter/init_1280_640.exr',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.lgtSGs*']
            OPTIMIZE_PARAMS += ['envmap.position'] + ['envmap.weight'] + ['envmap.std']
        
        else:
            scene_config['envmap'] = {
                'type': 'envmap',
                'filename': './emitter/init_1280_640.exr',
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

def register_optimizer(params, train_conf):
    opt = optimizers.BoundedAdam()

    ellipsoids = Ellipsoid.unravel(params['shape.data'])
    
    # register optimization parameters
    opt['centers'] = ellipsoids.center
    opt['scales']  = ellipsoids.scale
    opt['quats']   = mi.Vector4f(ellipsoids.quat)
    opt['opacities'] = params['shape.opacities']
    opt['normals'] = params['shape.normals']

    opt['albedos'] = params['shape.albedos']
    opt['roughnesses'] = params['shape.roughnesses']
    opt['metallics'] = params['shape.metallics']
    
    # set learning rate
    lr_dict = {
        'centers':     train_conf.optimizer.params.centers_lr,
        'scales':      train_conf.optimizer.params.scales_lr,
        'quats':       train_conf.optimizer.params.quats_lr,
        'opacities':   train_conf.optimizer.params.opacities_lr,
        'normals':     train_conf.optimizer.params.normals_lr,
        
        'albedos':     train_conf.optimizer.params.albedos_lr,
        'roughnesses': train_conf.optimizer.params.roughnesses_lr,
        'metallics': train_conf.optimizer.params.metallics_lr,
    }

    # register envmap parameters
    if args.envmap_optimization:
        if args.spherical_gaussian:
            opt['envmap.position'] = params['envmap.position']
            opt['envmap.weight'] = params['envmap.weight']
            opt['envmap.std'] = params['envmap.std']
            for i in range(args.num_sgs):
                opt[f'envmap.lgtSGslobe_{i}']   = params[f'envmap.lgtSGslobe_{i}']
                opt[f'envmap.lgtSGslambda_{i}'] = params[f'envmap.lgtSGslambda_{i}']
                opt[f'envmap.lgtSGsmu_{i}']     = params[f'envmap.lgtSGsmu_{i}']

            for i in range(args.num_sgs):
                lr_dict[f'envmap.lgtSGslobe_{i}']   = train_conf.optimizer.params.envmap.sg_lobe_lr
                lr_dict[f'envmap.lgtSGslambda_{i}'] = train_conf.optimizer.params.envmap.sg_lambda_lr
                lr_dict[f'envmap.lgtSGsmu_{i}']     = train_conf.optimizer.params.envmap.sg_mu_lr
        else:
            opt['envmap.data'] = params['envmap.data']
            lr_dict['envmap.data'] = train_conf.optimizer.params.envmap.data_lr

    opt.set_learning_rate(lr_dict)

    opt.set_bounds('scales',    lower=1e-6, upper=1e2)
    opt.set_bounds('opacities', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('normals', lower=-1, upper=1)

    opt.set_bounds('albedos', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('roughnesses', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('metallics', lower=1e-6, upper=1-1e-6)

    return opt

def update_params(opt, params):
    params['shape.data'] = Ellipsoid.ravel(opt['centers'], opt['scales'], mi.Quaternion4f(opt['quats']))
    params['shape.opacities'] = opt['opacities']
    params['shape.normals'] = opt['normals']
    
    params['shape.albedos'] = opt['albedos']
    params['shape.roughnesses'] = opt['roughnesses']
    params['shape.metallics'] = opt['metallics']
    
    if args.envmap_optimization:
        if args.spherical_gaussian:
            params['envmap.position'] = opt['envmap.position']
            params['envmap.weight'] = opt['envmap.weight']
            params['envmap.std'] = opt['envmap.std']
            for i in range(args.num_sgs):
                params[f'envmap.lgtSGslobe_{i}']   = opt[f'envmap.lgtSGslobe_{i}']
                params[f'envmap.lgtSGslambda_{i}'] = opt[f'envmap.lgtSGslambda_{i}']
                params[f'envmap.lgtSGsmu_{i}']     = opt[f'envmap.lgtSGsmu_{i}']
        else:
            #params['envmap.data'] = opt['envmap.data']
            params['envmap.data'] = gaussian_filter(opt['envmap.data'], sigma=1.0, radius=5)

    params.update()

def initialize_components():
    """Initialize all necessary components for training"""
    train_conf = OmegaConf.load('configs/train.yaml')    

    with time_measure("Loading dataset"):
        dataset = Dataset(args.dataset_path, train_iters=train_conf.optimizer.iterations if args.dash_reso_sche else None)

    with time_measure("Initializing gaussians"):
        gaussians = GaussianModel()
        if args.ply_path.endswith(".ply"):
            gaussians.restore_from_ply(args.ply_path, args.reset_attribute)
        elif args.ply_path.endswith(".pt"):
            gaussians.restore_from_ckpt(args.ply_path)
        else:
            raise ValueError(f"Unsupported file type: {args.ply_path}")
    
    gsstrategy = GSStrategyModel('configs/gs.yaml')
    
    with time_measure("Loading gaussian to ellipsoids factory"):
        ellipsoidsfactory = EllipsoidsFactory()
        gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)
    
    # Setup output directories
    setup_output_directories()

    # Save original envmap
    envmap = mi.Bitmap(args.envmap_path)
    envmap = np.array(envmap)
    mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'ref' + ('.exr')), envmap)

    return train_conf, dataset, gaussians, gsstrategy, gaussians_attributes

def setup_scene_and_optimizer(gaussians_attributes, train_conf):
    """Setup scene configuration and optimizer"""
    # Load scene config and create scene
    scene_config = load_scene_config()
    scene_dict = mi.load_dict(scene_config)

    # Setup parameters for optimization
    params = mi.traverse(scene_dict)
    params.keep(OPTIMIZE_PARAMS)
    for _, param in params.items():
        dr.enable_grad(param)

    # Register optimizer
    opt = register_optimizer(params, train_conf)
    update_params(opt, params)

    return scene_dict, params, opt

def compute_metrics(img, aovs, ref_imgs, normal_mask):
    """Compute metrics (PSNR, MSE, MAE) for the current iteration"""

    img, aovs, ref_imgs, normal_mask = map(dr.detach, [img, aovs, ref_imgs, normal_mask])
    
    rgb_psnr = lpsnr(ref_imgs['rgb'], img, convert_to_srgb=True)
    albedo_psnr = None
    roughness_mse = None
    normal_mae = None

    if args.dataset_type in ["TensoIR", "RT4Relight"]:
        albedo_img = aovs['albedo']
        roughness_img = aovs['roughness']
        normal_img = aovs['normal']
        
        ref_albedo = ref_imgs['albedo']
        ref_roughness = ref_imgs['roughness']
        ref_normal = ref_imgs['normal']

        _, three_channel_ratio = compute_rescale_ratio([ref_albedo], [albedo_img])
        albedo_img = three_channel_ratio * albedo_img
        albedo_img = mi.Bitmap(albedo_img)

        albedo_psnr = lpsnr(ref_albedo, albedo_img)
        roughness_mse = l2(ref_roughness, roughness_img)
        normal_mae = lmae(ref_normal, normal_img, normal_mask.squeeze())
    
    loss_metrics = {
        'rgb_psnr': rgb_psnr,
        'albedo_psnr': albedo_psnr,
        'roughness_mse': roughness_mse,
        'normal_mae': normal_mae
    }

    return loss_metrics

def save_iteration_results(i, scene_dict, params, train_conf, loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list, gaussians):
    """Save results at the end of each iteration"""
    # Save envmap if optimization is enabled
    if args.envmap_optimization:
        if args.spherical_gaussian:
            envmap_img = render_envmap_bitmap(params=params, num_sgs=args.num_sgs)
            mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'{i:04d}' + ('.exr')), envmap_img)
            if (i in SAVE_ENVMAP_ITER) or i == train_conf.optimizer.iterations - 1:
                save_sg_envmap(params, args.num_sgs, i)
        else:
            envmap_data = params['envmap.data']
            envmap_img = mi.Bitmap(envmap_data)
            mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'{i:04d}' + ('.exr')), envmap_img)

    # Save plots and PLY files at specified iterations
    if (i in dataset.render_upsample_iter) or i == train_conf.optimizer.iterations - 1:
        plot_loss(loss_list, label='Total Loss', output_file=join(OUTPUT_DIR, 'total_loss.png'))
        plot_loss(rgb_PSNR_list, label = "RGB PSNR", output_file=join(OUTPUT_DIR, 'rgb_psnr.png'))
        
        if args.dataset_type == "TensoIR":
            plot_loss(albedo_PSNR_list, label='Albedo PSNR', output_file=join(OUTPUT_DIR, 'albedo_psnr.png'))
            plot_loss(roughness_MSE_list, label='Roughness MSE', output_file=join(OUTPUT_DIR, 'roughness_mse.png'))
            plot_loss(normal_MAE_list, label='Normal MAE', output_file=join(OUTPUT_DIR, 'normal_mae.png'))

        gaussians.restore_from_params(params)
        save_path = f"{OUTPUT_PLY_DIR}/iter_{i:03d}.ply"
        gaussians.save_ply(save_path)
        print(f"[Iter {i}] Saved PLY to {save_path}")

def train_loop(train_conf, dataset, gaussians, gsstrategy, scene_dict, params, opt):
    """Main training loop"""
    seed = 0
    loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list = [], [], [], [], []

    pbar = tqdm.tqdm(range(train_conf.optimizer.iterations))
    for i in pbar:
        loss = mi.Float(0.0)
        
        # Update learning rate schedule
        gsstrategy.lr_schedule(opt, i, train_conf.optimizer.iterations, train_conf.optimizer.scheduler.min_factor)

        sensor, ref_imgs, priors_imgs = dataset.get_sensor_iterator()
            
        # Render scene
        buffer = mi.render(scene_dict, sensor=sensor, params=params, 
                                spp=args.training_spp * args.primal_spp_mult, spp_grad=args.training_spp,
                                seed=seed, seed_grad=seed + 1 + len(dataset.sensors))
            
        dr.eval(buffer)

        seed += 1 + dataset.batch_size

        # Unpack buffer
        img, aovs = unpack_buffer(buffer)
    
        # Compute all losses
        total_loss, normal_mask = compute_losses(
            img, aovs, ref_imgs, priors_imgs, sensor, i, train_conf
        )

        # Backward pass
        dr.backward(total_loss)

        loss += dr.detach(total_loss)

        # Save rendering results
        save_render_results(i, img, aovs, ref_imgs, normal_mask, dataset)

        # Compute metrics
        loss_metrics = compute_metrics(img, aovs, ref_imgs, normal_mask)

        # Update loss lists
        loss_list.append(np.asarray(dr.detach(total_loss)))
        rgb_PSNR_list.append(np.asarray(loss_metrics['rgb_psnr']))

        if args.dataset_type in ["TensoIR", "RT4Relight"]:
            normal_MAE_list.append(np.asarray(loss_metrics['normal_mae']))
            albedo_PSNR_list.append(np.asarray(loss_metrics['albedo_psnr']))
            roughness_MSE_list.append(np.asarray(loss_metrics['roughness_mse']))

        # Update parameters
        opt.step()
        update_params(opt, params)
        dataset.update_sensors(i)

        # Update progress bar
        loss_np = np.asarray(loss)
        pbar.set_description(f"Loss: {loss_np[0]:.4f}")

        if args.dataset_type in ["TensoIR", "RT4Relight"]:
            pbar.set_postfix({'rgb': loss_metrics['rgb_psnr'], 'albedo': loss_metrics['albedo_psnr'], 'roughness': loss_metrics['roughness_mse'], 'normal': loss_metrics['normal_mae']})
        else:
            pbar.set_postfix({'rgb': loss_metrics['rgb_psnr']})

        # Save iteration results
        save_iteration_results(i, scene_dict, params, train_conf, loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list, gaussians)

if __name__ == "__main__":
    # Initialize all components
    train_conf, dataset, gaussians, gsstrategy, gaussians_attributes = initialize_components()
    
    # Setup scene and optimizer
    scene_dict, params, opt = setup_scene_and_optimizer(gaussians_attributes, train_conf)
    
    # Run training loop
    train_loop(train_conf, dataset, gaussians, gsstrategy, scene_dict, params, opt)