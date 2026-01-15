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
                'type': 'SGEmitter',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.lgtSGs*']
        
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
            for i in range(args.num_sgs):
                params[f'envmap.lgtSGslobe_{i}']   = opt[f'envmap.lgtSGslobe_{i}']
                params[f'envmap.lgtSGslambda_{i}'] = opt[f'envmap.lgtSGslambda_{i}']
                params[f'envmap.lgtSGsmu_{i}']     = opt[f'envmap.lgtSGsmu_{i}']
        else:
            params['envmap.data'] = opt['envmap.data']

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
    
    with time_measure("Creating KDTree"):
        KD_Tree = cKDTree(gaussians._xyz)
        _, kdtree_idx = KD_Tree.query(gaussians._xyz, k=32)
    
    gsstrategy = GSStrategyModel('configs/gs.yaml')
    
    with time_measure("Loading gaussian to ellipsoids factory"):
        ellipsoidsfactory = EllipsoidsFactory()
        gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)
    
    # Ensure output directories exist
    ensure_dir(OUTPUT_ENVMAP_DIR)
    ensure_dir(OUTPUT_OPT_DIR)
    ensure_dir(OUTPUT_EXTRA_DIR)
    ensure_dir(OUTPUT_PLY_DIR)

    # Save original envmap
    envmap = mi.Bitmap(args.envmap_path)
    envmap = np.array(envmap)
    mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'ref' + ('.exr')), envmap)

    return train_conf, dataset, gaussians, kdtree_idx, gsstrategy, gaussians_attributes

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

def compute_losses(img, aovs, ref_img, idx, sensor, dataset, opt, kdtree_idx, i):
    """Compute all losses for the current iteration"""
    # Extract aovs
    albedo_img = aovs['albedo'][:, :, :3]
    roughness_img = aovs['roughness'][:, :, :1]
    metallic_img = aovs['metallic'][:, :, :1]
    depth_img = aovs['depth'][:, :, :1]
    normal_img = aovs['normal'][:, :, :3]
    
    # Normal mask processing
    normal_norm = np.linalg.norm(normal_img, axis=2, keepdims=True)
    normal_mask = normal_norm > 0.1
    normal_mask_flat = np.reshape(normal_mask, (-1,1)).squeeze()

    # Get prior images
    albedo_priors_img = dataset.albedo_priors_images[idx][sensor.film().crop_size()[0]]
    roughness_priors_img = dataset.roughness_priors_images[idx][sensor.film().crop_size()[0]]
    normal_priors_img = dataset.normal_priors_images[idx][sensor.film().crop_size()[0]]

    # Compute view loss
    view_loss = l1(img, ref_img, convert_to_srgb=True) / dataset.batch_size

    # Compute priors losses
    albedo_priors_loss = l2(albedo_img, albedo_priors_img) / dataset.batch_size
    roughness_priors_loss = l2(roughness_img, roughness_priors_img) / dataset.batch_size
    normal_priors_loss = l2(normal_img, normal_priors_img) / dataset.batch_size
    priors_loss = albedo_priors_loss + roughness_priors_loss + normal_priors_loss

    # Compute TV losses
    rgb_tv_loss = TV(ref_img, img) / dataset.batch_size
    material_tv_loss = TV(ref_img, dr.concat([albedo_img, roughness_img, metallic_img], axis=2)) / dataset.batch_size
    tv_loss = rgb_tv_loss + material_tv_loss
    
    # Compute lambda loss
    lamb_loss = dr.mean(1.0-roughness_img) / dataset.batch_size

    # Compute normal loss from depth
    fake_normal_img = convert_depth_to_normal(depth_img, sensor)
    normal_loss = lnormal_sqr(normal_img, fake_normal_img, normal_mask_flat) / dataset.batch_size
    
    # Compute smooth losses
    albedo_laplacian_loss = ldiscrete_laplacian_reg_3dims(opt['albedos'], kdtree_idx) / dataset.batch_size
    roughness_laplacian_loss = ldiscrete_laplacian_reg_1dim(opt['roughnesses'], kdtree_idx) / dataset.batch_size
    laplacian_loss = albedo_laplacian_loss + roughness_laplacian_loss

    # ref_albedo_img = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
    # ref_roughness_img = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]
    # l1_albedo_loss = l1(ref_albedo_img, albedo_img) / dataset.batch_size
    # l1_roughness_loss = l1(ref_roughness_img, roughness_img) / dataset.batch_size

    # Compute total loss with warmup weights
    total_loss = mi.TensorXf([0.0])
    if i < train_conf.optimizer.warmup_iterations: # warm up
        total_loss += view_loss + 0.1 * normal_loss + 0.001 * lamb_loss + 0.01 * tv_loss + 1e-5 * laplacian_loss + priors_loss + 1e-4 * envmap_reg(opt, args.num_sgs)
    else:
        total_loss += view_loss + 0.1 * normal_loss + 0.001 * lamb_loss + 0.01 * tv_loss + 0.05 * priors_loss + 1e-4 * envmap_reg(opt, args.num_sgs)

    return total_loss, img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, normal_mask, ref_img, sensor

def save_render_results(i, idx, img, ref_img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, normal_mask, aovs, dataset):
    """Save rendering results for the current iteration"""
    rgb_bmp = resize_img(mi.Bitmap(img), dataset.target_res)
    rgb_ref_bmp = resize_img(mi.Bitmap(ref_img), dataset.target_res)
    albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
    roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
    metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)

    depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), dataset.target_res)
    normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img + 1.0)/2 , 0))), dataset.target_res)

    write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
    write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)
    write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_albedo' + ('.png')), albedo_bmp)
    write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_roughness' + ('.png')), roughness_bmp)
    write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_metallic' + ('.png')), metallic_bmp)
    write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_depth' + ('.png')), depth_bmp)
    write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_normal' + ('.png')), normal_bmp)
            
    # Save direct/indirect light if available
    if args.separate_direct_indirect and 'direct_light' in aovs and 'indirect_light' in aovs:
        direct_light_img = aovs['direct_light'][:, :, :3]
        indirect_light_img = aovs['indirect_light'][:, :, :3]

        direct_light_bmp = resize_img(mi.Bitmap(direct_light_img), dataset.target_res)
        indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img), dataset.target_res)
        
        write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_direct_light' + ('.png')), direct_light_bmp)
        write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_indirect_light' + ('.png')), indirect_light_bmp)

def compute_metrics(img, ref_img, aovs, dataset, sensor, idx, normal_mask):
    """Compute metrics (PSNR, MSE, MAE) for the current iteration"""
    rgb_psnr = lpsnr(ref_img, img, convert_to_srgb=True) / dataset.batch_size
    albedo_psnr = None
    roughness_mse = None
    normal_mae = None

    if args.dataset_type in ["TensoIR", "RTIR"]:
        albedo_img = aovs['albedo'][:, :, :3]
        roughness_img = aovs['roughness'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]
        
        ref_albedo = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
        ref_roughness = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]
        ref_normal = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]

        # _, three_channel_ratio = compute_rescale_ratio([ref_albedo], [albedo_img])
        # albedo_img = three_channel_ratio * albedo_img
        # albedo_img = mi.Bitmap(albedo_img)

        albedo_psnr = lpsnr(ref_albedo, albedo_img) / dataset.batch_size
        roughness_mse = l2(ref_roughness, roughness_img) / dataset.batch_size
        normal_mae = lmae(ref_normal, normal_img, normal_mask.squeeze()) / dataset.batch_size
    
    return rgb_psnr, albedo_psnr, roughness_mse, normal_mae

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
        
        if args.dataset_type in ["TensoIR", "RTIR"]:
            plot_loss(albedo_PSNR_list, label='Albedo PSNR', output_file=join(OUTPUT_DIR, 'albedo_psnr.png'))
            plot_loss(roughness_MSE_list, label='Roughness MSE', output_file=join(OUTPUT_DIR, 'roughness_mse.png'))
            plot_loss(normal_MAE_list, label='Normal MAE', output_file=join(OUTPUT_DIR, 'normal_mae.png'))

        gaussians.restore_from_params(params)
        save_path = f"{OUTPUT_PLY_DIR}/iter_{i:03d}.ply"
        gaussians.save_ply(save_path)
        print(f"[Iter {i}] Saved PLY to {save_path}")

def train_loop(train_conf, dataset, gaussians, kdtree_idx, gsstrategy, scene_dict, params, opt):
    """Main training loop"""
    seed = 0
    loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list = [], [], [], [], []

    pbar = tqdm.tqdm(range(train_conf.optimizer.iterations))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        albedo_psnr = mi.Float(0.0)
        roughness_mse = mi.Float(0.0)
        normal_mae = mi.Float(0.0)
        
        # Update learning rate schedule
        gsstrategy.lr_schedule(opt, i, train_conf.optimizer.iterations, train_conf.optimizer.scheduler.min_factor)

        for idx, sensor in dataset.get_sensor_iterator():
            # Render scene
            img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=args.training_spp * args.primal_spp_mult, spp_grad=args.training_spp,
                                  seed=seed, seed_grad=seed + 1 + len(dataset.sensors))
            
            dr.eval(img)
            dr.eval(aovs)

            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            
            # Compute all losses
            total_loss, img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, normal_mask, ref_img, sensor = compute_losses(
                img, aovs, ref_img, idx, sensor, dataset, opt, kdtree_idx, i
            )

            # Backward pass
            dr.backward(total_loss)

            loss += total_loss

            # Save rendering results
            save_render_results(i, idx, img, ref_img, albedo_img, roughness_img, metallic_img, depth_img, normal_img, normal_mask, aovs, dataset)

            # Compute metrics
            rgb_psnr_iter, albedo_psnr_iter, roughness_mse_iter, normal_mae_iter = compute_metrics(
                img, ref_img, aovs, dataset, sensor, idx, normal_mask
            )
            
            rgb_psnr += rgb_psnr_iter
            if args.dataset_type in ["TensoIR", "RTIR"]:
                albedo_psnr += albedo_psnr_iter
                roughness_mse += roughness_mse_iter
                normal_mae += normal_mae_iter
            
        # Update loss lists
        loss_list.append(np.asarray(total_loss))
        rgb_PSNR_list.append(np.asarray(rgb_psnr))

        if args.dataset_type in ["TensoIR", "RTIR"]:
            normal_MAE_list.append(np.asarray(normal_mae))
            albedo_PSNR_list.append(np.asarray(albedo_psnr))
            roughness_MSE_list.append(np.asarray(roughness_mse))

        # Update parameters
        opt.step()
        update_params(opt, params)
        dataset.update_sensors(i)

        # Update progress bar
        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        if args.dataset_type in ["TensoIR", "RTIR"]:
            pbar.set_postfix({'rgb': rgb_psnr, 'albedo': albedo_psnr, 'roughness': roughness_mse, 'normal': normal_mae})
        else:
            pbar.set_postfix({'rgb': rgb_psnr})

        # Save iteration results
        save_iteration_results(i, scene_dict, params, train_conf, loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list, gaussians)

if __name__ == "__main__":
    # Initialize all components
    train_conf, dataset, gaussians, kdtree_idx, gsstrategy, gaussians_attributes = initialize_components()
    
    # Setup scene and optimizer
    scene_dict, params, opt = setup_scene_and_optimizer(gaussians_attributes, train_conf)
    
    # Run training loop
    train_loop(train_conf, dataset, gaussians, kdtree_idx, gsstrategy, scene_dict, params, opt)