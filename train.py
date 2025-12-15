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
            'type': INTEGRATOR,
            'max_depth': MAX_BOUNCE_NUM,
            'pt_rate': SPP_PT_RATE,
            'gaussian_max_depth': 128,
            'hide_emitters': HIDE_EMITTER,
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
        if SPHERICAL_GAUSSIAN:
            # register SG envmap
            SGModel(
                num_sgs = NUM_SGS,
                #sg_init = np.load("output/final_optimized_sgs.npy")
            )
            
            scene_config['envmap'] = {
                'type': 'vMF',
                'filename': '/home/zjk/datasets/TensoIR/Environment_Maps/high_res_envmaps_1k/sunset.hdr',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.lgtSGs*']
            OPTIMIZE_PARAMS += ['envmap.position'] + ['envmap.weight'] + ['envmap.std']
        
        else:
            scene_config['envmap'] = {
                'type': 'envmap',
                'filename': 'D:/dataset/Environment_Maps/high_res_envmaps_1k/init_1280_640.exr',
                'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 90) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], 90)
            }
            OPTIMIZE_PARAMS += ['envmap.data']

    else:
        scene_config['emitter'] = {
            'type': 'envmap',
            'id': 'EnvironmentMapEmitter',
            'filename': ENVMAP_PATH,
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

    # set learning rate
    lr_dict = {
        'centers':     train_conf.optimizer.params.centers_lr,
        'scales':      train_conf.optimizer.params.scales_lr,
        'quats':       train_conf.optimizer.params.quats_lr,
        'opacities':   train_conf.optimizer.params.opacities_lr,
        'normals':     train_conf.optimizer.params.normals_lr,
        
        'albedos':     train_conf.optimizer.params.albedos_lr,
        'roughnesses': train_conf.optimizer.params.roughnesses_lr,
    }

    # register envmap parameters
    if OPTIMIZE_ENVMAP:
        opt['envmap.position'] = params['envmap.position']
        opt['envmap.weight'] = params['envmap.weight']
        opt['envmap.std'] = params['envmap.std']
        for i in range(NUM_SGS):
            opt[f'envmap.lgtSGslobe_{i}']   = params[f'envmap.lgtSGslobe_{i}']
            opt[f'envmap.lgtSGslambda_{i}'] = params[f'envmap.lgtSGslambda_{i}']
            opt[f'envmap.lgtSGsmu_{i}']     = params[f'envmap.lgtSGsmu_{i}']

        for i in range(NUM_SGS):
            lr_dict[f'envmap.lgtSGslobe_{i}']   = train_conf.optimizer.params.envmap.sg_lobe_lr
            lr_dict[f'envmap.lgtSGslambda_{i}'] = train_conf.optimizer.params.envmap.sg_lambda_lr
            lr_dict[f'envmap.lgtSGsmu_{i}']     = train_conf.optimizer.params.envmap.sg_mu_lr

    opt.set_learning_rate(lr_dict)

    opt.set_bounds('scales',    lower=1e-6, upper=1e2)
    opt.set_bounds('opacities', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('normals', lower=-1, upper=1)

    opt.set_bounds('albedos', lower=1e-6, upper=1-1e-6)
    opt.set_bounds('roughnesses', lower=1e-6, upper=1-1e-6)

    return opt

def update_params(opt, params):
    params['shape.data'] = Ellipsoid.ravel(opt['centers'], opt['scales'], mi.Quaternion4f(opt['quats']))
    params['shape.opacities'] = opt['opacities']
    params['shape.normals'] = opt['normals']
    
    params['shape.albedos'] = opt['albedos']
    params['shape.roughnesses'] = opt['roughnesses']
    
    if OPTIMIZE_ENVMAP:
        params['envmap.position'] = opt['envmap.position']
        params['envmap.weight'] = opt['envmap.weight']
        params['envmap.std'] = opt['envmap.std']
        for i in range(NUM_SGS):
            params[f'envmap.lgtSGslobe_{i}']   = opt[f'envmap.lgtSGslobe_{i}']
            params[f'envmap.lgtSGslambda_{i}'] = opt[f'envmap.lgtSGslambda_{i}']
            params[f'envmap.lgtSGsmu_{i}']     = opt[f'envmap.lgtSGsmu_{i}']

    params.update()

if __name__ == "__main__":
    train_conf = OmegaConf.load('configs/train.yaml')
    dataset = Dataset(DATASET_PATH)

    gaussians = GaussianModel()
    if PLY_PATH.endswith(".ply"):
        gaussians.restore_from_ply(PLY_PATH, RESET_ATTRIBUTE)
    elif PLY_PATH.endswith(".pt"):
        gaussians.restore_from_ckpt(PLY_PATH)
    else:
        raise ValueError(f"Unsupported file type: {PLY_PATH}")

    # cKDTree: find the nearest k gaussians
    KD_Tree = cKDTree(gaussians._xyz)
    _, kdtree_idx = KD_Tree.query(gaussians._xyz, k=32)

    gsstrategy = GSStrategyModel('configs/gs.yaml')

    ellipsoidsfactory = EllipsoidsFactory()
    gaussians_attributes = ellipsoidsfactory.load_gaussian(gaussians=gaussians)

    # original envmap
    envmap = mi.Bitmap(ENVMAP_PATH)
    envmap = np.array(envmap)
    mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'ref' + ('.png')), envmap)


    #---------------------------------- config ----------------------------------
    scene_config = load_scene_config()

    scene_dict = mi.load_dict(scene_config)

    params = mi.traverse(scene_dict)
    params.keep(OPTIMIZE_PARAMS)
    for _, param in params.items():
        dr.enable_grad(param)

    opt = register_optimizer(params, train_conf)
    update_params(opt, params)

    seed = 0

    loss_list, rgb_PSNR_list, albedo_PSNR_list, roughness_MSE_list, normal_MAE_list = [], [], [], [], []

    pbar = tqdm.tqdm(range(train_conf.optimizer.iterations))
    for i in pbar:
        loss = mi.Float(0.0)
        rgb_psnr = mi.Float(0.0)
        albedo_psnr = mi.Float(0.0)
        roughness_mse = mi.Float(0.0)
        normal_mae = mi.Float(0.0)
        
        gsstrategy.lr_schedule(opt, i, train_conf.optimizer.iterations, train_conf.optimizer.scheduler.min_factor)

        for idx, sensor in dataset.get_sensor_iterator(i):
            img, aovs = mi.render(scene_dict, sensor=sensor, params=params, 
                                  spp=SPP * PRIMAL_SPP_MULT, spp_grad=SPP,
                                  seed=seed, seed_grad=seed + 1 + len(dataset.sensors))
            
            seed += 1 + len(dataset.sensors)

            ref_img = dataset.ref_images[idx][sensor.film().crop_size()[0]]
            
            albedo_priors_img = dataset.albedo_priors_images[idx][sensor.film().crop_size()[0]]
            roughness_priors_img = dataset.roughness_priors_images[idx][sensor.film().crop_size()[0]]
            normal_priors_img = dataset.normal_priors_images[idx][sensor.film().crop_size()[0]]

            # aovs
            albedo_img = aovs['albedo'][:, :, :3]
            roughness_img = aovs['roughness'][:, :, :1]
            metallic_img = aovs['metallic'][:, :, :1]
            depth_img = aovs['depth'][:, :, :1]
            normal_img = aovs['normal'][:, :, :3]
            
            normal_norm = np.linalg.norm(normal_img, axis=2, keepdims=True)
            normal_mask = normal_norm > 0.5
            normal_mask_flat = np.reshape(normal_mask, (-1,1)).squeeze()

            view_loss = l1(img, ref_img) / dataset.batch_size

            # priors loss follow GS-ID
            albedo_priors_loss = l2(albedo_img, albedo_priors_img) / dataset.batch_size
            roughness_priors_loss = l2(roughness_img, roughness_priors_img) / dataset.batch_size
            normal_priors_loss = l2(normal_img, normal_priors_img) / dataset.batch_size
            priors_loss = 0.5 * albedo_priors_loss + roughness_priors_loss + normal_priors_loss

            # loss follow GS-IR
            rgb_tv_loss = TV(ref_img, img) / dataset.batch_size
            material_tv_loss = TV(ref_img, dr.concat([albedo_img, roughness_img, metallic_img], axis=2)) / dataset.batch_size
            tv_loss = rgb_tv_loss + material_tv_loss
            
            lamb_loss = dr.mean(1.0-roughness_img) / dataset.batch_size

            # convert depth to fake_normal
            fake_normal_img = convert_depth_to_normal(depth_img, sensor)
            normal_loss = lnormal_sqr(normal_img, fake_normal_img, normal_mask_flat) / dataset.batch_size
              
            # smooth loss
            albedo_laplacian_loss = ldiscrete_laplacian_reg_3dims(opt['albedos'], kdtree_idx) / dataset.batch_size
            roughness_laplacian_loss = ldiscrete_laplacian_reg_1dim(opt['roughnesses'], kdtree_idx) / dataset.batch_size
            laplacian_loss = albedo_laplacian_loss + roughness_laplacian_loss

            # total loss
            total_loss = mi.TensorXf([0.0])
            if i < 64: # warm up
                total_loss += view_loss + 0.1 * normal_loss + 0.001 * lamb_loss + 0.1 * tv_loss + 1e-4 * laplacian_loss + 0.1 * priors_loss
            else:
                total_loss += view_loss + 0.1 * normal_loss + 0.001 * lamb_loss + 0.1 * tv_loss + 0.05 * priors_loss

            dr.backward(total_loss)

            loss += total_loss

            #----------------------------------save the results----------------------------------
            rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
            rgb_ref_bmp = resize_img(mi.Bitmap(ref_img),dataset.target_res)
            albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
            roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
            #metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
            depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), dataset.target_res)
            normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img + 1.0)/2 , 0))), dataset.target_res) 

            mi.util.write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}' + ('.png')), rgb_bmp)
            mi.util.write_bitmap(join(OUTPUT_OPT_DIR, f'opt-{i:04d}-{idx:02d}_ref' + ('.png')), rgb_ref_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_albedo' + ('.png')), albedo_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_roughness' + ('.png')), roughness_bmp)
            #mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_metallic' + ('.png')), metallic_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_depth' + ('.png')), depth_bmp)
            mi.util.write_bitmap(join(OUTPUT_EXTRA_DIR, f'opt-{i:04d}-{idx:02d}_normal' + ('.png')), normal_bmp)            

            rgb_psnr += lpsnr(ref_img, img) / dataset.batch_size

            if DATASET_TYPE == "TensoIR":
                ref_albedo = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
                ref_roughness = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]
                ref_normal = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]

                albedo_psnr += lpsnr(ref_albedo, albedo_img) / dataset.batch_size
                roughness_mse += l2(ref_roughness, roughness_img) / dataset.batch_size
                normal_mae += lmae(ref_normal, normal_img, normal_mask.squeeze()) / dataset.batch_size
            
        loss_list.append(np.asarray(total_loss))
        rgb_PSNR_list.append(np.asarray(rgb_psnr))

        if DATASET_TYPE == "TensoIR":
            normal_MAE_list.append(np.asarray(normal_mae))
            albedo_PSNR_list.append(np.asarray(albedo_psnr))
            roughness_MSE_list.append(np.asarray(roughness_mse))

        opt.step()
        update_params(opt, params)

        dataset.update_sensors(i)

        loss_np = np.asarray(loss)
        loss_str = f'Loss: {loss_np[0]:.4f}'
        pbar.set_description(loss_str)

        if DATASET_TYPE == "TensoIR":
            pbar.set_postfix({'rgb': rgb_psnr, 'albedo': albedo_psnr, 'roughness': roughness_mse, 'normal': normal_mae})
        else:
            pbar.set_postfix({'rgb': rgb_psnr})


        # save envmap
        if OPTIMIZE_ENVMAP:
            if SPHERICAL_GAUSSIAN:
                envmap_img = render_envmap_bitmap(params=params, num_sgs=NUM_SGS)
                mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'{i:04d}' + ('.png')), envmap_img)
                if (i in SAVE_ENVMAP_ITER) or i == train_conf.optimizer.iterations - 1:
                    save_sg_envmap(params, NUM_SGS, i)
            else:
                envmap_data = params['envmap.data']
                envmap_img = mi.Bitmap(envmap_data)
                mi.util.write_bitmap(join(OUTPUT_ENVMAP_DIR, f'{i:04d}' + ('.png')), envmap_img)

        if (i in dataset.render_upsample_iter) or i == train_conf.optimizer.iterations - 1:
            plot_loss(loss_list, label='Total Loss', output_file=join(OUTPUT_DIR, 'total_loss.png'))
            plot_loss(rgb_PSNR_list, label = "RGB PSNR", output_file=join(OUTPUT_DIR, 'rgb_psnr.png'))
            plot_loss(albedo_PSNR_list, label='Albedo PSNR', output_file=join(OUTPUT_DIR, 'albedo_psnr.png'))
            plot_loss(roughness_MSE_list, label='Roughness MSE', output_file=join(OUTPUT_DIR, 'roughness_mse.png'))
            plot_loss(normal_MAE_list, label='Normal MAE', output_file=join(OUTPUT_DIR, 'normal_mae.png'))

            gaussians.restore_from_params(params)
            save_path = f"{OUTPUT_PLY_DIR}/iter_{i:03d}.ply"
            gaussians.save_ply(save_path)
            print(f"[Iter {i}] Saved PLY to {save_path}")