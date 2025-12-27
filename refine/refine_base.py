import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import drjit and mitsuba first, set variant immediately
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import torch
import tqdm
import numpy as np
from os.path import join
from omegaconf import OmegaConf

from constants import *
import optimizers
from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

class RefineBase:
    """Base class for refinement process"""
    
    def __init__(self):
        """Initialize the refinement base class"""
        self.refine_conf = OmegaConf.load('configs/refine.yaml')
        self.dataset = None
        self.gaussians = None
        self.gsstrategy = None
        self.ellipsoidsfactory = None
        self.gaussians_attributes = None
        self.scene_dict = None
        self.params = None
        self.opt = None
        self.seed = 0
        self.psnr_list = []
        self.mae_list = []
        self.opacities_scale = 1.0  # Default value, can be overridden in subclasses
    
    def load_data(self):
        """Load dataset and gaussian model"""
        self.dataset = Dataset(args.dataset_path, REFINE_UPSAMPLE_ITER)
        
        self.gaussians = GaussianModel()
        if args.ply_path.endswith(".ply"):
            self.gaussians.restore_from_ply(args.ply_path, args.reset_attribute)
        elif args.ply_path.endswith(".pt"):
            self.gaussians.restore_from_ckpt(args.ply_path)
        else:
            raise ValueError(f"Unsupported file type: {args.ply_path}")
    
    def initialize_models(self):
        """Initialize GS strategy and ellipsoids factory"""
        self.gsstrategy = GSStrategyModel('configs/gs.yaml')
        self.gsstrategy.sensors_normal = self.dataset.sensors_normal
        self.gsstrategy.sensors_intrinsic = self.dataset.sensors_intrinsic
        
        self.ellipsoidsfactory = EllipsoidsFactory()
        self.gaussians_attributes = self.ellipsoidsfactory.load_gaussian(gaussians=self.gaussians)
    
    def setup_scene(self):
        """Set up the Mitsuba scene"""
        self.scene_dict = mi.load_dict({
            'type': 'scene',
            'integrator': {
                'type': 'volprim_refine',
                'max_depth': 128,
                'geometry_threshold': args.geometry_threshold
            },
            'shape': {
                'type': 'ellipsoidsmesh',
                'centers': self.gaussians_attributes['centers'],
                'scales': self.gaussians_attributes['scales'],
                'quaternions': self.gaussians_attributes['quats'],
                'opacities': self.gaussians_attributes['sigmats'],
                'sh_coeffs': self.gaussians_attributes['features'],
                'normals': self.gaussians_attributes['normals']
            }
        })
    
    def setup_optimization(self):
        """Set up optimization parameters"""
        self.params = mi.traverse(self.scene_dict)
        self.params.keep(REFINE_PARAMS) 
        for _, param in self.params.items():
            dr.enable_grad(param)
        
        # Clear opacity and initialize normals
        n_ellipsoids = self.params['shape.opacities'].shape[0]
        self.params['shape.opacities'] = self.params['shape.opacities'] * self.opacities_scale
        
        m_normals = self.params['shape.normals'].shape[0] // n_ellipsoids
        self.params['shape.normals'] = dr.full(mi.Float, 0.1, n_ellipsoids * m_normals)
        
        # Set up optimizer
        self.opt = optimizers.BoundedAdam()
        ellipsoids = Ellipsoid.unravel(self.params['shape.data'])
        self.opt['centers'] = ellipsoids.center
        self.opt['scales']  = ellipsoids.scale
        self.opt['quats']   = mi.Vector4f(ellipsoids.quat)
        self.opt['opacities'] = self.params['shape.opacities']
        self.opt['sh_coeffs'] = self.params['shape.sh_coeffs']
        self.opt['normals'] = self.params['shape.normals']
        
        # Set learning rates
        self.opt.set_learning_rate({
            'centers':   self.refine_conf.optimizer.params.centers_lr,
            'scales':    self.refine_conf.optimizer.params.scales_lr,
            'quats':     self.refine_conf.optimizer.params.quats_lr,
            'opacities': self.refine_conf.optimizer.params.opacities_lr,
            'sh_coeffs': self.refine_conf.optimizer.params.sh_coeffs_lr,
            'normals':   self.refine_conf.optimizer.params.normals_lr
        })
        
        # Set bounds
        self.opt.set_bounds('scales',    lower=1e-6, upper=1e2)
        self.opt.set_bounds('opacities', lower=1e-6, upper=1-1e-6)
        self.opt.set_bounds('sh_coeffs', lower=-1, upper=1)
        self.opt.set_bounds('normals',   lower=-1, upper=1)
    
    def update_params(self):
        """Update scene parameters from optimizer"""
        self.params['shape.data'] = Ellipsoid.ravel(self.opt['centers'], self.opt['scales'], mi.Quaternion4f(self.opt['quats']))
        self.params['shape.opacities'] = self.opt['opacities']
        self.params['shape.sh_coeffs'] = self.opt['sh_coeffs']
        self.params['shape.normals'] = self.opt['normals']
        self.params.update()
    
    def calculate_loss(self, idx, sensor, img, aovs, ref_img):
        """Calculate loss for a single view (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement calculate_loss method")
    
    def render_and_optimize(self):
        """Main rendering and optimization loop"""
        # Ensure output directory exists
        ensure_dir(OUTPUT_REFINE_DIR)
        
        self.update_params()
        
        pbar = tqdm.tqdm(range(self.refine_conf.optimizer.iterations))
        for self.i in pbar:
            loss = mi.Float(0.0)
            rgb_psnr = mi.Float(0.0)
            normal_mae = mi.Float(0.0)
            
            self.gsstrategy.lr_schedule(self.opt, self.i, self.refine_conf.optimizer.iterations, self.refine_conf.optimizer.scheduler.min_factor)
            
            for idx, sensor in self.dataset.get_sensor_iterator():
                img, aovs = mi.render(self.scene_dict, sensor=sensor, params=self.params, 
                                      spp=args.refine_spp, spp_grad=1,
                                      seed=self.seed, seed_grad=self.seed+1+len(self.dataset.sensors))
                
                self.seed += 1 + len(self.dataset.sensors)
                
                ref_img = self.dataset.ref_images[idx][sensor.film().crop_size()[0]]
                
                # Calculate loss using subclass implementation
                view_loss, total_loss, rgb_psnr_val, normal_mae_val = self.calculate_loss(idx, sensor, img, aovs, ref_img)
                
                loss += total_loss
                rgb_psnr += rgb_psnr_val
                normal_mae += normal_mae_val
            
            # Update gradients and parameters
            self.gsstrategy.update_grad_norm(opt=self.opt)
            self.opt.step()
            
            self.gsstrategy.update_gs(opt=self.opt, step=self.i)
            self.update_params()
            
            # Scale value restraint
            data = self.params['shape.data']
            data_np = np.array(data)
            N = data_np.shape[0] // 10
            data_np = data_np.reshape(N, 10)
            data_np[:, 3:6] = np.clip(data_np[:, 3:6], 1e-3, 0.05)
            self.params['shape.data'] = dr.cuda.ad.Float(data_np.reshape(-1))
            
            self.dataset.update_sensors(self.i)
            
            self.psnr_list.append(rgb_psnr)
            self.mae_list.append(normal_mae)
            
            loss_np = np.asarray(loss)
            loss_str = f'Loss: {loss_np[0]:.4f}'
            pbar.set_description(loss_str)
            pbar.set_postfix({'rgb': rgb_psnr, 'normal': normal_mae})
            
            # Save intermediate results
            if (self.i > 0) and (self.i % 50 == 0): 
                self.gaussians.restore_from_params(self.params)
                self.gaussians.save_ply(args.refine_path)
    
    def save_results(self):
        """Save final results"""
        plot_loss(self.psnr_list, label='PSNR', output_file=join(OUTPUT_REFINE_DIR, 'psnr.png'))
        plot_loss(self.mae_list, label='MAE', output_file=join(OUTPUT_REFINE_DIR, 'mae.png'))
        
        # Save final PLY file
        self.gaussians.restore_from_params(self.params)
        self.gaussians.save_ply(args.refine_path)
    
    def run(self):
        """Run the complete refinement process"""
        self.load_data()
        self.initialize_models()
        self.setup_scene()
        self.setup_optimization()
        self.render_and_optimize()
        self.save_results()