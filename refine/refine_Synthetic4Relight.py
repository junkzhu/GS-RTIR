import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import numpy as np
from os.path import join

from constants import *
from utils import *
from losses import *

try:
    from .refine_base import RefineBase
except ImportError:
    from refine_base import RefineBase

class RefineSynthetic4Relight(RefineBase):
    """Refinement class for Synthetic4Relight dataset"""
    
    def __init__(self):
        """Initialize the refinement class"""
        super().__init__()
        self.opacities_scale = 2.0
    
    def calculate_loss(self, idx, sensor, img, aovs, ref_img):
        """Calculate loss for Synthetic4Relight dataset"""
        # Get normal priors image
        normal_priors_img = self.dataset.normal_priors_images[idx][sensor.film().crop_size()[0]]
        
        # Extract AOVs
        depth_img = aovs['depth'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]
        normal_norm = np.linalg.norm(normal_img, axis=2, keepdims=True)
        normal_mask = normal_norm > 0.1
        normal_mask_flat = np.reshape(normal_mask, (-1, 1)).squeeze()
        
        # Calculate losses
        view_loss = l1(ref_img, img, convert_to_srgb=True) / self.dataset.batch_size
        
        # Normal priors loss
        normal_priors_loss = l2(normal_priors_img, normal_img) / self.dataset.batch_size
        
        # Convert depth to fake normal and calculate loss
        fake_normal_img = convert_depth_to_normal(depth_img, sensor)
        normal_tv_loss = TV(normal_priors_img, normal_img) / self.dataset.batch_size
        fake_normal_loss = lnormal_sqr(dr.detach(fake_normal_img), normal_img, normal_mask_flat) / self.dataset.batch_size
        
        # Total loss
        total_loss = normal_priors_loss + normal_tv_loss + fake_normal_loss
        
        # Backward pass
        dr.backward(total_loss)
        
        # Render and save intermediate results
        rgb_bmp = resize_img(mi.Bitmap(img), self.dataset.target_res)
        rgb_ref_bmp = resize_img(mi.Bitmap(ref_img), self.dataset.target_res)
        depth_bmp = resize_img(mi.Bitmap(depth_img/dr.max(depth_img)), self.dataset.target_res)
        normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), self.dataset.target_res)
        fake_normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (fake_normal_img+1)/2, 0))), self.dataset.target_res)
        
        write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{self.i:04d}-{idx:02d}.png'), rgb_bmp)
        write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{self.i:04d}-{idx:02d}_ref.png'), rgb_ref_bmp)
        write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{self.i:04d}-{idx:02d}_depth.png'), depth_bmp)
        write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{self.i:04d}-{idx:02d}_normal.png'), normal_bmp)      
        write_bitmap(join(OUTPUT_REFINE_DIR, f'opt-{self.i:04d}-{idx:02d}_fake_normal.png'), fake_normal_bmp)      
        
        # Calculate PSNR and MAE
        rgb_psnr_val = lpsnr(ref_img, img, convert_to_srgb=True) / self.dataset.batch_size
        normal_mae_val = lmae(normal_priors_img, normal_img, normal_mask.squeeze()) / self.dataset.batch_size
        
        return view_loss, total_loss, rgb_psnr_val, normal_mae_val

if __name__ == "__main__":
    refine = RefineSynthetic4Relight()
    refine.run()