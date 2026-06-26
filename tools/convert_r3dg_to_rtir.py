import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import torch
import numpy as np
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *
from utils import *
from models import *
from integrators import *
from datasets import *

scene_list = ['mic']

three_channel_ratio = {
    'armadillo': [0.5625, 0.44444444, 0.22727273],
    'hotdog': [2.11320755, 1.90909091, 1.12727273],
    'lego': [1.5, 1.50588235, 1.18181818],
    'jugs': [0.95238095, 0.875, 0.42857143],
    'ficus': [0.20652174, 0.24183007, 0.07843137],
    'mic': [0.80, 0.65, 0.45]
}

if __name__ == "__main__":
    gaussians = GaussianModel()

    for scene_name in scene_list:
        ply_path = f'./demos/scenes/TensoIR/{scene_name}.ply'
        out_path = f'./demos/scenes/TensoIR/{scene_name}_rtir.ply'

        gaussians.restore_from_ply(ply_path, False)
        
        gaussians.rescale_albedo(three_channel_ratio[scene_name])
        
        gaussians.save_ply(out_path)
