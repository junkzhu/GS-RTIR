import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from os.path import join
import numpy as np
import torch
import imageio.v2 as imageio
from natsort import natsorted

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *

from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

def read_img(path, convert_to_linear=False):
    """Read and preprocess an image file"""
    img = imageio.imread(path).astype(np.float32) / 255.0
    if img.shape[-1] == 4:  # RGBA
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        rgb = rgb * alpha
        img = rgb
    if convert_to_linear:
        img = srgb_to_linear(img)  # Use the global function from utils.py
    return img

def readImages(path):
    renders = {
        "albedo_ori": [],
        "albedo_ref": []
    }
    image_names = []

    for fname in natsorted(os.listdir(path)):
        if not fname.endswith(".png"):
            continue
        
        name_no_ext = os.path.splitext(fname)[0]
        base_name = name_no_ext.split("_")[0]

        if base_name not in image_names:
            image_names.append(base_name)

    for idx, name in enumerate(image_names):
        real_idx = idx * args.stride
        name = f"{real_idx:02d}"

        renders["albedo_ori"].append(read_img(os.path.join(path, f"{name}_albedo_ori.png"), convert_to_linear=True)) # read_img func will convert the image to np array
        renders["albedo_ref"].append(read_img(os.path.join(path, f"{name}_albedo_ref.png"), convert_to_linear=True))
    
    return renders

def read_RGB_images(renders_dir):
    renders = {
        "rgb": []
    }
    image_paths = []

    for fname in natsorted(os.listdir(renders_dir)):
        if not fname.endswith(".png"):
            continue

        image_paths.append(fname)

    for image_path in image_paths:
        renders["rgb"].append(read_img(os.path.join(renders_dir, image_path)))
    
    return renders

def to_torch_image(img):
    if isinstance(img, dr.ArrayBase):
        img = np.array(img)

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported type {type(img)}")
    
    img = img.astype(np.float32)
    
    img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return img_torch

def rescale_albedo(path):

    renders = readImages(path)

    single_channel_ratio, three_channel_ratio = compute_rescale_ratio(renders["albedo_ref"], renders["albedo_ori"])
    print("Albedo scale:", three_channel_ratio)

    for idx, albedo_img in enumerate(renders["albedo_ori"]):
        real_idx = idx * args.stride

        albedo_img = mi.Bitmap(albedo_img)
        albedo_bmp = albedo_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32)
        albedo_img = three_channel_ratio * mi.TensorXf(albedo_bmp)
        albedo_bmp = mi.Bitmap(albedo_img)

        write_bitmap(join(path, f'{real_idx:02d}_albedo' + ('.png')), albedo_bmp)

if __name__ == "__main__":
    # python tools/rescale_albedo.py
    
    args.dataset_name = "hotdog"
    path = f"outputs/TensoIR/{args.dataset_name}/renders"
    
    rescale_albedo(path)
