import tqdm
import numpy as np
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

import lpips
llpips = lpips.LPIPS(net='vgg')

def readImages(renders_dir):
    renders = {
        "rgb": [],
        "albedo": [],
        "roughness": [],
        "normal": []
    }
    image_names = []

    for fname in natsorted(os.listdir(renders_dir)):
        if not fname.endswith(".png"):
            continue
        base_name = fname.split("_")[0]

        if base_name not in image_names:
            image_names.append(base_name)

    for name in image_names:
        def srgb_to_linear(img):
            """Convert sRGB to linear RGB (gamma correction)"""
            return img ** 2.2

        def read_img(path):
            img = imageio.imread(path).astype(np.float32) / 255.0
            if img.shape[-1] == 4:  # RGBA
                rgb = img[..., :3]
                alpha = img[..., 3:4]
                rgb = rgb * alpha
                img = rgb
            img = srgb_to_linear(img)
            return img

        renders["rgb"].append(read_img(os.path.join(renders_dir, f"{name}_rgb.png")))
        renders["albedo"].append(read_img(os.path.join(renders_dir, f"{name}_albedo.png")))
        renders["roughness"].append(read_img(os.path.join(renders_dir, f"{name}_roughness.png")))
        renders["normal"].append(read_img(os.path.join(renders_dir, f"{name}_normal.png")))
    
    return renders

def to_torch_image(img):
    if isinstance(img, dr.ArrayBase):
        img = np.array(img)

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported type {type(img)}")
    
    img = img.astype(np.float32)
    
    img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return img_torch

if __name__ == "__main__":
    
    dataset = Dataset(DATASET_PATH, RENDER_UPSAMPLE_ITER, "test")

    renders = readImages(OUTPUT_RENDER_DIR)

    metrics = {
        "psnr_rgb": [],
        "ssim_rgb": [],
        "lpips_rgb": [],

        "psnr_albedo": [],
        "ssim_albedo": [],
        "lpips_albedo": [],

        "l2_roughness": [],
        "lmae_normal": []
    }

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Metrics")

    for idx, sensor in pbar:
        rgb_img       = renders["rgb"][idx][:, :, :3]
        albedo_img    = renders["albedo"][idx][:, :, :3]
        roughness_img = renders["roughness"][idx][:, :, :3]
        normal_img    = renders["normal"][idx][:, :, :3]

        ref_rgb       = dataset.ref_images[idx][sensor.film().crop_size()[0]]
        ref_albedo    = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
        ref_roughness = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]
        ref_normal    = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]

        normal_mask = np.any(ref_normal != 0, axis=2, keepdims=True)

        psnr_rgb_val = lpsnr(ref_rgb, rgb_img)
        ssim_rgb_val = lssim(ref_rgb, rgb_img)
        lpips_rgb_val = llpips(to_torch_image(ref_rgb), to_torch_image(rgb_img)).detach().item()

        psnr_alb_val = lpsnr(ref_albedo, albedo_img)
        ssim_alb_val = lssim(ref_albedo, albedo_img)
        lpips_alb_val = llpips(to_torch_image(ref_albedo), to_torch_image(albedo_img)).detach().item()

        l2_rough_val = l2(ref_roughness, roughness_img).numpy()
        lmae_norm_val = lmae(ref_normal, normal_img, normal_mask.squeeze())

        metrics["psnr_rgb"].append(psnr_rgb_val)
        metrics["ssim_rgb"].append(ssim_rgb_val)
        metrics["lpips_rgb"].append(lpips_rgb_val)

        metrics["psnr_albedo"].append(psnr_alb_val)
        metrics["ssim_albedo"].append(ssim_alb_val)
        metrics["lpips_albedo"].append(lpips_alb_val)

        metrics["l2_roughness"].append(l2_rough_val)
        metrics["lmae_normal"].append(lmae_norm_val)

    print("\n=== Final Averages ===")
    print(f"PSNR (RGB):       {np.mean(metrics['psnr_rgb']):.3f}")
    print(f"SSIM (RGB):       {np.mean(metrics['ssim_rgb']):.3f}")
    print(f"LPIPS (RGB):      {np.mean(metrics['lpips_rgb']):.3f}")

    print(f"PSNR (Albedo):    {np.mean(metrics['psnr_albedo']):.3f}")
    print(f"SSIM (Albedo):    {np.mean(metrics['ssim_albedo']):.3f}")
    print(f"LPIPS (Albedo):   {np.mean(metrics['lpips_albedo']):.3f}")

    print(f"L2 (Roughness):   {np.mean(metrics['l2_roughness']):.4f}")
    print(f"LMAE (Normal):    {np.mean(metrics['lmae_normal']):.4f}")

    save_path = os.path.join(OUTPUT_RENDER_DIR, "results_metrics.json")
    np.savez(save_path, **metrics)
    print(f"[INFO] Metrics saved to {save_path}")
