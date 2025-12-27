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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

llpips = lpips.LPIPS(net='vgg').to(device)

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
        
        name_no_ext = os.path.splitext(fname)[0]
        base_name = name_no_ext.split("_")[0]

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

        renders["rgb"].append(read_img(os.path.join(renders_dir, f"{name}.png")))
        renders["albedo"].append(read_img(os.path.join(renders_dir, f"{name}_albedo.png")))
        renders["roughness"].append(read_img(os.path.join(renders_dir, f"{name}_roughness.png")))
        renders["normal"].append(read_img(os.path.join(renders_dir, f"{name}_normal.png")))
    
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

def metrics_training_envmap():
    dataset = Dataset(args.dataset_path, RENDER_UPSAMPLE_ITER, "test")

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

    lpips_rgb_ref_batch = []
    lpips_rgb_pred_batch = []

    lpips_alb_ref_batch = []
    lpips_alb_pred_batch = []

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Metrics")

    for idx, sensor in pbar:
        rgb_img       = renders["rgb"][idx][:, :, :3]
        albedo_img    = renders["albedo"][idx][:, :, :3]
        roughness_img = renders["roughness"][idx][:, :, :3]
        normal_img    = renders["normal"][idx][:, :, :3]

        ref_rgb       = dataset.ref_images[idx][sensor.film().crop_size()[0]]
        ref_albedo    = dataset.ref_albedo_images[idx][sensor.film().crop_size()[0]]
        ref_roughness = dataset.ref_roughness_images[idx][sensor.film().crop_size()[0]]

        #TODO: temporarily use a mask to exclude the black pixels
        mask = (ref_rgb != 0)
        rgb_img       = dr.select(mask, rgb_img, 0.0)
        albedo_img    = dr.select(mask, albedo_img, 0.0)
        roughness_img = dr.select(mask, roughness_img, 0.0)
        normal_img    = dr.select(mask, normal_img, 0.0)

        psnr_rgb_val = lpsnr(ref_rgb, rgb_img)
        ssim_rgb_val = lssim(ref_rgb, rgb_img)

        psnr_alb_val = lpsnr(ref_albedo, albedo_img)
        ssim_alb_val = lssim(ref_albedo, albedo_img)
        
        l2_rough_val = l2(ref_roughness, roughness_img).numpy()
       
        metrics["psnr_rgb"].append(psnr_rgb_val)
        metrics["ssim_rgb"].append(ssim_rgb_val)

        metrics["psnr_albedo"].append(psnr_alb_val)
        metrics["ssim_albedo"].append(ssim_alb_val)

        metrics["l2_roughness"].append(l2_rough_val)

        lpips_rgb_ref_batch.append(
            to_torch_image(ref_rgb).to(device).squeeze(0)
        )
        lpips_rgb_pred_batch.append(
            to_torch_image(rgb_img).to(device).squeeze(0)
        )

        lpips_alb_ref_batch.append(
            to_torch_image(ref_albedo).to(device).squeeze(0)
        )
        lpips_alb_pred_batch.append(
            to_torch_image(albedo_img).to(device).squeeze(0)
        )

        if args.dataset_type == "TensoIR":
            ref_normal    = dataset.ref_normal_images[idx][sensor.film().crop_size()[0]]
            normal_mask = np.any(ref_normal != 0, axis=2, keepdims=True)
            lmae_norm_val = lmae(ref_normal, normal_img, normal_mask.squeeze())
            metrics["lmae_normal"].append(lmae_norm_val)
        
    print("Start caculate lpips...")
    lpips_rgb_vals = []
    lpips_alb_vals = []
    with torch.no_grad():
        B = 4

        for i in range(0, len(lpips_rgb_ref_batch), B):
            ref = torch.stack(lpips_rgb_ref_batch[i:i+B], dim=0).to(device).float()
            pred = torch.stack(lpips_rgb_pred_batch[i:i+B], dim=0).to(device).float()

            v = llpips(ref, pred)
            lpips_rgb_vals.append(v.detach().cpu())

            del ref, pred, v
            torch.cuda.empty_cache()

        lpips_rgb_vals = torch.cat(lpips_rgb_vals, dim=0)

        for i in range(0, len(lpips_alb_ref_batch), B):
            ref = torch.stack(lpips_alb_ref_batch[i:i+B], dim=0).to(device).float()
            pred = torch.stack(lpips_alb_pred_batch[i:i+B], dim=0).to(device).float()

            v = llpips(ref, pred)
            lpips_alb_vals.append(v.detach().cpu())

            del ref, pred, v
            torch.cuda.empty_cache()

        lpips_alb_vals = torch.cat(lpips_alb_vals, dim=0)

    for v in lpips_rgb_vals:
        metrics["lpips_rgb"].append(float(v))

    for v in lpips_alb_vals:
        metrics["lpips_albedo"].append(float(v))

    metrics["psnr_rgb_mean"] = float(np.mean(metrics['psnr_rgb']))
    metrics["ssim_rgb_mean"] = float(np.mean(metrics['ssim_rgb']))
    metrics["lpips_rgb_mean"] = float(np.mean(metrics['lpips_rgb']))

    metrics["psnr_albedo_mean"] = float(np.mean(metrics['psnr_albedo']))
    metrics["ssim_albedo_mean"] = float(np.mean(metrics['ssim_albedo']))
    metrics["lpips_albedo_mean"] = float(np.mean(metrics['lpips_albedo']))

    metrics["l2_roughness_mean"] = float(np.mean(metrics['l2_roughness']))

    print("\n=== Final Averages ===")
    print(f"PSNR (RGB):       {np.mean(metrics['psnr_rgb']):.3f}")
    print(f"SSIM (RGB):       {np.mean(metrics['ssim_rgb']):.3f}")
    print(f"LPIPS (RGB):      {np.mean(metrics['lpips_rgb']):.3f}")

    print(f"PSNR (Albedo):    {np.mean(metrics['psnr_albedo']):.3f}")
    print(f"SSIM (Albedo):    {np.mean(metrics['ssim_albedo']):.3f}")
    print(f"LPIPS (Albedo):   {np.mean(metrics['lpips_albedo']):.3f}")

    print(f"L2 (Roughness):   {np.mean(metrics['l2_roughness']):.4f}")
    
    if args.dataset_type == "TensoIR":
        metrics["lmae_normal_mean"] = float(np.mean(metrics['lmae_normal']))
        print(f"LMAE (Normal):    {np.mean(metrics['lmae_normal']):.4f}")

    save_path = os.path.join(OUTPUT_RENDER_DIR, "results_metrics.json")
    np.savez(save_path, **metrics)
    print(f"[INFO] Metrics saved to {save_path}")

def metrics_relighting_envmap(envmap_name):
    dataset = Dataset(args.dataset_path, RENDER_UPSAMPLE_ITER, env=envmap_name, dataset_type="test", load_ref_relight_images=args.relight)

    current_env_images_path = os.path.join(OUTPUT_RENDER_DIR, f'./relight/{envmap_name}')
    renders = read_RGB_images(current_env_images_path)

    metrics = {
        "psnr_rgb": [],
        "ssim_rgb": [],
        "lpips_rgb": []
    }

    lpips_ref_batch = []
    lpips_pred_batch = []

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Relighting Metrics")

    for idx, sensor in pbar:
        rgb_img       = renders["rgb"][idx][:, :, :3]
        ref_rgb       = dataset.ref_relight_images[envmap_name][idx]

        psnr_rgb_val = lpsnr(ref_rgb, rgb_img)
        ssim_rgb_val = lssim(ref_rgb, rgb_img)

        metrics["psnr_rgb"].append(psnr_rgb_val)
        metrics["ssim_rgb"].append(ssim_rgb_val)

        lpips_ref_batch.append(
            to_torch_image(ref_rgb).to(device).squeeze(0)
        )
        lpips_pred_batch.append(
            to_torch_image(rgb_img).to(device).squeeze(0)
        )

    print("Start caculate lpips...")
    lpips_rgb_vals = []
    with torch.no_grad():
        B = 4
        for i in range(0, len(lpips_ref_batch), B):
            ref = torch.stack(lpips_ref_batch[i:i+B], dim=0).to(device).float()
            pred = torch.stack(lpips_pred_batch[i:i+B], dim=0).to(device).float()

            v = llpips(ref, pred)
            lpips_rgb_vals.append(v.detach().cpu())

            del ref, pred, v
            torch.cuda.empty_cache()
    
    lpips_rgb_vals = torch.cat(lpips_rgb_vals, dim=0)

    for v in lpips_rgb_vals:
        metrics["lpips_rgb"].append(float(v))

    metrics["psnr_rgb_mean"] = float(np.mean(metrics['psnr_rgb']))
    metrics["ssim_rgb_mean"] = float(np.mean(metrics['ssim_rgb']))
    metrics["lpips_rgb_mean"] = float(np.mean(metrics['lpips_rgb']))
    
    print(f"\n=== {envmap_name} Final Averages ===")
    print(f"PSNR (RGB):       {np.mean(metrics['psnr_rgb']):.3f}")
    print(f"SSIM (RGB):       {np.mean(metrics['ssim_rgb']):.3f}")
    print(f"LPIPS (RGB):      {np.mean(metrics['lpips_rgb']):.3f}")

    save_path = os.path.join(OUTPUT_RENDER_DIR, f"{envmap_name}_results_metrics.json")
    np.savez(save_path, **metrics)
    print(f"[INFO] Metrics saved to {save_path}")

if __name__ == "__main__":
    
    metrics_training_envmap()

    if args.relight:
        envmap_name_list = get_relighting_envmap_names(args.envmap_root)
        for envmap_name in envmap_name_list:
            metrics_relighting_envmap(envmap_name)