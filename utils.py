import drjit as dr
import mitsuba as mi
import tqdm
from jinja2 import Template
import numpy as np
import torch
from emitter.sgenvmap_util import SG2Envmap
import glob
import os
from pathlib import Path
from constants import *

def resize_img(img, target_res, smooth=False):
    """Resizes a Mitsuba Bitmap using either a box filter (smooth=False)
       or a gaussian filter (smooth=True)"""
    assert isinstance(img, mi.Bitmap)
    source_res = img.size()
    if target_res[0] == source_res[0] and target_res[1] == source_res[1]:
        return img
    return img.resample(mi.ScalarVector2u(target_res[1], target_res[0]))

def set_sensor_res(sensor, res):
    """Sets the resolution of an existing Mitsuba sensor"""
    params = mi.traverse(sensor)
    params['film.size'] = res
    sensor.parameters_changed()
    params.update()

def _shift_tensor_2d(x, dy, dx):
    """
    Shift a (H, W) tensor by (dy, dx) with edge replication.
    dy > 0 : shift down
    dy < 0 : shift up
    dx > 0 : shift right
    dx < 0 : shift left
    """
    H, W = x.shape[:2]

    # --- shift in y ---
    if dy > 0:
        # pad top with dy rows
        pad = x[0:1, :]
        pad = dr.concat([pad] * dy, axis=0)
        x = dr.concat([pad, x[:H-dy, :]], axis=0)

    elif dy < 0:
        # pad bottom with -dy rows
        pad = x[H-1:H, :]
        pad = dr.concat([pad] * (-dy), axis=0)
        x = dr.concat([x[-dy:, :], pad], axis=0)

    # --- shift in x ---
    if dx > 0:
        # pad left with dx columns
        pad = x[:, 0:1]
        pad = dr.concat([pad] * dx, axis=1)
        x = dr.concat([pad, x[:, :W-dx]], axis=1)

    elif dx < 0:
        # pad right with -dx columns
        pad = x[:, W-1:W]
        pad = dr.concat([pad] * (-dx), axis=1)
        x = dr.concat([x[:, -dx:], pad], axis=1)

    return x


def convert_depth_to_normal(depth_map, sensor, fov_deg=39.0, pad=2, depth_thresh=0.01):
    """
    Fully differentiable wrt depth_map (no NumPy ops that depend on depth).
    - depth_map: (H, W) or (H, W, 1) drjit tensor (TensorXf) preferred
    - returns: (H, W, 3) normal in world space
    """

    # --- Ensure depth is a Dr.Jit tensor (keep AD graph if it exists) ---
    # Accept (H,W) or (H,W,1)
    if hasattr(depth_map, "shape"):
        pass
    else:
        # fallback: make a tensor (this path is non-AD if input is plain array)
        depth_map = mi.TensorXf(depth_map)

    H, W = depth_map.shape[:2]

    # Make z shape (H,W)
    if len(depth_map.shape) == 3:
        z = depth_map[:, :, 0]
    else:
        z = depth_map

    # --- Camera intrinsics ---
    fov_rad = float(fov_deg * np.pi / 180.0)
    fx = fy = float(W / (2.0 * np.tan(fov_rad / 2.0)))
    cx, cy = float(W / 2.0), float(H / 2.0)

    # --- Differentiable "valid mask": window min-filter > thresh ---
    # This replaces NumPy sliding_window_view + np.all(...)
    depth_min = z
    if pad > 0:
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                shifted = _shift_tensor_2d(z, dy, dx)
                depth_min = dr.minimum(depth_min, shifted)

    mask2d = depth_min > depth_thresh  # (H,W) boolean tensor (drjit)
    mask = mask2d[:, :, None]          # (H,W,1)

    # --- Build pixel grids (xx, yy) ---
    # This is OK to do with NumPy because it does NOT depend on depth,
    # and it won't break depth gradients.
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    xx = mi.TensorXf(xx)
    yy = mi.TensorXf(yy)

    # --- Backproject to camera space points P(x,y) ---
    X = (xx - cx) * z / fx
    Y = (yy - cy) * z / fy
    P = dr.concat([X[:, :, None], Y[:, :, None], z[:, :, None]], axis=2)  # (H,W,3)

    # Neighbor points (edge replicated)
    P_left  = dr.concat([P[:, 0:1, :],  P[:, :-1, :]], axis=1)
    P_right = dr.concat([P[:, 1:, :],   P[:, -1:, :]], axis=1)
    P_up    = dr.concat([P[0:1, :, :],  P[:-1, :, :]], axis=0)
    P_down  = dr.concat([P[1:, :, :],   P[-1:, :, :]], axis=0)

    # Depth neighbors for adaptive differencing (still differentiable)
    depth_left  = dr.concat([z[:, 0:1],  z[:, :-1]], axis=1)
    depth_right = dr.concat([z[:, 1:],   z[:, -1:]], axis=1)
    depth_up    = dr.concat([z[0:1, :],  z[:-1, :]], axis=0)
    depth_down  = dr.concat([z[1:, :],   z[-1:, :]], axis=0)

    cond_x = dr.abs(depth_left - z) < dr.abs(depth_right - z)
    ddx = dr.select(cond_x[:, :, None], P - P_left, P_right - P)

    cond_y = dr.abs(depth_down - z) < dr.abs(depth_up - z)
    ddy = dr.select(cond_y[:, :, None], P - P_down, P_up - P)

    # Cross product ddx x ddy
    nx = ddx[..., 1] * ddy[..., 2] - ddx[..., 2] * ddy[..., 1]
    ny = ddx[..., 2] * ddy[..., 0] - ddx[..., 0] * ddy[..., 2]
    nz = ddx[..., 0] * ddy[..., 1] - ddx[..., 1] * ddy[..., 0]

    # Normalize
    norm = dr.sqrt(nx * nx + ny * ny + nz * nz)
    norm = dr.maximum(norm, 1e-8)
    nx = -nx / norm
    ny = -ny / norm
    nz =  nz / norm

    # Rotate to world (constant w.r.t depth; does NOT break depth gradients)
    c2w = sensor.world_transform().matrix
    normal_x = nx * c2w[0, 0] + ny * c2w[0, 1] + nz * c2w[0, 2]
    normal_y = nx * c2w[1, 0] + ny * c2w[1, 1] + nz * c2w[1, 2]
    normal_z = nx * c2w[2, 0] + ny * c2w[2, 1] + nz * c2w[2, 2]

    normal = dr.concat([normal_x[:, :, None], normal_y[:, :, None], normal_z[:, :, None]], axis=2)  # (H,W,3)

    # Apply mask (mask comes from Dr.Jit ops on depth_min, so graph is intact)
    normal = dr.select(mask, normal, dr.zeros_like(normal))

    return normal

def opacity_lamb_loss(opacities: mi.Float):
    return dr.mean(dr.square(1 - opacities))

def opacity_entropy_loss(opacities: mi.Float):
    """
    Entropy-style regularizer encouraging α→0 or 1
    L = -mean(α*log(α+eps) + (1-α)*log(1-α+eps))
    """
    eps = 1e-6
    return -dr.mean(opacities * dr.log(opacities + eps) + (1 - opacities) * dr.log(1 - opacities + eps))

def plot_loss(data, label, output_file):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(data)
    fig.canvas.toolbar_visible = 'fade-in-fade-out'
    fig.canvas.footer_visible = False
    fig.canvas.header_visible = False
    ax.set_xlabel('Iteration')
    plt.grid(True)
    plt.ylabel(label)
    plt.title(label + ' plot')
    plt.savefig(output_file)

def get_lgtSGs(params, num_sgs):
    envmap_template = """{% for i in range(num_sgs) %}
lobe_{{ i }} = params['envmap.lgtSGslobe_{{ i }}'].numpy()
lambda_{{ i }} = params['envmap.lgtSGslambda_{{ i }}'].numpy()
mu_{{ i }} = params['envmap.lgtSGsmu_{{ i }}'].numpy()
{% endfor %}
lgtSGs = [ {% for i in range(num_sgs) %} lobe_{{ i }}, lambda_{{ i }}, mu_{{ i }} {% if not loop.last %}, {% endif %} {% endfor %} ]
"""
    envmap_template = Template(envmap_template)
    envmap_data = envmap_template.render(num_sgs=num_sgs)
    # Execute into an isolated local dict and include params in that dict
    envmap_locals = {'params': params}
    exec(envmap_data, globals(), envmap_locals)

    # extract lgtSGs and params (if modified by the exec code)
    return np.array(np.concatenate([np.ravel(x).astype(np.float32) for x in envmap_locals.get('lgtSGs')]), dtype=np.float32).reshape(-1, 7)


def render_envmap_bitmap(params, num_sgs):
    lgtSGs = get_lgtSGs(params, num_sgs)
    lgtSGs_np = np.array(lgtSGs, dtype=np.float32)

    envmap = SG2Envmap(lgtSGs_np, 256, 512)
    return mi.Bitmap(envmap)

def save_sg_envmap(params, num_sgs, iter):
    lgtSGs = get_lgtSGs(params, num_sgs)
    np.save(f"{OUTPUT_ENVMAP_DIR}/optimized_sgs_{iter:04d}.npy", lgtSGs)

def compute_rescale_ratio(gt_albedo_list, albedo_list):
    
    gt_albedo_flat_list = []
    albedo_flat_list = []
    
    for gt_albedo, albedo in zip(gt_albedo_list, albedo_list):
        
        gt_albedo = torch.from_numpy(np.array(gt_albedo))
        albedo = torch.from_numpy(np.array(albedo))

        gt_albedo_flat = gt_albedo.reshape(-1, 3)
        albedo_flat = albedo.reshape(-1, 3)
    
        mask = ~(albedo_flat == 0).all(dim=1)

        gt_albedo_flat = gt_albedo_flat[mask]
        albedo_flat = albedo_flat[mask]

        gt_albedo_flat_list.append(gt_albedo_flat)
        albedo_flat_list.append(albedo_flat)

    gt_all = torch.cat(gt_albedo_flat_list, dim=0)
    albedo_all = torch.cat(albedo_flat_list, dim=0)

    single_channel_ratio = (gt_all / albedo_all.clamp(min=1e-6))[..., 0].median()
    three_channel_ratio, _ = (gt_all / albedo_all.clamp(min=1e-6)).median(dim=0)

    if (three_channel_ratio < 0.1).any():
        three_channel_ratio = [(gt_all/albedo_all.clamp_min(1e-6))[..., 0].median().item()] * 3 # follow IRGS

    return mi.TensorXf(single_channel_ratio), mi.TensorXf(three_channel_ratio)

def load_hdr_paths(root_dir):
    hdr_paths = glob.glob(os.path.join(root_dir, "*.hdr"))
    hdr_paths.sort()
    return hdr_paths

def get_relighting_envmap_names(root_dir):
    name_list = []
    envmaps = load_hdr_paths(root_dir)
    for envmap in envmaps:
        #create folder
        envmap_name = Path(envmap).stem
        name_list.append(envmap_name)
    return name_list