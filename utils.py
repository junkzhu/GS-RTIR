import drjit as dr
import mitsuba as mi
import tqdm
from jinja2 import Template
import numpy as np
import torch
import time
from functools import wraps
from contextlib import contextmanager
from emitter.sgenvmap_util import SG2Envmap
import glob
import os
from pathlib import Path
from constants import *
from os.path import join

def make_half_topdown_orbit_sensors_zup(
    resx: int,
    resy: int,
    fov: float,
    n_frames: int = 120,
    center=(0.0, 0.0, 0.0),
    radius: float = 2.0,
    height: float = 1.0,
    start_azimuth_deg: float = 0.0,
    clockwise: bool = True,
    up=(0.0, 0.0, 1.0),           # Z-up
    spp: int = 1,
):
    sensors = []

    sampler = mi.load_dict({
        'type': 'independent',
        'sample_count': int(spp),
    })

    film = mi.load_dict({
        'type': 'hdrfilm',
        'width': int(resx),
        'height': int(resy),
        'pixel_format': 'rgb',
        'pixel_filter': {'type': 'tent'},
        'sample_border': True
    })

    center = np.array(center, dtype=np.float32)
    up_v = mi.ScalarVector3f(*up)

    sign = -1.0 if clockwise else 1.0
    start = np.deg2rad(start_azimuth_deg)

    for i in range(int(n_frames)):
        t = i / float(n_frames)
        az = start + sign * (2.0 * np.pi * t)

        cam_pos = mi.ScalarPoint3f(
            float(center[0] + radius * np.cos(az)),
            float(center[1] + radius * np.sin(az)),
            float(center[2] + height),
        )

        target = mi.ScalarPoint3f(float(center[0]), float(center[1]), float(center[2]))

        to_world = mi.ScalarTransform4f.look_at(
            origin=cam_pos,
            target=target,
            up=up_v
        )

        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': float(fov),
            'to_world': to_world,
            'sampler': sampler,
            'film': film
        }))

    return sensors

def load_sensors(data_path, resx, resy, fov):
    sensors = []
    sampler = mi.load_dict({'type': 'independent'})
    film = mi.load_dict({
        'type': 'hdrfilm', 
        'width': resx, 
        'height': resy,
        'pixel_format': 'rgb', 
        'pixel_filter': {'type': 'tent'}, 
        'sample_border': True
    })
    
    transforms_file = os.path.join(data_path, f'transforms_test.json')
    if not os.path.exists(transforms_file):
        print(f"Transforms file not found: {transforms_file}")
        return []
    
    try:
        with open(transforms_file, 'r') as f:
            transforms_data = json.load(f)
    except Exception as e:
        print(f"Error reading transforms file: {e}")
        return []
    
    camera_angle_x = transforms_data.get('camera_angle_x', None)
    if camera_angle_x is None:
        print("Warning: camera_angle_x not found in transforms file")
        return []
    
    focal_length = 0.5 * resx / np.tan(0.5 * camera_angle_x)
    if fov is None:
        fov = np.degrees(camera_angle_x)
    
    frames = transforms_data.get('frames', [])
    n_cameras = len(frames)
    
    camera_indices = list(range(n_cameras))

    scene_center = np.array([0.0, 0.0, 0.0])

    for idx, frame_idx in enumerate(camera_indices):
        if frame_idx >= n_cameras:
            print(f"Warning: Camera index {frame_idx} exceeds available cameras ({n_cameras}), skipping")
            continue
        
        frame = frames[frame_idx]
        
        if 'transform_matrix' not in frame:
            print(f"Warning: Camera {frame_idx} missing transform_matrix, skipping")
            continue
        
        transform_matrix = np.array(frame['transform_matrix'])
        
        if transform_matrix.shape != (4, 4):
            print(f"Warning: Camera {frame_idx} has incorrect transform matrix shape, skipping")
            continue
        
        c2w = transform_matrix.copy()
        
        c2w[:3, 3] *= 1.0
        c2w[:3, 3] += np.array([0.0, 0.0, 0.0])

        #https://euruson.com/post/225c1af2-1435-804f-a7c8-d3a78006d48c
        coord_transform = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        c2w = c2w @ coord_transform
                
        rotation_part = c2w[:3, :3]         
        u, s, vh = np.linalg.svd(rotation_part)         
        c2w[:3, :3] = u @ vh

        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f(c2w),
            'sampler': sampler,
            'film': film
        })
        
        sensors.append(sensor)
    return sensors

def write_bitmap(path, img, keep_gamma=False):
    if keep_gamma:
        img = srgb_to_linear(img)
    mi.util.write_bitmap(path, img)

def srgb_to_linear(img, clip=True):
    img = np.asarray(img, dtype=np.float32)

    if clip:
        img = np.clip(img, 0.0, 1.0)

    return np.where(
        img < 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )

def linear_to_srgb(img, clip=True):
    img = np.asarray(img, dtype=np.float32)

    if clip:
        img = np.clip(img, 0.0, 1.0)

    return np.where(
        img < 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055
    )

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

@contextmanager
def time_measure(description="Operation"):
    """
    Context manager to measure execution time of a block of code.
    
    Args:
        description: Description of the operation being timed
    
    Yields:
        None
    
    Example:
        with time_measure("Loading dataset"):
            dataset = Dataset(args.dataset_path)
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{description} time: {elapsed_time:.4f} seconds")

def time_it(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to be timed
    
    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} execution time: {elapsed_time:.4f} seconds")
        return result
    return wrapper

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
    import matplotlib
    matplotlib.use('agg')
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
    plt.close(fig)

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
    
        mask = (gt_albedo_flat > 0.0).all(dim=1)

        gt_albedo_flat = gt_albedo_flat[mask]
        albedo_flat = albedo_flat[mask]

        gt_albedo_flat_list.append(gt_albedo_flat)
        albedo_flat_list.append(albedo_flat)

    gt_all = torch.cat(gt_albedo_flat_list, dim=0)
    albedo_all = torch.cat(albedo_flat_list, dim=0)

    single_channel_ratio = (gt_all / albedo_all.clamp(min=1e-6))[..., 0].median()
    three_channel_ratio, _ = (gt_all / albedo_all.clamp(min=1e-6)).median(dim=0)

    #if "air_baloons" in args.dataset_name:
        #three_channel_ratio = [(gt_all/albedo_all.clamp_min(1e-6))[..., 0].median().item()] * 3 # follow IRGS

    return mi.TensorXf(single_channel_ratio), mi.TensorXf(three_channel_ratio)

def load_hdr_paths(root_dir):
    hdr_paths = []
    hdr_paths += glob.glob(os.path.join(root_dir, "*.hdr"))
    hdr_paths += glob.glob(os.path.join(root_dir, "*.exr"))
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

def unpack_buffer(buffer):
    """Unpack the buffer into individual components"""
    rgb = buffer[:, :, :3]
    alpha = buffer[:, :, 3:4]
    albedo = buffer[:, :, 4:7]
    roughness = buffer[:, :, 7:8]
    metallic = buffer[:, :, 8:9]
    direct = buffer[:, :, 9:12]
    indirect = buffer[:, :, 12:15]
    normal = buffer[:, :, 15:18]
    depth = buffer[:, :, 18:19]

    aovs = {
        'alpha': alpha,
        'albedo': albedo,
        'roughness': roughness,
        'metallic': metallic,
        'direct_light': direct,
        'indirect_light': indirect,
        'normal': normal,
        'depth': depth,
    }

    return rgb, aovs

# ==================== Saving Functions ====================
def setup_output_directories():
    """Create all necessary output directories"""
    directories = [
        OUTPUT_ENVMAP_DIR,
        OUTPUT_RGB_DIR,
        OUTPUT_GBUFFER_DIR,
        OUTPUT_ALBEDO_DIR,
        OUTPUT_ROUGHNESS_DIR,
        OUTPUT_METALLIC_DIR,
        OUTPUT_DEPTH_DIR,
        OUTPUT_NORMAL_DIR,
        OUTPUT_DIRECT_LIGHT_DIR,
        OUTPUT_INDIRECT_LIGHT_DIR,
        OUTPUT_PLY_DIR,
    ]
    for directory in directories:
        ensure_dir(directory)

def save_render_results(i, img, aovs, ref_imgs, normal_mask, dataset):
    """Save rendering results for the current iteration"""
    
    target_res = [dataset.target_res[0], dataset.target_res[1] * dataset.batch_size]
    
    filename_base = f'opt-{i:04d}'
    
    # Prepare images for saving
    images_to_save = {
        (OUTPUT_RGB_DIR, filename_base): img,
        (OUTPUT_RGB_DIR, f'{filename_base}_ref'): ref_imgs['rgb'],
        (OUTPUT_ALBEDO_DIR, filename_base): aovs['albedo'],
        (OUTPUT_ROUGHNESS_DIR, filename_base): aovs['roughness'],
        (OUTPUT_METALLIC_DIR, filename_base): aovs['metallic'],
        (OUTPUT_DEPTH_DIR, filename_base): aovs['depth'] / dr.max(aovs['depth']),
        (OUTPUT_NORMAL_DIR, filename_base): mi.TensorXf(np.where(normal_mask, (aovs['normal'] + 1.0) / 2, 0)),
    }
    
    # Save RGB and gbuffer images
    for (output_dir, filename), image_data in images_to_save.items():
        bmp = resize_img(mi.Bitmap(image_data), target_res)
        write_bitmap(join(output_dir, f'{filename}.png'), bmp)
            
    # Save direct/indirect light if available
    if args.separate_direct_indirect and 'direct_light' in aovs and 'indirect_light' in aovs:
        direct_light_img = aovs['direct_light']
        indirect_light_img = aovs['indirect_light']

        direct_light_bmp = resize_img(mi.Bitmap(direct_light_img), target_res)
        indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img), target_res)
        
        write_bitmap(join(OUTPUT_DIRECT_LIGHT_DIR, f'{filename_base}.png'), direct_light_bmp)
        write_bitmap(join(OUTPUT_INDIRECT_LIGHT_DIR, f'{filename_base}.png'), indirect_light_bmp)

def gaussian_filter(tensor, sigma=1.0, radius=2):
    """Separable Gaussian blur in drjit (differentiable). tensor: (H, W, C)."""
    H, W, C = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (k / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    flat = dr.ravel(tensor)
    arr = flat.array if hasattr(flat, 'array') else flat

    def indices_shift_j(dj):
        i = np.arange(H, dtype=np.uint32)[:, None, None]
        j = np.arange(W, dtype=np.uint32)[None, :, None]
        cc = np.arange(C, dtype=np.uint32)[None, None, :]
        j_shift = np.mod(np.int64(j) + int(dj), W).astype(np.uint32)
        idx = (i * (W * C) + j_shift * C + cc).ravel()
        return mi.UInt32(idx)

    def indices_shift_i(di):
        i = np.arange(H, dtype=np.uint32)[:, None, None]
        j = np.arange(W, dtype=np.uint32)[None, :, None]
        cc = np.arange(C, dtype=np.uint32)[None, None, :]
        i_shift = np.mod(np.int64(i) + int(di), H).astype(np.uint32)
        idx = (i_shift * (W * C) + j * C + cc).ravel()
        return mi.UInt32(idx)

    out = dr.zeros_like(flat)
    for r, w in enumerate(kernel):
        idx = indices_shift_j(r - radius)
        out += float(w) * dr.gather(mi.Float, arr, idx)
    out_3d = mi.TensorXf(out, shape=(H, W, C))
    flat2 = dr.ravel(out_3d)
    arr2 = flat2.array if hasattr(flat2, 'array') else flat2
    out2 = dr.zeros_like(flat2)
    for r, w in enumerate(kernel):
        idx = indices_shift_i(r - radius)
        out2 += float(w) * dr.gather(mi.Float, arr2, idx)
    return mi.TensorXf(out2, shape=(H, W, C))