import json
import os
import numpy as np
import drjit as dr
import mitsuba as mi
from pathlib import Path
from PIL import Image
from utils import resize_img, srgb_to_linear
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from constants import args

mipmap_min_res = 100

def load_bitmap(fn, bsrgb2linear = True, normalize = False, mitsuba_axis = False):
    """Load bitmap as float32 and apply gamma correction"""
    img = np.array(Image.open(fn)).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Apply inverse gamma correction (sRGB → linear)
    if bsrgb2linear:
        img = srgb_to_linear(img)
    
    if normalize:
        alpha = img[..., 3:]
        rgb = img[..., :3]

        rgb = rgb * 2 -1

        if mitsuba_axis:
            if "genprior" in fn: # blender -> mitsuba
                rgb[..., 2] = -rgb[..., 2]
            if "rgb2x" in fn: # colmap -> mitsuba                 
                rgb[..., 0] = -rgb[..., 0]
                rgb[..., 2] = -rgb[..., 2]
        
        norm = np.linalg.norm(rgb, axis=-1, keepdims=True)
        img = rgb / np.maximum(norm, 1e-8) * alpha

    if img.shape[-1] == 4:  # Handle alpha channel
        alpha = img[..., 3:]
        rgb = img[..., :3]
        img = rgb * alpha  # Premultiply alpha (if needed)
        
    return mi.Bitmap(img)

def load_mipmaps(fn, bsrgb2linear = True, normalize = False, mitsuba_axis = False):
    bmp = load_bitmap(fn, bsrgb2linear, normalize, mitsuba_axis)
    d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
    new_res = bmp.size()
    while np.min(new_res) > mipmap_min_res:
        new_res = new_res // 2
        d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
    return d

def load_normal_prior_mipmaps(args, sensors, bsrgb2linear = True, normalize = False, mitsuba_axis = False):
    idx, fn = args
    
    bmp = load_bitmap(fn, bsrgb2linear, normalize, mitsuba_axis)
    bmp = np.array(bmp, dtype=np.float32)

    sensor = sensors[idx]
    R_c2w = np.array(sensor.world_transform().matrix)[:3, :3].astype(np.float32).squeeze()
    
    H, W, _ = bmp.shape
    bmp = bmp.reshape(-1, 3).T
    bmp = R_c2w @ bmp
    bmp = bmp.T.reshape(H, W, 3)
    bmp /= np.maximum(np.linalg.norm(bmp, axis=-1, keepdims=True), 1e-8)

    bmp = mi.Bitmap(bmp)

    d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
    new_res = bmp.size()
    while np.min(new_res) > mipmap_min_res:
        new_res = new_res // 2
        d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
    
    return d

def read_nerf_synthetic(nerf_data_path, format, camera_indices=None, resx=800, resy=800, radius=2.0, scale_factor=1.0, split='train', env='sunset', filter_type="tent", normalize_distance=False, offset=np.array([0.0, 0.0, 0.0]), relight_envmap_names=None, load_ref_relight_images=False):
    #-------------------------SENSORS------------------------
    sensors = []
    sensors_normal = []
    sensors_intrinsic = []
    sampler = mi.load_dict({'type': 'independent'})
    film = mi.load_dict({
        'type': 'hdrfilm', 
        'width': resx, 
        'height': resy,
        'pixel_format': format, 
        'pixel_filter': {'type': filter_type}, 
        'sample_border': True
    })
    
    transforms_file = os.path.join(nerf_data_path, f'transforms_{split}.json')
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
    fov = np.degrees(camera_angle_x)
    
    frames = transforms_data.get('frames', [])
    n_cameras = len(frames)
    
    if camera_indices is None:
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
        
        c2w[:3, 3] *= scale_factor
        c2w[:3, 3] += offset

        #https://euruson.com/post/225c1af2-1435-804f-a7c8-d3a78006d48c
        coord_transform = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        c2w = c2w @ coord_transform
        
        if normalize_distance:
            camera_pos = c2w[:3, 3]
            relative_pos = camera_pos - scene_center
            current_distance = np.linalg.norm(relative_pos)

            if current_distance > 1e-8:
                normalized_relative_pos = relative_pos / current_distance * radius
                c2w[:3, 3] = scene_center + normalized_relative_pos
        
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
        
        sensor_normal = -c2w[:3, 2]
        sensors_normal.append(sensor_normal)

        fx = 0.5 * resx / np.tan(0.5 * camera_angle_x)
        fy = 0.5 * resy / np.tan(0.5 * camera_angle_x)
        cx = resx / 2.0
        cy = resy / 2.0
        intrinsic = np.array([fx, fy, cx, cy], dtype=np.float32)
        sensors_intrinsic.append(intrinsic)

    #-------------------------IMAGES------------------------
    image_paths=[]
    for frame in frames:
        file_path = frame["file_path"]
        img_path = Path(nerf_data_path) / (file_path + '_' + env + ".png")
        if not img_path.exists():
            img_path = Path(nerf_data_path) / (file_path + '_' + env + ".jpg")
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image_paths.append(str(img_path.resolve()))
    
    ref_images=[]
    ref_albedo_images=[]
    ref_normal_images=[]
    ref_roughness_images=[]

    albedo_paths = [path.replace('rgba_sunset.png', 'albedo.png') for path in image_paths]
    normal_paths = [path.replace('rgba_sunset.png', 'normal.png') for path in image_paths]
    roughness_paths = [path.replace('rgba_sunset.png', 'roughness.png') for path in image_paths]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        ref_images = list(
            executor.map(
                lambda fn: load_mipmaps(fn, True),
                image_paths
            )
        )

        ref_albedo_images = list(
            executor.map(
                lambda fn: load_mipmaps(fn, False),
                albedo_paths
            )
        )

        ref_normal_images = list(
            executor.map(
                lambda fn: load_mipmaps(fn, False, normalize=True, mitsuba_axis=False),
                normal_paths
            )
        )

        ref_roughness_images = list(
            executor.map(
                lambda fn: load_mipmaps(fn, False),
                roughness_paths
            )
        )
    
    # relight images
    if load_ref_relight_images:
        ref_relight_images=defaultdict(list)
        for envmap_name in relight_envmap_names:
            relight_paths = [path.replace('rgba_sunset.png', f'rgba_{envmap_name}.png') for path in image_paths] 
            for idx, fn in enumerate(relight_paths):
                bmp = load_bitmap(fn)
                d = mi.TensorXf(bmp)
                ref_relight_images[envmap_name].append(d)

    if split != 'train' and load_ref_relight_images:
        return sensors, sensors_normal, sensors_intrinsic, ref_images, ref_albedo_images, ref_normal_images, ref_roughness_images, None, None, None, ref_relight_images
    elif split != 'train':
        return sensors, sensors_normal, sensors_intrinsic, ref_images, ref_albedo_images, ref_normal_images, ref_roughness_images, None, None, None, None

    albedo_priors_images=[]
    roughness_priors_images=[]
    normal_priors_images=[]

    if args.diffusion_model == "rgb2x":
        albedo_priors_paths = [path.replace('rgba_sunset.png', 'albedo_sunset.png') for path in image_paths]
        roughness_priors_paths = [path.replace('rgba_sunset.png', 'roughness_sunset.png') for path in image_paths]
        normal_priors_paths = [path.replace('rgba_sunset.png', 'normal_sunset.png') for path in image_paths]
    if args.diffusion_model == "genprior":
        albedo_priors_paths = [path.replace('rgba_sunset.png', 'albedo_sunset_genprior.png') for path in image_paths]
        roughness_priors_paths = [path.replace('rgba_sunset.png', 'roughness_sunset_genprior.png') for path in image_paths]
        normal_priors_paths = [path.replace('rgba_sunset.png', 'normal_sunset_genprior.png') for path in image_paths]


    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        albedo_priors_images = list(
            executor.map(
                lambda fn: load_mipmaps(fn, False),
                albedo_priors_paths
            )
        )

        roughness_priors_images = list(
            executor.map(
                lambda fn: load_mipmaps(fn, False),
                roughness_priors_paths
            )
        )

        normal_priors_images = list(
            executor.map(
                lambda args: load_normal_prior_mipmaps(args, sensors, False, normalize=True, mitsuba_axis=True),
                enumerate(normal_priors_paths)
            )
        )

    return sensors, sensors_normal, sensors_intrinsic, ref_images, ref_albedo_images, ref_normal_images, ref_roughness_images, albedo_priors_images, roughness_priors_images, normal_priors_images, None

def read_Synthetic4Relight(nerf_data_path, format, camera_indices=None, resx=800, resy=800, radius=2.0, scale_factor=1.0, split='train', env='sunset', filter_type="tent", normalize_distance=False, offset=np.array([0.0, 0.0, 0.0]), relight_envmap_names=None, load_ref_relight_images=False):
    #-------------------------SENSORS------------------------
    sensors = []
    sensors_normal = []
    sensors_intrinsic = []
    sampler = mi.load_dict({'type': 'independent'})
    film = mi.load_dict({
        'type': 'hdrfilm', 
        'width': resx, 
        'height': resy,
        'pixel_format': format, 
        'pixel_filter': {'type': filter_type}, 
        'sample_border': True
    })
    
    transforms_file = os.path.join(nerf_data_path, f'transforms_{split}.json')
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
    
    fov = np.degrees(camera_angle_x)
    
    frames = transforms_data.get('frames', [])
    n_cameras = len(frames)
    
    if camera_indices is None:
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
        
        c2w[:3, 3] *= scale_factor
        c2w[:3, 3] += offset

        #https://euruson.com/post/225c1af2-1435-804f-a7c8-d3a78006d48c
        coord_transform = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        c2w = c2w @ coord_transform
        
        if normalize_distance:
            camera_pos = c2w[:3, 3]
            relative_pos = camera_pos - scene_center
            current_distance = np.linalg.norm(relative_pos)

            if current_distance > 1e-8:
                normalized_relative_pos = relative_pos / current_distance * radius
                c2w[:3, 3] = scene_center + normalized_relative_pos
        
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
    
        sensor_normal = -c2w[:3, 2]
        sensors_normal.append(sensor_normal)

        fx = 0.5 * resx / np.tan(0.5 * camera_angle_x)
        fy = 0.5 * resy / np.tan(0.5 * camera_angle_x)
        cx = resx / 2.0
        cy = resy / 2.0
        intrinsic = np.array([fx, fy, cx, cy], dtype=np.float32)
        sensors_intrinsic.append(intrinsic)

    #-------------------------IMAGES------------------------
    image_paths=[]
    for frame in frames:
        file_path = frame["file_path"]
        
        img_path = None

        if split == 'train':
            img_path = Path(nerf_data_path) / (file_path + ".png")
        else:
            img_path = Path(nerf_data_path) / (file_path + "_rgba.png")
        
        if not img_path.exists():
            img_path = Path(nerf_data_path) / (file_path + ".jpg")
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image_paths.append(str(img_path.resolve()))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        ref_images = list(executor.map(lambda fn: load_mipmaps(fn, True), image_paths))

    ref_albedo_images=[]
    ref_roughness_images=[]

    if split == 'test':
        albedo_paths = [path.replace('_rgba.png', '_albedo.png') for path in image_paths]  
        roughness_paths = [path.replace('_rgba.png', '_rough.png') for path in image_paths]

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            ref_albedo_images = list(executor.map(lambda fn: load_mipmaps(fn, True), albedo_paths))
            ref_roughness_images = list(executor.map(lambda fn: load_mipmaps(fn, False), roughness_paths))

    # relight images
    if load_ref_relight_images:
        ref_relight_images = defaultdict(list)

        for envmap_name in relight_envmap_names:
            relight_paths = []

            for path in image_paths:
                new_dir = os.path.dirname(path).replace(
                    "test", f"test_rli"
                )

                fname = os.path.basename(path)
                idx = fname.split("_")[0]
                new_fname = f"{envmap_name}_{idx}.png"

                new_path = os.path.join(new_dir, new_fname)
                relight_paths.append(new_path)

            for fn in relight_paths:
                bmp = load_bitmap(fn)
                d = mi.TensorXf(bmp)
                ref_relight_images[envmap_name].append(d)

    if split != 'train' and load_ref_relight_images:
        return sensors, sensors_normal, sensors_intrinsic, ref_images, ref_albedo_images, None, ref_roughness_images, None, None, None, ref_relight_images
    elif split != 'train':
        return sensors, sensors_normal, sensors_intrinsic, ref_images, ref_albedo_images, None, ref_roughness_images, None, None, None, None

    albedo_priors_images=[]
    roughness_priors_images=[]
    normal_priors_images=[]

    if args.diffusion_model == "rgb2x":
        albedo_priors_paths = [path.replace('.png', '_albedo_rgb2x.png') for path in image_paths]
        roughness_priors_paths = [path.replace('.png', '_roughness_rgb2x.png') for path in image_paths]
        normal_priors_paths = [path.replace('.png', '_normal_rgb2x.png') for path in image_paths]
    if args.diffusion_model == "genprior":
        albedo_priors_paths = [path.replace('.png', '_albedo_genprior.png') for path in image_paths]
        roughness_priors_paths = [path.replace('.png', '_roughness_genprior.png') for path in image_paths]
        normal_priors_paths = [path.replace('.png', '_normal_genprior.png') for path in image_paths]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        albedo_priors_images = list(executor.map(lambda fn: load_mipmaps(fn, True), albedo_priors_paths))
        roughness_priors_images = list(executor.map(lambda fn: load_mipmaps(fn, False), roughness_priors_paths))
        normal_priors_images = list(executor.map(lambda args: load_normal_prior_mipmaps(args, sensors, False, normalize=True, mitsuba_axis=True), enumerate(normal_priors_paths)))

    return sensors, sensors_normal, sensors_intrinsic, ref_images, ref_albedo_images, None, ref_roughness_images, albedo_priors_images, roughness_priors_images, normal_priors_images, None

def read_Stanford_orb(nerf_data_path, format, camera_indices=None, resx=800, resy=800, radius=2.0, scale_factor=1.0, split='train', env='sunset', filter_type="tent", normalize_distance=False, offset=np.array([0.0, 0.0, 0.0])):
    #-------------------------SENSORS------------------------
    sensors = []
    sampler = mi.load_dict({'type': 'independent'})
    
    film = mi.load_dict({
        'type': 'hdrfilm', 
        'width': resx, 
        'height': resy,
        'pixel_format': format, 
        'pixel_filter': {'type': filter_type}, 
        'sample_border': True
    })
    
    transforms_file = os.path.join(nerf_data_path, f'transforms_{split}.json')
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
    fov = np.degrees(camera_angle_x)
    
    frames = transforms_data.get('frames', [])
    n_cameras = len(frames)
    
    if camera_indices is None:
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
        
        c2w[:3, 3] *= scale_factor
        c2w[:3, 3] += offset

        #https://euruson.com/post/225c1af2-1435-804f-a7c8-d3a78006d48c
        coord_transform = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        c2w = c2w @ coord_transform
        
        if normalize_distance:
            camera_pos = c2w[:3, 3]
            relative_pos = camera_pos - scene_center
            current_distance = np.linalg.norm(relative_pos)

            if current_distance > 1e-8:
                normalized_relative_pos = relative_pos / current_distance * radius
                c2w[:3, 3] = scene_center + normalized_relative_pos
        
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
    
    #-------------------------IMAGES------------------------
    image_paths=[]
    for frame in frames:
        file_path = frame["file_path"]
        img_path = Path(nerf_data_path) / (file_path + ".png")
        if not img_path.exists():
            img_path = Path(nerf_data_path) / (file_path + ".jpg")
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image_paths.append(str(img_path.resolve()))

    def srgb_to_linear(img):
        """Convert sRGB to linear RGB (gamma correction)"""
        return img ** 2.2
    
    def load_bitmap(fn, bsrgb2linear = True, normalize = False, mitsuba = False):
        """Load bitmap as float32 and apply gamma correction"""
        img = np.array(Image.open(fn)).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Apply inverse gamma correction (sRGB → linear)
        if bsrgb2linear:
            img = srgb_to_linear(img)
        
        if normalize:
            alpha = img[..., 3:]
            rgb = img[..., :3]

            rgb = rgb * 2 -1

            if mitsuba:
                rgb[..., 0] = -rgb[..., 0]
                rgb[..., 2] = -rgb[..., 2]
            
            norm = np.linalg.norm(rgb, axis=-1, keepdims=True)
            img = rgb / np.maximum(norm, 1e-8) * alpha

        if img.shape[-1] == 4:  # Handle alpha channel
            alpha = img[..., 3:]
            rgb = img[..., :3]
            img = rgb * alpha  # Premultiply alpha (if needed)
            
        return mi.Bitmap(img)

    ref_images=[]
    for idx, fn in enumerate(image_paths):
        bmp = load_bitmap(fn)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        ref_images.append(d)

    ref_albedo_images=[]
    ref_roughness_images=[]

    if split == 'test':
        albedo_paths = [path.replace('.png', '_albedo.png') for path in image_paths]  
        for idx, fn in enumerate(albedo_paths):
            bmp = load_bitmap(fn, False)
            d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
            new_res = bmp.size()
            while np.min(new_res) > 4:
                new_res = new_res // 2
                d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
            ref_albedo_images.append(d)
 
        roughness_paths = [path.replace('.png', '_rough.png') for path in image_paths]
        for idx, fn in enumerate(roughness_paths):
            bmp = load_bitmap(fn, False)

            d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
            new_res = bmp.size()
            while np.min(new_res) > 4:
                new_res = new_res // 2
                d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
            ref_roughness_images.append(d)

    if split != 'train':
        return sensors, ref_images, ref_albedo_images, None, ref_roughness_images, None, None, None

    albedo_priors_paths = [path.replace('.png', '_albedo_rgb2x.png') for path in image_paths]
    albedo_priors_images=[]
    for idx, fn in enumerate(albedo_priors_paths):
        bmp = load_bitmap(fn, True)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        albedo_priors_images.append(d)

    roughness_priors_paths = [path.replace('.png', '_roughness_rgb2x.png') for path in image_paths]
    roughness_priors_images=[]
    for idx, fn in enumerate(roughness_priors_paths):
        bmp = load_bitmap(fn, False)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        roughness_priors_images.append(d)

    normal_priors_paths = [path.replace('.png', '_normal_rgb2x.png') for path in image_paths]
    normal_priors_images=[]
    for idx, fn in enumerate(normal_priors_paths):
        bmp = load_bitmap(fn, False, normalize=True, mitsuba=True)
        bmp = np.array(bmp, dtype=np.float32)

        sensor = sensors[idx]
        R_c2w = np.array(sensor.world_transform().matrix)[:3, :3].astype(np.float32).squeeze()
        
        H, W, _ = bmp.shape
        bmp = bmp.reshape(-1, 3).T
        bmp = R_c2w @ bmp
        bmp = bmp.T.reshape(H, W, 3)
        bmp /= np.maximum(np.linalg.norm(bmp, axis=-1, keepdims=True), 1e-8)

        bmp = mi.Bitmap(bmp)

        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        normal_priors_images.append(d)

    return sensors, ref_images, ref_albedo_images, None, ref_roughness_images, albedo_priors_images, roughness_priors_images, normal_priors_images


sceneLoadTypeCallbacks = {"TensoIR": read_nerf_synthetic,
                          "Synthetic4Relight": read_Synthetic4Relight,
                          "Stanford_orb": read_Stanford_orb}