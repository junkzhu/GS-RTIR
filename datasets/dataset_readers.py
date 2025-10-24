import json
import os
import numpy as np
import drjit as dr
import mitsuba as mi
from pathlib import Path
from PIL import Image
from utils import resize_img

def read_nerf_synthetic(nerf_data_path, format, camera_indices=None, resx=800, resy=800, radius=2.0, scale_factor=1.0, split='train', env='sunset', normalize_distance=False, offset=np.array([0.0, 0.0, 0.0])):
    #-------------------------SENSORS------------------------
    sensors = []
    sampler = mi.load_dict({'type': 'independent'})
    film = mi.load_dict({
        'type': 'hdrfilm', 
        'width': resx, 
        'height': resy,
        'pixel_format': format, 
        'pixel_filter': {'type': 'gaussian'}, 
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
        img_path = Path(nerf_data_path) / (file_path + '_' + env + ".png")
        if not img_path.exists():
            img_path = Path(nerf_data_path) / (file_path + '_' + env + ".jpg")
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

    albedo_paths = [path.replace('rgba_sunset.png', 'albedo.png') for path in image_paths]
    ref_albedo_images=[]
    for idx, fn in enumerate(albedo_paths):
        bmp = load_bitmap(fn, False)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        ref_albedo_images.append(d)

    normal_paths = [path.replace('rgba_sunset.png', 'normal.png') for path in image_paths]
    ref_normal_images=[]
    for idx, fn in enumerate(normal_paths):
        bmp = load_bitmap(fn, False, normalize=True, mitsuba=False)
        bmp = mi.TensorXf(bmp)
        bmp = mi.Bitmap(bmp)

        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        ref_normal_images.append(d)

    roughness_paths = [path.replace('rgba_sunset.png', 'roughness.png') for path in image_paths]
    ref_roughness_images=[]
    for idx, fn in enumerate(roughness_paths):
        bmp = load_bitmap(fn, False)

        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        ref_roughness_images.append(d)

    albedo_priors_paths = [path.replace('rgba_sunset.png', 'albedo_sunset.png') for path in image_paths]
    albedo_priors_images=[]
    for idx, fn in enumerate(albedo_priors_paths):
        bmp = load_bitmap(fn, True)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        albedo_priors_images.append(d)

    roughness_priors_paths = [path.replace('rgba_sunset.png', 'roughness_sunset.png') for path in image_paths]
    roughness_priors_images=[]
    for idx, fn in enumerate(roughness_priors_paths):
        bmp = load_bitmap(fn, False)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = dr.clamp(mi.TensorXf(resize_img(bmp, new_res, smooth=False)), 0.0, 1.0)
        roughness_priors_images.append(d)

    normal_priors_paths = [path.replace('rgba_sunset.png', 'normal_sunset.png') for path in image_paths]
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

    return sensors, ref_images, ref_albedo_images, ref_normal_images, ref_roughness_images, albedo_priors_images, roughness_priors_images, normal_priors_images

sceneLoadTypeCallbacks = {"Blender": read_nerf_synthetic}