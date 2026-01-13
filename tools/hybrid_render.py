import sys, os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tqdm
import numpy as np
import os
from os.path import join
from pathlib import Path

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from constants import *

from utils import *
from models import *
from integrators import *
from datasets import *
from losses import *

def make_sensor(
    cam_pos,
    look_at,
    up=(0, 0, 1),
    fov=45.0,
    resx=1600,
    resy=600
):
    film = mi.load_dict({
        'type': 'hdrfilm',
        'width': resx,
        'height': resy,
        'pixel_format': 'rgb',
        'pixel_filter': {'type': 'tent'},
        'sample_border': True
    })

    sensor = mi.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=cam_pos,
            target=look_at,
            up=up
        ),
        'film': film
    })

    return sensor

def set_scene_config(envmap_path):
    scene_config = {
        'type': 'scene',
        'integrator': {
            'type': 'hybrid_rt',
            'max_depth': 3,
            'pt_rate': 1.0,
            'gaussian_max_depth': 128,
            'hide_emitters': False,
            'use_mis': True,
            'selfocc_offset_max': 0.1,
            'geometry_threshold': 0.2,
            'separate_direct_indirect': True,
        },
        'rectangle': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                .scale([5, 8, 1])
                .translate([0, 0, 0]),
            'material': {
                'type': 'principled',
                'base_color': {
                    'type': 'rgb',
                    'value': [0.05, 0.05, 0.05]
                },
                'metallic': 1.0,
                'roughness': 0.05,
                'specular': 0.5
            }
        },
        # 'sphere': {
        #     'type': 'sphere',
        #     'center': [0, 1, 1],
        #     'radius': 0.3,
        #     'bsdf': {
        #         'type': 'diffuse'
        #     }
        # }
    }

    scene_config['emitter'] = {
        'type': 'envmap',
        'id': 'EnvironmentMapEmitter',
        'filename': envmap_path,
        'to_world': mi.ScalarTransform4f.rotate([0, 0, 1], 0) @
                    mi.ScalarTransform4f.rotate([1, 0, 0], 90)
    }

    # scene_config['point_light'] = {
    #     'type': 'point',
    #     'position': [0, 0, 3],
    #     'intensity': {
    #         'type': 'spectrum',
    #         'value': 10.0,
    #     }
    # }

    scene_config['sphere_area_light'] = {
        'type': 'sphere',
        'center': [0, 0, 1],
        'radius': 0.3,
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': 10.0,
            }
        }
    }
    
    return scene_config

def add_gaussian_config(scene_config, gaussians_attributes, name_prefix='gaussian'):
    # Check if we have multiple gaussian models
    is_multiple = isinstance(gaussians_attributes, list)
    
    if is_multiple:
        # Add each gaussian model with a unique name
        for i, attrs in enumerate(gaussians_attributes):
            gaussian_name = f'{name_prefix}{i+1}'
             
            # Add gaussian to scene configuration without to_world attribute
            scene_config[gaussian_name] = {
                'type': 'ellipsoidsmesh',
                'centers': attrs['centers'],
                'scales': attrs['scales'],
                'quaternions': attrs['quats'],
                'opacities': attrs['sigmats'],
                'sh_coeffs': attrs['features'],
                'normals': attrs['normals'],
                'albedos': attrs['albedos'],
                'roughnesses': attrs['roughnesses'],
                'metallics': attrs['metallics']
            }
    else:
        # Add gaussian to scene configuration without to_world attribute
        scene_config['gaussians'] = {
            'type': 'ellipsoidsmesh',
            'centers': gaussians_attributes['centers'],
            'scales': gaussians_attributes['scales'],
            'quaternions': gaussians_attributes['quats'],
            'opacities': gaussians_attributes['sigmats'],
            'sh_coeffs': gaussians_attributes['features'],
            'normals': gaussians_attributes['normals'],
            'albedos': gaussians_attributes['albedos'],
            'roughnesses': gaussians_attributes['roughnesses'],
            'metallics': gaussians_attributes['metallics']
        }
    
    return scene_config

def apply_transform_to_gaussian(attrs, transform):
    """
    Apply a transform to the gaussian attributes
    
    Args:
        attrs: Gaussian attributes dictionary
        transform: Transform matrix to apply
        
    Returns:
        Updated attributes dictionary with transformed centers, normals, and quaternions
    """
    # Create a copy of the attributes
    transformed_attrs = attrs.copy()
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Get transform matrix as numpy array
    if hasattr(transform, 'matrix'):
        # For Mitsuba's ScalarAffineTransform4f objects, get the matrix
        matrix = transform.matrix
        # Convert Matrix4f to numpy array
        transform_np = np.array([[matrix[i, j] for j in range(4)] for i in range(4)])
    elif hasattr(transform, 'numpy'):
        # For other transform types that have a numpy() method
        transform_np = transform.numpy()
    else:
        # Try to convert directly to numpy array
        transform_np = np.array(transform)
    
    # Extract the 3x3 rotation+scaling matrix
    rot_scale_matrix = transform_np[:3, :3]
    
    # Apply transform to centers
    centers = transformed_attrs['centers']
    if hasattr(centers, 'numpy'):
        centers_np = centers.numpy()
    else:
        centers_np = centers
    
    transformed_centers = []
    for center in centers_np:
        # Convert to homogeneous coordinates
        center_homogeneous = np.append(center, 1.0)
        # Apply transform using numpy matrix multiplication
        transformed_center_homogeneous = transform_np.dot(center_homogeneous)
        # Convert back to 3D coordinates
        transformed_center = transformed_center_homogeneous[:3]
        transformed_centers.append(transformed_center)
    
    # Convert transformed centers back to original type
    if hasattr(centers, 'numpy'):
        if hasattr(centers, 'shape') and centers.shape[-1] == 3:
            transformed_centers_np = np.array(transformed_centers)
            transformed_attrs['centers'] = centers.__class__(transformed_centers_np)
        else:
            transformed_attrs['centers'] = np.array(transformed_centers)
    else:
        transformed_attrs['centers'] = np.array(transformed_centers)
    
    # Apply transform to normals if they exist
    if 'normals' in transformed_attrs:
        normals = transformed_attrs['normals']
        if hasattr(normals, 'numpy'):
            normals_np = normals.numpy()
        else:
            normals_np = normals
        
        # For normals, we only want to apply the rotation part (not scaling)
        # Calculate the inverse transpose of the rotation matrix
        rot_matrix = rot_scale_matrix
        # Normalize rotation matrix to handle scaling
        scale_factor = np.linalg.norm(rot_matrix, axis=0)
        rot_matrix = rot_matrix / scale_factor[np.newaxis, :]
        
        # Apply rotation and normalize (avoid division by zero)
        transformed_normals = []
        for normal in normals_np:
            transformed = rot_matrix.dot(normal)
            norm = np.linalg.norm(transformed)
            transformed_normals.append(transformed / norm if norm > 1e-8 else transformed)
        
        # Convert back to original type
        transformed_np = np.array(transformed_normals)
        if hasattr(normals, 'numpy'):
            if hasattr(normals, 'shape') and normals.shape[-1] == 3:
                transformed_attrs['normals'] = normals.__class__(transformed_np)
            else:
                transformed_attrs['normals'] = transformed_np
        else:
            transformed_attrs['normals'] = transformed_np
    
    # Apply transform to quaternions if they exist
    if 'quats' in transformed_attrs:
        quats = transformed_attrs['quats']
        if hasattr(quats, 'numpy'):
            quats_np = quats.numpy()
        else:
            quats_np = quats
        
        # Convert the rotation matrix to a quaternion
        rot_matrix = rot_scale_matrix
        # Remove scaling effect
        scale_factor = np.linalg.norm(rot_matrix, axis=0)
        rot_matrix = rot_matrix / scale_factor[np.newaxis, :]
        
        # Ensure it's a proper rotation matrix (orthogonal)
        if not np.isclose(np.linalg.det(rot_matrix), 1.0):
            # If determinant is not 1, it's not a pure rotation
            # We can use polar decomposition to extract the rotation part
            from scipy.linalg import polar
            rot_matrix, _ = polar(rot_matrix)
        
        transform_quat = R.from_matrix(rot_matrix).as_quat()
        # Convert to [w, x, y, z] format if needed
        transform_quat = np.roll(transform_quat, 1)  # Convert from [x,y,z,w] to [w,x,y,z]
        
        transformed_quats = []
        for quat in quats_np:
            # Convert quaternion from [w, x, y, z] to [x, y, z, w] for scipy
            quat_scipy = np.roll(quat, -1)
            # Apply the transform rotation to the quaternion
            combined_rot = R.from_quat(quat_scipy) * R.from_quat(np.roll(transform_quat, -1))
            combined_quat = combined_rot.as_quat()
            # Convert back to [w, x, y, z] format
            combined_quat = np.roll(combined_quat, 1)
            transformed_quats.append(combined_quat)
        
        # Convert transformed quats back to original type
        if hasattr(quats, 'numpy'):
            if hasattr(quats, 'shape') and quats.shape[-1] == 4:
                transformed_quats_np = np.array(transformed_quats)
                transformed_attrs['quats'] = quats.__class__(transformed_quats_np)
            else:
                transformed_attrs['quats'] = np.array(transformed_quats)
        else:
            transformed_attrs['quats'] = np.array(transformed_quats)
    
    # Apply transform to scales if they exist
    if 'scales' in transformed_attrs:
        scales = transformed_attrs['scales']
        if hasattr(scales, 'numpy'):
            scales_np = scales.numpy()
        else:
            scales_np = scales
        
        # Apply scaling from the transform matrix
        scale_factor = np.linalg.norm(rot_scale_matrix, axis=0)
        
        transformed_scales = []
        for scale in scales_np:
            # Apply the scaling factors
            transformed_scale = scale * scale_factor
            transformed_scales.append(transformed_scale)
        
        # Convert transformed scales back to original type
        if hasattr(scales, 'numpy'):
            if hasattr(scales, 'shape') and scales.shape[-1] == 3:
                transformed_scales_np = np.array(transformed_scales)
                transformed_attrs['scales'] = scales.__class__(transformed_scales_np)
            else:
                transformed_attrs['scales'] = np.array(transformed_scales)
        else:
            transformed_attrs['scales'] = np.array(transformed_scales)
    
    return transformed_attrs

def render_dataset_scene(dataset, scene_dict, render_spp):    
    # Ensure render directory exists
    ensure_dir(OUTPUT_HYBRID_DIR)

    pbar = tqdm.tqdm(enumerate(dataset.sensors), total=len(dataset.sensors), desc="Rendering the scene")

    for idx, sensor in pbar:
        img, aovs = mi.render(scene_dict, sensor=sensor, spp=render_spp) #img is linear space #TODO

        #aovs
        albedo_img = aovs['albedo'][:, :, :3]
        roughness_img = aovs['roughness'][:, :, :1]
        #metallic_img = aovs['metallic'][:, :, :1]
        normal_img = aovs['normal'][:, :, :3]

        normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)

        albedo_bmp = resize_img(mi.Bitmap(albedo_img), dataset.target_res)
        roughness_bmp = resize_img(mi.Bitmap(roughness_img), dataset.target_res)
        #metallic_bmp = resize_img(mi.Bitmap(metallic_img), dataset.target_res)
        normal_bmp = resize_img(mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))), dataset.target_res) 

        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_albedo' + ('.png')), albedo_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_roughness' + ('.png')), roughness_bmp)
        #write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_metallic' + ('.png')), metallic_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_normal' + ('.png')), normal_bmp)

        if args.separate_direct_indirect:
            direct_light_img = aovs['direct_light'][:, :, :3]
            indirect_light_img = aovs['indirect_light'][:, :, :3]

            direct_light_bmp = resize_img(mi.Bitmap(direct_light_img),dataset.target_res)
            indirect_light_bmp = resize_img(mi.Bitmap(indirect_light_img),dataset.target_res)

            write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_direct_light' + ('.png')), direct_light_bmp)
            write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_indirect_light' + ('.png')), indirect_light_bmp)

        rgb_bmp = resize_img(mi.Bitmap(img),dataset.target_res)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}' + ('.png')), rgb_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'{idx:02d}_ref' + ('.png')), dataset.ref_images[idx][sensor.film().crop_size()[0]])

def render_define_scene(scene_dict, render_spp):
    # Ensure render directory exists
    ensure_dir(OUTPUT_HYBRID_DIR)
    resx = 1600
    resy = 600

    #---- Define the Sensor ----#
    sensor = make_sensor(
        cam_pos=(-40, 0.0, 20.0),
        look_at=(0.0, 0.0, 0.0),
        fov=22,
        resx=resx, 
        resy=resy
    )

    img, aovs = mi.render(scene_dict, sensor=sensor, spp=render_spp) #img is linear space #TODO

    #aovs
    albedo_img = aovs['albedo'][:, :, :3]
    roughness_img = aovs['roughness'][:, :, :1]
    metallic_img = aovs['metallic'][:, :, :1]
    normal_img = aovs['normal'][:, :, :3]

    normal_mask = np.any(normal_img != 0, axis=2, keepdims=True)

    albedo_bmp = mi.Bitmap(albedo_img)
    roughness_bmp = mi.Bitmap(roughness_img)
    metallic_bmp = mi.Bitmap(metallic_img)
    normal_bmp = mi.Bitmap(mi.TensorXf(np.where(normal_mask, (normal_img+1)/2, 0))) 

    write_bitmap(join(OUTPUT_HYBRID_DIR, f'albedo' + ('.png')), albedo_bmp)
    write_bitmap(join(OUTPUT_HYBRID_DIR, f'roughness' + ('.png')), roughness_bmp)
    write_bitmap(join(OUTPUT_HYBRID_DIR, f'metallic' + ('.png')), metallic_bmp)
    write_bitmap(join(OUTPUT_HYBRID_DIR, f'normal' + ('.png')), normal_bmp)

    if args.separate_direct_indirect:
        direct_light_img = aovs['direct_light'][:, :, :3]
        indirect_light_img = aovs['indirect_light'][:, :, :3]

        direct_light_bmp = mi.Bitmap(direct_light_img)
        indirect_light_bmp = mi.Bitmap(indirect_light_img)

        write_bitmap(join(OUTPUT_HYBRID_DIR, f'direct_light' + ('.png')), direct_light_bmp)
        write_bitmap(join(OUTPUT_HYBRID_DIR, f'indirect_light' + ('.png')), indirect_light_bmp)

    rgb_bmp = mi.Bitmap(img)
    write_bitmap(join(OUTPUT_HYBRID_DIR, f'rgb' + ('.png')), rgb_bmp)

if __name__ == "__main__":
    render_spp = 32
    dataset_path = '/home/zjk/datasets/TensoIR/lego'
    envmap_path = '/home/zjk/datasets/bloem_train_track_cloudy_4k.exr'
    
    # Render switch: True to use dataset sensors, False to use predefined sensor
    use_dataset_render = False
    
    # Model paths
    ply_paths = [
        '/home/zjk/code/GS-RTIR/outputs/TensoIR#0/lego/ply/iter_799_rescaled.ply',
        '/home/zjk/code/GS-RTIR/outputs/TensoIR#0/armadillo/ply/iter_799_rescaled.ply',
        '/home/zjk/code/GS-RTIR/outputs/TensoIR#0/ficus/ply/iter_799_rescaled.ply',
        '/home/zjk/code/GS-RTIR/outputs/TensoIR#0/hotdog/ply/iter_799_rescaled.ply',

        '/home/zjk/code/GS-RTIR/outputs/Synthetic4Relight#0/air_baloons/ply/iter_799_rescaled.ply',
        '/home/zjk/code/GS-RTIR/outputs/Synthetic4Relight#0/chair/ply/iter_799_rescaled.ply',
        '/home/zjk/code/GS-RTIR/outputs/Synthetic4Relight#0/jugs/ply/iter_799_rescaled.ply',
    ]
    
    rotations = [
        [0, 0, -30], [0, 0, 90], [0, 0, 180], [0, 0, 140], #TensoIR
        
        [0, 0, 150], [0, 0, -60], [0, 0, -90], #Synthetic4Relight
    ]
    scales = [
        None, None, [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], #TensoIR
        
        [0.5, 0.5, 0.5], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], #Synthetic4Relight
    ]
    translations = [
        [2.0, -6.0, 0.35], [2.0, -3.5, 0.63], [2.0, -2, 0.7], [-0.1, -4.8, 0.12], #TensoIR
        
        [3.0, 6.0, 1.0], [2.0, 3.0, 0.4], [2.0, 0, 0.05], #Synthetic4Relight
    ]
    
    # Initialize factory
    ellipsoidsfactory = EllipsoidsFactory()
    
    # Load all Gaussian models
    all_gaussians_attributes = []
    with time_measure("Loading multiple gaussians"):
        for idx, ply_path in enumerate(ply_paths):
            gaussians = GaussianModel()
            gaussians.restore_from_ply(ply_path, False)
            
            # Apply model-specific transformations
            # 1. Apply translation if specified
            if translations[idx] is not None:
                gaussians.translate(torch.tensor(translations[idx], dtype=torch.float32))
            
            # 2. Apply scale if specified
            if scales[idx] is not None:
                gaussians.scale(torch.tensor(scales[idx], dtype=torch.float32))
            
            # 3. Apply rotation if specified
            if rotations[idx] is not None:
                # Convert Euler angles (in degrees) to rotation matrix
                rx, ry, rz = rotations[idx]
                rx_rad = torch.deg2rad(torch.tensor(rx, dtype=torch.float32))
                ry_rad = torch.deg2rad(torch.tensor(ry, dtype=torch.float32))
                rz_rad = torch.deg2rad(torch.tensor(rz, dtype=torch.float32))
                
                # Create rotation matrices around each axis
                R_x = torch.tensor([
                    [1, 0, 0],
                    [0, torch.cos(rx_rad), -torch.sin(rx_rad)],
                    [0, torch.sin(rx_rad), torch.cos(rx_rad)]
                ], dtype=torch.float32)
                
                R_y = torch.tensor([
                    [torch.cos(ry_rad), 0, torch.sin(ry_rad)],
                    [0, 1, 0],
                    [-torch.sin(ry_rad), 0, torch.cos(ry_rad)]
                ], dtype=torch.float32)
                
                R_z = torch.tensor([
                    [torch.cos(rz_rad), -torch.sin(rz_rad), 0],
                    [torch.sin(rz_rad), torch.cos(rz_rad), 0],
                    [0, 0, 1]
                ], dtype=torch.float32)
                
                # Combine rotations: R_z * R_y * R_x
                rotmat = R_z @ R_y @ R_x
                gaussians.rotate(rotmat)

            gaussians_attrs = ellipsoidsfactory.load_gaussian(gaussians=gaussians)
            all_gaussians_attributes.append(gaussians_attrs)
        
        # Create scene config
        scene_config = set_scene_config(envmap_path)
        # Add gaussians to scene
        scene_config = add_gaussian_config(scene_config, all_gaussians_attributes)
        
        # Load scene
        scene_dict = mi.load_dict(scene_config)
        params = mi.traverse(scene_dict)

    # Render according to switch
    if use_dataset_render:
        # Load dataset if needed
        from datasets import Dataset
        dataset = Dataset(dataset_path)
        render_dataset_scene(dataset, scene_dict, render_spp)
    else:
        render_define_scene(scene_dict, render_spp)