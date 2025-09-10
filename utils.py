import drjit as dr
import mitsuba as mi
import tqdm

import numpy as np

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

def convert_depth_to_normal(depth_map, sensor, fov_deg=39.0, pad=2, depth_thresh=0.01):

    if not isinstance(depth_map, dr.cuda.TensorXf):
        depth_map = dr.cuda.TensorXf(depth_map)
    
    H, W = depth_map.shape[:2]
    fov_rad = float(fov_deg * np.pi / 180.0)
    fx = fy = float(W / (2.0 * np.tan(fov_rad / 2.0)))
    cx, cy = float(W / 2.0), float(H / 2.0)
    
    windows = np.lib.stride_tricks.sliding_window_view(depth_map, (2*pad+1,2*pad+1), axis=(0, 1))
    mask_region = np.all(windows > depth_thresh, axis=(3, 4)) #H-4,W-4
    mask = np.zeros((H, W, 1), dtype=bool)
    mask[pad:H-pad, pad:W-pad] = mask_region

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = dr.cuda.TensorXf(xx)
    yy = dr.cuda.TensorXf(yy)    
    
    z = depth_map
    X = (xx[:,:,np.newaxis] - cx) * z / fx
    Y = (yy[:,:,np.newaxis] - cy) * z / fy
    P = dr.concat([X, Y, z], axis=2)

    P_left   = dr.concat([P[:,0:1,:],   P[:,:-1,:]], axis=1)
    P_right  = dr.concat([P[:,1:,:],    P[:,-1:,:]], axis=1)
    P_up     = dr.concat([P[0:1,:,:],   P[:-1,:,:]], axis=0)
    P_down   = dr.concat([P[1:,:,:],    P[-1:,:,:]], axis=0)

    depth = z

    depth_left  = dr.concat([depth[:,0:1],  depth[:,:-1]], axis=1)
    depth_right = dr.concat([depth[:,1:],   depth[:,-1:]], axis=1)
    depth_up    = dr.concat([depth[0:1,:],  depth[:-1,:]], axis=0)
    depth_down  = dr.concat([depth[1:,:],   depth[-1:,:]], axis=0)

    cond_x = dr.abs(depth_left - depth) < dr.abs(depth_right - depth)
    ddx = dr.select(cond_x, P - P_left, P_right - P)

    cond_y = dr.abs(depth_down - depth) < dr.abs(depth_up - depth)
    ddy = dr.select(cond_y, P - P_down, P_up - P)

    nx = ddx[...,1]*ddy[...,2] - ddx[...,2]*ddy[...,1]
    ny = ddx[...,2]*ddy[...,0] - ddx[...,0]*ddy[...,2]
    nz = ddx[...,0]*ddy[...,1] - ddx[...,1]*ddy[...,0]

    #normalize
    norm = dr.sqrt(nx*nx + ny*ny + nz*nz)
    norm = dr.maximum(norm, 1e-8)
    nx = -nx/norm
    ny = -ny/norm
    nz = nz/norm

    #rotation
    c2w = sensor.world_transform().matrix
    normal_x = nx * c2w[0, 0] + ny * c2w[0, 1] + nz * c2w[0, 2]
    normal_y = nx * c2w[1, 0] + ny * c2w[1, 1] + nz * c2w[1, 2]
    normal_z = nx * c2w[2, 0] + ny * c2w[2, 1] + nz * c2w[2, 2]

    normal = dr.concat([normal_x[:,:,None],
                        normal_y[:,:,None],
                        normal_z[:,:,None]], axis=2)

    normal = dr.select(mask, normal, dr.zeros_like(normal))

    return normal