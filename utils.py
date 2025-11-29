import drjit as dr
import mitsuba as mi
import tqdm
from jinja2 import Template
import numpy as np
import torch
from emitter.sgenvmap_util import SG2Envmap

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
    lgtSGs_torch = torch.tensor(lgtSGs, dtype=torch.float32)
    envmap = SG2Envmap(lgtSGs_torch, 256, 512).detach().cpu().numpy()
    return mi.Bitmap(envmap)

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

    return mi.TensorXf(single_channel_ratio), mi.TensorXf(three_channel_ratio)