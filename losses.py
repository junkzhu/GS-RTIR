import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils import linear_to_srgb

def l1(reference, image, convert_to_srgb=False):
    '''
    L1 loss function
    '''
    if convert_to_srgb:
        reference = dr.linear_to_srgb(reference)
        image = dr.linear_to_srgb(image)

    return dr.mean(dr.abs(reference - image), axis=None)

def l2(reference, image):
    '''
    L2 loss function
    '''
    return dr.mean(dr.power(reference - image, 2), axis=None)

def l2_np(reference, image):
    '''
    L2 loss function
    '''
    return np.mean(np.power(reference - image, 2), axis=None)

def lpsnr(reference, image, convert_to_srgb=False):
    '''
    PSNR loss function
    '''
    reference = np.asarray(reference, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)

    reference = np.clip(reference, 0.0, None)
    image = np.clip(image, 0.0, None)

    if convert_to_srgb:
        reference = linear_to_srgb(reference)
        image = linear_to_srgb(image)

    mse = np.mean((reference - image)**2)
    return -10.0 * np.log(mse) / np.log(10.0)

def TV(reference, image):
    rgb_grad_h = dr.exp(-dr.mean(dr.abs(reference[1:, :, :] - reference[:-1, :, :]), axis=2))
    rgb_grad_w = dr.exp(-dr.mean(dr.abs(reference[:, 1:, :] - reference[:, :-1, :]), axis=2))
    tv_h = dr.power((image[1:, :, :] - image[:-1, :, :]), 2)
    tv_w = dr.power((image[:, 1:, :] - image[:, :-1, :]), 2)
    tv_loss = dr.mean(tv_h * rgb_grad_h[:,:,dr.newaxis]) + dr.mean(tv_w * rgb_grad_w[:,:,dr.newaxis])
    return tv_loss

def lnormal(reference, image, mask_flat):
    img_flat = dr.reshape(image, (-1, 3))
    ref_flat = dr.reshape(reference, (-1, 3))
    cos_sim = dr.sum(img_flat * ref_flat, axis=-1)

    loss = dr.select(mask_flat, 1.0 - cos_sim, 0.0)

    return dr.mean(loss)

def lnormal_sqr(reference, image, mask_flat):
    ref = dr.reshape(reference, (-1, 3))
    img = dr.reshape(image, (-1, 3))
    cos_sim = dr.sum(ref * img, axis=-1)

    loss = dr.select(mask_flat, dr.square(1.0 - cos_sim), 0.0)
    
    return dr.mean(loss)

def lmae(reference, image, mask):
    '''
    Normal Mean Angular Error
    '''    
    reference = np.array(reference)
    image = np.array(image)
    
    cos = np.where(mask, np.clip(np.sum(reference * image, axis=-1), -1.0, 1.0), 1.0)
    
    mae = np.mean(np.arccos(cos) * 180.0 / np.pi)

    return mae

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def lssim(img1, img2, window_size=11, size_average=True, convert_to_srgb=False):
    
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    
    if convert_to_srgb:
        img1 = linear_to_srgb(img1)
        img2 = linear_to_srgb(img2)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ldiscrete_laplacian_reg_1dim(data, idx):
    
    num_neighbors = idx.shape[1] - 1
    
    c = dr.gather(mi.Float, data.array, idx[:, 0])
    
    neighbor_sum = mi.Float(0.0)
    for i in range(1, idx.shape[1]):
        val = dr.gather(mi.Float, data.array, idx[:, i])
        neighbor_sum += val
    neighbor_avg = neighbor_sum / float(num_neighbors)
    
    laplacian = dr.sqr(c - neighbor_avg)

    return dr.sum(laplacian)

def ldiscrete_laplacian_reg_3dims(data, idx):

    num_neighbors = idx.shape[1] - 1
    c_idx = idx[:, 0] * 3

    c = mi.Vector3f(
        dr.gather(mi.Float, data.array, c_idx),
        dr.gather(mi.Float, data.array, c_idx + 1),
        dr.gather(mi.Float, data.array, c_idx + 2),
    )
    
    neighbor_sum = mi.Vector3f(0.0)
    for i in range(1, idx.shape[1]):
        neighbor_idx = idx[:, i] * 3
        val_0 = dr.gather(mi.Float, data.array, neighbor_idx)
        val_1 = dr.gather(mi.Float, data.array, neighbor_idx + 1)
        val_2 = dr.gather(mi.Float, data.array, neighbor_idx + 2)
        
        val = mi.Vector3f(val_0, val_1, val_2)
        neighbor_sum += val
    neighbor_avg = neighbor_sum / float(num_neighbors)
    
    laplacian = dr.squared_norm(c - neighbor_avg)

    return dr.sum(laplacian)

def envmap_reg(opt, n_sg):
    try:
        L_reg = 0.0
        for i in range(n_sg):
            mu = opt[f'envmap.lgtSGsmu_{i}']
            L_reg += dr.squared_norm(mu)
        return L_reg
    except KeyError:
        # envmap optimization is disabled, opt has no SG params
        return 0.0