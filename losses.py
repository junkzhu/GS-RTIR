import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1(reference, image):
    '''
    L1 loss function
    '''
    return dr.mean(dr.abs(reference - image), axis=None)

def l2(reference, image):
    '''
    L2 loss function
    '''
    return dr.mean(dr.power(reference - image, 2), axis=None)

def lpsnr(reference, image):
    '''
    PSNR loss function
    '''
    reference = np.asarray(reference, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)

    reference = reference**(1/2.2)
    image = image**(1/2.2)

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

def lssim(img1, img2, window_size=11, size_average=True):
    
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
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
