import drjit as dr
import mitsuba as mi
import numpy as np

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
    return 20 * dr.log(1.0 * dr.rsqrt(l2(reference, image))) * dr.rcp(dr.log(10))

def TV(reference, image):
    rgb_grad_h = dr.exp(-dr.mean(dr.abs(reference[1:, :, :] - reference[:-1, :, :]), axis=2))
    rgb_grad_w = dr.exp(-dr.mean(dr.abs(reference[:, 1:, :] - reference[:, :-1, :]), axis=2))
    tv_h = dr.power((image[1:, :, :] - image[:-1, :, :]), 2)
    tv_w = dr.power((image[:, 1:, :] - image[:, :-1, :]), 2)
    tv_loss = dr.mean(tv_h * rgb_grad_h[:,:,dr.newaxis]) + dr.mean(tv_w * rgb_grad_w[:,:,dr.newaxis])
    return tv_loss

def lnormal(reference, image):
    img_flat = dr.reshape(image, (-1, 3))
    ref_flat = dr.reshape(reference, (-1, 3))
    cos_sim = dr.sum(img_flat * ref_flat, axis=-1)
    loss = 1.0 - cos_sim
    return dr.mean(loss)

def lmae(reference, image):
    '''
    Mean Angular Error
    '''    
    eps = 1e-8

    reference = np.array(reference)
    image = np.array(image)

    mask = (np.linalg.norm(reference, axis=-1) > eps) & (np.linalg.norm(image, axis=-1) > eps)

    ref_norm = reference / np.clip(np.linalg.norm(reference, axis=-1, keepdims=True), eps, None)
    img_norm = image / np.clip(np.linalg.norm(image, axis=-1, keepdims=True), eps, None)

    cos_theta = np.sum(ref_norm * img_norm, axis=-1)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

    mae = np.mean(angle[mask])

    return mae