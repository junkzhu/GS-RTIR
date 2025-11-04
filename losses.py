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