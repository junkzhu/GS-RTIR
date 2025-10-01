import drjit as dr
import mitsuba as mi

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