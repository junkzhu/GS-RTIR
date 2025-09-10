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