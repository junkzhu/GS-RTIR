import os
import numpy as np
import mitsuba as mi

from constants import *
from utils import set_sensor_res
from .dataset_readers import sceneLoadTypeCallbacks

class Dataset:

    def __init__(
        self,
        source_path
    ) -> None:
        
        self.batch_size = BATCH_SIZE
        self.render_upsample_iter = None

        self.sensors = {}
        self.ref_images = {}
        self.ref_albedo_images = {}
        self.ref_normal_images = {}
        self.ref_roughness_images = {}

        self.target_res = [800,800]

        if os.path.exists(os.path.join(source_path, "sparse")):
            assert(1==2) #TODO COLMAP
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            self.sensors, self.ref_images, self.ref_albedo_images, self.ref_normal_images, self.ref_roughness_images = sceneLoadTypeCallbacks["Blender"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1]
            )

            self.render_upsample_iter = list(RENDER_UPSAMPLE_ITER)
            self.init_res = mi.ScalarPoint2i(np.array(self.target_res)//2**len(self.render_upsample_iter))

            for sensor in self.sensors:
                set_sensor_res(sensor, self.init_res)

        else:
            assert False, "Could not recognize scene type!"

    def get_sensor_iterator(self, i):
        n_sensors = len(self.sensors)
        if self.batch_size and (self.batch_size < n_sensors):
            # Access sensors in a strided way, assuming that this will maximize angular coverage per iteration
            steps = int(np.ceil(n_sensors / self.batch_size))
            indices = [(j * steps + i % steps) % n_sensors for j in range(self.batch_size)]
            sensors = [self.sensors[idx] for idx in indices]
            return zip(indices, sensors)
        else:
            return enumerate(self.sensors)
        
    def update_sensors(self, i):
        if self.render_upsample_iter is not None and i in self.render_upsample_iter:
            target_res = self.init_res * 2 ** (sorted(self.render_upsample_iter).index(i) + 1)
            for sensor in self.sensors:
                set_sensor_res(sensor, target_res)