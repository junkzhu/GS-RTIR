import os
import numpy as np
import mitsuba as mi
from collections import defaultdict

from constants import *
from utils import set_sensor_res, get_relighting_envmap_names
from .dataset_readers import sceneLoadTypeCallbacks

class Dataset:

    def __init__(
        self,
        source_path,
        render_upsampler_iters=TRAIN_UPSAMPLE_ITER,
        dataset_type="train",
        env='sunset',
        load_ref_relight_images=False,
        train_iters=None
    ) -> None:
        
        self.batch_size = args.batch_size
        
        self.shuffle = args.shuffle
        self.rng = np.random.RandomState(42)
        self.sensor_perm = None
        self.sensor_ptr = 0

        self.render_upsample_iter = list(render_upsampler_iters)

        self.sensors = {}
        self.sensors_normal = {}
        self.sensors_intrinsic = {}

        self.ref_images = {}
        self.ref_albedo_images = {}
        self.ref_normal_images = {}
        self.ref_roughness_images = {}
        self.albedo_priors_images = {}
        self.roughness_priors_images = {}
        self.normal_priors_images = {}

        self.relight_envmap_names = [env]
        self.ref_relight_images = defaultdict(list)

        self.target_res = [800,800]

        self.max_reso_scale_init = 8 if train_iters != None and args.dash_reso_sche else None

        if args.dataset_type == 'COLMAP':
            assert(1==2) #TODO COLMAP
        elif args.dataset_type in ["TensoIR", "RTIR"]:
            self.sensors, self.sensors_normal, self.sensors_intrinsic, self.ref_images, self.ref_albedo_images, self.ref_normal_images, self.ref_roughness_images, self.albedo_priors_images, self.roughness_priors_images, self.normal_priors_images, self.ref_relight_images = sceneLoadTypeCallbacks["TensoIR"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1], split=dataset_type, env=env, 
                relight_envmap_names=self.relight_envmap_names, load_ref_relight_images=load_ref_relight_images, 
                train_iters=train_iters, max_reso_scale_init=self.max_reso_scale_init,
            )
        elif args.dataset_type == 'Synthetic4Relight':
            self.sensors, self.sensors_normal, self.sensors_intrinsic, self.ref_images, self.ref_albedo_images, self.ref_normal_images, self.ref_roughness_images, self.albedo_priors_images, self.roughness_priors_images, self.normal_priors_images, self.ref_relight_images, new_render_upsample_iter  = sceneLoadTypeCallbacks["Synthetic4Relight"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1], split=dataset_type, env=env, 
                relight_envmap_names=self.relight_envmap_names, load_ref_relight_images=load_ref_relight_images,
                train_iters=train_iters, max_reso_scale_init=self.max_reso_scale_init,
            )
        elif args.dataset_type == 'Stanford_orb':
            assert False
        else:
            assert False, "Could not recognize scene type!"

        if train_iters == None:
            self.init_res = mi.ScalarPoint2i(np.array(self.target_res)//2**len(self.render_upsample_iter))
        else:
            self.init_res = mi.ScalarPoint2i(np.array(self.target_res)//self.max_reso_scale_init)
            self.render_upsample_iter = new_render_upsample_iter

        for sensor in self.sensors:
            set_sensor_res(sensor, self.init_res)

    def _reset_epoch(self):
        n = len(self.sensors)
        if self.shuffle:
            self.sensor_perm = self.rng.permutation(n)
        else:
            self.sensor_perm = np.arange(n)
        self.sensor_ptr = 0

    def get_sensor_iterator(self):
        if self.sensor_perm is None or self.sensor_ptr >= len(self.sensors):
            self._reset_epoch()

        indices = self.sensor_perm[self.sensor_ptr : self.sensor_ptr + self.batch_size]

        self.sensor_ptr += len(indices)

        sensors = [self.sensors[idx] for idx in indices]

        return zip(indices, sensors)
        
    def update_sensors(self, i):
        if self.render_upsample_iter is not None and i in self.render_upsample_iter:
            if self.max_reso_scale_init == None:
                target_res = self.init_res * 2 ** (sorted(self.render_upsample_iter).index(i) + 1)
            else:
                target_res = (self.init_res * self.max_reso_scale_init) // (self.max_reso_scale_init - sorted(self.render_upsample_iter).index(i))
            for sensor in self.sensors:
                set_sensor_res(sensor, target_res)