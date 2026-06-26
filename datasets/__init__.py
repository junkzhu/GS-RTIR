import os
import json
import numpy as np
import mitsuba as mi
from collections import defaultdict
from pathlib import Path

from constants import *
from utils import set_sensor_res, get_relighting_envmap_names
from .dataset_readers import sceneLoadTypeCallbacks
from typing import List
import drjit as dr
from PIL import Image

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

        self.target_res = self._infer_target_res(source_path, dataset_type)
        self.current_res = None

        self.max_reso_scale_init = 8 if train_iters != None and args.dash_reso_sche else None

        if args.dataset_type == 'COLMAP':
            assert False #TODO COLMAP
        elif args.dataset_type in ["TensoIR", "RT4Relight"]:
            self.sensors, self.sensors_normal, self.sensors_intrinsic, self.ref_images, self.ref_albedo_images, self.ref_normal_images, self.ref_roughness_images, self.albedo_priors_images, self.roughness_priors_images, self.normal_priors_images, self.ref_relight_images, new_render_upsample_iter, self.reso_scales, self.reso_level_begin = sceneLoadTypeCallbacks["TensoIR"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1], split=dataset_type, env=env, 
                relight_envmap_names=self.relight_envmap_names, load_ref_relight_images=load_ref_relight_images, 
                train_iters=train_iters, max_reso_scale_init=self.max_reso_scale_init,
            )
        elif args.dataset_type == 'Synthetic4Relight':
            self.sensors, self.sensors_normal, self.sensors_intrinsic, self.ref_images, self.ref_albedo_images, self.ref_normal_images, self.ref_roughness_images, self.albedo_priors_images, self.roughness_priors_images, self.normal_priors_images, self.ref_relight_images, new_render_upsample_iter, self.reso_scales, self.reso_level_begin  = sceneLoadTypeCallbacks["Synthetic4Relight"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1], split=dataset_type, env=env, 
                relight_envmap_names=self.relight_envmap_names, load_ref_relight_images=load_ref_relight_images,
                train_iters=train_iters, max_reso_scale_init=self.max_reso_scale_init,
            )
        elif args.dataset_type == 'Nerf_Synthetic':
            self.sensors, self.sensors_normal, self.sensors_intrinsic, self.ref_images, self.ref_albedo_images, self.ref_normal_images, self.ref_roughness_images, self.albedo_priors_images, self.roughness_priors_images, self.normal_priors_images, self.ref_relight_images, new_render_upsample_iter, self.reso_scales, self.reso_level_begin  = sceneLoadTypeCallbacks["Nerf_Synthetic"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1], split=dataset_type, 
                relight_envmap_names=self.relight_envmap_names, load_ref_relight_images=load_ref_relight_images,
                train_iters=train_iters, max_reso_scale_init=self.max_reso_scale_init,
            )
        elif args.dataset_type == 'Stanford_orb':
            self.sensors, self.sensors_normal, self.sensors_intrinsic, self.ref_images, self.albedo_priors_images, self.roughness_priors_images, self.normal_priors_images, self.ref_relight_images, new_render_upsample_iter, self.reso_scales, self.reso_level_begin = sceneLoadTypeCallbacks["Stanford_orb"](
                source_path, 'rgb', resx=self.target_res[0], resy=self.target_res[1], split=dataset_type, env=env, 
                relight_envmap_names=self.relight_envmap_names, load_ref_relight_images=load_ref_relight_images,
                train_iters=train_iters, max_reso_scale_init=self.max_reso_scale_init,
            )
        else:
            assert False, "Could not recognize scene type!"

        if train_iters == None:
            self.init_res = mi.ScalarPoint2i(np.array(self.target_res)//2**len(self.render_upsample_iter))
            self.train_iters = None
        else:
            self.init_res = mi.ScalarPoint2i(np.array(self.target_res)//self.max_reso_scale_init)
            self.render_upsample_iter = new_render_upsample_iter
            self.render_upsample_iter.append(train_iters)
            self.train_iters = train_iters

        for sensor in self.sensors:
            set_sensor_res(sensor, self.init_res)
            self.current_res = self.init_res

    def _infer_target_res(self, source_path, split):
        """Infer target resolution from reference image size."""
        default_res = [800, 800]
        root = Path(source_path)

        transforms_candidates = [
            root / f"transforms_{split}.json",
            root / "transforms_train.json",
            root / "transforms_test.json",
        ]

        def _resolve_image_path(file_path_str):
            p = Path(file_path_str)
            if p.is_absolute():
                base = p
            else:
                base = root / p
            candidates = [
                base,
                Path(str(base) + ".png"),
                Path(str(base) + "_rgba.png"),
                Path(str(base) + ".jpg"),
                Path(str(base) + ".jpeg"),
                Path(str(base) + ".exr"),
            ]
            for c in candidates:
                if c.exists() and c.is_file():
                    return c
            return None

        for tf_path in transforms_candidates:
            if not tf_path.exists():
                continue
            try:
                with open(tf_path, "r") as f:
                    transforms_data = json.load(f)
                frames = transforms_data.get("frames", [])
                for frame in frames:
                    file_path = frame.get("file_path", None)
                    if not file_path:
                        continue
                    img_path = _resolve_image_path(file_path)
                    if img_path is None:
                        continue
                    with Image.open(img_path) as img:
                        w, h = img.size
                    return [w, h]
            except Exception:
                continue

        # Fallback: scan common image files directly under source_path.
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.exr"):
            for img_path in sorted(root.glob(pattern)):
                if not img_path.is_file():
                    continue
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    return [w, h]
                except Exception:
                    continue

        print(f"[Dataset] Failed to infer target resolution from {source_path}, fallback to {default_res}")
        return default_res

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

        indices = []
        while len(indices) < self.batch_size:
            remaining = self.batch_size - len(indices)
            available = len(self.sensors) - self.sensor_ptr
            
            if available == 0:
                self._reset_epoch()
                available = len(self.sensors)
            
            take = min(remaining, available)
            indices.extend(self.sensor_perm[self.sensor_ptr : self.sensor_ptr + take])
            self.sensor_ptr += take

        sensors = [self.sensors[idx] for idx in indices]

        # Prepare image lists for concatenation
        if self.current_res is None:
            raise ValueError("current_res is None. Please ensure update_sensors has been called or init_res is set.")
        
        res_key = self.current_res[0]
        
        # Define image sources mapping
        ref_image_sources = {
            'rgb': self.ref_images,
            'albedo': self.ref_albedo_images,
            'roughness': self.ref_roughness_images,
            'normal': self.ref_normal_images,
        }
        prior_image_sources = {
            'albedo': self.albedo_priors_images,
            'roughness': self.roughness_priors_images,
            'normal': self.normal_priors_images,
        }
        n_sensors = len(self.sensors)
        # Only include sources that have the same length as sensors (e.g. Synthetic4Relight
        # train has empty ref_albedo/ref_roughness/ref_normal; skip them to avoid IndexError)
        ref_image_lists = {
            key: [source[idx][res_key] for idx in indices]
            for key, source in ref_image_sources.items()
            if source is not None and len(source) == n_sensors
        }
        prior_image_lists = {
            key: [source[idx][res_key] for idx in indices]
            for key, source in prior_image_sources.items()
            if source is not None and len(source) == n_sensors
        }

        # Concatenate sensors and images sequentially
        # IMPORTANT: Cannot use ThreadPoolExecutor here because drjit operations (dr.zeros, dr.eval)
        # create computation graphs that are thread-local. When operations are split across
        # parent/child threads, drjit's scope management fails, causing "scope ID" errors.
        # The error message indicates: "Very likely, a computation is split across a parent/child thread"
        batch_sensor = self.concatenate_sensors(sensors)
        ref_imgs = {
            key: self.concatenate_tensors(img_list)
            for key, img_list in ref_image_lists.items()
        }
        priors_imgs = {
            key: self.concatenate_tensors(img_list)
            for key, img_list in prior_image_lists.items()
        }

        return batch_sensor, ref_imgs, priors_imgs
        
    def update_sensors(self, i):
        """Update sensor resolution based on iteration. For resume: i may be any completed iter, not necessarily in render_upsample_iter."""
        if self.max_reso_scale_init is None:
            # Standard resolution scheduling: resolution = init_res * 2^level where level = number of upsample iters already passed
            if self.render_upsample_iter is not None:
                sorted_ups = sorted(self.render_upsample_iter)
                level = sum(1 for u in sorted_ups if u <= i)
                target_res = self.init_res * 2 ** level
                for sensor in self.sensors:
                    set_sensor_res(sensor, target_res)
                self.current_res = target_res
        else:
            # DashGaussian resolution scheduling - update every iteration
            from .dataset_readers import get_res_scale
            if self.reso_scales is not None and self.reso_level_begin is not None and self.train_iters is not None:
                scale = get_res_scale(i, self.reso_scales, self.reso_level_begin, self.train_iters)
                target_res = mi.ScalarPoint2i(np.array(self.target_res) // scale)
                for sensor in self.sensors:
                    set_sensor_res(sensor, target_res)
                self.current_res = target_res
                # Only print when resolution changes or at key iterations
                if i == 0 or (self.render_upsample_iter is not None and i in self.render_upsample_iter):
                    print(f"Iter {i}: scale={scale}, res={target_res}")

    def concatenate_tensors(self, images) -> mi.TensorXf:
        '''
        Concatenate a list of numpy arrays on the X axis
        Converts numpy arrays to TensorXf for concatenation
        '''
        if not images:
            raise ValueError("Cannot concatenate empty list of images")
        
        # Convert numpy arrays to TensorXf
        tensor_images = [mi.TensorXf(img) for img in images]
        
        shape = tensor_images[0].shape
        h = shape[0]
        w = shape[1]
        if len(shape) == 2:
            concatenated = dr.zeros(mi.TensorXf, shape=(h, len(tensor_images) * w))
            for i, img in enumerate(tensor_images):
                concatenated[:, (w*i):(w*(i+1))] = img.array
            concatenated = concatenated[:, :, None]
        else:
            concatenated = dr.zeros(mi.TensorXf, shape=(h, len(tensor_images) * w, shape[2]))
            for i, img in enumerate(tensor_images):
                concatenated[:, (w*i):(w*(i+1)), :] = img.array
        dr.eval(concatenated) # TODO necessary?
        return concatenated

    def concatenate_sensors(self, sensors: List[mi.Sensor]) -> mi.Sensor:
        if not sensors:
            raise ValueError("Cannot concatenate empty list of sensors")
        if self.current_res is None:
            raise ValueError("current_res is None. Cannot concatenate sensors without resolution.")
        
        res = self.current_res
        batch_sensor_dict = {
            'type': 'batch',
            'film': {
                'type': 'hdrfilm',
                'width': res[0] * len(sensors), 'height': res[1],
                'filter': { 'type': 'tent' },
            }
        }
        # Use enumerate for more pythonic iteration
        for i, sensor in enumerate(sensors):
            batch_sensor_dict[f'cam_{i:04d}'] = sensor
        batch_sensor = mi.load_dict(batch_sensor_dict)  
        return batch_sensor