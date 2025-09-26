import torch
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Dict, Tuple

class GaussianModel:
    def __init__(self) -> None:
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0)
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

    #output of GS-IR
    def restore_from_chkpnt(
        self,
        model_args: Tuple[
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Dict,
            float,
        ]
    ) -> None:
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal,
            self._albedo,
            self._roughness,
            self._metallic,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args

    def restore_from_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        features_dc = np.zeros((xyz.shape[0], 1, 3))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        rest_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        rest_names = sorted(rest_names, key=lambda x: int(x.split('_')[-1]))
        features_rest = np.zeros((xyz.shape[0], len(rest_names)))
        for idx, attr_name in enumerate(rest_names):
            features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        num_coeffs = len(rest_names) // 3
        features_rest = features_rest.reshape(xyz.shape[0], 3, num_coeffs)
        features_rest = np.transpose(features_rest, (0, 2, 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        prop_names = [p.name for p in plydata.elements[0].properties]
        if all(n in prop_names for n in ["nx", "ny", "nz"]):
            normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                                np.asarray(plydata.elements[0]["ny"]),
                                np.asarray(plydata.elements[0]["nz"])), axis=1)
        else:
            normals = np.zeros((xyz.shape[0], 3))
            normals[:, 2] = 1.0

        self._xyz = torch.from_numpy(xyz)
        self._opacity = torch.from_numpy(opacities)    
        self._scaling = torch.from_numpy(scales)
        self._rotation = torch.from_numpy(rots)
        self._features_dc = torch.from_numpy(features_dc)
        self._features_rest = torch.from_numpy(features_rest) 
        self._normal = torch.from_numpy(normals) 

    def restore_from_params(self, params):
        data = np.array(params['shape.data'])
        n = data.shape[0] // 10
        data = data.reshape(n, 10)
        
        self._xyz = data[:, 0:3]
        self._scaling = data[:, 3:6]
        self._rotation = data[:, 6:10]

        sh_coeffs = np.array(params['shape.sh_coeffs'])
        sh_coeffs = sh_coeffs.reshape(n, -1)
        self._features_dc   = sh_coeffs[:, 0:3]
        self._features_rest = sh_coeffs[:, 3:]

        opacities = np.array(params['shape.opacities'])
        self._opacity = opacities.reshape(n, 1)

        normals = np.array(params['shape.normals'])
        self._normal = normals.reshape(n, 3)

    def save_ply(self, path):
        N = self._xyz.shape[0]

        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]

        num_coeffs = self._features_rest.shape[1]
        for i in range(num_coeffs):
            dtype.append((f'f_rest_{i}', 'f4'))

        for i in range(self._scaling.shape[1]):
            dtype.append((f'scale_{i}', 'f4'))

        for i in range(self._rotation.shape[1]):
            dtype.append((f'rot_{i}', 'f4'))

        vertices = np.zeros(N, dtype=dtype)

        vertices['x'] = self._xyz[:, 0].astype(np.float32)
        vertices['y'] = self._xyz[:, 1].astype(np.float32)
        vertices['z'] = self._xyz[:, 2].astype(np.float32)

        normals = self._normal.numpy() if isinstance(self._normal, torch.Tensor) else self._normal
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        vertices['nx'] = normals[:, 0].astype(np.float32)
        vertices['ny'] = normals[:, 1].astype(np.float32)
        vertices['nz'] = normals[:, 2].astype(np.float32)

        opacity = self._opacity.numpy() if isinstance(self._opacity, torch.Tensor) else self._opacity
        opacity = np.clip(opacity, 1e-6, 1-1e-6)
        opacity = np.log(opacity / (1 - opacity))
        vertices['opacity'] = opacity[:, 0].astype(np.float32)

        features_dc = self._features_dc.numpy() if isinstance(self._features_dc, torch.Tensor) else self._features_dc
        vertices['f_dc_0'] = features_dc[:, 0].astype(np.float32)
        vertices['f_dc_1'] = features_dc[:, 1].astype(np.float32)
        vertices['f_dc_2'] = features_dc[:, 2].astype(np.float32)

        f_rest = self._features_rest.numpy() if isinstance(self._features_rest, torch.Tensor) else self._features_rest
        f_rest = f_rest.reshape(N, -1)
        for i in range(f_rest.shape[1]):
            vertices[f'f_rest_{i}'] = f_rest[:, i].astype(np.float32)

        scales = self._scaling.numpy() if isinstance(self._scaling, torch.Tensor) else self._scaling
        scales = np.maximum(scales, 1e-6)
        scales = np.log(scales)
        for i in range(scales.shape[1]):
            vertices[f'scale_{i}'] = scales[:, i].astype(np.float32)

        rots = self._rotation.numpy() if isinstance(self._rotation, torch.Tensor) else self._rotation
        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-8)
        rots = np.roll(rots, shift=1, axis=1)
        for i in range(rots.shape[1]):
            vertices[f'rot_{i}'] = rots[:, i].astype(np.float32)

        ply_el = PlyElement.describe(vertices, 'vertex')
        PlyData([ply_el], text=False).write(path)

