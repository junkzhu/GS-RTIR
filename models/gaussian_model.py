import torch
import numpy as np
from plyfile import PlyData
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

        self._xyz = torch.from_numpy(xyz)
        self._opacity = torch.from_numpy(opacities)    
        self._scaling = torch.from_numpy(scales)
        self._rotation = torch.from_numpy(rots)
        self._features_dc = torch.from_numpy(features_dc)
        self._features_rest = torch.from_numpy(features_rest) 
        self._normal = torch.zeros((xyz.shape[0], 3))
        self._normal[:, 2] = 1.0