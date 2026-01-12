import torch
from utils import srgb_to_linear, linear_to_srgb
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Dict, Tuple

def rotmat_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        s = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s

    return torch.stack([w, x, y, z])

def quat_mul(q1, q2):
    # q = (w,x,y,z)
    w1,x1,y1,z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    w2,x2,y2,z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

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

    def restore_from_ply(self, path, reset_attribute = False):
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

        if all(n in prop_names for n in ["a_0", "a_1", "a_2"]) and not reset_attribute:
            albedos = np.stack((np.asarray(plydata.elements[0]["a_0"]),
                                np.asarray(plydata.elements[0]["a_1"]),
                                np.asarray(plydata.elements[0]["a_2"])), axis=1)
        else:
            albedos = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.5

        if "r" in prop_names and not reset_attribute:
            roughnesses = np.asarray(plydata.elements[0]["r"])[..., np.newaxis]
        else:
            roughnesses = np.ones((xyz.shape[0], 1), dtype=np.float32)

        if "m" in prop_names:
            metallics = np.asarray(plydata.elements[0]["m"])[..., np.newaxis]
        else:
            metallics = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        
        self._xyz = torch.from_numpy(xyz)
        self._opacity = torch.from_numpy(opacities)    
        self._scaling = torch.from_numpy(scales)
        self._rotation = torch.from_numpy(rots)
        self._features_dc = torch.from_numpy(features_dc)
        self._features_rest = torch.from_numpy(features_rest) 
        self._normal = torch.from_numpy(normals) 
        self._albedo = torch.from_numpy(albedos)
        self._roughness = torch.from_numpy(roughnesses)
        self._metallic = torch.from_numpy(metallics)

        self._opacity = torch.sigmoid(self._opacity)
        self._opacity = torch.clamp(self._opacity, 1e-8, 1.0 - 1e-8)

        self._scaling = torch.exp(self._scaling)
        self._scaling = torch.clamp(self._scaling, min=1e-6)

        self._rotation = torch.roll(self._rotation, shifts=-1, dims=1)
        self._rotation = self._rotation / (torch.norm(self._rotation, dim=1, keepdim=True) + 1e-8)
        
        #self._normal = self._normal / (torch.norm(self._normal, dim=1, keepdim=True) + 1e-8)
        #normal = [x * 2 - 1 for x in normal]

    def restore_from_ckpt(self, path, reset_attribute = False):
        '''
        Read checkpoint file from 3dgrt 
        '''
        
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        
        self._xyz = checkpoint["positions"]
        self._scaling = checkpoint["scale"]
        self._rotation = checkpoint["rotation"]
        self._opacity = checkpoint["density"]

        N = self._xyz.shape[0]

        if "features_albedo" in checkpoint:
            features_dc = checkpoint["features_albedo"].unsqueeze(1)
        else:
            features_dc = torch.zeros((N, 1, 3))
        
        if "features_specular" in checkpoint:
            features_rest = checkpoint["features_specular"].reshape(N, 15, 3)
            #features_rest = torch.transpose(features_rest, 1, 2)
        else:
            features_rest = torch.zeros((N, 15, 3))

        self._features_dc = features_dc
        self._features_rest = features_rest

        self._normal = torch.from_numpy(np.zeros((N, 3))) 
        self._albedo = torch.from_numpy(np.ones((N, 3), dtype=np.float32) * 0.5)
        self._roughness = torch.from_numpy(np.ones((N, 1), dtype=np.float32))
        self._metallic = torch.from_numpy(np.zeros((N, 1), dtype=np.float32))

        self._opacity = torch.sigmoid(self._opacity)
        self._opacity = torch.clamp(self._opacity, 1e-8, 1.0 - 1e-8)

        self._scaling = torch.exp(self._scaling)
        self._scaling = torch.clamp(self._scaling, min=1e-6)

        self._rotation = torch.roll(self._rotation, shifts=-1, dims=1)
        self._rotation = self._rotation / (torch.norm(self._rotation, dim=1, keepdim=True) + 1e-8)

        self._normal = self._normal / (torch.norm(self._normal, dim=1, keepdim=True) + 1e-8)

    def restore_from_params(self, params):
        n = int(params['shape.normals'].shape[0]/3)

        if 'shape.data' in params:
            data = np.array(params['shape.data'])
            data = data.reshape(n, 10)
            
            self._xyz = data[:, 0:3]
            self._scaling = data[:, 3:6]
            self._rotation = data[:, 6:10]

        if 'shape.sh_coeffs' in params:
            sh_coeffs = np.array(params['shape.sh_coeffs'])
            sh_coeffs = sh_coeffs.reshape(n, -1)
            self._features_dc   = sh_coeffs[:, 0:3]
            self._features_rest = sh_coeffs[:, 3:]

        if 'shape.opacities' in params:
            opacities = np.array(params['shape.opacities'])
            self._opacity = opacities.reshape(n, 1)
        
        if 'shape.normals' in params:
            normals = np.array(params['shape.normals'])
            self._normal = normals.reshape(n, 3)

        if 'shape.albedos'in params:
            albedos = np.array(params['shape.albedos'])
            self._albedo = albedos.reshape(n, 3)
        else:
            self._albedo = torch.from_numpy(np.ones((n, 3), dtype=np.float32) * 0.5)
        
        if 'shape.roughnesses' in params:
            roughnesses = np.array(params['shape.roughnesses'])
            self._roughness = roughnesses.reshape(n, 1)
        else:
            self._roughness = torch.from_numpy(np.ones((n, 1), dtype=np.float32))
        
        if 'shape.metallics' in params:
            metallics = np.array(params['shape.metallics'])
            self._metallic = metallics.reshape(n, 1)
        else:
            self._metallic = torch.from_numpy(np.zeros((n, 1), dtype=np.float32))

    def save_ply(self, path):
        N = self._xyz.shape[0]

        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('a_0', 'f4'), ('a_1', 'f4'), ('a_2', 'f4'),
            ('r', 'f4'),
            ('m', 'f4')
        ]
        
        num_coeffs = self._features_rest.reshape(N, -1).shape[1]
        for i in range(num_coeffs):
            dtype.append((f'f_rest_{i}', 'f4'))

        for i in range(self._scaling.shape[1]):
            dtype.append((f'scale_{i}', 'f4'))

        for i in range(self._rotation.shape[1]):
            dtype.append((f'rot_{i}', 'f4'))

        vertices = np.zeros(N, dtype=dtype)

        xyzs = self._xyz.numpy() if isinstance(self._xyz, torch.Tensor) else self._xyz
        vertices['x'] = xyzs[:, 0].astype(np.float32)
        vertices['y'] = xyzs[:, 1].astype(np.float32)
        vertices['z'] = xyzs[:, 2].astype(np.float32)

        normals = self._normal.numpy() if isinstance(self._normal, torch.Tensor) else self._normal
        #normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        vertices['nx'] = normals[:, 0].astype(np.float32)
        vertices['ny'] = normals[:, 1].astype(np.float32)
        vertices['nz'] = normals[:, 2].astype(np.float32)

        opacity = self._opacity.numpy() if isinstance(self._opacity, torch.Tensor) else self._opacity
        opacity = np.log(opacity / (1.0 - opacity))
        vertices['opacity'] = opacity[:, 0].astype(np.float32)

        features_dc = self._features_dc.numpy() if isinstance(self._features_dc, torch.Tensor) else self._features_dc
        features_dc = features_dc.reshape(N, -1)
        vertices['f_dc_0'] = features_dc[:, 0].astype(np.float32)
        vertices['f_dc_1'] = features_dc[:, 1].astype(np.float32)
        vertices['f_dc_2'] = features_dc[:, 2].astype(np.float32)

        f_rest = self._features_rest.numpy() if isinstance(self._features_rest, torch.Tensor) else self._features_rest
        f_rest = f_rest.reshape(N, 15, 3).transpose(0, 2, 1).reshape(N, -1)
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

        albedos = self._albedo.numpy() if isinstance(self._albedo, torch.Tensor) else self._albedo
        vertices['a_0'] = albedos[:, 0].astype(np.float32)
        vertices['a_1'] = albedos[:, 1].astype(np.float32)
        vertices['a_2'] = albedos[:, 2].astype(np.float32)

        roughness = self._roughness.numpy() if isinstance(self._roughness, torch.Tensor) else self._roughness
        vertices['r'] = roughness[:, 0].astype(np.float32)

        metallic = self._metallic.numpy() if isinstance(self._metallic, torch.Tensor) else self._metallic
        vertices['m'] = metallic[:, 0].astype(np.float32)

        ply_el = PlyElement.describe(vertices, 'vertex')
        PlyData([ply_el], text=False).write(path)

    def rescale_albedo(self, scale):
        #albedo_linear = srgb_to_linear(self._albedo)
        #scale = np.array(scale)
        #albedo_linear = albedo_linear * scale
        #self._albedo = torch.from_numpy(linear_to_srgb(albedo_linear))

        scale = torch.from_numpy(np.array(scale))
        self._albedo = self._albedo * scale

    def translate(self, translate):
        new_xyz = self._xyz
        new_xyz = new_xyz + translate
        self._xyz = new_xyz

    def scale(self, scale):
        new_xyz = self._xyz
        mean_xyz = torch.mean(new_xyz,0)
        new_xyz = new_xyz - mean_xyz
        new_xyz = new_xyz * scale
        self._xyz = new_xyz + mean_xyz

        # scale gaussians scale
        new_scaling = self._scaling * scale
        self._scaling = new_scaling

    def rotate(self, rotmat):
        self.rotate_xyz(rotmat)
        self.rotate_rot(rotmat)
        self.rotate_normal(rotmat)

    def rotate_xyz(self, rotmat):
        new_xyz = self._xyz
        mean_xyz = torch.mean(new_xyz,0)
        new_xyz = new_xyz - mean_xyz
        new_xyz = new_xyz @ rotmat.T
        self._xyz = new_xyz + mean_xyz
    
    def rotate_rot(self, rotmat):
        qR = rotmat_to_quat(rotmat).to(self._rotation.device)
        qR = qR.unsqueeze(0).expand(self._rotation.shape[0], -1)
        
        rot = torch.roll(self._rotation, shifts=1, dims=1)
        rot = quat_mul(qR, rot)
        self._rotation = torch.roll(rot, shifts=-1, dims=1)

    def rotate_normal(self, rotmat):
        self._normal = self._normal @ rotmat.T