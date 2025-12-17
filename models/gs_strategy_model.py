from asyncio.log import logger
import torch
import math
import numpy as np
import mitsuba as mi
import drjit as dr
from omegaconf import OmegaConf

def check_step_condition(step: int, start: int, end: int, freq: int) -> bool:
    """Checks if an operation should occur for the given step."""
    if (start >= 0 and step > start) and (step < end or end == -1) and step % freq == 0:
        return True
    return False

class GSStrategyModel:
    def __init__(self, gs_strategy_yaml):
        self.conf = OmegaConf.load(gs_strategy_yaml)
        self.split_n_gaussians = self.conf.densify.split_n_gaussians
        self.sensors_normal = None
        self.sensors_intrinsic = None

        self.num_gaussian = torch.empty(0)

        self.centers = torch.empty(0)
        self.scales = torch.empty(0)
        self.quats = torch.empty(0)
        self.opacities = torch.empty(0)
        self.sh_coeffs = torch.empty(0)
        self.normals = torch.empty(0)

        self.grad_norm = torch.empty(0)

    def quaternion_to_so3(self, r):
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), dtype=r.dtype, device=r.device)

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    def check_step_condition(step: int, start: int, end: int, freq: int) -> bool:
        """Checks if an operation should occur for the given step."""
        if (start >= 0 and step > start) and (step < end or end == -1) and step % freq == 0:
            return True
        return False
    
    def lr_schedule(self, opt, step, max_iterations, min_factor):
        if (step > 0 and step % 20 == 0):
            factor = min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * step / max_iterations))
            for key in opt.lr.keys():
                opt.lr[key] = opt.lr[key] * factor

    def update_opt(self, opt):
        centers = self.centers.transpose(0,1)
        
        scales = self.scales.transpose(0,1)
        quats = self.quats.transpose(0,1)
        opacities = self.opacities.squeeze()
        sh_coeffs = self.sh_coeffs.flatten()
        normals = self.normals.flatten()

        opt['centers'] = mi.Point3f(centers)
        opt['scales'] = mi.Vector3f(scales)
        opt['quats'] = mi.Vector4f(quats)
        opt['opacities'] = mi.Float(opacities)
        opt['sh_coeffs'] = mi.Float(sh_coeffs)
        opt['normals'] = mi.Float(normals)

    def update_gs(self, opt, step):
        self.centers = opt['centers'].torch().transpose(0,1) # (N,3)
        self.scales = opt['scales'].torch().transpose(0,1) # (N,3)
        self.quats = opt['quats'].torch().transpose(0,1) # (N,4)

        self.num_gaussian = self.centers.shape[0]
        
        self.opacities = opt['opacities'].torch().unsqueeze(1) # (N,1)
        self.sh_coeffs = opt['sh_coeffs'].torch().reshape(self.num_gaussian, -1) # (N,3)
        self.normals = opt['normals'].torch().reshape(self.num_gaussian, -1) # (N,3)

        scene_updated = False

        if check_step_condition(
            step,
            self.conf.densify.start_iteration,
            self.conf.densify.end_iteration,
            self.conf.densify.frequency,
        ):
            self.densify_gaussians()
            scene_updated = True
        
        if check_step_condition(
            step,
            self.conf.prune.start_iteration,
            self.conf.prune.end_iteration,
            self.conf.prune.frequency,
        ):
            self.prune_gaussians_opacity(self.conf.prune.prune_opacity_threshold)
            scene_updated = True
        
        if check_step_condition(
            step,
            self.conf.prune_scale.start_iteration,
            self.conf.prune_scale.end_iteration,
            self.conf.prune_scale.frequency,
        ):
            self.prune_gaussians_scale(self.conf.prune_scale.prune_scale_threshold)
            scene_updated = True

        if check_step_condition(
            step,
            self.conf.decay_opacities.start_iteration,
            self.conf.decay_opacities.end_iteration,
            self.conf.decay_opacities.frequency,
        ):
            self.decay_opacities(self.conf.decay_opacities.gamma)
            scene_updated = True

        if check_step_condition(
            step,
            self.conf.reset_opacities.start_iteration,
            self.conf.reset_opacities.end_iteration,
            self.conf.reset_opacities.frequency,
        ):
            self.reset_opacities(self.conf.reset_opacities.new_max_density)
            scene_updated = True

        if scene_updated:
            self.update_opt(opt)

    def update_grad_norm(self, opt):
        grad_center  = dr.grad(opt['centers']).torch().transpose(0,1) # (N,3)
        self.num_gaussian = grad_center.shape[0]
        grad_scale   = dr.grad(opt['scales']).torch().transpose(0,1) # (N,3)
        grad_quat    = dr.grad(opt['quats']).torch().transpose(0,1) # (N,4)
        grad_opacity = dr.grad(opt['opacities']).torch().unsqueeze(1) # (N,1)
        grad_sh      = dr.grad(opt['sh_coeffs']).torch().reshape(self.num_gaussian, -1) # (N,3)
        grad_normal  = dr.grad(opt['normals']).torch().reshape(self.num_gaussian, -1) # (N,3)

        grad_norm = torch.sqrt(
          (grad_center ** 2).sum(dim=1)
        + (grad_scale ** 2).sum(dim=1)
        + (grad_quat ** 2).sum(dim=1)
        + (grad_opacity ** 2).sum(dim=1)
        + (grad_sh ** 2).sum(dim=1)
        + (grad_normal ** 2).sum(dim=1)
        )

        self.grad_norm = grad_norm

    def update_num_gaussian(self):
        self.num_gaussian = self.centers.shape[0]

    def densify_gaussians(self):
        self.clone_gaussians(self.grad_norm, self.conf.densify.clone_grad_threshold, self.conf.densify.relative_size_threshold)
        self.split_gaussians(self.grad_norm, self.conf.densify.split_grad_threshold, self.conf.densify.split_n_gaussians, self.conf.densify.relative_size_threshold)
        #self.split_extreme_shape_gaussian(self.grad_norm, self.conf.densify.split_extreme_shape_grad_threshold, self.conf.densify.anisotropy_threshold, self.conf.densify.size_min, self.conf.densify.size_max)

    #Density-pruned
    def prune_gaussians_opacity(self, prune_opacity_threshold=0.1):
        mask = (self.opacities.squeeze() > prune_opacity_threshold)

        if self.conf.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            print(f" Density-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.centers = self.centers[mask]
        self.scales = self.scales[mask]
        self.quats = self.quats[mask]
        self.opacities = self.opacities[mask]
        self.sh_coeffs = self.sh_coeffs[mask]
        self.normals = self.normals[mask]
        self.grad_norm = self.grad_norm[mask]

        self.update_num_gaussian()

    #Cloned
    def clone_gaussians(self, grad, clone_grad_threshold = 0.0002, relative_size_threshold = 0.01):
        mask = (grad >= clone_grad_threshold)
        mask &= (torch.max(self.scales, dim=1).values <= relative_size_threshold)

        if self.conf.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            print(f" Cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        self.centers = torch.cat([self.centers, self.centers[mask]])
        self.scales = torch.cat([self.scales, self.scales[mask]])
        self.quats = torch.cat([self.quats, self.quats[mask]])
        self.opacities = torch.cat([self.opacities, self.opacities[mask]])
        self.sh_coeffs = torch.cat([self.sh_coeffs, self.sh_coeffs[mask]])
        self.normals = torch.cat([self.normals, self.normals[mask]])
        self.grad_norm = torch.cat([self.grad_norm, self.grad_norm[mask]])

        self.update_num_gaussian()

    def split_extreme_shape_gaussian(self, grad, split_extreme_shape_grad_threshold = 0.0002, anisotropy_threshold = 10.0, size_min = 0.002, size_max = 0.2):
        mask = (grad >= split_extreme_shape_grad_threshold)
        
        anisotropy = torch.max(self.scales, dim=1).values / torch.min(self.scales, dim=1).values
        extreme_shape = anisotropy > anisotropy_threshold

        valid_shape = (torch.max(self.scales, dim=1).values > size_min) & \
                (torch.max(self.scales, dim=1).values < size_max)
        
        extreme_big = torch.mean(self.scales, dim=1) > size_max

        mask &= extreme_shape & valid_shape
        mask |= extreme_big
        
        stds = self.scales[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = self.quaternion_to_so3(self.quats[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        
        if self.conf.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            print(f" Splitted {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) extreme shape gaussians")

        old_scales = self.scales[mask] # [N, 3]  
        avg_r = old_scales.mean(dim=1, keepdim=True)  # [N, 1]
        split_scales = avg_r * torch.ones_like(old_scales) * 0.5

        split_centers = self.centers[mask].repeat(self.split_n_gaussians, 1) + offsets * 0.1  # [2N, 3]
        split_scales = split_scales.repeat(self.split_n_gaussians, 1)
        split_quats = self.quats[mask].repeat(self.split_n_gaussians, 1)
        split_opacities = self.opacities[mask].repeat(self.split_n_gaussians, 1)
        split_sh_coeffs = self.sh_coeffs[mask].repeat(self.split_n_gaussians, 1)
        split_normals = self.normals[mask].repeat(self.split_n_gaussians, 1)
        split_grad_norm = self.grad_norm[mask].repeat(self.split_n_gaussians)

        self.centers = torch.cat([self.centers[~mask], split_centers])
        self.scales = torch.cat([self.scales[~mask], split_scales])
        self.quats = torch.cat([self.quats[~mask], split_quats])
        self.opacities = torch.cat([self.opacities[~mask], split_opacities])
        self.sh_coeffs = torch.cat([self.sh_coeffs[~mask], split_sh_coeffs])
        self.normals = torch.cat([self.normals[~mask], split_normals])
        self.grad_norm = torch.cat([self.grad_norm[~mask], split_grad_norm])

        self.update_num_gaussian()

    #Splitted
    def split_gaussians(self, grad, split_grad_threshold = 0.0002, split_n_gaussians = 2, relative_size_threshold = 0.01):
        mask = (grad >= split_grad_threshold)
        mask &= (torch.max(self.scales, dim=1).values > relative_size_threshold)

        stds = self.scales[mask].repeat(split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = self.quaternion_to_so3(self.quats[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        
        if self.conf.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            print(f" Splitted {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        split_centers = self.centers[mask].repeat(self.split_n_gaussians, 1) + offsets * 0.1  # [2N, 3]
        split_scales = self.scales[mask].repeat(self.split_n_gaussians, 1) / (0.8 * self.split_n_gaussians)
        split_quats = self.quats[mask].repeat(self.split_n_gaussians, 1)
        split_opacities = self.opacities[mask].repeat(self.split_n_gaussians, 1)
        split_sh_coeffs = self.sh_coeffs[mask].repeat(self.split_n_gaussians, 1)
        split_normals = self.normals[mask].repeat(self.split_n_gaussians, 1)
        split_grad_norm = self.grad_norm[mask].repeat(self.split_n_gaussians)

        self.centers = torch.cat([self.centers[~mask], split_centers])
        self.scales = torch.cat([self.scales[~mask], split_scales])
        self.quats = torch.cat([self.quats[~mask], split_quats])
        self.opacities = torch.cat([self.opacities[~mask], split_opacities])
        self.sh_coeffs = torch.cat([self.sh_coeffs[~mask], split_sh_coeffs])
        self.normals = torch.cat([self.normals[~mask], split_normals])
        self.grad_norm = torch.cat([self.grad_norm[~mask], split_grad_norm])

        self.update_num_gaussian()

    def decay_opacities(self, gamma=0.9):
        self.opacities *= gamma
        
        if self.conf.print_stats:
            print(f" Decay opacities by a factor of {gamma}")

    def reset_opacities(self, value=0.01):
        self.opacities = torch.clamp(self.opacities, max=torch.ones_like(self.opacities) * value)
        
        if self.conf.print_stats:
            print(f" Reset opacities to {value}")

    def prune_gaussians_scale(self, prune_scale_threshold=1.0):
        intrinsic_max = torch.tensor(self.sensors_intrinsic[0], dtype=torch.float32, device="cuda").max()
        
        cam_normals = torch.tensor(np.array(self.sensors_normal), dtype=torch.float32, device="cuda") # N，3
        similarities = torch.matmul(self.centers, cam_normals.T)
        cam_dists = similarities.min(dim=1)[0].clamp(min=1e-8) # N，1
        ratio = self.scales.min(dim=1)[0] / cam_dists * intrinsic_max

        mask = (ratio >= prune_scale_threshold)
        if self.conf.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            print(f" Scale-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.centers = self.centers[mask]
        self.scales = self.scales[mask]
        self.quats = self.quats[mask]
        self.opacities = self.opacities[mask]
        self.sh_coeffs = self.sh_coeffs[mask]
        self.normals = self.normals[mask]
        self.grad_norm = self.grad_norm[mask]

        self.update_num_gaussian()