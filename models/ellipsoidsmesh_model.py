import mitsuba as mi
import numpy as np

class EllipsoidsFactory:
    def __init__(self):
        self.centers = []
        self.scales = []
        self.quaternions = []
        self.sigmats = []
        self.feature = []

        self.normals = []
        self.albedos = []
        self.roughnesses = []
        self.metallics = []

    def add(self, mean, scale, sigmat=1.0, albedo=[0.5, 0.5, 0.5], quaternion=[0.0, 0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], roughness = 0.8, metallic = 0.0, feature=np.zeros((16, 3))):
        sigmat = 1 / (1 + np.exp(-sigmat))
        sigmat = np.clip(sigmat, 1e-8, 1.0 - 1e-8)
        scale = np.exp(scale)
        scale = np.maximum(scale, 1e-6)    
        quaternion /= np.linalg.norm(quaternion, keepdims=True)
        quaternion = np.roll(quaternion, shift=-1)
        normal /= np.linalg.norm(normal, keepdims=True)
        #normal = [x * 2 - 1 for x in normal]

        self.centers.append(mi.ScalarPoint3f(mean))
        self.scales.append(mi.ScalarVector3f(scale))
        self.quaternions.append(mi.ScalarQuaternion4f(quaternion))
        self.sigmats.append(sigmat)
        self.feature.append(feature)

        self.normals.append(mi.ScalarVector3f(normal))
        self.roughnesses.append(roughness)
        self.metallics.append(metallic)

        if isinstance(albedo, float):
            albedo = mi.ScalarColor3f(albedo)
        self.albedos.append(albedo)

    def build(self):
        num_gaussians = len(self.centers)
        centers = mi.TensorXf(np.ravel(np.array(self.centers)), shape=(num_gaussians, 3))
        scales  = mi.TensorXf(np.ravel(np.array(self.scales)), shape=(num_gaussians, 3))
        quats   = mi.TensorXf(np.ravel(np.array(self.quaternions)), shape=(num_gaussians, 4))
        normals  = mi.TensorXf(np.ravel(np.array(self.normals)), shape=(num_gaussians, 3))
        sigmats = mi.TensorXf(self.sigmats, shape=(num_gaussians, 1))
        self.feature = np.array(self.feature).reshape((num_gaussians, -1))
        features = mi.TensorXf(np.array(self.feature))

        self.albedos = np.array(self.albedos).reshape((num_gaussians, -1))
        albedos = mi.TensorXf(np.array(self.albedos))
        roughnesses = mi.TensorXf(self.roughnesses, shape=(num_gaussians, 1))
        metallics = mi.TensorXf(self.metallics, shape=(num_gaussians, 1))
        
        attributes = {
            'centers': centers,
            'scales': scales,
            'quats': quats,
            'normals': normals,
            'sigmats': sigmats,
            'features': features,
            'albedos': albedos,
            'roughnesses': roughnesses,
            'metallics': metallics
        }

        return attributes
    
    def load_gaussian(self, gaussians):
        xyzs = gaussians._xyz.detach().numpy()
        opacitys = gaussians._opacity.detach().numpy()
        scales = gaussians._scaling.detach().numpy()
        rots = gaussians._rotation.detach().numpy()
        normal = gaussians._normal.detach().numpy()
        features = gaussians._features_dc.detach().numpy()
        feature_rests = gaussians._features_rest.detach().numpy()
        
        for xyz, opacity, scale, rot, normal, feature, feature_rest in zip(xyzs, opacitys, scales, rots, normal, features, feature_rests):            
            self.add(
                mean=xyz.tolist(),
                scale=scale.tolist(),
                sigmat=float(opacity.item()),
                quaternion=rot.tolist(),
                feature=np.concatenate((feature * 0.0, feature_rest * 0.0), axis=0).tolist(),
                
                normal=normal.tolist(),
            )

        return self.build()
