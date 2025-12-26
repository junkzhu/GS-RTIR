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
        roughnesses = mi.TensorXf(np.array(self.roughnesses))
        metallics = mi.TensorXf(np.array(self.metallics))
        
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
        albedo = gaussians._albedo.detach().numpy()
        roughness = gaussians._roughness.detach().numpy()
        metallic = gaussians._metallic.detach().numpy()
        
        for xyz, opacity, scale, rot, normal, feature, feature_rest, albedo, roughness, metallic in zip(xyzs, opacitys, scales, rots, normal, features, feature_rests, albedo, roughness, metallic):            
            self.add(
                mean=xyz.tolist(),
                scale=scale.tolist(),
                sigmat=float(opacity.item()),
                quaternion=rot.tolist(),
                feature=np.concatenate((feature, feature_rest), axis=0).tolist(),
                
                normal=normal.tolist(),
                albedo=albedo.tolist(),
                roughness=roughness.tolist(),
                metallic=metallic.tolist()
            )

        return self.build()
