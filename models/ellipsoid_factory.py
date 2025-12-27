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
        # Extract all attributes as numpy arrays in batch
        num_gaussians = gaussians._xyz.shape[0]
        
        xyzs = gaussians._xyz.detach().numpy()
        opacitys = gaussians._opacity.detach().numpy()
        scales = gaussians._scaling.detach().numpy()
        rots = gaussians._rotation.detach().numpy()
        normals = gaussians._normal.detach().numpy()
        features_dc = gaussians._features_dc.detach().numpy()
        feature_rests = gaussians._features_rest.detach().numpy()
        albedos = gaussians._albedo.detach().numpy()
        roughnesses = gaussians._roughness.detach().numpy()
        metallics = gaussians._metallic.detach().numpy()
        
        # Batch concatenate features
        features = np.concatenate((features_dc, feature_rests), axis=1)
        
        # Convert to Mitsuba tensors directly in batch
        centers = mi.TensorXf(xyzs.reshape(num_gaussians, 3))
        scales = mi.TensorXf(scales.reshape(num_gaussians, 3))
        quats = mi.TensorXf(rots.reshape(num_gaussians, 4))
        normals_tensor = mi.TensorXf(normals.reshape(num_gaussians, 3))
        sigmats = mi.TensorXf(opacitys.reshape(num_gaussians, 1))
        features_tensor = mi.TensorXf(features.reshape(num_gaussians, -1))
        albedos_tensor = mi.TensorXf(albedos.reshape(num_gaussians, -1))
        roughnesses_tensor = mi.TensorXf(roughnesses.reshape(num_gaussians, 1))
        metallics_tensor = mi.TensorXf(metallics.reshape(num_gaussians, 1))
        
        attributes = {
            'centers': centers,
            'scales': scales,
            'quats': quats,
            'normals': normals_tensor,
            'sigmats': sigmats,
            'features': features_tensor,
            'albedos': albedos_tensor,
            'roughnesses': roughnesses_tensor,
            'metallics': metallics_tensor
        }
        
        return attributes