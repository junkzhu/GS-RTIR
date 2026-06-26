import gc
import drjit as dr
import mitsuba as mi

from models import DisneyBSDF

PI = dr.pi
EPS = 1e-8

class ReparamIntegrator(mi.SamplingIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 4)
        self.pt_rate = props.get('pt_rate', 1.0)
        self.gaussian_max_depth = props.get('gaussian_max_depth', 128)
        self.hide_emitters = props.get('hide_emitters', False)
        self.use_mis = props.get('use_mis', False)
        
        # self-occlusion method
        self.selfocc_offset_max = props.get('selfocc_offset_max', -1)
        self.selfocc_mode = props.get('selfocc_mode', 'normal')
        self.selfocc_offset_fixed = props.get('selfocc_offset_fixed', 0.1)  # used when selfocc_mode == 'fixed'
        
        self.geometry_threshold = props.get('geometry_threshold', 0.5)
        self.separate_direct_indirect = props.get('separate_direct_indirect', False)
        
        self.optimize_mesh = props.get('optimize_mesh', False)

        self._bsdf = DisneyBSDF()

        if self.selfocc_offset_max < 0:
            self.selfocc_offset_max = float(1e8)

    
    def safe_normalize(self, v):
        n = dr.norm(v)
        n_safe = dr.maximum(n, 1e-8)
        v_normalized = v / n_safe
        return dr.replace_grad(v_normalized, v)
    
    def safe_clamp(self, x, min_val=0.0, max_val=1.0):
        clamped = dr.clamp(x, min_val, max_val)
        return dr.replace_grad(clamped, x)

    def SurfaceInteraction3f(self, ray, D, N, valid = True, offset = 0.0):
        #create a new si as gaussian intersection
        if isinstance(D, mi.Spectrum):
            D = D[0]
        
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.sh_frame = mi.Frame3f(N)

        si.n = N
        si.wi = -ray.d
        si.t = dr.select((D > 0) & valid, D, si.t)
        si.p = ray.o + (1 - offset) * ray.d * D
        si.wavelengths = ray.wavelengths
        return si

    def prepare_film(self, sensor, aovs=[]):
        film = sensor.film()
        film.prepare(aovs)

    def prepare(self, sensor, seed, spp):
        film = sensor.film()
        sampler = sensor.sampler().clone()
        if spp != 0:
            sampler.set_sample_count(spp)
        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)
        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()
        wavefront_size = dr.prod(film_size) * spp

        wavefront_size_limit = 0xffffffff if dr.is_jit_v(mi.Float) else 0x40000000
        if wavefront_size > wavefront_size_limit:
            raise Exception(f"Wavefront {wavefront_size} exceeds {wavefront_size_limit}")
        sampler.seed(seed, wavefront_size)
        return sampler, spp
    
    def sample_rays(self, scene, sensor, sampler, reparam=None):
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray

        When a reparameterization function is provided via the 'reparam'
        argument, it will be applied to the returned image-space position (i.e.
        the sample positions will be moving). The other two return values
        remain detached.
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2u()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(mi.Int(-film_size[0]), pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += film.crop_offset()

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        ray, weight = sensor.sample_ray_differential(time=wavelength_sample, sample1=sampler.next_1d(),
                                                     sample2=pos_adjusted,sample3=aperture_sample)
        det = mi.Float(1.0)
        if reparam is not None:
            assert not rfilter.is_box_filter()
            assert film.sample_border()

            with dr.resume_grad():
                reparam_d, det = reparam(ray=ray, depth=mi.UInt32(0))

                # Create a fake interaction along the sampled ray and use it to the
                # position with derivative tracking
                it = dr.zeros(mi.Interaction3f)
                it.p = ray.o + reparam_d
                ds, _ = sensor.sample_direction(it, aperture_sample)
                # Return a reparameterized image position
                pos_f = ds.uv + film.crop_offset()

        return ray, weight, pos_f, det

    @dr.syntax
    def is_self_occlusion(self, origin_pos, si_hit, ray, active):
        """True where the hit is self-occlusion: (si_hit.p - origin_pos.p) dot normal > 0; normal from eval_normal at hit."""
        
        dist = dr.norm(si_hit.p - origin_pos.p)
        close_hit = dist < 0.02

        normal = self.eval_normal(si_hit, ray, active)
        return (close_hit | (dr.dot(origin_pos.n, normal) >= 0.0)) & active

    @dr.syntax
    def next_ray(self, scene, si, dir, offset, active):
        ray = mi.Ray3f(si.spawn_ray(dir))
        ray.o = ray.o + ray.d * offset

        active = mi.Mask(active)

        while dr.hint(active, label=f"Ray Start Test"):
            si_cur = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            active_gs = active & si_cur.is_valid() & si_cur.shape.is_ellipsoids()

            if self.selfocc_mode == 'normal':
                dist = dr.norm(si_cur.p - si.p)
                within_max = dist <= self.selfocc_offset_max
                self_occ = self.is_self_occlusion(si, si_cur, ray, active_gs) & within_max
                # Only continue for rays that hit a Gaussian and are self-occlusion; others exit (avoid infinite loop when ray misses)
                active = active_gs & self_occ
            else:
                active = active_gs

            ray.o[active] = si_cur.p + ray.d * 1e-4  

        return ray

    @dr.syntax
    def ray_intersect(self, scene, sampler, ray, active):
        ray = mi.Ray3f(ray)
        
        # Ray first intersects with 3D Gaussians
        A_raw, R_raw, M_raw, D_raw, N_raw, hit_valid, ray_valid, weight_acc = self.ray_marching_loop(scene, sampler, True, ray, None, None, None, None, None, None, active)    

        state_out = {
            'albedo': dr.select(hit_valid, A_raw, 0.0),
            'roughness': dr.select(hit_valid, R_raw, 0.0),
            'metallic': dr.select(hit_valid, M_raw, 0.0),
            'depth': dr.select(hit_valid, D_raw, 0.0),
            'normal': dr.select(hit_valid, N_raw, 0.0),
            'weight_acc': dr.select(hit_valid, weight_acc, 0.0),
            'hit_valid': hit_valid
        }

        A_gs = self.safe_clamp(A_raw, 0.0, 1.0)
        R_gs = self.safe_clamp(R_raw, 0.0, 1.0)
        M_gs = self.safe_clamp(M_raw, 0.0, 1.0)
        N_gs = self.safe_normalize(N_raw)
        D_gs = D_raw

        # If ray doesn't intersect with 3D Gaussians, it's still valid and continues to check mesh intersection;
        # If ray directly intersects mesh in Gaussian intersection logic, it doesn't affect transmittance, ray remains valid and will still execute this.
        ray_o, ray_d = dr.copy(ray.o), dr.copy(ray.d)
        si_mesh = scene.ray_intersect(ray, active)
        
        # If ray intersects GS first, execute mesh detection logic, skip surface GS
        mesh_detection_active = ~hit_valid & si_mesh.is_valid() & si_mesh.shape.is_ellipsoids()
        while dr.hint(mesh_detection_active, label=f"Mesh Intersect Skip Gaussians"):
            ray.o = si_mesh.p + ray.d * 1e-6
            si_mesh = scene.ray_intersect(ray, mesh_detection_active)
            mesh_detection_active &= (si_mesh.is_valid() & si_mesh.shape.is_ellipsoids())

        mesh_active = ~hit_valid & si_mesh.is_valid() & ~si_mesh.shape.is_ellipsoids() & ~si_mesh.shape.is_emitter()

        A_mesh = si_mesh.bsdf(ray).eval_diffuse_reflectance(si_mesh, mesh_active)
        R_mesh = si_mesh.bsdf(ray).eval_attribute_1("roughness", si_mesh, mesh_active)
        M_mesh = si_mesh.bsdf(ray).eval_attribute_1("metallic", si_mesh, mesh_active)
        N_mesh = dr.select(mesh_active, mi.Spectrum(si_mesh.n), mi.Spectrum(0.0))
        D_mesh = dr.select(mesh_active, mi.Float(dr.dot(si_mesh.p - ray_o, ray_d)), mi.Float(0.0))

        # Identify the first valid intersection point
        hit_valid |= mesh_active

        # Determine validity of next ray bounce
        ray_valid &= ~mesh_active

        # Unified Gbuffer output
        A = dr.select(~mesh_active, A_gs, A_mesh)
        R = dr.select(~mesh_active, R_gs, R_mesh)
        M = dr.select(~mesh_active, M_gs, M_mesh)
        N = dr.select(~mesh_active, N_gs, N_mesh)
        D = dr.select(~mesh_active, D_gs, D_mesh)

        return A, R, M, N, D, hit_valid, ray_valid, state_out

    @dr.syntax
    def shadow_ray_test(self, scene, sampler, pos, ray, active):
        #Stochastic Ray Tracing of Transparent 3D Gaussians, 3.3 section
        active=mi.Mask(active)
        ray = mi.Ray3f(dr.detach(ray))
    
        occluded = ~active
        while dr.hint(active, label=f"Shadow Ray Test"):
            si_cur = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            
            # Hybrid rendering supplement: if intersects with mesh and intersection is not a light source, directly treat as occluded
            intersect_mesh = si_cur.is_valid() & ~si_cur.shape.is_ellipsoids() & ~si_cur.shape.is_emitter()
            occluded[active & intersect_mesh] = mi.Bool(True)
            active &= ~intersect_mesh

            # Gaussian part
            active_gs = active & si_cur.is_valid() & si_cur.shape.is_ellipsoids()

            self_occ = mi.Bool(False)
            if self.selfocc_mode == 'normal':
                if pos is not None:
                    dist = dr.norm(si_cur.p - pos.p)
                    within_max = dist <= self.selfocc_offset_max
                    self_occ = self.is_self_occlusion(pos, si_cur, ray, active_gs) & within_max
                    ray.o[self_occ] = si_cur.p + ray.d * 1e-4
                    active_gs &= ~self_occ
            # else: strategy 'fixed' — no is_self_occlusion, all Gaussian hits go to transmission test below
             
            transmission = self.eval_transmission(si_cur, ray, active_gs)
            alpha = 1.0 - transmission # opacity as a probability

            rand = sampler.next_1d()
            hit_occluded = rand < alpha
            occluded[active_gs & hit_occluded] = mi.Bool(True)
            active = (active_gs & ~hit_occluded) | self_occ
            ray.o[active] = si_cur.p + ray.d * 1e-4

        return occluded

    #-------------------- 3DGS --------------------
    @dr.syntax
    def ray_intersect_emitter(self, scene, ray, active):
        with dr.suspend_grad():
            ray = mi.Ray3f(ray)
            si = scene.ray_intersect(ray, active)

            emitter_detection_active = active & si.is_valid() & si.shape.is_ellipsoids()
            while dr.hint(emitter_detection_active, label=f"Skip Gaussians"):
                ray.o = si.p + ray.d * 1e-6
                si = scene.ray_intersect(ray, emitter_detection_active)
                emitter_detection_active &= (si.is_valid() & si.shape.is_ellipsoids())

        return si

    def eval_sh_emission(self, si, ray, active):
        """
        Evaluate the SH directionally emission on intersected volumetric primitives
        """
        def eval(shape, si, ray, active):
            if shape is not None and shape.is_ellipsoids():
                sh_coeffs = shape.eval_attribute_x("sh_coeffs", si, active)
                sh_degree = int(dr.sqrt((sh_coeffs.shape[0] // 3) - 1))
                sh_dir_coef = dr.sh_eval(ray.d, sh_degree)
                emission = mi.Color3f(0.0)
                for i, sh in enumerate(sh_dir_coef):
                    emission += sh * mi.Color3f(
                        [sh_coeffs[i * 3 + j] for j in range(3)]
                    )
                return dr.maximum(emission + 0.5, 0.0)
            else:
                return mi.Color3f(0.0)

        return dr.dispatch(si.shape, eval, si, ray, active)

    def eval_transmission(self, si, ray, active):
        """
        Evaluate the transmission model on intersected volumetric primitives
        """
        def gather_ellipsoids_props(self, prim_index, active):
            if self is not None and self.is_ellipsoids():
                si = dr.zeros(mi.SurfaceInteraction3f)
                si.prim_index = prim_index
                data = self.eval_attribute_x("ellipsoid", si, active)
                center  = mi.Point3f([data[i] for i in range(3)])
                scale   = mi.Vector3f([data[i + 3] for i in range(3)])
                quat    = mi.Quaternion4f([data[i + 6] for i in range(4)])
                rot     = dr.quat_to_matrix(quat, size=3)
                return center, scale, rot
            else:
                return mi.Point3f(0), mi.Vector3f(0), mi.Matrix3f(0)

        center, scale, rot = dr.dispatch(si.shape, gather_ellipsoids_props, si.prim_index, active)

        opacity = si.shape.eval_attribute_1("opacities", si, active)

        # Gaussian splatting transmittance model
        # Find the peak location along the ray, from "3D Gaussian Ray Tracing"
        o = rot.T * (ray.o - center) / scale
        d = rot.T * ray.d / scale
        t_peak = -dr.dot(o, d) / dr.dot(d, d)
        p_peak = ray(t_peak)

        # Gaussian kernel evaluation
        p = rot.T * (p_peak - center)
        density = dr.exp(-0.5 * (p.x**2 / scale.x**2 + p.y**2 / scale.y**2 + p.z**2 / scale.z**2))

        return 1.0 - dr.minimum(opacity * density, 0.9999)
    
    def eval_transmission_w_dist(self, si, ray, active):
        """
        Evaluate the transmission model on intersected volumetric primitives
        """
        def gather_ellipsoids_props(self, prim_index, active):
            if self is not None and self.is_ellipsoids():
                si = dr.zeros(mi.SurfaceInteraction3f)
                si.prim_index = prim_index
                data = self.eval_attribute_x("ellipsoid", si, active)
                center  = mi.Point3f([data[i] for i in range(3)])
                scale   = mi.Vector3f([data[i + 3] for i in range(3)])
                quat    = mi.Quaternion4f([data[i + 6] for i in range(4)])
                rot     = dr.quat_to_matrix(quat, size=3)
                return center, scale, rot
            else:
                return mi.Point3f(0), mi.Vector3f(0), mi.Matrix3f(0)

        center, scale, rot = dr.dispatch(si.shape, gather_ellipsoids_props, si.prim_index, active)

        opacity = si.shape.eval_attribute_1("opacities", si, active)

        # Gaussian splatting transmittance model
        # Find the peak location along the ray, from "3D Gaussian Ray Tracing"
        o = rot.T * (ray.o - center) / scale
        d = rot.T * ray.d / scale
        t_peak = -dr.dot(o, d) / dr.dot(d, d)
        p_peak = ray(t_peak)

        # Gaussian kernel evaluation
        p = rot.T * (p_peak - center)
        density = dr.exp(-0.5 * (p.x**2 / scale.x**2 + p.y**2 / scale.y**2 + p.z**2 / scale.z**2))

        # Get influence distance
        dist = dr.norm(p_peak - si.p) * 2 

        return 1.0 - dr.minimum(opacity * density, 0.9999), dist

    def eval_normal(self, si, ray, active):
        def eval(shape, si, ray, active):
            if shape is not None and shape.is_ellipsoids():
                normals = shape.eval_attribute_3("normals", si, active)
                normals = dr.normalize(mi.Vector3f(normals))
                return normals
            else:
                return mi.Vector3f(0.0)
        return dr.dispatch(si.shape, eval, si, ray, active)
            
    def eval_bsdf_component(self, si, ray, active):
        def eval(shape, si, ray, active):
            if shape is not None and shape.is_ellipsoids():
                normals = shape.eval_attribute_3("normals", si, active)
                normals = mi.Vector3f(normals)

                albedos = shape.eval_attribute_3("albedos", si, active)
                albedos = dr.maximum(albedos, 0.0)

                roughnesses = shape.eval_attribute_1("roughnesses", si, active)
                roughnesses = dr.clamp(roughnesses, 0.0, 1.0)

                metallics = shape.eval_attribute_1("metallics", si, active)
                metallics = dr.clamp(metallics, 0.0, 1.0)

                # high_rough_mask = roughnesses > 0.8
                # red_color = mi.Color3f(1.0, 0.0, 0.0)
                # default_color = mi.Color3f(0.0, 0.0, 0.0)
                # albedos = dr.select(high_rough_mask, red_color, default_color)

                return normals, albedos, roughnesses, metallics
            else:
                return mi.Vector3f(0.0), mi.Color3f(0.0), mi.Float(0.0), mi.Float(0.0)

        return dr.dispatch(si.shape, eval, si, ray, active)

    @dr.syntax
    def ray_marching_loop_wo_rt(self, scene, ray, active):
        #copy from volprim_rf_basic     
        num = mi.UInt32(0)
        active = mi.Mask(active)

        ray = mi.Ray3f(dr.detach(ray))

        L = mi.Spectrum(0.0)
        A = mi.Spectrum(0.0)
        R = mi.Float(0.0)
        M = mi.Float(0.0)
        D = mi.Float(0.0)
        N = mi.Spectrum(0.0)
        weight_acc = mi.Float(0.0)

        T = mi.Float(1.0)
        β = mi.Spectrum(1.0)
        depth_acc = mi.Float(0.0)
        
        while dr.hint(active, label="Primitive splatting"):
            si = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            active &= si.is_valid() & si.shape.is_ellipsoids()

            depth_acc += dr.select(active, si.t, 0.0)

            Le = mi.Spectrum(0.0)
            depth = mi.Float(0.0)
            normal = mi.Spectrum(0.0)
            albedo = mi.Spectrum(0.0)
            roughness = mi.Float(0.0)
            metallic = mi.Float(0.0)
            weight = mi.Float(0.0)

            emission = self.eval_sh_emission(si, ray, active)
            normals_val, albedo_val, roughness_val, metallic_val = self.eval_bsdf_component(si, ray, active)
            transmission = self.eval_transmission(si, ray, active)
            
            Le[active] = β * (1.0 - transmission) * emission
            Le[~dr.isfinite(Le)] = 0.0

            #valid_gs = dr.dot(ray.d, normals_val) < 0.0
            #weight = dr.select(valid_gs, T * (1.0 - transmission), 0.0)

            weight = T * (1.0 - transmission)

            albedo = weight * albedo_val
            albedo[~dr.isfinite(albedo)] = 0.0

            roughness = weight * roughness_val
            roughness[~dr.isfinite(roughness)] = 0.0

            metallic = weight * metallic_val
            metallic[~dr.isfinite(metallic)] = 0.0

            depth = weight * depth_acc
            depth[~dr.isfinite(depth)] = 0.0

            normal = weight * normals_val
            normal[~dr.isfinite(normal)] = 0.0

            L[active] = (L + Le)
            A[active] = (A + albedo)
            R[active] = (R + roughness)
            M[active] = (M + metallic)
            D[active] = (D + depth)
            N[active] = (N + normal)
            weight_acc[active]= (weight_acc + weight)
            
            β[active] *= transmission
            #T[active] *= dr.select(valid_gs, transmission, 1.0)
            T[active] *= transmission

            ray.o[active] = si.p + ray.d * 1e-4
 
            depth_acc[active] += 1e-4

            active &= si.is_valid()
            num[active] += 1

            active &= T > 0.01
            active &= num < self.gaussian_max_depth

        L = mi.math.srgb_to_linear(L)
        D = D / dr.maximum(weight_acc, 1e-8)
        N = N / dr.maximum(weight_acc, 1e-8)

        return L, A, R, M, D, N

    @dr.syntax
    def ray_marching_loop(self, scene, sampler, primal, ray, δA, δR, δM, δD, δN, state_in, active):
        
        num = mi.UInt32(0)
        active = mi.Mask(active)

        ray = mi.Ray3f(dr.detach(ray)) #clone a new ray

        A = mi.Spectrum(0.0 if primal else state_in['albedo'])
        R = mi.Float(0.0 if primal else state_in['roughness'])
        M = mi.Float(0.0 if primal else state_in['metallic'])
        D = mi.Float(0.0 if primal else state_in['depth'])
        N = mi.Spectrum(0.0 if primal else state_in['normal'])
        weight_acc = mi.Float(0.0 if primal else state_in['weight_acc'])

        δA = mi.Spectrum(δA if δA is not None else 0)
        δR = mi.Float(δR if δR is not None else 0)
        δM = mi.Float(δM if δM is not None else 0)
        δD = mi.Float(δD if δD is not None else 0)
        δN = mi.Spectrum(δN if δN is not None else 0)

        T = mi.Float(1.0)

        depth_acc = mi.Float(0.0)
        while dr.hint(active, label=f"BSDF ray tracing"):
            si_cur = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            active &= si_cur.is_valid() & si_cur.shape.is_ellipsoids()

            depth_acc += dr.select(active, si_cur.t, 0.0)

            depth = mi.Float(0.0)
            normal = mi.Spectrum(0.0)

            albedo = mi.Spectrum(0.0)
            roughness = mi.Float(0.0)
            metallic = mi.Float(0.0)
            
            weight = mi.Float(0.0)
            with dr.resume_grad(when=not primal):
                normals_val, albedo_val, roughness_val, metallic_val = self.eval_bsdf_component(si_cur, ray, active)
                transmission, dist = self.eval_transmission_w_dist(si_cur, ray, active)
                
                #valid_gs = dr.dot(ray.d, normals_val) < 0.0
                #weight = dr.select(valid_gs, T * (1.0 - transmission), 0.0)

                weight = T * (1.0 - transmission)
                weight_det = dr.detach(weight)

                albedo = weight_det * albedo_val
                albedo[~dr.isfinite(albedo)] = 0.0

                roughness = weight_det * roughness_val
                roughness[~dr.isfinite(roughness)] = 0.0

                metallic = weight_det * metallic_val
                metallic[~dr.isfinite(metallic)] = 0.0

                normal = weight_det * normals_val
                normal[~dr.isfinite(normal)] = 0.0

                depth = weight * depth_acc
                depth = dr.select(dr.isfinite(depth), depth, 0.0)

            A[active] = (A + albedo) if primal else (A - albedo / weight_acc)
            R[active] = (R + roughness) if primal else (R - roughness / weight_acc)
            M[active] = (M + metallic) if primal else (M - metallic / weight_acc)

            D[active] = (D + depth) if primal else (D - depth / weight_acc)
            N[active] = (N + normal) if primal else (N - normal / weight_acc)
            weight_acc[active]= (weight_acc + weight) if primal else (weight_acc - weight)

            #T[active] *= dr.select(valid_gs, transmission, 1.0)
            T[active] *= transmission

            ray.o[active] = si_cur.p + ray.d * 1e-4
            depth_acc[active] += 1e-4

            with dr.resume_grad(when=not primal):
                if not primal:
                    Ar_ind = A * transmission / dr.detach(transmission)
                    Rr_ind = R * transmission / dr.detach(transmission)
                    Mr_ind = M * transmission / dr.detach(transmission)
                    Dr_ind = D * transmission / dr.detach(transmission)
                    Nr_ind = N * transmission / dr.detach(transmission)
                    
                    Ao = albedo + Ar_ind
                    Ro = roughness + Rr_ind
                    Mo = metallic + Mr_ind
                    Do = depth + Dr_ind
                    No = normal + Nr_ind

                    LA = δA * Ao
                    LR = δR * Ro
                    LM = δM * Mo
                    LD = δD * Do
                    LN = δN * No

                    LA = dr.select(active & dr.isfinite(LA), LA, 0.0)
                    LR = dr.select(active & dr.isfinite(LR), LR, 0.0)
                    LM = dr.select(active & dr.isfinite(LM), LM, 0.0)
                    LD = dr.select(active & dr.isfinite(LD), LD, 0.0)
                    LN = dr.select(active & dr.isfinite(LN), LN, 0.0)

                    loss = LA + LR + LM + LD + LN
                    dr.backward_from(loss)
            
            active &= si_cur.is_valid()
            num[active] += 1

            active &= (T > 0.01)
            active &= num < self.gaussian_max_depth

            # sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            # if primal and num >= 10:
            #     rr_prob = dr.maximum(β_max, 0.1)
            #     rr_active = β_max < 0.1
            #     β[rr_active] *= dr.rcp(rr_prob)
            #     rr_continue = sample_rr < rr_prob
            #     active &= ~rr_active | rr_continue

        A = A / dr.maximum(weight_acc, 1e-8)
        R = R / dr.maximum(weight_acc, 1e-8)
        D = D / dr.maximum(weight_acc, 1e-8)
        N = N / dr.maximum(weight_acc, 1e-8)

        # Trick: srgb -> linear is much easier to optimize, follow 3dgs
        #R = mi.math.srgb_to_linear(R) 
        #A = mi.math.srgb_to_linear(A)

        # Hybrid visibility decision: 
        # 1) Use Monte Carlo sampling with transmittance T to decide whether the ray continues.
        #    If rand < T, the ray is allowed to pass through; otherwise, it is treated as blocked.
        # 2) Use the accumulated geometric weight as a high-confidence geometric prior.
        #    If weight_acc exceeds the threshold, the current sample is considered a solid object hit.
        # 3) The final hit state is the union of:
        #    - geometrically confident solid hits, and
        #    - Monte Carlo blocked hits.
        # 4) Rays are only allowed to continue if they pass the Monte Carlo test
        #    and are not classified as solid by the geometric criterion.
        
        # Abalation study: Geometry Judgement

        rand = sampler.next_1d()
        ray_active = (rand < T)

        solid_hit = (weight_acc > self.geometry_threshold)
        mc_hit = ~ray_active

        hit_active = solid_hit | mc_hit
        ray_active &= ~hit_active

        return A, R, M, D, N, hit_active, ray_active, weight_acc

    @dr.syntax
    def inject_mesh_gradient(self, scene, ray, δA_in, δR_in, δM_in, δD_in, δN_in, state, active):           
        ray = mi.Ray3f(ray)
        si_mesh = scene.ray_intersect(ray, active)

        δA = mi.Spectrum(δA_in)
        δR = mi.Float(δR_in)
        δM = mi.Float(δM_in)

        mesh_detection_active = ~state['hit_valid'] & si_mesh.is_valid() & si_mesh.shape.is_ellipsoids()
        while dr.hint(mesh_detection_active, label=f"Mesh Intersect Skip Gaussians"):
            ray.o = si_mesh.p + ray.d * 1e-6
            si_mesh = scene.ray_intersect(ray, mesh_detection_active)
            mesh_detection_active &= (si_mesh.is_valid() & si_mesh.shape.is_ellipsoids())

        mesh_active = ~state['hit_valid'] & si_mesh.is_valid() & ~si_mesh.shape.is_ellipsoids() & ~si_mesh.shape.is_emitter()

        with dr.resume_grad():
            bsdf = si_mesh.bsdf(ray)
            A_mesh = bsdf.eval_diffuse_reflectance(si_mesh, mesh_active)
            R_mesh = bsdf.eval_attribute_1("roughness", si_mesh, mesh_active)
            M_mesh = bsdf.eval_attribute_1("metallic", si_mesh, mesh_active)
            
            Ao = A_mesh * δA
            Ro = R_mesh * δR
            Mo = M_mesh * δM
            
            Ao = dr.select(mesh_active & dr.isfinite(Ao), Ao, 0.0)
            Ro = dr.select(mesh_active & dr.isfinite(Ro), Ro, 0.0)
            Mo = dr.select(mesh_active & dr.isfinite(Mo), Mo, 0.0)

            dr.backward_from(Ao + Ro + Mo)

    #-------------------- BSDF --------------------
    def fresnel_schlick(self, F0, cosTheta):
        # F0: Color3f, cosTheta: scalar or array in [0,1]
        # Schlick approximation: F = F0 + (1-F0)*(1-cosTheta)^5
        c = dr.clamp(1.0 - cosTheta, 0.0, 1.0)
        c5 = dr.power(c, 5.0)
        return F0 + (1.0 - F0) * c5

    def ggx_D(self, N, H, roughness):
        α = roughness * roughness
        α2 = α * α

        NdotH = dr.clamp(dr.dot(N, H), 0.0, 1.0)
        NdotH2 = NdotH * NdotH
        denom = NdotH2 * (α2 - 1.0) + 1.0
        D = α2 / (PI * denom * denom + EPS)

        return D

    def smith_G1(self, N, V, alpha):
        # G1 for GGX (Heitz) ; alpha is roughness^2
        NdotV = dr.clamp(dr.dot(N, V), 0.0, 1.0)
        # avoid division by 0
        a = alpha
        # common stable form:
        tmp = dr.sqrt(a * a + (1.0 - a * a) * (NdotV * NdotV))
        G1 = 2.0 * NdotV / (NdotV + tmp + EPS)
        return G1

    def ggx_G(self, N, V, L, roughness):
        alpha = roughness * roughness
        G1V = self.smith_G1(N, V, alpha)
        G1L = self.smith_G1(N, L, alpha)
        return G1V * G1L

    def eval_bsdf(self, albedo, roughness, metallic, N, V, L, H):
        return self._bsdf.eval_bsdf(albedo, roughness, metallic, N, V, L, H)
    
    def sample_bsdf(self, sampler, si, roughness, metallic, V_world):
        return self._bsdf.sample_bsdf(sampler, si, roughness, metallic, V_world)
    
    def bsdf(self, sampler, si, albedo, roughness, metallic, N, Vdir):
        return self._bsdf.bsdf(sampler, si, albedo, roughness, metallic, N, Vdir)