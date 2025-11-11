import gc
import drjit as dr
import mitsuba as mi

PI = dr.pi
EPS = 1e-8
CLOSE_THRESHOLD = 1e-3
MID_THRESHOLD = 3e-3

class ReparamIntegrator(mi.SamplingIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 4)
        self.gaussian_max_depth = props.get('gaussian_max_depth', 128)
        self.hide_emitters = props.get('hide_emitters', False)
        self.use_mis = props.get('use_mis', False)
    
    def SurfaceInteraction3f(self, ray, D, N, valid = True, offset = 0.0):
        #create a new si as gaussian intersection
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.sh_frame = mi.Frame3f(N)

        si.n = N
        si.wi = -ray.d
        si.t = dr.select((D > 0) & valid, D, si.t)
        si.p = ray.o + (1 - offset) * ray.d * D
        si.wavelengths = ray.wavelengths
        return si

    def prepare(self, sensor, seed, spp, aovs=[]):
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
        film.prepare(aovs)
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
    def next_ray(self, scene, si, dir, active):
        ray = mi.Ray3f(si.spawn_ray(dir))

        active = mi.Mask(active)
        o, d = ray.o, ray.d

        while dr.hint(active, label=f"Ray Start Test"):
            si_cur = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            active &= si_cur.is_valid() & si_cur.shape.is_ellipsoids()

            self_occ = mi.Bool(False)

            ray_dist = dr.abs(dr.dot(si_cur.p - o, d))

            self_occ[ray_dist < CLOSE_THRESHOLD] |= mi.Bool(True)

            N = self.eval_normal(si_cur, ray, active) 
            self_occ[ray_dist < MID_THRESHOLD] |= (dr.dot(N, ray.d) >= 0)

            active &= (self_occ & (ray_dist < MID_THRESHOLD))
            ray.o[active] = si_cur.p + ray.d * 1e-4  

        return ray

    @dr.syntax
    def shadow_ray_test(self, scene, sampler, ray, active):
        #Stochastic Ray Tracing of Transparent 3D Gaussians, 3.3 section
        active=mi.Mask(active)
        ray = mi.Ray3f(dr.detach(ray))
        o, d = ray.o, ray.d
    
        occluded = ~active
        while dr.hint(active, label=f"Shadow Ray Test"):
            si_cur = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            active &= si_cur.is_valid() & si_cur.shape.is_ellipsoids()

            self_occ = mi.Bool(False)

            ray_dist = dr.abs(dr.dot(si_cur.p - o, d))
            self_occ[ray_dist < CLOSE_THRESHOLD] |= mi.Bool(True)

            N = self.eval_normal(si_cur, ray, active) 
            self_occ[ray_dist < MID_THRESHOLD] |= (dr.dot(N, ray.d) >= 0)
             
            transmission = self.eval_transmission(si_cur, ray, active)
            alpha = 1.0 - transmission # opacity as a probability

            rand = sampler.next_1d()
            hit_occluded = rand < alpha
            occluded[active & hit_occluded & ~self_occ] = mi.Bool(True)
            active = active & ((~hit_occluded) | (hit_occluded & self_occ)) 
            ray.o[active] = si_cur.p + ray.d * 1e-4

        return occluded

    #-------------------- 3DGS --------------------
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
                normals = dr.normalize(mi.Vector3f(normals))

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
    def ray_marching_loop(self, scene, sampler, primal, ray, δA, δR, δM, δD, δN, state_in, active):
        
        num = mi.UInt32(0)
        active = mi.Mask(active)

        ray = mi.Ray3f(dr.detach(ray)) #clone a new ray

        A = mi.Spectrum(0.0 if primal else state_in['albedo'])
        R = mi.Float(0.0 if primal else state_in['roughness'][0])
        M = mi.Float(0.0 if primal else state_in['metallic'][0])
        D = mi.Float(0.0 if primal else state_in['depth'][0])
        N = mi.Spectrum(0.0 if primal else state_in['normal'])
        weight_acc = mi.Float(0.0 if primal else state_in['weight_acc'])

        δA = mi.Spectrum(δA if δA is not None else 0)
        δR = mi.Spectrum(δR if δR is not None else 0)
        δM = mi.Spectrum(δM if δM is not None else 0)
        δD = mi.Spectrum(δD if δD is not None else 0)
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
                transmission = self.eval_transmission(si_cur, ray, active)

                valid_gs = dr.dot(ray.d, normals_val) < 0.0

                weight = dr.select(valid_gs, T * (1.0 - transmission), 0.0)

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

            A[active] = (A + albedo) if primal else (A - albedo)
            R[active] = (R + roughness) if primal else (R - roughness)
            M[active] = (M + metallic) if primal else (M - metallic)

            D[active] = (D + depth) if primal else (D - depth)
            N[active] = (N + normal) if primal else (N - normal)
            weight_acc[active]= (weight_acc + weight) if primal else (weight_acc - weight)

            T[active] *= dr.select(valid_gs, transmission, 1.0)

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

                    Ao = dr.select(active & dr.isfinite(Ao), Ao, 0.0)
                    Ro = dr.select(active & dr.isfinite(Ro), Ro, 0.0)
                    Mo = dr.select(active & dr.isfinite(Mo), Mo, 0.0)
                    Do = dr.select(active & dr.isfinite(Do), Do, 0.0)
                    No = dr.select(active & dr.isfinite(No), No, 0.0)

                    loss = δA * Ao + δR * Ro + δM * Mo + δD * Do + δN * No
                    dr.backward_from(loss)
            
            active &= si_cur.is_valid()
            num[active] += 1

            active &= T > 0.01
            active &= num < self.gaussian_max_depth

            # sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            # if primal and num >= 10:
            #     rr_prob = dr.maximum(β_max, 0.1)
            #     rr_active = β_max < 0.1
            #     β[rr_active] *= dr.rcp(rr_prob)
            #     rr_continue = sample_rr < rr_prob
            #     active &= ~rr_active | rr_continue

        D = D / dr.maximum(weight_acc, 1e-8)
        N = dr.normalize(N)
        
        A = dr.clamp(A, 0.0, 1.0)
        R = dr.clamp(R, 0.05, 1.0)
        M = dr.clamp(M, 0.0, 1.0)

        #R = mi.math.srgb_to_linear(R) #TODO: 属性中存储的roughness如果是srgb空间的，优化中更容易收敛

        rand = sampler.next_1d()
        active = (rand < (1-T))

        return A, R, M, D, N, active , weight_acc

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
        """
        Evaluate BSDF Value
        """
        # Follow from https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        NdotL = dr.maximum(dr.dot(N, L), 0.0)
        NdotV = dr.maximum(dr.dot(N, V), 0.0)
        NdotH = dr.maximum(dr.dot(N, H), 0.0)
        VdotH = dr.maximum(dr.dot(V, H), 0.0)

        # --- Fresnel base F0 mix (dielectric 0.04 vs albedo for metal) ---
        F0_dielectric = mi.Color3f(0.04)
        # ensure types broadcast correctly: metallic may be scalar or per-item
        F0 = dr.lerp(F0_dielectric, albedo, metallic)  # Color3f

        # Fresnel term (Schlick)
        F = self.fresnel_schlick(F0, VdotH)  # Color3f

        # --- D, G ---
        D = self.ggx_D(N, H, roughness)  # scalar per item
        G = self.ggx_G(N, V, L, roughness)  # scalar per item

        # Specular numerator (Color) = D * G * F
        spec_num = F * D * G  # Color3f (broadcast D,G)

        denom = 4.0 * dr.detach(NdotV * NdotL + 1e-8)  # scalar
        specular = spec_num / denom  # Color3f

        F_avg = (F[0] + F[1] + F[2]) / 3.0
        diffuse = (1.0 - metallic) * (albedo / PI) * (1.0 - F_avg)

        bsdf_val = specular + diffuse

        # The emitter is just a envmap, so we need to add cosθ here
        cosθ = dr.maximum(dr.dot(N, L), 0.0)
        bsdf_val = bsdf_val * cosθ

        # ---------- PDF ----------
        diffuse_prob  = (1.0 - metallic) * 0.5
        specular_prob = 1.0 - diffuse_prob
        #Sepcular
        pdf_H = D * dr.maximum(0.0, NdotH)
        pdf_spec = pdf_H / (4.0 * dr.maximum(1e-4, VdotH))
        #Diffuse
        cosθ = dr.maximum(dr.dot(N, L), 0.0)
        pdf_diff = dr.select(cosθ > 0, cosθ * dr.rcp(dr.pi), 0.0)

        bsdf_pdf = specular_prob * pdf_spec + diffuse_prob * pdf_diff

        return bsdf_val, bsdf_pdf
    
    def sample_bsdf(self, sampler, si, roughness, metallic, V_world):
        """
        Sample BSDF direction (Disney simplified: Diffuse + GGX Specular)
        Input: V_world (view dir in world space, pointing away from surface)
        Output: L_world (sampled direction in world space), pdf
        """
        #TODO: 当法线未收敛时，存在V和N反向的可能性，会出现预期之外的错误。（复现方法 GS法线属性初始化为(0,0,-1)）

        # Local coordinates
        V = dr.normalize(si.to_local(V_world))
        N = mi.Vector3f(0.0, 0.0, 1.0) # local normal

        # randoms
        r1 = sampler.next_1d()
        r2 = sampler.next_1d()

        diffuse_prob  = (1.0 - metallic) * 0.5
        specular_prob = 1.0 - diffuse_prob
        choose_specular = (r1 < specular_prob)

        # ---------- Specular ----------
        u1_spec = r1
        u2_spec = r2
        alpha = roughness * roughness

        # https://schuttejoe.github.io/post/ggximportancesamplingpart1/
        phi_h = 2.0 * dr.pi * u1_spec
        cos_theta_h = dr.sqrt((1.0 - u2_spec) / (1.0 + (alpha * alpha - 1.0) * u2_spec))
        sin_theta_h = dr.sqrt(dr.maximum(0.0, 1.0 - cos_theta_h * cos_theta_h))
        H = mi.Vector3f(sin_theta_h * dr.cos(phi_h),
                        sin_theta_h * dr.sin(phi_h),
                        cos_theta_h)

        VdotH = dr.dot(V, H)
        L_spec = dr.normalize(2.0 * VdotH * H - V)

        NdotH = dr.clamp(H.z, 0.0, 1.0)
        D = self.ggx_D(N, H, roughness)
        pdf_H = D * dr.maximum(0.0, NdotH)
        pdf_spec = pdf_H / (4.0 * dr.maximum(1e-4, VdotH))

        # ---------- Diffuse ----------
        u1_diff = r1
        u2_diff = r2

        phi = 2.0 * dr.pi * u1_diff
        cos_theta = dr.sqrt(1.0 - u2_diff)
        sin_theta = dr.sqrt(u2_diff)
        L_diff = mi.Vector3f(sin_theta * dr.cos(phi),
                            sin_theta * dr.sin(phi),
                            cos_theta)

        pdf_diff = cos_theta / dr.pi

        # --- Merge branches ------
        L_local   = dr.select(choose_specular, L_spec, L_diff)
        pdf = dr.select(choose_specular, pdf_spec, pdf_diff)

        # To world
        L_world = si.to_world(L_local)
        pdf = dr.select(dr.dot(si.n, L_world) > 0, pdf, 0.0)

        return L_world, pdf
    
    def bsdf(self, sampler, si, albedo, roughness, metallic, N, Vdir):
        Ldir, pdf0 = self.sample_bsdf(sampler, si, roughness, metallic, Vdir)

        Halfvector = dr.normalize(Ldir + Vdir)
        val, pdf1 = self.eval_bsdf(albedo, roughness, metallic, N, Vdir, Ldir, Halfvector)

        bsdf_pdf = pdf1 # pdf0 == pdf1 
        bsdf_dir = Ldir
        bsdf_val = dr.select(bsdf_pdf > 0.0, val, 0.0)
        
        return bsdf_val, bsdf_dir, bsdf_pdf