import gc
import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator

class GaussianPrimitiveRadianceFieldIntegrator(ReparamIntegrator):
    def __init__(self, props):
        super().__init__(props)

    def aovs(self):
        return []

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aovs())
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            L, valid, _aovs= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
                depth=mi.UInt32(0), δL=None, δA=None, δR=None, δM=None, δD=None, δN=None, state_in=None, reparam=None, active=mi.Bool(True))
            
            #color
            block = sensor.film().create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))
            block.put(pos, ray.wavelengths, L * weight, alpha)
            sensor.film().put_block(block)
            self.primal_image = sensor.film().develop()

            #extra aovs
            self.extra_images = {}
            for aov_name, aov_value in _aovs.items():
                if(dr.shape(aov_value)[0]==3):
                    sensor.film().clear()
                    extra_block = sensor.film().create_block() 
                    extra_block.set_coalesce(extra_block.coalesce() and spp >= 4)
                    
                    aovs = [None]*3
                    aovs[0] = aov_value[0]
                    aovs[1] = aov_value[1]
                    aovs[2] = aov_value[2]
                    extra_block.put(pos, ray.wavelengths, aovs * weight)

                    sensor.film().put_block(extra_block)
                    current_image = sensor.film().develop()

                    self.extra_images[aov_name] = current_image

            del sampler, ray, weight, pos, L, valid, _aovs, alpha
            gc.collect()

            return self.primal_image, self.extra_images

    def render_backward(self, scene, params, grad_in, sensor=0, seed=0, spp=0):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        
        film = sensor.film()
        aovs = self.aovs()

        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp, aovs)
            ray, weight, pos, det = self.sample_rays(scene, sensor, sampler)

            L, valid, state_out = self.sample(
                mode=dr.ADMode.Primal, scene=scene, sampler=sampler.clone(), ray=ray, depth=mi.UInt32(0), 
                δL=None, δA=None, δR=None, δM=None, δD=None, δN=None, 
                state_in=None, reparam=None, active=mi.Bool(True))
            
            albedo_state_out = state_out['albedo']
            roughness_state_out = state_out['roughness']
            metallic_state_out = state_out['metallic']
            depth_state_out = state_out['depth']
            normal_state_out = state_out['normal']

            with dr.resume_grad():
                film.clear()
                dr.enable_grad(L)
                block = film.create_block()
                block.set_coalesce(block.coalesce() and spp >= 4)
                block.put(pos=pos, wavelengths=ray.wavelengths,
                        value=L * weight * det, weight=det, alpha=dr.select(valid, mi.Float(1), mi.Float(0)))
                film.put_block(block)
                dr.schedule(L, block.tensor())
                color_image = film.develop()
                dr.set_grad(color_image, grad_in[0])
                dr.enqueue(dr.ADMode.Backward, color_image)

                for data_name, extra_data in zip(['albedo', 'roughness', 'metallic', 'depth', 'normal'], [albedo_state_out, roughness_state_out, metallic_state_out, depth_state_out, normal_state_out]):
                    film.clear()
                    dr.enable_grad(extra_data)
                    extra_block = film.create_block()
                    extra_block.set_coalesce(extra_block.coalesce() and spp >= 4)
                    extra_block.put(pos, ray.wavelengths, extra_data)
                    dr.schedule(extra_data, extra_block.tensor())
                    film.put_block(extra_block)
                    extra_image = film.develop() 
                    dr.set_grad(extra_image, grad_in[1][data_name])
                    dr.enqueue(dr.ADMode.Backward, extra_image)
                
                dr.traverse(dr.ADMode.Backward)
                δL = dr.grad(L)                     # ∂loss/∂L
                δA = dr.grad(albedo_state_out) 
                δR = dr.grad(roughness_state_out) 
                δM = dr.grad(metallic_state_out) 
                δD = dr.grad(depth_state_out)
                δN = dr.grad(normal_state_out)

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_2 = self.sample(
                mode=dr.ADMode.Backward, scene=scene, sampler=sampler,
                ray=ray, depth=mi.UInt32(0), δL=δL, δA=δA, δR=δR, δM=δM, δD=δD, δN=δN,
                state_in=state_out, active=mi.Bool(True))

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, δA, δR, δM, δD, δN, ray, weight, pos, block, extra_block, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()


    @dr.syntax
    def ray_marching_loop(self, mode, scene, sampler, primal, ray, δL, δA, δR, δM, δD, δN, state_in, active):
        
        num = mi.UInt32(0)
        active = mi.Mask(active)

        ray = mi.Ray3f(dr.detach(ray)) #clone a new ray

        A = mi.Spectrum(0.0 if primal else state_in['albedo'])
        R = mi.Float(0.0 if primal else state_in['roughness'][0])
        M = mi.Float(0.0 if primal else state_in['metallic'][0])
        D = mi.Float(0.0 if primal else state_in['depth'][0])
        N = mi.Spectrum(0.0 if primal else state_in['normal'])
        weight_acc = mi.Float(0.0 if primal else state_in['weight_acc'][0])

        δA = mi.Spectrum(δA if δA is not None else 0)
        δR = mi.Spectrum(δR if δR is not None else 0)
        δM = mi.Spectrum(δM if δM is not None else 0)
        δD = mi.Spectrum(δD if δD is not None else 0)
        δN = mi.Spectrum(δN if δN is not None else 0)

        β = mi.Spectrum(1.0) #for rgb
        T = mi.Float(1.0) #for aovs

        with dr.resume_grad(when=not primal):
            if not primal:
                Vdirection = state_in['Vdirection']
                Ldirection = state_in['Ldirection']
                Halfvector = dr.normalize(Ldirection + Vdirection) 

                dr.enable_grad(A)
                dr.enable_grad(R)
                dr.enable_grad(N)
                #dr.enable_grad(M)

                cosθ = dr.clamp(dr.dot(N, Ldirection), 1e-6, 1.0)
                BSDF = self.eval_bsdf(A, R, M, N, Vdirection, Ldirection, Halfvector)
                BSDF = BSDF * cosθ

                dr.backward_from(BSDF)
                
                δA = δL * dr.grad(A) + δA 
                δR = δL * dr.grad(R)
                #δM = δL * dr.grad(M)
                δN = δL * dr.grad(N) + δN

        depth_acc = mi.Float(0.0)
        while dr.hint(active, label=f"BSDF ray tracing"):
            si_cur = scene.ray_intersect(ray)
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

                weight = T * (1.0 - transmission)

                albedo = β * (1.0 - transmission) * albedo_val
                albedo[~dr.isfinite(albedo)] = 0.0

                roughness = weight * roughness_val
                roughness[~dr.isfinite(roughness)] = 0.0

                metallic = weight * metallic_val
                metallic[~dr.isfinite(metallic)] = 0.0

                depth[active] = weight * depth_acc
                depth[~dr.isfinite(depth)] = 0.0

                normal[active] = weight * normals_val
                normal[~dr.isfinite(normal)] = 0.0

            A[active] = (A + albedo) if primal else (A - albedo)
            R[active] = (R + roughness) if primal else (R - roughness)
            M[active] = (M + metallic) if primal else (M - metallic)

            D[active] = (D + depth) if primal else (D - depth)
            N[active] = (N + normal) if primal else (N - normal)
            weight_acc[active]= (weight_acc + weight) if primal else (weight_acc - weight)

            β[active] *= transmission
            T[active] *= transmission

            ray.o[active] = si_cur.p + ray.d * 1e-4

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

                    if mode == dr.ADMode.Backward:
                        loss = δA * Ao + δR * Ro + δM * Mo + δD * Do + δN * No
                        dr.backward_from(loss)
                    else:
                        δA += dr.forward_to(Ao)
                        δR += dr.forward_to(Ro)
                        δM += dr.forward_to(Mo)
                        δD += dr.forward_to(Do)
                        δN += dr.forward_to(No)
            
            active &= si_cur.is_valid()
            num[active] += 1

            β_max = dr.max(β)

            active &= β_max > 0.01
            active &= num < self.max_depth

            sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            if primal and num >= 10:
                rr_prob = dr.maximum(β_max, 0.1)
                rr_active = β_max < 0.1
                β[rr_active] *= dr.rcp(rr_prob)
                rr_continue = sample_rr < rr_prob
                active &= ~rr_active | rr_continue

        D = D / dr.maximum(weight_acc, 1e-8)
        N = dr.normalize(N)
        
        return A, R, M, D, N, (T < 0.9) , weight_acc

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, δA, δR, δM, δD, δN, state_in, active, **kwargs):
        
        primal = (mode == dr.ADMode.Primal)
        
        active = mi.Mask(active)
        result = mi.Spectrum(0.0)

        A, R, M, D, N, active, weight_acc = self.ray_marching_loop(mode, scene, sampler, primal, ray, δL, δA, δR, δM, δD, δN, state_in, active)

        #create a new si as gaussian intersection
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.sh_frame.n = N
        si.initialize_sh_frame() 
        si.n = si.sh_frame.n
        si.wi = -ray.d
        si.p = ray.o + ray.d * D
        si.wavelengths = ray.wavelengths

        #visualize the emitter
        if not self.hide_emitters:
            result += si.emitter(scene).eval(si) 
        
        #-----------eval emitter-----------
        with dr.suspend_grad():
            ds, _ = scene.sample_emitter_direction(si, sampler.next_2d(active), False, active)
            active &= (ds.pdf != 0.0)
            shadow_ray = si.spawn_ray_to(ds.p)

            occluded = self.ray_test(scene, sampler, shadow_ray, active)

            si_e = dr.zeros(mi.SurfaceInteraction3f)
            si_e.sh_frame.n = ds.n
            si_e.initialize_sh_frame()
            si_e.n = si_e.sh_frame.n
            si_e.wi = -shadow_ray.d
            si_e.wavelengths = ray.wavelengths
            emitter_val = dr.select(active, ds.emitter.eval(si_e, active), 0.0)

            emitter_val = dr.select(ds.pdf > 0, emitter_val / ds.pdf, 0.0)
            visibility = dr.select(~occluded, 1.0, 0.0)

        #-----------eval bsdf-----------
        Ldirection = shadow_ray.d # view direction (outgoing)
        Vdirection = dr.normalize(-ray.d) # light direction (incoming)
        Halfvector = dr.normalize(Ldirection + Vdirection) # half-vector

        cosθ = dr.clamp(dr.dot(N, Ldirection), 1e-6, 1.0)
        BSDF = self.eval_bsdf(A, R, M, N, Vdirection, Ldirection, Halfvector)
        BSDF = BSDF * cosθ
        
        #output
        nee_contrib = visibility * BSDF * emitter_val
        result[active] = nee_contrib

        #aov & state_out
        aovs = {
            'albedo': dr.select(active, A, 0.0),
            'roughness': dr.select(active, mi.Spectrum(R), 0.0),
            'metallic': dr.select(active, mi.Spectrum(M), 0.0),
            'depth': dr.select(active, mi.Spectrum(D), 0.0),
            'normal': dr.select(active, N, 0.0),
            
            'weight_acc': dr.select(active, weight_acc, 0.0),
            'Vdirection': dr.select(active, Vdirection, 0.0),
            'Ldirection': dr.select(active, Ldirection, 0.0)
        }

        return mi.Spectrum(result), True, aovs

    def to_string(self):
        return f"GaussianPrimitiveRadianceFieldIntegrator[]"
    
mi.register_integrator("gsprim_rf", lambda props: GaussianPrimitiveRadianceFieldIntegrator(props))