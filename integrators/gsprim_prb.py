import gc
import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import mis_weight

from .reparam import ReparamIntegrator

class GaussianPrimitivePrbIntegrator(ReparamIntegrator):
    def __init__(self, props):
        super().__init__(props)

    def aovs(self):
        return []

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aovs())
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            
            #hybrid trick
            mask_pt = sampler.next_1d() < self.pt_rate
            
            L, valid, _aovs, _= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
                depth=mi.UInt32(0), δL=None, δA=None, δR=None, δM=None, δD=None, δN=None, state_in=None, reparam=None, active=mi.Bool(mask_pt))
            
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

            L, valid, state_out, gradients = self.sample(
                mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
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
                δL = dr.grad(L)                     # ∂loss/∂RGB
                δA = dr.grad(albedo_state_out)      # ∂loss/∂Albedo
                δR = dr.grad(roughness_state_out)   # ∂loss/∂Roughness
                δM = dr.grad(metallic_state_out)    # ∂loss/∂Metallic
                δD = dr.grad(depth_state_out)       # ∂loss/∂Depth
                δN = dr.grad(normal_state_out)      # ∂loss/∂Normal

                #Convert ∂loss/∂RGB to ∂loss/∂A, ∂loss/∂R, ∂loss/∂M
                #δA = δL * gradients['albedo'] + δA  # ∂loss/∂RGB * ∂RGB/∂A + ∂loss/∂A = ∂loss/∂A 
                #δR = δL * gradients['roughness'] + δR
                #δM = δL * gradients['metallic'] + δM
                #δD = δL * gradients['depth'] + δD
                #δN = δN
            
            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_2, _= self.sample(
                mode=dr.ADMode.Backward, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
                δL=δL, δA=δA, δR=δR, δM=δM, δD=δD, δN=δN,
                state_in=state_out, active=mi.Bool(True))

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, δA, δR, δM, δD, δN, ray, weight, pos, block, extra_block, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, δA, δR, δM, δD, δN, state_in, active, **kwargs):
        
        primal = (mode == dr.ADMode.Primal)
        
        valid_ray = (not self.hide_emitters) and scene.environment() is not None

        # --------------------- Configure loop state ----------------------
        mask_pt = mi.Mask(active)
        active = mi.Mask(active)
        
        depth = mi.UInt32(0)

        result = mi.Spectrum(0.0)
        L = mi.Spectrum(0 if primal else state_in['result'])
        δL = mi.Spectrum(δL if δL is not None else 0)

        aov_A, aov_R, aov_M, aov_D, aov_N = mi.Spectrum(0.0), mi.Spectrum(0.0), mi.Spectrum(0.0), mi.Spectrum(0.0), mi.Spectrum(0.0)

        β = mi.Spectrum(1)
        mis_em = mi.Float(1)

        ray_prev = dr.zeros(mi.Ray3f)
        ray_cur = mi.Ray3f(ray)
        
        si_prev = dr.zeros(mi.SurfaceInteraction3f)

        A_prev = mi.Spectrum(0.0)
        R_prev = mi.Float(0.0)
        M_prev = mi.Float(0.0)
        D_prev = mi.Float(0.0)
        N_prev = mi.Spectrum(0.0)

        # hybrid trick (just for render)
        if primal:   
            L_wo_rt, A_wo_rt, R_wo_rt, M_wo_rt, D_wo_rt, N_wo_rt = self.ray_marching_loop_wo_rt(scene, ray, ~mask_pt)
            result[~mask_pt] += L_wo_rt
            
            aov_A[~mask_pt] += self.safe_clamp(A_wo_rt, 0.0, 1.0)
            aov_R[~mask_pt] += self.safe_clamp(R_wo_rt, 0.0, 1.0)
            aov_M[~mask_pt] += self.safe_clamp(M_wo_rt, 0.0, 1.0)
            aov_N[~mask_pt] += self.safe_normalize(N_wo_rt)
            aov_D[~mask_pt] += D_wo_rt

        # ray tracing
        A_cur_raw, R_cur_raw, M_cur_raw, D_cur_raw, N_cur_raw, si_valid, weight_acc = self.ray_marching_loop(scene, sampler, True, ray_cur, δA, δR, δM, δD, δN, state_in, active)    
        
        state_cur = {
            'albedo': dr.select(active, A_cur_raw, 0.0),
            'roughness': dr.select(active, mi.Spectrum(R_cur_raw), 0.0),
            'metallic': dr.select(active, mi.Spectrum(M_cur_raw), 0.0),
            'depth': dr.select(active, mi.Spectrum(D_cur_raw), 0.0),
            'normal': dr.select(active, N_cur_raw, 0.0),
            'weight_acc': dr.select(active, weight_acc, 0.0)
        }
        
        A_cur = self.safe_clamp(A_cur_raw, 0.0, 1.0)
        R_cur = self.safe_clamp(R_cur_raw, 0.0, 1.0)
        M_cur = self.safe_clamp(M_cur_raw, 0.0, 1.0)
        N_cur = self.safe_normalize(N_cur_raw)
        D_cur = D_cur_raw

        si_cur = self.SurfaceInteraction3f(ray_cur, D_cur, N_cur, si_valid)
        
        valid_ray |= si_cur.is_valid() # output mask
        
        #aov & state_outs
        aov_A[mask_pt] += dr.select(active, A_cur, 0.0)
        aov_R[mask_pt] += dr.select(active, mi.Spectrum(R_cur), 0.0)
        aov_M[mask_pt] += dr.select(active, mi.Spectrum(M_cur), 0.0)
        aov_N[mask_pt] += dr.select(active, N_cur, 0.0)
        aov_D[mask_pt] += dr.select(active, D_cur, 0.0)

        aovs = {
            'albedo': aov_A,
            'roughness': aov_R,
            'metallic': aov_M,
            'depth': aov_D,
            'normal': aov_N
        }

        active_prev = mi.Bool(active)
        while dr.hint(active, max_iterations=self.max_depth, label="Path Replay Backpropagation (%s)" % mode.name):
            first_vertex = mi.Bool(depth == 0)
            active_next = mi.Bool(active)
            
            if not primal:
                with dr.resume_grad():
                    dr.enable_grad(A_cur, R_cur, M_cur, D_cur, N_cur)
                    dr.disable_grad(si_prev)
            
            with dr.resume_grad(when=not primal):      
                Le = β * mis_em * si_cur.emitter(scene).eval(si_cur)
           
            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid()
            # Next event estimation
            active_em = mi.Bool(active_next)
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(active_em), False, active_em)
            active_em &= (ds.pdf != 0.0)

            with dr.resume_grad(when= not primal):
                em_ray = si_cur.spawn_ray(ds.d)
                em_ray.d = dr.detach(em_ray.d)

                em_ray_valid = dr.dot(N_cur, em_ray.d) > 0.0
                occluded = self.shadow_ray_test(scene, sampler, em_ray, active_em & em_ray_valid)

                visibility = dr.select(~occluded, 1.0, 0.0)
                active_em &= ~occluded
                
                if not primal:
                    ds.d = em_ray.d
                    em_val = scene.eval_emitter_direction(dr.detach(si_cur), ds, active_em)
                    em_weight = dr.select((ds.pdf != 0) & ~occluded, em_val / ds.pdf, 0)

                #eval pdf of the ray in bsdf sampling
                Ldirection = em_ray.d
                Vdirection = dr.normalize(-ray_cur.d) #view direction (outgoing) 
                Halfvector = dr.normalize(Ldirection + Vdirection)
                bsdf_value_em, bsdf_pdf_em = self.eval_bsdf(A_cur, R_cur, M_cur, N_cur, Vdirection, Ldirection, Halfvector)
                mis_direct = dr.detach(mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = visibility * β * mis_direct * bsdf_value_em * em_weight
                
            bsdf_val, bsdf_dir, bsdf_pdf = self.bsdf(sampler, si_cur, A_cur, R_cur, M_cur, N_cur, Vdirection) #get bsdf attributes
            bsdf_weight = dr.select(bsdf_pdf > 0.0, bsdf_val / bsdf_pdf, 0.0)

            active_next &= (bsdf_pdf > 0.0)
            β *= mi.Spectrum(bsdf_weight)
            L_prev = L 

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
                
            # render direct illumination
            # L += dr.select((depth == 1), Le, 0.0)
            # L += dr.select(first_vertex, Lr_dir, 0.0)

            # # render indirect illumination
            # L += dr.select((depth == 1), 0.0, Le)
            # L += dr.select(first_vertex, 0.0, Lr_dir)
                 
            # Intersect next surface
            ray_next = self.next_ray(scene, si_cur, bsdf_dir, active_next) # set offset to avoid self-occ
            ray_next_valid = dr.dot(N_cur, ray_next.d) > 0.0
            active_next &= ray_next_valid

            A_next_raw, R_next_raw, M_next_raw, D_next_raw, N_next_raw, si_next_valid, weight_acc_next = self.ray_marching_loop(scene, sampler, True, ray_next, δA, δR, δM, δD, δN, state_in, active_next)
            
            state_next = {
                'albedo': dr.select(active, A_next_raw, 0.0),
                'roughness': dr.select(active, mi.Spectrum(R_next_raw), 0.0),
                'metallic': dr.select(active, mi.Spectrum(M_next_raw), 0.0),
                'depth': dr.select(active, mi.Spectrum(D_next_raw), 0.0),
                'normal': dr.select(active, N_next_raw, 0.0),
                'weight_acc': dr.select(active, weight_acc_next, 0.0)
            }

            A_next = self.safe_clamp(A_next_raw, 0.0, 1.0)
            R_next = self.safe_clamp(R_next_raw, 0.0, 1.0)
            M_next = self.safe_clamp(M_next_raw, 0.0, 1.0)
            N_next = self.safe_normalize(N_next_raw)
            D_next = D_next_raw
            
            si_next = self.SurfaceInteraction3f(ray_next, D_next, N_next, si_next_valid)

            # Compute MIS weight for the next vertex
            ds = mi.DirectionSample3f(scene, si=si_next, ref=si_cur)
            em_pdf = scene.pdf_emitter_direction(ref=si_cur, ds=ds, active=active_next)
            mis_em = dr.detach(mis_weight(bsdf_pdf, em_pdf))

            if not primal:
                sampler_clone = sampler.clone()
                active_next_next = mi.Bool(active_next) & si_next.is_valid() & (depth + 2 < self.max_depth)

                # Lr_dir_next
                active_em_next = mi.Bool(active_next_next)
                ds_next, em_weight_next = scene.sample_emitter_direction(si_next, sampler_clone.next_2d(), False, active_em_next)      
                active_em_next &= (ds_next.pdf != 0.0)
                
                em_ray_next = si_next.spawn_ray(ds_next.d)
                Ldirection_next = em_ray_next.d
                Vdirection_next = dr.normalize(-ray_next.d)
                Halfvector_next = dr.normalize(Ldirection_next + Vdirection_next)
                bsdf_next_val, bsdf_next_pdf = self.eval_bsdf(A_next, R_next, M_next, N_next, Vdirection_next, Ldirection_next, Halfvector_next)

                mis_direct_next = mis_weight(ds_next.pdf, bsdf_next_pdf)
                Lr_dir_next = β * mis_direct_next * bsdf_next_val * em_weight_next
                
                # Generate a detached BSDF sample at the next vertex
                bsdf_dir_next, _ = self.sample_bsdf(sampler_clone, si_next, R_next, M_next, Vdirection_next)

                with dr.resume_grad(si_cur.p):
                    wo_prev = dr.normalize(si_cur.p - si_prev.p)
                    wi_next = dr.normalize(si_cur.p - si_next.p)

                    si_next.wi = si_next.to_local(wi_next)
                    Le_next = β * mis_em * si_next.emitter(scene).eval(si_next, active_next)
                    L_next = L - dr.detach(Le_next) - dr.detach(Lr_dir_next)

                    # prev bsdf val
                    Ldirection_prev = wo_prev
                    Vdirection_prev = dr.normalize(-ray_prev.d)
                    Halfvector_prev = dr.normalize(Ldirection_prev + Vdirection_prev)
                    bsdf_prev_val, _ = self.eval_bsdf(A_prev, R_prev, M_prev, N_prev, Vdirection_prev, Ldirection_prev, Halfvector_prev)

                    # next bsdf val
                    Ldirection_next = bsdf_dir_next
                    Vdirection_next = dr.normalize(-ray_next.d)
                    Halfvector_next = dr.normalize(Ldirection_next + Vdirection_next)
                    bsdf_next_val, _ = self.eval_bsdf(A_next, R_next, M_next, N_next, Vdirection_next, Ldirection_next, Halfvector_next)

                    extra = mi.Spectrum(Le_next)
                    extra[~first_vertex] += L_prev * bsdf_prev_val / dr.detach(bsdf_prev_val)
                    extra[si_next.is_valid()] += L_next * bsdf_next_val / dr.detach(bsdf_next_val)

                with dr.resume_grad():
                    # cur bsdf val
                    bsdf_val_det = dr.detach(bsdf_weight * bsdf_pdf)
                    inv_bsdf_val_det = dr.select((bsdf_val_det != 0), dr.rcp(bsdf_val_det), 0)
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    Lo = (Le + Lr_dir + Lr_ind) + extra
                    Lo[depth > self.max_depth] = 0
                    
                    dr.backward_from(Lo)

                    δA_cur, δR_cur, δM_cur, δD_cur, δN_cur = map(dr.grad, (A_cur, R_cur, M_cur, D_cur, N_cur))

                δA = dr.select(first_vertex, δA + δA_cur * δL, δA_cur * δL) # ∂loss/∂RGB * ∂RGB/∂A + ∂loss/∂A = ∂loss/∂A
                δR = dr.select(first_vertex, δR + δR_cur * δL, δR_cur * δL)
                δM = dr.select(first_vertex, δM + δM_cur * δL, δM_cur * δL)
                δD = dr.select(first_vertex, δD + δD_cur * δL, δD_cur * δL)
                δN = dr.select(first_vertex, δN + δN_cur * δL, δN_cur * δL)

                self.ray_marching_loop(scene, sampler_clone, False, ray_cur, δA, δR, δM, δD, δN, state_cur, active_prev)

            depth[si_cur.is_valid()] += 1
            
            state_cur = dr.detach(state_next)
            active_prev = mi.Bool(active)
            active = mi.Bool(active_next)
            si_prev, ray_prev = map(dr.detach, (si_cur, ray_cur))
            si_cur, ray_cur = map(dr.detach, (si_next, ray_next))
            A_prev, R_prev, M_prev, D_prev, N_prev = map(dr.detach, (A_cur, R_cur, M_cur, D_cur, N_cur))
            A_cur, R_cur, M_cur, D_cur, N_cur = map(dr.detach,(A_next, R_next, M_next, D_next, N_next))

        result[mask_pt] += dr.select(valid_ray, mi.Spectrum(L), 0.0)
        aovs['result'] = result

        gradients = {}
        
        return result, True, aovs, gradients

    def to_string(self):
        return f"GaussianPrimitivePrbIntegrator[]"
    
mi.register_integrator("gsprim_prb", lambda props: GaussianPrimitivePrbIntegrator(props))