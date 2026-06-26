import gc
import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import mis_weight

from .reparam import ReparamIntegrator

class GaussianPrimitivePrbIntegrator(ReparamIntegrator):
    def __init__(self, props):
        super().__init__(props)
        rr_depth       = int(props.get('rr_depth', 2))
        self.rr_depth  = mi.UInt32(rr_depth if rr_depth > 0 else 2**32-1)
        self.use_rr = rr_depth < self.max_depth

    def aovs(self):
        return [
            "alpha",
            "albedo.x",
            "albedo.y",
            "albedo.z",
            "roughness",
            "metallic",
            "direct.x",
            "direct.y",
            "direct.z",
            "indirect.x",
            "indirect.y",
            "indirect.z",
            "normal.x",
            "normal.y",
            "normal.z",
            "depth",
        ]

    def pack_values(self, L, alpha, aovs, weight):
        rgb = L * weight
        aovs['direct_light'] *= weight
        aovs['indirect_light'] *= weight

        return [
            rgb[0],
            rgb[1],
            rgb[2],
            mi.Float(1.0), #have calculated in the sample function

            alpha,
            aovs['albedo'][0],
            aovs['albedo'][1],
            aovs['albedo'][2],
            aovs['roughness'],
            aovs['metallic'],
            aovs['direct_light'][0],
            aovs['direct_light'][1],
            aovs['direct_light'][2],
            aovs['indirect_light'][0],
            aovs['indirect_light'][1],
            aovs['indirect_light'][2],
            aovs['normal'][0],
            aovs['normal'][1],
            aovs['normal'][2],
            aovs['depth'],
        ]

    def render_process(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp)
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
                        
            L, valid, aovs, _= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
                depth=mi.UInt32(0), δL=None, δA=None, δR=None, δM=None, δD=None, δN=None, state_in=None, reparam=None, active=mi.Bool(True))

            #color
            block = sensor.film().create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)
            
            alpha = dr.select(valid, mi.Float(0.0), mi.Float(1.0))

            values = self.pack_values(L, alpha, aovs, weight)

            block.put(pos, values, True)
            sensor.film().put_block(block)

            del L, aovs, ray, weight, alpha, pos, block, sampler, values
            gc.collect()

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp)
            self.prepare_film(sensor=sensor, aovs=self.aovs())
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
                        
            L, valid, aovs, _= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
                depth=mi.UInt32(0), δL=None, δA=None, δR=None, δM=None, δD=None, δN=None, state_in=None, reparam=None, active=mi.Bool(True))
            
            #color
            block = sensor.film().create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)
            
            alpha = dr.select(valid, mi.Float(0.0), mi.Float(1.0))
            values = self.pack_values(L, alpha, aovs, weight)

            block.put(pos, values, True)
            sensor.film().put_block(block)
            
            self.buffer = sensor.film().develop()
            del sampler, spp, ray, weight, pos, L, valid, aovs, block, alpha, values, sensor
            gc.collect()

            return self.buffer

    def render_backward(self, scene, params, grad_in, sensor=0, seed=0, spp=0):        
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        
        film = sensor.film()
        aovs = self.aovs()

        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp)
            self.prepare_film(sensor=sensor, aovs=aovs)
            ray, weight, pos, det = self.sample_rays(scene, sensor, sampler)

            L, valid, aovs, gradients = self.sample(
                mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
                δL=None, δA=None, δR=None, δM=None, δD=None, δN=None, 
                state_in=None, reparam=None, active=mi.Bool(True))
            
            with dr.resume_grad():
                film.clear()
                
                dr.enable_grad(L)
                dr.enable_grad(aovs)

                block = film.create_block()
                block.set_coalesce(block.coalesce() and spp >= 4)

                weight = weight * det
                alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                
                values = self.pack_values(L, alpha, aovs, weight)

                block.put(pos, values, True)

                film.put_block(block)

                del valid, alpha, values, weight
                gc.collect()

                dr.schedule(L, aovs, block.tensor())
                buffer = film.develop()
                
                dr.set_grad(buffer, grad_in)
                dr.enqueue(dr.ADMode.Backward, buffer)
                
                dr.traverse(dr.ADMode.Backward)
                δL = dr.grad(L)            # ∂loss/∂RGB
                δaovs = dr.grad(aovs)      # ∂loss/∂Attribute
            
            # Launch Monte Carlo sampling in backward AD mode (2)
            self.sample(
                mode=dr.ADMode.Backward, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
                δL=δL, δA=δaovs['albedo'], δR=δaovs['roughness'], δM=δaovs['metallic'], δD=δaovs['depth'], δN=δaovs['normal'],
                state_in=aovs, active=mi.Bool(True))

            # We don't need any of the outputs here
            del L, aovs, δL, δaovs, ray, pos, block, buffer, sampler, grad_in, det, spp
            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, δA, δR, δM, δD, δN, state_in, active, **kwargs):
        
        primal = (mode == dr.ADMode.Primal)
        
        valid_ray = mi.Mask(active)

        # --------------------- Configure loop state ----------------------
        active = mi.Mask(active)
        
        depth = mi.UInt32(0)

        result = mi.Spectrum(0.0)
        L = mi.Spectrum(0 if primal else state_in['result'])
        δL = mi.Spectrum(δL if δL is not None else 0)

        aov_A, aov_R, aov_M, aov_D, aov_N = mi.Spectrum(0.0), mi.Float(0.0), mi.Float(0.0), mi.Float(0.0), mi.Spectrum(0.0)
        L_direct, L_indirect = mi.Spectrum(0.0), mi.Spectrum(0.0)

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

        # ray tracing
        A_cur, R_cur, M_cur, N_cur, D_cur, hit_valid, ray_valid, state_cur = self.ray_intersect(scene, sampler, ray_cur, active)
        active &= (hit_valid | ray_valid)

        si_cur = self.SurfaceInteraction3f(ray_cur, D_cur, N_cur, hit_valid)
        hit_valid &= si_cur.is_valid()
        
        valid_ray &= ray_valid # output mask
        
        #aov & state_outs
        aov_A += dr.select(hit_valid, A_cur, 0.0)
        aov_R += dr.select(hit_valid, R_cur, 0.0)
        aov_M += dr.select(hit_valid, M_cur, 0.0)
        aov_N += dr.select(hit_valid, N_cur, 0.0)
        aov_D += dr.select(hit_valid, D_cur, 0.0)

        aovs = {
            'albedo': aov_A,
            'roughness': aov_R,
            'metallic': aov_M,
            'depth': aov_D,
            'normal': aov_N
        }

        # show emitter
        if (not self.hide_emitters) and scene.environment() is not None:
            si_e = self.ray_intersect_emitter(scene, ray_cur, ray_valid)
            emitter = si_e.emitter(scene)            
            result += dr.select(ray_valid, emitter.eval(si_e), 0.0)

        active_prev = mi.Bool(active)
        while dr.hint(active, max_iterations=self.max_depth, label="Path Replay Backpropagation (%s)" % mode.name):
            first_vertex = mi.Bool(depth == 0)
            active_next = mi.Bool(active)
            mis_direct = 0.0
            
            if not primal:
                with dr.resume_grad():
                    dr.enable_grad(A_cur, R_cur, M_cur, D_cur, N_cur)
                    dr.disable_grad(si_prev)
            
            with dr.resume_grad(when=not primal):
                si_e = self.ray_intersect_emitter(scene, ray_cur, ray_valid)
                emitter = si_e.emitter(scene)
                    
                emitter_val = dr.select(ray_valid, emitter.eval(si_e), 0.0)
                Le = dr.select(first_vertex, 0.0, β * mis_em * emitter_val)
           
            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid()
            
            # Next event estimation
            active_em = mi.Bool(active_next)
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(active_em), False, active_em)
            active_em &= (ds.pdf != 0.0)

            with dr.resume_grad(when= not primal):
                em_ray = si_cur.spawn_ray(ds.d)
                em_ray.d = dr.detach(em_ray.d)

                if self.selfocc_mode == 'fixed':
                    fix_offset = self.selfocc_offset_fixed
                    em_ray.o = dr.detach(em_ray.o) + fix_offset * em_ray.d

                em_ray_valid = dr.dot(dr.detach(N_cur), em_ray.d) > 0.0
                occluded = self.shadow_ray_test(scene, sampler, si_cur, em_ray, active_em & em_ray_valid)

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
            
            # BSDF sampling  
            bsdf_val, bsdf_dir, bsdf_pdf = self.bsdf(sampler, si_cur, A_cur, R_cur, M_cur, N_cur, Vdirection) #get bsdf attributes
            bsdf_weight = dr.select(bsdf_pdf > 0.0, bsdf_val / bsdf_pdf, 0.0)

            active_next &= (bsdf_pdf > 0.0)
            β *= mi.Spectrum(bsdf_weight)
            L_prev = L 

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)

            if self.separate_direct_indirect:
                # render direct illumination
                L_direct += dr.select((depth == 1), dr.detach(Le), 0.0) + dr.select(first_vertex, dr.detach(Lr_dir), 0.0)
                # render indirect illumination
                L_indirect += dr.select((depth == 1), 0.0, dr.detach(Le)) + dr.select(first_vertex, 0.0, dr.detach(Lr_dir))
                 
            # Intersect next surface: fixed strategy = ray origin offset by selfocc_offset_fixed; normal strategy = no offset
            if self.selfocc_mode == 'fixed':
                occ_offset = self.selfocc_offset_fixed
            else:
                occ_offset = 0.0
            
            ray_next = self.next_ray(scene, si_cur, bsdf_dir, occ_offset, active_next)
            ray_next_valid = dr.dot(N_cur, ray_next.d) > 0.0
            active_next &= ray_next_valid

            A_next, R_next, M_next, N_next, D_next, hit_valid_next, ray_next_valid, state_next = self.ray_intersect(scene, sampler, ray_next, active_next)
            active_next &= (hit_valid_next | ray_next_valid)

            si_next = self.SurfaceInteraction3f(ray_next, D_next, N_next, hit_valid_next)
            hit_valid_next &= si_next.is_valid()

            # Compute MIS weight for the next vertex
            si_mis = dr.zeros(mi.SurfaceInteraction3f)
            si_mis.wi = -ray_next.d
            ds = mi.DirectionSample3f(scene, si=si_mis, ref=si_cur)
            
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
                    si_e_next = self.ray_intersect_emitter(scene, ray_next, ray_next_valid)
                    Le_next = β * mis_em * dr.select(ray_next_valid, si_e_next.emitter(scene).eval(si_e_next), 0.0)
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
                    
                    dr.backward_from(δL * Lo)

                    δA_cur, δR_cur, δM_cur, δD_cur, δN_cur = map(dr.grad, (A_cur, R_cur, M_cur, D_cur, N_cur))
                    
                # Small trick: convert spectrum gradient to float, then multiply by δL
                # δA_cur = δA_cur / dr.maximum(mis_direct, 1e-8)
                # δR_cur = δL * dr.sum(δR_cur/δL) / dr.maximum(mis_direct, 1e-8)
                # δM_cur = δL * dr.sum(δM_cur/δL) / dr.maximum(mis_direct, 1e-8)
                # δD_cur = δL * dr.sum(δD_cur/δL) / dr.maximum(mis_direct, 1e-8)

                δA_in = dr.select(first_vertex, δA_cur + δA, δA_cur) # ∂loss/∂RGB * ∂RGB/∂A + ∂loss/∂A = ∂loss/∂A
                δR_in = dr.select(first_vertex, δR_cur + δR, δR_cur)
                δM_in = dr.select(first_vertex, δM_cur + δM, δM_cur)
                δD_in = dr.select(first_vertex, δD_cur + δD, δD_cur)
                δN_in = dr.select(first_vertex, δN_cur + δN, δN_cur)

                self.ray_marching_loop(scene, sampler_clone, False, ray_cur, δA_in, δR_in, δM_in, δD_in, δN_in, state_cur, active_prev)
                
                if self.optimize_mesh:
                    self.inject_mesh_gradient(scene, ray_cur, δA_in, δR_in, δM_in, δD_in, δN_in, state_cur, active_prev)

            depth[si_cur.is_valid()] += 1
            
            # Perform russian roulette
            sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            if primal and self.use_rr:
                q = dr.minimum(dr.max(β), 0.99)
                perform_rr = (depth > self.rr_depth)
                active_next &= (sample_rr < q) | ~perform_rr
                β[perform_rr] = β * dr.rcp(q)
            active_next &= dr.any(β > 0.005)
            active_next &= dr.any((β != 0.0))

            # Set config for next iteration
            ray_valid = dr.detach(ray_next_valid)
            state_cur = dr.detach(state_next)
            active_prev = mi.Bool(active)
            active = mi.Bool(active_next)
            si_prev, ray_prev = map(dr.detach, (si_cur, ray_cur))
            si_cur, ray_cur = map(dr.detach, (si_next, ray_next))
            A_prev, R_prev, M_prev, D_prev, N_prev = map(dr.detach, (A_cur, R_cur, M_cur, D_cur, N_cur))
            A_cur, R_cur, M_cur, D_cur, N_cur = map(dr.detach,(A_next, R_next, M_next, D_next, N_next))

        result += dr.select(valid_ray, 0.0, mi.Spectrum(L))
        aovs['result'] = result
        
        if self.separate_direct_indirect:
            aovs['direct_light'] = dr.select(valid_ray, 0.0, mi.Spectrum(L_direct))
            aovs['indirect_light'] = dr.select(valid_ray, 0.0, mi.Spectrum(L_indirect))

        gradients = {}
        
        return result, valid_ray, aovs, gradients

    def to_string(self):
        return f"GaussianPrimitivePrbIntegrator[]"
    
mi.register_integrator("gsprim_prb", lambda props: GaussianPrimitivePrbIntegrator(props))