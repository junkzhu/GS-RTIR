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
            L, valid, _aovs, _= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
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

            L, valid, state_out, gradients = self.sample(
                mode=dr.ADMode.Forward, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
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
                δA = δL * gradients['albedo'] + δA  # ∂loss/∂RGB * ∂RGB/∂A + ∂loss/∂A = ∂loss/∂A 
                δR = δL * gradients['roughness'] + δR
                δM = δL * gradients['metallic'] + δM
                δD = δL * gradients['depth'] + δD
                δN = δN

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
        forward = (mode == dr.ADMode.Forward)
        
        valid_ray = (not self.hide_emitters) and scene.environment() is not None

        # --------------------- Configure loop state ----------------------
        active = mi.Bool(active)
        
        depth = mi.UInt32(0)
        L = mi.Spectrum(0)
        β = mi.Spectrum(1)
        mis_em = mi.Float(1)

        ray_prev = dr.zeros(mi.Ray3f)
        ray_cur = mi.Ray3f(ray)
        
        si_prev = dr.zeros(mi.SurfaceInteraction3f)
        #pi info
        A, R, M, D, N, si_valid, weight_acc = self.ray_marching_loop(scene, sampler, (primal|forward), ray_cur, δA, δR, δM, δD, δN, state_in, active)
        valid_ray |= si_valid    

        with dr.resume_grad(when=not primal):
            si_cur = self.SurfaceInteraction3f(ray_cur, D, N, si_valid)

        while dr.hint(active, max_iterations=self.max_depth, label="Path Replay Backpropagation (%s)" % mode.name):
            active_next = mi.Bool(active)
            
            with dr.resume_grad(when=not primal):         
                Le = β * mis_em * si_cur.emitter(scene).eval(si_cur, active_next)

            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid()

            # Next event estimation
            active_em = active_next & si_valid
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(), False, active_em)
            active_em &= (ds.pdf != 0.0)

            with dr.resume_grad(when= not primal):
                em_ray = si_cur.spawn_ray_to(ds.p)
                em_ray.d = dr.detach(em_ray.d)
                occluded = self.ray_test(scene, sampler, em_ray, active_em)
                active_em &= ~occluded
                
                #eval pdf of the ray in bsdf sampling
                Ldirection = em_ray.d
                Vdirection = dr.normalize(-ray_cur.d) #view direction (outgoing) 
                Halfvector = dr.normalize(Ldirection + Vdirection)
                bsdf_value_em, bsdf_pdf_em = self.eval_bsdf(A, R, M, N, Vdirection, Ldirection, Halfvector)
                mis_direct = mis_weight(ds.pdf, bsdf_pdf_em)
                Lr_dir = β * dr.detach(mis_direct) * bsdf_value_em * em_weight

            bsdf_val, bsdf_dir, bsdf_pdf = self.bsdf(sampler, si_cur, A, R, M, N, Vdirection) #get bsdf attributes
            bsdf_weight = bsdf_val / dr.maximum(1e-8, bsdf_pdf)
            β *= mi.Spectrum(bsdf_weight)
            L_prev = L
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)

            # Intersect next surface
            ray_next = si_cur.spawn_ray(bsdf_dir)
            A, R, M, D, N, si_next_valid, weight_acc = self.ray_marching_loop(scene, sampler,(primal|forward), ray_next, δA, δR, δM, δD, δN, state_in, active_next)
            si_next = self.SurfaceInteraction3f(ray_next, D, N, si_next_valid)

            # Compute MIS weight for the next vertex
            ds = mi.DirectionSample3f(scene, si=si_next, ref=si_cur)
            pdf_em = scene.pdf_emitter_direction(ref=si_cur, ds=ds, active=si_next_valid)
            mis_em = mis_weight(bsdf_pdf, pdf_em)

            if not primal:
                si_prev = si_cur
                ray_prev = ray_cur
            si_cur = si_next
            ray_cur = ray_next
            depth[si_cur.is_valid()] += 1
            active = active_next

        result = L
        gradients = {}
        aovs = {
            'albedo': dr.select(active, A, 0.0),
            'roughness': dr.select(active, mi.Spectrum(R), 0.0),
            'metallic': dr.select(active, mi.Spectrum(M), 0.0),
            'depth': dr.select(active, mi.Spectrum(D), 0.0),
            'normal': dr.select(active, N, 0.0)
        }

        return dr.select(valid_ray, mi.Spectrum(result), dr.detach(0.0)), True, aovs, gradients

    def to_string(self):
        return f"GaussianPrimitivePrbIntegrator[]"
    
mi.register_integrator("gsprim_prb", lambda props: GaussianPrimitivePrbIntegrator(props))