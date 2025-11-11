import gc
import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import mis_weight

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
                δN = δL * gradients['normal'] + δN

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
        
        active = mi.Mask(active)
        result = mi.Spectrum(0.0)

        A, R, M, D, N, active, weight_acc = self.ray_marching_loop(scene, sampler,(primal|forward), ray, δA, δR, δM, δD, δN, state_in, active)

        with dr.resume_grad(when= forward):
            if forward:
                dr.enable_grad(A, R, M, D, N)

            si = self.SurfaceInteraction3f(ray, D, N, active)

            #visualize the emitter
            if not self.hide_emitters:
                result += dr.select(~active, si.emitter(scene, ~active).eval(si, ~active), 0.0)
            
            # ---------------------- Emitter sampling ----------------------
            active_e = mi.Mask(active)
            with dr.suspend_grad():
                ds, _ = scene.sample_emitter_direction(si, sampler.next_2d(active_e), False, active)
                active_e &= (ds.pdf != 0.0)
                shadow_ray = si.spawn_ray_to(ds.p)

                shadow_ray_valid = dr.dot(N, shadow_ray.d) > 0.0
                occluded = self.shadow_ray_test(scene, sampler, shadow_ray, active_e & shadow_ray_valid)

                si_e = dr.zeros(mi.SurfaceInteraction3f)
                si_e.sh_frame.n = ds.n
                si_e.initialize_sh_frame()
                si_e.n = si_e.sh_frame.n
                si_e.wi = -shadow_ray.d
                si_e.wavelengths = ray.wavelengths
                emitter_val = dr.select(active_e, ds.emitter.eval(si_e, active_e), 0.0)

                emitter_val = dr.select(ds.pdf > 0, emitter_val / ds.pdf, 0.0)
                visibility = dr.select(~occluded, 1.0, 0.0)
            
            #-----------eval bsdf-----------
            Ldirection = shadow_ray.d #light direction (incoming)
            Vdirection = dr.normalize(-ray.d) #view direction (outgoing) 
            Halfvector = dr.normalize(Ldirection + Vdirection) # half-vector

            bsdf_val = mi.Spectrum(0.0)
            bsdf_pdf = dr.zeros(mi.Float)

            if self.use_mis:
                bsdf_val, bsdf_pdf = self.eval_bsdf(A, R, M, N, Vdirection, Ldirection, Halfvector)
                nee_contrib = visibility * bsdf_val * emitter_val * mis_weight(ds.pdf, dr.detach(bsdf_pdf))
            else:
                bsdf_val, _ = self.eval_bsdf(A, R, M, N, Vdirection, Ldirection, Halfvector)
                nee_contrib = visibility * bsdf_val * emitter_val

            result[active] += nee_contrib

            # ---------------------- BSDF sampling ----------------------
            if self.use_mis:
                with dr.suspend_grad():
                    bs_dir, bs_pdf = self.sample_bsdf(sampler, si, R, M, Vdirection)
                    active_bsdf = mi.Mask(active) & (bs_pdf > 0.0)

                    ds = dr.zeros(mi.DirectionSample3f)
                    ds.d = bs_dir
                    ds.dist = dr.inf
                    ds.emitter = scene.emitters()[0]

                    emitter_pdf = scene.pdf_emitter_direction(si, ds, active_bsdf)
                
                Halfvector = dr.normalize(bs_dir + Vdirection)
                bsdf_val, _ = self.eval_bsdf(A, R, M, N, Vdirection, bs_dir, Halfvector)
                shadow_ray = si.spawn_ray(bs_dir)
                shadow_ray_valid = dr.dot(N, shadow_ray.d) > 0.0
                occluded = self.shadow_ray_test(scene, sampler, shadow_ray, active_bsdf & shadow_ray_valid)
                visibility = dr.select(~occluded, 1.0, 0.0)

                si_e = dr.zeros(mi.SurfaceInteraction3f)
                si_e.sh_frame.n = ds.n
                si_e.initialize_sh_frame()
                si_e.n = si_e.sh_frame.n
                si_e.wi = -shadow_ray.d
                si_e.wavelengths = ray.wavelengths
                emitter_val = dr.select(active_bsdf, ds.emitter.eval(si_e, active_bsdf), 0.0)

                bsdf_contrib = visibility * bsdf_val / bs_pdf * emitter_val * mis_weight(bs_pdf, emitter_pdf)
                result[active_bsdf] += bsdf_contrib

            gradients = {}
            if forward:
                dr.backward_from(result)                
                δA = dr.grad(A) # ∂RGB/∂A
                δR = dr.grad(R)
                δM = dr.grad(M)
                δD = dr.grad(D)
                δN = dr.grad(N)
                gradients = {
                    'albedo': δA,
                    'roughness': δR,
                    'metallic': δM,
                    'depth': δD,
                    'normal': δN
                }

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

        return mi.Spectrum(result), True, aovs, gradients

    def to_string(self):
        return f"GaussianPrimitiveRadianceFieldIntegrator[]"
    
mi.register_integrator("gsprim_rf", lambda props: GaussianPrimitiveRadianceFieldIntegrator(props))