import gc
import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator

class VolumetricPrimitiveRadianceFieldIntegrator(ReparamIntegrator):
    def __init__(self, props):
        super().__init__(props)

    def aovs(self):
        return []

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aovs())
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            L, valid, _aovs, _, _= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
                depth=mi.UInt32(0), δL=None, δA=None, δD=None, δN=None, state_in=None, reparam=None, active=mi.Bool(True))
            
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

            L, valid,  _, gradients, state_out = self.sample(
                mode=dr.ADMode.Forward, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
                δL=None, δA=None, δD=None, δN=None, 
                state_in=None, reparam=None, active=mi.Bool(True))
            
            color_state_out = state_out['color']
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

                for data_name, extra_data in zip(['color', 'depth', 'normal'], [color_state_out, depth_state_out, normal_state_out]):
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
                δD = dr.grad(depth_state_out)       # ∂loss/∂Depth
                δN = dr.grad(normal_state_out)      # ∂loss/∂Normal

                #Convert ∂loss/∂RGB to ∂loss/∂A, ∂loss/∂R, ∂loss/∂M
                δA = δL * gradients['color'] # ∂loss/∂RGB * ∂RGB/∂A + ∂loss/∂A = ∂loss/∂A 
                δD = δD * gradients['depth']
                δN = δN * gradients['normal']

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, _, _, state_out_2 = self.sample(
                mode=dr.ADMode.Backward, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), 
                δL=δL, δA=δA, δD=δD, δN=δN,
                state_in=state_out, active=mi.Bool(True))

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, δA, δD, δN, ray, weight, pos, block, extra_block, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()


    @dr.syntax
    def ray_marching_loop(self, scene, sampler, primal, ray, δA, δD, δN, state_in, active):
        
        num = mi.UInt32(0)
        active = mi.Mask(active)

        ray = mi.Ray3f(dr.detach(ray)) #clone a new ray

        A = mi.Spectrum(0.0 if primal else state_in['color'])
        D = mi.Float(0.0 if primal else state_in['depth'][0])
        N = mi.Spectrum(0.0 if primal else state_in['normal'])
        weight_acc = mi.Float(0.0 if primal else state_in['weight_acc'])

        δA = mi.Spectrum(δA if δA is not None else 0)
        δD = mi.Spectrum(δD if δD is not None else 0)
        δN = mi.Spectrum(δN if δN is not None else 0)

        T = mi.Float(1.0)

        depth_acc = mi.Float(0.0)
        while dr.hint(active, label="Ray Marching"):
            si_cur = scene.ray_intersect(ray, coherent=True, ray_flags=mi.RayFlags.All, active=active)
            active &= si_cur.is_valid() & si_cur.shape.is_ellipsoids()

            depth_acc += dr.select(active, si_cur.t, 0.0)

            depth = mi.Float(0.0)
            normal = mi.Spectrum(0.0)
            color = mi.Spectrum(0.0)
            
            weight = mi.Float(0.0)
            with dr.resume_grad(when=not primal):
                sh_color = self.eval_sh_emission(si_cur, ray, active)
                normals_val, _, _, _ = self.eval_bsdf_component(si_cur, ray, active)
                transmission = self.eval_transmission(si_cur, ray, active)

                weight = T * (1.0 - transmission)

                color = weight * sh_color
                color[~dr.isfinite(color)] = 0.0

                depth[active] = weight * depth_acc
                depth[~dr.isfinite(depth)] = 0.0

                normal[active] = weight * normals_val
                normal[~dr.isfinite(normal)] = 0.0

            A[active] = (A + color) if primal else (A - color)

            D[active] = (D + depth) if primal else (D - depth / weight_acc)
            N[active] = (N + normal) if primal else (N - normal / weight_acc)
            weight_acc[active]= (weight_acc + weight) if primal else (weight_acc - weight)

            T[active] *= transmission

            ray.o[active] = si_cur.p + ray.d * 1e-4
            depth_acc[active] += 1e-4

            with dr.resume_grad(when=not primal):
                if not primal:
                    Ar_ind = A * transmission / dr.detach(transmission)
                    Dr_ind = D * transmission / dr.detach(transmission)
                    Nr_ind = N * transmission / dr.detach(transmission)
                    
                    Ao = color + Ar_ind
                    Do = depth + Dr_ind
                    No = normal + Nr_ind

                    Ao = dr.select(active & dr.isfinite(Ao), Ao, 0.0)
                    Do = dr.select(active & dr.isfinite(Do), Do, 0.0)
                    No = dr.select(active & dr.isfinite(No), No, 0.0)

                    loss = δA * Ao + δD * Do + δN * No
                    dr.backward_from(loss)
            
            active &= si_cur.is_valid()
            num[active] += 1

            active &= T > 0.01
            active &= num < self.gaussian_max_depth

            # Perform Russian Roulette
            # sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            # if primal and num >= 10:
            #     rr_prob = dr.maximum(T, 0.1)
            #     rr_active = T < 0.1
            #     T = dr.select(rr_active, T * dr.rcp(rr_prob), T)
            #     rr_continue = sample_rr < rr_prob
            #     active &= ~rr_active | rr_continue

        D = D / dr.maximum(weight_acc, 1e-8)
        N = N / dr.maximum(weight_acc, 1e-8)

        A = mi.math.srgb_to_linear(A)

        return A, D, N, (weight_acc>0.5), weight_acc

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, δA, δD, δN, state_in, active, **kwargs):
        
        primal = (mode == dr.ADMode.Primal)
        forward = (mode == dr.ADMode.Forward)
        
        active = mi.Mask(active)
        result = mi.Spectrum(0.0)
        result_N = mi.Spectrum(0.0)
        result_D = mi.Spectrum(0.0)

        A_raw, D_raw, N_raw, active, weight_acc = self.ray_marching_loop(scene, sampler, (primal|forward), ray, δA, δD, δN, state_in, active)

        with dr.resume_grad(when= forward):
            if forward:
                dr.enable_grad(A_raw, D_raw, N_raw)
            
            A = self.safe_clamp(A_raw, 0.0, 1.0)
            N = self.safe_normalize(N_raw)
            D = D_raw

            result[active] += A
            result_N[active] += N
            result_D[active] += D

            gradients = {}
            if forward:
                dr.backward_from(result)
                dr.backward_from(result_N)
                dr.backward_from(result_D)

                δA = dr.grad(A_raw) # ∂RGB/∂A
                δD = dr.grad(D_raw)
                δN = dr.grad(N_raw)

                gradients = {
                    'color': δA,
                    'depth': δD,
                    'normal': δN
                }

        #aov & state_out
        aovs = {
            'color': dr.select(active, result, 0.0),
            'depth': dr.select(active, result_D, 0.0),
            'normal': dr.select(active, result_N, 0.0)
        }

        state_out = {
            'color': dr.select(active, A_raw, 0.0),
            'depth': dr.select(active, mi.Spectrum(D_raw), 0.0),
            'normal': dr.select(active, N_raw, 0.0),
            'weight_acc': dr.select(active, weight_acc, 0.0)
        }

        return mi.Spectrum(result), True, aovs, gradients, state_out

    def to_string(self):
        return f"VolumetricPrimitiveRadianceFieldIntegrator[]"
    
mi.register_integrator("volprim_refine", lambda props: VolumetricPrimitiveRadianceFieldIntegrator(props))