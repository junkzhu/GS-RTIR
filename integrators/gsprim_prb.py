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

        A_prev = mi.Spectrum(0.0)
        R_prev = mi.Float(0.0)
        M_prev = mi.Float(0.0)
        D_prev = mi.Float(0.0)
        N_prev = mi.Spectrum(0.0)

        #pi info
        with dr.resume_grad(when=not primal):
            A_cur, R_cur, M_cur, D_cur, N_cur, si_valid, weight_acc = self.ray_marching_loop(scene, sampler, True, ray_cur, δA, δR, δM, δD, δN, state_in, active)    
            si_cur = self.SurfaceInteraction3f(ray_cur, D_cur, N_cur, si_valid)
            valid_ray |= si_cur.is_valid()
        
        aovs = {
            'albedo': dr.select(si_valid, A_cur, 0.0),
            'roughness': dr.select(si_valid, mi.Spectrum(R_cur), 0.0),
            'metallic': dr.select(si_valid, mi.Spectrum(M_cur), 0.0),
            'depth': dr.select(si_valid, mi.Spectrum(D_cur), 0.0),
            'normal': dr.select(si_valid, N_cur, 0.0)
        }

        while dr.hint(active, max_iterations=self.max_depth, label="Path Replay Backpropagation (%s)" % mode.name):
            first_vertex = (depth == 0)
            active_next = mi.Bool(active)
            
            if not primal:
                with dr.resume_grad():
                    dr.enable_grad(A_cur, R_cur, M_cur, D_cur, N_cur)
                    dr.disable_grad(si_prev)
            
            with dr.resume_grad(when=not primal):      
                Le = β * mis_em * si_cur.emitter(scene).eval(si_cur) #当没有与物体相交（无效si），那么就直接查找环境贴图的值（光源只有环境贴图）
           
            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid() #if 最后一轮 不进行后续的计算
            # Next event estimation
            active_em = active_next
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(active_em), False, active_em) #随机采样一个新的方向（envmap采样）
            active_em &= (ds.pdf != 0.0)

            with dr.resume_grad(when= not primal):
                em_ray = si_cur.spawn_ray_to(ds.p) #生成一条新的光线
                em_ray.d = dr.detach(em_ray.d)
                occluded = self.ray_test(scene, sampler, em_ray, active_em) #测试是否遮挡
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
                bsdf_value_em, bsdf_pdf_em = self.eval_bsdf(A_cur, R_cur, M_cur, N_cur, Vdirection, Ldirection, Halfvector) #计算交点的BSDF值（envmap采样）
                mis_direct = mis_weight(ds.pdf, bsdf_pdf_em)
                Lr_dir = visibility * β * mis_direct * bsdf_value_em * em_weight #直接光源（envmap）

                bsdf_val, bsdf_dir, bsdf_pdf = self.bsdf(sampler, si_cur, A_cur, R_cur, M_cur, N_cur, Vdirection) #get bsdf attributes
                bsdf_weight = bsdf_val / dr.maximum(1e-8, bsdf_pdf)
                active_next &= (bsdf_pdf > 0.0)
                β *= mi.Spectrum(bsdf_weight)
                L_prev = L 

                L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
                #L = L + dr.select(first_vertex, Le + Lr_dir, 0.0)
                 
            # Intersect next surface
            ray_next = si_cur.spawn_ray(bsdf_dir) #根据bsdf采样获得一个新的方向
            A_next, R_next, M_next, D_next, N_next, si_next_valid, weight_acc_next = self.ray_marching_loop(scene, sampler, True, ray_next, δA, δR, δM, δD, δN, state_in, active_next)
            si_next = self.SurfaceInteraction3f(ray_next, D_next, N_next, si_next_valid) #创建bsdf采样的交点

            # Compute MIS weight for the next vertex
            ds = mi.DirectionSample3f(scene, si=si_next, ref=si_cur)
            em_pdf = scene.pdf_emitter_direction(ref=si_cur, ds=ds, active=active_next)
            mis_em = mis_weight(bsdf_pdf, em_pdf)

            if not primal:
                sampler_clone = sampler.clone()
                active_next_next = active_next & si_next.is_valid() & (depth + 2 < self.max_depth)

                bsdf_next_val, bsdf_next_dir, bsdf_next_pdf = self.bsdf(sampler, si_next, A_next, R_next, M_next, N_next, ray_next.d)
                bsdf_prev_val, bsdf_prev_dir, bsdf_prev_pdf = self.bsdf(sampler, si_prev, A_prev, R_prev, M_prev, N_prev, ray_prev.d)

                active_em_next = active_next_next
                ds_next, em_weight_next = scene.sample_emitter_direction(si_next, sampler_clone.next_2d(), True, active_em_next)
                active_em_next &= (ds_next.pdf != 0.0)

                mis_direct_next = dr.select(ds_next.delta, 1, mis_weight(ds_next.pdf, bsdf_next_pdf))
                Lr_dir_next = β * mis_direct_next * bsdf_next_val * em_weight_next

                with dr.resume_grad(si_cur.p):
                    wo_prev = dr.normalize(si_cur.p - si_prev.p)
                    wi_next = dr.normalize(si_cur.p - si_next.p)

                    si_next.wi = si_next.to_local(wi_next)
                    Le_next = β * mis_em * si_next.emitter(scene).eval(si_next, active_next)
                    L_next = L - dr.detach(Le_next) - dr.detach(Lr_dir_next)

                    extra = mi.Spectrum(Le_next)
                    extra[~first_vertex] += L_prev * bsdf_prev_val / dr.detach(bsdf_prev_val)
                    extra[si_next.is_valid()] += L_next * bsdf_next_val / dr.detach(bsdf_next_val)

                with dr.resume_grad():
                    bsdf_val_det = bsdf_weight * bsdf_pdf
                    inv_bsdf_val_det = dr.select((bsdf_val_det != 0), dr.rcp(bsdf_val_det), 0)
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    Lo = (Le + Lr_dir + Lr_ind) + extra
                    Lo[depth > self.max_depth] = 0
                    
                    dr.backward_from(δL * Lo)

                    δA_cur, δR_cur, δM_cur, δD_cur, δN_cur = map(dr.grad, (A_cur, R_cur, M_cur, D_cur, N_cur))

                    result_temp = {
                        'albedo': dr.select(valid_ray, A_cur, 0.0),
                        'roughness': dr.select(valid_ray, mi.Spectrum(R_cur), 0.0),
                        'metallic': dr.select(valid_ray, mi.Spectrum(M_cur), 0.0),
                        'depth': dr.select(valid_ray, mi.Spectrum(D_cur), 0.0),
                        'normal': dr.select(valid_ray, N_cur, 0.0),
                        'weight_acc': dr.select(valid_ray, weight_acc, 0.0)
                    }

                if first_vertex:
                    δA = δA
                    δR = δR
                    δM = δM
                    δD = δD
                    δN = δN
                    self.ray_marching_loop(scene, sampler, False, ray_cur, δA, δR, δM, δD, δN, result_temp, active)

            depth[si_cur.is_valid()] += 1
            
            active = active_next
            si_prev, ray_prev = si_cur, ray_cur
            si_cur, ray_cur = si_next, ray_next
            A_prev, R_prev, M_prev, D_prev, N_prev = map(dr.detach, (A_cur, R_cur, M_cur, D_cur, N_cur))
            A_cur, R_cur, M_cur, D_cur, N_cur = map(dr.detach,(A_next, R_next, M_next, D_next, N_next))

        result = L
        gradients = {}
        
        return dr.select(valid_ray, mi.Spectrum(result), 0.0), True, aovs, gradients

    def to_string(self):
        return f"GaussianPrimitivePrbIntegrator[]"
    
mi.register_integrator("gsprim_prb", lambda props: GaussianPrimitivePrbIntegrator(props))