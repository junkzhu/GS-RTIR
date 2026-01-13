import gc
import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import mis_weight

from .reparam import ReparamIntegrator

class HybridRTIntegrator(ReparamIntegrator):
    def __init__(self, props):
        super().__init__(props)
        rr_depth       = int(props.get('rr_depth', 2))
        self.rr_depth  = mi.UInt32(rr_depth if rr_depth > 0 else 2**32-1)
        self.use_rr = rr_depth < self.max_depth

    def aovs(self):
        return []

    def render(self, scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aovs())
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            
            L, valid, _aovs, _= self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray, depth=mi.UInt32(0), active=mi.Bool(True))
            
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

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, active, **kwargs):
        
        primal = (mode == dr.ADMode.Primal)
        
        # --------------------- Configure loop state ----------------------
        active = mi.Mask(active)
        valid_ray = active # output mask
        
        depth = mi.UInt32(0)

        result = mi.Spectrum(0.0)
        L = mi.Spectrum(0)

        aov_A, aov_R, aov_M, aov_D, aov_N = mi.Spectrum(0.0), mi.Spectrum(0.0), mi.Spectrum(0.0), mi.Spectrum(0.0), mi.Spectrum(0.0)
        L_direct, L_indirect = mi.Spectrum(0.0), mi.Spectrum(0.0)

        β = mi.Spectrum(1)
        mis_em = mi.Float(1)

        ray_cur = mi.Ray3f(ray)
        
        A_cur, R_cur, M_cur, N_cur, D_cur, hit_valid, ray_valid, ray_depth, _ = self.ray_intersect(scene, sampler, ray, active)

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

        # 统一定义交点
        si_cur = self.SurfaceInteraction3f(ray_cur, D_cur, N_cur, hit_valid)        
        
        # 光源可视化
        if not self.hide_emitters:
            si_e = self.ray_intersect_emitter(scene, ray_cur, ray_valid)
            emitter = si_e.emitter(scene)            
            result += dr.select(hit_valid, 0.0, emitter.eval(si_e))
        
        #aov & state_outs
        while dr.hint(active, max_iterations=self.max_depth, label="Hybrid Ray Tracing (%s)" % mode.name):
            first_vertex = mi.Bool(depth == 0)
            active_next = mi.Bool(active)
            mis_direct = 0.0

            si_e = self.ray_intersect_emitter(scene, ray_cur, ray_valid)
            emitter = si_e.emitter(scene)
            emitter_val = dr.select(ray_valid, emitter.eval(si_e), 0.0)
            Le = dr.select(first_vertex, 0.0, β * mis_em * emitter_val)
        
            active_next &= (depth + 1 < self.max_depth) & si_cur.is_valid()
            # Next event estimation
            active_em = mi.Bool(active_next)
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(active_em), False, active_em)
            active_em &= (ds.pdf != 0.0)

            em_ray = si_cur.spawn_ray(ds.d)
            em_ray.d = dr.detach(em_ray.d)

            cosα = dr.abs(dr.dot(ray_cur.d, dr.detach(N_cur)))
            cosθ = dr.maximum(dr.abs(dr.dot(dr.detach(N_cur), em_ray.d)), 1e-8)
            occ_offset = dr.minimum((ray_depth*cosα/cosθ), self.selfocc_offset_max)
            em_ray.o = dr.detach(em_ray.o) + occ_offset * em_ray.d

            em_ray_valid = dr.dot(dr.detach(N_cur), em_ray.d) > 0.0
            occluded = self.shadow_ray_test(scene, sampler, em_ray, active_em & em_ray_valid)
            visibility = dr.select(~occluded, 1.0, 0.0)
            active_em &= ~occluded
            
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

            L = (L + Le + Lr_dir) 

            if self.separate_direct_indirect:
                # render direct illumination
                L_direct += dr.select((depth == 1), Le, 0.0) + dr.select(first_vertex, Lr_dir, 0.0)
                # render indirect illumination
                L_indirect += dr.select((depth == 1), 0.0, Le) + dr.select(first_vertex, 0.0, Lr_dir)
                 
            # Intersect next surface
            cosα = dr.abs(dr.dot(ray_cur.d, N_cur))
            cosθ = dr.maximum(dr.abs(dr.dot(N_cur, bsdf_dir)), 1e-8)
            occ_offset = dr.minimum((ray_depth*cosα/cosθ), self.selfocc_offset_max)
            ray_next = self.next_ray(scene, si_cur, bsdf_dir, occ_offset, active_next) # set offset to avoid self-occ
            ray_next_valid = dr.dot(N_cur, ray_next.d) > 0.0
            active_next &= ray_next_valid

            A_next, R_next, M_next, N_next, D_next, hit_valid_next, ray_next_valid, ray_depth_next, _ = self.ray_intersect(scene, sampler, ray_next, active_next)

            si_next = self.SurfaceInteraction3f(ray_next, D_next, N_next, hit_valid_next)

            # Compute MIS weight for the next vertex
            si_mis = dr.zeros(mi.SurfaceInteraction3f)
            si_mis.wi = -ray_next.d
            ds = mi.DirectionSample3f(scene, si=si_mis, ref=si_cur)

            em_pdf = scene.pdf_emitter_direction(ref=si_cur, ds=ds, active=active_next)
            mis_em = dr.detach(mis_weight(bsdf_pdf, em_pdf))

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
            ray_depth = dr.detach(ray_depth_next)
            active = mi.Bool(active_next)
            si_cur, ray_cur = map(dr.detach, (si_next, ray_next))
            A_cur, R_cur, M_cur, D_cur, N_cur = map(dr.detach,(A_next, R_next, M_next, D_next, N_next))

        result += dr.select(hit_valid, mi.Spectrum(L), 0.0)
        aovs['result'] = result
        
        if self.separate_direct_indirect:
            aovs['direct_light'] = dr.select(valid_ray, mi.Spectrum(L_direct), 0.0)
            aovs['indirect_light'] = dr.select(valid_ray, mi.Spectrum(L_indirect), 0.0)

        gradients = {}
        
        return result, True, aovs, gradients

    def to_string(self):
        return f"HybridRTIntegrator[]"

mi.register_integrator("hybrid_rt", lambda props: HybridRTIntegrator(props))