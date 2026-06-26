import drjit as dr
import mitsuba as mi

PI = dr.pi
EPS = 1e-8


class DisneyBSDF:
    """
    Disney-style PBR BSDF (Lambertian diffuse + GGX specular) for differentiable
    evaluation and importance sampling. Used by reparam integrators with
    albedo/roughness/metallic from Gaussians
    """

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

        if isinstance(bsdf_pdf, mi.Spectrum):
            bsdf_pdf = bsdf_pdf[0]

        return bsdf_val, dr.detach(bsdf_pdf)

    def sample_bsdf(self, sampler, si, roughness, metallic, V_world):
        """
        Sample BSDF direction (Disney simplified: Diffuse + GGX Specular)
        Input: V_world (view dir in world space, pointing away from surface)
        Output: L_world (sampled direction in world space), pdf
        """
        #TODO: When normal hasn't converged, there's a possibility of V and N being reversed, which can cause unexpected errors. (Reproduction method: initialize GS normal attribute to (0,0,-1))

        if isinstance(roughness, mi.Spectrum):
            roughness = roughness[0]

        if isinstance(metallic, mi.Spectrum):
            metallic = metallic[0]

        # Local coordinates
        V = dr.normalize(si.to_local(V_world))
        N = mi.Vector3f(0.0, 0.0, 1.0) # local normal

        # randoms
        r0 = sampler.next_1d()
        r1 = sampler.next_1d()
        r2 = sampler.next_1d()

        diffuse_prob  = (1.0 - metallic) * 0.5
        specular_prob = 1.0 - diffuse_prob
        choose_specular = (r0 < specular_prob)

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

        return L_world, dr.detach(pdf)

    def bsdf(self, sampler, si, albedo, roughness, metallic, N, Vdir):
        Ldir, pdf0 = self.sample_bsdf(sampler, si, roughness, metallic, Vdir)

        Halfvector = dr.normalize(Ldir + Vdir)
        val, pdf1 = self.eval_bsdf(albedo, roughness, metallic, N, Vdir, Ldir, Halfvector)

        bsdf_pdf = pdf1  # pdf0 == pdf1
        bsdf_dir = Ldir
        bsdf_val = dr.select(bsdf_pdf > 0.0, val, 0.0)

        return bsdf_val, bsdf_dir, bsdf_pdf
