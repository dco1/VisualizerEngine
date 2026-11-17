#ifndef ILLUMINATORAMA_HAIR_BSDF_H
#define ILLUMINATORAMA_HAIR_BSDF_H
// ── IlluminatoramaHairBSDF.h ────────────────────────────────────────────────
//
// Chiang et al. 2016 / pbrt-v3 hair fibre BSDF — the R, TT and TRT lobes (longitudinal Mp ×
// azimuthal Np × attenuation Ap, with the lobe-shift angles of the cuticle tilt) plus a residual
// multiple-scattering term — evaluated in the deferred lighting pass for pixels tagged as hair.
// Kajiya-Kay alone is one cosine lobe and reads as plastic thread; the coloured TT/TRT lobes and
// the secondary (shifted) highlight are what make fur read as fibre.
//
// Absorption comes from the fibre's *albedo* (the G-buffer colour) via d'Eon et al. 2011:
//   σa = ( ln(c) / (5.969 − 0.215β + 2.532β² − 10.73β³ + 5.574β⁴ + 0.245β⁵) )²
// so an orange pheomelanin coat is authored as the colour it should look, not as melanin amounts.
//
// The G-buffer carries the strand TANGENT in the normal slot for hair pixels (vertex alpha 0.995
// → normalRoughness.w 0.995h); the lighting pass turns that into the facing normal `Nh` that the
// rest of the pipeline (IBL, point lights) shades with.

#define kHairTagLo 0.99h
#define kHairTagHi 1.0h
#define kHairIOR 1.55f
#define kHairAlpha 0.0349f          // 2° cuticle tilt (radians)
#define kHairBetaM 0.30f            // longitudinal roughness
#define kHairBetaN 0.30f            // azimuthal roughness

static inline float hair_sqr(float x) { return x * x; }
static inline float hair_safeSqrt(float x) { return sqrt(max(0.0f, x)); }

static inline float hair_logI0(float x) {
    if (x > 12.0f) return x + 0.5f * (-log(6.2831853f) + log(1.0f / x) + 1.0f / (8.0f * x));
    float val = 0.0f, x2i = 1.0f, ifact = 1.0f; int i4 = 1;
    for (int i = 0; i < 10; ++i) {
        if (i > 1) ifact *= float(i);
        val += x2i / (i4 * hair_sqr(ifact));
        x2i *= x * x; i4 *= 4;
    }
    return log(val);
}

static inline float hair_Mp(float cosTi, float cosTo, float sinTi, float sinTo, float v) {
    float a = cosTi * cosTo / v, b = sinTi * sinTo / v;
    return (v <= 0.1f)
        ? exp(hair_logI0(a) - b - 1.0f / v + 0.6931f + log(1.0f / (2.0f * v)))
        : exp(hair_logI0(a) - b) / (2.0f * v * sinh(1.0f / v));
}

static inline float hair_frDielectric(float cosI, float eta) {
    cosI = clamp(cosI, -1.0f, 1.0f);
    if (cosI < 0.0f) { eta = 1.0f / eta; cosI = -cosI; }
    float sin2T = (1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f) return 1.0f;
    float cosT = hair_safeSqrt(1.0f - sin2T);
    float rpl = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rpe = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rpl * rpl + rpe * rpe);
}

static inline float hair_logistic(float x, float s) {
    x = abs(x);
    return exp(-x / s) / (s * hair_sqr(1.0f + exp(-x / s)));
}
static inline float hair_logisticCDF(float x, float s) { return 1.0f / (1.0f + exp(-x / s)); }
static inline float hair_trimmedLogistic(float x, float s, float a, float b) {
    return hair_logistic(x, s) / (hair_logisticCDF(b, s) - hair_logisticCDF(a, s));
}

/// Fibre scattering for light arriving from `wi` seen from `wo` (both world, unit), tangent `T`,
/// fibre albedo `c`, and the across-width offset `h` ∈ [−1, 1]. Returns the radiance-per-irradiance
/// factor (already includes the cosθ compensation pbrt divides back out — apply NO extra N·L).
static inline float3 hairFiberBSDF(float3 T, float3 wo, float3 wi, float3 c, float h) {
    // Fibre frame: x = T, (y, z) span the normal plane.
    float3 ref = abs(T.y) < 0.95f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 yA = normalize(cross(T, ref));
    float3 zA = cross(T, yA);
    float3 o = float3(dot(wo, T), dot(wo, yA), dot(wo, zA));
    float3 i = float3(dot(wi, T), dot(wi, yA), dot(wi, zA));
    float sinTo = o.x, cosTo = hair_safeSqrt(1.0f - sinTo * sinTo), phiO = atan2(o.z, o.y);
    float sinTi = i.x, cosTi = hair_safeSqrt(1.0f - sinTi * sinTi), phiI = atan2(i.z, i.y);
    float phi = phiI - phiO;
    if (phi > M_PI_F) phi -= 2.0f * M_PI_F;
    if (phi < -M_PI_F) phi += 2.0f * M_PI_F;

    const float eta = kHairIOR;
    float gammaO = asin(clamp(h, -1.0f, 1.0f));
    float sinTT = sinTo / eta, cosTT = hair_safeSqrt(1.0f - sinTT * sinTT);
    float etap = hair_safeSqrt(eta * eta - sinTo * sinTo) / max(cosTo, 1e-3f);
    float sinGT = h / etap, cosGT = hair_safeSqrt(1.0f - sinGT * sinGT);
    float gammaT = asin(clamp(sinGT, -1.0f, 1.0f));

    // σa from albedo.
    float bn = kHairBetaN;
    float den = 5.969f - 0.215f * bn + 2.532f * bn * bn - 10.73f * bn * bn * bn
              + 5.574f * pow(bn, 4.0f) + 0.245f * pow(bn, 5.0f);
    float3 sa = float3(hair_sqr(log(clamp(c.r, 1e-3f, 1.0f)) / den),
                       hair_sqr(log(clamp(c.g, 1e-3f, 1.0f)) / den),
                       hair_sqr(log(clamp(c.b, 1e-3f, 1.0f)) / den));
    float3 T1 = exp(-sa * (2.0f * cosGT / max(cosTT, 1e-3f)));

    float cosThAp = cosTo * hair_safeSqrt(1.0f - h * h);
    float fr = hair_frDielectric(cosThAp, eta);
    float3 ap[4];
    ap[0] = float3(fr);
    ap[1] = hair_sqr(1.0f - fr) * T1;
    ap[2] = ap[1] * T1 * fr;
    ap[3] = ap[2] * fr * T1 / max(float3(1.0f) - T1 * fr, float3(1e-3f));

    float bm = kHairBetaM;
    float v0 = hair_sqr(0.726f * bm + 0.812f * bm * bm + 3.7f * pow(bm, 20.0f));
    float vv[4] = { v0, 0.25f * v0, 4.0f * v0, 4.0f * v0 };
    float s = 0.626657069f * (0.265f * bn + 1.194f * bn * bn + 5.372f * pow(bn, 22.0f));

    float a = kHairAlpha;
    float sin2a = sin(2 * a), cos2a = cos(2 * a), sin1a = sin(a), cos1a = cos(a);
    float sin4a = sin(4 * a), cos4a = cos(4 * a);

    float3 fsum = float3(0.0f);
    for (int p = 0; p < 4; ++p) {
        float sTop, cTop;
        if (p == 0)      { sTop = sinTo * cos2a - cosTo * sin2a; cTop = cosTo * cos2a + sinTo * sin2a; }
        else if (p == 1) { sTop = sinTo * cos1a + cosTo * sin1a; cTop = cosTo * cos1a - sinTo * sin1a; }
        else if (p == 2) { sTop = sinTo * cos4a + cosTo * sin4a; cTop = cosTo * cos4a - sinTo * sin4a; }
        else             { sTop = sinTo;                          cTop = cosTo; }
        cTop = abs(cTop);
        float mp = hair_Mp(cosTi, cTop, sinTi, sTop, vv[p]);
        float np;
        if (p < 3) {
            float Phi = 2.0f * float(p) * gammaT - 2.0f * gammaO + float(p) * M_PI_F;
            float d = fmod(phi - Phi, 2.0f * M_PI_F);
            if (d > M_PI_F) d -= 2.0f * M_PI_F;
            if (d < -M_PI_F) d += 2.0f * M_PI_F;
            np = hair_trimmedLogistic(d, s, -M_PI_F, M_PI_F);
        } else {
            np = 1.0f / (2.0f * M_PI_F);
        }
        fsum += mp * ap[p] * np;
    }
    return fsum;
}

/// Residual multiple-scattering: the light that has bounced through the neighbouring fibres comes
/// out nearly isotropic and tinted by the *double* pass through the coat (dual scattering's global
/// term, collapsed to a single Lambert-in-the-normal-plane lobe). `c` = fibre albedo.
static inline float3 hairMultiScatter(float3 T, float3 wi, float3 c) {
    float sinI = hair_safeSqrt(1.0f - hair_sqr(dot(T, wi)));      // light component ⟂ the fibre
    return c * sqrt(max(c, float3(0.0f))) * (0.55f * sinI / M_PI_F);
}
#endif
