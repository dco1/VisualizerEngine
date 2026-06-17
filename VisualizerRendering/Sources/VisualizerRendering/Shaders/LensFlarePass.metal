// ── LENS FLARE PASS ─────────────────────────────────────────────────────────
//
// Screen-space additive lens-flare ghosts at the projected positions of
// active bursts. For each burst:
//   • Project its world position into NDC via the view-projection matrix.
//   • If on-screen, write a soft anamorphic streak (horizontal Gaussian)
//     + a faint colour-ghost on the line from source through screen centre.
//
// REUSABLE for any scene that wants a "camera responds to bright sources"
// look. Works against the FrameCompositor's HDR target — adds to it
// before bloom + tonemap, so the flare is naturally smeared by the bloom
// chain into the soft halos you expect from cinematic photography.

#include <metal_stdlib>
using namespace metal;

struct BurstLight {
    float4 positionIntensity;
    float4 colorAge;
    float4 radiusLifePad;
};

struct LensFlareUniforms {
    float4 viewProjRow0;     // rows of view-projection matrix for projecting
    float4 viewProjRow1;     // burst world positions to NDC.
    float4 viewProjRow2;
    float4 viewProjRow3;
    float4 countAndGain;     // x = burst count (uint bits), y = global gain,
                             // z = anamorphic stretch, w = ghost intensity
};

inline float3 lensProject(float3 worldPos, constant LensFlareUniforms& u) {
    float4 clip;
    clip.x = dot(u.viewProjRow0, float4(worldPos, 1.0f));
    clip.y = dot(u.viewProjRow1, float4(worldPos, 1.0f));
    clip.z = dot(u.viewProjRow2, float4(worldPos, 1.0f));
    clip.w = dot(u.viewProjRow3, float4(worldPos, 1.0f));
    if (clip.w <= 0.0f) return float3(-2, -2, -1);    // behind camera
    return float3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
}

kernel void lensFlarePass(
    texture2d<float, access::read_write> hdr     [[ texture(0) ]],
    device   const BurstLight*           bursts  [[ buffer(0) ]],
    constant LensFlareUniforms&          u       [[ buffer(1) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint w = hdr.get_width();
    uint h = hdr.get_height();
    if (gid.x >= w || gid.y >= h) return;

    uint burstCount = as_type<uint>(u.countAndGain.x);
    if (burstCount == 0u) return;

    float globalGain = u.countAndGain.y;
    float anamorphic = u.countAndGain.z;
    float ghostI    = u.countAndGain.w;

    // Pixel position in NDC (-1..+1, y up).
    float2 pixelNdc = float2(
        (float(gid.x) + 0.5f) / float(w) * 2.0f - 1.0f,
        1.0f - (float(gid.y) + 0.5f) / float(h) * 2.0f
    );

    float3 accum = float3(0.0f);

    for (uint i = 0u; i < burstCount; ++i) {
        BurstLight L = bursts[i];
        float intensity = L.positionIntensity.w;
        float life = L.radiusLifePad.y;
        if (intensity <= 0.0f || life <= 0.0f) continue;

        float age = L.colorAge.a;
        if (age >= life) continue;

        // Age fade (peak at age 0, zero at lifespan).
        float ageT = age / life;
        float ageScale;
        if (ageT < 0.08f) {
            ageScale = 1.0f;
        } else {
            float k = (ageT - 0.08f) / 0.92f;
            ageScale = (1.0f - k);
            ageScale = ageScale * ageScale;
        }

        // Project burst to NDC.
        float3 ndc = lensProject(L.positionIntensity.xyz, u);
        if (ndc.z < 0.0f || ndc.z > 1.0f) continue;     // off-camera

        // Anamorphic streak — horizontal Gaussian centred at the burst's
        // screen position. Wide along X (stretched), narrow along Y.
        float2 toBurst = pixelNdc - ndc.xy;
        float dx = toBurst.x / anamorphic;
        float dy = toBurst.y;
        float r2 = dx * dx + dy * dy * 9.0f;
        float streak = exp(-r2 * 12.0f);

        // Ghost dot on the line from burst-through-centre. Real lens
        // flares show inverted ghosts of bright sources reflected off
        // internal lens elements; one ghost at -0.4× the burst's NDC
        // position is a believable cheap version.
        float2 ghostPos = -ndc.xy * 0.4f;
        float2 toGhost = pixelNdc - ghostPos;
        float ghostR2 = dot(toGhost, toGhost);
        float ghost = exp(-ghostR2 * 60.0f) * ghostI;

        // Distance-from-burst falloff so the streak only paints near it.
        float radial = exp(-dot(toBurst, toBurst) * 4.0f);
        float combined = (streak + ghost) * radial;

        accum += L.colorAge.rgb * (intensity * ageScale * combined * globalGain);
    }

    float4 dst = hdr.read(gid);
    hdr.write(float4(dst.rgb + accum, dst.a), gid);
}
