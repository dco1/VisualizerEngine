// ── ILLUMINATORAMA — RAY-TRACED REFLECTION (SPECULAR) DENOISER · DH-0539 ─────
//
// The diffuse RT term has always had a denoise chain (temporal → SVGF à-trous → composite). The
// glossy reflection term did not: `illumi_rt_lighting_tlas` added it RAW into the HDR, so the only
// thing that ever cleaned it was TAA, and the host had to cap the roughness cutoff at 0.25 (the
// engine's own is 0.6) because anything rougher is a one-ray lobe: un-denoisable speckle.
//
// With `specDenoise` on, the TLAS kernel hands the reflection to this chain instead:
//
//   rtSpecular  rgb = (reflection − the deferred sky it replaces) ÷ Fresnel, a = mean ray hit distance
//               (0 ⇒ this pixel traced no reflection; the chain skips it)
//      │
//      ├─ illumi_spec_temporal  hit-distance-aware reprojection: a reflection does not live on the
//      │                        SURFACE, it lives on the VIRTUAL IMAGE behind it — the point along
//      │                        the view ray at (surface distance + hit distance). That point is
//      │                        static for a static scene and mirror, so IT is what is reprojected,
//      │                        not the surface's own motion. Blend floor and YCoCg clamp scale with
//      │                        roughness (a mirror keeps no history; a rough lobe keeps a lot).
//      ├─ illumi_spec_atrous    variance-guided à-trous, level count and radius scaled by roughness
//      │                        (a mirror runs 0 levels, so a sharp reflection of a mirror or a pane
//      │                        is never smeared); depth / normal / roughness / luminance stops.
//      └─ illumi_spec_composite adds the cleaned term into the HDR composite.
//
// Shares the SVGF colour space (`IlluminatoramaDenoise.h`) and edge-stop shapes, but NOT its
// kernels: the signal is signed (a delta over the sky it replaces) and its history is keyed by a
// virtual position, so the diffuse kernels would be wrong for it.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
#include "IlluminatoramaDenoise.h"
using namespace metal;

struct SpecDenoiseUniforms {
    uint  width; uint height; uint flags; uint stepSize;      // flags: bit0 = first frame (no history), bit1 = still (no temporal mean)
    float sigmaL; float sigmaZ; float sigmaN; float clampSigma;
    float mirrorRough;   // at or below this the lobe is a mirror: no history, no spatial filter
    float fullRough;     // at or above this the lobe is fully rough: the long-history / wide-kernel end
    float alphaRough;    // history blend floor (weight of the CURRENT frame) at fullRough
    uint  maxLevels;     // à-trous levels run for a fully rough pixel
    float lumFloor;      // relative floor of the luminance stop's bandwidth (a still raises it: see below)
    float _pad0; float _pad1; float _pad2;
};

static inline float specLum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
// The scaled YCoCg pair in IlluminatoramaDenoise.h clamps the inverse at 0 — right for radiance,
// wrong for this SIGNED delta, so the inverse is repeated here without the clamp.
static inline float3 specFromYCoCg(float3 c) { float t = c.x - c.z; return float3(t + c.y, c.x + c.z, t - c.y); }
static inline float3 specWorldPos(float2 ndcXY, float depth, float4x4 invVP) {
    float4 w = invVP * float4(ndcXY, depth, 1.0);
    return w.xyz / w.w;
}
static inline float specRoughness(texture2d<half, access::read> n, uint2 p) { return max(0.045, float(n.read(p).b)); }

// ── 1 · temporal ─────────────────────────────────────────────────────────────
kernel void illumi_spec_temporal(
    texture2d<half,  access::read>   raw       [[texture(0)]],   // rtSpecular: delta rgb, hit distance a
    texture2d<half,  access::sample> histColor [[texture(1)]],   // previous accumulated delta
    texture2d<float, access::sample> histAux   [[texture(2)]],   // previous (m1, m2, count, viewZ)
    texture2d<float, access::read>   gDepth    [[texture(3)]],
    texture2d<half,  access::read>   gNormal   [[texture(4)]],
    texture2d<half,  access::write>  outColor  [[texture(5)]],   // rgb accumulated, a = traced flag
    texture2d<float, access::write>  outAux    [[texture(6)]],
    texture2d<half,  access::write>  outVar    [[texture(7)]],
    constant SpecDenoiseUniforms&    u         [[buffer(0)]],
    constant FrameUniforms&          frame     [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    half4 r = raw.read(gid);
    float depth = gDepth.read(gid).r;
    float hitDist = float(r.a);
    if (depth >= 0.99999 || hitDist <= 0.0) {
        outColor.write(half4(0.0h), gid);
        if ((u.flags & 2u) == 0u) outAux.write(float4(0.0), gid);
        outVar.write(half4(0.0h), gid);
        return;
    }
    float3 cur = float3(r.rgb);
    float  rough = specRoughness(gNormal, gid);
    float  curLum = specLum(cur);

    float2 invWH = 1.0 / float2(u.width, u.height);
    float2 ndc = (float2(gid) + 0.5) * invWH * 2.0 - 1.0; ndc.y = -ndc.y;
    float  viewZ = abs(viewPosFromDepth(ndc, depth, frame.invProjection).z);

    // Spatial variance of the CURRENT frame, the fallback while the temporal moments are young.
    float s1 = 0.0, s2 = 0.0, sn = 0.0;
    float3 y1 = float3(0.0), y2 = float3(0.0);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j), int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            half4 q = raw.read(uint2(c));
            if (q.a <= 0.0h) continue;
            float3 qc = float3(q.rgb);
            float l = specLum(qc);
            s1 += l; s2 += l * l; sn += 1.0;
            float3 yy = RGBtoYCoCg(qc);
            y1 += yy; y2 += yy * yy;
        }
    }
    float spatialVar = 0.0;
    float3 mu = RGBtoYCoCg(cur), sigma = float3(0.02);
    if (sn > 0.0) {
        s1 /= sn; s2 /= sn; y1 /= sn; y2 /= sn;
        spatialVar = max(0.0, s2 - s1 * s1);
        mu = y1;
        sigma = max(sqrt(max(float3(0.0), y2 - y1 * y1)), float3(0.02));
    }

    float t = smoothstep(u.mirrorRough, u.fullRough, rough);
    float alphaFloor = mix(1.0, u.alphaRough, t);

    float N = 0.0;
    float3 outRGB = cur;
    float m1 = curLum, m2 = curLum * curLum;
    bool haveHistory = false;
    if ((u.flags & 3u) == 0u && alphaFloor < 0.999) {   // bit1: a frozen still's colour resolve IS the mean — a history here would correlate its frames
        // The virtual image: the view ray extended past the surface by the hit distance.
        float3 P = specWorldPos(ndc, depth, frame.invViewProjection);
        float3 dirV = P - frame.cameraWorldPos;
        float  dist = length(dirV);
        float3 X = frame.cameraWorldPos + dirV / max(dist, 1e-4) * (dist + hitDist);
        float4 clip = frame.previousViewProjection * float4(X, 1.0);
        if (clip.w > 1e-4) {
            float2 pn = clip.xy / clip.w;
            float2 histUV = float2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
            if (all(histUV > 0.0) && all(histUV < 1.0)) {
                constexpr sampler samp(filter::linear, address::clamp_to_edge);
                float4 aux = histAux.sample(samp, histUV);
                // The history texel must have held a TRACED surface at about this depth.
                float histZ = aux.w;
                bool consistent = aux.z >= 1.0 && abs(histZ - viewZ) <= 0.12 * max(viewZ, 0.5);
                if (consistent) {
                    float3 hist = float3(histColor.sample(samp, histUV).rgb);
                    // Tight neighbourhood clip in YCoCg — a reflection's history is only valid while it
                    // stays inside what the neighbours of THIS frame say it could be.
                    float3 hy = RGBtoYCoCg(hist);
                    hy = clamp(hy, mu - u.clampSigma * sigma, mu + u.clampSigma * sigma);
                    N = min(aux.z + 1.0, 64.0);
                    float alpha = max(alphaFloor, 1.0 / N);
                    outRGB = specFromYCoCg(mix(hy, RGBtoYCoCg(cur), alpha));
                    m1 = mix(aux.x, curLum, alpha);
                    m2 = mix(aux.y, curLum * curLum, alpha);
                    haveHistory = true;
                }
            }
        }
    }
    if (!haveHistory) N = 1.0;
    float tvar = max(0.0, m2 - m1 * m1);
    // Young history ⇒ the temporal moments say nothing yet; use the spatial estimate.
    float variance = N >= 4.0 ? tvar : max(tvar, spatialVar);
    outColor.write(half4(half3(outRGB), 1.0h), gid);
    if ((u.flags & 2u) == 0u) outAux.write(float4(m1, m2, N, viewZ), gid);   // a still keeps no history, so no aux (it is a stand-in there)
    outVar.write(half4(half(variance)), gid);
}

// ── 2 · à-trous (variance-guided, roughness-scaled) ───────────────────────────
constant float kSpecAtrousH[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };

kernel void illumi_spec_atrous(
    texture2d<half,  access::read>  cIn      [[texture(0)]],   // rgb delta, a = traced flag
    texture2d<half,  access::read>  vIn      [[texture(1)]],   // variance (r)
    texture2d<float, access::read>  gDepth   [[texture(2)]],
    texture2d<half,  access::read>  gNormal  [[texture(3)]],
    texture2d<half,  access::write> cOut     [[texture(4)]],
    texture2d<half,  access::write> vOut     [[texture(5)]],
    constant SpecDenoiseUniforms&   u        [[buffer(0)]],
    constant FrameUniforms&         frame    [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    half4 c0 = cIn.read(gid);
    float depth = gDepth.read(gid).r;
    if (depth >= 0.99999 || c0.a <= 0.0h) { cOut.write(c0, gid); vOut.write(vIn.read(gid), gid); return; }

    // How many levels THIS pixel runs — the roughness-scaled radius. A mirror runs none.
    float rough = specRoughness(gNormal, gid);
    float t = smoothstep(u.mirrorRough, u.fullRough, rough);
    uint levelsHere = (rough <= u.mirrorRough) ? 0u : max(1u, uint(round(t * float(u.maxLevels))));
    uint level = (u.stepSize == 1u) ? 0u : uint(log2(float(u.stepSize)));
    if (level >= levelsHere) { cOut.write(c0, gid); vOut.write(vIn.read(gid), gid); return; }

    float2 invWH = 1.0 / float2(u.width, u.height);
    float2 ndcC = (float2(gid) + 0.5) * invWH * 2.0 - 1.0; ndcC.y = -ndcC.y;
    float  zC = viewPosFromDepth(ndcC, depth, frame.invProjection).z;
    uint2  gx = uint2(min(gid.x + 1u, u.width - 1u), gid.y);
    uint2  gy = uint2(gid.x, min(gid.y + 1u, u.height - 1u));
    float  dX = gDepth.read(gx).r, dY = gDepth.read(gy).r;
    float2 ndcX = (float2(gx) + 0.5) * invWH * 2.0 - 1.0; ndcX.y = -ndcX.y;
    float2 ndcY = (float2(gy) + 0.5) * invWH * 2.0 - 1.0; ndcY.y = -ndcY.y;
    float dzdx = (dX >= 0.99999) ? 0.0 : (viewPosFromDepth(ndcX, dX, frame.invProjection).z - zC);
    float dzdy = (dY >= 0.99999) ? 0.0 : (viewPosFromDepth(ndcY, dY, frame.invProjection).z - zC);

    float3 nC = octDecode(float2(gNormal.read(gid).rg));
    float  lumC = specLum(float3(c0.rgb));

    // Prefiltered variance drives the luminance stop (canonical SVGF), floor relative to the signal.
    float gVar = 0.0;
    const float g3[3] = { 0.25, 0.5, 0.25 };
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = clamp(int2(gid) + int2(i, j), int2(0), int2(int(u.width) - 1, int(u.height) - 1));
            gVar += g3[i + 1] * g3[j + 1] * float(vIn.read(uint2(c)).r);
        }
    }
    float lumSigma = u.sigmaL * sqrt(max(gVar, 0.0)) + u.lumFloor * max(abs(lumC), 0.02);

    float3 sum = float3(0.0);
    float  vsum = 0.0, wsum = 0.0;
    int step = int(u.stepSize);
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            int2 p = int2(gid) + int2(i, j) * step;
            if (p.x < 0 || p.y < 0 || p.x >= int(u.width) || p.y >= int(u.height)) continue;
            uint2 np = uint2(p);
            half4 cn = cIn.read(np);
            if (cn.a <= 0.0h) continue;
            float nd = gDepth.read(np).r;
            if (nd >= 0.99999) continue;
            float hW = kSpecAtrousH[i + 2] * kSpecAtrousH[j + 2];
            float zN = viewPosFromDepth(((float2(np) + 0.5) * invWH * 2.0 - 1.0) * float2(1.0, -1.0), nd, frame.invProjection).z;
            float expDz = dzdx * float(i * step) + dzdy * float(j * step);
            float wDepth = exp(-abs(zN - zC) / (u.sigmaZ * abs(expDz) + 1e-2));
            half4 nrN = gNormal.read(np);
            float wNorm = pow(max(0.0, dot(nC, octDecode(float2(nrN.rg)))), u.sigmaN);
            // A different material roughness is a different lobe: its reflection is not this one's.
            float wRough = exp(-abs(max(0.045, float(nrN.b)) - rough) * 12.0);
            float lumN = specLum(float3(cn.rgb));
            float wLum = exp(-abs(lumC - lumN) / max(lumSigma, 1e-4));
            float w = hW * wDepth * wNorm * wRough * wLum;
            sum += w * float3(cn.rgb);
            vsum += w * w * float(vIn.read(np).r);
            wsum += w;
        }
    }
    if (wsum < 1e-6) { cOut.write(c0, gid); vOut.write(vIn.read(gid), gid); return; }
    cOut.write(half4(half3(sum / wsum), 1.0h), gid);
    vOut.write(half4(half(vsum / (wsum * wsum))), gid);
}

// ── 3 · composite ────────────────────────────────────────────────────────────
kernel void illumi_spec_composite(
    texture2d<half,  access::read>        spec   [[texture(0)]],
    texture2d<half,  access::read_write>  outHDR [[texture(1)]],
    texture2d<half,  access::read>        fres   [[texture(2)]],   // the Fresnel the signal was demodulated by
    constant SpecDenoiseUniforms&         u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.width || gid.y >= u.height) return;
    half4 s = spec.read(gid);
    if (s.a <= 0.0h) return;
    half4 prev = outHDR.read(gid);
    outHDR.write(half4(max(prev.rgb + s.rgb * fres.read(gid).rgb, half3(0.0h)), prev.a), gid);
}
