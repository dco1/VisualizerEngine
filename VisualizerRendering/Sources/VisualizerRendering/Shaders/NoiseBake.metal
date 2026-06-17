#include <metal_stdlib>
using namespace metal;

// ── NoiseBake.metal ──────────────────────────────────────────────────────────
//
// One-shot compute kernel that fills a tileable 3D RGBA16F volume with the
// FBM + Worley fields the volumetric cloud kernel used to compute inline at
// every march step. Baking moves the noise cost from O(64 march × 60 hash
// ops) per pixel to a single trilinear texture sample per march — the
// dominant speedup for VolumetricCloudRenderer.
//
// Channel packing:
//   R = fbm4Periodic(p)        — low-frequency cloud body, 4 octaves
//   G = 1 - worley3Periodic(p*4) — high-frequency erosion mask (already baked
//                                  at the 4× density the cloud kernel asks for,
//                                  so the cloud kernel reads both R and G from
//                                  ONE sample at noise-space position q)
//   B,A = unused
//
// Tileability: every hash lookup wraps the integer lattice coordinate into
// [0, tileSize) so the produced volume tiles seamlessly under
// `address::repeat`. The cloud kernel samples this volume at
// `(noisePos / tileSize)` and lets the sampler wrap.
//
// Cost: at tileSize=128 the bake is 128³ = 2M threads × (fbm4 + worley3) =
// ~300 ms one-time. Run at renderer init, never re-baked.

inline uint nbHash3(int3 p, int tileSize) {
    int m = tileSize;
    int3 t;
    t.x = ((p.x % m) + m) % m;
    t.y = ((p.y % m) + m) % m;
    t.z = ((p.z % m) + m) % m;
    uint h = uint(t.x * 374761393 + t.y * 668265263 + t.z * 1274126177);
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

inline float nbHashFloat(int3 p, int tileSize) {
    return float(nbHash3(p, tileSize) & 0x00FFFFFFu) / float(0x01000000u);
}

inline float nbFade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float nbValueNoise3(float3 p, int tileSize) {
    int3 i = int3(floor(p));
    float3 f = p - float3(i);
    float3 u = float3(nbFade(f.x), nbFade(f.y), nbFade(f.z));

    float n000 = nbHashFloat(i + int3(0,0,0), tileSize);
    float n100 = nbHashFloat(i + int3(1,0,0), tileSize);
    float n010 = nbHashFloat(i + int3(0,1,0), tileSize);
    float n110 = nbHashFloat(i + int3(1,1,0), tileSize);
    float n001 = nbHashFloat(i + int3(0,0,1), tileSize);
    float n101 = nbHashFloat(i + int3(1,0,1), tileSize);
    float n011 = nbHashFloat(i + int3(0,1,1), tileSize);
    float n111 = nbHashFloat(i + int3(1,1,1), tileSize);

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

// 4-octave FBM. The +offset on each octave matches the inline fbm4 in
// VolumetricSky.metal so the baked field is identical to the field the
// inline noise produced (modulo the periodicity wrap, which is invisible
// at the scales the cloud kernel samples).
inline float nbFbm4(float3 p, int tileSize) {
    float a = 0.5f;
    float sum = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum  += a * nbValueNoise3(p, tileSize);
        norm += a;
        p = p * 2.02f + float3(17.0f, 13.0f, 7.0f);
        a *= 0.5f;
    }
    return sum / max(norm, 1e-5f);
}

inline float nbWorley3(float3 p, int tileSize) {
    int3 ip = int3(floor(p));
    float3 fp = p - float3(ip);
    float minD2 = 1e9f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 cell = ip + int3(dx, dy, dz);
                uint h = nbHash3(cell, tileSize);
                float3 jitter = float3(
                    float((h      ) & 0xFFu),
                    float((h >> 8u) & 0xFFu),
                    float((h >> 16u) & 0xFFu)
                ) * (1.0f / 255.0f);
                float3 pt = float3(dx, dy, dz) + jitter;
                float3 d = pt - fp;
                float d2 = dot(d, d);
                minD2 = min(minD2, d2);
            }
        }
    }
    return clamp(sqrt(minD2), 0.0f, 1.0f);
}

kernel void noiseBake(
    texture3d<float, access::write> outTex [[texture(0)]],
    constant uint &tileSize                [[buffer(0)]],
    uint3 gid                              [[thread_position_in_grid]]
) {
    uint W = outTex.get_width();
    uint H = outTex.get_height();
    uint D = outTex.get_depth();
    if (gid.x >= W || gid.y >= H || gid.z >= D) return;

    int ts = int(tileSize);
    float3 p = float3(gid);

    float fbm = nbFbm4(p, ts);
    // Worley at 4× density baked at 1× lookup positions. The cloud kernel
    // can then read fbm and "erosion" with a single texture sample at q.
    float erosion = 1.0f - nbWorley3(p * 4.0f, ts);

    outTex.write(float4(fbm, erosion, 0.0f, 0.0f), gid);
}
