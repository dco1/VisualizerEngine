// ── CLOUD FLOOR MESH (Option B — real 3D cumulus geometry) ──────────────────
//
// GPU-driven cumulus heightfield. One thread per grid vertex; each thread
// samples the shared `NoiseVolume` to compute its cloud-height displacement,
// derives the analytic surface normal from neighbouring height samples, and
// accumulates per-vertex burst-light emission via a Lambert N·L term against
// every active burst. The host wraps the resulting vertex buffer in two
// SceneKit geometries — a PBR cloud body and an additive emission overlay
// — so the cloud has both ambient form and explosion-tinted highlights.
//
// Architecture options considered (in `VolumetricCloudFloorRenderer.swift`):
//   • Option A — screen-space raymarched overlay quad. Cheap to write but
//     reads as a flat 2D sheet because that's literally what it is, and
//     never depth-tests against scene geometry (shells "behind" the overlay
//     still get composited over). Deleted.
//   • Option B — THIS FILE. Real 3D heightfield mesh in scene space. Shells
//     fly through actual triangles, SceneKit's depth buffer handles
//     occlusion natively, cloud tops catch burst-light via Lambert per-vertex
//     (no triangulation artifact because diffuse is broad — unlike the ocean
//     specular that revealed mesh density).
//   • Option C — true volumetric with SCNRenderer delegate + scene depth
//     buffer integration. Cloud kernel terminates each march at scene depth
//     for proper depth-correct volumetric clouds. Requires hooking into
//     SCNRenderer's render passes — substantial work, not done.
//
// 48-byte CloudFloorVertex is float4-aligned per the project ALIGNMENT RULE.

#include <metal_stdlib>
using namespace metal;

struct CloudFloorVertex {
    float4 position;    // xyz = world, w = 1
    float4 normal;      // xyz = unit normal pointing OUT of cloud (mostly +Y)
    float4 emission;    // rgb = accumulated burst-light emission, a = pad
};

struct BurstLight {
    float4 positionIntensity;
    float4 colorAge;
    float4 radiusLifePad;
};

struct CloudMeshUniforms {
    float4 originSizeRes;   // x,y = origin (xz), z = world size (m),
                            // w = grid resolution (uint bits, verts per side)
    float4 baseShapeWind;   // x = baseY (cloud altitude), y = thickness (m),
                            // z = horizontalScale (noise lookup multiplier),
                            // w = time
    float4 coverageGain;    // x = coverage (0..1), y = shape exponent,
                            // z = burst lighting gain, w = burst count (uint bits)
    float4 windDir;         // xy = wind direction (unit), zw = pad
};

constexpr sampler volSampler(filter::linear, address::repeat, coord::normalized);

// Sample the cumulus HEIGHT function at world XZ. Returns the displacement
// above `baseY` in metres. Zero where there's no cloud (clear sky between
// puffs); peaks at `thickness` directly under the densest cumulus mounds.
//
// We AVERAGE 5 noise taps (centre + 4 cardinal offsets at ~3m radius) to
// filter out the FBM's high-frequency components. Without this the noise
// field's smallest octave creates jagged pyramidal facets at the mesh
// scale — wrong silhouette for cumulus. The average is a cheap box-filter
// that keeps low-frequency cloud bumps but kills the per-triangle noise.
inline float cloud_height(
    float2 xz,
    constant CloudMeshUniforms& u,
    texture3d<float, access::sample> noiseTex
) {
    float hScl = u.baseShapeWind.z;
    float t    = u.baseShapeWind.w;
    float2 wind = u.windDir.xy * t * 0.05f;       // slow drift — barely moves
    float tileSize = float(noiseTex.get_width());

    // 5-tap box filter, ~3m radius in world space. Filters out cloud
    // features smaller than ~6m so the mesh reads as rounded cumulus
    // mounds instead of fractal-noise spikes.
    const float SMOOTH_R = 3.0f;
    float n0, n1, n2, n3, n4;
    {
        float3 q = float3(xz.x - wind.x, 0.5f, xz.y - wind.y) * hScl;
        n0 = noiseTex.sample(volSampler, q / tileSize).r;
    }
    {
        float3 q = float3(xz.x + SMOOTH_R - wind.x, 0.5f, xz.y - wind.y) * hScl;
        n1 = noiseTex.sample(volSampler, q / tileSize).r;
    }
    {
        float3 q = float3(xz.x - SMOOTH_R - wind.x, 0.5f, xz.y - wind.y) * hScl;
        n2 = noiseTex.sample(volSampler, q / tileSize).r;
    }
    {
        float3 q = float3(xz.x - wind.x, 0.5f, xz.y + SMOOTH_R - wind.y) * hScl;
        n3 = noiseTex.sample(volSampler, q / tileSize).r;
    }
    {
        float3 q = float3(xz.x - wind.x, 0.5f, xz.y - SMOOTH_R - wind.y) * hScl;
        n4 = noiseTex.sample(volSampler, q / tileSize).r;
    }
    float n = (n0 + n1 + n2 + n3 + n4) * 0.2f;

    // Threshold-remap against coverage. Higher coverage → more sky becomes
    // cloud → height function is positive over a larger fraction of XZ.
    float coverage = u.coverageGain.x;
    float thresh = 1.0f - coverage;
    float d = max(0.0f, n - thresh) / max(1.0f - thresh, 1e-3f);

    // Shape curve: pow > 1 rounds peaks downward (cauliflower-bumpy);
    // pow < 1 puffs centres outward (cone-like). 1.3 = rounded cumulus
    // mounds with deeper valleys between them.
    float shape = pow(d, u.coverageGain.y);

    return shape * u.baseShapeWind.y;
}

// Sample the burst-light field — Lambert diffuse contribution per active
// burst. Clouds are diffuse (not specular), so this smooths broadly across
// the surface and never reveals the mesh tessellation as triangles (the
// "matrix" failure mode that the ocean's per-vertex specular hit).
inline float3 cloud_burstLambert(
    float3 vertexPos,
    float3 normal,
    device const BurstLight* lights,
    uint count,
    float gain
) {
    float3 sum = float3(0.0f);
    for (uint i = 0u; i < count; ++i) {
        BurstLight L = lights[i];
        float intensity = L.positionIntensity.w;
        float maxR      = L.radiusLifePad.x;
        float life      = L.radiusLifePad.y;
        if (intensity <= 0.0f || maxR <= 0.0f || life <= 0.0f) continue;

        float age = L.colorAge.a;
        if (age >= life) continue;

        float3 toBurst = L.positionIntensity.xyz - vertexPos;
        float dist = length(toBurst);
        // Generous reach — cloud illumination falls off with inverse-
        // square-ish behaviour but extends well past the burst's spark
        // radius (a burst illuminates a much wider area than its sparks
        // physically occupy).
        float reach = maxR * 3.5f;
        if (dist >= reach) continue;
        float3 lightDir = toBurst / max(dist, 1e-4f);

        float ageT = age / life;
        float ageScale;
        if (ageT < 0.08f) {
            ageScale = 1.0f;
        } else {
            float k = (ageT - 0.08f) / 0.92f;
            ageScale = (1.0f - k);
            ageScale = ageScale * ageScale;
        }

        float distFall = 1.0f - dist / reach;
        distFall *= distFall;

        // Lambert N·L — strong when the cloud surface faces the burst.
        // For a burst overhead and a cloud-top normal (~+Y), N·L ≈ 1.
        // For a side-facing normal, N·L falls off naturally.
        float NdotL = max(0.0f, dot(normal, lightDir));

        sum += L.colorAge.rgb * (intensity * NdotL * distFall * ageScale * gain);
    }
    return sum;
}

// ── Main kernel ─────────────────────────────────────────────────────────────

kernel void cloudFloorMeshBuild(
    device       CloudFloorVertex*   verts    [[ buffer(0) ]],
    constant     CloudMeshUniforms&  u        [[ buffer(1) ]],
    device const BurstLight*         bursts   [[ buffer(2) ]],
    texture3d<float, access::sample> noiseTex [[ texture(0) ]],
    uint id [[ thread_position_in_grid ]]
) {
    uint res = as_type<uint>(u.originSizeRes.w);
    uint total = res * res;
    if (id >= total) return;

    uint ix = id % res;
    uint iz = id / res;

    float fx = float(ix) / float(res - 1u);
    float fz = float(iz) / float(res - 1u);
    float ox = u.originSizeRes.x;
    float oz = u.originSizeRes.y;
    float worldSize = u.originSizeRes.z;
    float x = ox + (fx - 0.5f) * worldSize;
    float z = oz + (fz - 0.5f) * worldSize;

    // Central differences for the normal — sample at ±2 vertex widths in
    // each direction. Wider stencil averages over a 4-vertex window which
    // smooths the normal field significantly compared to forward
    // differences. Wider eps = smoother lighting at the cost of slightly
    // softened high-frequency cloud bumps (a feature, not a bug, for
    // cumulus look).
    float eps = (worldSize / float(res - 1u)) * 2.0f;
    float h_here = cloud_height(float2(x, z), u, noiseTex);
    float h_xp   = cloud_height(float2(x + eps, z), u, noiseTex);
    float h_xm   = cloud_height(float2(x - eps, z), u, noiseTex);
    float h_zp   = cloud_height(float2(x, z + eps), u, noiseTex);
    float h_zm   = cloud_height(float2(x, z - eps), u, noiseTex);

    float baseY = u.baseShapeWind.x;
    float y = baseY + h_here;

    // Normal from heightfield gradient (central differences).
    float dhdx = (h_xp - h_xm) / (2.0f * eps);
    float dhdz = (h_zp - h_zm) / (2.0f * eps);
    float3 normal = normalize(float3(-dhdx, 1.0f, -dhdz));

    // Per-vertex burst lighting — Lambert sum from every active burst.
    uint burstCount = as_type<uint>(u.coverageGain.w);
    float gain = u.coverageGain.z;
    float3 emission = cloud_burstLambert(
        float3(x, y, z), normal, bursts, burstCount, gain
    );
    // Soft Reinhard-style tone-map so multiple overlapping bursts stack
    // without clipping to pure white. Single burst stays close to linear;
    // 5 stacked bursts compress gracefully into the bloom-acceptable range.
    emission = emission / (1.0f + emission * 0.5f);

    verts[id].position = float4(x, y, z, 1.0f);
    verts[id].normal   = float4(normal, 0.0f);
    verts[id].emission = float4(emission, 0.0f);
}
