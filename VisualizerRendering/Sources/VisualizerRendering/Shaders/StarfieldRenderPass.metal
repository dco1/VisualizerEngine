// ── STARFIELD RENDER PASS SHADERS ───────────────────────────────────────────
//
// Vertex + fragment for the StarfieldRenderPass — expands per-corner
// vertices into screen-aligned billboards, draws each star as a tiny
// soft point via a Gaussian falloff computed in the fragment shader
// (no sprite texture needed — saves a sampler binding).

#include <metal_stdlib>
using namespace metal;

struct VSIn {
    float4 positionSize [[ attribute(0) ]];  // xyz = world, w = billboard half-size
    float4 color        [[ attribute(1) ]];  // rgb = HDR colour, a = corner idx 0..3
};

struct VSOut {
    float4 position [[ position ]];
    float3 color;
    float2 uv;       // -1..+1 across the billboard, for the Gaussian falloff
};

vertex VSOut starfieldVS(
    VSIn vin [[ stage_in ]],
    constant float4x4& viewProjection [[ buffer(1) ]],
    constant float4&   cameraRight    [[ buffer(2) ]],
    constant float4&   cameraUp       [[ buffer(3) ]]
) {
    int cornerIdx = int(vin.color.a);
    float2 cornerSign;
    float2 uv;
    if (cornerIdx == 0)      { cornerSign = float2( 1,  1); uv = float2( 1,  1); }
    else if (cornerIdx == 1) { cornerSign = float2(-1,  1); uv = float2(-1,  1); }
    else if (cornerIdx == 2) { cornerSign = float2(-1, -1); uv = float2(-1, -1); }
    else                     { cornerSign = float2( 1, -1); uv = float2( 1, -1); }

    float halfSize = vin.positionSize.w;
    float3 centre = vin.positionSize.xyz;
    float3 right = cameraRight.xyz * (halfSize * cornerSign.x);
    float3 up    = cameraUp.xyz    * (halfSize * cornerSign.y);
    float3 worldPos = centre + right + up;

    VSOut o;
    o.position = viewProjection * float4(worldPos, 1.0f);
    o.color = vin.color.rgb;
    o.uv = uv;
    return o;
}

fragment float4 starfieldFS(VSOut vin [[ stage_in ]]) {
    // Gaussian falloff — uv in [-1, +1], radius = length(uv).
    float r2 = dot(vin.uv, vin.uv);
    if (r2 >= 1.0f) discard_fragment();
    // Tight bright core + soft halo, both Gaussian. Matches the spark
    // sprite's profile so stars and sparks read as the same family of
    // bright points.
    float core = exp(-r2 * 22.0f);
    float halo = exp(-r2 *  3.2f) * 0.55f;
    float falloff = min(1.0f, core + halo);
    return float4(vin.color * falloff, 1.0f);
}
