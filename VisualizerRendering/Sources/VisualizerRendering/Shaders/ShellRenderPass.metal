// ── SHELL RENDER PASS SHADERS ───────────────────────────────────────────────
//
// Vertex + fragment for the ShellRenderPass — expands per-corner billboard
// vertices into screen-aligned quads, samples the soft-sparkle sprite.
// Writes both colour AND depth so downstream passes (volumetric cloud,
// other depth-correlated effects) can use shells as depth occluders.

#include <metal_stdlib>
using namespace metal;

struct VSIn {
    float4 positionRadius [[ attribute(0) ]];  // xyz = world centre, w = billboard radius
    float4 color          [[ attribute(1) ]];  // rgb = HDR colour, a = corner index 0..7
};

struct VSOut {
    float4 position [[ position ]];
    float3 color;
    float2 uv;
};

vertex VSOut shellPassVS(
    VSIn vin [[ stage_in ]],
    constant float4x4& viewProjection [[ buffer(1) ]],
    constant float4&   cameraRight    [[ buffer(2) ]],
    constant float4&   cameraUp       [[ buffer(3) ]]
) {
    // The corner index (color.a) lives in 0..7. 0..3 = core billboard
    // corners (clockwise from top-right), 4..7 = halo billboard corners.
    // We mod 4 to get the corner within a billboard.
    int cornerIdx = int(vin.color.a) % 4;

    // Corner offsets in normalised billboard space:
    //   0 = top-right    (+right, +up)
    //   1 = top-left     (-right, +up)
    //   2 = bottom-left  (-right, -up)
    //   3 = bottom-right (+right, -up)
    float2 cornerSign;
    float2 uv;
    if (cornerIdx == 0) { cornerSign = float2( 1,  1); uv = float2(1, 1); }
    else if (cornerIdx == 1) { cornerSign = float2(-1,  1); uv = float2(0, 1); }
    else if (cornerIdx == 2) { cornerSign = float2(-1, -1); uv = float2(0, 0); }
    else                     { cornerSign = float2( 1, -1); uv = float2(1, 0); }

    float radius = vin.positionRadius.w;
    float3 centre = vin.positionRadius.xyz;
    float3 right = cameraRight.xyz * (radius * cornerSign.x);
    float3 up    = cameraUp.xyz    * (radius * cornerSign.y);
    float3 worldPos = centre + right + up;

    VSOut o;
    o.position = viewProjection * float4(worldPos, 1.0f);
    o.color = vin.color.rgb;
    o.uv = uv;
    return o;
}

constexpr sampler spriteSampler(filter::linear, address::clamp_to_edge);

fragment float4 shellPassFS(
    VSOut vin [[ stage_in ]],
    texture2d<float, access::sample> sprite [[ texture(0) ]]
) {
    // The soft sparkle sprite carries falloff in RGB (additive blend
    // ignores alpha). Vertex colour modulates it. We boost intensity
    // here so the rising shell heads pop as clear light sources against
    // the cumulus body — keep them slightly tamer than the streak
    // sparks so the burst (when it fires) feels like a brighter event
    // than the rise.
    float4 spriteColor = sprite.sample(spriteSampler, vin.uv);
    // DISCARD low-coverage corners of the billboard so DEPTH isn't
    // written there. The pipeline has depth-write ON (so the cloud
    // raymarch can use shells as occluders), but the billboard quad
    // is square while the sprite falloff is round + soft — without
    // a discard the dim corners of every billboard write a "shell is
    // at this depth" stamp that the cloud kernel then uses to cut its
    // march short, leaving a square black halo around each shell head
    // where the cloud would have rendered. Discarding sub-threshold
    // pixels keeps depth-write confined to where the shell actually
    // contributes light.
    float coverage = max(spriteColor.r, max(spriteColor.g, spriteColor.b));
    if (coverage < 0.03f) {
        discard_fragment();
    }
    float intensity = 2.8;
    float3 hdr = spriteColor.rgb * vin.color * intensity;
    return float4(hdr, 1.0f);
}
