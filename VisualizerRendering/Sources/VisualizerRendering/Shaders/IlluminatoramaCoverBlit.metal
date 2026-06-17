#include <metal_stdlib>
using namespace metal;

// Fullscreen cover-blit — used by IlluminatoramaRenderer.present(to:) to
// display the fixed-canvas output texture directly into an MTKView drawable
// without going through SceneKit. Samples the source with a cover-fill UV
// crop (scaleU, scaleV, offsetU, offsetV) so the image fills the drawable
// without stretching or letterboxing, matching what the SceneKit
// contentsTransform path was doing.

struct CoverBlitVarying {
    float4 position [[position]];
    float2 texCoord;
};

// One oversized triangle covering the clip-space [-1,1]² quad.
// vid 0: (-1,-1,0,1), vid 1: (3,-1,0,1), vid 2: (-1,3,0,1)
vertex CoverBlitVarying illumi_coverblit_vs(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 1) ? 3.0 : -1.0,
                        (vid == 2) ? 3.0 : -1.0);
    // Metal NDC: Y-up.  Texture UV: Y-down (0=top, 1=bottom).
    // Screen top (NDC +1) → uv.y = 0; bottom (NDC -1) → uv.y = 1.
    float2 uv = float2(pos.x * 0.5 + 0.5,
                       1.0 - (pos.y * 0.5 + 0.5));
    CoverBlitVarying out;
    out.position = float4(pos, 0.0, 1.0);
    out.texCoord = uv;
    return out;
}

// uvRect: (scaleU, scaleV, offsetU, offsetV) from
// IlluminatoramaRenderer.coverUVRect(viewportWidth:viewportHeight:).
// Samples the fixed render canvas with a centred cover crop so the
// drawable is filled without stretching.
//
// sRGB note: src is bgra8Unorm_srgb → Metal auto-decodes stored bytes as
// sRGB → linear before handing them to the shader. Writing to a
// bgra8Unorm_srgb render target auto-encodes linear → sRGB. The round-trip
// is transparent; colours match what SceneKit's background.contentsTransform
// path produced.
fragment float4 illumi_coverblit_fs(CoverBlitVarying    in      [[stage_in]],
                                    texture2d<float>     src     [[texture(0)]],
                                    constant float4&     uvRect  [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = in.texCoord * uvRect.xy + uvRect.zw;
    return src.sample(s, uv);
}
