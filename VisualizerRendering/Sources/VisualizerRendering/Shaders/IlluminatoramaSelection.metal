// ── ILLUMINATORAMA — SELECTION OUTLINE ──────────────────────────────────────
//
// Selection mask rasterisation, separable dilation, and the outline composite.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Highlight outline mask pass ───────────────────────────────────────────────
//
// One render pipeline feeds a separable max-filter dilation + composite. Every
// highlighted element (selected OR hovered) supplies a SOLID bounding-box proxy
// to the mask buffer — never its detailed mesh. A box is a closed solid, so its
// screen-space mask has no internal holes; dilate − original then yields a clean
// outer ring with no internal edges and no fill. (Rasterizing the real mesh —
// open sofa frames, see-through bookshelf bays — produced internal rings and,
// once the small gaps merged under dilation, a full-object wash.)
//
//   illumi_selection_box_vs/fs   — draws boxes whose inst.highlight == wantMode.
//
// Then two compute passes:
//   illumi_selection_dilate_h/v  — separable max-filter dilation by `radius` px.
//   illumi_selection_composite   — ring = dilated − original; additive HDR blend.
//
// The pass runs once per mode: wantMode 1 (selected, blue) then 2 (hover, yellow).

struct SelMaskVSOut {
    float4 position [[position]];
};

// Mask boxes, filtered to the mode being composited this invocation: an instance
// whose highlight tag differs from wantMode is clipped (NDC z > 1 → zero fragments).
vertex SelMaskVSOut illumi_selection_box_vs(
    uint                       vid       [[vertex_id]],
    uint                       iid       [[instance_id]],
    const device Vertex*       verts     [[buffer(0)]],
    constant FrameUniforms&    frame     [[buffer(1)]],
    const device Instance*     instances [[buffer(2)]],
    constant int&              wantMode  [[buffer(3)]]
) {
    if (instances[iid].highlight != wantMode) {
        return { float4(2, 2, 2, 1) };   // outside clip → zero fragments
    }
    float4 worldP = instances[iid].modelMatrix * float4(verts[vid].position, 1.0);
    return { frame.viewProjection * worldP };
}

fragment float illumi_selection_mask_fs(SelMaskVSOut in [[stage_in]]) {
    return 1.0;
}

// Horizontal max-filter dilation pass (reads selectionMask, writes dilateH).
kernel void illumi_selection_dilate_h(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int&                   radius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(inTex.get_width()), h = int(inTex.get_height());
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float v = 0;
    for (int dx = -radius; dx <= radius; dx++) {
        int px = clamp(int(gid.x) + dx, 0, w - 1);
        v = max(v, inTex.read(uint2(px, gid.y)).r);
    }
    outTex.write(float4(v, 0, 0, 0), gid);
}

// Vertical max-filter dilation pass (reads dilateH, writes dilatedFinal).
kernel void illumi_selection_dilate_v(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant int&                   radius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int w = int(inTex.get_width()), h = int(inTex.get_height());
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    float v = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        int py = clamp(int(gid.y) + dy, 0, h - 1);
        v = max(v, inTex.read(uint2(gid.x, py)).r);
    }
    outTex.write(float4(v, 0, 0, 0), gid);
}

// Composites ring = (dilated − original) additively into the HDR texture.
struct SelectionOutlineParams {
    float4 colorIntensity;   // xyz = glow color, w = intensity (push past bloom threshold)
    int    width;
    int    height;
    int    _pad0;
    int    _pad1;
};

kernel void illumi_selection_composite(
    texture2d<float, access::read>       maskTex    [[texture(0)]],
    texture2d<float, access::read>       dilatedTex [[texture(1)]],
    texture2d<float, access::read_write> hdrTex     [[texture(2)]],
    constant SelectionOutlineParams&     p          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (int(gid.x) >= p.width || int(gid.y) >= p.height) return;
    float orig   = maskTex.read(gid).r;
    float dilated = dilatedTex.read(gid).r;
    float ring   = saturate(dilated - orig);
    float4 hdr   = hdrTex.read(gid);
    float3 glow  = ring * p.colorIntensity.xyz * p.colorIntensity.w;
    hdrTex.write(float4(hdr.rgb + glow, hdr.a), gid);
}
