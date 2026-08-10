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
//   illumi_selection_dilate_h/v  — separable dilation by `radius` px, ATTENUATED with
//                                  distance so the result is a distance ramp, not a plateau.
//   illumi_selection_composite   — ring = dilated − original; additive HDR blend, normalised
//                                  by the frame's exposure so the halo's DISPLAYED brightness
//                                  is the same in a night scene and a noon one.
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

// Distance attenuation for one axis of the separable dilation: a mask sample `k` texels away
// arrives at strength `1 − |k|/(radius+1)`. A PLAIN max filter (the original) hands back 1.0
// everywhere within the radius, so `dilated − mask` is a hard-edged band of uniform full
// intensity `radius` px wide — the "thick" half of the complaint, and the reason the ring had
// to be blasted past the bloom threshold to look like a glow at all. With the ramp, the ring
// is brightest against the silhouette and reaches zero at `radius+1`: a real feather, made by
// the outline pass itself instead of borrowed from bloom.
inline float sel_falloff(int k, int radius) {
    return 1.0 - float(abs(k)) / float(radius + 1);
}

// Horizontal dilation pass (reads selectionMask, writes dilateH).
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
        v = max(v, inTex.read(uint2(px, gid.y)).r * sel_falloff(dx, radius));
    }
    outTex.write(float4(v, 0, 0, 0), gid);
}

// Vertical dilation pass (reads dilateH, writes dilatedFinal).
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
        v = max(v, inTex.read(uint2(gid.x, py)).r * sel_falloff(dy, radius));
    }
    outTex.write(float4(v, 0, 0, 0), gid);
}

// Composites ring = (dilated − original) additively into the HDR texture.
struct SelectionOutlineParams {
    float4 colorIntensity;   // xyz = glow colour, w = PEAK intensity in tonemapper-input units
    int    width;
    int    height;
    float  ringGain;         // renormalises the attenuated ramp so its first step is a full 1.0
    float  manualExposure;   // `frame.exposure` — the multiplicative comp on top of auto-exposure
    int    autoExposure;     // != 0 → take the auto-exposure base from the state buffer, as the
                             //        tonemap does; 0 → the manual base of 1.0
    int    _pad0;
};

// The auto-exposure state buffer, exactly as `IlluminatoramaTonemap.metal` lays it out. The
// estimator kernel runs EARLIER in this frame's command buffer than the outline pass, so
// `smoothedExposure` here is the very value the tonemap will apply a few passes later.
struct SelExposureState {
    float prevTargetLogLum;
    float smoothedExposure;
    float newTargetLogLum;
    float deltaTime;
};

kernel void illumi_selection_composite(
    texture2d<float, access::read>       maskTex    [[texture(0)]],
    texture2d<float, access::read>       dilatedTex [[texture(1)]],
    texture2d<float, access::read_write> hdrTex     [[texture(2)]],
    constant SelectionOutlineParams&     p          [[buffer(0)]],
    device SelExposureState&             expo       [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (int(gid.x) >= p.width || int(gid.y) >= p.height) return;
    float orig   = maskTex.read(gid).r;
    float dilated = dilatedTex.read(gid).r;
    // `ringGain` puts the step immediately outside the silhouette back at 1.0 — the ramp above
    // costs it `1/(radius+1)` — so the halo's peak brightness is the intensity that was asked
    // for and does not quietly change when the width does.
    float ring   = saturate((dilated - orig) * p.ringGain);
    // The halo is composited BEFORE the tonemapper multiplies the frame by its exposure, so a
    // fixed additive value renders at a brightness that swings with the scene: blown white in a
    // dim room the auto-exposure is pushing up, washed out at noon. Dividing it out here makes
    // `intensity` mean the same displayed glow in every scene — the same units correction that
    // put `width` in output pixels.
    float autoBase = (p.autoExposure != 0) ? expo.smoothedExposure : 1.0;
    float exposure = autoBase * p.manualExposure;
    float4 hdr   = hdrTex.read(gid);
    float3 glow  = ring * p.colorIntensity.xyz * (p.colorIntensity.w / max(exposure, 1e-4));
    hdrTex.write(float4(hdr.rgb + glow, hdr.a), gid);
}
