// ── SIGNACK — HI-Z DEPTH PYRAMID (Daydream DH-1466) ───────────────────────────────────────────────
//
// A MAX-depth mip pyramid of the host's G-buffer depth (standard depth: 0 near, 1 far, cleared 1),
// for occlusion culling (`signack_occluded` in SignackCut.metal). Level 0 is half the depth's size
// rounded up to a power of two (texels past the edge repeat it — off screen), each further level
// exactly half; texel (x, y) of level L+1 is the max of level L's texels [2x, 2x+1] × [2y, 2y+1], so a
// texel's value is the FARTHEST depth over its whole 2^(L+1)-px footprint. A
// cluster whose nearest point is farther than that is hidden behind everything in the footprint.
//
// Self-contained (no includes) so the sources concatenate into one runtime library.

#include <metal_stdlib>
using namespace metal;

kernel void signack_hiz_seed(
    depth2d<float, access::read>      depth  [[texture(0)]],
    texture2d<float, access::write>   dst    [[texture(1)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 lim = uint2(depth.get_width() - 1, depth.get_height() - 1);
    uint2 s = gid * 2u;
    float m = max(max(depth.read(min(s, lim)), depth.read(min(s + uint2(1, 0), lim))),
                  max(depth.read(min(s + uint2(0, 1), lim)), depth.read(min(s + uint2(1, 1), lim))));
    dst.write(float4(m), gid);
}

kernel void signack_hiz_reduce(
    texture2d<float, access::read>    src    [[texture(0)]],
    texture2d<float, access::write>   dst    [[texture(1)]],
    uint2 gid                                [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 lim = uint2(src.get_width() - 1, src.get_height() - 1);
    uint2 s = gid * 2u;
    float m = max(max(src.read(min(s, lim)).r, src.read(min(s + uint2(1, 0), lim)).r),
                  max(src.read(min(s + uint2(0, 1), lim)).r, src.read(min(s + uint2(1, 1), lim)).r));
    dst.write(float4(m), gid);
}
