// Daydream DH-1360 — a STILL rendered as independent batches ("render until clean").
//
// The host renders a still as G batches, each a complete still of its own (fresh caches, fresh running
// means, F accumulated frames), and asks this file for three things between frames — never inside one:
//
//   illumi_still_batch_fold    — after a batch: add its mean image (the TAA accumulator, linear HDR,
//                                internal size) into an fp32 sum, and reduce it per TILE (tile edge in
//                                OUTPUT pixels) to the numbers the batch estimator and the tile classifier
//                                read: mean luminance, peak, 4 × 4 sub-block means, and the G-buffer's
//                                normal spread / emission / sky share.
//   illumi_still_display_tiles — the tonemapped picture per tile: mean display luma code, clipped share,
//                                near-black share (the `dark` class is judged in display codes).
//   illumi_still_write_mean    — the fp32 mean of every batch, written back into the accumulator so the
//                                host's next (held) frame prints it.
//
// One threadgroup per tile, so every pixel is visited by exactly one thread of exactly one group and
// the fp32 sum needs no atomics. Self-contained: no shared header (nothing here may change another pass).

#include <metal_stdlib>
using namespace metal;

/// Mirror of `IlluminatoramaStillBatches.Params` — 48 bytes.
struct StillTileParams {
    uint  internalW;  uint  internalH;  uint outputW;  uint outputH;
    uint  tilesX;     uint  tilesY;     float tileSize; float scaleX;   // internal px per output px
    float scaleY;     uint  srgbOutput; uint accumulate; float invBatches;
};

constant float3 kStillLuma = float3(0.2126, 0.7152, 0.0722);
constant uint kStillSub = 4u;                       // sub-blocks per tile edge
// Fold record per tile (floats): 0 lum Σ, 1 pixel count, 2 peak lum, 3–5 normal Σ, 6 geometry count,
// 7 emissive count, 8 sky count, 9–24 sub-block lum Σ, 25–40 sub-block count.
constant uint kStillFoldFields = 41u;
constant uint kStillDisplayFields = 4u;             // code Σ, pixel count, clipped count, dark count
constant uint kStillMaxSimdgroups = 32u;

static inline float3 stillOctDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return normalize(n);
}

/// The tile's pixel rectangle at a size `scale` × the output's: [lo, hi) on each axis.
static inline uint2 stillTileSpan(uint tile, float tileSize, float scale, uint extent, uint tiles) {
    uint lo = min(extent, uint(floor(float(tile) * tileSize * scale)));
    uint hi = (tile + 1u >= tiles) ? extent : min(extent, uint(floor(float(tile + 1u) * tileSize * scale)));
    return uint2(lo, max(lo, hi));
}

kernel void illumi_still_batch_fold(
    texture2d<half, access::read>  history   [[texture(0)]],
    texture2d<half, access::read>  normalRgh [[texture(1)]],
    texture2d<half, access::read>  emission  [[texture(2)]],
    depth2d<float, access::read>   depth     [[texture(3)]],
    device float4*                 sum       [[buffer(0)]],
    device float*                  tiles     [[buffer(1)]],
    constant StillTileParams&      p         [[buffer(2)]],
    uint2 tg      [[threadgroup_position_in_grid]],
    uint2 lid     [[thread_position_in_threadgroup]],
    uint2 tgSize  [[threads_per_threadgroup]],
    uint  sg      [[simdgroup_index_in_threadgroup]],
    uint  lane    [[thread_index_in_simdgroup]],
    uint  sgCount [[simdgroups_per_threadgroup]])
{
    if (tg.x >= p.tilesX || tg.y >= p.tilesY) return;
    uint2 xs = stillTileSpan(tg.x, p.tileSize, p.scaleX, p.internalW, p.tilesX);
    uint2 ys = stillTileSpan(tg.y, p.tileSize, p.scaleY, p.internalH, p.tilesY);
    float v[kStillFoldFields];
    for (uint i = 0u; i < kStillFoldFields; ++i) v[i] = 0.0;
    float spanX = float(max(1u, xs.y - xs.x)), spanY = float(max(1u, ys.y - ys.x));
    for (uint y = ys.x + lid.y; y < ys.y; y += tgSize.y) {
        for (uint x = xs.x + lid.x; x < xs.y; x += tgSize.x) {
            uint2 q = uint2(x, y);
            float3 c = max(float3(0.0), float3(history.read(q).rgb));
            uint idx = y * p.internalW + x;
            sum[idx] = (p.accumulate != 0u ? sum[idx] : float4(0.0)) + float4(c, 0.0);
            float l = dot(c, kStillLuma);
            v[0] += l; v[1] += 1.0; v[2] = max(v[2], l);
            if (depth.read(q) >= 0.99999) {
                v[8] += 1.0;                                    // sky: no geometry, no normal
            } else {
                float3 n = stillOctDecode(float2(normalRgh.read(q).xy));
                v[3] += n.x; v[4] += n.y; v[5] += n.z; v[6] += 1.0;
            }
            float3 e = float3(emission.read(q).rgb);
            if (max(e.r, max(e.g, e.b)) > 1e-4) v[7] += 1.0;
            uint sx = min(kStillSub - 1u, uint(float(x - xs.x) * float(kStillSub) / spanX));
            uint sy = min(kStillSub - 1u, uint(float(y - ys.x) * float(kStillSub) / spanY));
            uint s = sy * kStillSub + sx;
            v[9 + s] += l; v[25 + s] += 1.0;
        }
    }
    threadgroup float partial[kStillMaxSimdgroups * kStillFoldFields];
    for (uint i = 0u; i < kStillFoldFields; ++i) {
        float r = (i == 2u) ? simd_max(v[i]) : simd_sum(v[i]);
        if (lane == 0u && sg < kStillMaxSimdgroups) partial[sg * kStillFoldFields + i] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t = lid.y * tgSize.x + lid.x;
    if (t < kStillFoldFields) {
        float r = 0.0;
        for (uint k = 0u; k < min(sgCount, kStillMaxSimdgroups); ++k) {
            float w = partial[k * kStillFoldFields + t];
            r = (t == 2u) ? max(r, w) : r + w;
        }
        tiles[(tg.y * p.tilesX + tg.x) * kStillFoldFields + t] = r;
    }
}

static inline float stillSRGBEncode(float c) {
    c = saturate(c);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

kernel void illumi_still_display_tiles(
    texture2d<float, access::read> picture [[texture(0)]],
    device float*                  tiles   [[buffer(0)]],
    constant StillTileParams&      p       [[buffer(1)]],
    uint2 tg      [[threadgroup_position_in_grid]],
    uint2 lid     [[thread_position_in_threadgroup]],
    uint2 tgSize  [[threads_per_threadgroup]],
    uint  sg      [[simdgroup_index_in_threadgroup]],
    uint  lane    [[thread_index_in_simdgroup]],
    uint  sgCount [[simdgroups_per_threadgroup]])
{
    if (tg.x >= p.tilesX || tg.y >= p.tilesY) return;
    uint2 xs = stillTileSpan(tg.x, p.tileSize, 1.0, p.outputW, p.tilesX);
    uint2 ys = stillTileSpan(tg.y, p.tileSize, 1.0, p.outputH, p.tilesY);
    float v[kStillDisplayFields] = { 0.0, 0.0, 0.0, 0.0 };
    for (uint y = ys.x + lid.y; y < ys.y; y += tgSize.y) {
        for (uint x = xs.x + lid.x; x < xs.y; x += tgSize.x) {
            float3 c = picture.read(uint2(x, y)).rgb;   // an sRGB format reads back LINEAR: re-encode
            if (p.srgbOutput != 0u) c = float3(stillSRGBEncode(c.r), stillSRGBEncode(c.g), stillSRGBEncode(c.b));
            float code = 255.0 * dot(c, kStillLuma);
            v[0] += code; v[1] += 1.0;
            if (255.0 * max(c.r, max(c.g, c.b)) >= 250.0) v[2] += 1.0;
            if (code <= 2.0) v[3] += 1.0;
        }
    }
    threadgroup float partial[kStillMaxSimdgroups * kStillDisplayFields];
    for (uint i = 0u; i < kStillDisplayFields; ++i) {
        float r = simd_sum(v[i]);
        if (lane == 0u && sg < kStillMaxSimdgroups) partial[sg * kStillDisplayFields + i] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t = lid.y * tgSize.x + lid.x;
    if (t < kStillDisplayFields) {
        float r = 0.0;
        for (uint k = 0u; k < min(sgCount, kStillMaxSimdgroups); ++k) r += partial[k * kStillDisplayFields + t];
        tiles[(tg.y * p.tilesX + tg.x) * kStillDisplayFields + t] = r;
    }
}

kernel void illumi_still_write_mean(
    texture2d<half, access::write> history [[texture(0)]],
    device const float4*           sum     [[buffer(0)]],
    constant StillTileParams&      p       [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.internalW || gid.y >= p.internalH) return;
    float3 m = sum[gid.y * p.internalW + gid.x].rgb * p.invBatches;
    history.write(half4(half3(m), 1.0h), gid);
}
