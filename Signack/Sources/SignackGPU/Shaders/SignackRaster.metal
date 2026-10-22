// ── SIGNACK S4 — SOFTWARE (COMPUTE) RASTERIZER FOR SUB-PIXEL TRIANGLES (Daydream DH-1461) ──────
//
// A triangle smaller than a pixel still costs the hardware rasterizer a whole 2×2 quad of
// fragment work (and most of them cover no sample at all). Signack routes on-screen clusters that
// project smaller than `softRasterDiameter` px to this rasterizer instead (`signack_cut_classify`,
// flag 3), and the host's vertex stage turns their triangles into WINDOW-SPACE triangles in
// `SignackRasterTriangle` records (the same deformation its hardware vertex stage runs — so both
// paths see one surface). Then:
//
//   1. signack_raster_prepare   one thread: this frame's triangle count → the indirect dispatch;
//                               re-arms the append counter for the next frame.
//   2. signack_raster_depth     one thread per triangle: snap, coverage, depth → atomic MIN of the
//                               depth's float bits per pixel (non-negative floats order as uints).
//   3. signack_raster_ids       one thread per triangle, same coverage + depth: where its depth is
//                               the pixel's minimum, store its id.
//
// Two passes because the GPU this was built on (Apple7 / M1) has no 64-bit atomics — a packed
// (depth << 32 | id) atomic MIN, Nanite's single pass, is refused by the driver ("Unsupported i64
// atomic operation"). Depth is recomputed bit-identically in pass 3 (same function, explicit fma,
// precise divide), so the equality test is exact; equal-depth ties go to whichever store lands
// last — the same coplanar-tie race the S2 compaction already has.
//
// THE COVERAGE RULE is the hardware's, so a soft triangle and a hardware triangle that share an
// edge neither double-cover nor leave a crack between them (`SignackRasterParityTests` checks it
// against the real rasterizer, pixel for pixel):
//   - vertices snapped to the rasterizer's 1/256-pixel grid (round to nearest);
//   - sample at pixel centres (x + ½, y + ½), window space, y down;
//   - exact integer edge functions (64-bit), top-left fill rule on ties;
//   - front-facing = counter-clockwise in clip space (the host draws with `.counterClockwise`).
// Depth is the screen-linear interpolation of z/w — the hardware's plane, within rounding.
//
// OUTPUT CONTRACT (read by the host's merge): `depth[y·width + x]` = float bits of the nearest
// soft depth, 0xFFFFFFFF where no soft triangle covered the pixel; `ids[…]` = that triangle's id
// (host x with bit 31 replaced by front-facing, host y). SELF-CLEARING: the consumer resets every
// pixel it reads to 0xFFFFFFFF, so no clear pass ever runs; writes are confined to the scissor the
// consumer reads.
//
// Self-contained (no engine header), like SignackCut.metal; mirrors in SignackRasterTypes.swift.

#include <metal_stdlib>
using namespace metal;

struct SignackRasterTriangle {         // 48 B — mirror of Swift SignackRasterTriangle
    packed_float3 v0, v1, v2;          // window x, y (px, y down, origin top-left), z/w ∈ [0, 1]
    uint idX;                          // IN: host id, bit 31 = cull back faces; OUT bit 31 = front
    uint idY;
    uint flags;                        // reserved (0)
};
static_assert(sizeof(SignackRasterTriangle) == 48, "mirror of Swift SignackRasterTriangle");

struct SignackRasterUniforms {         // 32 B — mirror of Swift SignackRasterUniforms
    uint width, height;                // the target (px)
    uint x0, y0, x1, y1;               // scissor: writes confined to [x0, x1) × [y0, y1)
    uint capacity;                     // triangles the list holds
    uint maxSpan;                      // per-axis loop cap (px) — a larger triangle is clipped, counted
};
static_assert(sizeof(SignackRasterUniforms) == 32, "mirror of Swift SignackRasterUniforms");

// state[0] append counter (host vertex stage) · [1] this frame's triangle count · [2] dropped
// (over capacity, cumulative) · [3] clipped by maxSpan (cumulative) · dispatch args at byte 32.
constant uint SIGNACK_RASTER_THREADS = 64;   // SignackRaster.threadsPerGroup
constant uint SIGNACK_EMPTY = 0xFFFFFFFFu;

static inline long signack_edge(int2 a, int2 b, int2 p) {
    return long(b.x - a.x) * long(p.y - a.y) - long(b.y - a.y) * long(p.x - a.x);
}

/// Top-left rule for a triangle oriented so its interior is on the POSITIVE side of every edge
/// (clockwise on screen, y down): a top edge runs exactly horizontal left → right, a left edge runs
/// upward. A sample exactly on an edge belongs to the triangle only across a top or left edge.
static inline long signack_bias(int2 a, int2 b) {
    int2 d = b - a;
    return ((d.y == 0 && d.x > 0) || d.y < 0) ? 0 : -1;
}

static inline int signack_floor_div256(int v) { return v >= 0 ? v / 256 : -((-v + 255) / 256); }

/// Rasterise one triangle; `op(pixelIndex, depthBits, front)` per covered sample.
template <typename Op>
static inline void signack_raster(SignackRasterTriangle T, constant SignackRasterUniforms& U,
                                  device atomic_uint* state, thread Op& op) {
    float3 a = float3(T.v0), b = float3(T.v1), c = float3(T.v2);
    const float lim = 1048576.0f;   // 2^20 px: beyond it the 1/256 grid no longer fits an int
    if (any(fabs(float4(a.xy, b.xy)) > lim) || any(fabs(c.xy) > lim)) return;
    int2 s0 = int2(rint(a.xy * 256.0f)), s1 = int2(rint(b.xy * 256.0f)), s2 = int2(rint(c.xy * 256.0f));
    float z0 = a.z, z1 = b.z, z2 = c.z;
    long area = signack_edge(s0, s1, s2);
    if (area == 0) return;                       // the hardware draws nothing either
    const bool front = area < 0;                 // CCW on screen (y down) — the hardware's front, measured
    if ((T.idX & 0x80000000u) != 0u && !front) return;
    if (area < 0) { int2 t = s1; s1 = s2; s2 = t; float tz = z1; z1 = z2; z2 = tz; area = -area; }

    int minX = min3(s0.x, s1.x, s2.x), maxX = max3(s0.x, s1.x, s2.x);
    int minY = min3(s0.y, s1.y, s2.y), maxY = max3(s0.y, s1.y, s2.y);
    // Pixel centres p·256 + 128 inside the box: ceil((min − 128)/256) … floor((max − 128)/256).
    int px0 = -signack_floor_div256(128 - minX), px1 = signack_floor_div256(maxX - 128);
    int py0 = -signack_floor_div256(128 - minY), py1 = signack_floor_div256(maxY - 128);
    px0 = max(px0, int(U.x0)); py0 = max(py0, int(U.y0));
    px1 = min(px1, int(U.x1) - 1); py1 = min(py1, int(U.y1) - 1);
    if (px1 < px0 || py1 < py0) return;          // covers no sample — most sub-pixel triangles
    if (px1 - px0 >= int(U.maxSpan) || py1 - py0 >= int(U.maxSpan)) {
        atomic_fetch_add_explicit(&state[3], 1u, memory_order_relaxed);
        px1 = min(px1, px0 + int(U.maxSpan) - 1);
        py1 = min(py1, py0 + int(U.maxSpan) - 1);
    }
    const long b12 = signack_bias(s1, s2), b20 = signack_bias(s2, s0), b01 = signack_bias(s0, s1);
    const float invArea = precise::divide(1.0f, float(area));
    for (int py = py0; py <= py1; ++py) {
        for (int px = px0; px <= px1; ++px) {
            int2 p = int2(px * 256 + 128, py * 256 + 128);
            long w0 = signack_edge(s1, s2, p), w1 = signack_edge(s2, s0, p), w2 = signack_edge(s0, s1, p);
            if (w0 + b12 < 0 || w1 + b20 < 0 || w2 + b01 < 0) continue;
            // Screen-linear depth (z/w is affine in window space) — explicit fma so the two passes
            // round identically.
            float l1 = float(w1) * invArea, l2 = float(w2) * invArea;
            float z = fma(l2, z2 - z0, fma(l1, z1 - z0, z0));
            if (!(z >= 0.0f && z <= 1.0f)) continue;   // the hardware clips outside [0, 1]
            op(uint(py) * U.width + uint(px), as_type<uint>(z), front);
        }
    }
}

kernel void signack_raster_prepare(
    constant SignackRasterUniforms& U     [[buffer(0)]],
    device atomic_uint*             state [[buffer(1)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i != 0) return;
    uint appended = atomic_load_explicit(&state[0], memory_order_relaxed);
    uint n = min(appended, U.capacity);
    atomic_store_explicit(&state[1], n, memory_order_relaxed);
    if (appended > n) atomic_fetch_add_explicit(&state[2], appended - n, memory_order_relaxed);
    atomic_store_explicit(&state[0], 0u, memory_order_relaxed);
    device uint* dispatch = (device uint*)((device uchar*)state + 32);
    dispatch[0] = (n + SIGNACK_RASTER_THREADS - 1u) / SIGNACK_RASTER_THREADS;
    dispatch[1] = 1u; dispatch[2] = 1u;
}

struct SignackDepthOp {
    device atomic_uint* depth;
    void operator()(uint i, uint z, bool) thread { atomic_fetch_min_explicit(&depth[i], z, memory_order_relaxed); }
};

struct SignackIDOp {
    device atomic_uint* depth;
    device uint2*       ids;
    uint idX, idY;
    void operator()(uint i, uint z, bool front) thread {
        if (atomic_load_explicit(&depth[i], memory_order_relaxed) == z)
            ids[i] = uint2((idX & 0x7FFFFFFFu) | (front ? 0x80000000u : 0u), idY);
    }
};

kernel void signack_raster_depth(
    constant SignackRasterUniforms&       U     [[buffer(0)]],
    device atomic_uint*                   state [[buffer(1)]],
    device const SignackRasterTriangle*   tris  [[buffer(2)]],
    device atomic_uint*                   depth [[buffer(3)]],
    uint t                                      [[thread_position_in_grid]])
{
    if (t >= atomic_load_explicit(&state[1], memory_order_relaxed)) return;
    SignackDepthOp op = { depth };
    signack_raster(tris[t], U, state, op);
}

kernel void signack_raster_ids(
    constant SignackRasterUniforms&       U     [[buffer(0)]],
    device atomic_uint*                   state [[buffer(1)]],
    device const SignackRasterTriangle*   tris  [[buffer(2)]],
    device atomic_uint*                   depth [[buffer(3)]],
    device uint2*                         ids   [[buffer(4)]],
    uint t                                      [[thread_position_in_grid]])
{
    if (t >= atomic_load_explicit(&state[1], memory_order_relaxed)) return;
    SignackRasterTriangle T = tris[t];
    SignackIDOp op = { depth, ids, T.idX, T.idY };
    signack_raster(T, U, state, op);
}

// For a host that has no merge of its own (and for the tests): reset the scissor to empty.
kernel void signack_raster_clear(
    constant SignackRasterUniforms& U     [[buffer(0)]],
    device uint*                    depth [[buffer(3)]],
    uint2 g                               [[thread_position_in_grid]])
{
    uint x = U.x0 + g.x, y = U.y0 + g.y;
    if (x < U.x1 && y < U.y1) depth[y * U.width + x] = SIGNACK_EMPTY;
}
