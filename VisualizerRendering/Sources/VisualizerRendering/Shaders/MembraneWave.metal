#include <metal_stdlib>
using namespace metal;

// ── MembraneWave.metal ───────────────────────────────────────────────────────
//
// A damped 2-D wave equation on a mesh's own (rows × cols) vertex grid — a fluid-
// filled membrane (waterbed, jelly, drum skin). Each vertex carries a displacement
// h along its REST normal and a velocity v:
//
//     v += dt · (c² ∇²h − γ v + f)          (semi-implicit Euler, per substep)
//     h += dt · v
//
// ∇² uses each vertex's own rest spacing to its row / column neighbours (so a
// non-uniform grid propagates at the same physical speed everywhere). `mask`
// pins vertices that aren't part of the membrane (0 = fixed, 1 = free); a pinned
// row is a reflecting boundary, like the rim of a bed. Columns can wrap.
//
// Heights ping-pong between two buffers each substep (no neighbour read-during-write).
// A second kernel writes the displaced positions and recomputes normals from the
// displaced neighbours, into the packed_float3 streams that
// `IlluminatoramaRenderer.registerGPUMesh` repacks and refits.
//
// Swift mirror: `MembraneWaveMesh` (MembraneWaveMesh.swift).

struct MembraneParams {
    uint  rows;
    uint  cols;
    uint  wrapCols;      // 1 ⇒ column c-1 of c=0 is cols-1
    uint  impulseCount;
    float dt;
    float c2;            // wave speed² (units² / s²)
    float damping;       // γ, 1/s
    float uniformForce;  // added to every free vertex's acceleration (e.g. host inertia)
};

struct MembraneImpulse {
    float2 gridPos;      // (row, col) — fractional
    float  radius;       // in grid cells
    float  velocity;     // Δv at the centre (units / s)
};

static inline uint mw_index(uint r, uint c, uint cols) { return r * cols + c; }

kernel void membrane_step(
    const device float*            hIn      [[buffer(0)]],
    device float*                  v        [[buffer(1)]],
    const device float*            mask     [[buffer(2)]],
    const device float2*           invSpace [[buffer(3)]],   // (1/drow², 1/dcol²) per vertex
    constant MembraneParams&       p        [[buffer(4)]],
    const device MembraneImpulse*  imps     [[buffer(5)]],
    device float*                  hOut     [[buffer(6)]],   // ping-pong: no in-place races
    uint gid [[thread_position_in_grid]])
{
    uint n = p.rows * p.cols;
    if (gid >= n) return;
    float m = mask[gid];
    if (m <= 0.0f) { v[gid] = 0.0f; hOut[gid] = 0.0f; return; }
    uint r = gid / p.cols, c = gid % p.cols;
    uint rUp = r + 1 < p.rows ? r + 1 : r;          // Neumann at the grid ends
    uint rDn = r > 0 ? r - 1 : r;
    uint cR = c + 1 < p.cols ? c + 1 : (p.wrapCols != 0u ? 0u : c);
    uint cL = c > 0 ? c - 1 : (p.wrapCols != 0u ? p.cols - 1 : c);
    float hc = hIn[gid];
    // Pinned neighbours read as 0 (a fixed rim reflects the wave back inverted).
    uint iu = mw_index(rUp, c, p.cols), idn = mw_index(rDn, c, p.cols);
    uint ir = mw_index(r, cR, p.cols), il = mw_index(r, cL, p.cols);
    float hu = mask[iu] > 0.0f ? hIn[iu] : 0.0f;
    float hd = mask[idn] > 0.0f ? hIn[idn] : 0.0f;
    float hr = mask[ir] > 0.0f ? hIn[ir] : 0.0f;
    float hl = mask[il] > 0.0f ? hIn[il] : 0.0f;
    float2 is = invSpace[gid];
    float lap = (hu + hd - 2.0f * hc) * is.x + (hr + hl - 2.0f * hc) * is.y;

    float force = p.uniformForce;
    for (uint i = 0; i < p.impulseCount; ++i) {
        MembraneImpulse im = imps[i];
        float dc = abs(float(c) - im.gridPos.y);
        if (p.wrapCols != 0u) dc = min(dc, float(p.cols) - dc);
        float2 d = float2(float(r) - im.gridPos.x, dc) / max(0.5f, im.radius);
        force += im.velocity / p.dt * exp(-dot(d, d));      // Δv spread as a Gaussian
    }
    float vv = v[gid] + p.dt * (p.c2 * lap - p.damping * v[gid] + force * m);
    float hn = hc + p.dt * vv * m;
    // Guard: a non-finite value would spread across the whole membrane in a few steps.
    if (!isfinite(vv) || !isfinite(hn)) { vv = 0.0f; hn = 0.0f; }
    v[gid] = vv;
    hOut[gid] = hn;
}

static inline float3 mw_displaced(const device packed_float3* restPos, const device packed_float3* restNrm,
                                  const device float* h, uint k) {
    return float3(restPos[k]) + float3(restNrm[k]) * h[k];
}

kernel void membrane_displace(
    const device packed_float3*  restPos  [[buffer(0)]],
    const device packed_float3*  restNrm  [[buffer(1)]],
    const device float*          h        [[buffer(2)]],
    constant MembraneParams&     p        [[buffer(3)]],
    constant uint&               total    [[buffer(4)]],   // grid + any extra (pinned) vertices
    device packed_float3*        outPos   [[buffer(5)]],
    device packed_float3*        outNrm   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    uint n = p.rows * p.cols;
    float3 rp = float3(restPos[gid]);
    float3 rn = float3(restNrm[gid]);
    if (gid >= n) { outPos[gid] = packed_float3(rp); outNrm[gid] = packed_float3(rn); return; }
    uint r = gid / p.cols, c = gid % p.cols;
    float3 pc = mw_displaced(restPos, restNrm, h, gid);
    uint rUp = r + 1 < p.rows ? r + 1 : r;
    uint rDn = r > 0 ? r - 1 : r;
    uint cR = c + 1 < p.cols ? c + 1 : (p.wrapCols != 0u ? 0u : c);
    uint cL = c > 0 ? c - 1 : (p.wrapCols != 0u ? p.cols - 1 : c);
    float3 tr = mw_displaced(restPos, restNrm, h, mw_index(rUp, c, p.cols))
              - mw_displaced(restPos, restNrm, h, mw_index(rDn, c, p.cols));
    float3 tc = mw_displaced(restPos, restNrm, h, mw_index(r, cR, p.cols))
              - mw_displaced(restPos, restNrm, h, mw_index(r, cL, p.cols));
    float3 nn = cross(tc, tr);
    float l2 = dot(nn, nn);
    nn = l2 > 1e-20f ? nn * rsqrt(l2) : rn;
    if (dot(nn, rn) < 0.0f) nn = -nn;
    // Blend toward the rest normal where the grid is degenerate (poles, seams).
    nn = normalize(mix(rn, nn, saturate(sqrt(l2) * 4.0f)));
    outPos[gid] = packed_float3(pc);
    outNrm[gid] = packed_float3(nn);
}
