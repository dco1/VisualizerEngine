#include <metal_stdlib>
using namespace metal;

// ── ArcTube.metal ─────────────────────────────────────────────────────────────
//
// Constant-length "rubber-hose" tubes swept along circular arcs (ArcTubeBatch.swift).
// Each tube is a circular arc of fixed length between two points (or a straight run at
// full reach) with its exact moving frame; every vertex is closed-form, so one thread
// per vertex writes packed position + normal straight into the batch's shared buffers,
// which Illuminatorama repacks (and diff-captures for motion vectors) like any GPU mesh.

// Mirror of `ArcTubeBatch.Arc` (4 × float4, 64 bytes — the float4 ALIGNMENT RULE).
struct ArcTubeArc {
    float4 start;   // xyz = root point,        w = tube radius
    float4 chord;   // xyz = unit chord e,      w = arc length L
    float4 bow;     // xyz = unit bow u,        w = half-turn θ (0 = straight)
    float4 centre;  // xyz = arc centre,        w = arc radius R
};

// Mirror of `ArcTubeBatch.Params`.
struct ArcTubeParams {
    uint tubeCount;     // live tubes this dispatch; slots beyond collapse
    uint maxTubes;
    uint rings;         // rings along each tube
    uint radial;        // vertices round each ring
};

kernel void arcTubeSweep(constant ArcTubeArc*    arcs   [[buffer(0)]],
                         device packed_float3*   outPos [[buffer(1)]],
                         device packed_float3*   outNrm [[buffer(2)]],
                         constant ArcTubeParams& p      [[buffer(3)]],
                         uint vid [[thread_position_in_grid]]) {
    uint perTube = p.rings * p.radial;
    uint tube = vid / perTube;
    if (tube >= p.maxTubes) return;
    if (tube >= p.tubeCount) {
        // A free slot draws as zero-area triangles far below the world.
        outPos[vid] = packed_float3(0.0f, -100000.0f, 0.0f);
        outNrm[vid] = packed_float3(0.0f, 1.0f, 0.0f);
        return;
    }
    ArcTubeArc a = arcs[tube];
    uint local = vid - tube * perTube;
    uint i = local / p.radial;
    uint j = local - i * p.radial;
    float t = float(i) / float(max(p.rings, 2u) - 1u);

    float3 e = a.chord.xyz, u = a.bow.xyz;
    float3 b = cross(e, u);
    float th = a.bow.w;
    float3 c, n0;
    if (th > 1e-4f) {
        float phi = -th + 2.0f * th * t;
        n0 = e * sin(phi) + u * cos(phi);
        c = a.centre.xyz + n0 * a.centre.w;
    } else {
        n0 = u;
        c = a.start.xyz + e * (a.chord.w * t);
    }
    // (n0, b, tangent) is right-handed — n0 × b = tangent — so the ring runs n0 → b and the
    // batch's index pattern winds counter-clockwise seen from outside.
    float ang = float(j) / float(p.radial) * 2.0f * M_PI_F;
    float3 n = n0 * cos(ang) + b * sin(ang);
    outPos[vid] = packed_float3(c + n * a.start.w);
    outNrm[vid] = packed_float3(n);
}
