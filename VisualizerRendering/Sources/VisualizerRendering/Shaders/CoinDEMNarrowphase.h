// ══════════════════════════════════════════════════════════════════════════════
// CoinDEMNarrowphase.h — the EXACT polytope narrowphase (stage B1, 2026-09-27)
// ══════════════════════════════════════════════════════════════════════════════
//
// Included by CoinDEM.metal after the Stage-1 contact helpers (cdEmitContact, the
// manifold reduction, …) — it uses them, so it is not a standalone header.
//
// WHAT IT OWNS. Every contact that involves a POLYTOPE — a box, a convex hull that
// carries face topology (CoinHullTopology), or a child box of a COMPOUND (shape tag 6) —
// against another polytope, a swept sphere (sphere / capsule / egg) or (a compound child
// only) a disc, plus polytope bodies against static boxes. coinGenerateContacts' broadphase
// LISTS each such piece pair instead of generating it (cdAppendPolyPairs /
// cdAppendStaticBoxPairs), and coinPolyNarrow — one thread per listed pair, dispatched
// indirectly on the same encoder — runs its narrowphase into the same contact buffer.
// cdPairIsPoly is the single predicate, so every pair is generated once, and a world with
// no polytope body never allocates the list or dispatches the kernel (the host tracks
// hasPolyBodies): Daydream Home's egg rain steps bit-identically to the pre-B1 engine.
//
// WHY (each a measured failure of the old paths):
//   • VZ-0163 — box–box tested only the LOWER-index box's corners against the other box,
//     with a STRICT inside test (d > 0) and a speculative branch that rejected dist < 1e-8.
//     Two equal boxes stacked square put every corner EXACTLY on the other's edge: no
//     corner "inside", no near-contact, and the pair fell back to ONE central SAT contact
//     — no tipping resistance, so a 5-cube 10 mm tower fell at frame 3. (A box corner
//     exactly on a face, or exactly at the margin, also gave nothing.)
//   • T5 — hulls met static boxes through their VERTICES only: a box edge pressing into a
//     bar face, or two crossing edges, had no vertex inside anything.
//   • Hull–hull GJK/EPA at millimetre scale: a flat face–face simplex made EPA report a
//     0.26 mm depth for two bars 0.01 mm apart (the axis probe proved them separate and
//     was ignored), and the support-set band (1 mm + 2 % R, absolute) mixed a bar's face
//     ring with its chamfer ring — stacked bars were pushed apart and toppled.
//   • T8 — an egg's flank against a hull was two end spheres against the hull's support
//     PLANE: a bar edge touching the flank got contacts at the spheres' tangent points
//     (14.2 and 30.3 mm above the table) instead of where the edge actually is.
//
// THE METHOD (production-standard, Gregorius GDC 2013 / Box2D v3 / Jolt):
//   1. Separating-axis test in a pair-local frame: every face normal of both polytopes,
//      and every edge pair that forms a face of the Minkowski difference (Gauss-map arc
//      test — no other edge pair can be the minimum). Exact for polytopes, and the
//      separation it returns is the speculative gap for free.
//   2. Face contact: the reference face is the least-penetrated face (with the usual
//      relative/absolute tolerances so a resting stack never flips reference); the
//      incident face is the other body's face most anti-parallel to it; the incident
//      polygon is clipped by the reference face's side planes (Sutherland–Hodgman, with a
//      length tolerance so coincident edges never flicker), and every clipped point within
//      the margin is a contact with its OWN depth. Edge contact: the closest points of the
//      two edges. Reduced to ≤ 4 points (deepest + spread, cdReduceManifold).
//   3. Swept sphere (sphere / capsule / egg) vs polytope: the swept body is the union of
//      balls B(c(t), r(t)) along its axis (exactly — the convex hull of two spheres IS that
//      union), so f(t) = sd(c(t)) − r(t) is CONVEX in t and its minimum is the exact
//      signed distance to the polytope: golden-section search for t*, the exact signed
//      distance / closest point / normal of c(t*) (box: analytic; hull: faces, then the
//      visible faces' edges), plus the two end stations when they are within the margin
//      (the two-point rest of a capsule lying on a face).

// ── Shape predicates, compounds ──────────────────────────────────────────────────

static bool cdIsCompound(CoinBody c) { return c.shapeExtents.w > 5.5 && c.shapeExtents.w < 6.5; }

constant uint  CD_CMAX  = 128u;     // children per compound (CoinCompoundMath.maxChildren)
constant int   CD_PMAXV = 16;       // vertices per hull face loop (CoinHullTopology.maxFaceVertices)
constant int   CD_PMAXF = 64;       // faces per hull (CoinHullTopology.maxFaces)
constant int   CD_PCLIP = 34;       // clip buffer: CD_PMAXV incident + CD_PMAXV side planes + 2

// Box topology. Vertex i: bit0 → +x, bit1 → +y, bit2 → +z. Face f: 0 +x, 1 −x, 2 +y, 3 −y,
// 4 +z, 5 −z; loops CCW about the outward normal; edges (v0, v1, face, face).
constant uint  CD_BOX_LOOP[6][4] = { {1u,3u,7u,5u}, {0u,4u,6u,2u}, {2u,6u,7u,3u},
                                     {0u,1u,5u,4u}, {4u,5u,7u,6u}, {0u,2u,3u,1u} };
constant uint4 CD_BOX_EDGE[12] = {
    uint4(0,1,3,5), uint4(2,3,2,5), uint4(4,5,3,4), uint4(6,7,2,4),
    uint4(0,2,1,5), uint4(1,3,0,5), uint4(4,6,1,4), uint4(5,7,0,4),
    uint4(0,4,1,3), uint4(1,5,0,3), uint4(2,6,1,2), uint4(3,7,0,2) };

// A box piece in WORLD space (a box body, a compound child, a static box).
struct CdBox { float3 c; float4 q; float3 he; };

static uint cdCompoundCount(CoinBody b, device const uint2* hullRanges) {
    return min(hullRanges[uint(b.hullRef.x + 0.5)].y, CD_CMAX);
}
// Child k of compound `b`, in world space (table: centre, half-extents, orientation, in
// the principal frame with the COM at the origin — CoinDEMSolver+Compound.swift).
static CdBox cdCompoundChild(CoinBody b, uint k, device const float4* hullVerts, device const uint2* hullRanges) {
    uint base = hullRanges[uint(b.hullRef.x + 0.5)].x + 3u * k;
    float4 c = hullVerts[base], he = hullVerts[base + 1u], q = hullVerts[base + 2u];
    CdBox o;
    o.c  = b.posInvMass.xyz + cdQuatRotate(b.orient, c.xyz);
    o.q  = normalize(cdQuatMul(b.orient, q));
    o.he = he.xyz;
    return o;
}
static CdBox cdBoxOfBody(CoinBody b) {
    CdBox o; o.c = b.posInvMass.xyz; o.q = b.orient; o.he = b.shapeExtents.xyz; return o;
}

// ── Hull topology (CoinHullTopology.pack, right after the hull's vertices) ─────────

struct CdTopo { uint vBase, nV, nF, nE, fBase, lBase, eBase; };

static CdTopo cdHullTopoOf(CoinBody b, device const float4* hv, device const uint2* hr) {
    uint2 r = hr[uint(b.hullRef.x + 0.5)];
    uint4 h = as_type<uint4>(hv[r.x + r.y]);
    CdTopo t;
    // nV is NOT clamped: CoinHullTopology.fitsGPU caps faces, edges and loop sizes, not the
    // vertex count (a 5-ring × 14 barrel: 70 vertices, 58 faces). A clamp of 64 made
    // cdPolyMinDot — the SAT support — ignore vertices 64…69: along a static box's face the
    // barrel looked 0.31 mm further away, so a 0.05 mm penetration read as a 0.26 mm gap
    // and got NO contact (verifier, stage B1). cdPolyMinDot is the only reader; it loops.
    t.vBase = r.x; t.nV = r.y; t.nF = min(h.x, uint(CD_PMAXF)); t.nE = h.y;
    t.fBase = r.x + r.y + 1u;
    t.lBase = t.fBase + 2u * h.x;           // loop entries: (vertex.xyz, index bits), one slot each
    t.eBase = t.lBase + h.z;
    return t;
}
static bool cdHullHasTopo(CoinBody b, device const float4* hv, device const uint2* hr) {
    if (!cdIsHull(b)) return false;
    uint2 r = hr[uint(b.hullRef.x + 0.5)];
    return (as_type<uint4>(hv[r.x + r.y]).w & 1u) != 0u;
}

// A body the polytope narrowphase owns: a box, a compound, or a hull WITH topology.
static bool cdIsPolyBody(CoinBody c, device const float4* hv, device const uint2* hr) {
    return cdIsBox(c) || cdIsCompound(c) || cdHullHasTopo(c, hv, hr);
}
static bool cdIsRound(CoinBody c) { return cdIsSphere(c) || cdIsSwept(c); }

// THE pair predicate — coinGenerateContacts lists exactly the pairs this returns true for
// and coinPolyNarrow generates exactly those. Kept out of it: sphere–box (the old path is
// already exact), disc–box / disc–hull (the old paths; only a COMPOUND child meets a disc
// here, as its 16-gon prism, cdPolyDisc), and every pair without a polytope.
static bool cdPairIsPoly(CoinBody a, CoinBody b, device const float4* hv, device const uint2* hr) {
    if (cdIsCompound(a) || cdIsCompound(b)) return true;
    bool pa = cdIsPolyBody(a, hv, hr), pb = cdIsPolyBody(b, hv, hr);
    if (pa && pb) return true;
    if (pa && cdIsRound(b)) return !(cdIsBox(a) && cdIsSphere(b));
    if (pb && cdIsRound(a)) return !(cdIsBox(b) && cdIsSphere(a));
    return false;
}

// ── Polytope views (a box or a topology hull, in a pair-local working frame) ────────

struct CdPoly {
    float3x3 R;       // local → working frame
    float3   c;       // local origin in the working frame
    float3   he;      // box half-extents; a disc prism: (radius, half-thickness, 0)
    CdTopo   t;       // hull topology
    bool     box;
    bool     disc;    // a disc as a regular CD_DISC_N-gon prism (axis local +y)
};

static float3x3 cdQuatToMat(float4 q) { return cdQuatToMat3(q); }

// A DISC (only against a compound child — box–disc / hull–disc keep the old path) as a
// regular CD_DISC_N-gon prism with its vertices ON the true rim: the caps are the disc's
// faces, exact; the side is the inscribed polygon, its flats R(1 − cos(π/N)) = 1.9 % R
// inside the true rim. Vertices: 0..N−1 the top ring (θ_i = 2πi/N), N..2N−1 the bottom
// ring. Faces: 0..N−1 the sides (face i between vertices i and i+1), N the top cap, N+1
// the bottom cap. Edges: 0..N−1 the top ring, N..2N−1 the bottom ring, 2N..3N−1 the
// verticals.
//
// plausibility: shortcut — the rounded rim of a disc against a compound child is a 16-gon
// (≤ 1.9 % R radial error, caps exact). Tried: the disc's bounding box with the corners the
// disc does not occupy rejected (the lone box–disc path's approach) — it deletes the whole
// manifold of a disc lying on a face wider than itself (every clipped point is a phantom
// corner), and the disc fell through a compound tabletop (22 mm, testEveryShapeRestsOnACompound).
// An exact box–cylinder SAT (curved rim axes) was not attempted; no scene pairs compounds with
// discs yet (Digital Clock has no discs).
constant uint  CD_DISC_N = 16u;

static CdPoly cdPolyBox(CdBox B, float3 O) {
    CdPoly P; P.R = cdQuatToMat(B.q); P.c = B.c - O; P.he = B.he; P.box = true; P.disc = false;
    P.t.vBase = 0u; P.t.nV = 8u; P.t.nF = 6u; P.t.nE = 12u; P.t.fBase = 0u; P.t.lBase = 0u; P.t.eBase = 0u;
    return P;
}
static CdPoly cdPolyHull(CoinBody b, CdTopo t, float3 O) {
    CdPoly P; P.R = cdQuatToMat(b.orient); P.c = b.posInvMass.xyz - O; P.he = float3(0.0); P.t = t; P.box = false; P.disc = false;
    return P;
}
static CdPoly cdPolyDisc(CoinBody d, float3 O) {
    CdPoly P; P.R = cdQuatToMat(d.orient); P.c = d.posInvMass.xyz - O;
    P.he = float3(cdRadiusOf(d), cdHalfThickOf(d), 0.0); P.box = false; P.disc = true;
    P.t.vBase = 0u; P.t.nV = 2u * CD_DISC_N; P.t.nF = CD_DISC_N + 2u; P.t.nE = 3u * CD_DISC_N;
    P.t.fBase = 0u; P.t.lBase = 0u; P.t.eBase = 0u;
    return P;
}
static uint cdPolyNF(CdPoly P) { return P.box ? 6u : P.t.nF; }
static uint cdPolyNE(CdPoly P) { return P.box ? 12u : P.t.nE; }
static float3 cdDiscVLocal(float3 he, uint i) {
    uint k = i % CD_DISC_N;
    float a = float(k) * (6.2831853 / float(CD_DISC_N));
    return float3(he.x * cos(a), (i < CD_DISC_N) ? he.y : -he.y, he.x * sin(a));
}
static float3 cdPolyVLocal(CdPoly P, uint i, device const float4* hv) {
    if (P.box) return float3((i & 1u) ? P.he.x : -P.he.x, (i & 2u) ? P.he.y : -P.he.y, (i & 4u) ? P.he.z : -P.he.z);
    if (P.disc) return cdDiscVLocal(P.he, i);
    return hv[P.t.vBase + i].xyz;
}
static float3 cdPolyV(CdPoly P, uint i, device const float4* hv) { return P.c + P.R * cdPolyVLocal(P, i, hv); }
// Face f as a working-frame plane (outward unit normal, offset).
static float4 cdPolyFace(CdPoly P, uint f, device const float4* hv) {
    float4 l;
    if (P.box) {
        uint k = f >> 1; float s = (f & 1u) ? -1.0 : 1.0;
        float3 n = float3(k == 0u ? s : 0.0, k == 1u ? s : 0.0, k == 2u ? s : 0.0);
        l = float4(n, P.he[k]);
    } else if (P.disc) {
        if (f >= CD_DISC_N) l = float4(0.0, (f == CD_DISC_N) ? 1.0 : -1.0, 0.0, P.he.y);
        else {
            float a = (float(f) + 0.5) * (6.2831853 / float(CD_DISC_N));
            l = float4(cos(a), 0.0, sin(a), P.he.x * cos(3.14159265 / float(CD_DISC_N)));
        }
    } else {
        l = hv[P.t.fBase + 2u * f];
    }
    float3 n = P.R * l.xyz;
    return float4(n, l.w + dot(n, P.c));
}
static uint cdPolyLoopN(CdPoly P, uint f, device const float4* hv) {
    if (P.disc) return (f >= CD_DISC_N) ? CD_DISC_N : 4u;
    return P.box ? 4u : min(as_type<uint4>(hv[P.t.fBase + 2u * f + 1u]).y, uint(CD_PMAXV));
}
// Corner k of face f's loop, in the working frame, and its vertex index.
static float3 cdPolyLoopPos(CdPoly P, uint f, uint k, device const float4* hv, thread uint& vi) {
    if (P.box) { vi = CD_BOX_LOOP[f][k]; return cdPolyV(P, vi, hv); }
    if (P.disc) {
        const uint N = CD_DISC_N;
        if (f == N)           vi = (N - k) % N;                  // top cap: decreasing θ = CCW about +y
        else if (f == N + 1u) vi = N + k;                        // bottom cap: increasing θ = CCW about −y
        else {                                                   // side f: top f, top f+1, bottom f+1, bottom f
            uint f1 = (f + 1u) % N;
            vi = (k == 0u) ? f : ((k == 1u) ? f1 : ((k == 2u) ? N + f1 : N + f));
        }
        return cdPolyV(P, vi, hv);
    }
    float4 e = hv[P.t.lBase + as_type<uint4>(hv[P.t.fBase + 2u * f + 1u]).x + k];
    vi = as_type<uint>(e.w);
    return P.c + P.R * e.xyz;
}
static uint4 cdPolyEdge(CdPoly P, uint e, device const float4* hv) {
    if (P.disc) {
        const uint N = CD_DISC_N;
        if (e < N) return uint4(e, (e + 1u) % N, e, N);                            // top ring
        if (e < 2u * N) { uint j = e - N; return uint4(N + j, N + (j + 1u) % N, j, N + 1u); }   // bottom ring
        uint j = e - 2u * N; return uint4(j, N + j, (j + N - 1u) % N, j);           // vertical
    }
    return P.box ? CD_BOX_EDGE[e] : as_type<uint4>(hv[P.t.eBase + e]);
}
// min over the polytope's vertices of dot(n, v) (working frame).
static float cdPolyMinDot(CdPoly P, float3 n, device const float4* hv) {
    float3 nl = transpose(P.R) * n;
    float base = dot(n, P.c);
    if (P.box) return base - dot(abs(nl), P.he);
    float m = 1e30;
    if (P.disc) { for (uint i = 0; i < CD_DISC_N; ++i) m = min(m, dot(cdDiscVLocal(P.he, i).xz, nl.xz)); return base + m - P.he.y * abs(nl.y); }
    // 8 loads in flight (stage B3, VZ-0198): rolled, this loop paid a full device-load latency
    // per vertex on the in-order GPU — ~10 µs for a 32-vertex hull on one lane. Same min, same order.
    #pragma unroll 8
    for (uint i = 0; i < P.t.nV; ++i) m = min(m, dot(hv[P.t.vBase + i].xyz, nl));
    return base + m;
}

// ── Polytope–polytope: SAT + clipped manifold ─────────────────────────────────────

struct CdManifold {
    int    n;
    float3 p[4];        // contact points, working frame (midway between the surfaces)
    float  depth[4];    // > 0 penetrating, < 0 a speculative gap
    uint   tag[4];      // feature id (see cdPolyManifold)
    float3 nrm;         // unit normal from Q toward P
    float  sep;         // the SAT separation (max over the tested axes; < 0 = penetration depth)
};

// Relative + absolute SAT tie tolerances (Gregorius): an edge axis must beat the best
// face by 10 %, and Q's face must beat P's by 2 %, so a resting stack keeps its
// reference face instead of flickering between two near-equal choices.
constant float CD_SAT_REL_EDGE = 0.90;
constant float CD_SAT_REL_FACE = 0.98;

// The separating-axis test's result: the best face of P, the best face of Q and the best
// Minkowski edge pair (see cdPolySAT) — everything the manifold is then built from.
struct CdSAT {
    float  sepP;  uint faceP;
    float  sepQ;  uint faceQ;
    float  sepE;  uint edgeP, edgeQ;
    float3 axisE;
    bool   haveEdge;
};

// The SAT on ONE thread (the reference; the kernels run cdPolySATSG, the same answer bit for
// bit — CoinDEMSmallWorldTests.testPolytopeSATOnASIMDGroupMatchesOneThread). False at the
// first axis that separates beyond `lim`. `pf` / `qf` hold the face planes for the edge loop.
static bool cdPolySAT(CdPoly P, CdPoly Q, float lim, device const float4* hv, thread CdSAT& S) {
    uint nfP = cdPolyNF(P), nfQ = cdPolyNF(Q);
    // Face normals of both, working frame (cached for the edge loop).
    float4 pf[CD_PMAXF], qf[CD_PMAXF];
    #pragma unroll 8
    for (uint f = 0; f < nfP; ++f) pf[f] = cdPolyFace(P, f, hv);
    #pragma unroll 8
    for (uint f = 0; f < nfQ; ++f) qf[f] = cdPolyFace(Q, f, hv);

    // 1. Face axes of P, then of Q — each loop returns at the FIRST axis that separates
    //    beyond the margin (most candidate pairs are near but not touching). The face of P
    //    most aligned with the centre-to-centre direction is tried first: when the pair is
    //    apart it is nearly always the separating one.
    float3 dPQ = Q.c - P.c;
    uint f0 = 0u; float a0 = -1e30;
    for (uint f = 0; f < nfP; ++f) { float a = dot(pf[f].xyz, dPQ); if (a > a0) { a0 = a; f0 = f; } }
    float sepP = cdPolyMinDot(Q, pf[f0].xyz, hv) - pf[f0].w; uint faceP = f0;
    if (sepP > lim) return false;
    for (uint f = 0; f < nfP; ++f) {
        if (f == f0) continue;
        float s = cdPolyMinDot(Q, pf[f].xyz, hv) - pf[f].w;
        if (s > lim) return false;
        if (s > sepP) { sepP = s; faceP = f; }
    }
    float sepQ = -1e30; uint faceQ = 0u;
    for (uint f = 0; f < nfQ; ++f) {
        float s = cdPolyMinDot(P, qf[f].xyz, hv) - qf[f].w;
        if (s > lim) return false;
        if (s > sepQ) { sepQ = s; faceQ = f; }
    }

    // 2. Edge pairs that form a face of the Minkowski difference P − Q: P's edge arc
    //    (a, b) crosses the negated arc (−c, −d) of Q's edge on the Gauss map.
    float sepE = -1e30; uint edgeP = 0u, edgeQ = 0u; float3 axisE = float3(0.0, 1.0, 0.0);
    bool haveEdge = false;
    uint neP = cdPolyNE(P), neQ = cdPolyNE(Q);
    float sq[CD_PMAXF];
    for (uint e = 0; e < neP; ++e) {
        uint4 E = cdPolyEdge(P, e, hv);
        float3 a = pf[E.z].xyz, b = pf[E.w].xyz;
        float3 bxa = cross(b, a);
        for (uint f = 0; f < nfQ; ++f) sq[f] = dot(qf[f].xyz, bxa);   // = −(−n_Q)·(b×a)
        float3 p0 = cdPolyV(P, E.x, hv), p1 = cdPolyV(P, E.y, hv);
        float3 eP = p1 - p0;
        for (uint g = 0; g < neQ; ++g) {
            uint4 G = cdPolyEdge(Q, g, hv);
            float cba = -sq[G.z], dba = -sq[G.w];                     // c = −n_Q0, d = −n_Q1
            if (cba * dba >= 0.0) continue;
            float3 c = -qf[G.z].xyz, d = -qf[G.w].xyz;
            float3 dxc = cross(d, c);
            float adc = dot(a, dxc), bdc = dot(b, dxc);
            if (!(adc * bdc < 0.0 && cba * bdc > 0.0)) continue;
            float3 q0 = cdPolyV(Q, G.x, hv), q1 = cdPolyV(Q, G.y, hv);
            float3 eQ = q1 - q0;
            float3 ax = cross(eP, eQ);
            float l2 = dot(ax, ax);
            if (l2 <= 1e-6 * dot(eP, eP) * dot(eQ, eQ)) continue;   // near-parallel (sin < 1e-3)
            ax *= rsqrt(l2);
            if (dot(ax, p0 - P.c) < 0.0) ax = -ax;                     // outward from P
            float s = dot(ax, q0 - p0);
            if (s > lim) return false;
            if (s > sepE) { sepE = s; edgeP = e; edgeQ = g; axisE = ax; haveEdge = true; }
        }
    }
    S.sepP = sepP; S.faceP = faceP; S.sepQ = sepQ; S.faceQ = faceQ;
    S.sepE = sepE; S.edgeP = edgeP; S.edgeQ = edgeQ; S.axisE = axisE; S.haveEdge = haveEdge;
    return true;
}

// One Minkowski edge-pair axis of the SAT — cdPolySAT's inner loop body, on per-edge records
// computed once per edge (the same expressions as there): false when the pair is not a
// Gauss-map crossing or is near-parallel; else its separation `s` and outward axis `ax`.
struct CdSATEdgeP { float3 a, b, bxa, p0, eP; };
struct CdSATEdgeQ { float3 qz, qw, q0, eQ; };
static inline CdSATEdgeP cdSATEdgeOfP(CdPoly P, uint e, device const float4* hv) {
    uint4 E = cdPolyEdge(P, e, hv);
    CdSATEdgeP r;
    r.a = cdPolyFace(P, E.z, hv).xyz; r.b = cdPolyFace(P, E.w, hv).xyz;
    r.bxa = cross(r.b, r.a);
    r.p0 = cdPolyV(P, E.x, hv);
    float3 p1 = cdPolyV(P, E.y, hv);
    r.eP = p1 - r.p0;
    return r;
}
static inline CdSATEdgeQ cdSATEdgeOfQ(CdPoly Q, uint g, device const float4* hv) {
    uint4 G = cdPolyEdge(Q, g, hv);
    CdSATEdgeQ r;
    r.qz = cdPolyFace(Q, G.z, hv).xyz; r.qw = cdPolyFace(Q, G.w, hv).xyz;
    r.q0 = cdPolyV(Q, G.x, hv);
    float3 q1 = cdPolyV(Q, G.y, hv);
    r.eQ = q1 - r.q0;
    return r;
}
static inline bool cdSATEdgeAxis(CdPoly P, thread const CdSATEdgeP& ep, thread const CdSATEdgeQ& eq,
                                 thread float& s, thread float3& ax) {
    float cba = -dot(eq.qz, ep.bxa), dba = -dot(eq.qw, ep.bxa);     // c = −n_Q0, d = −n_Q1
    if (cba * dba >= 0.0) return false;
    float3 c = -eq.qz, d = -eq.qw;
    float3 dxc = cross(d, c);
    float adc = dot(ep.a, dxc), bdc = dot(ep.b, dxc);
    if (!(adc * bdc < 0.0 && cba * bdc > 0.0)) return false;
    ax = cross(ep.eP, eq.eQ);
    float l2 = dot(ax, ax);
    if (l2 <= 1e-6 * dot(ep.eP, ep.eP) * dot(eq.eQ, eq.eQ)) return false;   // near-parallel (sin < 1e-3)
    ax *= rsqrt(l2);
    if (dot(ax, ep.p0 - P.c) < 0.0) ax = -ax;                                // outward from P
    s = dot(ax, eq.q0 - ep.p0);
    return true;
}

// The SAT's FIRST test on one thread — cdPolySAT's opening step exactly: f0 = the face of P
// most aligned with the centre-to-centre direction (the first maximum above −1e30; 0 if none)
// and its separation s0. False when s0 > lim: the pair is apart and the SAT stops here, as the
// serial loop does. Most listed pairs end here (the Digital Clock's jib against every bar under
// it), so the kernels run it one thread per pair and hand only the survivors, with f0 / s0, to a
// SIMD group (cdPolySATSG).
static bool cdPolySATFirstFace(CdPoly P, CdPoly Q, float lim, device const float4* hv, thread uint& f0, thread float& s0) {
    uint nfP = cdPolyNF(P);
    float3 dPQ = Q.c - P.c;
    f0 = 0u; float a0 = -1e30;
    if (P.box || P.disc) {
        for (uint f = 0; f < nfP; ++f) { float a = dot(cdPolyFace(P, f, hv).xyz, dPQ); if (a > a0) { a0 = a; f0 = f; } }
    } else {
        // A hull's face planes straight from its table — cdPolyFace's hull branch (n = R·l.xyz), in
        // a BRANCH-FREE body so the unrolled loop keeps 8 loads in flight (inside cdPolyFace's
        // box / disc / hull branches each load waited its full latency: ≈ 26 µs per substep for
        // the clock's 56 listed bar pairs, stage B3).
        #pragma unroll 8
        for (uint f = 0; f < nfP; ++f) {
            float4 l = hv[P.t.fBase + 2u * f];
            float3 n = P.R * l.xyz;
            float a = dot(n, dPQ);
            if (a > a0) { a0 = a; f0 = f; }
        }
    }
    float4 pl0 = cdPolyFace(P, f0, hv);
    s0 = cdPolyMinDot(Q, pl0.xyz, hv) - pl0.w;
    return !(s0 > lim);
}

// The SAT on ONE SIMD GROUP (stage B3, VZ-0198) — the same axes, the same arithmetic per axis
// and the same answer as cdPolySAT, bit for bit. Every lane of the group calls it with the same
// P / Q / lim; `lane` is its index in the group, `w` the group's width (all `w` lanes active).
//
// WHY. cdPolySAT on one thread was the p95 of the Digital Clock's perf fence: a TOUCHING pair
// never early-outs, so it walks every face of both and every edge pair — 144 for two boxes,
// ~670 for the clock's bar hull on a box — through its face-plane / sq arrays, which live in
// the thread's stack (device) memory; one GPU thread issues ≈ one instruction per 4 ns and
// waits ≈ 16 ns on a dependent one (M1 Max). Measured: the crane's block touching its puck
// (box–box, 3 speculative points) cost ≈ 0.12 ms per substep in the narrowphase; the fence's
// frames with a touching polytope pair ran 4.9–5.9 ms against 3.3 ms without. Here the lanes
// share the axes: faces l, l + w, … per lane; the edge pairs with the LARGER edge set spread
// over the lanes and the smaller walked by each (per-edge data computed once per lane) — no
// arrays, nothing in stack memory.
//
// Selection = the serial loops': each keeps the maximum, and among equal maxima the FIRST in
// its visiting order (P's faces: f0 first, then by index; Q's faces by index; edge pairs by
// e·neQ + g), and a NaN never wins (`s > best` is false for it). Each lane keeps its best with
// that rule; the group takes the maximum, then the smallest visiting rank among lanes holding
// it, and broadcasts the winner's own values (so a ±0 tie resolves to the serial loop's).
static bool cdPolySATSG(CdPoly P, CdPoly Q, float lim, device const float4* hv, uint lane, uint w,
                        uint f0, float s0, thread CdSAT& S) {
    const uint BIG = 0xFFFFFFFFu;
    uint nfP = cdPolyNF(P), nfQ = cdPolyNF(Q);

    // P's faces other than f0 (f0 and its separation s0 come from cdPolySATFirstFace, which
    // already found s0 ≤ lim), by index.
    bool out = false;
    float bP = -INFINITY; uint iP = BIG;
    for (uint f = lane; f < nfP; f += w) {
        if (f == f0) continue;
        float4 pl = cdPolyFace(P, f, hv);
        float s = cdPolyMinDot(Q, pl.xyz, hv) - pl.w;
        out = out || (s > lim);
        if (s > bP || (s == bP && f < iP)) { bP = s; iP = f; }
    }
    // Q's faces, by index, above the serial loop's −1e30 start.
    float bQ = -INFINITY; uint iQ = BIG;
    for (uint f = lane; f < nfQ; f += w) {
        float4 ql = cdPolyFace(Q, f, hv);
        float s = cdPolyMinDot(P, ql.xyz, hv) - ql.w;
        out = out || (s > lim);
        if (s > -1e30 && (s > bQ || (s == bQ && f < iQ))) { bQ = s; iQ = f; }
    }
    if (simd_any(out)) return false;

    float mP = simd_max(bP);
    if (!isnan(s0) && mP > s0) {                    // a later face strictly beats f0
        uint win = simd_min((iP != BIG && bP == mP) ? iP : BIG);
        S.sepP = simd_shuffle(bP, ushort(win % w)); S.faceP = win;
    } else {
        S.sepP = s0; S.faceP = f0;
    }
    float mQ = simd_max(bQ);
    if (mQ > -1e30) {
        uint win = simd_min((iQ != BIG && bQ == mQ) ? iQ : BIG);
        S.sepQ = simd_shuffle(bQ, ushort(win % w)); S.faceQ = win;
    } else {
        S.sepQ = -1e30; S.faceQ = 0u;
    }

    // Edge pairs. Lanes take the larger edge set; flat = e·neQ + g is the serial visiting rank.
    uint neP = cdPolyNE(P), neQ = cdPolyNE(Q);
    float bE = -INFINITY; uint iE = BIG; float3 axE = float3(0.0, 1.0, 0.0);
    bool lanesOnP = neP >= neQ;
    if (lanesOnP) {
        for (uint e = lane; e < neP; e += w) {
            CdSATEdgeP ep = cdSATEdgeOfP(P, e, hv);
            for (uint g = 0; g < neQ; ++g) {
                CdSATEdgeQ eq = cdSATEdgeOfQ(Q, g, hv);
                float s; float3 ax;
                if (!cdSATEdgeAxis(P, ep, eq, s, ax)) continue;
                out = out || (s > lim);
                uint flat = e * neQ + g;
                if (s > -1e30 && (s > bE || (s == bE && flat < iE))) { bE = s; iE = flat; axE = ax; }
            }
        }
    } else {
        for (uint g = lane; g < neQ; g += w) {
            CdSATEdgeQ eq = cdSATEdgeOfQ(Q, g, hv);
            for (uint e = 0; e < neP; ++e) {
                CdSATEdgeP ep = cdSATEdgeOfP(P, e, hv);
                float s; float3 ax;
                if (!cdSATEdgeAxis(P, ep, eq, s, ax)) continue;
                out = out || (s > lim);
                uint flat = e * neQ + g;
                if (s > -1e30 && (s > bE || (s == bE && flat < iE))) { bE = s; iE = flat; axE = ax; }
            }
        }
    }
    if (simd_any(out)) return false;
    float mE = simd_max(bE);
    if (mE > -1e30) {
        uint win = simd_min((iE != BIG && bE == mE) ? iE : BIG);
        uint e = win / neQ, g = win % neQ;
        ushort owner = ushort((lanesOnP ? e : g) % w);
        S.sepE = simd_shuffle(bE, owner);
        S.axisE = float3(simd_shuffle(axE.x, owner), simd_shuffle(axE.y, owner), simd_shuffle(axE.z, owner));
        S.edgeP = e; S.edgeQ = g; S.haveEdge = true;
    } else {
        S.sepE = -1e30; S.edgeP = 0u; S.edgeQ = 0u; S.axisE = float3(0.0, 1.0, 0.0); S.haveEdge = false;
    }
    return true;
}

// Generate the manifold of P against Q within the speculative margin `spec`, from the pair's
// SAT result `S`. `tol` is the pair's length tolerance (scale-relative): the clip keeps a point
// up to `tol` outside a side plane (coincident edges never flicker) and the margin test is
// inclusive by `tol` (a corner exactly AT the margin, or exactly ON a face, is a contact —
// VZ-0163). `absTol` is the SAT tie tolerance. Feature tags: bits 30–31 kind (0 reference face
// on P, 1 on Q, 2 edge–edge), 15–21 the reference face / P edge, 0–14 the point (incident loop
// vertex k, or 16 + side-plane·16 + the segment's start tag for a clipped point / the Q edge).
// Which contact the SAT result makes: `edge` (an edge–edge contact) or, else, a face contact with
// the reference face on Q (`refIsQ`) or on P. Sets M.sep.
static inline void cdPolyContactKind(thread const CdSAT& S, float absTol, bool apartFaceFirst,
                                     thread bool& edge, thread bool& refIsQ, thread CdManifold& M) {
    float maxF = max(S.sepP, S.sepQ);
    M.sep = S.haveEdge ? max(maxF, S.sepE) : maxF;
    // The relative tie (an edge must beat the best face by 10 %) is written for OVERLAP, where
    // separations are negative. Within the speculative margin they are POSITIVE, and
    // 0.9·maxF < maxF flips its meaning: an edge pair whose axis is PARALLEL to the face
    // normal (two parallel faces approaching — every horizontal edge of a flat base against
    // the table's top edges crosses to a vertical axis, separation = the same gap) won the
    // tie as soon as the gap exceeded 10·absTol (0.5 mm), and emitted ONE point midway
    // between the base's edge and the table's far rim, so a flat-bottomed body landing flat
    // sank up to 2.6 mm into the table for a substep or two (VZ-0201, measured on the
    // Digital Clock's flat-based worker: 12 drop phases 0–2.6 mm). `apartFaceFirst`: while
    // apart, an edge must be strictly more separated than every face. Opt-in (the manifold
    // solve's flag at the call sites) so every other world is bit-identical.
    edge = S.haveEdge && ((apartFaceFirst && maxF > 0.0) ? S.sepE > maxF + absTol
                                                         : S.sepE > CD_SAT_REL_EDGE * maxF + absTol);
    refIsQ = S.sepQ > CD_SAT_REL_FACE * S.sepP + absTol;
}

// The edge–edge contact: the closest points of the two edges.
static inline void cdPolyEdgeContact(CdPoly P, CdPoly Q, thread const CdSAT& S, device const float4* hv,
                                     thread CdManifold& M) {
    uint4 E = cdPolyEdge(P, S.edgeP, hv), G = cdPolyEdge(Q, S.edgeQ, hv);
    float3 c1, c2;
    cdClosestSegSeg(cdPolyV(P, E.x, hv), cdPolyV(P, E.y, hv), cdPolyV(Q, G.x, hv), cdPolyV(Q, G.y, hv), c1, c2);
    M.n = 1;
    M.p[0] = 0.5 * (c1 + c2);
    M.depth[0] = -S.sepE;
    M.tag[0] = (2u << 30) | ((S.edgeP & 127u) << 15) | (S.edgeQ & 0x7FFFu);
    M.nrm = -S.axisE;
}

// The incident face: the face of `In` most anti-parallel to the reference normal `rn` — the
// FIRST minimum of dot(n_f, rn) below 1e30 (0 if none). One thread …
static uint cdPolyIncidentFace(CdPoly In, float3 rn, device const float4* hv) {
    uint fi = 0u; float mind = 1e30;
    uint nfI = cdPolyNF(In);
    for (uint f = 0; f < nfI; ++f) {
        float dd = dot(cdPolyFace(In, f, hv).xyz, rn);
        if (dd < mind) { mind = dd; fi = f; }
    }
    return fi;
}
// … or a SIMD group (every lane, the same arguments): the same face — each lane keeps its first
// minimum, the group takes the minimum and then the smallest index holding it.
static uint cdPolyIncidentFaceSG(CdPoly In, float3 rn, device const float4* hv, uint lane, uint w) {
    const uint BIG = 0xFFFFFFFFu;
    uint nfI = cdPolyNF(In);
    float bd = INFINITY; uint bi = BIG;
    for (uint f = lane; f < nfI; f += w) {
        float dd = dot(cdPolyFace(In, f, hv).xyz, rn);
        if (dd < 1e30 && (dd < bd || (dd == bd && f < bi))) { bd = dd; bi = f; }
    }
    float m = simd_min(bd);
    uint fi = simd_min((bi != BIG && bd == m) ? bi : BIG);
    return fi == BIG ? 0u : fi;
}

// The face contact against reference face `fr` (on Q when refIsQ, else on P) with incident face
// `fi`: the incident polygon clipped by the reference face's side planes (Sutherland–Hodgman,
// with the length tolerance `tol`), every clipped point within the margin a contact with its
// own depth, reduced to ≤ 4 points.
static bool cdPolyFaceContact(CdPoly P, CdPoly Q, bool refIsQ, uint fr, uint fi, float spec, float tol,
                              device const float4* hv, thread CdManifold& M) {
    float lim = spec + tol;
    CdPoly Rf = refIsQ ? Q : P;             // reference polytope
    CdPoly In = refIsQ ? P : Q;             // incident polytope
    float4 rpl = cdPolyFace(Rf, fr, hv);
    float3 poly[CD_PCLIP]; uint ptag[CD_PCLIP];
    int np = int(cdPolyLoopN(In, fi, hv));
    for (int k = 0; k < np; ++k) { uint vi; poly[k] = cdPolyLoopPos(In, fi, uint(k), hv, vi); ptag[k] = uint(k); }
    // Deepest incident vertex (the fallback if clipping leaves nothing — a vertex contact
    // just outside the reference face's extent).
    float sMin = 1e30; float3 pMin = poly[0]; uint tMin = 0u;
    for (int k = 0; k < np; ++k) { float s = dot(rpl.xyz, poly[k]) - rpl.w; if (s < sMin) { sMin = s; pMin = poly[k]; tMin = uint(k); } }

    uint nr = cdPolyLoopN(Rf, fr, hv);
    float3 nxt[CD_PCLIP]; uint ntag[CD_PCLIP];
    for (uint k = 0; k < nr && np > 0; ++k) {
        uint vi0, vi1;
        float3 v0 = cdPolyLoopPos(Rf, fr, k, hv, vi0);
        float3 v1 = cdPolyLoopPos(Rf, fr, (k + 1u) % nr, hv, vi1);
        float3 sn = cross(v1 - v0, rpl.xyz);                // outward side-plane normal
        float sl = length(sn);
        if (sl <= 1e-20) continue;
        sn /= sl;
        float off = dot(sn, v0) + tol;
        int no = 0;
        for (int i = 0; i < np; ++i) {
            float3 a = poly[i], b = poly[(i + 1) % np];
            float da = dot(sn, a) - off, db = dot(sn, b) - off;
            if (da <= 0.0 && no < CD_PCLIP) { nxt[no] = a; ntag[no] = ptag[i]; no++; }
            if (((da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0)) && no < CD_PCLIP) {
                float t = da / (da - db);
                nxt[no] = a + t * (b - a);
                ntag[no] = 16u + (k & 15u) * 16u + (ptag[i] & 15u);
                no++;
            }
        }
        np = no;
        for (int i = 0; i < np; ++i) { poly[i] = nxt[i]; ptag[i] = ntag[i]; }
    }
    float3 cp[CD_PCLIP]; float cd[CD_PCLIP]; uint ct[CD_PCLIP]; int nc = 0;
    for (int i = 0; i < np; ++i) {
        float s = dot(rpl.xyz, poly[i]) - rpl.w;
        if (s <= lim) { cp[nc] = poly[i] - 0.5 * s * rpl.xyz; cd[nc] = -s; ct[nc] = ptag[i]; nc++; }
    }
    if (nc == 0) {
        if (sMin > lim) return false;
        cp[0] = pMin - 0.5 * sMin * rpl.xyz; cd[0] = -sMin; ct[0] = tMin; nc = 1;
    }
    int keep[CD_MANIFOLD_MAX];
    int m = cdReduceManifold(cp, cd, nc, keep);
    uint kind = refIsQ ? 1u : 0u;
    for (int i = 0; i < m; ++i) {
        M.p[i] = cp[keep[i]];
        M.depth[i] = cd[keep[i]];
        M.tag[i] = (kind << 30) | ((fr & 127u) << 15) | (ct[keep[i]] & 0x7FFFu);
    }
    M.n = m;
    M.nrm = refIsQ ? rpl.xyz : -rpl.xyz;     // from Q toward P
    return true;
}

// Generate the manifold of P against Q within the speculative margin `spec`, from the pair's
// SAT result `S`, on one thread. `tol` is the pair's length tolerance (scale-relative): the clip
// keeps a point up to `tol` outside a side plane (coincident edges never flicker) and the margin
// test is inclusive by `tol` (a corner exactly AT the margin, or exactly ON a face, is a contact
// — VZ-0163). `absTol` is the SAT tie tolerance. Feature tags: bits 30–31 kind (0 reference
// face on P, 1 on Q, 2 edge–edge), 15–21 the reference face / P edge, 0–14 the point (incident
// loop vertex k, or 16 + side-plane·16 + the segment's start tag for a clipped point / the Q edge).
static bool cdPolyManifoldFrom(CdPoly P, CdPoly Q, thread const CdSAT& S, float spec, float tol, float absTol,
                               bool apartFaceFirst, device const float4* hv, thread CdManifold& M) {
    M.n = 0;
    M.nrm = float3(0.0, 1.0, 0.0);
    bool edge, refIsQ;
    cdPolyContactKind(S, absTol, apartFaceFirst, edge, refIsQ, M);
    if (edge) { cdPolyEdgeContact(P, Q, S, hv, M); return true; }
    uint fr = refIsQ ? S.faceQ : S.faceP;
    float4 rpl = cdPolyFace(refIsQ ? Q : P, fr, hv);
    uint fi = cdPolyIncidentFace(refIsQ ? P : Q, rpl.xyz, hv);
    return cdPolyFaceContact(P, Q, refIsQ, fr, fi, spec, tol, hv, M);
}

// SAT + manifold on ONE thread (the penetration probe, cdPolyPairDepth; the kernels use the
// SIMD-group SAT — cdPolyNarrowSG).
static bool cdPolyManifold(CdPoly P, CdPoly Q, float spec, float tol, float absTol, bool apartFaceFirst,
                           device const float4* hv, thread CdManifold& M) {
    M.n = 0;
    M.nrm = float3(0.0, 1.0, 0.0);
    CdSAT S;
    if (!cdPolySAT(P, Q, spec + tol, hv, S)) return false;
    return cdPolyManifoldFrom(P, Q, S, spec, tol, absTol, apartFaceFirst, hv, M);
}

// The SAT penetration depth alone (no manifold): max separation over the face axes and
// the Minkowski edge pairs, negated — exact for polytopes (the penetration probe).
static float cdPolyDepth(CdPoly P, CdPoly Q, device const float4* hv) {
    CdManifold M;
    if (!cdPolyManifold(P, Q, 1e30, 0.0, 0.0, false, hv, M)) return -1e30;
    return -M.sep;
}

// ── Signed distance to a polytope (point query) ──────────────────────────────────

struct CdSd { float d; float3 n; float3 cl; };    // signed distance, unit normal out of the solid, closest surface point

static CdSd cdSdBox(CdBox B, float3 p) {
    float3 lp = cdQuatRotateInv(B.q, p - B.c);
    float3 q = abs(lp) - B.he;
    float3 nl, cl;
    CdSd r;
    float mq = max(q.x, max(q.y, q.z));
    if (mq <= 0.0) {                                  // inside: nearest face
        uint k = (q.x >= q.y && q.x >= q.z) ? 0u : ((q.y >= q.z) ? 1u : 2u);
        float s = lp[k] >= 0.0 ? 1.0 : -1.0;
        nl = float3(k == 0u ? s : 0.0, k == 1u ? s : 0.0, k == 2u ? s : 0.0);
        cl = lp; cl[k] = s * B.he[k];
        r.d = mq;
    } else {
        cl = clamp(lp, -B.he, B.he);
        float3 dv = lp - cl;
        r.d = length(dv);
        nl = dv / max(r.d, 1e-30);
    }
    r.n = cdQuatRotate(B.q, nl);
    r.cl = B.c + cdQuatRotate(B.q, cl);
    return r;
}

// Hull (with topology): inside → the least-penetrated face; outside → the face with the
// largest plane distance if the projection lands inside its polygon (then it IS the
// closest point), else the nearest point on the edges of the faces that see p.
static CdSd cdSdHull(CoinBody h, CdTopo t, float3 p, device const float4* hv) {
    float3 lp = cdQuatRotateInv(h.orient, p - h.posInvMass.xyz);
    float s[CD_PMAXF];
    float best = -1e30; uint bf = 0u;
    for (uint f = 0; f < t.nF; ++f) {
        float4 pl = hv[t.fBase + 2u * f];
        s[f] = dot(pl.xyz, lp) - pl.w;
        if (s[f] > best) { best = s[f]; bf = f; }
    }
    float3 nl = hv[t.fBase + 2u * bf].xyz, cl;
    CdSd r;
    r.d = best;
    cl = lp - best * nl;
    if (best > 0.0) {
        // Is the projection inside face bf's polygon?
        uint4 lf = as_type<uint4>(hv[t.fBase + 2u * bf + 1u]);
        uint nL = min(lf.y, uint(CD_PMAXV));
        bool inside = true;
        for (uint k = 0; k < nL && inside; ++k) {
            float3 v0 = hv[t.lBase + lf.x + k].xyz;
            float3 v1 = hv[t.lBase + lf.x + (k + 1u) % nL].xyz;
            if (dot(cross(v1 - v0, cl - v0), nl) < 0.0) inside = false;
        }
        if (!inside) {
            float d2 = 1e30;
            for (uint f = 0; f < t.nF; ++f) {
                if (s[f] <= 0.0) continue;
                uint4 lf2 = as_type<uint4>(hv[t.fBase + 2u * f + 1u]);
                uint nL2 = min(lf2.y, uint(CD_PMAXV));
                for (uint k = 0; k < nL2; ++k) {
                    float3 v0 = hv[t.lBase + lf2.x + k].xyz;
                    float3 v1 = hv[t.lBase + lf2.x + (k + 1u) % nL2].xyz;
                    float3 q = cdClosestOnSegment(v0, v1, lp);
                    float dd = distance_squared(q, lp);
                    if (dd < d2) { d2 = dd; cl = q; }
                }
            }
            r.d = sqrt(d2);
            nl = (lp - cl) / max(r.d, 1e-30);
        }
    }
    r.n = cdQuatRotate(h.orient, nl);
    r.cl = h.posInvMass.xyz + cdQuatRotate(h.orient, cl);
    return r;
}

// A "piece" of a polytope body the swept query runs against: a box, or a topology hull.
struct CdPiece { bool box; CdBox b; };

static CdSd cdSdPiece(CdPiece P, CoinBody hullBody, CdTopo t, float3 p, device const float4* hv) {
    return P.box ? cdSdBox(P.b, p) : cdSdHull(hullBody, t, p, hv);
}

// ── Swept sphere (sphere / capsule / egg) vs a polytope piece ─────────────────────

struct CdSweptContact { int n; float3 p[3]; float3 nrm[3]; float depth[3]; uint tag[3]; };

constant int CD_GOLDEN_ITERS = 24;        // bracket 0.618^24 ≈ 1e-5 of the segment

// Segment s0→s1 swept by radius r0→r1, against piece P (normals point OUT of P, toward
// the swept body). Contacts: the global minimum t* of the convex f(t) = sd(c(t)) − r(t),
// and the end stations t = 0, 1 when their own f is within the margin (a capsule lying
// on a face rests on both ends; an egg's flank on a bar EDGE touches only at t*, where
// the edge actually is). Returns the contacts with depth −f.
static void cdSweptVsPiece(float3 s0, float3 s1, float r0, float r1, CdPiece P, CoinBody hullBody, CdTopo t,
                           float spec, float tol, device const float4* hv, thread CdSweptContact& C) {
    C.n = 0;
    float lim = spec + tol;
    // Face-plane pre-test: a face plane both end spheres clear by more than the margin
    // separates the pair (the swept surface's plane distance is linear along the axis, so
    // its minimum is at an end) — the common near-but-apart pair costs one plane pass.
    {
        float4 q = P.box ? P.b.q : hullBody.orient;
        float3 c = P.box ? P.b.c : hullBody.posInvMass.xyz;
        float3 l0 = cdQuatRotateInv(q, s0 - c), l1 = cdQuatRotateInv(q, s1 - c);
        if (P.box) {
            // Face +k: min over the ends of (l_k − r) − he_k; face −k: −max(l_k + r) − he_k.
            float3 sPos = min(l0 - r0, l1 - r1) - P.b.he, sNeg = -max(l0 + r0, l1 + r1) - P.b.he;
            if (any(sPos > lim) || any(sNeg > lim)) return;
        } else {
            for (uint f = 0; f < t.nF; ++f) {
                float4 pl = hv[t.fBase + 2u * f];
                if (min(dot(pl.xyz, l0) - pl.w - r0, dot(pl.xyz, l1) - pl.w - r1) > lim) return;
            }
        }
    }
    float L = length(s1 - s0);
    float tStar = 0.0;
    CdSd sdS = cdSdPiece(P, hullBody, t, s0, hv);
    float fS = sdS.d - r0;
    if (L > 1e-7) {
        const float g = 0.6180339887;
        float lo = 0.0, hi = 1.0;
        float x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
        float f1 = cdSdPiece(P, hullBody, t, mix(s0, s1, x1), hv).d - mix(r0, r1, x1);
        float f2 = cdSdPiece(P, hullBody, t, mix(s0, s1, x2), hv).d - mix(r0, r1, x2);
        for (int it = 0; it < CD_GOLDEN_ITERS; ++it) {
            if (f1 <= f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - g * (hi - lo);
                            f1 = cdSdPiece(P, hullBody, t, mix(s0, s1, x1), hv).d - mix(r0, r1, x1); }
            else          { lo = x1; x1 = x2; f1 = f2; x2 = lo + g * (hi - lo);
                            f2 = cdSdPiece(P, hullBody, t, mix(s0, s1, x2), hv).d - mix(r0, r1, x2); }
        }
        tStar = 0.5 * (lo + hi);
        sdS = cdSdPiece(P, hullBody, t, mix(s0, s1, tStar), hv);
        fS = sdS.d - mix(r0, r1, tStar);
    }
    if (fS > lim) return;
    // End stations.
    float3 st[2] = { s0, s1 }; float rr[2] = { r0, r1 }; float tt[2] = { 0.0, 1.0 };
    bool endOK[2] = { false, false };
    CdSd sdE[2];
    float fE[2] = { 1e30, 1e30 };
    if (L > 1e-7) {
        for (int k = 0; k < 2; ++k) {
            sdE[k] = cdSdPiece(P, hullBody, t, st[k], hv);
            fE[k] = sdE[k].d - rr[k];
            endOK[k] = fE[k] <= lim && abs(tt[k] - tStar) * L > 0.25 * min(r0, r1);
        }
    }
    bool both = endOK[0] && endOK[1];
    if (!both) {
        float r = mix(r0, r1, tStar);
        float3 c = mix(s0, s1, tStar);
        C.p[C.n] = 0.5 * ((c - r * sdS.n) + sdS.cl);
        C.nrm[C.n] = sdS.n; C.depth[C.n] = -fS; C.tag[C.n] = 0u; C.n++;
    }
    for (int k = 0; k < 2; ++k) {
        if (!endOK[k]) continue;
        C.p[C.n] = 0.5 * ((st[k] - rr[k] * sdE[k].n) + sdE[k].cl);
        C.nrm[C.n] = sdE[k].n; C.depth[C.n] = -fE[k]; C.tag[C.n] = uint(k + 1); C.n++;
    }
}

// ── Emission with a full feature identity ─────────────────────────────────────────

// Static contacts carry the collider index in meta.z's low 16 bits and a 16-bit fold of
// the feature above it (the solve masks meta.z & 0xFFFF — every older path emits
// feature 0 there, so they are unchanged); dynamic contacts carry the full 32-bit feature
// in meta.z. pairKey keeps its layout with a hash of the feature in its top 8 bits; the
// warm-start match compares the whole meta, so a key collision can never seed a
// different contact.
static uint cdFold16(uint f) { return (f ^ (f >> 16) ^ (f >> 5)) & 0xFFFFu; }


static bool cdContactOK(float3 nBtoA, float3 cp, float depth) {
    return !(isnan(depth) || any(isnan(nBtoA)) || any(isnan(cp))) && length(nBtoA) > 1e-12;
}
static void cdWriteContactF(device CoinContact* contacts, uint slot,
                            uint a, uint b, uint feature, uint colliderIdx,
                            float3 nBtoA, float3 cp, float3 xa, float3 xb, float depth, float auxW) {
    float3 n = normalize(nBtoA);
    float3 t1, t2; cdContactTangents(n, t1, t2);
    bool st = (b == CD_STATIC);
    uint lo = min(a, b);
    uint hi = st ? (0xFFFu - min(colliderIdx, 0xFFFu)) : max(a, b);
    CoinContact c;
    // pairKey: the PAIR only, with the reserved feature byte 0xFF (every older path keys
    // its 8-bit feature there, all < 0xFF) — coinWarmStartMatch then matches a polytope
    // contact by exact identity first and, when a clipped point's feature id changed
    // (a corner overhanging the reference face by a hair is replaced by two clip points,
    // and back), by the nearest old point of the same pair (CD_KEY_POSMATCH).
    c.meta = uint4(a, b, st ? ((colliderIdx & 0xFFFFu) | (cdFold16(feature) << 16)) : feature,
                   (lo & 0xFFFu) | ((hi & 0xFFFu) << 12) | CD_KEY_POSMATCH);
    c.nrm  = float4(n, depth);
    c.rA   = float4(cp - xa, 0.0);
    c.rB   = float4(st ? float3(0.0) : (cp - xb), 0.0);
    c.tan1 = float4(t1, 0.0);
    c.tan2 = float4(t2, -1.0);
    c.aux  = float4(0.0, 0.0, 0.0, auxW);
    c.ext  = float4(0.0);
    contacts[slot] = c;
}
static void cdEmitContactF(device CoinContact* contacts, device atomic_uint* contactCount, uint maxContacts,
                           uint a, uint b, uint feature, uint colliderIdx,
                           float3 nBtoA, float3 cp, float3 xa, float3 xb, float depth) {
    if (!cdContactOK(nBtoA, cp, depth)) return;
    uint slot = atomic_fetch_add_explicit(contactCount, 1u, memory_order_relaxed);
    if (slot >= maxContacts) return;
    cdWriteContactF(contacts, slot, a, b, feature, colliderIdx, nBtoA, cp, xa, xb, depth, 0.0);
}
// A manifold of n ≤ 4 points of one pair: grouped (contiguous, head −n / points −0.25)
// under CD_FLAG_MANIFOLD_SOLVE, else n independent contacts (see cdEmitGroup).
static void cdEmitGroupF(device CoinContact* contacts, device atomic_uint* contactCount, uint maxContacts,
                         constant CoinUniforms& u, uint a, uint b, uint colliderIdx, int n,
                         thread const uint* feature, thread const float3* nBtoA, thread const float3* cp,
                         float3 xa, float3 xb, thread const float* depth) {
    int m = 0; int idx[4];
    for (int i = 0; i < n && m < 4; ++i) if (cdContactOK(nBtoA[i], cp[i], depth[i])) idx[m++] = i;
    if (m == 0) return;
    bool grouped = (u.solverFlags & CD_FLAG_MANIFOLD_SOLVE) != 0u && m > 1;
    if (grouped) {
        // Written in CANONICAL order (ascending feature id — cdCanonicalManifoldOrder, the
        // same rule cdEmitGroup applies to the older paths), because the manifold solve visits
        // the points in the order they are written. cdPolyManifold hands them over in
        // cdReduceManifold's deepest-first order, and on a flat rest the depths tie to float
        // noise — so, unsorted here while the plane / box paths were sorted, the three-bar
        // stack of CoinDEMContactsTests drifted 0.074 mm / 0.118° in 9 s awake (40 spawn
        // positions: worst 0.074 mm; sorted: 0.0003 mm / 0.00007°), and the fence's stocked
        // bars turned 0.61° cold (sorted: 0.052°).
        uint fs[4]; int ord[4];
        for (int i = 0; i < m; ++i) fs[i] = feature[idx[i]];
        cdCanonicalManifoldOrder(fs, m, ord);
        uint slot = atomic_fetch_add_explicit(contactCount, uint(m), memory_order_relaxed);
        bool fits = slot + uint(m) <= maxContacts;           // straddling the end: ungrouped, no holes
        for (int i = 0; i < m && slot + uint(i) < maxContacts; ++i) {
            int k = idx[ord[i]];
            cdWriteContactF(contacts, slot + uint(i), a, b, feature[k], colliderIdx, nBtoA[k], cp[k],
                            xa, xb, depth[k], !fits ? 0.0 : (i == 0 ? -float(m) : -0.25));
        }
        return;
    }
    for (int i = 0; i < m; ++i) {
        uint slot = atomic_fetch_add_explicit(contactCount, 1u, memory_order_relaxed);
        if (slot >= maxContacts) return;
        cdWriteContactF(contacts, slot, a, b, feature[idx[i]], colliderIdx, nBtoA[idx[i]], cp[idx[i]], xa, xb, depth[idx[i]], 0.0);
    }
}

// Make the tags of one emitted batch unique (the colouring's identity order and the warm
// start both assume a pair's contacts differ): a rare duplicate gets its batch index mixed
// in, deterministically.
static void cdUniqueTags(thread uint* tags, int n) {
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (tags[i] == tags[j]) tags[i] ^= (uint(i) << 12) | 0x800u;
}

// Tight bounding radius about the COM (broadphase neighbourhood sizing, VZ-0161).
static float cdTightBound(CoinBody c) {
    if (cdIsCapsule(c)) return cdHalfThickOf(c);                          // hl + r
    if (cdIsSphere(c) || cdIsBox(c) || cdIsHull(c) || cdIsEgg(c) || cdIsCompound(c)) return cdRadiusOf(c);
    float R = cdRadiusOf(c), h = cdHalfThickOf(c);
    return sqrt(R * R + h * h);                                            // disc
}

// Broadphase neighbourhood half-width in cells for body c: every pair it can touch lies
// within (its bound + the largest live bound) of its COM. 1 (the classic 3×3×3) whenever
// the solver's cell covers its bodies — bit-identical to before; wider only for a world
// whose bodies outgrew the cell (VZ-0161: a 0.08 m capsule in a 0.12 m cell missed pairs
// two cells apart, 20–30 mm deep).
static int cdScanCells(CoinBody c, constant CoinUniforms& u) {
    float reach = (cdTightBound(c) + u.maxBodyBound) * u.invCell;
    return clamp(int(ceil(reach - 1e-4)), 1, 4);
}

// ── The kernels: candidate piece pairs, then one thread per pair ────────────────────
//
// A per-body thread that ran every heavy pair it owned serially made the GPU wait on the
// slowest body (a capsule touching four hulls: four golden-section searches back to back).
// So generation is split: coinGenerateContacts' own per-body broadphase appends each
// surviving PIECE pair (a compound contributes one per child) of a pair cdPairIsPoly owns to
// a pair list (cdAppendPolyPairs / cdAppendStaticBoxPairs), and coinPolyNarrow (one thread
// per list entry, indirect dispatch) runs that pair's narrowphase and emits its manifold.
// The critical path is one pair, not one body's worth of pairs, and the neighbourhood is
// scanned once.

// Feature layout of a polytope-narrowphase contact (meta.z for a dynamic pair, folded
// into meta.z's high half for a static one): bits 30–31 kind (0/1 face on A/B, 2 edge,
// 3 swept), 26–29 the thread body's piece (compound child), 22–25 the other body's piece,
// 0–21 cdPolyManifold's reference id + point tag (or the swept station). A piece index
// past 15 (a compound of 17…128 children) keeps its low 4 bits there and puts its high 3
// bits in 9–11 (the thread body's) / 12–14 (the other's): every tag leaves bits 9–14 clear
// (a clipped point's tag is ≤ 271, an edge id ≤ 191, a swept station ≤ 2, a probe corner
// ≤ 7), so each (piece, piece, tag) keeps its own id (cdUniqueTags' rare de-duplication
// XOR of bits 11–13 aside) — and a piece below 16 has no high bits, so every compound of
// ≤ 16 children keeps its feature ids bit for bit.
static inline uint cdPieceFeat(uint k, bool threadSide) {
    return threadSide ? (((k & 15u) << 26) | ((k >> 4) << 9)) : (((k & 15u) << 22) | ((k >> 4) << 12));
}
constant uint CD_FEAT_SWEPT = 3u << 30;

// Pair-list record: x = body A (the lower index — the contact's A), y = body B or CD_STATIC,
// z = collider index (static), w = kind << 28 | polySideIsA << 16 | pieceB << 8 | pieceA.
constant uint CD_PP_POLY   = 1u;     // polytope piece × polytope piece
constant uint CD_PP_SWEPT  = 2u;     // swept sphere × polytope piece
constant uint CD_PP_DISC   = 3u;     // compound piece × disc (its prism, cdPolyDisc)
constant uint CD_PP_STATIC = 4u;     // polytope piece × static box (kinds 1, 3)
constant uint CD_PP_GJK    = 5u;     // compound piece × a hull WITHOUT topology (GJK/EPA)
// Feature of a CD_PP_GJK contact: the swept kind with station 7 (a compound–hull pair never
// has swept contacts, and swept stations are 0–2), plus the child's piece bits.
constant uint CD_FEAT_GJK = (3u << 30) | 7u;

// Pieces of a polytope body: a compound's children, or the body itself (index 0).
static uint cdPieceCount(CoinBody b, device const uint2* hr) { return cdIsCompound(b) ? cdCompoundCount(b, hr) : 1u; }

// Piece k of body b as a polytope view in the working frame O (box, child box, or hull).
static CdPoly cdPieceView(CoinBody b, uint k, float3 O, device const float4* hv, device const uint2* hr) {
    if (cdIsCompound(b)) return cdPolyBox(cdCompoundChild(b, k, hv, hr), O);
    if (cdIsBox(b)) return cdPolyBox(cdBoxOfBody(b), O);
    return cdPolyHull(b, cdHullTopoOf(b, hv, hr), O);
}
// Its bounding sphere (world centre, radius).
static float4 cdPieceSphere(CoinBody b, uint k, device const float4* hv, device const uint2* hr) {
    if (cdIsCompound(b)) { CdBox c = cdCompoundChild(b, k, hv, hr); return float4(c.c, length(c.he)); }
    return float4(b.posInvMass.xyz, cdRadiusOf(b));
}
// As a point-query piece (box or hull).
static CdPiece cdPieceQuery(CoinBody b, uint k, device const float4* hv, device const uint2* hr) {
    CdPiece P;
    P.box = !cdIsHull(b);
    P.b = cdIsCompound(b) ? cdCompoundChild(b, k, hv, hr) : cdBoxOfBody(b);
    return P;
}

// Compound child k as a stand-alone BOX body, for the GJK/EPA fallback against a hull that
// carries no face topology (cdSupport / cdAxisProbe read only its pose and half-extents).
//
// plausibility: real — GJK/EPA is exact for the overlap depth and normal of a box against a
// convex vertex hull; the contact is ONE point (the midpoint of the two supports along the
// normal, the old hull path's EPA contact), not a clipped face manifold, and nothing
// speculative. Only a hull beyond the topology caps (> 64 faces, a face loop > 16 or > 192
// edges) meets a compound this way; before, such a hull was taken for a DISC (its bounding
// radius × its smallest half-extent) and a compound 1.3 mm clear of a 20 mm icosphere got
// four 3.4–4.0 mm "contacts" (verifier, stage B1).
static CoinBody cdChildAsBoxBody(CoinBody b, CdBox c) {
    CoinBody t = b;
    t.posInvMass = float4(c.c, b.posInvMass.w);
    t.orient = c.q;
    t.shapeExtents = float4(c.he, 1.0);                        // w = 1 → box
    t.prevPos.w = length(c.he);
    t.vel.w = min(c.he.x, min(c.he.y, c.he.z));
    return t;
}

// Emit a polytope manifold (points in the working frame O).
static void cdEmitManifold(thread CdManifold& M, float3 O, uint featHi,
                           device CoinContact* contacts, device atomic_uint* contactCount, uint maxContacts,
                           constant CoinUniforms& u, uint a, uint b, uint colliderIdx, float3 xa, float3 xb) {
    uint tags[4]; float3 nn[4], pp[4];
    for (int m = 0; m < M.n; ++m) { tags[m] = M.tag[m] | featHi; nn[m] = M.nrm; pp[m] = O + M.p[m]; }
    cdUniqueTags(tags, M.n);
    cdEmitGroupF(contacts, contactCount, maxContacts, u, a, b, colliderIdx, M.n, tags, nn, pp, xa, xb, M.depth);
}

// Reach of a bounding-sphere pre-filter, padded for float rounding exactly like the
// per-collider reject's rCull (CoinDEM.metal, VZ-0151). Unpadded, a body whose nearest
// feature sits exactly AT the speculative margin along its bounding radius — a cube's
// corner pointing straight down at a box face — was culled by the rounding of its own
// centre distance (VZ-0163: 2.0000098e-4 > 2e-4 → no contact).
static inline float cdReach(float r, float spec) { return r * 1.0001 + 1e-6 + spec; }

static void cdAppendPair(device uint4* pairs, device atomic_uint* pairCount, uint maxPairs, uint4 rec) {
    uint slot = atomic_fetch_add_explicit(pairCount, 1u, memory_order_relaxed);
    if (slot < maxPairs) pairs[slot] = rec;
}

// ── Culling clusters of a compound of more than CD_CFLAT children ────────────────────
//
// (CoinCompound.swift step 4; table layout in CoinDEMSolver+Compound.swift.) The pair
// producers below test a cluster's world AABB before any of its children and skip the
// cluster whole when even that box is farther than the speculative margin (+ the SAT's
// length tolerance and a float slack) from the partner: no child in it could produce a
// contact, so the contact set is exactly the unclustered one — only candidate pairs that
// could never touch go unlisted. A 116-child crane jib walked every child against every
// static and every partner on one thread each substep (≈ 21 000 child poses): the
// clock-like fence's physics went 3.8 → 11.3 ms (small-world) and 6.5 → 42 ms (multi-
// dispatch). A compound of ≤ CD_CFLAT children has no clusters and takes the old loops.
constant uint CD_CFLAT = 16u;

struct CdCluster { float3 c; float3 he; uint first, count; };

// Clusters of body b (0: none — ≤ CD_CFLAT children, or not a compound).
static uint cdClusterCount(CoinBody b, device const float4* hv, device const uint2* hr) {
    if (!cdIsCompound(b)) return 0u;
    uint2 r = hr[uint(b.hullRef.x + 0.5)];
    return r.y > CD_CFLAT ? as_type<uint>(hv[r.x + 3u * r.y].x) : 0u;
}
// Cluster g in world space: its AABB's centre rotated with the body, half-extents |R|·he
// (R = cdQuatToMat(b.orient), computed once by the caller for all of b's clusters).
static CdCluster cdClusterOf(CoinBody b, float3x3 R, uint g, device const float4* hv, device const uint2* hr) {
    uint2 r = hr[uint(b.hullRef.x + 0.5)];
    uint t = r.x + 3u * r.y + 1u + 2u * g;
    float4 a = hv[t], e = hv[t + 1u];
    CdCluster k;
    k.c = b.posInvMass.xyz + R * a.xyz;
    k.he = abs(R[0]) * e.x + abs(R[1]) * e.y + abs(R[2]) * e.z;
    uint bits = as_type<uint>(a.w);
    k.first = bits & 0xFFFFu;
    k.count = bits >> 16;
    return k;
}
// Entry i of body b's cluster member list (child indices, grouped by cluster; G clusters).
static uint cdClusterChild(CoinBody b, uint G, uint i, device const float4* hv, device const uint2* hr) {
    uint2 r = hr[uint(b.hullRef.x + 0.5)];
    uint4 u = as_type<uint4>(hv[r.x + 3u * r.y + 1u + 2u * G + (i >> 3)]);
    uint w = u[(i >> 1) & 3u];
    return (i & 1u) ? (w >> 16) : (w & 0xFFFFu);
}
// The margin a cluster is tested with: the speculative margin, the SAT's length tolerance
// for its largest child (≤ 1e-5 × the cluster's half-diagonal) and a float slack.
static inline float cdClusterMargin(CdCluster K, float spec) { return spec + 1e-5 * length(K.he) + 1e-6; }
// Could anything inside cluster K come within `m` of the sphere (centre p, radius r)?
static inline bool cdClusterNearSphere(CdCluster K, float3 p, float r, float m) {
    float3 g = max(abs(p - K.c) - K.he, float3(0.0));
    float reach = r + m;
    return dot(g, g) <= reach * reach;
}
// Could anything inside cluster K come within `m` of the box (centre bc, half-extents bhe,
// orientation q)? Separating axes: the box's three and the world's three.
static inline bool cdClusterNearBox(CdCluster K, float3 bc, float3 bhe, float4 q, float m) {
    float3x3 Rb = cdQuatToMat(q);
    float3 d = K.c - bc;
    for (int i = 0; i < 3; ++i)
        if (abs(dot(Rb[i], d)) > dot(abs(Rb[i]), K.he) + bhe[i] + m) return false;
    float3 bw = abs(Rb[0]) * bhe.x + abs(Rb[1]) * bhe.y + abs(Rb[2]) * bhe.z;
    return all(abs(d) <= K.he + bw + m);
}

// Deepest penetration of a pair cdPairIsPoly owns, by the same geometry the narrowphase
// generates with (the penetration probe, CoinDEMSolver.measurePenetration). ≤ 0 = apart.
static float cdPolyPairDepth(CoinBody ci, CoinBody cj, constant CoinUniforms& u,
                             device const float4* hv, device const uint2* hr) {
    float best = -1e30;
    bool pi = cdIsPolyBody(ci, hv, hr), pj = cdIsPolyBody(cj, hv, hr);
    uint ni = cdPieceCount(ci, hr), nj = cdPieceCount(cj, hr);
    if (pi && pj) {
        for (uint a = 0; a < ni; ++a) {
            float4 sa = cdPieceSphere(ci, a, hv, hr);
            for (uint b = 0; b < nj; ++b) {
                float4 sb = cdPieceSphere(cj, b, hv, hr);
                if (distance(sa.xyz, sb.xyz) > sa.w + sb.w) continue;
                float3 O = 0.5 * (sa.xyz + sb.xyz);
                best = max(best, cdPolyDepth(cdPieceView(ci, a, O, hv, hr), cdPieceView(cj, b, O, hv, hr), hv));
            }
        }
        return best;
    }
    CoinBody P = pi ? ci : cj, S = pi ? cj : ci;
    uint nP = pi ? ni : nj;
    if (cdIsRound(S)) {
        float3 s0, s1; float r0, r1;
        if (cdIsSphere(S)) { s0 = S.posInvMass.xyz; s1 = s0; r0 = cdRadiusOf(S); r1 = r0; }
        else cdSweptSegment(S, s0, s1, r0, r1);
        CdTopo t = cdIsHull(P) ? cdHullTopoOf(P, hv, hr) : CdTopo{0u, 0u, 0u, 0u, 0u, 0u, 0u};
        for (uint k = 0; k < nP; ++k) {
            CdSweptContact C;
            cdSweptVsPiece(s0, s1, r0, r1, cdPieceQuery(P, k, hv, hr), P, t, 1e30, 0.0, hv, C);
            for (int m = 0; m < C.n; ++m) best = max(best, C.depth[m]);
        }
        return best;
    }
    if (cdIsHull(S)) {                                        // compound × topology-less hull
        for (uint k = 0; k < nP; ++k) {
            float3 n; float d;
            if (cdGJKEPA(cdChildAsBoxBody(P, cdCompoundChild(P, k, hv, hr)), S, hv, hr, n, d)) best = max(best, d);
        }
        return best;
    }
    for (uint k = 0; k < nP; ++k) {                           // compound × disc (its prism)
        float4 sp = cdPieceSphere(P, k, hv, hr);
        float3 O = 0.5 * (sp.xyz + S.posInvMass.xyz);
        best = max(best, cdPolyDepth(cdPieceView(P, k, O, hv, hr), cdPolyDisc(S, O), hv));
    }
    return best;
}

// ── Pair-list producers, called from coinGenerateContacts (its broadphase scan is shared) ──

// cdAppendPolyPairs for a pair with a clustered compound on either side: the same records
// (the same per-piece sphere tests, the same kinds and piece bits), each cluster first
// tested against the partner — its bounding sphere, or for two compounds the other side's
// piece sphere — and skipped whole when even its box is out of reach.
__attribute__((noinline)) static void cdAppendPolyPairsClustered(uint id, uint j, CoinBody ci, CoinBody cj, float spec, uint gi, uint gj,
                                       device const float4* hv, device const uint2* hr,
                                       device uint4* pairs, device atomic_uint* pairCount, uint maxPairs) {
    bool pi = cdIsPolyBody(ci, hv, hr), pj = cdIsPolyBody(cj, hv, hr);
    uint ni = cdPieceCount(ci, hr), nj = cdPieceCount(cj, hr);
    if (pi && pj) {
        float4 bj = float4(cj.posInvMass.xyz, cdRadiusOf(cj));
        uint ci0 = gi > 0u ? gi : 1u;
        float3x3 Ri = cdQuatToMat(ci.orient), Rj = cdQuatToMat(cj.orient);
        for (uint g = 0; g < ci0; ++g) {
            uint a0 = 0u, na = ni;
            if (gi > 0u) {
                CdCluster K = cdClusterOf(ci, Ri, g, hv, hr);
                if (!cdClusterNearSphere(K, bj.xyz, bj.w, cdClusterMargin(K, spec))) continue;
                a0 = K.first; na = K.count;
            }
            for (uint ia = 0; ia < na; ++ia) {
                uint a = gi > 0u ? cdClusterChild(ci, gi, a0 + ia, hv, hr) : ia;
                float4 sa = cdPieceSphere(ci, a, hv, hr);
                if (distance(sa.xyz, bj.xyz) > cdReach(sa.w + bj.w, spec)) continue;
                uint cj0 = gj > 0u ? gj : 1u;
                for (uint h = 0; h < cj0; ++h) {
                    uint b0 = 0u, nb = nj;
                    if (gj > 0u) {
                        CdCluster L = cdClusterOf(cj, Rj, h, hv, hr);
                        if (!cdClusterNearSphere(L, sa.xyz, sa.w, cdClusterMargin(L, spec))) continue;
                        b0 = L.first; nb = L.count;
                    }
                    for (uint ib = 0; ib < nb; ++ib) {
                        uint b = gj > 0u ? cdClusterChild(cj, gj, b0 + ib, hv, hr) : ib;
                        float4 sb = cdPieceSphere(cj, b, hv, hr);
                        if (distance(sa.xyz, sb.xyz) > cdReach(sa.w + sb.w, spec)) continue;
                        cdAppendPair(pairs, pairCount, maxPairs, uint4(id, j, 0u, (CD_PP_POLY << 28) | (b << 8) | a));
                    }
                }
            }
        }
        return;
    }
    bool polyIsA = pi;
    CoinBody P = polyIsA ? ci : cj, S = polyIsA ? cj : ci;
    uint nP = polyIsA ? ni : nj, gP = polyIsA ? gi : gj;
    // The partner's bounding sphere (a swept body's segment by its own bound).
    float3 sc; float sr;
    float3 s0 = S.posInvMass.xyz, s1 = s0; float rMax = 0.0;
    if (cdIsRound(S)) {
        float r0, r1;
        if (cdIsSphere(S)) { r0 = cdRadiusOf(S); r1 = r0; }
        else cdSweptSegment(S, s0, s1, r0, r1);
        rMax = max(r0, r1);
        sc = 0.5 * (s0 + s1); sr = 0.5 * distance(s0, s1) + rMax;
    } else if (cdIsHull(S)) {
        sc = S.posInvMass.xyz; sr = cdRadiusOf(S);
    } else {
        sc = S.posInvMass.xyz; sr = length(float2(cdRadiusOf(S), cdHalfThickOf(S)));
    }
    uint kindBits = cdIsRound(S) ? CD_PP_SWEPT : (cdIsHull(S) ? CD_PP_GJK : CD_PP_DISC);
    uint cp0 = gP > 0u ? gP : 1u;
    float3x3 RP = cdQuatToMat(P.orient);
    for (uint g = 0; g < cp0; ++g) {
        uint k0 = 0u, nk = nP;
        if (gP > 0u) {
            CdCluster K = cdClusterOf(P, RP, g, hv, hr);
            if (!cdClusterNearSphere(K, sc, sr, cdClusterMargin(K, spec))) continue;
            k0 = K.first; nk = K.count;
        }
        for (uint ik = 0; ik < nk; ++ik) {
            uint k = gP > 0u ? cdClusterChild(P, gP, k0 + ik, hv, hr) : ik;
            float4 sp = cdPieceSphere(P, k, hv, hr);
            bool near;
            if (kindBits == CD_PP_SWEPT)    near = distance(cdClosestOnSegment(s0, s1, sp.xyz), sp.xyz) <= cdReach(sp.w + rMax, spec);
            else                             near = distance(sp.xyz, S.posInvMass.xyz) <= cdReach(sp.w + sr, spec);
            if (!near) continue;
            cdAppendPair(pairs, pairCount, maxPairs, uint4(id, j, 0u, (kindBits << 28) | (polyIsA ? (1u << 16) : 0u) | k));
        }
    }
}

// A dynamic pair cdPairIsPoly assigned to the polytope narrowphase, past the pair gates
// (bounding reject, joint skip): append one record per candidate PIECE pair.
static void cdAppendPolyPairs(uint id, uint j, CoinBody ci, CoinBody cj, float spec,
                              device const float4* hv, device const uint2* hr,
                              device uint4* pairs, device atomic_uint* pairCount, uint maxPairs) {
    bool pi = cdIsPolyBody(ci, hv, hr), pj = cdIsPolyBody(cj, hv, hr);
    uint ni = cdPieceCount(ci, hr), nj = cdPieceCount(cj, hr);
    uint gi = cdClusterCount(ci, hv, hr), gj = cdClusterCount(cj, hv, hr);
    if (gi > 0u || gj > 0u) {                                  // a clustered compound: cdAppendPolyPairsClustered
        cdAppendPolyPairsClustered(id, j, ci, cj, spec, gi, gj, hv, hr, pairs, pairCount, maxPairs);
        return;
    }
    if (pi && pj) {
        float4 bj = float4(cj.posInvMass.xyz, cdRadiusOf(cj));
        for (uint a = 0; a < ni; ++a) {
            float4 sa = cdPieceSphere(ci, a, hv, hr);
            if (distance(sa.xyz, bj.xyz) > cdReach(sa.w + bj.w, spec)) continue;
            for (uint b = 0; b < nj; ++b) {
                float4 sb = cdPieceSphere(cj, b, hv, hr);
                if (distance(sa.xyz, sb.xyz) > cdReach(sa.w + sb.w, spec)) continue;
                cdAppendPair(pairs, pairCount, maxPairs, uint4(id, j, 0u, (CD_PP_POLY << 28) | (b << 8) | a));
            }
        }
        return;
    }
    bool polyIsA = pi;
    CoinBody P = polyIsA ? ci : cj, S = polyIsA ? cj : ci;
    uint nP = polyIsA ? ni : nj;
    if (cdIsRound(S)) {
        float3 s0, s1; float r0, r1;
        if (cdIsSphere(S)) { s0 = S.posInvMass.xyz; s1 = s0; r0 = cdRadiusOf(S); r1 = r0; }
        else cdSweptSegment(S, s0, s1, r0, r1);
        float rMax = max(r0, r1);
        for (uint k = 0; k < nP; ++k) {
            float4 sp = cdPieceSphere(P, k, hv, hr);
            if (distance(cdClosestOnSegment(s0, s1, sp.xyz), sp.xyz) > cdReach(sp.w + rMax, spec)) continue;
            cdAppendPair(pairs, pairCount, maxPairs, uint4(id, j, 0u, (CD_PP_SWEPT << 28) | (polyIsA ? (1u << 16) : 0u) | k));
        }
        return;
    }
    if (cdIsHull(S)) {                                         // compound × topology-less hull: GJK/EPA
        for (uint k = 0; k < nP; ++k) {
            float4 sp = cdPieceSphere(P, k, hv, hr);
            if (distance(sp.xyz, S.posInvMass.xyz) > cdReach(sp.w + cdRadiusOf(S), spec)) continue;
            cdAppendPair(pairs, pairCount, maxPairs, uint4(id, j, 0u, (CD_PP_GJK << 28) | (polyIsA ? (1u << 16) : 0u) | k));
        }
        return;
    }
    float rD = length(float2(cdRadiusOf(S), cdHalfThickOf(S)));   // compound × disc: its bounding sphere
    for (uint k = 0; k < nP; ++k) {
        float4 sp = cdPieceSphere(P, k, hv, hr);
        if (distance(sp.xyz, S.posInvMass.xyz) > cdReach(sp.w + rD, spec)) continue;
        cdAppendPair(pairs, pairCount, maxPairs, uint4(id, j, 0u, (CD_PP_DISC << 28) | (polyIsA ? (1u << 16) : 0u) | k));
    }
}

// cdAppendStaticBoxPairs for a clustered compound: each cluster's box tested against the
// static box first (the same per-piece records after it).
__attribute__((noinline)) static void cdAppendStaticBoxPairsClustered(uint id, CoinBody ci, uint k, CoinStaticCollider col, float spec,
                                            uint G, device const float4* hv, device const uint2* hr,
                                            device uint4* pairs, device atomic_uint* pairCount, uint maxPairs) {
    uint ck = as_type<uint>(col.a.w);
    float4 sq = (ck == 3u) ? col.orient : float4(0.0, 0.0, 0.0, 1.0);
    float3x3 R = cdQuatToMat(ci.orient);
    for (uint g = 0; g < G; ++g) {
        CdCluster K = cdClusterOf(ci, R, g, hv, hr);
        if (!cdClusterNearBox(K, col.a.xyz, col.b.xyz, sq, cdClusterMargin(K, spec))) continue;
        for (uint i = 0; i < K.count; ++i) {
            uint pc = cdClusterChild(ci, G, K.first + i, hv, hr);
            float4 sp = cdPieceSphere(ci, pc, hv, hr);
            float3 gl = max(abs(cdQuatRotateInv(sq, sp.xyz - col.a.xyz)) - col.b.xyz, float3(0.0));
            float reach = cdReach(sp.w, spec);
            if (dot(gl, gl) > reach * reach) continue;
            cdAppendPair(pairs, pairCount, maxPairs, uint4(id, CD_STATIC, k, (CD_PP_STATIC << 28) | pc));
        }
    }
}

// A polytope body against a static box (kinds 1, 3) that passed the per-collider reject:
// one record per piece within reach.
static void cdAppendStaticBoxPairs(uint id, CoinBody ci, uint k, CoinStaticCollider col, float spec,
                                   device const float4* hv, device const uint2* hr,
                                   device uint4* pairs, device atomic_uint* pairCount, uint maxPairs) {
    uint G = cdClusterCount(ci, hv, hr);
    if (G > 0u) {                                              // a clustered compound: its clusters first
        cdAppendStaticBoxPairsClustered(id, ci, k, col, spec, G, hv, hr, pairs, pairCount, maxPairs);
        return;
    }
    uint ck = as_type<uint>(col.a.w);
    float4 sq = (ck == 3u) ? col.orient : float4(0.0, 0.0, 0.0, 1.0);
    uint ni = cdPieceCount(ci, hr);
    for (uint pc = 0; pc < ni; ++pc) {
        float4 sp = cdPieceSphere(ci, pc, hv, hr);
        float3 gl = max(abs(cdQuatRotateInv(sq, sp.xyz - col.a.xyz)) - col.b.xyz, float3(0.0));
        float reach = cdReach(sp.w, spec);
        if (dot(gl, gl) > reach * reach) continue;
        cdAppendPair(pairs, pairCount, maxPairs, uint4(id, CD_STATIC, k, (CD_PP_STATIC << 28) | pc));
    }
}

// cdCompoundStaticProbes' plane case for a clustered compound: a cluster whose box stays
// above the plane by more than the margin is skipped whole (its children's corners are all
// inside that box); the rest take the same corner probes.
__attribute__((noinline)) static void cdCompoundPlaneProbesClustered(uint id, CoinBody ci, uint k, CoinStaticCollider col, uint G,
                                           constant CoinUniforms& u, device const float4* hv, device const uint2* hr,
                                           device CoinContact* contacts, device atomic_uint* contactCount, uint maxContacts) {
    float spec = max(u.speculativeMargin, 0.0);
    float3 xi = ci.posInvMass.xyz;
    float3 n = col.a.xyz; float d = col.b.w;
    float3x3 R = cdQuatToMat(ci.orient);
    for (uint g = 0; g < G; ++g) {
        CdCluster K = cdClusterOf(ci, R, g, hv, hr);
        float deepest = d - (dot(n, K.c) - dot(abs(n), K.he));   // the cluster box's deepest point
        if (deepest < -(spec + 1e-6 * (abs(d) + length(K.he)) + 1e-6)) continue;
        for (uint i = 0; i < K.count; ++i) {
            uint pc = cdClusterChild(ci, G, K.first + i, hv, hr);
            CdBox cb = cdCompoundChild(ci, pc, hv, hr);
            float3 pts[8];
            for (uint v = 0; v < 8u; ++v)
                pts[v] = cb.c + cdQuatRotate(cb.q, float3((v & 1u) ? cb.he.x : -cb.he.x, (v & 2u) ? cb.he.y : -cb.he.y, (v & 4u) ? cb.he.z : -cb.he.z));
            float3 mPos[8]; float mDep[8]; uint mFeat[8]; int nc = 0;
            for (uint v = 0; v < 8u; ++v) {
                float pen = d - dot(n, pts[v]);
                if (pen >= -(spec + 1e-6 * (abs(d) + length(cb.he)) + 1e-9)) { mPos[nc] = pts[v]; mDep[nc] = pen; mFeat[nc] = v; nc++; }
            }
            int keep[CD_MANIFOLD_MAX];
            int m = cdReduceManifold(mPos, mDep, nc, keep);
            uint gf[4]; float3 gn[4], gp[4]; float gd[4];
            for (int j = 0; j < m; ++j) { gf[j] = cdPieceFeat(pc, true) | mFeat[keep[j]]; gn[j] = n; gp[j] = mPos[keep[j]]; gd[j] = mDep[keep[j]]; }
            cdEmitGroupF(contacts, contactCount, maxContacts, u, id, CD_STATIC, k, m, gf, gn, gp, xi, xi, gd);
        }
    }
}

// A compound's children against a plane / tube / pusher (kinds 0, 4, 2): each child's 8
// corners (radius 0) — exactly the probe logic the box path uses for its corners.
static void cdCompoundStaticProbes(uint id, CoinBody ci, uint k, CoinStaticCollider col, uint kind,
                                   constant CoinUniforms& u, device const float4* hv, device const uint2* hr,
                                   device CoinContact* contacts, device atomic_uint* contactCount, uint maxContacts) {
    if (kind == 0u) {                                          // a clustered compound against a plane
        uint G = cdClusterCount(ci, hv, hr);
        if (G > 0u) { cdCompoundPlaneProbesClustered(id, ci, k, col, G, u, hv, hr, contacts, contactCount, maxContacts); return; }
    }
    float spec = max(u.speculativeMargin, 0.0);
    float3 xi = ci.posInvMass.xyz;
    uint ni = cdPieceCount(ci, hr);
    for (uint pc = 0; pc < ni; ++pc) {
        CdBox cb = cdCompoundChild(ci, pc, hv, hr);
        float3 pts[8];
        for (uint v = 0; v < 8u; ++v)
            pts[v] = cb.c + cdQuatRotate(cb.q, float3((v & 1u) ? cb.he.x : -cb.he.x, (v & 2u) ? cb.he.y : -cb.he.y, (v & 4u) ? cb.he.z : -cb.he.z));
        if (kind == 0u) {
            float3 n = col.a.xyz; float d = col.b.w;
            float3 mPos[8]; float mDep[8]; uint mFeat[8]; int nc = 0;
            for (uint v = 0; v < 8u; ++v) {
                float pen = d - dot(n, pts[v]);
                if (pen >= -(spec + 1e-6 * (abs(d) + length(cb.he)) + 1e-9)) { mPos[nc] = pts[v]; mDep[nc] = pen; mFeat[nc] = v; nc++; }
            }
            int keep[CD_MANIFOLD_MAX];
            int m = cdReduceManifold(mPos, mDep, nc, keep);
            uint gf[4]; float3 gn[4], gp[4]; float gd[4];
            for (int i = 0; i < m; ++i) { gf[i] = cdPieceFeat(pc, true) | mFeat[keep[i]]; gn[i] = n; gp[i] = mPos[keep[i]]; gd[i] = mDep[keep[i]]; }
            cdEmitGroupF(contacts, contactCount, maxContacts, u, id, CD_STATIC, k, m, gf, gn, gp, xi, xi, gd);
        } else if (kind == 4u) {
            float3 ctr = col.a.xyz, ax = col.b.xyz, up = col.vel.xyz;
            float R = col.b.w, halfLen = col.vel.w;
            bool lowerOnly = (col.meta.x & 1u) != 0u;
            for (uint v = 0; v < 8u; ++v) {
                float3 d = pts[v] - ctr;
                float along = dot(d, ax);
                if (along < -halfLen || along > halfLen) continue;
                float3 radial = d - ax * along;
                float dr = length(radial);
                if (dr <= 1e-5 || dr >= R) continue;
                float3 rn = radial / dr;
                bool active = !lowerOnly || dot(rn, up) < 0.05;
                float pen = dr - R;
                if (active && pen > -spec)
                    cdEmitContactF(contacts, contactCount, maxContacts, id, CD_STATIC, cdPieceFeat(pc, true) | v, k, -rn, pts[v], xi, xi, pen);
            }
        } else if (kind == 2u) {
            float3 cc2 = col.a.xyz, he2 = col.b.xyz;
            float frontZ = cc2.z + he2.z, backZ = cc2.z - he2.z;
            float maxPush = max(0.0, col.vel.z) * u.dt;
            if (maxPush <= 0.0) continue;
            float maxDepth = maxPush / max(u.baumgarteBeta, 0.05) + u.contactSlop;
            for (uint v = 0; v < 8u; ++v) {
                float3 sv = pts[v];
                if (abs(sv.x - cc2.x) < he2.x && sv.y > cc2.y - he2.y && sv.y < cc2.y + he2.y && sv.z < frontZ && sv.z > backZ) {
                    float depth = clamp(frontZ - sv.z, 0.0, maxDepth);
                    if (depth > 0.0)
                        cdEmitContactF(contacts, contactCount, maxContacts, id, CD_STATIC, cdPieceFeat(pc, true) | v, k, float3(0, 0, 1), sv, xi, xi, depth);
                }
            }
        }
    }
}

// Indirect threadgroup count for coinPolyNarrow (3 separate uints, 12 bytes).
kernel void coinWritePolyArgs(
    device const atomic_uint& pairCount [[ buffer(0) ]],
    device uint*              args      [[ buffer(1) ]],
    constant uint&            tgSize    [[ buffer(2) ]],
    constant uint&            maxPairs  [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id != 0) return;
    uint n = min(atomic_load_explicit(&pairCount, memory_order_relaxed), maxPairs);
    args[0] = (n + tgSize - 1u) / tgSize;
    args[1] = 1u;
    args[2] = 1u;
}

// One listed piece pair's narrowphase (`n` = the listed pair count, clamped to the list), in
// two halves by pair kind:
//   • cdPolyNarrowThreadBody — ONE thread: a swept sphere / capsule / egg against a polytope
//     piece (golden-section search) and a compound child against a topology-less hull (GJK/EPA);
//   • every pair that runs the separating-axis test (piece × piece, piece × static box, compound
//     child × disc prism): cdSATPairSetup, its first face on ONE thread (cdSATPairFirstFace —
//     most listed pairs end there), then the rest on ONE SIMD GROUP (cdSATPairSolveSG: the SAT
//     on all lanes, the manifold and its emission on lane 0).
// The first returns at once for the second's kinds and vice versa, so a kernel runs both over
// the list (coinPolyNarrow; cdSWPolyNarrow in the small-world kernel). Stage B3 (VZ-0198); the
// SAT answer is bit-identical to the one-thread SAT, which a test seam still runs
// (CD_FLAG_POLY_SERIAL: lane 0 alone, cdPolySAT).
// These four are NOT inlined: the two kernels then run one compiled body of each, and the paths
// agree bit for bit. Inlined into their different surroundings, Metal's fast math rounded the
// copies differently: a tumbling pile of compounds, discs, hulls, boxes, capsules and eggs parted
// between the small-world and multi-dispatch paths after 24 frames (7th digit of a velocity).
constant uint CD_FLAG_POLY_SERIAL = 32u;   // TEST SEAM (CoinDEMSolver.polySATSerialForTesting)

static inline bool cdPolyPairIsSAT(uint kind) { return kind == CD_PP_POLY || kind == CD_PP_STATIC || kind == CD_PP_DISC; }

__attribute__((noinline)) static void cdPolyNarrowThreadBody(uint p, uint n,
    device const CoinBody*           coins,
    constant CoinUniforms&           u,
    device CoinContact*              contacts,
    device atomic_uint*              contactCount,
    constant uint&                   maxContacts,
    device const float4*             hullVerts,
    device const uint2*              hullRanges,
    device const uint4*              pairs)
{
    if (p >= n) return;
    uint4 r = pairs[p];
    uint kind = r.w >> 28;
    if (cdPolyPairIsSAT(kind)) return;                       // the SIMD-group half's
    uint pa = r.w & 0xFFu;
    bool polyIsA = (r.w & (1u << 16)) != 0u;
    uint id = r.x, j = r.y;
    CoinBody ci = coins[id];
    float3 xi = ci.posInvMass.xyz;
    float spec = max(u.speculativeMargin, 0.0);
    CoinBody cj = coins[j];
    float3 xj = cj.posInvMass.xyz;
    CoinBody P = polyIsA ? ci : cj, S = polyIsA ? cj : ci;
    uint k = pa;
    if (kind == CD_PP_SWEPT) {
        float3 s0, s1; float r0, r1;
        if (cdIsSphere(S)) { s0 = S.posInvMass.xyz; s1 = s0; r0 = cdRadiusOf(S); r1 = r0; }
        else cdSweptSegment(S, s0, s1, r0, r1);
        CdTopo t = cdIsHull(P) ? cdHullTopoOf(P, hullVerts, hullRanges) : CdTopo{0u, 0u, 0u, 0u, 0u, 0u, 0u};
        float4 sp = cdPieceSphere(P, k, hullVerts, hullRanges);
        float tol = 1e-5 * min(sp.w, max(max(r0, r1), 1e-4)) + 1e-9;
        CdSweptContact C;
        cdSweptVsPiece(s0, s1, r0, r1, cdPieceQuery(P, k, hullVerts, hullRanges), P, t, spec, tol, hullVerts, C);
        uint tags[3]; float3 nBA[3];
        for (int m = 0; m < C.n; ++m) {
            tags[m] = CD_FEAT_SWEPT | cdPieceFeat(k, polyIsA) | C.tag[m];
            nBA[m] = polyIsA ? -C.nrm[m] : C.nrm[m];      // C.nrm points out of the polytope, toward the swept body
        }
        cdUniqueTags(tags, C.n);
        cdEmitGroupF(contacts, contactCount, maxContacts, u, id, j, 0u, C.n, tags, nBA, C.p, xi, xj, C.depth);
        return;
    }
    if (kind == CD_PP_GJK) {
        CoinBody C = cdChildAsBoxBody(P, cdCompoundChild(P, k, hullVerts, hullRanges));
        CoinBody A = polyIsA ? C : S, B = polyIsA ? S : C;
        float3 n; float depth;
        if (!cdGJKEPA(A, B, hullVerts, hullRanges, n, depth) || !(depth > 0.0)) return;
        float3 cp = 0.5 * (cdSupport(A, -n, hullVerts, hullRanges) + cdSupport(B, n, hullVerts, hullRanges));
        cdEmitContactF(contacts, contactCount, maxContacts, id, j, CD_FEAT_GJK | cdPieceFeat(k, polyIsA), 0u,
                       n, cp, xi, xj, depth);
        return;
    }
}

// A listed pair that runs the separating-axis test (CD_PP_POLY / STATIC / DISC), set up: its
// (first, second) polytopes in a working frame O, its length tolerance, and what it emits as
// (the body pair or the static collider, the feature's piece bits). False for any other kind.
struct CdSATPair {
    CdPoly A1, A2;
    float3 O, xa, xb;
    float  tol;
    uint   id, b, collider, featHi;
    bool   oneWay;
};
__attribute__((noinline)) static bool cdSATPairSetup(uint p, uint n,
    device const CoinBody*           coins,
    device const CoinStaticCollider* colliders,
    device const float4*             hullVerts,
    device const uint2*              hullRanges,
    device const uint4*              pairs,
    thread CdSATPair&                Q)
{
    if (p >= n) return false;
    uint4 r = pairs[p];
    uint kind = r.w >> 28;
    if (!cdPolyPairIsSAT(kind)) return false;                // the one-thread half's
    uint pa = r.w & 0xFFu, pb = (r.w >> 8) & 0xFFu;
    bool polyIsA = (r.w & (1u << 16)) != 0u;
    uint id = r.x, j = r.y;
    CoinBody ci = coins[id];
    Q.id = id; Q.xa = ci.posInvMass.xyz; Q.b = j; Q.collider = 0u; Q.oneWay = false;
    if (kind == CD_PP_STATIC) {
        CoinStaticCollider col = colliders[r.z];
        uint ck = as_type<uint>(col.a.w);
        CdBox sb; sb.c = col.a.xyz; sb.he = col.b.xyz;
        sb.q = (ck == 3u) ? col.orient : float4(0.0, 0.0, 0.0, 1.0);
        Q.oneWay = (ck == 1u) && ((col.meta.x & 1u) != 0u);
        float4 sp = cdPieceSphere(ci, pa, hullVerts, hullRanges);
        Q.O = sp.xyz;
        Q.A1 = cdPieceView(ci, pa, Q.O, hullVerts, hullRanges); Q.A2 = cdPolyBox(sb, Q.O);
        Q.tol = 1e-5 * sp.w + 1e-9;
        Q.featHi = cdPieceFeat(pa, true); Q.b = CD_STATIC; Q.collider = r.z; Q.xb = Q.xa;
        return true;
    }
    CoinBody cj = coins[j];
    Q.xb = cj.posInvMass.xyz;
    if (kind == CD_PP_POLY) {
        float4 sa = cdPieceSphere(ci, pa, hullVerts, hullRanges), sb = cdPieceSphere(cj, pb, hullVerts, hullRanges);
        Q.O = 0.5 * (sa.xyz + sb.xyz);
        Q.A1 = cdPieceView(ci, pa, Q.O, hullVerts, hullRanges); Q.A2 = cdPieceView(cj, pb, Q.O, hullVerts, hullRanges);
        Q.tol = 1e-5 * min(sa.w, sb.w) + 1e-9;
        Q.featHi = cdPieceFeat(pa, true) | cdPieceFeat(pb, false);
        return true;
    }
    // CD_PP_DISC: a compound piece against a disc — SAT + clipping against the disc's
    // CD_DISC_N-gon prism (cdPolyDisc; caps exact, rim inscribed).
    CoinBody P = polyIsA ? ci : cj, S = polyIsA ? cj : ci;
    uint k = pa;
    float Rd = cdRadiusOf(S);
    float4 sp = cdPieceSphere(P, k, hullVerts, hullRanges);
    Q.O = 0.5 * (sp.xyz + S.posInvMass.xyz);
    CdPoly PP = cdPieceView(P, k, Q.O, hullVerts, hullRanges), PD = cdPolyDisc(S, Q.O);
    Q.A1 = polyIsA ? PP : PD; Q.A2 = polyIsA ? PD : PP;
    Q.tol = 1e-5 * min(sp.w, Rd) + 1e-9;
    Q.featHi = cdPieceFeat(k, polyIsA);
    return true;
}

// The SAT pair's first face, one thread (cdPolySATFirstFace): false when the pair is apart.
__attribute__((noinline)) static bool cdSATPairFirstFace(thread const CdSATPair& Q, constant CoinUniforms& u, device const float4* hullVerts,
                                      thread uint& f0, thread float& s0) {
    return cdPolySATFirstFace(Q.A1, Q.A2, max(u.speculativeMargin, 0.0) + Q.tol, hullVerts, f0, s0);
}

// The rest of a surviving SAT pair on ONE SIMD GROUP — every lane calls it with the same pair
// (`lane` / `w` as in cdPolySATSG): the SAT past the first face on all lanes, then the manifold
// and its emission on lane 0. `serial` (the CD_FLAG_POLY_SERIAL test seam): lane 0 runs the
// whole one-thread SAT (cdPolySAT) instead, f0 / s0 unused.
__attribute__((noinline)) static void cdSATPairSolveSG(thread const CdSATPair& Q, uint f0, float s0, bool serial, uint lane, uint w,
    constant CoinUniforms&           u,
    device CoinContact*              contacts,
    device atomic_uint*              contactCount,
    constant uint&                   maxContacts,
    device const float4*             hullVerts)
{
    float spec = max(u.speculativeMargin, 0.0);
    float absTol = 0.5 * max(u.contactSlop, 1e-6);
    bool apartFaceFirst = (u.solverFlags & CD_FLAG_MANIFOLD_SOLVE) != 0u;   // VZ-0201 (see cdPolyManifoldFrom)
    CdSAT S;
    CdManifold M;
    bool made;
    if (serial) {
        if (lane != 0u || !cdPolySAT(Q.A1, Q.A2, spec + Q.tol, hullVerts, S)) return;
        made = cdPolyManifoldFrom(Q.A1, Q.A2, S, spec, Q.tol, absTol, apartFaceFirst, hullVerts, M);
    } else {
        if (!cdPolySATSG(Q.A1, Q.A2, spec + Q.tol, hullVerts, lane, w, f0, s0, S)) return;   // uniform
        M.n = 0;
        M.nrm = float3(0.0, 1.0, 0.0);
        bool edge, refIsQ;
        cdPolyContactKind(S, absTol, apartFaceFirst, edge, refIsQ, M);                        // uniform
        if (edge) {
            if (lane != 0u) return;
            cdPolyEdgeContact(Q.A1, Q.A2, S, hullVerts, M);
            made = true;
        } else {
            uint fr = refIsQ ? S.faceQ : S.faceP;
            float4 rpl = cdPolyFace(refIsQ ? Q.A2 : Q.A1, fr, hullVerts);
            uint fi = cdPolyIncidentFaceSG(refIsQ ? Q.A1 : Q.A2, rpl.xyz, hullVerts, lane, w);
            if (lane != 0u) return;
            made = cdPolyFaceContact(Q.A1, Q.A2, refIsQ, fr, fi, spec, Q.tol, hullVerts, M);
        }
    }
    if (!made) return;
    if (Q.oneWay && M.nrm.y < 0.5) return;                  // top-only ledge
    cdEmitManifold(M, Q.O, Q.featHi, contacts, contactCount, maxContacts, u, Q.id, Q.b, Q.collider, Q.xa, Q.xb);
}

kernel void coinPolyNarrow(
    device const CoinBody*           coins        [[ buffer(0) ]],
    device const CoinStaticCollider* colliders    [[ buffer(1) ]],
    constant CoinUniforms&           u            [[ buffer(2) ]],
    device CoinContact*              contacts     [[ buffer(3) ]],
    device atomic_uint*              contactCount [[ buffer(4) ]],
    constant uint&                   maxContacts  [[ buffer(5) ]],
    device const float4*             hullVerts    [[ buffer(6) ]],
    device const uint2*              hullRanges   [[ buffer(7) ]],
    device const uint4*              pairs        [[ buffer(8) ]],
    device const atomic_uint&        pairCount    [[ buffer(9) ]],
    constant uint&                   maxPairs     [[ buffer(10) ]],
    uint p    [[ threadgroup_position_in_grid ]],     // one SIMD-group-wide threadgroup per listed pair
    uint lane [[ thread_index_in_simdgroup ]],
    uint w    [[ threads_per_simdgroup ]])
{
    uint n = min(atomic_load_explicit(&pairCount, memory_order_relaxed), maxPairs);
    if (lane == 0u) cdPolyNarrowThreadBody(p, n, coins, u, contacts, contactCount, maxContacts, hullVerts, hullRanges, pairs);
    CdSATPair Q;
    if (!cdSATPairSetup(p, n, coins, colliders, hullVerts, hullRanges, pairs, Q)) return;   // uniform: one pair per group
    bool serial = (u.solverFlags & CD_FLAG_POLY_SERIAL) != 0u;
    uint f0 = 0u; float s0 = 0.0; uint live = 1u;
    if (!serial) {
        if (lane == 0u) live = cdSATPairFirstFace(Q, u, hullVerts, f0, s0) ? 1u : 0u;
        live = simd_broadcast(live, 0); f0 = simd_broadcast(f0, 0); s0 = simd_broadcast(s0, 0);
    }
    if (live == 0u) return;
    cdSATPairSolveSG(Q, f0, s0, serial, lane, w, u, contacts, contactCount, maxContacts, hullVerts);
}
