// ── ILLUMINATORAMA — G-BUFFER VERTEX DEFORMATION + PASS CONSTANTS (shared) ──────────────────
//
// The G-buffer pass's function constants, the per-vertex deformation (`illumiDeform`: model
// matrix, tree wind, drag/impact sway, field sink) and the small helpers the material evaluation
// shares with it. Included through IlluminatoramaGBufferShade.h by both the conventional raster
// path (IlluminatoramaGBuffer.metal) and Signack's visibility buffer (IlluminatoramaVisBuffer.metal,
// Daydream DH-1461), so a vertex lands on the SAME pose in all of them.
#pragma once

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Animated UV DOMAIN WARP ───────────────────────────────────────────────────
//
// Offsets the sampling UV as a smooth function of the UV itself and of time.
// A SCROLL translates a map rigidly and reads as the whole pattern sliding; a
// DOMAIN warp moves the sample POINT instead, so the pattern breathes and
// wobbles in place while staying where it is. That is the difference between a
// window grid drifting sideways and one that slowly morphs.
//
// Two octaves, the second at an irrational-ish frequency and speed ratio to the
// first so they never phase-lock into an obvious repeating pulse. Each axis is
// driven by the OPPOSITE one (x by uv.y, y by uv.x), which makes it shear and
// swirl rather than merely breathe in and out.
//
// `w.x <= 0` returns the UV untouched — the default for every instance, so this
// is an exact no-op everywhere it is not asked for.
static inline float2 warpUV(float2 uv, float4 w, float t) {
    if (w.x <= 0.0f) return uv;
    const float f = w.y;
    const float s = w.z * t;
    float2 d = float2(sin(uv.y * f + s),
                      cos(uv.x * f * 1.13f - s * 0.87f));
    d += 0.5f * float2(sin(uv.y * f * 2.31f - s * 1.71f),
                       cos(uv.x * f * 1.87f + s * 1.29f));
    return uv + d * w.x;
}

// TEMP DIAGNOSTIC (forest-tex session): set to 1 to flat-colour Forest wood by
// species marker (oak=red birch=white maple=blue log=green cap=yellow). REVERT
// to 0 before counting any fix; this is a marker-contract probe only.
#define FOREST_SPECIES_DEBUG 0
// ── Hierarchical tree wind (#58 #1) ──────────────────────────────────────────
// Vertex-shader vegetation wind (GG3-style): each tree vertex carries
// (swayWeight, phase, flutter) in its tangent (packed by ForestGeometry). The
// trunk base has swayWeight 0 so it stays planted while the canopy sways
// (cantilever). Layered: macro height-weighted bend + a coarse TRAVELLING GUST
// envelope (wind moves across the stand in waves) + high-freq leaf flutter.
// Gated by windStrength > 0 → an exact no-op for every scene that doesn't set it;
// windHeading (frame._padPhase2B) is the dir. `windStrength` reaching here is the
// frame's global (repurposed frame._padPhase2A) TIMES the per-instance `windScale`,
// which is 0 unless the host declared the draw vegetation. That per-instance term is
// load-bearing: `windAttr` IS `v.tangent`, and a tangent is the TBN basis every
// normal-mapped surface carries, so on ordinary geometry `windAttr.x` is a surface
// direction, not a sway weight. Without the instance gate this function swings any
// mesh whose tangents happen to point +X.
static inline float3 applyTreeWind(float3 wp, float4 windAttr, float time,
                                   float windStrength, float windHeading) {
    float sway = windAttr.x;
    if (sway < 0.0001 || windStrength <= 0.0) return wp;
    float phase   = windAttr.y;
    float flutter = windAttr.z;
    float2 wdir = float2(cos(windHeading), sin(windHeading));
    float gust  = 0.55 + 0.45 * sin(time * 0.43 + dot(wp.xz, wdir) * 0.07);
    float macro = sin(time * 0.9 + phase);
    float bend  = sway * windStrength * (0.5 + gust) * macro;
    wp.xz += wdir * bend;
    wp.y  -= sway * windStrength * 0.18 * fabs(macro);   // slight droop as it bends
    // Leaf flutter: high-freq shimmer on foliage. Kept SMALL (sub-card amplitude)
    // and at LOW spatial frequency so neighbouring outer-rind cards flutter
    // COHERENTLY — at the prior 0.15 coefficient the per-vertex chaotic jitter
    // (~3 cm vs ~8.5 cm cards) slid adjacent rind cards apart and re-opened the
    // sealed crown silhouette (side/top "flake cloud" regression). The macro sway
    // above stays coherent (shared tree phase), so the crown leans as one mass.
    if (flutter > 0.0001) {
        wp.x += flutter * windStrength * 0.05 * sin(time * 6.5 + phase * 4.0 + wp.y * 1.1);
        wp.z += flutter * windStrength * 0.05 * sin(time * 5.7 + phase * 3.0 + wp.x * 1.1);
        wp.y += flutter * windStrength * 0.035 * sin(time * 6.0 + phase * 3.5);
    }
    return wp;
}

// Issue #65 — per-vertex motion vectors for animated GPU meshes (fluid /
// marching-cubes / MLS-MPM surfaces). When false (the default for every
// ordinary instanced draw) `illumi_vs` is BYTE-FOR-BYTE its old self: the
// previous-frame object-space position is just `v.position`, so velocity comes
// purely from the per-instance model matrix delta (rigid spin/translation).
// When true (the dedicated G-buffer GPU-mesh draw specializes this on), the
// previous object-space position is read from a side `prevPositions` buffer the
// host blits each frame — so a deforming surface whose model matrix is identity
// still writes a real screen-space velocity (curr vertex − prev vertex). Both
// cases compose: a rigid GPU mesh has prevPos == currPos so velocity falls back
// to the matrix delta, exactly as before.
constant bool kUsePrevVerts [[function_constant(10)]];
// DH-0140 — the PHOTO-LANE G-buffer variant writes a sixth target, `GBufferOut.material`.
// The live canvas compiles this false and its output struct is the five targets it always
// was; the still compiles it true. This is the carrier Danny asked for (2026-08-31): quality
// that "would cost every pixel of every frame" is built for the highest-quality preset and
// Photo Export only, and the interactive path pays nothing.
constant bool kExtendedGBuffer [[function_constant(11)]];
// Opt-in INTERIOR MAPPING (van Dongen 2008) for the mesh kinds a host lists in
// `IlluminatoramaRenderer.interiorMappedMeshKinds`: the pane's emission becomes the radiance of
// a lamp-lit ROOM BOX behind it, ray-traced analytically per fragment (one draw, no geometry).
// Only the dedicated pipeline variant compiles it true; every other draw is unchanged.
constant bool kInteriorMap [[function_constant(12)]];

/// Host mirror: `IlluminatoramaInteriorMapping.GPU`. Metres, in the PANE's frame: u along the
/// mesh tangent (the pane's width), v world-up, w into the room (−normal).
struct InteriorMapParams {
    float4 room;     // x width, y height, z depth (mean; ±depthJitter per window), w sill height (floor → pane bottom)
    float4 pane;     // x pane width, y pane height, z uv bias subtracted from uv.x, w depth jitter (fraction)
    float4 wall;     // rgb wall albedo (linear), w floor albedo scale
    float4 lamp;     // x lamp height above the floor, y lamp inset from the back wall, z ceiling albedo scale, w ambient (one-bounce) fraction
};

/// A fast per-window hash in [0, 1)³ from the window's world origin.
static inline float3 imapHash3(float3 p) {
    uint3 q = uint3(int3(floor(p * 4.0f)) + 100000);
    q = q * 1664525u + 1013904223u;
    q.x += q.y * q.z; q.y += q.z * q.x; q.z += q.x * q.y;
    q ^= q >> 16u;
    q.x += q.y * q.z; q.y += q.z * q.x; q.z += q.x * q.y;
    return float3(q & 0xFFFFu) / 65536.0f;
}

/// Radiance (per unit lamp emission) leaving the room box behind a pane, along the camera ray
/// through the pane point: the ray is intersected analytically with the box's floor, ceiling,
/// side and back walls (the nearest exit plane), and that Lambertian surface is lit by a point
/// lamp inside the room (exact inverse-square × cosine — an empty convex box has no
/// occluders) plus a uniform one-bounce term. Normalised so the back wall facing a lamp one
/// room-depth away reads ≈ 1: `inst.emission` then sets the room's luminance directly.
static inline float3 interiorMapRadiance(float3 P, float3 N, float4 T, float2 uv, float3 cam,
                                         InteriorMapParams m) {
    float3 up = float3(0.0f, 1.0f, 0.0f);
    float3 tu = normalize(T.xyz - N * dot(T.xyz, N));
    float3 win = -N;                                    // into the room
    float pu = clamp(uv.x - m.pane.z, 0.0f, 1.0f) * m.pane.x;   // metres from the pane's left edge
    float pv = clamp(uv.y, 0.0f, 1.0f) * m.pane.y;              // metres above the pane's bottom
    float3 origin = P - tu * pu - up * pv;              // the pane's bottom-left corner (world)
    float3 h = imapHash3(origin);
    float W = m.room.x, H = m.room.y;
    float D = m.room.z * (1.0f + m.pane.w * (2.0f * h.x - 1.0f));
    // Room frame: x across (0…W), y up from the floor (0…H), z depth (0 at the pane … D).
    float x0 = 0.5f * (W - m.pane.x) + (h.y - 0.5f) * 0.6f * (W - m.pane.x);   // pane offset along the wall
    float3 ro = float3(x0 + pu, m.room.w + pv, 0.0f);
    float3 dw = normalize(P - cam);
    float3 rd = float3(dot(dw, tu), dot(dw, up), dot(dw, win));
    if (rd.z <= 1e-4f) return float3(0.0f);
    // Exit distance through the box (the ray starts on the front face, inside the box).
    float tx = rd.x > 0.0f ? (W - ro.x) / rd.x : (rd.x < 0.0f ? -ro.x / rd.x : 1e9f);
    float ty = rd.y > 0.0f ? (H - ro.y) / rd.y : (rd.y < 0.0f ? -ro.y / rd.y : 1e9f);
    float tz = (D - ro.z) / rd.z;
    float t = max(min(tx, min(ty, tz)), 0.0f);
    float3 hit = ro + rd * t;
    float3 n; float3 alb = m.wall.rgb;
    if (t == tz)      { n = float3(0.0f, 0.0f, -1.0f); }
    else if (t == ty) { n = rd.y > 0.0f ? float3(0.0f, -1.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
                        alb *= rd.y > 0.0f ? m.lamp.z : m.wall.w; }
    else              { n = rd.x > 0.0f ? float3(-1.0f, 0.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f); }
    // The lamp: a point at a hashed spot near the back of the room.
    float3 lp = float3(W * (0.2f + 0.6f * h.z), m.lamp.x, max(D - m.lamp.y, 0.3f * D));
    float3 l = lp - hit;
    float r2 = max(dot(l, l), 0.04f);
    float cosI = max(dot(n, l * rsqrt(r2)), 0.0f);
    float direct = cosI / r2;
    // Normalisation: the back wall straight across from the lamp, at the lamp's inset.
    float ref = 1.0f / max(m.lamp.y * m.lamp.y, 0.04f);
    return alb / max(dot(m.wall.rgb, float3(0.2126f, 0.7152f, 0.0722f)), 1e-3f)
         * (direct / ref + m.lamp.w);
}

// ── Drag/impact sway (generic rigid secondary motion) ────────────────────────
// The non-foliage sibling of applyTreeWind: a placed object the host is dragging
// (or that just knocked into something) leans + hops, driven entirely by the
// per-instance swayMode/swayLean/swayJostle the host's DragSwayTracker fills.
//
// Applied in WORLD space about the instance's bottom-pivot so it composes cleanly
// with the per-instance non-uniform scale already baked into modelMatrix (rotating
// unit-box object space then scaling would shear). Pivot = base centre of the
// box; lean axis = the object's local +Z in world (modelMatrix column 2). swayMode
// 0 ⇒ identity, so every non-swaying instance is an exact no-op.
//
// swayMode reference:
//   0 · none                — hard no-op (all vegetation instances that don't opt in).
//   1 · bottom-pivot lean   — rigid rotation about the box BASE (object y=-0.5) by the
//                             host-supplied static `lean` angle (books, upright shelf
//                             contents, dragged/knocked props). `lean` IS the angle.
//   2 · top-pivot pendulum  — rigid rotation about the model ORIGIN (object y=0), for a
//                             HANGING object (ceiling pendant). The pendant mesh is
//                             authored with its ceiling anchor at object y=0 and its body
//                             hanging DOWN into −Y, and it's placed by an unscaled matrix,
//                             so object y=0 is exactly the ceiling attach point — the
//                             pivot. Self-oscillates in the shader: angle = `lean` *
//                             sin(time·ω + phase), so `lean` here is the AMPLITUDE (max
//                             swing, radians) and the host sets it ONCE (static
//                             per-instance) — no per-frame drive. Phase is derived from
//                             the pivot's world XZ so neighbouring pendants swing out of
//                             step. Displacement is 0 at the top anchor and grows toward
//                             the hanging bottom (pivot about the top) — the inverse of
//                             mode 1 (which pivots at the box base y=-0.5).
//   3 · wind hinge flap     — rigid rotation about the model ORIGIN about local +Z (the hinge),
//                             like mode 2, but ONE-SIDED and driven by the frame's WIND, like a
//                             sheet: angle = `lean` × wind × catch × sway(time, phase), sway ∈
//                             [0, 1] a steady lift carrying a gusting sway and a flutter, so the
//                             flap (a toilet seat or lid) rides up with the wind and never swings
//                             through what it rests on. `catch` is how squarely the wind meets it:
//                             a flap lifting toward its back (lean > 0 — a closed lid) catches
//                             wind blowing INTO its free edge; one swinging forward (lean < 0 —
//                             an open seat) catches wind from behind; any facing keeps some
//                             turbulent lift. `lean` = the flap at full wind (radians), signed.
//                             Phase comes from the pivot, so parts on one hinge move together (a
//                             seat whose lean is a fraction of its lid's stays under it).
//
// Returns the rotated world position; `worldN`/`worldT` are rotated in place by the
// same rigid rotation so lighting/normal-mapping track the lean.
static inline float3 applySway(float3 wp, float4x4 model, int mode,
                               float lean, float jostle, float time, float wind, float windHeading,
                               thread float3& worldN, thread float3& worldT) {
    if (mode == 0) return wp;
    // Mode 1 pivots at the box base (y=-0.5) and applies `lean` directly. Mode 2 pivots
    // at the model origin (y=0) — the hanging pendant's ceiling anchor — and
    // self-oscillates `lean` (as amplitude) from `time`.
    float  pivotY = (mode >= 2) ? 0.0 : -0.5;
    float3 pivot  = (model * float4(0.0, pivotY, 0.0, 1.0)).xyz;
    float3 axis   = normalize((model * float4(0.0, 0.0, 1.0, 0.0)).xyz);  // local +Z in world
    float  angle  = lean;
    if (mode == 2) {
        // Gentle pendulum ~0.42 Hz (ω ≈ 2.65 rad/s); per-instance phase from the top
        // anchor's world XZ so a row of pendants doesn't swing in lock-step.
        float phase = pivot.x * 1.7 + pivot.z * 2.3;
        angle = lean * sin(time * 2.65 + phase);
    } else if (mode == 3) {
        float phase = pivot.x * 1.7 + pivot.z * 2.3;
        // The flap's free edge points along cross(up, hinge axis) — independent of its rest pose.
        float3 fwd = cross(float3(0.0f, 1.0f, 0.0f), axis);
        float2 f = length(fwd.xz) > 1e-4f ? normalize(fwd.xz) : float2(0.0f, -1.0f);
        float2 wdir = float2(cos(windHeading), sin(windHeading));           // where the wind blows TO
        float catchWind = 0.35f + 0.65f * saturate(dot(wdir, f) * (lean > 0.0f ? -1.0f : 1.0f));
        // A sheet in the wind: a steady lift (~0.2 Hz gust swell) with a faster flutter on top.
        float sway = 0.55f + 0.40f * sin(time * 1.25f + phase) + 0.15f * sin(time * 4.6f + phase * 1.9f);
        angle = lean * max(wind, 0.0f) * catchWind * saturate(sway);
    }
    float  c = cos(angle), s = sin(angle);
    // Rodrigues rotation of (wp - pivot) about `axis`, then restore pivot.
    float3 r = wp - pivot;
    float3 rot = r * c + cross(axis, r) * s + axis * dot(axis, r) * (1.0 - c);
    worldN = worldN * c + cross(axis, worldN) * s + axis * dot(axis, worldN) * (1.0 - c);
    worldT = worldT * c + cross(axis, worldT) * s + axis * dot(axis, worldT) * (1.0 - c);
    return pivot + rot + float3(0.0, jostle, 0.0);
}

// ── Field sink ──────────────────────────────────────────────────────────────
// A host's whole field of objects slurps into the ground and slides back out: each instance with
// a sink depth (Instance.tagGlow.y, metres) drops by depth × p², p the frame's progress
// (FrameUniforms.fieldSink), staggered per instance by a hash of its origin so the field goes
// down in a ripple rather than as one plate. Squared: it ACCELERATES as it goes down and
// DECELERATES as it comes up. Below the ground it is simply hidden by the (opaque) terrain.
// depth 0 or progress 0 ⇒ exactly 0 — every instance and host that never opts in is untouched.
// `key` (Instance.tagGlow.z) > 0 is an explicit stagger key, so the parts of ONE object placed by
// different matrices (a toilet's body and its hinged lid) go down together; 0 = hash the origin.
static inline float fieldSinkOffset(float depth, float progress, float key, float4x4 model) {
    if (depth <= 0.0f || progress <= 0.0f) return 0.0f;
    float2 o = model[3].xz;
    float h = key > 0.0f ? fract(key) : fract(sin(dot(o, float2(12.9898f, 78.233f))) * 43758.5453f);
    const float stagger = 0.35f;
    float p = saturate(progress * (1.0f + stagger) - stagger * h);
    return depth * p * p;
}

// ── The vertex deformation, ONCE ─────────────────────────────────────────────────────────────
// Everything `illumi_vs` does to a vertex before the camera matrices: model matrix, tree wind
// (this frame at `frame.time`, the previous pose at `frame.time − windPrevDelta`), drag/impact
// sway, field sink. Factored out (Signack S3, Daydream DH-1461) because the VISIBILITY-BUFFER
// path re-derives the same three vertices of a triangle in its raster pass AND again, per pixel,
// in its resolve — and both must land on the pose `illumi_vs` would have produced, bit for bit.
// One function is the only way the three cannot drift. Arithmetic order is exactly the old
// inline code's, so the conventional path is byte-identical.
struct IllumiDeformed {
    float4 worldP;
    float4 prevWorldP;
    float3 worldN;
    float3 worldT;
};

static inline IllumiDeformed illumiDeform(Vertex v, float3 prevObjP, Instance inst, Instance prevInst,
                                          constant FrameUniforms& frame) {
    IllumiDeformed d;
    float4 worldP = inst.modelMatrix * float4(v.position, 1.0);
    // Tree wind displacement — see the long note in `illumi_vs`.
    float windStrength = frame._padPhase2A * inst.windScale;
    float prevWindTime = frame.time - frame.windPrevDelta;
    worldP.xyz = applyTreeWind(worldP.xyz, v.tangent, frame.time,
                               windStrength, frame._padPhase2B);
    float3 worldN = (inst.normalMatrix * float4(v.normal, 0.0)).xyz;
    float4 prevWorldP = prevInst.modelMatrix * float4(prevObjP, 1.0);
    prevWorldP.xyz = applyTreeWind(prevWorldP.xyz, v.tangent, prevWindTime,
                                   windStrength, frame._padPhase2B);
    float3 worldT = (inst.modelMatrix * float4(v.tangent.xyz, 0.0)).xyz;
    worldP.xyz = applySway(worldP.xyz, inst.modelMatrix, inst.swayMode,
                           inst.swayLean, inst.swayJostle, frame.time, frame._padPhase2A, frame._padPhase2B, worldN, worldT);
    float3 prevN = worldN, prevT = worldT;   // throwaway: prev normal/tangent unused
    prevWorldP.xyz = applySway(prevWorldP.xyz, prevInst.modelMatrix, prevInst.swayMode,
                               prevInst.swayLean, prevInst.swayJostle, prevWindTime, frame._padPhase2A, frame._padPhase2B, prevN, prevT);
    worldP.y -= fieldSinkOffset(inst.tagGlow.y, frame.fieldSink, inst.tagGlow.z, inst.modelMatrix);
    prevWorldP.y -= fieldSinkOffset(prevInst.tagGlow.y, frame.prevFieldSink, prevInst.tagGlow.z, prevInst.modelMatrix);
    d.worldP = worldP; d.prevWorldP = prevWorldP; d.worldN = worldN; d.worldT = worldT;
    return d;
}

