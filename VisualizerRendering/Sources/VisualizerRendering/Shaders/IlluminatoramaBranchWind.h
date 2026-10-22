// ── ILLUMINATORAMA — PER-BRANCH WIND (shared) ─────────────────────────────────────────────────
//
// Signack S5 (Daydream DH-1461): the limb layer of the tree wind, Pivot-Painter style. The trunk
// sway (`applyTreeWind`: one height² weight + one phase per tree) moves a crown as ONE mass; real
// limbs also swing on their own, each at its own phase and a little faster than the trunk, with
// the swing growing from nothing at the limb's base to its fullest at its tips, and the leaves
// fluttering on top. The bake (VisualizerVegetation `ForestTreeGeometry.Soup.bakeBranchWind`)
// writes each tree vertex's limb record into its UV, which tree meshes do not otherwise use:
//
//     uv = (kBranchWindMarker + b, φ)     b ∈ [0, 1] — the vertex's weight along its limb
//                                          (0 at the limb's base ⇒ the joint never tears)
//                                          φ ∈ [0, 2π) — the limb's phase (one per limb)
//
// Every vertex of a limb's wood, its sub-branches and the leaf cards hanging from it carries the
// same φ, and a leaf card carries ONE b (its attachment point), so a card moves rigidly with its
// branch. The motion is a pure function of the vertex's own attributes and the clock, so two
// vertices that share a position (a weld, a cluster boundary at any Signack level) move together:
// no LOD cut can crack under wind. A UV outside the marker band (every other mesh) is untouched.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// The UV band that marks a limb record: x ∈ [marker, marker + 1]. Far outside any material UV a
/// tree mesh could carry (tree meshes ship uv = 0; the soil marker is negative; venation is 10–11).
constant float kBranchWindMarker = 20.0f;
/// Peak limb displacement per unit wind strength (metres per unit of `windStrength`, at b = 1),
/// horizontal along the wind + a smaller cross-wind and vertical share — see `windReach`
/// (Swift, IlluminatoramaMesh+IndirectDraw.swift) for the bound: ≤ kBranchWindAmp·(1 + 0.45 + 0.25).
constant float kBranchWindAmp = 0.55f;

/// The limb layer's displacement of a vertex at world position `wp` (already trunk-swayed).
static inline float3 applyBranchWind(float3 wp, float2 uv, float time, float windStrength,
                                     float windHeading, float gust) {
    float bx = uv.x - kBranchWindMarker;
    if (bx < 0.0f || bx > 1.001f || windStrength <= 0.0f) return wp;
    float b = min(bx, 1.0f);
    float phi = uv.y;
    // Each limb its own rate: 1.7–2.6 rad/s (0.27–0.41 Hz), faster than the trunk's 0.9.
    float omega = 1.7f + 0.9f * fract(phi * 0.6180339f);
    float s1 = sin(time * omega + phi);
    float s2 = sin(time * omega * 1.37f + phi * 2.1f);
    float2 wdir = float2(cos(windHeading), sin(windHeading));
    float2 perp = float2(-wdir.y, wdir.x);
    float a = windStrength * kBranchWindAmp * b;
    wp.xz += a * (wdir * ((0.6f + 0.4f * gust) * s1) + perp * (0.45f * s2));
    wp.y  += a * 0.25f * s2;
    return wp;
}
