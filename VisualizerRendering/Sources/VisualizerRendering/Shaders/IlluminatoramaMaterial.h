#pragma once

// ── ILLUMINATORAMA — MATERIAL ATLAS SAMPLING AND SURFACE NOISE ──────────────
//
// The texture-atlas samplers (aspect-corrected + hex-lattice anti-tiling) and
// the soil/bark procedural noise the G-buffer pass shades terrain with.

#include <metal_stdlib>
using namespace metal;

// ── Procedural soil material helpers (#58 dirt items #11/#12/#13) ─────────────
// Cheap value-noise + a finite-difference gradient for world-space macro/micro
// normal detail on the ground, so soil gets roughness/normal variation without a
// baked texture or extra geometry. Used only by the gated soil branch below.
static inline float soilHash(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
static inline float soilValueNoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = soilHash(i), b = soilHash(i + float2(1, 0));
    float c = soilHash(i + float2(0, 1)), d = soilHash(i + float2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// Cellular (Worley F1) field: returns x = distance to the nearest jittered cell
// centre (small = near a crack between cells), y = a per-cell ID in [0,1] (a stable
// random value shared by every pixel inside one cell). Used by the OAK bark branch
// to carve the trunk into DISCRETE shedding PLATES — the cell interior takes a
// coherent ridge tint, the inter-cell distance drives the dark furrow crack. Two
// hashes off the same integer cell give an independent centre offset + ID.
static inline float2 barkCellF1(float2 p) {
    float2 ip = floor(p), fp = fract(p);
    float bestD = 8.0;
    float bestID = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float2 g = float2(float(i), float(j));
            float2 cell = ip + g;
            float2 jitter = float2(soilHash(cell), soilHash(cell + 37.7));
            float2 diff = g + jitter - fp;
            float d = dot(diff, diff);
            if (d < bestD) { bestD = d; bestID = soilHash(cell + 91.3); }
        }
    }
    return float2(sqrt(bestD), bestID);
}
// 2D height gradient summed over a macro (~1.4 m) and a micro (~0.4 m) octave.
static inline float2 soilNormalGrad(float2 p) {
    const float e = 0.15;
    float2 g = float2(0.0);
    float2 pm = p * 0.7;
    g.x += soilValueNoise(pm + float2(e, 0)) - soilValueNoise(pm - float2(e, 0));
    g.y += soilValueNoise(pm + float2(0, e)) - soilValueNoise(pm - float2(0, e));
    float2 pf = p * 2.6;
    g.x += 0.5 * (soilValueNoise(pf + float2(e, 0)) - soilValueNoise(pf - float2(e, 0)));
    g.y += 0.5 * (soilValueNoise(pf + float2(0, e)) - soilValueNoise(pf - float2(0, e)));
    return g / (2.0 * e);
}

// Phase 4.25 (issue #60 item 5) — aspect-correct atlas sample. `uvScale[slice]`
// is the fraction of the square slice that the *letterboxed* source image fills
// (see IlluminatoramaTextureAtlas): (1,1) for a square texture, which takes the
// hardware-`repeat` fast path — bit-identical to the pre-aspect code, no seam.
// For a non-square (letterboxed) slice we tile MANUALLY with fract(uv)*scale and
// inset the lookup by a half-texel so the bilinear footprint never bleeds into
// the empty letterbox band. This manual wrap reintroduces a thin filter seam at
// tile boundaries on aspect-corrected textures only — the declared cost of
// keeping one uniform-size square slice array.
static inline float4 sampleAtlasAspect(texture2d_array<float, access::sample> atlas,
                                       sampler s, float2 uv, uint slice,
                                       const device float2* uvScale,
                                       float2 duvdx, float2 duvdy) {
    float2 sc = uvScale[slice];
    if (sc.x >= 0.999f && sc.y >= 0.999f) {
        // Square → hardware repeat. S1.1: EXPLICIT gradients, taken from the caller's
        // ORIGINAL uv. Now that the atlas has a mip chain, letting the texture unit
        // infer the LOD would differ two per-cell hex-hash offsets across a quad at a
        // lattice boundary (see sampleAtlasHex) and select the 1x1 mip — a triangular
        // grid of mean-coloured lines over every surface.
        return atlas.sample(s, uv, slice, gradient2d(duvdx, duvdy));
    }
    // Per-axis: the axis that fills the slice (sc == 1) keeps the raw uv so the
    // hardware `repeat` sampler wraps it seamlessly; only the LETTERBOXED axis
    // (sc < 1) is tiled manually with fract()*sc, half-texel-inset so the
    // bilinear footprint never bleeds into the empty band. So a 4:1 wall tiles
    // seamlessly along its long axis and carries the manual-tiling seam only on
    // the short (banded) axis.
    float2 texSize = float2(atlas.get_width(), atlas.get_height());
    float2 halfTexel = 0.5f / texSize;
    float2 manual = clamp(fract(uv) * sc, halfTexel, sc - halfTexel);
    float2 st;
    st.x = (sc.x >= 0.999f) ? uv.x : manual.x;
    st.y = (sc.y >= 0.999f) ? uv.y : manual.y;
    // S1.1: st = fract(uv)*sc, so d(st) = sc*d(uv) everywhere EXCEPT the fract() step,
    // where the implicit derivative would be +/-sc — the coarsest mip along a grid line
    // at every tile boundary. Rebuild it from the PRE-fract uv.
    float2 axisScale = float2((sc.x >= 0.999f) ? 1.0f : sc.x,
                              (sc.y >= 0.999f) ? 1.0f : sc.y);
    return atlas.sample(s, st, slice,
                        gradient2d(duvdx * axisScale, duvdy * axisScale));
}

// Phase 7 — hex-stochastic atlas sample. Breaks repeating tiling on large planar
// surfaces (floors, walls) by blending three stochastically-offset samples whose
// cell borders align on a triangular lattice. GPU port of DaydreamCore's
// MaterialChannels.sampleAlbedoHex / hexHash (CPU reference in MaterialChannels.swift).
//
// `uv` — the same UV that sampleAtlasAspect would use (world-metres ÷ tile-metres,
//        so one UV unit = one texture tile). The hex cells and offset vectors are
//        expressed in that same space — the offset shifts into a different region of
//        the infinitely-tiling texture, so it's aspect-safe and letterbox-safe.
static inline float2 hexHash2D(float2 p) {
    // Integer bit-mixing hash — no sin/cos. Same constants as the CPU reference.
    // p is in skewed-lattice coordinates (integer-valued at cell vertices).
    //
    // The multiply happens in INTEGERS, after the coordinate is rounded — not in float
    // with an int() cast on the product. That cast was an overflow: at 73856093 per
    // lattice unit, `int()` leaves signed 32-bit range at |p| ≈ 26 cells. Daydream's
    // ground is ONE quad spanning ±750 m sampled at 2 m per tile, so the lattice runs to
    // ±375 — past the break about 51 m from the origin, across nearly the whole yard.
    // Out-of-range float→int is undefined in Metal; where it saturates, every cell beyond
    // that hashes identically, the three-tap blend collapses to one sample, and the
    // anti-tiling it exists to provide is silently off. The tell was that it only ever
    // looked right in tests, which all frame a room near the origin.
    //
    // Unsigned throughout: wrapping multiply is defined behaviour for uint (it is not for
    // int), and `>>` is a logical shift, so the mix cannot be perturbed by sign extension.
    int cx = int(floor(p.x + 0.5));
    int cy = int(floor(p.y + 0.5));
    uint ix = uint(cx) * 73856093u ^ uint(cy) * 19349663u;
    uint iy = uint(cx) * 83492791u ^ uint(cy) * 23994923u;
    ix ^= ix >> 11; ix *= 0x45d9f3bu; ix ^= ix >> 16;
    iy ^= iy >> 11; iy *= 0x45d9f3bu; iy ^= iy >> 16;
    return float2(float(ix & 0xFFu) / 128.0f - 1.0f,
                  float(iy & 0xFFu) / 128.0f - 1.0f);
}

// S2.5 half 2 — one scalar hash in [-1, 1) for the per-PATTERN-CELL value jitter
// (consumed in IlluminatoramaGBuffer.metal; the field contract lives on
// `Instance.patternCells` in IlluminatoramaCommon.h). Same integer bit-mixing family and
// the same overflow discipline as `hexHash2D` above: round in float, convert to int ONCE,
// then mix in uint where wrapping multiply is defined behaviour. The input is already
// integer-valued (`floor(uv * patternCells)`); the +0.5 round only guards FP noise.
static inline float patternCellHash(float2 cell) {
    int cx = int(floor(cell.x + 0.5f));
    int cy = int(floor(cell.y + 0.5f));
    uint h = uint(cx) * 73856093u ^ uint(cy) * 19349663u;
    h ^= h >> 11; h *= 0x45d9f3bu; h ^= h >> 16;
    return float(h & 0xFFFFu) / 32768.0f - 1.0f;
}


// ── Wood knots, in WORLD SPACE (not baked into the tile) ─────────────────────────────────
//
// **Why this cannot be a texture.** A knot is a sparse LANDMARK, and a landmark baked into a
// tiling texture reappears on a lattice at the tile period, forever. Daydream Home's wood tile is
// ~1 m (the resolution its grain needs), so baked knots drew a 1 m grid across every floor —
// "it looks like a pattern rather than organic". No amount of scatter inside the tile fixes that,
// because the scatter is what repeats.
//
// `uv` keeps counting up across the whole surface — only the atlas LOOKUP wraps — so a field
// evaluated on it IS world-space and never repeats. That is the same property `patternCells`
// relies on, used for geometry instead of tone.
//
// **And why it cannot be a decal quad.** The thing that separates a knot from a dark sticker is
// that the trunk's grain FLOWS AROUND it. A decal composited on top cannot bend what is under it.
// Here the knot returns a UV `warp` that is applied to the material samples themselves — pushed
// radially by `r²/d`, the streamline displacement for flow past a cylinder — so the grain, the
// pores and the ring figure all sweep around the knot exactly as they do in real wood.
//
// Everything is in UV units, which are isotropic in world space (uv = worldPos / uvMetres on both
// axes), so no aspect correction is needed and a radius means the same thing on both axes.
struct WoodKnotSample {
    float2 warp;    // UV displacement — the grain flowing around the knot
    float  eye;     // 1 on the branch's cut face, 0 off it
    float  rings;   // the branch's own concentric end grain, [0,1]
    float  rim;     // the bark line where the branch met the face
    float  crack;   // radial drying checks
    float  pith;    // 1 at the centre, falling to 0 at the rim
    float2 radial;  // outward unit direction, for the core's normal dip
};

static inline float woodKnotHash(float2 cell, uint salt) {
    int cx = int(floor(cell.x + 0.5f));
    int cy = int(floor(cell.y + 0.5f));
    uint h = uint(cx) * 73856093u ^ uint(cy) * 19349663u ^ salt * 83492791u;
    h ^= h >> 11; h *= 0x45d9f3bu; h ^= h >> 16;
    return float(h & 0xFFFFu) / 65536.0f;
}

// `knot` packs the material's knot parameters:
//   x = lattice cells per UV unit (0 disables the whole feature — the default, and an exact
//       no-op for every scene and every material that never opts in)
//   y = knot radius in UV units
//   z = darkness of the core against the wood it grew through, [0,1]
//   w = fraction of lattice cells that carry a knot
//
// **`patternCells` is what makes these PLANKS rather than a sheet of wood, and that matters for
// more than tidiness.** It carries the board comb the material already ships for its tone
// de-repeat, and this function uses it to enforce the one thing a saw does:
//
//   * A knot belongs to ONE board. Its influence is clipped to the plank its centre is in,
//     because the cut ended the grain — a knot cannot reach into the next board any more than a
//     branch can grow across a saw kerf.
//   * The cut itself is a FIXED POINT of the warp. The displacement is scaled to zero at both
//     seams, so no knot can ever move a board edge.
//
// Both were missing in the first version, and both are the same defect wearing different
// clothes: the warp was applied to the whole texture, and the texture contains the saw cuts.
// Measured on the shipped oak — a large knot reaches 176 mm against a 127 mm board (1.39x the
// board width, so it MUST spill across), and displaced the seam itself by 28.7 mm, a quarter of
// a board. Danny, 2026-08-19: *"it looks like the knots are pulling the straight-shape of the
// boards, this is not realistic"* — and then the framing that names the fix: *"couldn't you
// actually build real planks, then CUT those planks to the correct size?"* This is that, in the
// only form a per-pixel shader can express it: the grain is whatever the log had, the cut is
// straight by definition.
static inline WoodKnotSample sampleWoodKnots(float2 uv, float4 knot, float2 patternCells) {
    WoodKnotSample k;
    k.warp = float2(0.0f);
    k.eye = 0.0f; k.rings = 0.0f; k.rim = 0.0f; k.crack = 0.0f; k.pith = 0.0f;
    k.radial = float2(0.0f);
    if (knot.x <= 0.0f || knot.y <= 0.0f || knot.w <= 0.0f) { return k; }

    float cells = knot.x;
    float2 base = floor(uv * cells);
    float nearest = 1e9f;

    // The plank this pixel is on, and how far into it we are. `bw <= 0` means the material is a
    // continuous panel with no cuts at all, so nothing below clips or pins.
    float bw = patternCells.x > 0.0f ? 1.0f / patternCells.x : 0.0f;
    float pixelBoard = bw > 0.0f ? floor(uv.x / bw) : 0.0f;
    // Zero AT each cut, one across the middle of the board.
    float seamPin = 1.0f;
    if (bw > 0.0f) {
        float f = uv.x / bw - pixelBoard;
        seamPin = smoothstep(0.0f, 0.16f, f) * smoothstep(0.0f, 0.16f, 1.0f - f);
    }

    // 3x3, because a knot's influence reaches a few radii and its site is jittered inside its
    // own cell. The cell size is chosen (CPU side) so that is enough.
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float2 cell = base + float2(float(i), float(j));
            if (woodKnotHash(cell, 1u) > knot.w) { continue; }

            // Size varies a LOT, and from a squared draw: mostly small knots with the
            // occasional big one. A field of same-sized discs reads as a stamp.
            //
            // The FLOOR matters as much as the spread. At 0.45x the smallest draws were a few
            // pixels across at room distance and the floor read as scattered pepper — the same
            // "grime, not tooth" failure the glaze occlusion hit. A knot has to be big enough to
            // be recognisable as one, or it is dirt.
            float sz = woodKnotHash(cell, 5u);
            float rad = knot.y * (0.70f + 1.00f * sz * sz);

            float2 centre = (cell + float2(woodKnotHash(cell, 2u), woodKnotHash(cell, 3u))) / cells;
            // Keep it off the plank seams. `patternCells.x` boards span one UV unit.
            if (patternCells.x > 0.0f) {
                float bw = 1.0f / patternCells.x;
                float board = floor(centre.x / bw);
                float inset = min(0.42f, rad * 1.8f / bw);
                centre.x = clamp(centre.x, (board + inset) * bw, (board + 1.0f - inset) * bw);
            }

            // A knot is a feature of ONE plank: the cut ended it. Without this a big knot's
            // influence (~4.5 radii) is wider than the board it sits on and bleeds into the
            // neighbour, which reads as a single warped sheet rather than as laid boards.
            if (bw > 0.0f && floor(centre.x / bw) != pixelBoard) { continue; }

            float2 d2 = uv - centre;
            d2.y /= 1.35f;                       // oval along the board, as a branch is
            float d = max(length(d2), 1e-6f);
            if (d > rad * 5.0f) { continue; }

            float theta = atan2(d2.y, d2.x);
            float ph = woodKnotHash(cell, 7u) * 6.28318531f;
            // A SUBTLY irregular outline: a perfect circle is a drilled hole, a lumpy one is a
            // blob. Whole harmonics of theta, so it closes on itself.
            float lobe = sin(theta * 2.0f + ph) * 0.6f + sin(theta * 3.0f + ph * 2.1f) * 0.4f;
            float r = rad * (1.0f + 0.10f * lobe);

            // The grain flows around it, and is straight again a few radii out.
            //
            // ACROSS the board only. Displacing ALONG the grain slides the lines along
            // themselves — invisible — while visibly bowing the one family of features that runs
            // the other way: the butt joints. There is nothing to gain and a defect to lose.
            //
            // Scaled by `seamPin`, so the displacement is zero at both cuts however close the
            // knot is to one. 0.55 because a full radius of deflection at the rim is more than
            // a branch actually pushes its trunk's rings aside.
            float push = (r * r / max(d, r)) * (1.0f - smoothstep(r * 1.8f, r * 4.5f, d));
            k.warp.x += (d2.x / d) * push * 0.55f * seamPin;

            k.eye  = max(k.eye,  1.0f - smoothstep(r * 0.94f, r * 1.10f, d));
            k.rim  = max(k.rim,  (1.0f - smoothstep(0.0f, 0.16f, abs(d / r - 1.0f)))
                               * (1.0f - smoothstep(r, r * 1.35f, d)));
            k.pith = max(k.pith, 1.0f - smoothstep(0.0f, r * 0.95f, d));

            // Two hairline drying checks, unequal, from near the pith to just past the rim.
            // Four fat wedges read as a peace sign.
            float spikes = pow(abs(sin(theta + ph * 1.7f)), 120.0f)
                         * (0.30f + 0.70f * (0.5f + 0.5f * sin(theta + ph)));
            k.crack = max(k.crack, spikes * smoothstep(0.0f, r * 0.15f, d)
                                          * (1.0f - smoothstep(r * 0.95f, r * 1.35f, d)));

            // The NEAREST knot owns the end grain — two overlapping eyes would interleave
            // their rings into a moire. TWO rings across the face, not five: a 34 mm knot is
            // ~17 texels at the shipped bake and five of them cannot be resolved.
            if (d < nearest) {
                nearest = d;
                k.rings = pow(saturate(0.5f - 0.5f * cos(d / (r * 0.42f) * 6.28318531f)), 1.4f);
                k.radial = d2 / d;
            }
        }
    }
    return k;
}

// ── Carpet pile-lay tone bands (DH-0472) ──────────────────────────────────────────────────────
//
// The metre-scale value variation a pile carpet shows because the fibres LIE one way and shift
// value where the nap turns — the broad soft "shading" true of a brand-new rug, not wear. It cannot
// be baked: the carpet tile is ~0.30 m, so any low-frequency term authored inside it repeats every
// 0.30 m and then dissolves under the hex de-repeat, and the rug reads as one flat tone at room
// distance. So it is drawn here, per pixel, on the UNWRAPPED uv — which counts up across the whole
// rug and never repeats at the tile period. Exactly the wood-knot mechanism, one scale up.
//
// `macro` packs (see HouseRenderBridge, the one place metres become UV):
//   x = band lattice cells per UV unit (0 disables the whole feature — the default, and an exact
//       no-op for every scene and every material that never opts in)
//   y = ± tone amplitude
//
// Returns an ACHROMATIC multiplier around 1.0 — a value shift only, never a hue shift (a hue wander
// would read as dirt, not nap). Achromatic is also why the caller applies it to the sampled albedo
// rather than compositing a colour, the same modelling choice `sampleWoodKnots` makes.
static inline float carpetMacroHash(int cx, int cy) {
    uint h = uint(cx) * 73856093u ^ uint(cy) * 19349663u ^ 0x9E3779B9u;
    h ^= h >> 11; h *= 0x45d9f3bu; h ^= h >> 16;
    return float(h & 0xFFFFu) / 65536.0f;                 // [0,1]
}
static inline float carpetMacroNoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0f - 2.0f * f);                 // smoothstep interpolation
    int cx = int(i.x), cy = int(i.y);
    float a = carpetMacroHash(cx,     cy);
    float b = carpetMacroHash(cx + 1, cy);
    float c = carpetMacroHash(cx,     cy + 1);
    float d = carpetMacroHash(cx + 1, cy + 1);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);      // [0,1]
}
static inline float sampleCarpetMacro(float2 uv, float2 macro) {
    if (macro.x <= 0.0f || macro.y <= 0.0f) { return 1.0f; }
    // Elongate the field along the lay direction so the tone varies as broad STREAKS (the pile
    // laid one way), not isotropic blobs. Two octaves so the banding is organic, not a sine grid.
    float2 p = uv * macro.x;
    p.x *= 0.35f;
    float n = carpetMacroNoise(p) * 0.65f + carpetMacroNoise(p * 2.3f + 7.1f) * 0.35f;
    float sgn = n * 2.0f - 1.0f;                          // [-1,1]
    return 1.0f + sgn * macro.y;
}

// `strength` gates the whole effect. When strength <= 0 (the DEFAULT for every
// scene that never opts in) this returns EXACTLY `sampleAtlasAspect(atlas, s, uv,
// slice, uvScale, duvdx, duvdy)` — the identical single texture read the pre-anti-tiling shader
// did — so opted-out scenes (Visualizer) are byte-for-byte unchanged. For
// strength in (0,1] the three-tap hex blend is mixed in by `strength`, so the
// caller can dial the de-repetition from subtle to full.
//
// S2.5 — the blend is VARIANCE-PRESERVING, not the naive linear `mix` this shipped with.
//
// The three taps read three uncorrelated regions of the same texture, so they behave as
// i.i.d. draws from one distribution with mean μ and variance σ². A weighted average with
// Σwᵢ = 1 keeps the mean but carries variance σ²·Σwᵢ², and Σwᵢ² runs from 1 (a cell centre,
// one tap) down to 1/3 (a triangle centroid, three equal taps). So the old blend threw away
// up to **⅔ of the material's contrast**, worst exactly in the middle of every lattice
// triangle — a de-repeat that pays for itself in a soft, washed floor, which is the
// contrast-flattening docs/MATERIALS_AND_TEXTURES §6 warns about.
//
// The fix restores the second moment: `out = μ + (blend − μ)/√(Σwᵢ²)`. It is exact at a
// cell centre (Σwᵢ² = 1 ⇒ no change ⇒ still continuous with the single-tap fast path) and
// scales by √3 at a centroid. μ comes from `sliceMean[slice]`, computed host-side in the
// SAME space the sampler returns (see IlluminatoramaTextureAtlas.sliceMeanBuffer); when the
// host never recorded one (`w == 0`, e.g. a live-blitted slice) this degrades to the old
// linear blend rather than rescaling around garbage.
//
// This is the cheap half of Mikkelsen's histogram-preserving hex tiling: it preserves the
// first two moments but not the tails, so the [0,1] clamp below still clips where the true
// inverse-CDF would have curved. See the S2.5 note in VISUAL_QUALITY_STRETCH for why the
// full Gaussianised-texture + inverse-CDF-LUT form is NOT worth its baked resource here.
static inline float4 sampleAtlasHex(texture2d_array<float, access::sample> atlas,
                                     sampler s, float2 uv, uint slice,
                                     const device float2* uvScale,
                                     const device float4* sliceMean,
                                     float strength,
                                     float2 duvdx, float2 duvdy) {
    // Exact single-sample fast path — identical to sampleAtlasAspect, no extra taps.
    float4 single = sampleAtlasAspect(atlas, s, uv, slice, uvScale, duvdx, duvdy);
    if (strength <= 0.0f) { return single; }
    const float sq3over2 = 0.8660254f;         // sqrt(3)/2
    // Skew UV to triangular lattice: (u, v) → (u + v*0.5, v*sqrt(3)/2)
    float su = uv.x + uv.y * 0.5f;
    float sv = uv.y * sq3over2;
    float si = floor(su), sj = floor(sv);
    float fu = su - si, fv = sv - sj;

    // Barycentric weights and cell-vertex integer coords in skewed space.
    // Lower triangle: verts at (si,sj), (si+1,sj), (si,sj+1)
    // Upper triangle: verts at (si+1,sj+1), (si,sj+1), (si+1,sj)
    float2 sk0, sk1, sk2;
    float  w0, w1, w2;
    if (fu + fv < 1.0f) {
        sk0 = float2(si,       sj);
        sk1 = float2(si + 1.0f, sj);
        sk2 = float2(si,       sj + 1.0f);
        w0 = 1.0f - fu - fv; w1 = fu; w2 = fv;
    } else {
        sk0 = float2(si + 1.0f, sj + 1.0f);
        sk1 = float2(si,        sj + 1.0f);
        sk2 = float2(si + 1.0f, sj);
        w0 = fu + fv - 1.0f; w1 = 1.0f - fu; w2 = 1.0f - fv;
    }

    // Hash each vertex (in skewed-lattice integer space) to a UV offset.
    float2 h0 = hexHash2D(sk0), h1 = hexHash2D(sk1), h2 = hexHash2D(sk2);

    // Cubic blend weight (w^3 normalised) — smooth at cell boundaries.
    float p0 = w0 * w0 * w0, p1 = w1 * w1 * w1, p2 = w2 * w2 * w2;
    float pSum = max(p0 + p1 + p2, 1e-6f);

    float4 c0 = sampleAtlasAspect(atlas, s, uv + h0, slice, uvScale, duvdx, duvdy);
    float4 c1 = sampleAtlasAspect(atlas, s, uv + h1, slice, uvScale, duvdx, duvdy);
    float4 c2 = sampleAtlasAspect(atlas, s, uv + h2, slice, uvScale, duvdx, duvdy);
    float4 hex = (c0 * p0 + c1 * p1 + c2 * p2) / pSum;

    // Variance-preserving rescale (see the header note). `w` is the normalised weight
    // triple, so dot(w,w) == Σwᵢ² ∈ [1/3, 1] and rsqrt of it ∈ [1, √3].
    float4 mu = sliceMean[slice];
    if (mu.w > 0.5f) {
        float3 w = float3(p0, p1, p2) / pSum;
        float rescale = rsqrt(max(dot(w, w), 1e-6f));
        // rgb only: nothing reads alpha off a hex tap (albedo takes .rgb, the normal map
        // .xyz, roughness .g), and the albedo atlas's alpha is premultiplication state,
        // not a signal whose contrast means anything.
        hex.rgb = clamp(mu.rgb + (hex.rgb - mu.rgb) * rescale, 0.0f, 1.0f);
    }
    // Blend the de-repeated hex result toward the plain single sample by strength.
    // strength==1 → full hex, strength==0 handled by the early-out above.
    return mix(single, hex, saturate(strength));
}
