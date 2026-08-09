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
