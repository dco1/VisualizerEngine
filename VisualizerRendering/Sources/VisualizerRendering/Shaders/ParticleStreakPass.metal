// ── PARTICLE STREAK PASS SHADERS ────────────────────────────────────────────
//
// Vertex + fragment for the reusable ParticleStreakPass — renders the
// FireworkParticleSolver's `FWStreakVertex` buffer (positions + colors)
// + parallel UV buffer through a custom Metal render pipeline, writing
// HDR additive into the FrameCompositor's target.
//
// REUSABLE for any particle scene that drives a streak-quad buffer via
// the same kernel pattern (Fireworks+, future Dust, Snow, Sparks scenes).

#include <metal_stdlib>
using namespace metal;

// ── VERTEX INPUTS ───────────────────────────────────────────────────────────
//
// `color` is now a float4: rgb is the per-particle hue (already pre-
// multiplied by tailFade in the kernel for backwards compatibility with
// the SCN-pipeline scenes), and `color.a` carries the STRETCH FACTOR
// (vTile = streakLen / width) — how elongated this particle's quad is
// along its velocity axis. Constant per particle, varies between
// particles based on their speed and current age.
struct VSIn {
    float3 position [[ attribute(0) ]];
    float4 color    [[ attribute(1) ]];
    float2 uv       [[ attribute(2) ]];
};

// ── VERTEX → FRAGMENT INTERPOLANTS ──────────────────────────────────────────
//
// `stretchFactor` is the same vTile at every vertex of the quad, so the
// rasterizer interpolates it to the same constant value at every fragment.
// The fragment shader uses it to build an anisotropic SDF — the visible
// region stretches by stretchFactor along the velocity (UV.y) direction
// for cinematic motion blur on fast-moving particles.
struct VSOut {
    float4 position [[ position ]];
    float3 color;
    float2 uv;
    float  stretchFactor;
};

vertex VSOut particleStreakVS(
    VSIn vin [[ stage_in ]],
    constant float4x4& viewProjection [[ buffer(2) ]]
) {
    VSOut o;
    o.position = viewProjection * float4(vin.position, 1.0f);
    o.color = vin.color.rgb;
    o.uv = vin.uv;
    o.stretchFactor = vin.color.a;
    return o;
}

// ── SPRITE SAMPLER ──────────────────────────────────────────────────────────
//
// LINEAR magnification + minification + mipmaps for anti-aliased
// sampling at any on-screen size. CRITICAL for kill-the-squares
// behaviour: at small sizes (a few pixels per particle) the GPU
// picks a mip level whose pre-filtered values are already smooth,
// so we don't get pixel-quantised squares from raster aliasing.
//
// V-axis WRAP = REPEAT so the streak quad (whose UV.v runs 0..vTile)
// tiles the sprite as a chain of round Gaussian dots along the
// velocity vector — that's how the kernel was designed to render
// streaks. Without `.repeat` on V, a fast-moving spark with vTile=5
// would clamp to the sprite's edge (zero) past UV.v=1 and show only
// a single dim disc at the tail. With `.repeat`, every tile-position
// along the streak gets a fresh Gaussian sample.
//
// U-axis WRAP = CLAMP_TO_EDGE so the sprite stays single-width
// across the streak — no horizontal tiling. The sprite's outer
// pixels are transparent black, so clamping creates a clean
// silhouette edge across the streak width.
constexpr sampler spriteSampler(
    filter::linear,
    mip_filter::linear,
    s_address::clamp_to_edge,
    t_address::repeat
);

struct ParticleStreakParams {
    // Per-fragment multiplier on the final HDR contribution. Larger
    // values push more bloom + a brighter pop through the tonemap.
    float intensity;
    // Gamma applied to the per-particle colour after normalising to
    // its brightest channel — pushes secondary channels toward zero
    // so a red-ish particle reads as bold pure red instead of pink.
    // Higher values = more saturated.
    float saturationPow;
};

fragment float4 particleStreakFS(
    VSOut vin [[ stage_in ]],
    texture2d<float, access::sample> sprite [[ texture(0) ]],
    constant ParticleStreakParams& params [[ buffer(0) ]]
) {
    // ── SPHERE IMPOSTER — REAL 3D, NO 2D FAKE GLOW ──────────────────
    //
    // Each particle quad is rendered as a true sphere imposter:
    //
    //   1. INSIDE the inscribed circle (r ≤ 1) we treat each fragment
    //      as a point on the surface of a unit sphere centred on the
    //      quad. The sphere's surface normal at any screen position
    //      (x, y) is `(x, y, sqrt(1 - x² - y²))` — actual unit
    //      sphere geometry projected onto the quad.
    //
    //   2. Shade that normal with Lambert + Blinn-Phong against a
    //      fixed-direction light. The OFFSET specular highlight is
    //      the single most important 3D cue — a real photographed
    //      glowing ball has a bright spot OFFSET from the centre,
    //      not centred. Any centred radial profile reads as flat.
    //
    //   3. CHORD term `sqrt(1-r²)` provides the physically-correct
    //      volumetric emission brightness (path length through the
    //      sphere along the line of sight).
    //
    //   4. fwidth-AA on the SILHOUETTE — the chord function goes to
    //      zero at r=1 with a vertical tangent (the sharp silhouette
    //      that defines a sphere). fwidth feathers this transition
    //      over one screen pixel regardless of particle size, so
    //      small particles don't pixel-quantise to squares.
    //
    // CRITICAL: NO 2D OUTER HALO.
    //
    // The previous version added a soft Gaussian `halo` lobe that
    // extended past the sphere silhouette to fake atmospheric glow.
    // That created the "plastic 2D rim" the user spotted: inside
    // r=1 the sphere was shaded in 3D (Lambert, spec, offset
    // highlight); outside r=1 only the flat 2D halo Gaussian
    // contributed. The eye sees the discontinuity between those two
    // visual languages as a thin plastic ring around the 3D ball.
    //
    // The correct way to get atmospheric glow around a bright light
    // is the compositor's BLOOM PASS, which already runs after this
    // shader. Bloom thresholds bright pixels, downsamples them,
    // Gaussian-blurs the result, and adds it back to the HDR — that
    // IS the optical-scatter atmospheric halo a real camera lens
    // produces around a real glowing point source. Hand-painted 2D
    // halos in the particle shader are just an inferior reinvention
    // of the same effect. So we render an HONEST 3D sphere here
    // and let the bloom pass do the glow downstream.

    // ── ANISOTROPIC CAPSULE SDF — CINEMATIC MOTION BLUR ─────────────
    //
    // Two upgrades from the previous version:
    //
    //   1. The shape is now a CAPSULE (2D rounded-rect / 3D pill),
    //      not a circle. Static particles still render as spheres
    //      (the capsule's cylinder body has zero length when
    //      stretchFactor=1, so the shape degenerates to a sphere);
    //      fast-moving particles render as elongated capsules
    //      stretched along their velocity vector. That's the
    //      physical model for camera motion blur: a moving point
    //      source integrated over the shutter duration sweeps out
    //      a capsule-shaped path on the sensor.
    //
    //   2. The SDF is built from per-particle `stretchFactor`
    //      (a.k.a. vTile = streakLen / width) which the kernel
    //      packed into color.a and the vertex shader passed
    //      through. With this number per fragment, we can shape
    //      the capsule to match the streak quad's geometry
    //      exactly.
    //
    // QUAD COORDINATE SETUP:
    //
    //   `vin.uv.x` ranges 0..1 across the streak WIDTH
    //              (perpendicular to velocity in world space)
    //   `vin.uv.y` ranges 0..stretchFactor along the streak LENGTH
    //              (parallel to velocity; 0 = tail vertex, vTile = head)
    //
    // CAPSULE PARAMETERISATION — CENTERED FORM:
    //
    // Earlier version used `axisEnd = 2·stretch − 2` with the axis
    // anchored at `p.y = 0` (the tail end). That worked but baked
    // a directional bias into UV space: the capsule was anchored
    // to one end rather than centered on the quad. Subtle but
    // matters for symmetric operations (e.g. matching the
    // head→tail fade against the AA gradient at both ends, or
    // future motion-direction reversal).
    //
    // Symmetric form, centered on the quad's geometric centre:
    //
    //   q.x = 2·(uv.x − 0.5)              → −1..+1 across width
    //   q.y = 2·(uv.y − 0.5) − (stretch−1) → centred along length
    //   halfLen = stretch − 1              → half-length of the
    //                                        cylinder body (0 for
    //                                        stretch=1 sphere)
    //
    // After this remap, the capsule axis is the segment from
    // (0, −halfLen) to (0, +halfLen), symmetric about origin.
    // The cylinder body is the strip |q.y| ≤ halfLen with radius 1;
    // the two hemisphere caps extend |q.y| past halfLen by up to 1.
    //
    // Closest point on axis: `q' = (0, clamp(q.y, −halfLen, +halfLen))`.
    // Then the standard capsule SDF is:
    //
    //   sdf = length(q − q') − 1
    //
    // For stretch = 1: halfLen = 0, q' = origin, sdf = length(q) − 1
    // (the unit circle — pure sphere imposter, identical to the
    // sphere-only build).
    float2 p = (vin.uv - 0.5f) * 2.0f;

    // Stretch factor — clamp to >= 1.0 just in case the kernel
    // returns a sub-1 value at some boundary (it shouldn't).
    float stretch = max(1.0f, vin.stretchFactor);

    // Re-center p.y so the capsule is symmetric about origin.
    float halfLen = stretch - 1.0f;
    p.y -= halfLen;

    // Closest point on the symmetric capsule axis.
    float axisY = clamp(p.y, -halfLen, halfLen);
    float2 axisPoint = float2(0.0f, axisY);

    // Vector from axis to this fragment. This is the unified
    // surface-radial direction that the normal reconstruction below
    // uses everywhere on the capsule (cylinder body OR cap) — it's
    // simply `normalize(p − closestPointOnSegment)` extended with
    // the depth (chord) component. No cap-vs-cylinder branching
    // means no discontinuity at the boundary.
    float2 toAxis     = p - axisPoint;
    float  distToAxis = length(toAxis);

    // Standard 2D capsule SDF.
    float sdf = distToAxis - 1.0f;

    // ── ANISOTROPIC SCREEN-SPACE AA ─────────────────────────────────
    //
    // Previous version used `smoothstep(0, fwidth(sdf), -sdf)`.
    // `fwidth(x) = |dFdx(x)| + |dFdy(x)|` — a cheap L1 estimate of
    // the screen-space gradient magnitude. For an isotropic shape
    // (sphere imposter) this is fine. For a long thin capsule
    // viewed at a grazing angle, the L1 estimate over-weights one
    // axis and the AA band becomes inconsistent along the streak,
    // producing edge shimmer along the long axis.
    //
    // The robust analytic form: use the actual screen-space SDF
    // gradient `length((dFdx(sdf), dFdy(sdf)))` so the AA band is
    // exactly one screen pixel wide in the direction the SDF is
    // changing fastest, regardless of capsule orientation. This
    // is the textbook "thick line" AA used in font rasterisers
    // and signed-distance UI primitives.
    float2 sdfGrad = float2(dfdx(sdf), dfdy(sdf));
    float aaWidth = max(length(sdfGrad), 1e-4f);
    float alpha = smoothstep(0.0f, aaWidth, -sdf);
    if (alpha <= 0.0f) discard_fragment();

    // ── UNIFIED SPHERE/CAPSULE SHADING (no cap↔cylinder branch) ─────
    //
    // The shading is derived from the SAME `toAxis` vector the SDF
    // used. That's the unified-capsule-normal trick: the surface
    // normal at any point on a capsule is `normalize(p − closestPointOnSegment)`,
    // which automatically gives:
    //
    //   • Cylinder body  → toAxis = (p.x, 0)   → purely radial-X normal
    //   • Head cap       → toAxis = p − headPt → sphere-radial normal
    //   • Tail cap       → toAxis = p − tailPt → sphere-radial normal
    //
    // No branching, no thresholds, no cap-vs-body region detection.
    // The transition between cylinder and cap is C¹-continuous
    // because the clamp on `axisY` smoothly transitions toAxis.y
    // from 0 (inside cylinder) to ±(p.y − ±halfLen) (in caps).
    //
    // `chord` is the line-of-sight depth through the unit-radius
    // capsule at this position — used both as the Z-component of
    // the surface normal (for Lambert/spec shading) AND as the
    // volumetric emission brightness (a uniformly-glowing capsule
    // is brightest where the line of sight passes deepest through
    // the volume).
    float chord = sqrt(max(0.0f, 1.0f - distToAxis * distToAxis));
    float3 normal = normalize(float3(toAxis.x, toAxis.y, max(chord, 0.001f)));

    // ── MOTION BLUR BRIGHTNESS: PERCEPTUAL, NOT RADIOMETRIC ─────────
    //
    // A real camera with a finite shutter integrates emission over
    // time. For a particle moving at constant velocity during the
    // shutter, the RADIOMETRICALLY CORRECT per-pixel brightness
    // scales as 1 / stretch — total energy stays constant, spread
    // over `stretch` times as many pixels.
    //
    // We use 1 / sqrt(stretch) instead. That's a deliberate
    // PERCEPTUAL choice, not a physical one. Pure radiometric
    // (1/stretch) makes fast streaks visually invisible (a 5×
    // streak is 80% dimmer per pixel than a static particle, which
    // looks underexposed compared to the reference). The sqrt
    // compromise keeps fast streaks legibly bright while still
    // visually distinguishing them from slow particles.
    //
    // This is what cinema film stocks actually do — they're not
    // radiometrically linear; the response curve compresses
    // motion-blurred frames toward "perceptually correct" rather
    // than "energy conserved." If you ever want strict radiometric
    // correctness, change to `1.0 / stretch` and ALSO drop the
    // head→tail fade below (which is a separate perceptual choice
    // that would otherwise double-count).
    float motionBlurConservation = 1.0f / sqrt(stretch);

    // Light direction in view space. +Z = toward camera. UV +Y is
    // DOWN on screen, so a NEGATIVE Y component lights from the top.
    // Upper-right-front yields a top-right specular spot — visually
    // clear and intuitive.
    float3 lightDir = normalize(float3(0.45f, -0.55f, 0.70f));
    float NdotL = max(0.0f, dot(normal, lightDir));

    // ── LAMBERT BIAS — HIGH CONTRAST ────────────────────────────────
    //
    // Was `0.55 + 0.45 · NdotL` — too subtle. At small on-screen
    // sizes the gradient compressed into a few pixels and the eye
    // saw approximate uniform brightness ⇒ "flat".
    //
    // New mix `0.18 + 0.82 · NdotL` produces a 5× brightness ratio
    // between the unlit and lit hemispheres. The dark side is
    // visibly DARK; the lit side is visibly BRIGHT. That clear
    // gradient is the directional asymmetry that screams 3D rather
    // than "soft 2D blob". The unlit side isn't pure black (real
    // emitters self-illuminate through their volume) so there's
    // always some body brightness there to read as the sphere's
    // back-half.
    float bodyBrightness = chord * (0.18f + 0.82f * NdotL);

    // ── SPECULAR — DOMINANT OFFSET HIGHLIGHT ────────────────────────
    //
    // Blinn-Phong with exponent 18 + multiplier 3.5. Lower exponent
    // (vs the previous 32) widens the hot spot so it covers several
    // pixels even on small particles — guarantees the offset
    // highlight is visible at any size. Higher multiplier (vs the
    // previous 1.6) makes the spot push past the body brightness
    // and saturate toward white in HDR, mimicking a real camera's
    // over-exposed hot-pixel response.
    //
    // This bright OFFSET spot — sitting upper-right of the sphere
    // centre, not centred — is the single most important 3D cue.
    // A centred radial peak reads as a flat disc no matter how soft;
    // an off-centre peak reads as the lit pole of a sphere.
    float3 viewDir = float3(0.0f, 0.0f, 1.0f);
    float3 halfway = normalize(lightDir + viewDir);
    float NdotH = max(0.0f, dot(normal, halfway));
    float spec = pow(NdotH, 18.0f) * 3.5f;

    // ── PER-PARTICLE HUE ────────────────────────────────────────────
    float3 c = vin.color;
    float maxc = max(c.r, max(c.g, c.b));
    float3 norm = (maxc > 1e-4) ? c / maxc : float3(1.0);
    float3 saturated = pow(norm, float3(params.saturationPow));

    // ── COLOUR PRESERVATION — NO HARDCODED WHITE LAYER ──────────────
    //
    // Previous iterations multiplied the specular term by a fixed
    // warm-white `(1.0, 0.96, 0.88)` and added that on TOP of the
    // coloured body. At the user's HDR intensity settings (3.5–5×
    // baseline × the slider) the white component completely
    // dominated the coloured component, washing every particle to
    // a near-white blob with only a faint colour halo.
    //
    // That was physically wrong. A real spark is an incandescent
    // emitter — the WHOLE BODY emits the same spectrum (Wien's law:
    // a uniformly-hot body emits a single colour temperature). The
    // "white-hot centre" we see in real photographs is the camera
    // SENSOR clipping, not a separate spectral channel. Sensor
    // saturation belongs in the tonemap, not the particle shader.
    //
    // So: render the entire sphere — body, spec, everything — in
    // the particle's actual hue. The compositor's downstream
    // chroma-preserving ACES tonemap clips bright pixels naturally,
    // producing the "white-hot core" effect where (and only where)
    // the HDR value really does exceed the tonemap ceiling. Dimmer
    // pixels retain their colour as they should. Colours stop
    // washing out across the board.
    //
    // The 3D LOOK STILL COMES THROUGH because:
    //   • bodyBrightness alone already encodes the Lambert gradient
    //     (5× contrast between lit and unlit hemispheres)
    //   • spec alone already encodes the offset bright spot
    //   • their SUM is brightest where the spec peaks (the offset
    //     hot spot), which after tonemap clipping reads as
    //     overexposed white relative to the dimmer body — same
    //     visual cue, but the white emerges from physics, not from
    //     a hardcoded `hotWhite` constant.
    // ── HEAD → TAIL FADE — EMISSION-DECAY HEURISTIC ─────────────────
    //
    // For a real box-shutter camera and a particle moving at
    // constant velocity, every position along the streak gets the
    // SAME exposure time (the shutter sweeps each position once).
    // A box exposure of a constant-emission particle would produce
    // a uniform-brightness streak, NOT a head-to-tail gradient.
    //
    // So a head→tail gradient is NOT a true exposure integral.
    // What it IS modelling, accurately, is the fact that our
    // particles' EMISSION DECAYS OVER THEIR LIFETIME (hot-start
    // burst, fading embers). The pixel near the head saw the
    // particle's freshly-spawned hot emission; the pixel near the
    // tail saw the particle's slightly-older (slightly-dimmer)
    // emission a few ms earlier. For a fast-moving young spark
    // this temporal difference is significant — exactly the
    // dragon-tail visual signature of real cinematic fireworks
    // photography.
    //
    // `headness` ∈ [0, 1] tracks position along the streak axis
    // measured in the CENTERED p-space (post-recentering by
    // halfLen). Mapping: tail end at p.y = -halfLen → headness=0;
    // head end at p.y = +halfLen → headness=1. For stretch=1
    // (no streak), halfLen=0 and we divide-by-zero-guard so
    // headness collapses to ~0.5 — the static case sees no fade
    // anyway because the `blurStrength` gate is zero.
    //
    // `mix(0.35, 1.0, headness)` keeps the tail at 35% of peak —
    // a pure 0..1 fade would make the very tail invisible. The
    // gate `smoothstep(1.0, 2.0, stretch)` ensures static
    // dying embers (stretch≈1) see no fade and stay uniformly
    // bright; fade only ramps in for genuinely fast-moving
    // particles where the emission-decay signature is real.
    //
    // FUTURE: replace this heuristic with a true 1D exposure
    // integral along the capsule axis. That would let the kernel
    // pass an "emission curve" function and the fragment shader
    // would analytically integrate `emission(t) × shutter(t)` for
    // the segment from t_head to t_tail. More accurate camera
    // model; same cost; same code footprint.
    float headness = saturate(0.5f + p.y / max(2.0f * halfLen, 1e-4f));
    float blurStrength = smoothstep(1.0f, 2.0f, stretch);
    float headTailFade = mix(1.0f, mix(0.35f, 1.0f, headness), blurStrength);

    // ── FINAL HDR COMPOSITE ─────────────────────────────────────────
    //
    // body + spec → multiplied by per-particle hue × user intensity
    // × motion-blur conservation × head→tail fade × SDF silhouette
    // AA alpha. The compositor's chroma-preserving ACES tonemap
    // handles the final bright-pixel-to-white clipping downstream.
    float totalBrightness = bodyBrightness + spec;
    float3 hdr = totalBrightness
               * saturated * maxc
               * params.intensity
               * motionBlurConservation
               * headTailFade
               * alpha;

    // ── DEPTH-BASED ATMOSPHERIC HAZE ───────────────────────────────
    // Far particles fade toward a cool moonlit haze, near particles
    // stay vivid. Establishes the parallax depth cue ChatGPT flagged
    // as missing — without this every spark renders at the same
    // saturation regardless of how far away it is, so the scene reads
    // flat.
    //
    // ndcZ is in [0, 1] (Metal NDC). Sparks within the burst zone
    // (~50 m from camera, depth ≈ 0.85 typical) get a mild haze; far
    // sparks (near the slab edge) get more.
    float ndcZ = saturate(vin.position.z / max(vin.position.w, 1e-4f));
    float hazeT = smoothstep(0.85f, 0.999f, ndcZ);
    float3 hazeColor = float3(0.05f, 0.07f, 0.11f);  // cool moonlit haze
    hdr = mix(hdr, hazeColor * params.intensity * 0.2f, hazeT * 0.7f);

    return float4(hdr, 1.0f);
}
