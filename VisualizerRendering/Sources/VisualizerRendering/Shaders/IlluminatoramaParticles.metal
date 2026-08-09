// ── ILLUMINATORAMA — PARTICLES ──────────────────────────────────────────────
//
// The GPU particle step and the billboard vertex/fragment stages, including the
// externally-simulated variant.

#include <metal_stdlib>
#include "IlluminatoramaCommon.h"
using namespace metal;

// ── Phase 4.11 — Particles ───────────────────────────────────────────────────
//
// Compute-driven particle integration (kernel) + forward additive HDR
// render (vertex + fragment) for in-flight particles. Layout matches the
// Swift `IlluminatoramaParticle` / `IlluminatoramaParticleFrameUniforms`
// structs byte-for-byte.

struct Particle {
    float3 position;
    float  life;
    float3 velocity;
    float  size;
    float3 color;
    float  _pad;
};

struct ParticleFrameUniforms {
    float  dt;
    float3 gravity;
    float  drag;
    uint   capacity;
    float  _pad0;
    float  _pad1;
    float  _pad2;
};

kernel void illumi_particles_step(
    device Particle*                    particles  [[buffer(0)]],
    constant ParticleFrameUniforms&     pf         [[buffer(1)]],
    uint                                gid        [[thread_position_in_grid]]
) {
    if (gid >= pf.capacity) return;
    Particle p = particles[gid];
    if (p.life <= 0.0) return;
    // Symplectic Euler: velocity first, then position. Stable for gravity
    // + drag.
    p.velocity += pf.gravity * pf.dt;
    p.velocity *= exp(-pf.drag * pf.dt);
    p.position += p.velocity * pf.dt;
    // Decay life at ~1/lifetime. We don't store lifetime per-particle in
    // 4.11 — life decays at a fixed rate so each emit's particles last
    // ~1.0s. Adjust on the host by scaling input `life`.
    p.life -= pf.dt;
    if (p.life < 0.0) p.life = 0.0;
    particles[gid] = p;
}

struct ParticleVSOut {
    float4 clipPos      [[position]];
    float3 color;
    float  life;
    float2 uv;          // (0,0) bottom-left → (1,1) top-right of the streak quad
};

// Phase 4.11 round 4 — velocity-aligned billboard quad per particle.
// Six vertices form two triangles; the vertex shader builds a tangent
// (along screen-space velocity) and bitangent so the quad stretches in
// the direction the spark is moving. This is what makes a single
// snapshot READ as motion — a 6 m/s ember now draws as a 4:1 streak
// rather than a round bead.
//
// Dispatch is `drawPrimitives(.triangle, vertexCount: capacity * 6)`.
// `vid / 6` indexes the particle; `vid % 6` selects the quad corner.
vertex ParticleVSOut illumi_particles_vs(
    uint                       vid        [[vertex_id]],
    const device Particle*     particles  [[buffer(0)]],
    constant FrameUniforms&    frame      [[buffer(1)]]
) {
    uint pid    = vid / 6;
    uint corner = vid % 6;
    Particle p  = particles[pid];

    // Quad corner offsets in (tangent, bitangent) coords. Layout is
    // two triangles sharing the diagonal:
    //   0: (-1,-1)  1: ( 1,-1)  2: (-1, 1)   (first triangle)
    //   3: (-1, 1)  4: ( 1,-1)  5: ( 1, 1)   (second triangle)
    const float2 cornerLUT[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
        float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
    };
    float2 c = cornerLUT[corner];

    ParticleVSOut o;
    if (p.life <= 0.0) {
        // Dead → emit a degenerate vertex that the rasteriser clips.
        o.clipPos = float4(0, 0, -1, -1);
        o.color = float3(0);
        o.uv = float2(0);
        o.life = 0;
        return o;
    }

    // Centre clip-space position.
    float4 cClip = frame.viewProjection * float4(p.position, 1.0);
    // First-order projection of the velocity into NDC. Treats velocity
    // as a small displacement; for streak-rendering speeds this is
    // accurate to within a few percent.
    float4 vClip = frame.viewProjection * float4(p.velocity, 0.0);
    float2 velNDC = vClip.xy / max(cClip.w, 0.1);
    float  velMag = length(velNDC);

    // Tangent / bitangent in NDC space. Fall back to horizontal when
    // the spark is essentially stationary (so a particle at apex still
    // renders as a round dot rather than a zero-width line).
    float2 tangent   = velMag > 0.001 ? velNDC / velMag : float2(1.0, 0.0);
    float2 bitangent = float2(-tangent.y, tangent.x);

    // Convert pixel size to NDC. 1400 ≈ half of internal-resolution
    // 1080p plus a fudge to keep the streak's short axis (~spark
    // thickness) thinner than the round-dot equivalent — anisotropic
    // sprites read better when narrow.
    float wAttn   = sqrt(max(cClip.w, 0.5));
    float ndcShort = (p.size / 1400.0) / wAttn;
    // Streak length along tangent. Linear in velocity above a small
    // floor so:
    //   v = 0       → ndcLong = ndcShort (square round dot)
    //   v moderate  → ndcLong = ndcShort + k·perFrameNDC (elongated)
    //   v very high → capped at ~10× short axis so the streak doesn't
    //                  exit-screen on the first frame.
    // k = 3.0 with shutter time 1.0/60 gives roughly a 4–6:1 streak at
    // the lab's spawn velocities — what the reviewer asked for.
    float perFrameNDC = velMag * (1.0 / 60.0);
    float ndcLong = ndcShort + min(3.0 * perFrameNDC, 9.0 * ndcShort);

    float2 ndcOffset = c.x * tangent * ndcLong
                     + c.y * bitangent * ndcShort;
    o.clipPos = cClip;
    // Add the corner offset in pre-divide space (×cClip.w) so it lands
    // at the intended NDC offset after the perspective divide.
    o.clipPos.xy += ndcOffset * cClip.w;

    // Colour ramp (Planckian-locus mix) × twinkle envelope.
    // Twinkle: 4·t·(1-t) is zero at birth (life=1.0) and at death
    // (life=0.0), peaks at life=0.5. Gives embers a "fade-in, peak,
    // fade-out" envelope that reads as living embers rather than
    // pop-in/pop-out flashes. Reviewer round-4 polish item.
    float t = clamp(p.life, 0.0, 1.0);
    float3 cool = float3(0.6, 0.08, 0.03);
    float3 hot  = float3(1.0, 1.0, 1.0);
    float3 ramp = mix(cool, hot, smoothstep(0.0, 0.85, t));
    float twinkle = 4.0 * t * (1.0 - t);
    o.color = p.color * ramp * twinkle;

    // UV in [0,1] for the fragment falloff.
    o.uv  = c * 0.5 + 0.5;
    o.life = p.life;
    return o;
}

// ── Phase 4.23 — host-buffer point-sprite renderer ───────────────────────────
//
// Many scenes ship their own GPU particle pipelines that write into
// `MTLBuffer`s of `(position, color)` per particle (FireworksUltra's
// starfield + burst particles, Foam's spray clouds, anything that
// previously bound to SceneKit via a `.point` primitive SCNGeometry).
// These can't go through `IlluminatoramaParticleEmitter`'s integrate→draw
// path because the host owns the simulation; we just need to render the
// buffers as additive HDR point sprites.
//
// The VS reads from two `device float*` buffers with HOST-PROVIDED
// strides (in float units, NOT bytes — Metal indexes `float*` in
// floats). FireworksUltra ships `SIMD4<Float>` (stride-4 floats);
// other scenes might ship `packed_float3` (stride-3 floats). The
// indexing math handles both without a per-buffer pipeline variant.
//
// Output is a `.point` primitive with size in pixels via `[[point_size]]`;
// the fragment shader does a soft Gaussian falloff inside `point_coord`
// for the glowing-orb look bloom turns into twinkles.

struct ExtParticleVSOut {
    float4 clipPos    [[position]];
    float  pointSize  [[point_size]];
    float3 color;
};

struct ExtParticleParams {
    uint  posStrideFloats;
    uint  colorStrideFloats;
    uint  posOffsetFloats;    // float offset of position within its stride slot
    uint  colorOffsetFloats;  // float offset of colour within its stride slot
    float pointSize;     // pixels
    float colorScale;    // host-tunable scalar applied to color before output
};

vertex ExtParticleVSOut illumi_extparticle_vs(
    uint                       vid       [[vertex_id]],
    const device float*        positions [[buffer(0)]],
    const device float*        colors    [[buffer(1)]],
    constant FrameUniforms&    frame     [[buffer(2)]],
    constant ExtParticleParams& params   [[buffer(3)]]
) {
    uint pOff = vid * params.posStrideFloats   + params.posOffsetFloats;
    uint cOff = vid * params.colorStrideFloats + params.colorOffsetFloats;
    float3 p = float3(positions[pOff + 0],
                       positions[pOff + 1],
                       positions[pOff + 2]);
    float3 c = float3(colors[cOff + 0],
                       colors[cOff + 1],
                       colors[cOff + 2]);
    ExtParticleVSOut o;
    o.clipPos   = frame.viewProjection * float4(p, 1.0);
    o.color     = c * params.colorScale;
    o.pointSize = params.pointSize;
    return o;
}

fragment float4 illumi_extparticle_fs(
    ExtParticleVSOut in       [[stage_in]],
    float2           pcoord   [[point_coord]]
) {
    // Soft Gaussian inside the point quad — gives the bright-core +
    // bloomable-halo look without an external texture. `pcoord` is
    // (0,0)..(1,1) over the point's footprint; recentre to the disc.
    float2 d = pcoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard_fragment();
    float fall = exp(-r2 * 16.0);
    return float4(in.color * fall, fall);
}

fragment float4 illumi_particles_fs(
    ParticleVSOut             in           [[stage_in]]
) {
    if (in.life <= 0.0) discard_fragment();
    // Soft Gaussian falloff over the quad's local UV. `uv` is mapped
    // from corner offsets so dx runs along tangent (long axis) and dy
    // along bitangent (short axis). The Gaussian gives a hot core with
    // a soft outer halo — the additive blend then produces glowing
    // streaks instead of the round 4 / round 5 "dumpling" look reviewer
    // flagged (hard-edged opaque blobs).
    //
    // Round 6 — debug-pass confirmed that the vertex geometry is in
    // fact producing elongated quads; the structural concern from
    // rounds 4 / 5 was a polish-stage falloff issue, not a geometry
    // bug.
    float2 d = in.uv - 0.5;
    float r2 = d.x * d.x + d.y * d.y;
    if (r2 > 0.25) discard_fragment();
    // Half-width at half-max ~0.2 in UV space; outer ~30% gets soft.
    float falloff = exp(-12.0 * r2);
    return float4(in.color * falloff, falloff);
}
