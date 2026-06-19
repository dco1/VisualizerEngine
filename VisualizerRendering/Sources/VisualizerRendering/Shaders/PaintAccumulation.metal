#include <metal_stdlib>
using namespace metal;

// ── Paint accumulation ────────────────────────────────────────────────────────
//
// A persistent, never-cleared RGBA16F accumulation texture for the Tennis Ball
// Painter scene. Tennis balls splat against the white wall (and spatter on the
// floor); each impact STAMPS a soft paint dab at the impact UV in the ball's
// picked-up colour. The texture is bound to the wall / floor material's
// diffuse.contents and sampled directly by SceneKit — so the painting builds up
// in GPU memory and is never rebuilt CPU-side and never erased.
//
// Drips are REAL: each splat may seed drip "rivulets" (agents with position +
// velocity + colour + life) that march DOWN the wall under gravity, stamping a
// thin wet streak as they go. No scrolling-texture fake — the streak is the
// integrated path of a gravity-advected agent.

// ── Stamp ─────────────────────────────────────────────────────────────────────
// All float4-packed to satisfy the project ALIGNMENT RULE (no bare float3 in a
// Swift<->Metal shared struct).
struct PaintStamp {
    float4 centerRadiusAlpha;   // xy = centerUV, z = radiusUV, w = alpha
    float4 colorEdge;           // xyz = colour (linear RGB), w = edgeNoise
    float4 seedFlick;           // x = seed, yz = flick dir (UV, ball incoming vel), w = wetness
    float4 extra;               // x = streakCount (# directional flicks), yzw spare
};

static inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// 2D value noise (smooth) for the irregular dab interior.
static inline float vnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + float2(1, 0));
    float c = hash21(i + float2(0, 1));
    float d = hash21(i + float2(1, 1));
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// One dispatch per impact. Grid is a small tile centred on the stamp.
//
// A wet-paint impact is NOT a soft gaussian airbrush bloom. It is a slapped
// splat: a HARD nearly-flat core (the bulk of the paint that stuck on impact),
// a short noise-ragged transition, an IRREGULAR (noise-perturbed) outer edge,
// and a few DIRECTIONAL flick streaks radiating from the impact, biased along
// the ball's incoming velocity direction (paint that sprayed off on contact).
kernel void paintStamp(texture2d<float, access::read_write> tex [[texture(0)]],
                       constant PaintStamp& s [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint W = tex.get_width();
    uint H = tex.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float2 centerUV = s.centerRadiusAlpha.xy;
    float  radiusUV = s.centerRadiusAlpha.z;
    float  alpha    = s.centerRadiusAlpha.w;
    float3 color    = s.colorEdge.xyz;
    float  edgeNoise= s.colorEdge.w;
    float  seed     = s.seedFlick.x;
    float2 flickDir = s.seedFlick.yz;          // UV-space incoming velocity dir
    float  wetness  = s.seedFlick.w;            // fresh-dab wet sheen [0,1]
    int    nStreaks = max(0, int(s.extra.x + 0.5)); // # directional flicks (varied per splat)
    // Aspect correction: when the canvas maps onto a non-square face (the wide back
    // wall), `aspectY` (= faceHeight/faceWidth) scales the v-axis so a dab that is
    // round in UV becomes round IN WORLD on the stretched face. 1 = square (floor).
    float  aspectY  = s.extra.y > 1e-3 ? s.extra.y : 1.0;

    float2 uv = (float2(gid) + 0.5) / float2(W, H);
    float2 d  = uv - centerUV;
    d.y *= aspectY;                             // v contributes less → dab taller in UV → round in world
    float dist = length(d);
    if (dist > radiusUV * 2.4) return;          // outside affected tile (flicks reach)

    float ang = atan2(d.y, d.x);

    // ── Core blob: hard core + noise-perturbed irregular edge ──────────────
    // Per-angle radius wobble (low-freq lobes) + interior value-noise so the
    // boundary is genuinely irregular, not a wavy circle.
    float lobe  = (hash21(float2(ang * 2.5, seed)) - 0.5)
                + 0.5 * (hash21(float2(ang * 5.3, seed + 7.0)) - 0.5);
    float ninterior = vnoise(d / max(radiusUV, 1e-4) * 4.0 + seed) - 0.5;
    float r = radiusUV * (1.0 + edgeNoise * lobe + 0.30 * edgeNoise * ninterior);

    // HARD core: full opacity out to ~0.62·r, then a SHORT ragged falloff to the
    // edge (vs the old broad smoothstep(r, 0.45r) airbrush gradient).
    float core = 1.0 - smoothstep(r * 0.62, r * 1.02, dist);
    // Break the core with faint interior mottling so it reads as slapped wet
    // paint (uneven film), not a flat decal.
    float mottle = 0.85 + 0.15 * vnoise(d / max(radiusUV, 1e-4) * 9.0 + seed * 1.7);
    float a = alpha * core * mottle;

    // ── Directional flick streaks ──────────────────────────────────────────
    // A handful of thin streaks shoot outward, biased along flickDir (the ball's
    // incoming velocity). Each streak is a narrow angular wedge that reaches past
    // the core; the dominant ones point downstream of the impact.
    // A FEW deliberate streaks only — too many faint long streaks from every dab
    // composite into grey-brown mud over a long settle. `nStreaks` (varied 1–6 per
    // splat host-side) short crisp streaks, strongly biased downstream of the flick
    // direction, opaque so they read as distinct paint flicks (not a faint spray).
    float flickLen = length(flickDir);
    float baseAng  = flickLen > 1e-4 ? atan2(flickDir.y, flickDir.x) : seed * 6.2831853;
    float streak = 0.0;
    for (int k = 0; k < nStreaks; ++k) {
        float jit = (hash21(float2(seed, float(k) * 3.1)) - 0.5);
        float sa = baseAng + jit * 0.9;                                     // tight ±~25° fan
        float reach = radiusUV * (1.15 + 0.55 * hash21(float2(seed + 11.0, float(k))));
        float halfw = 0.07 + 0.06 * hash21(float2(seed + 21.0, float(k)));  // narrow wedge
        float dang = atan2(sin(ang - sa), cos(ang - sa));
        float along = smoothstep(reach, radiusUV * 0.6, dist);             // 1 near core, 0 at tip
        float across = 1.0 - smoothstep(0.0, halfw, abs(dang));
        streak = max(streak, along * across);
    }
    // streaks only where there IS a flick direction (an actual impact velocity).
    if (flickLen > 1e-4) a = max(a, alpha * streak);

    if (a <= 0.001) return;

    float4 prev = tex.read(gid);
    // Wet paint over: source-over composite. Accumulates permanently — we never
    // clear. Alpha channel tracks WETNESS (fresh-dab sheen) for the wall's wet/dry
    // clearcoat read and the drip seeding pass; it decays via paintDecay.
    float3 rgb = mix(prev.rgb, color, a);
    float aOut = max(prev.a, wetness * smoothstep(0.0, 0.35, a));
    tex.write(float4(rgb, aOut), gid);
}

// Decay the WETNESS (alpha) channel toward 0 so a just-landed dab visibly
// wet-shines, then dries to matte over a couple of seconds. RGB is untouched —
// the painting is permanent; only the wet SHEEN fades. Runs once per tick.
struct DecayUniforms { float keep; float pad0; float pad1; float pad2; };
kernel void paintDecay(texture2d<float, access::read_write> tex [[texture(0)]],
                       constant DecayUniforms& u [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
    float4 px = tex.read(gid);
    if (px.a <= 0.0) return;
    px.a *= u.keep;
    if (px.a < 0.004) px.a = 0.0;
    tex.write(px, gid);
}

// ── Drips ──────────────────────────────────────────────────────────────────────
struct DripAgent {
    float4 posVel;          // xy = posUV, zw = velUV (+y/+w is DOWN the wall)
    float4 colorLife;       // xyz = colour (linear RGB), w = life (s, <=0 dead)
    float4 widthFlowPad;    // x = width (UV half-width), y = flow (opacity), zw = pad
};

struct DripUniforms {
    float dt;
    float gravity;      // UV/s^2 downward acceleration
    uint  agentCount;
    float pad;
};

// Advance each drip down the wall and stamp its streak into the same texture.
// One thread per agent.
kernel void paintDripStep(texture2d<float, access::read_write> tex [[texture(0)]],
                          device DripAgent* agents [[buffer(0)]],
                          constant DripUniforms& u [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= u.agentCount) return;
    DripAgent a = agents[gid];
    float life = a.colorLife.w;
    if (life <= 0.0) return;

    uint W = tex.get_width();
    uint H = tex.get_height();

    float2 posUV = a.posVel.xy;
    float2 velUV = a.posVel.zw;
    float3 color = a.colorLife.xyz;
    float  width = a.widthFlowPad.x;
    float  flow  = a.widthFlowPad.y;

    // Integrate gravity (drips run straight down, slowing as paint thins).
    velUV.y += u.gravity * u.dt;
    velUV   *= 0.985;                          // viscous drag
    float2 newPos = posUV + velUV * u.dt;
    life -= u.dt;

    // Stamp a short segment between old and new position so fast drips don't gap.
    int steps = 6;
    for (int i = 0; i <= steps; ++i) {
        float t = float(i) / float(steps);
        float2 p = mix(posUV, newPos, t);
        if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) continue;
        // paint a small vertical-ish dab; thinning toward the tail.
        float fade = life > 0.0 ? clamp(life * 1.5, 0.2, 1.0) : 0.2;
        int rad = max(1, int(width * float(W)));
        int cx = int(p.x * float(W));
        int cy = int(p.y * float(H));
        for (int dy = -rad; dy <= rad; ++dy) {
            for (int dx = -rad; dx <= rad; ++dx) {
                int xx = cx + dx;
                int yy = cy + dy;
                if (xx < 0 || yy < 0 || xx >= int(W) || yy >= int(H)) continue;
                float dd = length(float2(dx, dy)) / float(rad + 0.0001);
                float op = flow * fade * smoothstep(1.0, 0.2, dd);
                if (op <= 0.001) continue;
                uint2 gg = uint2(xx, yy);
                float4 prev = tex.read(gg);
                float3 rgb = mix(prev.rgb, color, op);
                tex.write(float4(rgb, max(prev.a, op)), gg);
            }
        }
    }
    if (newPos.y > 1.0) life = 0.0;            // ran off the bottom
    a.posVel = float4(newPos, velUV);
    a.colorLife = float4(color, life);
    agents[gid] = a;
}

// ── Resolve linear RGBA16F → sRGB-encoded BGRA8 (for the Illuminatorama atlas) ──
//
// The persistent paint canvas is RGBA16F holding LINEAR colour (the stamp kernel
// writes linear RGB). The Illuminatorama albedo atlas is `bgra8Unorm_srgb`, which
// the G-buffer shader samples as sRGB→linear. So to round-trip correctly we must
// write sRGB-ENCODED bytes into a `bgra8Unorm` staging texture (same byte layout,
// blit-compatible with the `_srgb` atlas). `paintResolve` does exactly that: read
// the linear src texel, encode pow(1/2.2), write BGRA8. The alpha (wetness) channel
// is dropped — the atlas albedo carries colour only.
struct ResolveUniforms { uint pad; };
kernel void paintResolve(texture2d<float, access::read>  src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint W = dst.get_width();
    uint H = dst.get_height();
    if (gid.x >= W || gid.y >= H) return;
    // src and dst may differ in resolution; sample src by normalised coord.
    uint2 sgid = uint2(uint(float(gid.x) / float(W) * float(src.get_width())),
                       uint(float(gid.y) / float(H) * float(src.get_height())));
    float3 lin = src.read(sgid).rgb;
    // Linear → sRGB encode (the inverse of the atlas's sample-time decode).
    float3 srgb = pow(clamp(lin, 0.0, 1.0), float3(1.0 / 2.2));
    dst.write(float4(srgb, 1.0), gid);
}

// Initialise the persistent texture to a base colour (called once at setup;
// thereafter the texture is never cleared).
struct ClearColor { float4 rgba; };
kernel void paintClear(texture2d<float, access::write> tex [[texture(0)]],
                       constant ClearColor& c [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
    tex.write(c.rgba, gid);
}

// ── Repaint flood front ───────────────────────────────────────────────────────
// A wave of a NEW colour pours over the canvas, covering everything (old paint and
// freshly-stamped ball dabs alike). It is NOT a composite-time override: it WRITES
// the new colour into the buffer along an advancing front, so once a texel is
// covered it becomes the new substrate and balls paint ON TOP of it afterward. To
// avoid re-erasing those later dabs, each dispatch writes ONLY the band the front
// swept THIS frame — (prevPos, curPos] in "front-coordinate" space. A per-column
// hash offset makes the leading edge drip in fingers (a waterfall, not a wipe).
struct FloodUniforms {
    float4 color;    // rgb = new colour, w = wetness written into alpha
    float4 front;    // x = prevPos, y = curPos, z = seamV (floor), w = mode (0 = wall-down, 1 = floor-radial)
    float4 drip;     // x = seed, y = fingerAmp, z = fingerFreq, w = feather
};

static inline float pf_hash(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }

// SMOOTH value noise — hash at integer lattice points, smoothstep-interpolated
// between. Using floor()'d hashes directly (a step function per column) is what made
// the front read as hard stepped angles; interpolating gives a continuous liquid edge.
static inline float pf_vnoise(float x, float seed) {
    float i = floor(x), f = fract(x);
    float w = f * f * (3.0 - 2.0 * f);
    return mix(pf_hash(i + seed), pf_hash(i + 1.0 + seed), w);
}

kernel void paintFlood(texture2d<float, access::read_write> tex [[texture(0)]],
                       constant FloodUniforms& u [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint W = tex.get_width(), H = tex.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);

    // Per-column drip "finger" lead from SMOOTH fractal value noise so the leading
    // edge undulates continuously (no column steps). Three octaves give broad lobes +
    // finer ripples; a cubed sparse octave biases toward occasional long thin tendrils
    // (drips that outrun the sheet). lead ∈ ~[0, fingerAmp].
    float f = uv.x * u.drip.z;
    float s = u.drip.x;
    float broad = pf_vnoise(f,        s);
    float fine  = pf_vnoise(f * 2.7,  s + 11.0);
    float tend  = pf_vnoise(f * 0.6,  s + 23.0);
    float lead = (0.50 * broad + 0.22 * fine + 0.50 * tend * tend * tend) * u.drip.y;

    // Front coordinate: how "deep" this texel is along the flow. Columns with a
    // bigger lead are reached earlier (their fc is smaller).
    float fc = (u.front.w < 0.5)
             ? (uv.y - lead)                       // wall: flow DOWN in +v
             : (abs(uv.y - u.front.z) - lead);     // floor: spread OUT from the seam row

    // Write only the band the front crossed THIS frame: (prevPos, curPos]. Each
    // texel is written exactly once (the frame the front reaches it), so dabs balls
    // lay on already-covered texels survive. FULL replacement — a wet paint front has
    // a defined leading edge; its organic raggedness comes from the per-column finger
    // lead above (and the drip agents seeded ahead of it), not a per-texel fade,
    // which — with write-once — would leave 15% of the old colour showing forever.
    if (fc <= u.front.x || fc > u.front.y) return;
    float4 prev = tex.read(gid);
    tex.write(float4(u.color.rgb, max(prev.a, u.color.w)), gid);
}

// ── Front-profile measurement (instrumentation) ───────────────────────────────
// Per column x, the DEEPEST v (0…1) covered by the target colour — i.e. how far the
// flood front has descended in that column. Used by the repaint probe to quantify
// the leading-edge geometry: a blocky (column-quantised) front shows long flat runs
// with sharp jumps between them; a smooth liquid front changes gradually. -1 = none.
kernel void paintFrontProfile(texture2d<float, access::read> tex [[texture(0)]],
                              constant float4& target [[buffer(0)]],   // rgb + tol
                              device float* profile [[buffer(1)]],
                              uint x [[thread_position_in_grid]]) {
    uint W = tex.get_width(), H = tex.get_height();
    if (x >= W) return;
    float deepest = -1.0;
    for (uint y = 0; y < H; y++) {
        float3 c = tex.read(uint2(x, y)).rgb;
        if (distance(c, target.rgb) <= target.w) deepest = max(deepest, float(y) / float(H));
    }
    profile[x] = deepest;
}

// ── Coverage measurement (instrumentation) ────────────────────────────────────
// Counts texels whose RGB matches a target colour within a tolerance. counter[0] =
// matches, counter[1] = total. Used by the repaint probe to prove the wave reaches
// 100% coverage and to time it; never on the per-frame render path.
kernel void paintCoverage(texture2d<float, access::read> tex [[texture(0)]],
                          constant float4& target [[buffer(0)]],   // rgb + tol in w
                          device atomic_uint* counter [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
    atomic_fetch_add_explicit(&counter[1], 1u, memory_order_relaxed);
    float3 c = tex.read(gid).rgb;
    if (distance(c, target.rgb) <= target.w)
        atomic_fetch_add_explicit(&counter[0], 1u, memory_order_relaxed);
}
