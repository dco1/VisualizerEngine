#include <metal_stdlib>
using namespace metal;

// ── ILLUMINATORAMA CRYSTAL FLAKES ────────────────────────────────────────────
//
// The near-camera snow: a small pool of flakes drawn as REAL ice-crystal shapes, not dots. Each is a
// camera-independent quad lying in the flake's own plane (it tumbles as it falls), shaded in the
// fragment shader by a signed-distance snowflake with true six-fold symmetry:
//
//   kind 0  stellar dendrite   six main arms, three paired side branches each
//   kind 1  fernlike dendrite  six arms, five finer branch pairs (the big feathery ones)
//   kind 2  hexagonal plate    outer hexagon, an inner hexagon, six spokes
//   kind 3  broad-arm star     six tapered, wide arms with a short branch
//
// Ice is translucent: the outline and branches read bright, the body dim. A tumbling facet also
// catches the light as a brief GLINT (specular off the flake's face against the moon). The nearest
// flakes are OUT OF FOCUS — they open into soft, bright bokeh discs, as in a real night photograph.
//
// Two kernels share one 64-byte struct: `illumi_crystal_step` advances the pool (inertial fall under
// gusting wind + flutter + a pendulum tumble, recycled inside the camera's frustum volume), and
// `illumi_crystal_vs/fs` draw it as additive HDR quads, depth-tested against the scene.

struct CrystalFlake {
    float4 posR;     // xyz world position, w = radius (m)
    float4 velAge;   // xyz velocity, w = age (s; negative = not yet born)
    float4 nrmSpin;  // xyz unit plane normal, w = spin angle about it
    float4 colKind;  // rgb lit colour (already faded), w = kind + seed (fractional)
};

struct CrystalStepUniforms {
    float dt; float time; uint count; float relaxTau;
    float4 camPos;       // xyz, w = far distance of the pool
    float4 camFwd;       // xyz forward, w = near distance
    float4 camRightT;    // xyz right * tan(fovx/2)
    float4 camUpT;       // xyz up    * tan(fovy/2)
    float4 wind;         // xyz m/s, w = gust depth
    float4 dyn;          // x mean fall speed, y turbulence, z flutter, w tumble amplitude (rad)
    float4 size;         // x min radius, y max radius, z ground y, w fade-in time
    float4 moon;         // rgb moonlit flake radiance
    float4 lightPos[8];
    float4 lightCol[8];
};

static inline float chash(float3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
static inline float cnoise(float3 x) {
    float3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float a = chash(i), b = chash(i + float3(1,0,0)), c = chash(i + float3(0,1,0)), d = chash(i + float3(1,1,0));
    float e = chash(i + float3(0,0,1)), g = chash(i + float3(1,0,1)), h = chash(i + float3(0,1,1)), k = chash(i + float3(1,1,1));
    return mix(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), mix(mix(e, g, f.x), mix(h, k, f.x), f.y), f.z) * 2.0 - 1.0;
}
static inline float3 cnoise3(float3 x) {
    return float3(cnoise(x), cnoise(x + float3(31.4, 17.2, 47.1)), cnoise(x + float3(-19.3, 53.7, 11.9)));
}

kernel void illumi_crystal_step(
    device CrystalFlake*           flakes [[buffer(0)]],
    constant CrystalStepUniforms&  u      [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.count) return;
    CrystalFlake f = flakes[gid];
    float3 p = f.posR.xyz;
    float3 v = f.velAge.xyz;
    float age = f.velAge.w + u.dt;
    float seed = fract(f.colKind.w);
    float kind = floor(f.colKind.w);

    // ── Recycle: outside the camera's pool volume, or on the ground → respawn inside it ──
    float3 rel = p - u.camPos.xyz;
    float depth = dot(rel, u.camFwd.xyz);
    float ndcX = dot(rel, u.camRightT.xyz) / max(1e-3, dot(u.camRightT.xyz, u.camRightT.xyz) * max(depth, 0.05));
    float ndcY = dot(rel, u.camUpT.xyz)    / max(1e-3, dot(u.camUpT.xyz, u.camUpT.xyz) * max(depth, 0.05));
    bool outside = depth < u.camFwd.w * 0.75 || depth > u.camPos.w || fabs(ndcX) > 1.35 || fabs(ndcY) > 1.35
                   || p.y < u.size.z + 0.02 || age < -1.0 || !all(isfinite(p));
    if (outside) {
        float3 r = float3(chash(float3(gid, u.time * 7.1, 1.0)), chash(float3(gid, u.time * 3.7, 2.0)),
                          chash(float3(gid, u.time * 5.3, 3.0)));
        float d = mix(u.camFwd.w, u.camPos.w, r.x * r.x);                 // denser near the lens
        float sx = (r.y * 2.0 - 1.0) * 1.15, sy = (r.z * 2.0 - 1.0) * 1.15;
        p = u.camPos.xyz + u.camFwd.xyz * d + u.camRightT.xyz * (sx * d) + u.camUpT.xyz * (sy * d);
        p.y = max(p.y, u.size.z + 0.15);
        v = float3(0.0, -0.4, 0.0);
        seed = chash(float3(gid, u.time * 11.3, 9.0));
        kind = floor(chash(float3(gid, u.time * 2.9, 4.0)) * 4.0);
        age = -(chash(float3(gid, u.time * 13.7, 5.0)) * 0.6);             // staggered births
    }
    float h1 = chash(float3(seed * 91.7, 3.1, 7.7));
    float h2 = chash(float3(seed * 17.3, 11.9, 5.3));

    // ── Motion: inertial fall (big flakes are slow), gusting wind, curl turbulence, flutter ──
    float fall = u.dyn.x * (0.55 + 0.75 * h1);
    float gust = cnoise(p * 0.12 + float3(0.0, 0.0, u.time * 0.35));
    float3 wind = u.wind.xyz * (1.0 + u.wind.w * gust);
    float3 q = p * 0.6 + float3(0.0, u.time * 0.5, 0.0);
    const float e = 0.15;
    float3 dx = float3(e,0,0), dy = float3(0,e,0), dz = float3(0,0,e);
    float3 px0 = cnoise3(q - dx), px1 = cnoise3(q + dx), py0 = cnoise3(q - dy), py1 = cnoise3(q + dy);
    float3 pz0 = cnoise3(q - dz), pz1 = cnoise3(q + dz);
    float3 curl = float3((py1.z - py0.z) - (pz1.y - pz0.y), (pz1.x - pz0.x) - (px1.z - px0.z),
                         (px1.y - px0.y) - (py1.x - py0.x)) / (2.0 * e);
    float ph = u.time * (1.6 + 2.2 * h2) + seed * 6.2831;
    float3 flutter = float3(cos(ph), 0.2 * sin(ph * 1.7), sin(ph * 0.83)) * u.dyn.z * (0.4 + 0.8 * h2);
    float3 target = wind + curl * u.dyn.y + flutter + float3(0.0, -fall, 0.0);
    v += (target - v) * (1.0 - exp(-u.dt / max(u.relaxTau, 1e-3)));
    p += v * u.dt;

    // ── Orientation: a falling plate rocks like a pendulum while it spins about its own normal ──
    float theta = u.dyn.w * sin(u.time * (1.1 + 1.6 * h1) + seed * 17.0);
    float phi = u.time * (0.4 + 0.5 * h2) + seed * 6.2831;
    float3 n = float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    float spin = f.nrmSpin.w + u.dt * (0.7 + 2.4 * h1) * (h2 < 0.5 ? -1.0 : 1.0);

    // ── Light it: moon + the scene's warm lights (same model as the far snow) ──
    float3 lit = u.moon.rgb;
    for (int i = 0; i < 8; ++i) {
        float3 L = u.lightPos[i].xyz - p;
        lit += u.lightCol[i].rgb / (dot(L, L) + 0.15);
    }
    float fade = saturate(age / max(u.size.w, 1e-3));
    float radius = mix(u.size.x, u.size.y, seed * seed);

    f.posR = float4(p, radius);
    f.velAge = float4(v, age);
    f.nrmSpin = float4(n, spin);
    f.colKind = float4(lit * fade, kind + seed * 0.999);
    flakes[gid] = f;
}

// ── Draw ─────────────────────────────────────────────────────────────────────

struct CrystalDrawParams {
    float4 camPos;    // xyz
    float4 moonDir;   // xyz TOWARD the moon
    float4 glint;     // rgb glint colour, w = glint strength
    float4 misc;      // x intensity, y bokeh-full distance (m), z sharp distance (m), w bokeh disc strength
};

struct CrystalVSOut {
    float4 position [[position]];
    float2 uv;
    float3 color;
    float  kind;
    float  glint;
    float  blur;
};

vertex CrystalVSOut illumi_crystal_vs(
    uint vid [[vertex_id]],
    const device CrystalFlake* flakes [[buffer(0)]],
    constant float4x4& viewProjection [[buffer(2)]],
    constant CrystalDrawParams& dp [[buffer(3)]])
{
    const float2 corner[6] = { float2(-1,-1), float2(1,-1), float2(1,1), float2(-1,-1), float2(1,1), float2(-1,1) };
    CrystalFlake f = flakes[vid / 6];
    float2 q = corner[vid % 6];

    float3 n = normalize(f.nrmSpin.xyz);
    float3 ref = fabs(n.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(ref, n));
    float3 b = cross(n, t);
    float s = f.nrmSpin.w, cs = cos(s), sn = sin(s);
    float3 tt = t * cs + b * sn, bb = -t * sn + b * cs;

    float3 pos = f.posR.xyz;
    float dist = length(pos - dp.camPos.xyz);
    float blur = smoothstep(dp.misc.z, dp.misc.y, dist);            // 1 when very near the lens
    float grow = 1.0 + 0.7 * blur;                                  // a defocused flake blooms wider
    float3 world = pos + (tt * q.x + bb * q.y) * f.posR.w * grow;

    CrystalVSOut o;
    o.position = viewProjection * float4(world, 1.0);
    o.uv = q * grow;
    o.color = f.colKind.rgb;
    o.kind = floor(f.colKind.w);
    // A facet catches the moon: a tight lobe of the half-vector against the flake's (two-sided) face.
    float3 V = normalize(dp.camPos.xyz - pos);
    float3 H = normalize(dp.moonDir.xyz + V);
    o.glint = pow(saturate(fabs(dot(n, H))), 90.0);
    o.blur = blur;
    return o;
}

static inline float sdSeg(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = saturate(dot(pa, ba) / dot(ba, ba));
    return length(pa - ba * h);
}

fragment float4 illumi_crystal_fs(
    CrystalVSOut in [[stage_in]],
    constant CrystalDrawParams& dp [[buffer(3)]])
{
    float2 p = in.uv;
    float r = length(p);
    if (r > 2.6) discard_fragment();

    // Fold into one 30° wedge about an arm axis: six-fold mirror symmetry for free.
    float a = atan2(p.y, p.x);
    float wedge = fmod(a + 6.2831853 + 0.5235988, 1.0471976) - 0.5235988;
    float2 q = float2(cos(wedge), fabs(sin(wedge))) * r;

    const float2 dir60 = float2(0.5, 0.8660254);
    float kind = in.kind;
    float d;
    float fill = 0.0;
    if (kind < 0.5) {                                                // stellar dendrite
        d = sdSeg(q, float2(0.0), float2(0.92, 0.0)) - 0.040;
        for (int i = 0; i < 3; ++i) {
            float tb = 0.30 + 0.21 * float(i);
            float len = 0.34 * (1.0 - tb) + 0.10;
            d = min(d, sdSeg(q, float2(tb, 0.0), float2(tb, 0.0) + dir60 * len) - 0.030);
        }
        d = min(d, r - 0.13);
    } else if (kind < 1.5) {                                         // fernlike dendrite
        d = sdSeg(q, float2(0.0), float2(0.96, 0.0)) - 0.034;
        for (int i = 0; i < 5; ++i) {
            float tb = 0.18 + 0.13 * float(i);
            float len = 0.30 * (1.0 - tb) + 0.07;
            d = min(d, sdSeg(q, float2(tb, 0.0), float2(tb, 0.0) + dir60 * len) - 0.022);
        }
        d = min(d, r - 0.10);
    } else if (kind < 2.5) {                                         // hexagonal plate
        float hex = dot(q, float2(0.8660254, 0.5)) - 0.66;            // outer hexagon edge
        float hin = dot(q, float2(0.8660254, 0.5)) - 0.36;            // inner hexagon
        d = min(fabs(hex), fabs(hin)) - 0.034;
        d = min(d, sdSeg(q, float2(0.0), float2(0.62, 0.0)) - 0.026);   // spokes
        fill = (hex < 0.0) ? 0.07 : 0.0;                              // a faint translucent body
    } else {                                                         // broad-arm star
        float wArm = mix(0.115, 0.030, saturate(q.x / 0.95));
        d = (fabs(q.y) - wArm) ;                                       // tapered arm
        d = max(d, q.x - 0.95);
        d = min(d, sdSeg(q, float2(0.50, 0.0), float2(0.50, 0.0) + dir60 * 0.26) - 0.030);
        d = min(d, r - 0.17);
    }

    float aa = max(fwidth(d), 0.004);
    float shape = (1.0 - smoothstep(-aa, aa, d)) + fill * (1.0 - smoothstep(-aa, aa, d + 0.04) * 0.0);

    // Out-of-focus: the crystal dissolves into a bright, soft-edged disc (bokeh), brighter at its rim.
    float disc = (1.0 - smoothstep(0.78, 1.0, r)) * (0.55 + 0.45 * smoothstep(0.55, 0.95, r));
    float m = mix(shape, disc * dp.misc.w, in.blur * 0.45);   // a crystal never fully becomes a disc

    // A gentle inner glow along the lines so thin arms don't alias to nothing at distance.
    float halo = exp(-max(d, 0.0) * 14.0) * 0.18 * (1.0 - in.blur);
    float3 col = in.color * (m + halo) * dp.misc.x;
    col += dp.glint.rgb * dp.glint.w * in.glint * m * 4.0;
    return float4(col, saturate(m));
}
