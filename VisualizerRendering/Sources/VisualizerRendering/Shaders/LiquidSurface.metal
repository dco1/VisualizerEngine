#include <metal_stdlib>
using namespace metal;

// ── LIQUID SURFACE (LiquidSurfaceRenderer.swift) ───────────────────────────
//
// A simulated liquid's particles drawn as ONE continuous surface, in screen space (van der Laan, Green & Sainz 2009, "Screen
// Space Fluid Rendering with Curvature Flow"; the smoothing is Truong & Yuksel 2018, "A Narrow-Range Filter for Screen-Space
// Fluid Rendering"). The same passes draw the stream in the air, the sheet and droplets of a splash and the pool it fills:
//
//   1. liquidSplatVS + liquidDepthFS      every live particle as a sphere (stretched along its flight by what the liquid has
//                                         stretched, and by the shutter): the nearest surface's view depth, stopped by the scene's.
//   2. liquidSplatVS + liquidThicknessFS  the same spheres added up: how much liquid the eye looks through, and how much of that
//                                         is aerated (white water).
//   3. liquidSmooth (×N, separable)       the depth smoothed into a surface without bridging to what is behind it or in front.
//   4. liquidThicknessBlur                the thickness, softened.
//   5. liquidCompositeFS                  normals off the surface; the scene behind refracted through it and absorbed by its depth
//                                         (Beer–Lambert), the room reflected by Fresnel, the lights' highlights, the foam.
//
// Particle layout: `MLSParticle` (MLSMPM.metal), 144 bytes.

struct LSParticle {
    float4   positionMass;
    float4   velocityJp;
    float3x3 C;
    float3x3 F;
    float4   misc;           // x = density, w = aeration
};

struct LSUniforms {
    float4x4 view;           // world → view
    float4x4 proj;
    float4x4 invProj;
    float4x4 invView;
    float4x4 worldFromSim;
    float4x4 simFromView;
    float4 viewport;         // xy = this pass's target size (px), zw = 1 / size
    float4 particle;         // x = radius (world m), y = most a sphere stretches, z = shutter (s), w = rest density
    float4 absorption;       // rgb = absorption (1/m), w = thickness scale
    float4 scatter;          // rgb = the colour the body scatters back, w = how much (1/m)
    float4 optics;           // x = F0, y = refraction strength, z = environment gain, w = 1: an environment map is bound
    float4 foam;             // rgb = foam colour, w = amount
    float4 filter;           // x = sigma (world m), y = depth window (m), z = ambient light, w = spray whiteness
    float4 envRot;           // x = environment yaw (rad), y = reflection highlight cap, z = edge roundness, w = 1 / the simulation's distance
    float4 clip;             // x = 1: on, y = the height (sim) under which nothing is drawn, z = the thickness the silhouette is drawn at (m), w = this pass's sphere scale
    float4 lightPos[4];      // xyz = position (w = 1) or direction to the light (w = 0), world
    float4 lightCol[4];      // rgb
    uint4  counts;           // x = lights
};

constexpr sampler lsNearest(filter::nearest, address::clamp_to_edge, coord::normalized);
constexpr sampler lsLinear(filter::linear, address::clamp_to_edge, coord::normalized);
constexpr sampler lsEnv(filter::linear, mip_filter::linear, s_address::repeat, t_address::clamp_to_edge);

struct LSVOut {
    float4 position [[position]];
    float2 c;                // the sphere's disc, −1…1
    float3 vpos;             // view position of this point of the disc
    float  radius [[flat]];
    float  foam [[flat]];
};

vertex LSVOut liquidSplatVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                            const device LSParticle* particles [[buffer(0)]],
                            constant LSUniforms& U [[buffer(1)]])
{
    LSVOut o;
    LSParticle p = particles[iid];
    if (p.positionMass.w <= 0.0f) {            // dead
        o.position = float4(2, 2, 2, 1); o.c = float2(0); o.vpos = float3(0); o.radius = 0; o.foam = 0;
        return o;
    }
    float2 q = float2(float(vid & 1u), float(vid >> 1u)) * 2.0f - 1.0f;
    float3 c = (U.view * (U.worldFromSim * float4(p.positionMass.xyz, 1.0f))).xyz;
    float3 v = (U.view * (U.worldFromSim * float4(p.velocityJp.xyz, 0.0f))).xyz;
    float  r = U.particle.x * U.clip.w;
    // Under the clip height it is someone else's to draw (a pool with a surface of its own): the spheres are cut by that plane
    // in the fragment stage (a clean waterline, not a string of shrinking beads); one wholly under it is not drawn at all.
    if (U.clip.x > 0.5f && p.positionMass.y < U.clip.y - 1.5f * r) {
        o.position = float4(2, 2, 2, 1); o.c = float2(0); o.vpos = float3(0); o.radius = 0; o.foam = 0;
        return o;
    }
    float  speed = length(v), flat2 = length(v.xy);
    float2 dir = flat2 > 1e-4f ? v.xy / flat2 : float2(1.0f, 0.0f);
    // Along its flight a sphere is as long as the liquid has stretched (J: a falling stream's particles draw apart, and what
    // joins them is liquid) and as the shutter smears it.
    float  inPlane = speed > 1e-4f ? flat2 / speed : 0.0f;
    float  s = 1.0f + inPlane * (clamp(p.velocityJp.w - 1.0f, 0.0f, 1.5f) + speed * U.particle.z / (2.0f * r));
    s = min(s, U.particle.y);
    float2 off = dir * (q.x * r * s) + float2(-dir.y, dir.x) * (q.y * r);
    o.vpos = c + float3(off, 0.0f);
    o.position = U.proj * float4(o.vpos, 1.0f);
    o.c = q;
    o.radius = r;
    // White water: the air it has taken in, and spray (a lone fast drop is a cloud of finer ones).
    float dens = p.misc.x / max(U.particle.w, 1e-3f);
    float spray = smoothstep(0.45f, 0.12f, dens) * smoothstep(0.4f, 1.6f, speed) * U.filter.w;
    o.foam = saturate(p.misc.w + spray);
    return o;
}

// Spray: `FoamParticle`s (FoamSolver), small spheres in the same surface.
struct LSDroplet { float4 positionLife; float4 velocityAge; };

inline float ls_hash(uint v) {
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return float((v >> 22u) ^ v) * (1.0f / 4294967296.0f);
}

vertex LSVOut liquidDropletVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                              const device LSDroplet* drops [[buffer(0)]],
                              constant LSUniforms& U [[buffer(1)]])
{
    LSVOut o;
    LSDroplet p = drops[iid];
    if (p.positionLife.w <= 0.0f) {
        o.position = float4(2, 2, 2, 1); o.c = float2(0); o.vpos = float3(0); o.radius = 0; o.foam = 0;
        return o;
    }
    float2 q = float2(float(vid & 1u), float(vid >> 1u)) * 2.0f - 1.0f;
    float3 c = (U.view * (U.worldFromSim * float4(p.positionLife.xyz, 1.0f))).xyz;
    float3 v = (U.view * (U.worldFromSim * float4(p.velocityAge.xyz, 0.0f))).xyz;
    // Sizes from a fine drop to a fat one, and a drop dwindles as it dies.
    float  r = U.lightCol[3].w * (0.45f + 0.9f * ls_hash(iid)) * smoothstep(0.0f, 0.12f, p.positionLife.w) * U.clip.w;
    float  speed = length(v), flat2 = length(v.xy);
    float2 dir = flat2 > 1e-4f ? v.xy / flat2 : float2(1.0f, 0.0f);
    float  s = min(1.0f + flat2 * U.particle.z / (2.0f * max(r, 1e-5f)), U.particle.y);
    float2 off = dir * (q.x * r * s) + float2(-dir.y, dir.x) * (q.y * r);
    o.vpos = c + float3(off, 0.0f);
    o.position = U.proj * float4(o.vpos, 1.0f);
    o.c = q;
    o.radius = r;
    o.foam = 0.0f;
    return o;
}

inline bool ls_clipped(float3 vp, constant LSUniforms& U) {
    return U.clip.x > 0.5f && (U.simFromView * float4(vp, 1.0f)).y < U.clip.y;
}

struct LSDepthOut {
    float z [[color(0)]];
    float depth [[depth(any)]];
};

fragment LSDepthOut liquidDepthFS(LSVOut in [[stage_in]], constant LSUniforms& U [[buffer(0)]],
                                  depth2d<float> sceneDepth [[texture(0)]])
{
    float r2 = dot(in.c, in.c);
    if (r2 > 1.0f) discard_fragment();
    float3 vp = in.vpos + float3(0.0f, 0.0f, sqrt(1.0f - r2) * in.radius);
    if (ls_clipped(vp, U)) discard_fragment();
    float4 clip = U.proj * float4(vp, 1.0f);
    float d = clip.z / clip.w;
    if (d > sceneDepth.sample(lsNearest, in.position.xy * U.viewport.zw)) discard_fragment();
    LSDepthOut o;
    o.z = -vp.z;
    o.depth = d;
    return o;
}

fragment float4 liquidThicknessFS(LSVOut in [[stage_in]], constant LSUniforms& U [[buffer(0)]],
                                  depth2d<float> sceneDepth [[texture(0)]])
{
    float r2 = dot(in.c, in.c);
    if (r2 > 1.0f) discard_fragment();
    float nz = sqrt(1.0f - r2);
    float3 vp = in.vpos + float3(0.0f, 0.0f, nz * in.radius);
    if (ls_clipped(in.vpos, U)) discard_fragment();
    float4 clip = U.proj * float4(vp, 1.0f);
    if (clip.z / clip.w > sceneDepth.sample(lsNearest, in.position.xy * U.viewport.zw)) discard_fragment();
    float t = 2.0f * nz * in.radius;
    return float4(t, t * in.foam, 0.0f, 0.0f);
}

// One pass of the narrow-range filter along `axis`. A neighbour well in FRONT of the centre is another layer (skipped); one
// well BEHIND counts as a little behind (so a near surface is not dragged back to a far one, and its edge still rounds away).
kernel void liquidSmooth(texture2d<float, access::read>  src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         constant LSUniforms& U [[buffer(0)]],
                         constant int2& axis [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]])
{
    int2 size = int2(src.get_width(), src.get_height());
    if (any(int2(gid) >= size)) return;
    float z = src.read(gid).x;
    if (z <= 0.0f) { dst.write(float4(0.0f), gid); return; }
    float pxPerM = U.proj[1][1] * U.viewport.y * 0.5f / z;
    float sigma = clamp(U.filter.x * pxPerM, 0.75f, 40.0f);
    const int taps = 20;
    float stride = max(1.0f, 2.5f * sigma / float(taps));
    float window = U.filter.y, behind = U.filter.y * 0.35f, edge = U.envRot.z;
    float sum = z, wsum = 1.0f;
    // Where the taps are more than a pixel apart they are jittered, a different phase each pixel and pass: what would be
    // bands is noise, and the next round smooths it.
    float phase = (stride > 1.0f && U.counts.z == 0u) ? ls_hash(gid.x * 1973u + gid.y * 9277u + uint(axis.x) * 26699u + U.counts.y * 7919u) - 0.5f : 0.0f;
    for (int i = 1; i <= taps; i++) {
        float d = (float(i) + phase) * stride;
        float w = exp(-d * d / (2.0f * sigma * sigma));
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            int2 q = int2(gid) + axis * int(round(d)) * sgn;
            float zj = (any(q < 0) || any(q >= size)) ? 0.0f : src.read(uint2(q)).x;
            if (zj <= 0.0f) {                      // nothing there: the surface turns away toward its edge
                sum += w * edge * (z + behind); wsum += w * edge;
                continue;
            }
            if (zj < z - window) continue;
            if (zj > z + window) zj = z + behind;
            sum += w * zj; wsum += w;
        }
    }
    dst.write(float4(sum / wsum, 0.0f, 0.0f, 0.0f), gid);
}

kernel void liquidThicknessBlur(texture2d<float, access::read>  src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                constant LSUniforms& U [[buffer(0)]],
                                constant int2& axis [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]])
{
    int2 size = int2(src.get_width(), src.get_height());
    if (any(int2(gid) >= size)) return;
    // A fixed reach, the particle's size at a metre scaled by this target's height: thickness has no depth to scale by.
    float sigma = clamp(U.filter.x * U.proj[1][1] * U.viewport.y * 0.5f * U.envRot.w, 1.0f, 24.0f);
    const int taps = 8;
    float stride = max(1.0f, 2.0f * sigma / float(taps));
    float4 sum = src.read(gid);
    float wsum = 1.0f;
    for (int i = 1; i <= taps; i++) {
        float d = float(i) * stride, w = exp(-d * d / (2.0f * sigma * sigma));
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            int2 q = clamp(int2(gid) + axis * int(round(d)) * sgn, int2(0), size - 1);
            sum += w * src.read(uint2(q)); wsum += w;
        }
    }
    dst.write(sum / wsum, gid);
}

struct LSQuad { float4 position [[position]]; float2 uv; };

vertex LSQuad liquidQuadVS(uint id [[vertex_id]]) {
    float2 c = float2((id << 1) & 2, id & 2);
    return LSQuad{ float4(c * 2.0f - 1.0f, 0.0f, 1.0f), float2(c.x, 1.0f - c.y) };
}

inline float3 ls_view_pos(float2 uv, float z, constant LSUniforms& U) {
    float4 h = U.invProj * float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 1.0f, 1.0f);
    float3 d = h.xyz / h.w;
    return d * (z / -d.z);
}

inline float3 ls_environment(float3 dirWorld, float lod, constant LSUniforms& U, texture2d<float> env) {
    if (U.optics.w < 0.5f) {
        // No map: a dim room, lighter overhead.
        return float3(0.02f) + float3(0.10f) * smoothstep(0.2f, 1.0f, dirWorld.y);
    }
    // The engine's equirectangular convention (IlluminatoramaRT.metal `equirect_uv`).
    float a = atan2(dirWorld.z, dirWorld.x) + U.envRot.x;
    float2 uv = float2(a / 6.2831853f, acos(clamp(dirWorld.y, -1.0f, 1.0f)) / 3.14159265f);
    return env.sample(lsEnv, uv, level(lod)).rgb;
}

fragment float4 liquidCompositeFS(LSQuad in [[stage_in]], constant LSUniforms& U [[buffer(0)]],
                                  texture2d<float> depthT [[texture(0)]],
                                  texture2d<float> thickT [[texture(1)]],
                                  texture2d<float> scene [[texture(2)]],
                                  depth2d<float> sceneDepth [[texture(3)]],
                                  texture2d<float> env [[texture(4)]])
{
    int2 size = int2(depthT.get_width(), depthT.get_height());
    int2 ip = clamp(int2(in.position.xy), int2(0), size - 1);
    float z = depthT.read(uint2(ip)).x;
    if (z <= 0.0f) discard_fragment();
    float2 uv = in.uv, px = U.viewport.zw;
    auto at = [&](int2 o) { int2 q = clamp(ip + o, int2(0), size - 1); return depthT.read(uint2(q)).x; };
    float zl = at(int2(-1, 0)), zr = at(int2(1, 0)), zu = at(int2(0, -1)), zd = at(int2(0, 1));
    float3 P = ls_view_pos(uv, z, U);
    // The smaller step each way: across a silhouette the other one jumps to another surface.
    float3 dx, dy;
    bool hasL = zl > 0.0f, hasR = zr > 0.0f, hasU = zu > 0.0f, hasD = zd > 0.0f;
    if (hasR && (!hasL || fabs(zr - z) < fabs(z - zl))) dx = ls_view_pos(uv + float2(px.x, 0), zr, U) - P;
    else if (hasL) dx = P - ls_view_pos(uv - float2(px.x, 0), zl, U);
    else dx = ls_view_pos(uv + float2(px.x, 0), z, U) - P;
    if (hasD && (!hasU || fabs(zd - z) < fabs(z - zu))) dy = ls_view_pos(uv + float2(0, px.y), zd, U) - P;
    else if (hasU) dy = P - ls_view_pos(uv - float2(0, px.y), zu, U);
    else dy = ls_view_pos(uv + float2(0, px.y), z, U) - P;
    float3 n = normalize(cross(dy, dx));
    float3 V = normalize(-P);
    if (dot(n, V) < 0.0f) n = -n;

    float4 th = thickT.sample(lsLinear, uv);
    float thick = max(th.x, 0.0f) * U.absorption.w;
    float aer = th.x > 1e-5f ? saturate(th.y / th.x) : 0.0f;

    float NoV = saturate(dot(n, V));
    float fres = U.optics.x + (1.0f - U.optics.x) * pow(1.0f - NoV, 5.0f);

    // What is behind, bent by the surface (a body of water is a lens: it shows what is beyond its far side) and stopped if
    // the bent look would land on something in front of the water.
    float2 ruv = uv + float2(-n.x, n.y) * (U.optics.y * min(thick, 0.08f) / max(z, 0.05f));
    float sd = sceneDepth.sample(lsNearest, ruv);
    float4 sv = U.invProj * float4(ruv.x * 2.0f - 1.0f, 1.0f - ruv.y * 2.0f, sd, 1.0f);
    if (sd < 1.0f && -sv.z / sv.w < z - 0.01f) ruv = uv;
    float3 behind = scene.sample(lsLinear, ruv).rgb;

    // Light on the liquid's body and foam: the lamps, wrapped (light goes into it), and the room.
    float3 lit = float3(U.filter.z);
    float3 spec = float3(0.0f);
    float3 Pw = (U.invView * float4(P, 1.0f)).xyz;
    for (uint i = 0; i < U.counts.x; i++) {
        float3 Lw = U.lightPos[i].w > 0.5f ? normalize(U.lightPos[i].xyz - Pw) : normalize(U.lightPos[i].xyz);
        float3 L = (U.view * float4(Lw, 0.0f)).xyz;
        float wrap = saturate(0.5f + 0.5f * dot(n, L));
        lit += U.lightCol[i].rgb * wrap * wrap;
        float3 H = normalize(L + V);
        float nh = saturate(dot(n, H));
        spec += U.lightCol[i].rgb * (pow(nh, 900.0f) * 6.0f + pow(nh, 90.0f) * 0.25f) * saturate(dot(n, L) * 4.0f + 0.2f);
    }
    float3 through = behind * exp(-U.absorption.rgb * thick)
                   + U.scatter.rgb * lit * (1.0f - exp(-U.scatter.w * thick));

    float3 Rw = (U.invView * float4(reflect(-V, n), 0.0f)).xyz;
    float3 refl = ls_environment(Rw, 0.0f, U, env) * U.optics.z;
    refl = min(refl, float3(U.envRot.y));
    float3 c = mix(through, refl, fres) + min(spec, float3(U.envRot.y));

    // White water: many small bubbles and drops, so matte and bright by the light on it.
    // …and bubbles need water to be in: a film a couple of millimetres deep is clear whatever struck it.
    float white = saturate(aer * U.foam.w) * smoothstep(0.003f, 0.02f, th.x);
    c = mix(c, U.foam.rgb * (0.25f + 0.75f * lit), white);

    // Coverage at the silhouette (the surface is drawn after the scene's antialiasing: its own edge needs one).
    // The depth pass draws its spheres a little large; the outline is where the (blurred) thickness reaches `clip.z`: one smooth
    // contour round a body of particles, not the scalloped union of their discs.
    float cover = (1.0f + float(hasL) + float(hasR) + float(hasU) + float(hasD)) / 5.0f;
    float alpha = smoothstep(0.3f, 0.9f, cover) * smoothstep(0.55f * U.clip.z, U.clip.z, th.x);
    return float4(c, alpha);
}
