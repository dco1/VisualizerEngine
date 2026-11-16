//  Modifier+GPUKernels.swift — the Metal source for the GPU modifiers (ticket item M2).
//
//  One library, compiled from this string at first use (this target has no resources; same pattern as SDF+GPU.swift).
//  Every kernel is one thread per mesh vertex (or welded vertex / triangle / rim edge) in Float; the CPU modifiers in
//  this folder are the oracle. The 64-bit integer hash of `ProceduralNoise.value` is run bit-for-bit (Metal has
//  ulong), only the lattice position and the interpolation are Float.

#if canImport(Metal)
enum ModifierKernels {
    static let common = """
    #include <metal_stdlib>
    using namespace metal;

    static float angle3(float3 a, float3 b, float3 c) {
        float3 u = b - a, v = c - a; float lu = length(u), lv = length(v);
        if (lu <= 1e-12f || lv <= 1e-12f) return 0;
        return acos(clamp(dot(u, v) / (lu * lv), -1.0f, 1.0f));
    }

    // ---- welded-vertex normals: one thread per welded vertex gathers its incident welded triangles -------------
    struct WP { uint nW; uint even; uint pad0, pad1; };
    kernel void weld_normals(device const float4* pos [[buffer(0)]], device const uint* rep [[buffer(1)]],
                             device const uint* wStart [[buffer(2)]], device const uint* wEnt [[buffer(3)]],
                             device const uint* wtris [[buffer(4)]], device float4* wN [[buffer(5)]],
                             constant WP& P [[buffer(6)]], uint w [[thread_position_in_grid]]) {
        if (w >= P.nW) return;
        float3 acc = float3(0);
        for (uint e = wStart[w]; e < wStart[w + 1]; e++) {
            uint ent = wEnt[e], t = ent / 3, s = ent % 3;
            float3 a = pos[rep[wtris[3 * t]]].xyz, b = pos[rep[wtris[3 * t + 1]]].xyz, c = pos[rep[wtris[3 * t + 2]]].xyz;
            float3 fn = cross(b - a, c - a);
            if (length(fn) <= 1e-14f) continue;
            fn = normalize(fn);
            float ang = s == 0 ? angle3(a, b, c) : (s == 1 ? angle3(b, c, a) : angle3(c, a, b));
            acc += fn * ang;
        }
        float l = length(acc);
        wN[w] = l > 1e-12f ? float4(acc / l, 0) : float4(0, 1, 0, 0);
    }

    // Solidify's even-thickness scale per welded vertex: 1 / max(0.25, mean(dot(face normal, vertex normal))).
    kernel void solid_scale(device const float4* pos [[buffer(0)]], device const uint* rep [[buffer(1)]],
                            device const uint* wStart [[buffer(2)]], device const uint* wEnt [[buffer(3)]],
                            device const uint* wtris [[buffer(4)]], device const float4* wN [[buffer(5)]],
                            device float* scale [[buffer(6)]], constant WP& P [[buffer(7)]],
                            uint w [[thread_position_in_grid]]) {
        if (w >= P.nW) return;
        float sum = 0, cnt = 0;
        if (P.even != 0) {
            for (uint e = wStart[w]; e < wStart[w + 1]; e++) {
                uint t = wEnt[e] / 3;
                float3 a = pos[rep[wtris[3 * t]]].xyz, b = pos[rep[wtris[3 * t + 1]]].xyz, c = pos[rep[wtris[3 * t + 2]]].xyz;
                float3 fn = cross(b - a, c - a);
                if (length(fn) <= 1e-14f) continue;
                sum += dot(normalize(fn), wN[w].xyz); cnt += 1;
            }
        }
        scale[w] = cnt > 0 ? 1.0f / max(0.25f, sum / cnt) : 1.0f;
    }

    // ---- Mesh3.moved(to:): turn each corner normal by the rotation of the first valid face round the vertex ----
    struct TP { uint n; uint pad0, pad1, pad2; };
    kernel void turn_normals(device const float4* oldP [[buffer(0)]], device const float4* newP [[buffer(1)]],
                             device const float4* oldN [[buffer(2)]], device float4* outN [[buffer(3)]],
                             device const uint* idx [[buffer(4)]], device const uint* vStart [[buffer(5)]],
                             device const uint* vTris [[buffer(6)]], constant TP& P [[buffer(7)]],
                             uint v [[thread_position_in_grid]]) {
        if (v >= P.n) return;
        float3 n0 = oldN[v].xyz;
        outN[v] = oldN[v];
        for (uint e = vStart[v]; e < vStart[v + 1]; e++) {
            uint t = vTris[e], i0 = idx[3 * t], i1 = idx[3 * t + 1], i2 = idx[3 * t + 2];
            float3 o = cross(oldP[i1].xyz - oldP[i0].xyz, oldP[i2].xyz - oldP[i0].xyz);
            float3 w = cross(newP[i1].xyz - newP[i0].xyz, newP[i2].xyz - newP[i0].xyz);
            if (length(o) <= 1e-14f || length(w) <= 1e-14f) continue;
            float3 from = normalize(o), to = normalize(w);
            float3 axis = cross(from, to); float s = length(axis), c = dot(from, to);
            float3 r = n0;
            if (s >= 1e-7f) { float3 k = axis / s; r = n0 * c + cross(k, n0) * s + k * (dot(k, n0) * (1 - c)); }
            outN[v] = float4(normalize(r), 0);
            break;
        }
    }

    // ---- Displace, procedural fBm (ProceduralNoise.value / fbm) ------------------------------------------------
    static ulong hash3(ulong seed, long x, long y, long z) {
        ulong h = seed + as_type<ulong>(x) * 0x9E3779B97F4A7C15UL + as_type<ulong>(y) * 0xBF58476D1CE4E5B9UL
                + as_type<ulong>(z) * 0x94D049BB133111EBUL;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
        h ^= h >> 31;
        return h;
    }
    static float hval(ulong seed, long x, long y, long z) { return float(hash3(seed, x, y, z) >> 11) * 1.1102230246251565e-16f; }
    static float noise_value(float3 p, ulong seed) {
        float3 f = floor(p), t = p - f, s = t * t * (3.0f - 2.0f * t);
        long ix = long(f.x), iy = long(f.y), iz = long(f.z);
        float x00 = mix(hval(seed, ix, iy, iz), hval(seed, ix + 1, iy, iz), s.x);
        float x10 = mix(hval(seed, ix, iy + 1, iz), hval(seed, ix + 1, iy + 1, iz), s.x);
        float x01 = mix(hval(seed, ix, iy, iz + 1), hval(seed, ix + 1, iy, iz + 1), s.x);
        float x11 = mix(hval(seed, ix, iy + 1, iz + 1), hval(seed, ix + 1, iy + 1, iz + 1), s.x);
        return mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z);
    }
    struct NP { float4 fixedDir; uint n, dirMode, octaves, seedLo; uint seedHi; float strength, mid, freq; float lac, gain, pad0, pad1; };
    kernel void displace_noise(device const float4* pos [[buffer(0)]], device float4* outP [[buffer(1)]],
                               device const uint* cid [[buffer(2)]], device const uint* rep [[buffer(3)]],
                               device const float4* wN [[buffer(4)]], constant NP& P [[buffer(5)]],
                               uint v [[thread_position_in_grid]]) {
        if (v >= P.n) return;
        uint id = cid[v];
        float3 p = pos[rep[id]].xyz;
        ulong seed = (ulong(P.seedHi) << 32) | ulong(P.seedLo);
        float sum = 0, amp = 1, norm = 0, f = P.freq;
        for (uint o = 0; o < max(1u, P.octaves); o++) {
            sum += amp * noise_value(p * f, seed + ulong(o) * 7919UL); norm += amp;
            amp *= P.gain; f *= P.lac;
        }
        float3 d = P.dirMode == 0 ? wN[id].xyz : P.fixedDir.xyz;
        outP[v] = float4(pos[v].xyz + d * (P.strength * (sum / norm - P.mid)), 0);
    }

    // ---- SimpleDeform --------------------------------------------------------------------------------------------
    struct SDP { float4 origin, w, u, bv, w2; uint mode, n; float angle, lower, upper, factor, radius, pad; };
    kernel void simple_deform(device const float4* pos [[buffer(0)]], device float4* outP [[buffer(1)]],
                              constant SDP& P [[buffer(2)]], uint v [[thread_position_in_grid]]) {
        if (v >= P.n) return;
        float3 p0 = pos[v].xyz, W = P.w.xyz, o = P.origin.xyz;
        float3 r = p0 - o;
        float a = dot(r, W);
        float c = min(max(a, P.lower), P.upper);
        float t = (c - P.lower) / (P.upper - P.lower);
        float3 perp = r - W * a;
        float3 q = p0;
        if (P.mode == 0) {                        // bend
            if (fabs(P.angle) > 1e-9f) {
                float3 V = P.bv.xyz, W2 = P.w2.xyz;
                float b = dot(perp, V), cc = dot(perp, W2);
                float th = (c - P.lower) / P.radius;
                float3 centre = W * (P.lower + P.radius * sin(th)) + V * (P.radius * (1 - cos(th)));
                float3 tangent = W * cos(th) + V * sin(th);
                float3 normal = -W * sin(th) + V * cos(th);
                q = o + centre + normal * b + W2 * cc + tangent * (a - c);
            }
        } else if (P.mode == 1) {                 // twist
            float th = P.angle * t, cs = cos(th), sn = sin(th);
            q = o + W * a + (perp * cs + cross(W, perp) * sn);
        } else if (P.mode == 2) {                 // taper
            q = o + W * a + perp * (1 + P.factor * t);
        } else {                                  // stretch
            float s = 1 + P.factor, k = 1 / sqrt(s);
            q = o + W * (a + (s - 1) * (c - P.lower)) + perp * k;
        }
        outP[v] = float4(q, 0);
    }

    // ---- Lattice ---------------------------------------------------------------------------------------------------
    struct LP { float4 boxMin, size; uint dx, dy, dz, n; uint cubic, pad0, pad1, pad2; };
    static float3 lat_pt(device const float4* P, constant LP& L, int i, int j, int k) {
        return P[uint(i) + L.dx * (uint(j) + L.dy * uint(k))].xyz;
    }
    static float3 lat_cp(device const float4* P, constant LP& L, int i, int j, int k) {
        int nx = int(L.dx), ny = int(L.dy), nz = int(L.dz);
        int ci = i, ni = i, cj = j, nj = j, ck = k, nk = k; bool wi = false, wj = false, wk = false;
        if (i < 0) { ci = 0; ni = 1; wi = true; } else if (i >= nx) { ci = nx - 1; ni = nx - 2; wi = true; }
        if (j < 0) { cj = 0; nj = 1; wj = true; } else if (j >= ny) { cj = ny - 1; nj = ny - 2; wj = true; }
        if (k < 0) { ck = 0; nk = 1; wk = true; } else if (k >= nz) { ck = nz - 1; nk = nz - 2; wk = true; }
        float3 base = lat_pt(P, L, ci, cj, ck), r = base;
        if (wi) r += base - lat_pt(P, L, ni, cj, ck);
        if (wj) r += base - lat_pt(P, L, ci, nj, ck);
        if (wk) r += base - lat_pt(P, L, ci, cj, nk);
        return r;
    }
    static float3 lat_cr(float3 a, float3 b, float3 c, float3 d, float t) {
        return 0.5f * ((2 * b) + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * (t * t) + (3 * b - a - 3 * c + d) * (t * t * t));
    }
    kernel void lattice(device const float4* pos [[buffer(0)]], device float4* outP [[buffer(1)]],
                        device const float4* pts [[buffer(2)]], constant LP& L [[buffer(3)]],
                        uint v [[thread_position_in_grid]]) {
        if (v >= L.n) return;
        float3 p = pos[v].xyz;
        float3 u = (p - L.boxMin.xyz) / L.size.xyz;
        float3 uc = clamp(u, 0.0f, 1.0f);
        float3 pc = L.boxMin.xyz + uc * L.size.xyz;
        float3 g = uc * float3(float(L.dx - 1), float(L.dy - 1), float(L.dz - 1));
        int i0 = min(int(g.x), int(L.dx) - 2), j0 = min(int(g.y), int(L.dy) - 2), k0 = min(int(g.z), int(L.dz) - 2);
        float3 f = float3(g.x - float(i0), g.y - float(j0), g.z - float(k0));
        float3 s;
        if (L.cubic == 0) {
            #define AT(a, b, c) lat_pt(pts, L, i0 + a, j0 + b, k0 + c)
            float3 c00 = mix(AT(0,0,0), AT(1,0,0), f.x), c10 = mix(AT(0,1,0), AT(1,1,0), f.x);
            float3 c01 = mix(AT(0,0,1), AT(1,0,1), f.x), c11 = mix(AT(0,1,1), AT(1,1,1), f.x);
            s = mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
            #undef AT
        } else {
            float3 pl[4];
            for (int dk = -1; dk <= 2; dk++) {
                float3 rw[4];
                for (int dj = -1; dj <= 2; dj++) {
                    rw[dj + 1] = lat_cr(lat_cp(pts, L, i0 - 1, j0 + dj, k0 + dk), lat_cp(pts, L, i0, j0 + dj, k0 + dk),
                                        lat_cp(pts, L, i0 + 1, j0 + dj, k0 + dk), lat_cp(pts, L, i0 + 2, j0 + dj, k0 + dk), f.x);
                }
                pl[dk + 1] = lat_cr(rw[0], rw[1], rw[2], rw[3], f.y);
            }
            s = lat_cr(pl[0], pl[1], pl[2], pl[3], f.z);
        }
        outP[v] = float4(p + (s - pc), 0);
    }

    // ---- Solidify: vertices for both layers and the rim, one thread per valid triangle / rim edge ---------------
    struct SP { uint nt, ne, hasN, hasUV; float aOff, bOff, pad0, pad1; };
    static float3 layer(device const float4* pos, device const uint* rep, device const float4* wN, device const float* scale,
                        uint id, float off) {
        return pos[rep[id]].xyz + wN[id].xyz * (off * scale[id]);
    }
    static void put_flat(device float4* oP, device float4* oN, device float2* oU, uint at, float3 a, float3 b, float3 c) {
        float3 n = cross(b - a, c - a); float l = length(n);
        n = l > 1e-14f ? n / l : float3(0);
        oP[at] = float4(a, 0); oP[at + 1] = float4(b, 0); oP[at + 2] = float4(c, 0);
        oN[at] = float4(n, 0); oN[at + 1] = float4(n, 0); oN[at + 2] = float4(n, 0);
        oU[at] = float2(0); oU[at + 1] = float2(0); oU[at + 2] = float2(0);
    }
    kernel void solid_tris(device const float4* pos [[buffer(0)]], device const float4* nrm [[buffer(1)]],
                           device const float2* uv [[buffer(2)]], device const uint* idx [[buffer(3)]],
                           device const uint* cid [[buffer(4)]], device const uint* rep [[buffer(5)]],
                           device const float4* wN [[buffer(6)]], device const float* scale [[buffer(7)]],
                           device const uint* src [[buffer(8)]], device float4* oP [[buffer(9)]],
                           device float4* oN [[buffer(10)]], device float2* oU [[buffer(11)]],
                           constant SP& P [[buffer(12)]], uint j [[thread_position_in_grid]]) {
        if (j >= P.nt) return;
        uint t = src[j];
        uint c0 = idx[3 * t], c1 = idx[3 * t + 1], c2 = idx[3 * t + 2];
        uint i0 = cid[c0], i1 = cid[c1], i2 = cid[c2];
        float3 a0 = layer(pos, rep, wN, scale, i0, P.aOff), a1 = layer(pos, rep, wN, scale, i1, P.aOff), a2 = layer(pos, rep, wN, scale, i2, P.aOff);
        float3 b0 = layer(pos, rep, wN, scale, i0, P.bOff), b1 = layer(pos, rep, wN, scale, i1, P.bOff), b2 = layer(pos, rep, wN, scale, i2, P.bOff);
        uint o = 6 * j;
        float2 u0 = P.hasUV != 0 ? uv[c0] : float2(0), u1 = P.hasUV != 0 ? uv[c1] : float2(0), u2 = P.hasUV != 0 ? uv[c2] : float2(0);
        if (P.hasN != 0) {
            oP[o] = float4(a0, 0); oP[o + 1] = float4(a1, 0); oP[o + 2] = float4(a2, 0);
            oN[o] = nrm[c0]; oN[o + 1] = nrm[c1]; oN[o + 2] = nrm[c2];
            oU[o] = u0; oU[o + 1] = u1; oU[o + 2] = u2;
            oP[o + 3] = float4(b0, 0); oP[o + 4] = float4(b2, 0); oP[o + 5] = float4(b1, 0);
            oN[o + 3] = -nrm[c0]; oN[o + 4] = -nrm[c2]; oN[o + 5] = -nrm[c1];
            oU[o + 3] = u0; oU[o + 4] = u2; oU[o + 5] = u1;
        } else {
            put_flat(oP, oN, oU, o, a0, a1, a2);
            oU[o] = u0; oU[o + 1] = u1; oU[o + 2] = u2;
            put_flat(oP, oN, oU, o + 3, b0, b2, b1);
            oU[o + 3] = u0; oU[o + 4] = u2; oU[o + 5] = u1;
        }
    }
    kernel void solid_rim(device const float4* pos [[buffer(0)]], device const uint* rep [[buffer(1)]],
                          device const float4* wN [[buffer(2)]], device const float* scale [[buffer(3)]],
                          device const uint* edges [[buffer(4)]], device float4* oP [[buffer(5)]],
                          device float4* oN [[buffer(6)]], device float2* oU [[buffer(7)]],
                          constant SP& P [[buffer(8)]], uint e [[thread_position_in_grid]]) {
        if (e >= P.ne) return;
        uint a = edges[2 * e], b = edges[2 * e + 1];
        float3 Aa = layer(pos, rep, wN, scale, a, P.aOff), Ba = layer(pos, rep, wN, scale, a, P.bOff);
        float3 Ab = layer(pos, rep, wN, scale, b, P.aOff), Bb = layer(pos, rep, wN, scale, b, P.bOff);
        uint o = 6 * (P.nt + e);
        put_flat(oP, oN, oU, o, Aa, Ba, Bb);
        put_flat(oP, oN, oU, o + 3, Aa, Bb, Ab);
    }
    """
}
#endif
