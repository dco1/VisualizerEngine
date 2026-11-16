//  Metal source, part 3 of 3, for MeshIO's GPU path: binary PLY. Vertex records are fixed-stride (each property a
//  scalar of one of 8 types at a known offset, either byte order); face records are fixed-stride too when every face
//  is a triangle (`ply_faces` counts the faces that aren't, and the caller then falls back to the CPU reader, which
//  triangulates n-gons). Writing mirrors the CPU's layout: 8 floats (+ 3 uchar colours) a vertex, `3 i32 i32 i32` a face.

#if canImport(Metal)
extension MeshIOGPU {
    static let sourcePLY = """
    // type codes: 0 int8 1 uint8 2 int16 3 uint16 4 int32 5 uint32 6 float32 7 float64
    struct PlyMap { int off[11]; uint typ[11]; uint be; uint hasN; uint hasC; uint pad; };
    struct PlyVParams { uint base; uint stride; uint count; uint pad; };
    struct PlyFParams { uint base; uint stride; uint count; uint listOff; uint cntType; uint idxType; uint be; uint nV; };

    static inline uint tsize(uint ty) { return ty < 2 ? 1u : (ty < 4 ? 2u : (ty < 7 ? 4u : 8u)); }
    static inline uint rdbits(device const uchar* b, ulong o, uint size, bool be) {   // size <= 4
        uint u = 0;
        for (uint k = 0; k < size; k++) u = (u << 8) | uint(b[o + (be ? k : size - 1 - k)]);
        return u;
    }
    static inline float rdprop(device const uchar* b, ulong o, uint ty, bool be) {
        if (ty == 7) {                                              // float64 -> float32 (round toward zero)
            uint hi = rdbits(b, o + (be ? 0 : 4), 4, be), lo = rdbits(b, o + (be ? 4 : 0), 4, be);
            uint sign = hi >> 31, ex = (hi >> 20) & 0x7ff;
            if (ex == 0) return 0;
            int e = int(ex) - 1023 + 127;
            if (e <= 0) return 0;
            if (e >= 255) return sign ? -INFINITY : INFINITY;
            uint mant = ((hi & 0xfffffu) << 3) | (lo >> 29);
            return as_type<float>((sign << 31) | (uint(e) << 23) | mant);
        }
        uint u = rdbits(b, o, tsize(ty), be);
        switch (ty) {
        case 0: return float(int(as_type<char>(uchar(u))));
        case 1: return float(u);
        case 2: return float(int(as_type<short>(ushort(u))));
        case 3: return float(u);
        case 4: return float(as_type<int>(u));
        case 5: return float(u);
        default: return as_type<float>(u);
        }
    }
    static inline uint rdindex(device const uchar* b, ulong o, uint ty, bool be, thread bool& bad) {
        uint u = rdbits(b, o, tsize(ty), be);
        if (ty == 0 && (u & 0x80)) bad = true;
        if (ty == 2 && (u & 0x8000)) bad = true;
        if (ty == 4 && (u & 0x80000000u)) bad = true;
        return u;
    }

    /// Four flat arrays: position(3) normal(3, normalised, zero -> +z) uv(2) colour(3, uchar -> 0...1).
    kernel void ply_vertices(device const uchar* raw [[buffer(0)]], device float* P [[buffer(1)]],
                             device float* N [[buffer(2)]], device float* U [[buffer(3)]], device float* C [[buffer(4)]],
                             constant PlyVParams& vp [[buffer(5)]], constant PlyMap& m [[buffer(6)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= vp.count) return;
        ulong r0 = ulong(vp.base) + ulong(vp.stride) * i;
        float v[11];
        for (uint c = 0; c < 11; c++) {
            if (m.off[c] < 0) { v[c] = (c == 5 && m.hasN == 0) ? 1.0f : 0.0f; continue; }
            v[c] = rdprop(raw, r0 + uint(m.off[c]), m.typ[c], m.be != 0);
            if (c >= 8 && m.typ[c] < 2) v[c] = precise::divide(v[c], 255.0f);
        }
        if (m.hasN != 0) {
            float3 n = float3(v[3], v[4], v[5]); float l = length(n);
            n = l > 1e-30f ? n / l : float3(0, 0, 1);
            v[3] = n.x; v[4] = n.y; v[5] = n.z;
        }
        st3(P, i, float3(v[0], v[1], v[2])); st3(N, i, float3(v[3], v[4], v[5]));
        U[2 * i] = v[6]; U[2 * i + 1] = v[7];
        if (m.hasC != 0) st3(C, i, float3(v[8], v[9], v[10]));
    }
    kernel void ply_faces(device const uchar* raw [[buffer(0)]], device uint* I [[buffer(1)]],
                          constant PlyFParams& fp [[buffer(2)]], device atomic_uint* bad [[buffer(3)]],
                          uint f [[thread_position_in_grid]]) {
        if (f >= fp.count) return;
        ulong r0 = ulong(fp.base) + ulong(fp.stride) * f + fp.listOff;
        bool be = fp.be != 0, badIdx = false;
        uint csz = tsize(fp.cntType), isz = tsize(fp.idxType);
        uint n = rdbits(raw, r0, csz, be);
        if (n != 3) { atomic_fetch_add_explicit(&bad[0], 1u, memory_order_relaxed); I[3 * f] = 0; I[3 * f + 1] = 0; I[3 * f + 2] = 0; return; }
        for (uint k = 0; k < 3; k++) {
            uint v = rdindex(raw, r0 + csz + isz * k, fp.idxType, be, badIdx);
            if (v >= fp.nV) badIdx = true;
            I[3 * f + k] = badIdx ? 0 : v;
        }
        if (badIdx) atomic_fetch_add_explicit(&bad[1], 1u, memory_order_relaxed);
    }

    /// Write: x y z nx ny nz s t as float32 (+ r g b as uchar) a vertex, at `out` + `base`. Inputs are flat floats.
    kernel void ply_write_vertices(device const float* P [[buffer(0)]], device const float* N [[buffer(1)]],
                                   device const float* U [[buffer(2)]], device const float* C [[buffer(3)]],
                                   device uchar* out [[buffer(4)]], constant uint& n [[buffer(5)]],
                                   constant uint& base [[buffer(6)]], constant uint& hasC [[buffer(7)]],
                                   uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint stride = hasC ? 35u : 32u;
        ulong o = ulong(base) + ulong(stride) * i;
        float3 p = ld3(P, i), nn = ld3(N, i);
        stf(out, o, p.x); stf(out, o + 4, p.y); stf(out, o + 8, p.z);
        stf(out, o + 12, nn.x); stf(out, o + 16, nn.y); stf(out, o + 20, nn.z);
        stf(out, o + 24, U[2 * i]); stf(out, o + 28, U[2 * i + 1]);
        if (hasC) for (uint k = 0; k < 3; k++) out[o + 32 + k] = uchar(clamp(round(C[3 * i + k] * 255.0f), 0.0f, 255.0f));
    }
    kernel void ply_write_faces(device const uint* I [[buffer(0)]], device uchar* out [[buffer(1)]],
                                constant uint& nT [[buffer(2)]], constant uint& base [[buffer(3)]],
                                uint t [[thread_position_in_grid]]) {
        if (t >= nT) return;
        ulong o = ulong(base) + 13ul * t;
        out[o] = 3; st32(out, o + 1, I[3 * t]); st32(out, o + 5, I[3 * t + 1]); st32(out, o + 9, I[3 * t + 2]);
    }
    """
}
#endif
