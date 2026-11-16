//  Mesh3+WeldGPU.swift — merge-by-distance on the GPU (ticket item M1). `Mesh3.welded(…, backend:)`.
//
//  Guarantees: the same contract as the CPU `welded` (see `Mesh3+Weld.swift`): vertices within `distance` that agree
//  in UV (`uvTolerance`) and normal direction (`normalAngleDegrees`) become one, the representative's position is
//  kept exactly (gathered back from the input doubles by index), normals of merged vertices are averaged, triangles
//  that collapse are dropped, winding is untouched. The representative choice is the CPU's exactly: the lowest index
//  of each cluster that is not within `distance` (and attribute-compatible) of a lower representative (greedy,
//  non-transitive), found by an iterated parallel resolve, so vertex counts match the CPU's.
//  What runs where: every step is a kernel: cell hashing, a bitonic sort of (cell hash, vertex) pairs, a binary-search
//  27-cell neighbour walk, the greedy representative resolve (rounds until no vertex is undecided), a prefix-scan
//  compaction of representatives, the normal averaging (a gather over each representative's cells, no atomics), and
//  the triangle remap / degenerate drop / compaction. The CPU only converts to Float relative to the bounding box,
//  gathers the doubles of the representatives and assembles the `Mesh3`.
//  Does not guarantee: cluster ties across hash buckets at a distance within 1e-5 of `distance` (float rounding),
//  or distances below ~1e-7 of the mesh's extent (Float's resolution): those return nil and the CPU runs.

import simd

#if canImport(Metal)
import Metal

/// The clusters of a GPU weld, still on the GPU: `rep[i]` the representative (an original vertex index), `remap[i]` its
/// new index, `repList[k]` the original index of new vertex k.
struct WeldCore {
    var rep: MTLBuffer, remap: MTLBuffer, repList: MTLBuffer, keys: MTLBuffer, sidx: MTLBuffer
    var nreps: Int, p: GP

    /// Cluster `n` float positions (any frame: cells are floor(p / distance)). `uv` / `nrm` (xyz) veto merges as in
    /// `Mesh3.welded`. nil on a GPU error.
    static func make(_ g: ModelGPU, pos: MTLBuffer, uv: MTLBuffer?, nrm: MTLBuffer?, n: Int, distance: Double,
                     uvTolerance: Double, normalAngleDegrees: Double) -> WeldCore? {
        var npad = 2; while npad < n { npad <<= 1 }
        let keys = g.buffer(count: npad, UInt64.self), sidx = g.buffer(count: npad, UInt32.self)
        let status = g.buffer(count: n, UInt32.self), rep = g.buffer(count: n, UInt32.self)
        let counter = g.buffer(count: 1, UInt32.self)
        memset(status.contents(), 0, n * 4)
        let uvB = uv ?? g.buffer(count: 1, SIMD2<Float>.self), nB = nrm ?? g.buffer(count: 1, SIMD4<Float>.self)
        let flags: UInt32 = (uv != nil ? 1 : 0) | (nrm != nil ? 2 : 0)
        let p = GP(a: UInt32(n), b: UInt32(npad), c: flags, d: UInt32(npad), x: Float(distance),
                   y: Float(distance * distance * (1 + 1e-5)), z: Float(uvTolerance), w: Float(cos(normalAngleDegrees * .pi / 180) - 1e-12))
        guard g.run({ s in
            s.go("weld_keys", npad, [pos, keys, sidx], p)
            s.sort(keys: keys, values: sidx, pow2: npad)
        }) else { return nil }
        var rounds = 0
        repeat {
            counter.ptr(UInt32.self).pointee = 0
            guard g.run({ s in for _ in 0 ..< 4 { s.go("weld_resolve", n, [pos, uvB, nB, keys, sidx, status, counter], p) } }) else { return nil }
            rounds += 4
        } while counter.ptr(UInt32.self).pointee != 0 && rounds < 4096
        guard rounds < 4096 else { return nil }
        let scanB = g.buffer(count: n + 1, UInt32.self), remap = g.buffer(count: n, UInt32.self)
        let repList = g.buffer(count: n, UInt32.self)
        guard g.run({ s in
            s.go("weld_pick", n, [pos, uvB, nB, keys, sidx, status, rep], p)
            s.go("weld_flag", n + 1, [rep, scanB], GP(a: UInt32(n)))
            s.scan(scanB, n + 1)
            s.go("weld_remap", n, [rep, scanB, remap, repList], GP(a: UInt32(n)))
        }) else { return nil }
        return WeldCore(rep: rep, remap: remap, repList: repList, keys: keys, sidx: sidx, nreps: Int(scanB.ptr(UInt32.self)[n]), p: p)
    }
}

public extension Mesh3 {

    /// `welded` on the chosen backend. `.gpu` falls back to the CPU when there is no Metal device or the distance
    /// is finer than Float can resolve against the mesh's extent.
    func welded(distance: Double, uvTolerance: Double = 1e-6, normalAngleDegrees: Double = 180, backend: MeshBackend) -> Mesh3 {
        if backend.useGPU(triangles: triangleCount),
           let m = weldedGPU(distance: distance, uvTolerance: uvTolerance, normalAngleDegrees: normalAngleDegrees) { return m }
        return welded(distance: distance, uvTolerance: uvTolerance, normalAngleDegrees: normalAngleDegrees)
    }

    internal func weldedGPU(distance: Double, uvTolerance: Double, normalAngleDegrees: Double) -> Mesh3? {
        guard let g = ModelGPU.shared, let b = bounds, !indices.isEmpty, distance > 0 else { return nil }
        let n = positions.count, tris = indices.count / 3
        guard len3(b.max - b.min) / distance < 1e8 else { return nil }
        let hasUV = uvs.count == n, hasN = normals.count == n
        let posB = g.buffer(positions.map { SIMD4<Float>(Float($0.x - b.min.x), Float($0.y - b.min.y), Float($0.z - b.min.z), 0) })
        let uvB = hasUV ? g.buffer(uvs.map { SIMD2<Float>(Float($0.x), Float($0.y)) }) : nil
        let nB = hasN ? g.buffer(normals.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 0) }) : nil
        let useN = hasN && normalAngleDegrees < 180
        guard let core = WeldCore.make(g, pos: posB, uv: uvB, nrm: useN ? nB : nil, n: n, distance: distance,
                                       uvTolerance: uvTolerance, normalAngleDegrees: normalAngleDegrees) else { return nil }
        let nreps = core.nreps
        let outN = g.buffer(count: nreps, SIMD4<Float>.self)
        let idxB = g.buffer(indices), tkeep = g.buffer(count: tris + 1, UInt32.self), outIdx = g.buffer(count: tris * 3, UInt32.self)
        guard g.run({ s in
            if hasN {
                s.go("weld_normals", nreps, [posB, uvB ?? posB, nB!, core.keys, core.sidx, core.rep, core.repList, outN], core.p)
            }
            s.go("weld_tri_flag", tris + 1, [idxB, core.remap, tkeep], GP(a: UInt32(tris)))
            s.scan(tkeep, tris + 1)
            s.go("weld_tri_scatter", tris, [idxB, core.remap, tkeep, outIdx], GP(a: UInt32(tris)))
        }) else { return nil }
        let kept = Int(tkeep.ptr(UInt32.self)[tris])
        var out = Mesh3()
        let rl = core.repList.ptr(UInt32.self)
        out.positions = (0 ..< nreps).map { positions[Int(rl[$0])] }
        if hasUV { out.uvs = (0 ..< nreps).map { uvs[Int(rl[$0])] } }
        if hasN {
            let on = outN.ptr(SIMD4<Float>.self)
            out.normals = (0 ..< nreps).map { Vec3(Double(on[$0].x), Double(on[$0].y), Double(on[$0].z)) }
        }
        out.indices = Array(UnsafeBufferPointer(start: outIdx.ptr(UInt32.self), count: kept * 3))
        return out
    }
}

extension ModelGPU {
    static let weldSource = """
    kernel void weld_keys(device const float4* pos [[buffer(0)]], device ulong* key [[buffer(1)]], device uint* idx [[buffer(2)]],
                          constant GP& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
        idx[i] = i;
        key[i] = i >= p.a ? 0xFFFFFFFFFFFFFFFFul : cell_key(int3(floor(pos[i].xyz / p.x)));
    }

    // flags: bit0 compare UVs, bit1 compare normals. p.y = d², p.z = uv tolerance, p.w = cos of the normal angle.
    static bool weld_ok(uint a, uint b, device const float4* pos, device const float2* uv, device const float4* nrm,
                        constant GP& p) {
        float3 d = pos[a].xyz - pos[b].xyz;
        if (dot(d, d) > p.y) return false;
        if (p.c & 1) { float2 e = fabs(uv[a] - uv[b]); if (e.x > p.z || e.y > p.z) return false; }
        if (p.c & 2) { if (dot(nrm[a].xyz, nrm[b].xyz) < p.w) return false; }
        return true;
    }

    #define WELD_NEIGHBOURS(I, J, BODY) { \\
        int3 c0 = int3(floor(pos[I].xyz / p.x)); \\
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) { \\
            ulong ck = cell_key(c0 + int3(dx, dy, dz)); \\
            for (uint m = lower_bound_k(keys, p.d, ck); m < p.d && keys[m] == ck; m++) { uint J = sidx[m]; BODY } } }

    // Greedy representatives, in parallel: i is a representative when every lower neighbour is decided and none is
    // one; it is not when some lower neighbour is one. 0 undecided, 1 representative, 2 merged.
    kernel void weld_resolve(device const float4* pos [[buffer(0)]], device const float2* uv [[buffer(1)]],
                             device const float4* nrm [[buffer(2)]], device const ulong* keys [[buffer(3)]],
                             device const uint* sidx [[buffer(4)]], device uint* status [[buffer(5)]],
                             device atomic_uint* undecided [[buffer(6)]], constant GP& p [[buffer(15)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= p.a || status[i] != 0) return;
        bool lowerRep = false, lowerOpen = false;
        WELD_NEIGHBOURS(i, j, {
            if (j < i && weld_ok(j, i, pos, uv, nrm, p)) {
                uint s = status[j];
                if (s == 1) lowerRep = true; else if (s == 0) lowerOpen = true;
            }
        })
        if (lowerRep) status[i] = 2;
        else if (!lowerOpen) status[i] = 1;
        else atomic_fetch_add_explicit(undecided, 1, memory_order_relaxed);
    }

    kernel void weld_pick(device const float4* pos [[buffer(0)]], device const float2* uv [[buffer(1)]],
                          device const float4* nrm [[buffer(2)]], device const ulong* keys [[buffer(3)]],
                          device const uint* sidx [[buffer(4)]], device const uint* status [[buffer(5)]],
                          device uint* rep [[buffer(6)]], constant GP& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
        if (i >= p.a) return;
        uint best = i;
        if (status[i] == 2) {
            best = 0xFFFFFFFFu;
            WELD_NEIGHBOURS(i, j, { if (j < i && status[j] == 1 && j < best && weld_ok(j, i, pos, uv, nrm, p)) best = j; })
        }
        rep[i] = best;
    }

    kernel void weld_flag(device const uint* rep [[buffer(0)]], device uint* out [[buffer(1)]], constant GP& p [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
        out[i] = (i < p.a && rep[i] == i) ? 1 : 0;
    }
    kernel void weld_remap(device const uint* rep [[buffer(0)]], device const uint* sc [[buffer(1)]],
                           device uint* remap [[buffer(2)]], device uint* list [[buffer(3)]], constant GP& p [[buffer(15)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= p.a) return;
        remap[i] = sc[rep[i]];
        if (rep[i] == i) list[sc[i]] = i;
    }

    // Sum of the normals of every vertex merged into representative list[t], renormalised.
    kernel void weld_normals(device const float4* pos [[buffer(0)]], device const float2* uv [[buffer(1)]],
                             device const float4* nrm [[buffer(2)]], device const ulong* keys [[buffer(3)]],
                             device const uint* sidx [[buffer(4)]], device const uint* rep [[buffer(5)]],
                             device const uint* list [[buffer(6)]], device float4* outN [[buffer(7)]],
                             constant GP& p [[buffer(15)]], uint t [[thread_position_in_grid]]) {
        uint r = list[t];
        float3 sum = float3(0);
        WELD_NEIGHBOURS(r, j, { if (rep[j] == r) sum += nrm[j].xyz; })
        float l = length(sum);
        outN[t] = float4(l > 1e-12f ? sum / l : nrm[r].xyz, 0);
    }

    static bool tri_ok(device const uint* idx, device const uint* remap, uint t) {
        uint a = remap[idx[3 * t]], b = remap[idx[3 * t + 1]], c = remap[idx[3 * t + 2]];
        return a != b && b != c && a != c;
    }
    kernel void weld_tri_flag(device const uint* idx [[buffer(0)]], device const uint* remap [[buffer(1)]],
                              device uint* keep [[buffer(2)]], constant GP& p [[buffer(15)]], uint t [[thread_position_in_grid]]) {
        keep[t] = (t < p.a && tri_ok(idx, remap, t)) ? 1 : 0;
    }
    kernel void weld_tri_scatter(device const uint* idx [[buffer(0)]], device const uint* remap [[buffer(1)]],
                                 device const uint* sc [[buffer(2)]], device uint* out [[buffer(3)]], constant GP& p [[buffer(15)]],
                                 uint t [[thread_position_in_grid]]) {
        if (t >= p.a || !tri_ok(idx, remap, t)) return;
        uint o = sc[t];
        for (uint k = 0; k < 3; k++) out[3 * o + k] = remap[idx[3 * t + k]];
    }
    """
}
#else
public extension Mesh3 {
    func welded(distance: Double, uvTolerance: Double = 1e-6, normalAngleDegrees: Double = 180, backend: MeshBackend) -> Mesh3 {
        welded(distance: distance, uvTolerance: uvTolerance, normalAngleDegrees: normalAngleDegrees)
    }
}
#endif
