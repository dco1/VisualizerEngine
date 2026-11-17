import Foundation
import simd

//  UVPackGPU.swift — pack scoring on the GPU: minimum-area rectangles, island bounds and the overlap check (item M3).
//
//  Guarantees: `aligned` gives each island the rotation `UVGeometry.alignedToMinimumRectangle` (or, `wide`, the
//  `wideAlignedToMinimumRectangle`) would, moved to the origin: the candidate angles are the island's convex hull
//  edges (hull on the CPU), every candidate's rotated bounding box is scored by one threadgroup over the hull points,
//  a kernel picks the best (the first of any that tie within 2e-5 (Float noise on a thin box), as the CPU picks the first strict improvement),
//  and one kernel transforms every UV of every island. `score` reduces each island's bounds on the GPU and tests every
//  pair of island rectangles in one N×N dispatch (`overlapPairs`, same 1e-9 slack as the CPU packer's tests).
//  `pack` = GPU alignment + the CPU `UVPacker` shelf search (60 steps over a few hundred boxes: sequential, tiny) on
//  already-rotated islands. Doesn't: nest islands. Float: the alignment angle agrees to ~1e-6 rad.

public struct UVPackScore: Sendable, Equatable {
    /// Σ of the islands' bounding-rectangle areas.
    public var boxArea: Double
    /// Pairs of islands whose rectangles overlap.
    public var overlapPairs: Int
    /// Whether every rectangle lies inside 0…1.
    public var insideUnitSquare: Bool

    /// The CPU oracle.
    public static func cpu(_ layouts: [UVIslandLayout]) -> UVPackScore {
        let b = layouts.map { $0.bounds }
        var area = 0.0, inside = true, pairs = 0
        for r in b {
            area += (r.max.x - r.min.x) * (r.max.y - r.min.y)
            if r.min.x < -1e-9 || r.min.y < -1e-9 || r.max.x > 1 + 1e-9 || r.max.y > 1 + 1e-9 { inside = false }
        }
        for i in 0..<b.count { for j in (i + 1)..<Swift.max(b.count, i + 1) {
            if b[i].min.x < b[j].max.x - 1e-9 && b[j].min.x < b[i].max.x - 1e-9
                && b[i].min.y < b[j].max.y - 1e-9 && b[j].min.y < b[i].max.y - 1e-9 { pairs += 1 }
        } }
        return UVPackScore(boxArea: area, overlapPairs: pairs, insideUnitSquare: inside)
    }
}

public enum UVPackGPU {
    /// Islands scored on `backend`.
    public static func score(_ layouts: [UVIslandLayout], backend: UVBackend) -> UVPackScore {
        #if canImport(Metal)
        if backend.useGPU(work: layouts.reduce(0) { $0 + $1.uv.count }, threshold: 4096), let g = UVGPU.shared,
           let s = gpuScore(layouts, g) { return s }
        #endif
        return .cpu(layouts)
    }

    /// `UVPacker.pack` with the rotation search on `backend`.
    public static func pack(_ layouts: [UVIslandLayout], mesh: Mesh3, options: UVPackOptions, backend: UVBackend) -> [UVIslandLayout] {
        #if canImport(Metal)
        if backend.useGPU(work: layouts.reduce(0) { $0 + $1.uv.count }, threshold: 4096), let g = UVGPU.shared {
            var o = options
            var items = layouts
            if options.rotateToFit, let r = aligned(layouts, wide: true, g) { items = r; o.rotateToFit = false }
            return UVPacker.pack(items, mesh: mesh, options: o)
        }
        #endif
        return UVPacker.pack(layouts, mesh: mesh, options: options)
    }

    static var source: String {
        #if canImport(Metal)
        kernelSource
        #else
        ""
        #endif
    }
}

#if canImport(Metal)
import Metal

extension UVPackGPU {
    struct Cand { var hs: UInt32, hc: UInt32, angle: Float, pad: UInt32 }

    /// Layouts rotated to their minimum rectangle (and quarter-turned wide if `wide`), moved to the origin.
    static func aligned(_ layouts: [UVIslandLayout], wide: Bool, _ g: UVGPU) -> [UVIslandLayout]? {
        guard !layouts.isEmpty else { return [] }
        var hull: [SIMD2<Float>] = [], cands: [Cand] = [], candRange: [SIMD2<UInt32>] = []
        var pts: [SIMD2<Float>] = [], owner: [UInt32] = []
        for (i, l) in layouts.enumerated() {
            let lo0 = l.bounds.min   // centre on the corner: Float keeps its precision for the island's own size
            let h = UVGeometry.hull(l.uv).map { $0 - lo0 }
            let hs = UInt32(hull.count)
            hull += h.map { SIMD2(Float($0.x), Float($0.y)) }
            let c0 = UInt32(cands.count)
            if h.count >= 2 {
                for k in 0..<h.count {
                    let d = h[(k + 1) % h.count] - h[k]
                    guard len(d) > 1e-15 else { continue }
                    cands.append(Cand(hs: hs, hc: UInt32(h.count), angle: Float(-atan2(d.y, d.x)), pad: 0))
                }
            }
            if cands.count == Int(c0) { cands.append(Cand(hs: hs, hc: UInt32(h.count), angle: 0, pad: 0)) }
            candRange.append(SIMD2(c0, UInt32(cands.count) - c0))
            pts += l.uv.map { SIMD2(Float($0.x - lo0.x), Float($0.y - lo0.y)) }
            owner += [UInt32](repeating: UInt32(i), count: l.uv.count)
        }
        let hb = g.buffer(hull), cb = g.buffer(cands), rb = g.buffer(candRange)
        let box = g.empty(cands.count * 16), tf = g.empty(layouts.count * 16)
        let pb = g.buffer(pts), ob = g.buffer(owner), outp = g.empty(pts.count * 8)
        var wideFlag: UInt32 = wide ? 1 : 0, nIsl = UInt32(layouts.count), nPts = UInt32(pts.count)
        guard g.run({ enc in
            enc.setBuffer(hb, offset: 0, index: 0); enc.setBuffer(cb, offset: 0, index: 1); enc.setBuffer(box, offset: 0, index: 2)
            g.dispatchGroups(enc, "pack_rect_score", groups: cands.count, size: 64)
            enc.setBuffer(box, offset: 0, index: 0); enc.setBuffer(cb, offset: 0, index: 1); enc.setBuffer(rb, offset: 0, index: 2)
            enc.setBuffer(tf, offset: 0, index: 3); enc.setBytes(&wideFlag, length: 4, index: 4); enc.setBytes(&nIsl, length: 4, index: 5)
            g.dispatch(enc, "pack_rect_best", layouts.count)
            enc.setBuffer(pb, offset: 0, index: 0); enc.setBuffer(ob, offset: 0, index: 1); enc.setBuffer(tf, offset: 0, index: 2)
            enc.setBuffer(outp, offset: 0, index: 3); enc.setBytes(&nPts, length: 4, index: 4)
            g.dispatch(enc, "pack_xform", pts.count)
        }) else { return nil }
        let out = UVGPU.read(outp, SIMD2<Float>.self, count: pts.count)
        var result: [UVIslandLayout] = [], at = 0
        for l in layouts {
            result.append(UVIslandLayout(island: l.island, uv: (0..<l.uv.count).map { Vec2(Double(out[at + $0].x), Double(out[at + $0].y)) }))
            at += l.uv.count
        }
        return result
    }

    static func gpuScore(_ layouts: [UVIslandLayout], _ g: UVGPU) -> UVPackScore? {
        guard !layouts.isEmpty else { return UVPackScore(boxArea: 0, overlapPairs: 0, insideUnitSquare: true) }
        var pts: [SIMD2<Float>] = [], range: [SIMD2<UInt32>] = []
        for l in layouts { range.append(SIMD2(UInt32(pts.count), UInt32(l.uv.count))); pts += l.uv.map { SIMD2(Float($0.x), Float($0.y)) } }
        let n = layouts.count
        let pb = g.buffer(pts), rb = g.buffer(range), bb = g.empty(n * 16), cnt = g.empty(16)
        memset(cnt.contents(), 0, 16)
        var nn = UInt32(n)
        guard g.run({ enc in
            enc.setBuffer(pb, offset: 0, index: 0); enc.setBuffer(rb, offset: 0, index: 1); enc.setBuffer(bb, offset: 0, index: 2)
            g.dispatchGroups(enc, "pack_bounds", groups: n)
            enc.setBuffer(bb, offset: 0, index: 0); enc.setBuffer(cnt, offset: 0, index: 1); enc.setBytes(&nn, length: 4, index: 2)
            enc.setComputePipelineState(g.pipeline("pack_overlap"))
            enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }) else { return nil }
        let b = UVGPU.read(bb, SIMD4<Float>.self, count: n)
        let c = UVGPU.read(cnt, UInt32.self, count: 2)
        var area = 0.0, inside = true
        for r in b {
            area += Double(r.z - r.x) * Double(r.w - r.y)
            if r.x < -1e-6 || r.y < -1e-6 || r.z > 1 + 1e-6 || r.w > 1 + 1e-6 { inside = false }
        }
        return UVPackScore(boxArea: area, overlapPairs: Int(c[0]), insideUnitSquare: inside)
    }

    static let kernelSource = """
    // One threadgroup per candidate angle: bounding box (minx, miny, maxx, maxy) of the rotated hull.
    struct Cand { uint hs, hc; float angle; uint pad; };
    kernel void pack_rect_score(device const float2* hull [[buffer(0)]], device const Cand* cands [[buffer(1)]],
                                device float4* box [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                                uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        threadgroup float sh[32];
        Cand c = cands[grp];
        float cs = cos(c.angle), sn = sin(c.angle);
        float4 b = float4(INFINITY, INFINITY, -INFINITY, -INFINITY);
        for (uint i = tid; i < c.hc; i += tgs) {
            float2 p = hull[c.hs + i];
            float2 q = float2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
            b = float4(min(b.xy, q), max(b.zw, q));
        }
        float x0 = tg_min(b.x, sh, tid, tgs), y0 = tg_min(b.y, sh, tid, tgs);
        float x1 = tg_max(b.z, sh, tid, tgs), y1 = tg_max(b.w, sh, tid, tgs);
        if (tid == 0) box[grp] = float4(x0, y0, x1, y1);
    }

    // One thread per island: best candidate; tf = (cos, sin, shiftx, shifty) so that p' = R p - shift.
    kernel void pack_rect_best(device const float4* box [[buffer(0)]], device const Cand* cands [[buffer(1)]],
                               device const uint2* range [[buffer(2)]], device float4* tf [[buffer(3)]],
                               constant uint& wide [[buffer(4)]], constant uint& n [[buffer(5)]],
                               uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        uint2 r = range[i];
        float best = INFINITY;
        for (uint k = r.x; k < r.x + r.y; k++) { float4 b = box[k]; best = min(best, (b.z - b.x) * (b.w - b.y)); }
        uint pick = r.x;
        for (uint k = r.x; k < r.x + r.y; k++) {
            float4 b = box[k];
            if ((b.z - b.x) * (b.w - b.y) <= best * (1 + 2e-5f)) { pick = k; break; }
        }
        float4 b = box[pick]; float ang = cands[pick].angle;
        float2 lo = b.xy, hi = b.zw;
        if (wide != 0 && (hi.y - lo.y) > (hi.x - lo.x)) { ang += M_PI_F / 2; lo = float2(-hi.y, lo.x); }
        tf[i] = float4(cos(ang), sin(ang), lo.x, lo.y);
    }

    kernel void pack_xform(device const float2* pts [[buffer(0)]], device const uint* owner [[buffer(1)]],
                           device const float4* tf [[buffer(2)]], device float2* out [[buffer(3)]],
                           constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float4 t = tf[owner[i]]; float2 p = pts[i];
        out[i] = float2(t.x * p.x - t.y * p.y, t.y * p.x + t.x * p.y) - t.zw;
    }

    kernel void pack_bounds(device const float2* pts [[buffer(0)]], device const uint2* range [[buffer(1)]],
                            device float4* bounds [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                            uint grp [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
        threadgroup float sh[32];
        uint2 r = range[grp];
        float4 b = float4(INFINITY, INFINITY, -INFINITY, -INFINITY);
        for (uint i = tid; i < r.y; i += tgs) { float2 p = pts[r.x + i]; b = float4(min(b.xy, p), max(b.zw, p)); }
        float x0 = tg_min(b.x, sh, tid, tgs), y0 = tg_min(b.y, sh, tid, tgs);
        float x1 = tg_max(b.z, sh, tid, tgs), y1 = tg_max(b.w, sh, tid, tgs);
        if (tid == 0) bounds[grp] = float4(x0, y0, x1, y1);
    }

    kernel void pack_overlap(device const float4* b [[buffer(0)]], device atomic_uint* count [[buffer(1)]],
                             constant uint& n [[buffer(2)]], uint2 ij [[thread_position_in_grid]]) {
        uint i = ij.x, j = ij.y;
        if (i >= n || j >= n || i >= j) return;
        float4 a = b[i], c = b[j];
        if (a.x < c.z - 1e-9f && c.x < a.z - 1e-9f && a.y < c.w - 1e-9f && c.y < a.w - 1e-9f)
            atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    }
    """
}
#endif
