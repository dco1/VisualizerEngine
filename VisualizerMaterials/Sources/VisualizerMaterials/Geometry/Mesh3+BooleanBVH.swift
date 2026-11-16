//  Bounding-volume hierarchy over triangles (or any boxes) for the mesh-intersection boolean, plus the
//  small parallel helper and spatial point pool it shares.
//
//  Guarantees: `query` visits every item whose box overlaps the query box (closed, so touching counts) and
//  nothing else. The build is a median split on the longest centroid axis; leaves hold up to 4 items.
//  CPU only: tree walks are branchy, pointer-chasing work that does not suit a compute pass at this size.

import Foundation
import simd

struct BoxBVH {
    private struct Node { var lo: Vec3; var hi: Vec3; var start: Int32; var count: Int32; var right: Int32 }
    private var nodes: [Node] = []
    private var order: [Int32] = []
    private var ilo: [Vec3] = [], ihi: [Vec3] = []

    init(lo: [Vec3], hi: [Vec3]) {
        let n = lo.count
        order = (0..<n).map { Int32($0) }
        ilo = lo; ihi = hi
        guard n > 0 else { return }
        nodes.reserveCapacity(2 * n / 2 + 2)
        let centre = (0..<n).map { (lo[$0] + hi[$0]) * 0.5 }
        _ = build(0, n, lo, hi, centre)
    }

    private mutating func build(_ s: Int, _ e: Int, _ lo: [Vec3], _ hi: [Vec3], _ c: [Vec3]) -> Int {
        var bl = lo[Int(order[s])], bh = hi[Int(order[s])]
        var cl = c[Int(order[s])], ch = cl
        for i in (s + 1)..<e {
            let k = Int(order[i])
            bl = simd_min(bl, lo[k]); bh = simd_max(bh, hi[k])
            cl = simd_min(cl, c[k]); ch = simd_max(ch, c[k])
        }
        let id = nodes.count
        nodes.append(Node(lo: bl, hi: bh, start: Int32(s), count: Int32(e - s), right: -1))
        if e - s <= 4 { return id }
        let ext = ch - cl
        let axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : (ext.y >= ext.z ? 1 : 2)
        if ext[axis] <= 0 { return id }
        order[s..<e].sort { c[Int($0)][axis] < c[Int($1)][axis] }
        let m = (s + e) / 2
        _ = build(s, m, lo, hi, c)
        let r = build(m, e, lo, hi, c)
        nodes[id].count = 0
        nodes[id].right = Int32(r)
        return id
    }

    /// Calls `visit` for every item whose box overlaps `[qlo, qhi]`.
    func query(_ qlo: Vec3, _ qhi: Vec3, _ visit: (Int) -> Void) {
        guard !nodes.isEmpty else { return }
        var stack: [Int32] = [0]
        while let id = stack.popLast() {
            let nd = nodes[Int(id)]
            if nd.lo.x > qhi.x || nd.hi.x < qlo.x || nd.lo.y > qhi.y || nd.hi.y < qlo.y
                || nd.lo.z > qhi.z || nd.hi.z < qlo.z { continue }
            if nd.count > 0 {
                for i in Int(nd.start)..<Int(nd.start + nd.count) {
                    let k = Int(order[i])
                    let l = ilo[k], h = ihi[k]
                    if l.x > qhi.x || h.x < qlo.x || l.y > qhi.y || h.y < qlo.y || l.z > qhi.z || h.z < qlo.z { continue }
                    visit(k)
                }
            } else {
                stack.append(id + 1); stack.append(nd.right)
            }
        }
    }
}

/// `@unchecked Sendable` buffer so parallel loops can write disjoint slots without exclusivity traps.
final class SharedBuffer<T>: @unchecked Sendable {
    private let base: UnsafeMutablePointer<T>
    let count: Int
    init(_ items: [T]) {
        count = items.count
        base = UnsafeMutablePointer<T>.allocate(capacity: max(1, items.count))
        base.initialize(from: items, count: items.count)
    }
    deinit { base.deinitialize(count: count); base.deallocate() }
    var items: [T] { Array(UnsafeBufferPointer(start: base, count: count)) }
    func withPointer<R>(_ body: (UnsafeMutablePointer<T>) -> R) -> R { body(base) }
}

enum BooleanParallel {
    /// Runs `body(range)` over `count` items split into chunks, on all cores. Each body call must touch
    /// only its own range's output slots.
    static func chunks(_ count: Int, minChunk: Int = 64, _ body: @escaping @Sendable (Range<Int>) -> Void) {
        guard count > 0 else { return }
        let cores = max(1, ProcessInfo.processInfo.activeProcessorCount)
        var chunk = max(minChunk, (count + cores * 4 - 1) / (cores * 4))
        chunk = (chunk + minChunk - 1) / minChunk * minChunk
        let n = (count + chunk - 1) / chunk
        if n <= 1 { body(0..<count); return }
        DispatchQueue.concurrentPerform(iterations: n) { i in
            body((i * chunk)..<min(count, (i + 1) * chunk))
        }
    }
}

/// Welded point store: points closer than `eps` share one id. Grid-hashed.
struct BooleanPool {
    private(set) var points: [Vec3] = []
    let eps: Double
    private let origin: Vec3
    private let cell: Double
    private var head: [SIMD3<Int32>: Int32] = [:]
    private var next: [Int32] = []

    init(origin: Vec3, eps: Double) {
        self.origin = origin; self.eps = eps; self.cell = eps * 16
    }

    private func key(_ p: Vec3) -> SIMD3<Int32> {
        let q = (p - origin) / cell
        return SIMD3(Int32(clamping: Int(q.x.rounded(.down))), Int32(clamping: Int(q.y.rounded(.down))),
                     Int32(clamping: Int(q.z.rounded(.down))))
    }

    mutating func add(_ p: Vec3) -> Int {
        let k = key(p)
        let f = (p - origin) / cell
        let fr = SIMD3(f.x - f.x.rounded(.down), f.y - f.y.rounded(.down), f.z - f.z.rounded(.down))
        let t = eps / cell
        func offs(_ v: Double) -> [Int32] {
            var o: [Int32] = [0]
            if v < t { o.append(-1) }
            if v > 1 - t { o.append(1) }
            return o
        }
        for dx in offs(fr.x) { for dy in offs(fr.y) { for dz in offs(fr.z) {
            var j = head[SIMD3(k.x + dx, k.y + dy, k.z + dz)] ?? -1
            while j >= 0 {
                if len3(points[Int(j)] - p) <= eps { return Int(j) }
                j = next[Int(j)]
            }
        } } }
        let id = points.count
        points.append(p)
        next.append(head[k] ?? -1)
        head[k] = Int32(id)
        return id
    }

    /// Appends a point that is never welded (a private interior vertex).
    mutating func addUnshared(_ p: Vec3) -> Int {
        points.append(p); next.append(-1)
        return points.count - 1
    }
}
