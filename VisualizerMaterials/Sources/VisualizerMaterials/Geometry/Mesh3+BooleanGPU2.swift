//  Host side of the GPU-heavy boolean (item M6b): device, pipelines, small buffer helpers, the GPU exclusive scan
//  and the two primitives everything else is built from — a uniform grid of boxes (CSR, built by counting sort on
//  the GPU) and the grid query (box overlap / point near an edge / point near a point), whose pair lists come
//  back in a deterministic order. The pipeline that uses them is in `Mesh3+BooleanGPU2Pipeline.swift`.
//
//  What the GPU may and may not decide: it works in Float, so every verdict is either a superset of the exact
//  answer (candidate pairs) or a verdict whose error bound it checked (the narrow phase's definite hit / miss);
//  anything near a bound is "uncertain" and goes to the CPU's exact predicates. The CPU path stays the oracle.

import Foundation
import simd

#if canImport(Metal)
import Metal

struct GridP {
    var lo: SIMD4<Float>
    var inv: SIMD4<Float>
    var dims: SIMD4<UInt32>
    var cells: Int { Int(dims.x) * Int(dims.y) * Int(dims.z) }
}

enum GArg {
    case buf(MTLBuffer)
    case bytes([UInt8])
    static func val<T>(_ v: T) -> GArg { var x = v; return .bytes(withUnsafeBytes(of: &x) { Array($0) }) }
}

/// A uniform grid over a set of boxes: `start[c] ..< start[c + 1]` indexes `entries`, the boxes touching cell c.
final class BoxGrid2 {
    let spec: GridP
    let n: Int
    let lo: MTLBuffer, hi: MTLBuffer
    var start: MTLBuffer, entries: MTLBuffer
    init(spec: GridP, n: Int, lo: MTLBuffer, hi: MTLBuffer, start: MTLBuffer, entries: MTLBuffer) {
        self.spec = spec; self.n = n; self.lo = lo; self.hi = hi; self.start = start; self.entries = entries
    }
}

final class BooleanGPU2: @unchecked Sendable {
    static let shared: BooleanGPU2? = BooleanGPU2()
    /// Booleans with at least this many triangles (both operands) take the GPU-heavy pipeline.
    nonisolated(unsafe) static var minTriangles = 20_000
    let device: MTLDevice, queue: MTLCommandQueue
    private var pso: [String: MTLComputePipelineState] = [:]
    /// Seconds spent compiling the shaders on first use (reported by the benchmarks as the cold-start cost).
    private(set) var compileSeconds = 0.0

    private init?() {
        let t0 = Date()
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: BooleanGPU2Source.text, options: nil) else { return nil }
        for name in lib.functionNames {
            guard let f = lib.makeFunction(name: name), let p = try? d.makeComputePipelineState(function: f) else { return nil }
            pso[name] = p
        }
        guard let sb = pso["scan_block"], sb.maxTotalThreadsPerThreadgroup >= 1024, sb.threadExecutionWidth == 32,
              let wt = pso["winding_tg"], wt.maxTotalThreadsPerThreadgroup >= 256 else { return nil }
        (device, queue) = (d, q)
        compileSeconds = Date().timeIntervalSince(t0)
    }

    /// Compiles the shaders now (call at launch to take the cold start off the first boolean).
    static func prewarm() { _ = shared }

    // MARK: buffers

    func buf<T>(_ a: [T]) -> MTLBuffer? {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress ?? UnsafeRawPointer(bitPattern: 16)!, length: Swift.max($0.count, 16), options: .storageModeShared) }
    }
    func zeros(_ bytes: Int) -> MTLBuffer? {
        guard let b = device.makeBuffer(length: Swift.max(bytes, 16), options: .storageModeShared) else { return nil }
        memset(b.contents(), 0, b.length)
        return b
    }
    func fill(_ b: MTLBuffer, _ v: UInt32) {
        let p = b.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<(b.length / 4) { p[i] = v }
    }
    func read<T>(_ b: MTLBuffer, _ type: T.Type, _ count: Int, offset: Int = 0) -> [T] {
        guard count > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: (b.contents() + offset).assumingMemoryBound(to: T.self), count: count))
    }
    func u32(_ b: MTLBuffer, _ i: Int) -> Int { Int(b.contents().assumingMemoryBound(to: UInt32.self)[i]) }

    // MARK: encoding

    /// Runs `body` in one command buffer / compute encoder and waits. False when the GPU reports an error.
    @discardableResult
    func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Bool {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        body(enc)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }

    func go(_ enc: MTLComputeCommandEncoder, _ name: String, _ n: Int, tg: Int = 64, _ args: [GArg]) {
        guard n > 0, let p = pso[name] else { return }
        enc.setComputePipelineState(p)
        for (i, a) in args.enumerated() {
            switch a {
            case .buf(let b): enc.setBuffer(b, offset: 0, index: i)
            case .bytes(let v): v.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: v.count, index: i) }
            }
        }
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: Swift.min(tg, p.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }

    /// Like `go`, but dispatches `groups` threadgroups of `tg` threads.
    func goGroups(_ enc: MTLComputeCommandEncoder, _ name: String, groups: Int, tg: Int, _ args: [GArg]) {
        guard groups > 0, let p = pso[name] else { return }
        enc.setComputePipelineState(p)
        for (i, a) in args.enumerated() {
            switch a {
            case .buf(let b): enc.setBuffer(b, offset: 0, index: i)
            case .bytes(let v): v.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: v.count, index: i) }
            }
        }
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    /// Exclusive prefix sum in place over `n` entries (the buffer should hold `n + 1` zero-padded entries when the
    /// total is wanted: it ends up in entry `n`).
    func scan(_ enc: MTLComputeCommandEncoder, _ b: MTLBuffer, _ n: Int) {
        guard n > 0 else { return }
        let blocks = (n + 1023) / 1024
        guard let sums = zeros(blocks * 4 + 4) else { return }
        go(enc, "scan_block", blocks * 1024, tg: 1024, [.buf(b), .buf(sums), .val(UInt32(n))])
        if blocks > 1 {
            scan(enc, sums, blocks)
            go(enc, "scan_add", n, tg: 1024, [.buf(b), .buf(sums), .val(UInt32(n))])
        }
    }

    // MARK: grid

    /// Builds one grid per box set (all on the given spec) with two command buffers: count + scan, then fill.
    func makeGrids(_ sets: [(lo: MTLBuffer, hi: MTLBuffer, n: Int)], spec: GridP) -> [BoxGrid2]? {
        let cells = spec.cells
        var counts: [MTLBuffer] = []
        for _ in sets { guard let c = zeros((cells + 1) * 4) else { return nil }; counts.append(c) }
        let ok = run { enc in
            for (k, s) in sets.enumerated() {
                go(enc, "grid_count", s.n, [.buf(s.lo), .buf(s.hi), .val(spec), .buf(counts[k]), .val(UInt32(s.n))])
                scan(enc, counts[k], cells + 1)
            }
        }
        guard ok else { return nil }
        var entries: [MTLBuffer] = [], cursors: [MTLBuffer] = []
        for k in 0..<sets.count {
            let total = u32(counts[k], cells)
            guard total < 200_000_000, let e = zeros(total * 4), let c = zeros(cells * 4) else { return nil }
            entries.append(e); cursors.append(c)
        }
        let ok2 = run { enc in
            for (k, s) in sets.enumerated() {
                go(enc, "grid_fill", s.n, [.buf(s.lo), .buf(s.hi), .val(spec), .buf(counts[k]), .buf(cursors[k]),
                                           .buf(entries[k]), .val(UInt32(s.n))])
            }
        }
        guard ok2 else { return nil }
        return sets.enumerated().map { BoxGrid2(spec: spec, n: $1.n, lo: $1.lo, hi: $1.hi, start: counts[$0], entries: entries[$0]) }
    }

    /// Pairs (query index, grid entry) from querying `q` boxes against `grid`. `mode`: 0 box overlap, 1 query is a
    /// point (index = its pool id) within `margin` of an edge of the grid's triangles (`tris`, `pos`), 3 points
    /// within `margin` of earlier points. Returns the pairs as a GPU buffer in (query, entry) order.
    func query(grid: BoxGrid2, loQ: MTLBuffer, hiQ: MTLBuffer, nQ: Int, mode: UInt32, margin: Float,
               tris: MTLBuffer, pos: MTLBuffer) -> (pairs: MTLBuffer, count: Int)? {
        guard nQ > 0, let counts = zeros((nQ + 1) * 4), let dummy = zeros(16), let one = zeros(16) else { return nil }
        struct QP { var n: UInt32; var mode: UInt32; var phase: UInt32; var cap: UInt32; var margin: Float; var p0: Float = 0; var p1: Float = 0; var p2: Float = 0 }
        func args(_ phase: UInt32, _ cap: UInt32, _ out: MTLBuffer) -> [GArg] {
            [.buf(loQ), .buf(hiQ), .buf(grid.lo), .buf(grid.hi), .buf(grid.start), .buf(grid.entries),
             .val(grid.spec), .val(QP(n: UInt32(nQ), mode: mode, phase: phase, cap: cap, margin: margin)),
             .buf(counts), .buf(counts), .buf(out), .buf(dummy), .buf(tris), .buf(pos)]
        }
        guard run({ enc in
            go(enc, "query", nQ, args(0, 0, one))
            scan(enc, counts, nQ + 1)
        }) else { return nil }
        let total = u32(counts, nQ)
        guard total < 400_000_000, let out = zeros(total * 8) else { return nil }
        guard run({ enc in go(enc, "query", nQ, args(1, UInt32(total), out)) }) else { return nil }
        return (out, total)
    }
}
#else
final class BooleanGPU2: @unchecked Sendable {
    static let shared: BooleanGPU2? = nil
    nonisolated(unsafe) static var minTriangles = 20_000
    static func prewarm() {}
}
#endif
