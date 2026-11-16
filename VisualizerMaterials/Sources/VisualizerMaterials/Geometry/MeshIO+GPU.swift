//  MeshIO+GPU.swift — the GPU path of MeshIO (ticket item M5): the Metal context and the `backend` switch.
//
//  What runs on the GPU (kernels in MeshIO+GPUKernels*.swift, one source string compiled at first use, like SDF+GPU):
//    · binary STL decode and encode (50-byte facets <-> flat float arrays, facet-normal rule of the CPU reader),
//    · binary PLY decode and encode (fixed-stride vertex and triangle-face records, the 8 scalar types, either byte
//      order for reading, LE for writing),
//    · STL weld (grid-hash, 32-bit CAS table, prefix-scan compaction) + angle-weighted crease-smoothed normals
//      (atomics), the whole import in ONE command buffer with no CPU read-back in the middle,
//    · per-triangle volume / area / degenerate / flipped checks and a directed-edge winding table (`MeshIO.gpuReport`).
//  What stays on the CPU, and why: OBJ and ASCII STL / PLY text (a tokenizer + float parse is bandwidth-trivial
//  next to a serial `Double(String)`, but the branchy, variable-length grammar — n-gons, negative indices, `solid`
//  blocks, mixed `v/vt/vn` — is where bugs live, and measured below the CPU parse is not the bottleneck on typical
//  OBJ sizes; see the timings in `MeshIOGPUTests.testTimings1M`); binary PLY with n-gons, float64 / 32-bit-integer
//  positions, lists other than the face list, or ASCII (a CPU reader keeps the Double precision and triangulates);
//  `Mesh3` itself, whose arrays are Double (Metal has no Double), so a GPU decode ends by WIDENING float to double
//  on the CPU (parallel). Callers that can stay in float get `MeshIOFloatMesh` and skip it.
//  Guarantees: output equals the CPU reader's — STL positions bit-identical, facet normals within 1e-6, welded
//  meshes with the same vertex order and index list; PLY positions / uvs / colours bit-identical for float32 files;
//  smooth normals within ~1e-4 (float math, float atomics add in no fixed order). `.gpu` that meets an input it can't
//  do (above) quietly runs the CPU path; `.auto` picks the GPU only for big binary inputs (>= `autoThreshold` bytes).

import Foundation
import simd

public extension MeshIO {
    enum Backend: Sendable { case cpu, gpu, auto }
    /// `.auto` uses the GPU at or above this many bytes of input (below it, upload + dispatch costs more than parsing).
    static let autoThreshold = 1 << 20
    static var gpuAvailable: Bool { gpuContextAvailable() }

    static func read(data: Data, format: Format? = nil, options: ReadOptions = .init(), backend: Backend) throws -> Result {
        let f = format ?? detect(data)
        guard let f else { throw Failure.unknownFormat }
        if backend == .cpu || (backend == .auto && data.count < autoThreshold) { return try read(data: data, format: f, options: options) }
        switch f {
        case .stl: if let r = gpuReadSTL(data, options: options) { return r }
        case .ply: if let r = gpuReadPLY(data) { return r }
        case .obj: break                                   // text: CPU (see header)
        }
        return try read(data: data, format: f, options: options)
    }

    static func write(_ mesh: Mesh3, format: Format, options: WriteOptions = .init(), backend: Backend) -> Data {
        write([mesh], format: format, options: options, backend: backend)
    }

    static func write(_ meshes: [Mesh3], format: Format, options: WriteOptions = .init(), backend: Backend) -> Data {
        let size = meshes.reduce(0) { $0 + $1.indices.count / 3 * 50 }
        if backend != .cpu, !(backend == .auto && size < autoThreshold), options.binary {
            switch format {
            case .stl: if let d = gpuWriteSTL(meshes) { return d }
            case .ply: if let d = gpuWritePLY(meshes, options: options) { return d }
            case .obj: break
            }
        }
        return write(meshes, format: format, options: options)
    }
}

/// A mesh as the GPU leaves it: flat float arrays, no widening to Double. `uvs` is 2 floats a vertex, `colors` 3 or [].
public struct MeshIOFloatMesh: Sendable {
    public var positions: [Float] = [], normals: [Float] = [], uvs: [Float] = [], colors: [Float] = []
    public var indices: [UInt32] = []
    public var vertexCount: Int { positions.count / 3 }
    public var triangleCount: Int { indices.count / 3 }
    public init() {}
    public func toMesh3() -> Mesh3 {
        let n = vertexCount
        var m = Mesh3()
        m.positions = MeshIOGPUUtil.widen3(positions, n)
        m.normals = normals.count == positions.count ? MeshIOGPUUtil.widen3(normals, n) : [Vec3](repeating: Vec3(0, 0, 1), count: n)
        m.uvs = uvs.count == 2 * n ? MeshIOGPUUtil.widen2(uvs, n) : [Vec2](repeating: .zero, count: n)
        m.indices = indices
        return m
    }
    public var colors3: [SIMD3<Float>] {
        (0 ..< colors.count / 3).map { SIMD3(colors[3 * $0], colors[3 * $0 + 1], colors[3 * $0 + 2]) }
    }
}

/// CPU helpers, parallel where the work is a per-element conversion.
enum MeshIOGPUUtil {
    struct Ptr<T>: @unchecked Sendable { let p: UnsafeMutablePointer<T> }
    static func chunks(_ n: Int, _ body: @Sendable (Range<Int>) -> Void) {
        let c = max(1, min(ProcessInfo.processInfo.activeProcessorCount, n / 8192))
        let per = (n + c - 1) / c
        DispatchQueue.concurrentPerform(iterations: c) { k in body(min(n, k * per) ..< min(n, (k + 1) * per)) }
    }
    static func widen3(_ f: [Float], _ n: Int) -> [Vec3] {
        [Vec3](unsafeUninitializedCapacity: n) { buf, count in
            let dst = Ptr(p: buf.baseAddress!)
            f.withUnsafeBufferPointer { src in
                let s = src
                nonisolated(unsafe) let sp = s
                chunks(n) { r in for i in r { dst.p[i] = Vec3(Double(sp[3 * i]), Double(sp[3 * i + 1]), Double(sp[3 * i + 2])) } }
            }
            count = n
        }
    }
    static func widen2(_ f: [Float], _ n: Int) -> [Vec2] {
        [Vec2](unsafeUninitializedCapacity: n) { buf, count in
            let dst = Ptr(p: buf.baseAddress!)
            f.withUnsafeBufferPointer { src in
                nonisolated(unsafe) let sp = src
                chunks(n) { r in for i in r { dst.p[i] = Vec2(Double(sp[2 * i]), Double(sp[2 * i + 1])) } }
            }
            count = n
        }
    }
    /// Double Vec3 -> flat Float (the GPU's input form).
    static func narrow3(_ v: [Vec3]) -> [Float] {
        let n = v.count
        return [Float](unsafeUninitializedCapacity: n * 3) { buf, count in
            let dst = Ptr(p: buf.baseAddress!)
            v.withUnsafeBufferPointer { src in
                nonisolated(unsafe) let sp = src
                chunks(n) { r in for i in r { let p = sp[i]; dst.p[3 * i] = Float(p.x); dst.p[3 * i + 1] = Float(p.y); dst.p[3 * i + 2] = Float(p.z) } }
            }
            count = n * 3
        }
    }
    static func narrow2(_ v: [Vec2]) -> [Float] {
        let n = v.count
        return [Float](unsafeUninitializedCapacity: n * 2) { buf, count in
            let dst = Ptr(p: buf.baseAddress!)
            v.withUnsafeBufferPointer { src in
                nonisolated(unsafe) let sp = src
                chunks(n) { r in for i in r { dst.p[2 * i] = Float(sp[i].x); dst.p[2 * i + 1] = Float(sp[i].y) } }
            }
            count = n * 2
        }
    }
}

#if canImport(Metal)
import Metal

func gpuContextAvailable() -> Bool { MeshIOGPU.shared != nil }

/// Device, queue and pipelines, made once; nil without a Metal device.
final class MeshIOGPU: @unchecked Sendable {
    static let shared: MeshIOGPU? = MeshIOGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private let pipes: [String: MTLComputePipelineState]

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        let opts = MTLCompileOptions(); opts.languageVersion = .version3_0
        guard let lib = try? d.makeLibrary(source: MeshIOGPU.sourceCore + MeshIOGPU.sourceWeld + MeshIOGPU.sourcePLY, options: opts)
        else { return nil }
        var p: [String: MTLComputePipelineState] = [:]
        for name in lib.functionNames {
            guard let f = lib.makeFunction(name: name), let ps = try? d.makeComputePipelineState(function: f) else { return nil }
            p[name] = ps
        }
        guard (p["scan_blocks"]?.maxTotalThreadsPerThreadgroup ?? 0) >= 1024,
              (p["tri_check"]?.maxTotalThreadsPerThreadgroup ?? 0) >= 256 else { return nil }
        (device, queue, pipes) = (d, q, p)
    }

    enum Arg { case buf(MTLBuffer), u(UInt32), f(Float), bytes(Data) }
    /// A POD value (not an array, which would copy its reference) by bytes.
    static func bytes<T>(_ v: T) -> Arg { var x = v; return withUnsafeBytes(of: &x) { .bytes(Data($0)) } }
    static func bytes<T>(array a: [T]) -> Arg { a.withUnsafeBytes { .bytes(Data($0)) } }

    /// Encode one dispatch over `n` threads (1-D); args are bound to buffer indices 0, 1, 2 … in order.
    func go(_ e: MTLComputeCommandEncoder, _ name: String, _ n: Int, _ args: [Arg], group: Int = 0) {
        guard n > 0, let ps = pipes[name] else { return }
        e.setComputePipelineState(ps)
        for (i, a) in args.enumerated() {
            switch a {
            case .buf(let b): e.setBuffer(b, offset: 0, index: i)
            case .u(var v): e.setBytes(&v, length: 4, index: i)
            case .f(var v): e.setBytes(&v, length: 4, index: i)
            case .bytes(let d): d.withUnsafeBytes { e.setBytes($0.baseAddress!, length: d.count, index: i) }
            }
        }
        let g = group > 0 ? group : min(256, ps.maxTotalThreadsPerThreadgroup)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: g, height: 1, depth: 1))
    }

    /// Exclusive prefix sum of `flags[0..<n]` into `out`; the total goes to `cnt[slot]`.
    func scan(_ e: MTLComputeCommandEncoder, flags: MTLBuffer, out: MTLBuffer, n: Int, cnt: MTLBuffer, slot: Int) {
        let nb = (n + 1023) / 1024
        let sums = alloc((nb + 1) * 4)
        e.setComputePipelineState(pipes["scan_blocks"]!)
        e.setBuffer(flags, offset: 0, index: 0); e.setBuffer(out, offset: 0, index: 1); e.setBuffer(sums, offset: 0, index: 2)
        var nn = UInt32(n); e.setBytes(&nn, length: 4, index: 3)
        e.dispatchThreadgroups(MTLSize(width: nb, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        go(e, "scan_totals", 1, [.buf(sums), .u(UInt32(nb)), .buf(cnt), .u(UInt32(slot))])
        go(e, "scan_add", n, [.buf(out), .buf(sums), .u(UInt32(n))])
    }

    func alloc(_ bytes: Int) -> MTLBuffer {            // zero-filled by Metal
        device.makeBuffer(length: max(16, bytes), options: .storageModeShared)!
    }
    func upload<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes { $0.count == 0 ? alloc(16) : device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }
    func upload(_ d: Data) -> MTLBuffer {
        d.withUnsafeBytes { $0.count == 0 ? alloc(16) : device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }
    func pow2Table(for n: Int) -> Int { var s = 1024; while s < 2 * n { s <<= 1 }; return s }

    /// Run encoded work to completion. Returns false if the GPU reported an error.
    func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Bool {
        guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return false }
        body(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }
    static func read<T>(_ b: MTLBuffer, _ type: T.Type, count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: T.self), count: count))
    }
}
#else
func gpuContextAvailable() -> Bool { false }
extension MeshIO {
    static func gpuReadSTL(_ data: Data, options: ReadOptions) -> Result? { nil }
    static func gpuReadPLY(_ data: Data) -> Result? { nil }
    static func gpuWriteSTL(_ meshes: [Mesh3]) -> Data? { nil }
    static func gpuWritePLY(_ meshes: [Mesh3], options: WriteOptions) -> Data? { nil }
}
#endif
