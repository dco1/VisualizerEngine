//  ModellingGPU.swift — one Metal context for the modelling kit's GPU topology work (ticket item M4).
//
//  Holds the device, queue and ONE library compiled from source at first use (the SDF grid / refine kernels of
//  `SDFGPU`, plus the prefix scan here, the surface-nets topology kernels and the half-edge bulk kernels), a pipeline
//  cache, and tiny encoding helpers. `ModellingGPU.shared` is nil without a Metal device; every user falls back to its
//  CPU twin then.
//
//  `scan` is a work-efficient exclusive prefix sum over uint32 (1024-element blocks, simdgroup scans inside, block sums
//  scanned recursively), the one primitive stream compaction needs: flag → scan → scatter. To get a total, scan n + 1
//  elements with a zero last flag: the total lands in element n.
//  Guarantees: exact (integer). Not guaranteed: totals past 2^32.

import Foundation

#if canImport(Metal)
import Metal

enum GPUArg {
    case buf(MTLBuffer)
    case bytes([UInt8])
    static func value<T>(_ v: T) -> GPUArg { var x = v; return .bytes(withUnsafeBytes(of: &x) { Array($0) }) }
}

final class ModellingGPU: @unchecked Sendable {
    static let shared: ModellingGPU? = ModellingGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private let library: MTLLibrary
    private var cache: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        let src = [SDFGPU.source, ModellingGPU.scanSource, SDFNetGPU.source, HalfEdgeBulkGPU.source,
                   HalfEdgeTopologyGPU.source].joined(separator: "\n")
        do { library = try d.makeLibrary(source: src, options: nil) } catch {
            FileHandle.standardError.write(Data("ModellingGPU: shader compile failed: \(error)\n".utf8))
            return nil
        }
        (device, queue) = (d, q)
    }

    func pipe(_ name: String) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = cache[name] { return p }
        let f = library.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: f)
        cache[name] = p
        return p
    }

    // MARK: buffers

    func buffer(length: Int) -> MTLBuffer { device.makeBuffer(length: Swift.max(length, 16), options: .storageModeShared)! }
    func buffer<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress ?? UnsafeRawPointer(bitPattern: 16)!, length: Swift.max($0.count, 16), options: .storageModeShared)! }
    }
    func read<T>(_ b: MTLBuffer, _ type: T.Type, count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: T.self), count: count))
    }

    // MARK: encoding

    private func bind(_ enc: MTLComputeCommandEncoder, _ name: String, _ args: [GPUArg]) -> MTLComputePipelineState {
        let p = pipe(name)
        enc.setComputePipelineState(p)
        for (i, a) in args.enumerated() {
            switch a {
            case .buf(let b): enc.setBuffer(b, offset: 0, index: i)
            case .bytes(let x): x.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: x.count, index: i) }
            }
        }
        return p
    }

    func dispatch1D(_ enc: MTLComputeCommandEncoder, _ name: String, _ n: Int, _ args: [GPUArg]) {
        guard n > 0 else { return }
        let p = bind(enc, name, args)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Swift.min(256, p.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }

    func dispatch3D(_ enc: MTLComputeCommandEncoder, _ name: String, _ w: Int, _ h: Int, _ d: Int, _ args: [GPUArg]) {
        guard w > 0, h > 0, d > 0 else { return }
        _ = bind(enc, name, args)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: d), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
    }

    /// Threadgroup counts read from `indirect` (3 × uint32) on the GPU: how a kernel decides, itself, to run or not.
    func dispatchIndirect(_ enc: MTLComputeCommandEncoder, _ name: String, _ args: [GPUArg], indirect: MTLBuffer) {
        _ = bind(enc, name, args)
        enc.dispatchThreadgroups(indirectBuffer: indirect, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
    }

    func dispatchGroups(_ enc: MTLComputeCommandEncoder, _ name: String, groups: Int, threads: Int, _ args: [GPUArg]) {
        guard groups > 0 else { return }
        _ = bind(enc, name, args)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// Exclusive prefix sum of `input[0..<n]` into `output[0..<n]` (distinct buffers).
    func scan(_ enc: MTLComputeCommandEncoder, input: MTLBuffer, output: MTLBuffer, n: Int) {
        guard n > 0 else { return }
        let blocks = (n + 1023) / 1024
        let sums = buffer(length: blocks * 4)
        dispatchGroups(enc, "mk_scan_block", groups: blocks, threads: 256, [.buf(input), .buf(output), .buf(sums), .value(UInt32(n))])
        if blocks > 1 {
            let scanned = buffer(length: blocks * 4)
            scan(enc, input: sums, output: scanned, n: blocks)
            dispatch1D(enc, "mk_scan_add", n, [.buf(output), .buf(scanned), .value(UInt32(n))])
        }
    }

    func fill(_ enc: MTLComputeCommandEncoder, _ b: MTLBuffer, count: Int, value: UInt32) {
        dispatch1D(enc, "mk_fill_u32", count, [.buf(b), .value(UInt32(count)), .value(value)])
    }

    /// Run `body` in one command buffer and wait. False if the GPU failed.
    @discardableResult
    func run(_ body: (MTLComputeCommandEncoder) -> Void) -> Bool {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        body(enc)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }

    static let scanSource = """
    kernel void mk_scan_block(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], device uint* sums [[buffer(2)]],
                              constant uint& n [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
                              uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup uint part[32];
        uint base = gid * 1024 + tid * 4;
        uint v0 = base + 0 < n ? in[base + 0] : 0, v1 = base + 1 < n ? in[base + 1] : 0;
        uint v2 = base + 2 < n ? in[base + 2] : 0, v3 = base + 3 < n ? in[base + 3] : 0;
        uint s = v0 + v1 + v2 + v3;
        uint ex = simd_prefix_exclusive_sum(s);
        uint tot = simd_sum(s);
        if (lane == 0) part[sg] = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint off = 0;
        for (uint i = 0; i < sg; i++) off += part[i];
        if (tid == 255) sums[gid] = off + tot;
        uint run = off + ex;
        if (base + 0 < n) out[base + 0] = run;
        run += v0; if (base + 1 < n) out[base + 1] = run;
        run += v1; if (base + 2 < n) out[base + 2] = run;
        run += v2; if (base + 3 < n) out[base + 3] = run;
    }
    kernel void mk_scan_add(device uint* out [[buffer(0)]], device const uint* offs [[buffer(1)]], constant uint& n [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
        if (i < n) out[i] += offs[i / 1024];
    }
    kernel void mk_fill_u32(device uint* b [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& v [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
        if (i < n) b[i] = v;
    }
    """
}
#else
final class ModellingGPU: @unchecked Sendable { static let shared: ModellingGPU? = nil }
#endif
