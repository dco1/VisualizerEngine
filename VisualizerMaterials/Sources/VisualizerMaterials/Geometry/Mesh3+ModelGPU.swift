//  Mesh3+ModelGPU.swift — the shared Metal plumbing of the GPU weld / decimate / remesh (ticket item M1).
//
//  Guarantees: one device, queue and library (compiled from source strings at first use, like `SDFGPU`; this target
//  has no resources) for every M1 kernel; `ModelPass` encodes dispatches into one serial compute encoder so a
//  pipeline of passes runs without CPU round trips; `scan` is a recursive exclusive prefix sum; `sort` is a bitonic
//  sort of (64-bit key, index) pairs. `ModelGPU.shared` is nil without a Metal device, and every GPU entry point
//  then returns nil so the caller falls back to the CPU code, which stays the oracle.
//  Does NOT guarantee: double precision (Metal has none; callers centre and scale their data first and gather exact
//  doubles back by index where they can), or bit-identical results with the CPU (order of equal-cost choices differs).

import Foundation
import simd

/// Which implementation a modelling operation runs on. `.auto` picks the GPU for meshes big enough to repay it.
public enum MeshBackend: Sendable { case cpu, gpu, auto }

extension MeshBackend {
    /// Triangle count from which `.auto` goes to the GPU (a dispatch chain costs ~1 ms before it pays).
    static let autoThreshold = 20_000
    func useGPU(triangles: Int) -> Bool {
        switch self {
        case .cpu: return false
        case .gpu: return ModelGPU.shared != nil
        case .auto: return ModelGPU.shared != nil && triangles >= MeshBackend.autoThreshold
        }
    }
}

/// Generic kernel parameters (`GP` in the MSL): four ints, four floats, read by name in each kernel.
struct GP {
    var a: UInt32 = 0, b: UInt32 = 0, c: UInt32 = 0, d: UInt32 = 0
    var x: Float = 0, y: Float = 0, z: Float = 0, w: Float = 0
}

#if canImport(Metal)
import Metal

final class ModelGPU: @unchecked Sendable {
    static let shared: ModelGPU? = ModelGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        do { library = try d.makeLibrary(source: ModelGPU.source, options: nil) } catch {
            print("ModelGPU: MSL failed to compile: \(error)")
            return nil
        }
        (device, queue) = (d, q)
    }

    static var source: String { commonSource + weldSource + collapseSource + collapseSource2 + remeshSource + remeshSource2 }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let f = library.makeFunction(name: name), let p = try? device.makeComputePipelineState(function: f)
        else { fatalError("ModelGPU: no kernel \(name)") }
        pipelines[name] = p
        return p
    }

    func buffer(bytes: Int) -> MTLBuffer {
        device.makeBuffer(length: Swift.max(bytes, 16), options: .storageModeShared)!
    }
    func buffer<T>(count: Int, _ type: T.Type) -> MTLBuffer { buffer(bytes: count * MemoryLayout<T>.stride) }
    func buffer<T>(_ a: [T]) -> MTLBuffer {
        if a.isEmpty { return buffer(bytes: 16) }
        return a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }

    /// Run `body` in one command buffer and wait. nil when the GPU reports an error.
    @discardableResult
    func run(_ body: (ModelPass) -> Void) -> Bool {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        let pass = ModelPass(self, enc)
        body(pass)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }
}

/// Records dispatches into one serial compute encoder.
final class ModelPass {
    let gpu: ModelGPU, enc: MTLComputeCommandEncoder
    private var keep: [MTLBuffer] = []
    init(_ g: ModelGPU, _ e: MTLComputeCommandEncoder) { gpu = g; enc = e }

    /// One thread per item. Buffers bind to indices 0…; `p` goes to index 15.
    func go(_ kernel: String, _ n: Int, _ bufs: [MTLBuffer?], _ p: GP = GP()) {
        guard n > 0 else { return }
        let pl = gpu.pipeline(kernel)
        enc.setComputePipelineState(pl)
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        var pp = p
        enc.setBytes(&pp, length: MemoryLayout<GP>.stride, index: 15)
        let w = Swift.min(64, pl.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// Exclusive prefix sum of `m` uints in place. Put a 0 at `[m - 1]` and read the total there after the pass.
    func scan(_ data: MTLBuffer, _ m: Int) {
        guard m > 0 else { return }
        let blocks = (m + 1023) / 1024
        let sums = gpu.buffer(count: blocks, UInt32.self)
        keep.append(sums)
        var pp = GP(a: UInt32(m))
        enc.setComputePipelineState(gpu.pipeline("scan_block"))
        enc.setBuffer(data, offset: 0, index: 0); enc.setBuffer(sums, offset: 0, index: 1)
        enc.setBytes(&pp, length: MemoryLayout<GP>.stride, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        if blocks > 1 {
            scan(sums, blocks)
            enc.setComputePipelineState(gpu.pipeline("scan_add"))
            enc.setBuffer(data, offset: 0, index: 0); enc.setBuffer(sums, offset: 0, index: 1)
            enc.setBytes(&pp, length: MemoryLayout<GP>.stride, index: 15)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        }
    }

    /// Bitonic sort of `n` (key, value) pairs ascending by key then value; both arrays must hold `pow2 ≥ n` items
    /// (the tail is filled with max keys by the caller).
    func sort(keys: MTLBuffer, values: MTLBuffer, pow2: Int) {
        var k = 2
        while k <= pow2 {
            var j = k >> 1
            while j > 0 {
                go("bitonic_step", pow2, [keys, values], GP(a: UInt32(j), b: UInt32(k)))
                j >>= 1
            }
            k <<= 1
        }
    }
}

extension MTLBuffer {
    func array<T>(_ type: T.Type, count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: contents().assumingMemoryBound(to: T.self), count: count))
    }
    func ptr<T>(_ type: T.Type) -> UnsafeMutablePointer<T> { contents().assumingMemoryBound(to: T.self) }
}

extension ModelGPU {
    static let commonSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct GP { uint a, b, c, d; float x, y, z, w; };

    kernel void scan_block(device uint* d [[buffer(0)]], device uint* sums [[buffer(1)]], constant GP& p [[buffer(15)]],
                           uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                           uint grp [[threadgroup_position_in_grid]]) {
        threadgroup uint t[1024];
        uint v = gid < p.a ? d[gid] : 0;
        t[lid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = 1; o < 1024; o <<= 1) {
            uint x = lid >= o ? t[lid - o] : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            t[lid] += x;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (gid < p.a) d[gid] = t[lid] - v;
        if (lid == 1023) sums[grp] = t[lid];
    }
    kernel void scan_add(device uint* d [[buffer(0)]], device const uint* sums [[buffer(1)]], constant GP& p [[buffer(15)]],
                         uint gid [[thread_position_in_grid]], uint grp [[threadgroup_position_in_grid]]) {
        if (gid < p.a && grp > 0) d[gid] += sums[grp];
    }

    kernel void bitonic_step(device ulong* k [[buffer(0)]], device uint* v [[buffer(1)]], constant GP& p [[buffer(15)]],
                             uint i [[thread_position_in_grid]]) {
        uint j = p.a, kk = p.b;
        uint ixj = i ^ j;
        if (ixj <= i) return;
        bool asc = (i & kk) == 0;
        ulong ki = k[i], kj = k[ixj];
        uint vi = v[i], vj = v[ixj];
        bool gt = ki > kj || (ki == kj && vi > vj);
        if (gt == asc) { k[i] = kj; k[ixj] = ki; v[i] = vj; v[ixj] = vi; }
    }

    static ulong cell_key(int3 c) {
        ulong h = (ulong)(uint)c.x * 0x9E3779B97F4A7C15ul ^ ((ulong)(uint)c.y * 0xC2B2AE3D27D4EB4Ful)
                ^ ((ulong)(uint)c.z * 0x165667B19E3779F9ul);
        h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ul; h ^= h >> 32;
        return h == 0xFFFFFFFFFFFFFFFFul ? h - 1 : h;
    }
    static uint lower_bound_k(device const ulong* k, uint n, ulong key) {
        uint lo = 0, hi = n;
        while (lo < hi) { uint m = (lo + hi) >> 1; if (k[m] < key) lo = m + 1; else hi = m; }
        return lo;
    }
    """
}

#else
final class ModelGPU: @unchecked Sendable { static let shared: ModelGPU? = nil }
#endif
