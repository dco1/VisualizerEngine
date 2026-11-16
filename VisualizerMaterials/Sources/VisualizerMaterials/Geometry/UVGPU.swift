import Foundation
import simd

//  UVGPU.swift — Metal side of the UV kit (ticket item M3): the backend switch, the device, and shared MSL helpers.
//
//  Guarantees: every UV operation that has a GPU twin (`UVProjectionGPU`, `UVUnwrapGPU`, `UVReportGPU`,
//  `UVPackGPU`) takes a `UVBackend`; `.cpu` runs the original code untouched (the oracle), `.gpu` runs Metal when a
//  device exists (otherwise silently the CPU), `.auto` picks Metal only above a size where it pays. The MSL is
//  compiled from source at first use (the target has no resources), as `SDFGPU` does; with no device `UVGPU.shared`
//  is nil and the GPU tests XCTSkip. All kernels compute in Float: the GPU results match the CPU's Double ones to
//  ~1e-4 relative, not bit for bit.
//  Stays on the CPU, and why: welding/adjacency/island discovery (`UVTopology`: hash joins and union-find, data
//  dependent branching), seam paths, convex hulls (a sort, and tiny next to the point counts the kernels see), and
//  the shelf packer's 60-step binary search over a few hundred boxes.

public enum UVBackend: Sendable {
    case cpu, gpu, auto

    /// Whether to use Metal for `work` units (triangles, points) when a device exists.
    func useGPU(work: Int, threshold: Int = 2048) -> Bool {
        #if canImport(Metal)
        guard UVGPU.shared != nil else { return false }
        switch self { case .cpu: return false; case .gpu: return true; case .auto: return work >= threshold }
        #else
        return false
        #endif
    }
}

#if canImport(Metal)
import Metal

final class UVGPU: @unchecked Sendable {
    static let shared: UVGPU? = UVGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()
    private let library: MTLLibrary

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        let src = [UVGPU.common, UVProjectionGPU.source, UVUnwrapGPU.source, UVReportGPU.source, UVPackGPU.source]
            .joined(separator: "\n")
        guard let lib = try? d.makeLibrary(source: src, options: nil) else { return nil }
        (device, queue, library) = (d, q, lib)
    }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        let p = try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
        pipelines[name] = p
        return p
    }

    func buffer<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress ?? UnsafeRawPointer(bitPattern: 16)!,
                                              length: Swift.max($0.count, 16), options: .storageModeShared)! }
    }
    func empty(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: Swift.max(bytes, 16), options: .storageModeShared)! }

    static func read<T>(_ b: MTLBuffer, _ type: T.Type, count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: T.self), count: count))
    }

    /// One 1-D dispatch of `n` threads (a no-op for 0).
    func dispatch(_ enc: MTLComputeCommandEncoder, _ name: String, _ n: Int, group: Int = 256) {
        guard n > 0 else { return }
        let p = pipeline(name)
        enc.setComputePipelineState(p)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: Swift.min(group, p.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }
    /// `groups` threadgroups of `size` threads (reductions with one group per item).
    func dispatchGroups(_ enc: MTLComputeCommandEncoder, _ name: String, groups: Int, size: Int = 256) {
        guard groups > 0 else { return }
        enc.setComputePipelineState(pipeline(name))
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: size, height: 1, depth: 1))
    }

    /// Run `encode` in one command buffer; false if it failed.
    @discardableResult
    func run(_ encode: (MTLComputeCommandEncoder) -> Void) -> Bool {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        encode(enc); enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }

    /// Positions as float4 (the GPU's layout for every mesh buffer).
    static func float4s(_ p: [Vec3]) -> [SIMD4<Float>] { p.map { SIMD4(Float($0.x), Float($0.y), Float($0.z), 0) } }

    static let common = """
    #include <metal_stdlib>
    using namespace metal;

    // Sum over a threadgroup (all threads must call it; `sh` has 32 floats).
    static float tg_sum(float v, threadgroup float* sh, uint tid, uint tgs) {
        float s = simd_sum(v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & 31) == 0) sh[tid >> 5] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float r = 0; for (uint i = 0; i < (tgs + 31) / 32; i++) r += sh[i];
        return r;
    }
    static float tg_max(float v, threadgroup float* sh, uint tid, uint tgs) {
        float s = simd_max(v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & 31) == 0) sh[tid >> 5] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float r = -INFINITY; for (uint i = 0; i < (tgs + 31) / 32; i++) r = max(r, sh[i]);
        return r;
    }
    static float tg_min(float v, threadgroup float* sh, uint tid, uint tgs) { return -tg_max(-v, sh, tid, tgs); }
    """
}
#endif
