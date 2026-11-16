//  Metal kernels for the mesh-intersection boolean (`BooleanGPU`), each with a CPU twin that is the oracle.
//
//   · `bool_broadphase` — every A-triangle box against every B-triangle box (brute force, one thread per A
//     triangle, hits appended with an atomic counter). Boxes are widened in Float by a relative margin so the
//     GPU set is a SUPERSET of the exact set; the CPU narrow phase (exact signs) discards the extras.
//     CPU twin: `BoxBVH.query` (see `BooleanJob.findPairs`). Used when |A|·|B| ≥ 1e6.
//   · `bool_winding` — generalized winding number of many points against a closed mesh (one thread per point,
//     Float). CPU twin: `BooleanJob.windingCPU` (Double). Used when points·triangles ≥ 2e5.
//
//  NOT on the GPU: the narrow phase (exact-sign triangle–triangle tests), the constrained retriangulation and
//  the welding/assembly: branchy, exact-arithmetic or hash-heavy work. The MSL is compiled from source on first
//  use (this package has no resources); `BooleanGPU.shared` is nil without a Metal device.

import Foundation
import simd

#if canImport(Metal)
import Metal

final class BooleanGPU: @unchecked Sendable {
    static let shared: BooleanGPU? = BooleanGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private let broad: MTLComputePipelineState, wind: MTLComputePipelineState

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: BooleanGPU.source, options: nil),
              let bf = lib.makeFunction(name: "bool_broadphase"), let wf = lib.makeFunction(name: "bool_winding"),
              let bp = try? d.makeComputePipelineState(function: bf), let wp = try? d.makeComputePipelineState(function: wf)
        else { return nil }
        (device, queue, broad, wind) = (d, q, bp, wp)
    }

    private func buffer<T>(_ a: [T]) -> MTLBuffer? {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: Swift.max($0.count, 16), options: .storageModeShared) }
    }

    /// Candidate (A, B) pairs whose widened boxes overlap, or nil when the output overflowed / no device.
    func broadphase(_ a: ([Vec3], [Vec3]), _ b: ([Vec3], [Vec3])) -> [SIMD2<Int32>]? {
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for v in a.0 + b.0 { lo = simd_min(lo, v) }
        for v in a.1 + b.1 { hi = simd_max(hi, v) }
        let scale = max(len3(hi - lo), 1e-12)
        let margin = Float(scale * 2e-6)
        func f4(_ v: Vec3, _ m: Float) -> SIMD4<Float> {
            SIMD4(Float(v.x - lo.x) + m, Float(v.y - lo.y) + m, Float(v.z - lo.z) + m, 0)
        }
        let la = a.0.map { f4($0, -margin) }, ha = a.1.map { f4($0, margin) }
        let lb = b.0.map { f4($0, -margin) }, hb = b.1.map { f4($0, margin) }
        let capacity = 16_000_000
        guard let bla = buffer(la), let bha = buffer(ha), let blb = buffer(lb), let bhb = buffer(hb),
              let out = device.makeBuffer(length: capacity * 8, options: .storageModeShared),
              let counter = device.makeBuffer(length: 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        counter.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        var params = SIMD4<UInt32>(UInt32(la.count), UInt32(lb.count), UInt32(capacity), 0)
        enc.setComputePipelineState(broad)
        for (i, bf) in [bla, bha, blb, bhb, out, counter].enumerated() { enc.setBuffer(bf, offset: 0, index: i) }
        enc.setBytes(&params, length: 16, index: 6)
        enc.dispatchThreads(MTLSize(width: la.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        let n = Int(counter.contents().load(as: UInt32.self))
        guard n <= capacity else { return nil }
        let p = out.contents().assumingMemoryBound(to: UInt32.self)
        return (0..<n).map { SIMD2(Int32(bitPattern: p[$0 * 2]), Int32(bitPattern: p[$0 * 2 + 1])) }
    }

    /// Winding number of each point against the triangles `corners` (3 consecutive entries each).
    func winding(points: [Vec3], corners: [Vec3]) -> [Float]? {
        guard !points.isEmpty, !corners.isEmpty else { return nil }
        var lo = Vec3(repeating: .infinity)
        for c in corners { lo = simd_min(lo, c) }
        for p in points { lo = simd_min(lo, p) }
        func f4(_ v: Vec3) -> SIMD4<Float> { SIMD4(Float(v.x - lo.x), Float(v.y - lo.y), Float(v.z - lo.z), 0) }
        let pts = points.map(f4), cs = corners.map(f4)
        guard let bp = buffer(pts), let bc = buffer(cs),
              let out = device.makeBuffer(length: points.count * 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var params = SIMD2<UInt32>(UInt32(points.count), UInt32(corners.count / 3))
        enc.setComputePipelineState(wind)
        enc.setBuffer(bp, offset: 0, index: 0); enc.setBuffer(bc, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&params, length: 8, index: 3)
        enc.dispatchThreads(MTLSize(width: points.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: points.count))
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void bool_broadphase(device const float4* loA [[buffer(0)]], device const float4* hiA [[buffer(1)]],
                                device const float4* loB [[buffer(2)]], device const float4* hiB [[buffer(3)]],
                                device uint2* out [[buffer(4)]], device atomic_uint* counter [[buffer(5)]],
                                constant uint4& n [[buffer(6)]], uint i [[thread_position_in_grid]]) {
        if (i >= n.x) return;
        float3 l = loA[i].xyz, h = hiA[i].xyz;
        for (uint j = 0; j < n.y; j++) {
            float3 bl = loB[j].xyz, bh = hiB[j].xyz;
            if (bl.x > h.x || bh.x < l.x || bl.y > h.y || bh.y < l.y || bl.z > h.z || bh.z < l.z) continue;
            uint k = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
            if (k < n.z) out[k] = uint2(i, j);
        }
    }

    kernel void bool_winding(device const float4* pts [[buffer(0)]], device const float4* cs [[buffer(1)]],
                             device float* out [[buffer(2)]], constant uint2& n [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
        if (i >= n.x) return;
        float3 p = pts[i].xyz;
        float sum = 0;
        for (uint t = 0; t < n.y; t++) {
            float3 a = cs[t * 3].xyz - p, b = cs[t * 3 + 1].xyz - p, c = cs[t * 3 + 2].xyz - p;
            float la = length(a), lb = length(b), lc = length(c);
            float num = dot(a, cross(b, c));
            float den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
            sum += 2.0f * atan2(num, den);
        }
        out[i] = sum / (4.0f * M_PI_F);
    }
    """
}
#else
final class BooleanGPU: @unchecked Sendable {
    static let shared: BooleanGPU? = nil
    func broadphase(_ a: ([Vec3], [Vec3]), _ b: ([Vec3], [Vec3])) -> [SIMD2<Int32>]? { nil }
    func winding(points: [Vec3], corners: [Vec3]) -> [Float]? { nil }
}
#endif
