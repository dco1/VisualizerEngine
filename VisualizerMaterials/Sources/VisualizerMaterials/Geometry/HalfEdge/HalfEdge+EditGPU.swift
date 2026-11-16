//  HalfEdge+EditGPU — the bulk per-vertex arithmetic of the edit tools on the GPU (Metal compute, MSL compiled
//  from source at first use like `HalfEdgeGPU` / `SDFGPU`; `HalfEdgeEditGPU.shared` is nil without a device).
//    he_offsets: one thread per displaced vertex gathers the area vectors of its selected faces (a CSR list the
//                CPU builds, O(selected corners)) and writes the push/pull displacement (averaged normal, or the
//                per-face mitre, capped by the mitre limit). CPU twin: `HalfEdgeMesh.normalOffset`.
//    he_smooth:  one Laplacian step per vertex (Jacobi: reads the old buffer, writes the new one), ping-ponged
//                for all iterations inside one command buffer; pinned vertices copy through. CPU twin:
//                `HalfEdgeMesh.laplacianSmoothCPU`.
//  Float on the GPU against Double on the CPU: they agree to ~1e-5 relative (the tests hold them to that).
//  NOT on the GPU: the topology edits and the CSR / neighbour lists themselves (pointer chasing on small rings).

import Foundation
import simd

#if canImport(Metal)
import Metal

public final class HalfEdgeEditGPU: @unchecked Sendable {
    public static let shared: HalfEdgeEditGPU? = HalfEdgeEditGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    let offsetPipe: MTLComputePipelineState, smoothPipe: MTLComputePipelineState

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: HalfEdgeEditGPU.source, options: nil),
              let of = lib.makeFunction(name: "he_offsets"), let sf = lib.makeFunction(name: "he_smooth"),
              let op = try? d.makeComputePipelineState(function: of), let sp = try? d.makeComputePipelineState(function: sf)
        else { return nil }
        (device, queue, offsetPipe, smoothPipe) = (d, q, op, sp)
    }

    private func buf<T>(_ a: [T]) -> MTLBuffer? {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress ?? UnsafeRawPointer(bitPattern: 16)!, length: Swift.max($0.count, 16), options: .storageModeShared) }
    }

    /// Displacements for vertices given the area vectors of the faces displacing each.
    func offsets(_ lists: [[Vec3]], distance: Double, perFace: Bool) -> [Vec3]? {
        let n = lists.count
        guard n > 0 else { return [] }
        var starts: [UInt32] = [0], vecs: [SIMD4<Float>] = []
        for l in lists { for v in l { vecs.append(SIMD4(Float(v.x), Float(v.y), Float(v.z), 0)) }; starts.append(UInt32(vecs.count)) }
        guard let sb = buf(starts), let vb = buf(vecs), let out = device.makeBuffer(length: n * 16, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var params = SIMD4<Float>(Float(distance), perFace ? 1 : 0, Float(HEEdit.miterLimit), 0)
        var count = UInt32(n)
        enc.setComputePipelineState(offsetPipe)
        enc.setBuffer(sb, offset: 0, index: 0); enc.setBuffer(vb, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&params, length: 16, index: 3); enc.setBytes(&count, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        let p = UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: n)
        return p.map { Vec3(Double($0.x), Double($0.y), Double($0.z)) }
    }

    /// `iterations` Jacobi Laplacian steps over every vertex slot; `neighbours[i]` empty or `pinned[i]` = stays.
    func smooth(positions: [Vec3], neighbours: [[Int]], pinned: [Bool], lambda: Double, iterations: Int) -> [Vec3]? {
        let n = positions.count
        guard n > 0 else { return [] }
        var starts: [UInt32] = [0], nbrs: [UInt32] = []
        for l in neighbours { nbrs += l.map { UInt32($0) }; starts.append(UInt32(nbrs.count)) }
        let pos = positions.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 0) }
        guard let sb = buf(starts), let nb = buf(nbrs), let pb = buf(pinned.map { UInt32($0 ? 1 : 0) }),
              let a = buf(pos), let b = device.makeBuffer(length: n * 16, options: .storageModeShared),
              let cb = queue.makeCommandBuffer() else { return nil }
        var lam = Float(lambda), count = UInt32(n)
        var src = a, dst = b
        for _ in 0..<Swift.max(iterations, 0) {
            guard let enc = cb.makeComputeCommandEncoder() else { return nil }
            enc.setComputePipelineState(smoothPipe)
            enc.setBuffer(sb, offset: 0, index: 0); enc.setBuffer(nb, offset: 0, index: 1); enc.setBuffer(pb, offset: 0, index: 2)
            enc.setBuffer(src, offset: 0, index: 3); enc.setBuffer(dst, offset: 0, index: 4)
            enc.setBytes(&lam, length: 4, index: 5); enc.setBytes(&count, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
            swap(&src, &dst)
        }
        cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        let p = UnsafeBufferPointer(start: src.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: n)
        return p.map { Vec3(Double($0.x), Double($0.y), Double($0.z)) }
    }

    static let source = """
#include <metal_stdlib>
using namespace metal;

kernel void he_offsets(device const uint* starts [[buffer(0)]], device const float4* vecs [[buffer(1)]],
                       device float4* out [[buffer(2)]], constant float4& prm [[buffer(3)]],
                       constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint a = starts[i], b = starts[i + 1];
    float d = prm.x;
    float3 s = float3(0), m = float3(0);
    for (uint k = a; k < b; k++) {
        float3 v = vecs[k].xyz; s += v;
        float l = length(v); if (l > 1e-20f) m += v / l;
    }
    float3 r = float3(0);
    if (prm.y < 0.5f) {
        float l = length(s); r = l > 1e-20f ? s / l * d : float3(0);
    } else {
        m /= max(float(b - a), 1.0f);
        float mm = dot(m, m);
        if (mm > 1e-8f) {
            r = m * (d / mm);
            float cap = prm.z * fabs(d), l = length(r);
            if (l > cap) r *= cap / l;
        }
    }
    out[i] = float4(r, 0);
}

kernel void he_smooth(device const uint* starts [[buffer(0)]], device const uint* nbrs [[buffer(1)]],
                      device const uint* pinned [[buffer(2)]], device const float4* src [[buffer(3)]],
                      device float4* dst [[buffer(4)]], constant float& lambda [[buffer(5)]],
                      constant uint& n [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint a = starts[i], b = starts[i + 1];
    float3 p = src[i].xyz;
    if (pinned[i] != 0 || b == a) { dst[i] = float4(p, 0); return; }
    float3 avg = float3(0);
    for (uint k = a; k < b; k++) avg += src[nbrs[k]].xyz;
    avg /= float(b - a);
    dst[i] = float4(p + lambda * (avg - p), 0);
}
"""
}
#else
public final class HalfEdgeEditGPU: @unchecked Sendable {
    public static let shared: HalfEdgeEditGPU? = nil
    func offsets(_ lists: [[Vec3]], distance: Double, perFace: Bool) -> [Vec3]? { nil }
    func smooth(positions: [Vec3], neighbours: [[Int]], pinned: [Bool], lambda: Double, iterations: Int) -> [Vec3]? { nil }
}
#endif
