//  HalfEdgeMesh+GPU — flat (SoA) topology for GPU kernels, and the bulk per-element maths on the GPU.
//
//  `soa()` exports the topology as contiguous Int32 / Float arrays (the layout a kernel reads): halfedge
//  origin / next / prev / twin / face, face -> halfedge, vertex -> halfedge; dead slots are -1 (`heFace == -1`).
//  On the GPU (Metal compute, MSL compiled from source at first use like `SDFGPU`; `HalfEdgeGPU.shared` is nil
//  without a device): per-face area vectors (Newell, so |v|/2 is the area and v/|v| the normal) and per-vertex
//  area-weighted normals (each vertex walks its own fan through next / twin). Both have CPU twins
//  (`faceAreaVectors`, `vertexNormals`, the oracle and the fallback); the GPU runs in Float, so it matches the
//  CPU to ~1e-5 relative. NOT on the GPU: the topology edits themselves, edge loops / rings (a handful of
//  steps along one path) and the Mesh3 weld and triangulation (hash-table and ear-clip work); those stay CPU.

import Foundation
import simd

public struct HalfEdgeSoA: Sendable {
    public var origin: [Int32], next: [Int32], prev: [Int32], twin: [Int32], face: [Int32]
    public var faceHalfedge: [Int32], vertexHalfedge: [Int32]
    public var positions: [SIMD4<Float>]
}

public extension HalfEdgeMesh {
    func soa() -> HalfEdgeSoA {
        func i32(_ a: [Int]) -> [Int32] { a.map { Int32($0) } }
        return HalfEdgeSoA(origin: i32(heOrigin), next: i32(heNext), prev: i32(hePrev), twin: i32(heTwin), face: i32(heFace),
                           faceHalfedge: i32(faceHE), vertexHalfedge: i32(vertexHE),
                           positions: positions.map { SIMD4(Float($0.x), Float($0.y), Float($0.z), 0) })
    }

    /// Newell vector per face slot (zero for dead faces): |v| / 2 = area, v / |v| = normal. CPU twin of the kernel.
    func faceAreaVectors() -> [Vec3] {
        (0..<faceHE.count).map { faceHE[$0] >= 0 ? HalfEdgeMesh.newell(corners(of: $0)) : .zero }
    }
    /// Area-weighted vertex normals (unit; zero for isolated vertices). CPU twin of the kernel.
    func vertexNormals() -> [Vec3] {
        let av = faceAreaVectors()
        return (0..<positions.count).map { v in
            normalize3(faces(around: v).reduce(Vec3.zero) { $0 + av[$1] })
        }
    }
    /// Same results from the GPU when a device exists, else the CPU twin.
    func faceAreaVectorsGPU() -> [Vec3] { HalfEdgeGPU.shared?.faceVectors(soa()) ?? faceAreaVectors() }
    func vertexNormalsGPU() -> [Vec3] { HalfEdgeGPU.shared?.vertexNormals(soa()) ?? vertexNormals() }
}

#if canImport(Metal)
import Metal

public final class HalfEdgeGPU: @unchecked Sendable {
    public static let shared: HalfEdgeGPU? = HalfEdgeGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    let facePipe: MTLComputePipelineState, vertPipe: MTLComputePipelineState

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: HalfEdgeGPU.source, options: nil),
              let ff = lib.makeFunction(name: "he_face_vectors"), let vf = lib.makeFunction(name: "he_vertex_normals"),
              let fp = try? d.makeComputePipelineState(function: ff), let vp = try? d.makeComputePipelineState(function: vf)
        else { return nil }
        (device, queue, facePipe, vertPipe) = (d, q, fp, vp)
    }

    private func buf<T>(_ a: [T]) -> MTLBuffer? {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress ?? UnsafeRawPointer(bitPattern: 16)!, length: Swift.max($0.count, 16), options: .storageModeShared) }
    }

    private func run(_ pipe: MTLComputePipelineState, count: Int, _ buffers: [MTLBuffer], out: MTLBuffer) -> Bool {
        guard count > 0, let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return count == 0 }
        var n = UInt32(count)
        enc.setComputePipelineState(pipe)
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        enc.setBuffer(out, offset: 0, index: buffers.count)
        enc.setBytes(&n, length: 4, index: buffers.count + 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }

    private func vectors(_ b: MTLBuffer, _ n: Int) -> [Vec3] {
        let p = UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: n)
        return p.map { Vec3(Double($0.x), Double($0.y), Double($0.z)) }
    }

    func faceVectors(_ s: HalfEdgeSoA) -> [Vec3]? {
        let n = s.faceHalfedge.count
        guard let nx = buf(s.next), let fh = buf(s.faceHalfedge), let org = buf(s.origin), let pos = buf(s.positions),
              let out = device.makeBuffer(length: Swift.max(n, 1) * 16, options: .storageModeShared),
              run(facePipe, count: n, [fh, nx, org, pos], out: out) else { return nil }
        return vectors(out, n)
    }

    func vertexNormals(_ s: HalfEdgeSoA) -> [Vec3]? {
        guard let fv = faceVectors(s) else { return nil }
        let fvec = fv.map { SIMD4(Float($0.x), Float($0.y), Float($0.z), 0) }
        let n = s.vertexHalfedge.count
        guard let vh = buf(s.vertexHalfedge), let nx = buf(s.next), let pv = buf(s.prev), let tw = buf(s.twin),
              let fc = buf(s.face), let fb = buf(fvec),
              let out = device.makeBuffer(length: Swift.max(n, 1) * 16, options: .storageModeShared),
              run(vertPipe, count: n, [vh, nx, pv, tw, fc, fb], out: out) else { return nil }
        return vectors(out, n)
    }

    static let source = """
#include <metal_stdlib>
using namespace metal;

kernel void he_face_vectors(device const int* faceHE [[buffer(0)]], device const int* nxt [[buffer(1)]],
                            device const int* org [[buffer(2)]], device const float4* pos [[buffer(3)]],
                            device float4* out [[buffer(4)]], constant uint& n [[buffer(5)]],
                            uint f [[thread_position_in_grid]]) {
    if (f >= n) return;
    int h0 = faceHE[f];
    if (h0 < 0) { out[f] = float4(0); return; }
    float3 sum = 0; int h = h0; int guard = 0;
    do {
        float3 a = pos[org[h]].xyz, b = pos[org[nxt[h]]].xyz;
        sum += float3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
        h = nxt[h]; guard++;
    } while (h != h0 && h >= 0 && guard < 4096);
    out[f] = float4(sum, 0);
}

// one thread per vertex: walk the fan forward (next(twin(h))), then backward (twin(prev(h))) if it was open.
kernel void he_vertex_normals(device const int* vhe [[buffer(0)]], device const int* nxt [[buffer(1)]],
                              device const int* prv [[buffer(2)]], device const int* twn [[buffer(3)]],
                              device const int* fac [[buffer(4)]], device const float4* fv [[buffer(5)]],
                              device float4* out [[buffer(6)]], constant uint& n [[buffer(7)]],
                              uint v [[thread_position_in_grid]]) {
    if (v >= n) return;
    int h0 = vhe[v];
    if (h0 < 0) { out[v] = float4(0); return; }
    float3 sum = fv[fac[h0]].xyz;
    int h = h0; bool closed = false; int guard = 0;
    while (guard++ < 4096) {
        int t = twn[h]; if (t < 0) break;
        h = nxt[t];
        if (h == h0) { closed = true; break; }
        sum += fv[fac[h]].xyz;
    }
    if (!closed) {
        h = h0; guard = 0;
        while (guard++ < 4096) {
            int t = twn[prv[h]]; if (t < 0) break;
            h = t;
            sum += fv[fac[h]].xyz;
        }
    }
    float l = length(sum);
    out[v] = float4(l > 0 ? sum / l : sum, 0);
}
"""
}
#else
public final class HalfEdgeGPU: @unchecked Sendable {
    public static let shared: HalfEdgeGPU? = nil
    func faceVectors(_ s: HalfEdgeSoA) -> [Vec3]? { nil }
    func vertexNormals(_ s: HalfEdgeSoA) -> [Vec3]? { nil }
}
#endif
