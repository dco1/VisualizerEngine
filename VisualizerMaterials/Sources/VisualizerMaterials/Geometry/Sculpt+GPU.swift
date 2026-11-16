//  Sculpt+GPU.swift — Metal brush kernels for the sculpt kit (ticket item I).
//
//  The field lives in a `SculptStore`'s page-aligned memory, wrapped with no copy as a shared `MTLBuffer`; a stamp is two
//  dispatches over its brush box only (`sculpt_copy` to a scratch, then `sculpt_apply` reading the scratch and writing
//  the field), encoded in one command buffer and NOT waited on: the queue is serial, and `SculptVolume.sync()` (called by
//  anything that reads the field on the CPU, by `end()`, `mesh()`, `undo()`) waits for the last one. Also here: the 2×
//  upsample of `refine()` and the vertex projection of `mesh()`. The CPU twin is `SculptKernel.swift`; the MSL below is
//  the same arithmetic in the same order, and `SculptTests` holds the two within 1e-4.
//  Compiled from source at first use (as `SDF+GPU.swift`), on the same device; everything returns false / nil without one
//  and the callers fall back to the CPU.

import Foundation
import simd

#if canImport(Metal)
import Metal

final class SculptGPUContext: @unchecked Sendable {
    static let shared: SculptGPUContext? = SculptGPUContext()
    let device: MTLDevice, queue: MTLCommandQueue
    let copyPipe: MTLComputePipelineState, applyPipe: MTLComputePipelineState
    let upPipe: MTLComputePipelineState, refinePipe: MTLComputePipelineState
    var scratch: MTLBuffer?

    private init?() {
        guard let g = SDFGPU.shared, let lib = try? g.device.makeLibrary(source: SculptGPUContext.source, options: nil) else { return nil }
        func pipe(_ n: String) -> MTLComputePipelineState? {
            lib.makeFunction(name: n).flatMap { try? g.device.makeComputePipelineState(function: $0) }
        }
        guard let c = pipe("sculpt_copy"), let a = pipe("sculpt_apply"), let u = pipe("sculpt_upsample"),
              let r = pipe("sculpt_refine") else { return nil }
        (device, queue, copyPipe, applyPipe, upPipe, refinePipe) = (g.device, g.queue, c, a, u, r)
    }

    func scratch(floats: Int) -> MTLBuffer? {
        if let s = scratch, s.length >= floats * 4 { return s }
        scratch = device.makeBuffer(length: max(floats, 1024) * 4 * 2, options: .storageModeShared)
        return scratch
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct SculptParams { float4 centerRadius; float4 normalAmount; float4 drag; float4 gridOrigin; uint4 lo; uint4 size; uint4 dims; };
    struct RefineParams { float4 origin; uint nx, ny, nz, count; float cell, h; uint iterations; };

    static float falloff(uint kind, float t) {
        if (t >= 1) return 0;
        switch (kind) {
        case 0: { float s = 1 - t * t; return s * s; }
        case 1: return 1 - t;
        case 2: { float s = 1 - t; return s * s; }
        case 3: return 1;
        case 4: return sqrt(1 - t * t);
        default: return 1 - (3 * t * t - 2 * t * t * t);
        }
    }

    struct View {
        device float* field; device float* scratch; constant SculptParams& P;
        float read(int i, int j, int k) const {
            int3 lo = int3(P.lo.xyz), sz = int3(P.size.xyz);
            int3 l = int3(i, j, k) - lo;
            if (all(l >= 0) && all(l < sz)) return scratch[(l.z * sz.y + l.y) * sz.x + l.x];
            return field[(k * (int)P.dims.y + j) * (int)P.dims.x + i];
        }
        float tri(float3 q) const {
            float3 g = (q - P.gridOrigin.xyz) / P.gridOrigin.w;
            float3 n = float3(P.dims.xyz);
            g = clamp(g, float3(0), n - 1);
            int i = min(int(g.x), (int)P.dims.x - 2), j = min(int(g.y), (int)P.dims.y - 2), k = min(int(g.z), (int)P.dims.z - 2);
            float3 f = g - float3(i, j, k);
            float x00 = read(i, j, k) * (1 - f.x) + read(i + 1, j, k) * f.x;
            float x10 = read(i, j + 1, k) * (1 - f.x) + read(i + 1, j + 1, k) * f.x;
            float x01 = read(i, j, k + 1) * (1 - f.x) + read(i + 1, j, k + 1) * f.x;
            float x11 = read(i, j + 1, k + 1) * (1 - f.x) + read(i + 1, j + 1, k + 1) * f.x;
            return (x00 * (1 - f.y) + x10 * f.y) * (1 - f.z) + (x01 * (1 - f.y) + x11 * f.y) * f.z;
        }
    };

    kernel void sculpt_copy(device const float* field [[buffer(0)]], device float* scratch [[buffer(1)]],
                            constant SculptParams& P [[buffer(2)]], uint3 id [[thread_position_in_grid]]) {
        if (id.x >= P.size.x || id.y >= P.size.y || id.z >= P.size.z) return;
        uint3 g = P.lo.xyz + id;
        scratch[(id.z * P.size.y + id.y) * P.size.x + id.x] = field[(g.z * P.dims.y + g.y) * P.dims.x + g.x];
    }

    kernel void sculpt_apply(device float* field [[buffer(0)]], device float* scratch [[buffer(1)]],
                             constant SculptParams& P [[buffer(2)]], device const float* mask [[buffer(3)]],
                             uint3 id [[thread_position_in_grid]]) {
        if (id.x >= P.size.x || id.y >= P.size.y || id.z >= P.size.z) return;
        int3 gi = int3(P.lo.xyz + id);
        int nx = P.dims.x, ny = P.dims.y, nz = P.dims.z;
        uint idx = (gi.z * ny + gi.y) * nx + gi.x;
        float3 p = P.gridOrigin.xyz + float3(gi) * P.gridOrigin.w;
        float3 c = P.centerRadius.xyz; float r = P.centerRadius.w;
        float3 nrm = P.normalAmount.xyz; float amt = P.normalAmount.w;
        float3 drag = P.drag.xyz; float factor = P.drag.w;
        uint kind = P.lo.w, fk = P.size.w;
        float3 d = p - c;
        float t = sqrt(d.x * d.x + d.y * d.y + d.z * d.z) / r;
        float w = falloff(fk, t);
        if (w <= 0) return;
        if (P.dims.w != 0) w *= 1 - mask[idx];
        if (w <= 0) return;
        View V = { field, scratch, P };
        float v = scratch[(id.z * P.size.y + id.y) * P.size.x + id.x];
        float out = v;
        switch (kind) {
        case 0: out = V.tri(p - nrm * (amt * w)); break;
        case 1: out = V.tri(p - drag * (w * factor)); break;
        case 2: {
            float avg = (V.read(max(gi.x - 1, 0), gi.y, gi.z) + V.read(min(gi.x + 1, nx - 1), gi.y, gi.z)
                       + V.read(gi.x, max(gi.y - 1, 0), gi.z) + V.read(gi.x, min(gi.y + 1, ny - 1), gi.z)
                       + V.read(gi.x, gi.y, max(gi.z - 1, 0)) + V.read(gi.x, gi.y, min(gi.z + 1, nz - 1))) / 6;
            out = v + (avg - v) * min(max(w * factor, 0.0f), 1.0f); break; }
        case 3: out = v - amt * w; break;
        case 4: {
            float wn = falloff(fk, min(t / 0.45f, 1.0f));
            if (P.dims.w != 0) wn *= 1 - mask[idx];
            out = v + amt * wn; break; }
        case 5: {
            float plane = d.x * nrm.x + d.y * nrm.y + d.z * nrm.z;
            float band = min(max(1 - fabs(v) / r, 0.0f), 1.0f);
            out = v + (plane - v) * min(max(w * factor, 0.0f), 1.0f) * band; break; }
        default: {
            float along = d.x * nrm.x + d.y * nrm.y + d.z * nrm.z;
            float3 perp = d - nrm * along;
            out = V.tri(p + perp * (min(factor, 0.9f) * w)); break; }
        }
        field[idx] = out;
    }

    kernel void sculpt_upsample(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                                constant uint4* dims [[buffer(2)]], uint3 id [[thread_position_in_grid]]) {
        uint3 n = dims[0].xyz, m = dims[1].xyz;
        if (id.x >= m.x || id.y >= m.y || id.z >= m.z) return;
        float3 g = float3(id) * 0.5f;
        g = clamp(g, float3(0), float3(n) - 1);
        int i = min(int(g.x), (int)n.x - 2), j = min(int(g.y), (int)n.y - 2), k = min(int(g.z), (int)n.z - 2);
        float3 f = g - float3(i, j, k);
        #define S(a, b, c) src[((k + c) * n.y + (j + b)) * n.x + (i + a)]
        float x00 = S(0,0,0) * (1 - f.x) + S(1,0,0) * f.x;
        float x10 = S(0,1,0) * (1 - f.x) + S(1,1,0) * f.x;
        float x01 = S(0,0,1) * (1 - f.x) + S(1,0,1) * f.x;
        float x11 = S(0,1,1) * (1 - f.x) + S(1,1,1) * f.x;
        dst[(id.z * m.y + id.y) * m.x + id.x] = (x00 * (1 - f.y) + x10 * f.y) * (1 - f.z) + (x01 * (1 - f.y) + x11 * f.y) * f.z;
    }

    struct FView {
        device const float* f; constant RefineParams& R;
        float at(int i, int j, int k) const { return f[(k * (int)R.ny + j) * (int)R.nx + i]; }
        float tri(float3 p) const {
            float3 g = clamp((p - R.origin.xyz) / R.cell, float3(0), float3(R.nx, R.ny, R.nz) - 1);
            int i = min(int(g.x), (int)R.nx - 2), j = min(int(g.y), (int)R.ny - 2), k = min(int(g.z), (int)R.nz - 2);
            float3 q = g - float3(i, j, k);
            float x00 = at(i, j, k) * (1 - q.x) + at(i + 1, j, k) * q.x;
            float x10 = at(i, j + 1, k) * (1 - q.x) + at(i + 1, j + 1, k) * q.x;
            float x01 = at(i, j, k + 1) * (1 - q.x) + at(i + 1, j, k + 1) * q.x;
            float x11 = at(i, j + 1, k + 1) * (1 - q.x) + at(i + 1, j + 1, k + 1) * q.x;
            return (x00 * (1 - q.y) + x10 * q.y) * (1 - q.z) + (x01 * (1 - q.y) + x11 * q.y) * q.z;
        }
        float3 grad(float3 p, float h) const {
            return float3(tri(p + float3(h, 0, 0)) - tri(p - float3(h, 0, 0)), tri(p + float3(0, h, 0)) - tri(p - float3(0, h, 0)),
                          tri(p + float3(0, 0, h)) - tri(p - float3(0, 0, h))) / (2 * h);
        }
    };

    kernel void sculpt_refine(device float4* pos [[buffer(0)]], device float4* nrm [[buffer(1)]],
                              device const float4* cmin [[buffer(2)]], device const float* field [[buffer(3)]],
                              constant RefineParams& R [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= R.count) return;
        FView F = { field, R };
        float3 p = pos[i].xyz, lo = cmin[i].xyz, hi = lo + R.cell;
        for (uint it = 0; it < R.iterations; it++) {
            float d = F.tri(p); float3 g = F.grad(p, R.h); float g2 = dot(g, g);
            if (g2 > 1e-12f) p = clamp(p - g * (d / g2), lo, hi);
        }
        float3 g = F.grad(p, R.h); float l = length(g);
        pos[i] = float4(p, 0);
        nrm[i] = float4(l > 1e-6f ? g / l : float3(0, 1, 0), 0);
    }
    """
}

enum SculptGPU {
    static var available: Bool { SculptGPUContext.shared != nil }

    static func wrap(_ s: SculptStore) -> AnyObject? {
        guard let c = SculptGPUContext.shared else { return nil }
        return c.device.makeBuffer(bytesNoCopy: s.ptr, length: s.bytes, options: .storageModeShared, deallocator: nil)
    }

    static func wait(_ v: SculptVolume) {
        guard let cb = v.pending as? MTLCommandBuffer else { return }
        cb.waitUntilCompleted()
        v.pending = nil
    }

    static func apply(_ P: SculptParams, to v: SculptVolume) -> Bool {
        guard let c = SculptGPUContext.shared, let field = v.store.gpu as? MTLBuffer,
              let scratch = c.scratch(floats: Int(P.size.x * P.size.y * P.size.z)),
              let cb = c.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        let mask = (v.maskStore?.gpu as? MTLBuffer) ?? field
        var p = P
        let grid = MTLSize(width: Int(P.size.x), height: Int(P.size.y), depth: Int(P.size.z))
        let tg = MTLSize(width: 8, height: 8, depth: 4)
        enc.setComputePipelineState(c.copyPipe)
        enc.setBuffer(field, offset: 0, index: 0); enc.setBuffer(scratch, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SculptParams>.stride, index: 2)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.setComputePipelineState(c.applyPipe)
        enc.setBuffer(field, offset: 0, index: 0); enc.setBuffer(scratch, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<SculptParams>.stride, index: 2)
        enc.setBuffer(mask, offset: 0, index: 3)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding(); cb.commit()
        v.pending = cb
        return true
    }

    static func upsample(from a: SculptStore, _ n: SIMD3<Int>, to b: SculptStore, _ m: SIMD3<Int>) -> Bool {
        guard let c = SculptGPUContext.shared, let src = a.gpu as? MTLBuffer, let dst = b.gpu as? MTLBuffer,
              let cb = c.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        var dims = [SIMD4<UInt32>(UInt32(n.x), UInt32(n.y), UInt32(n.z), 0), SIMD4<UInt32>(UInt32(m.x), UInt32(m.y), UInt32(m.z), 0)]
        enc.setComputePipelineState(c.upPipe)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1)
        enc.setBytes(&dims, length: 32, index: 2)
        enc.dispatchThreads(MTLSize(width: m.x, height: m.y, depth: m.z), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.status == .completed
    }

    /// Vertex positions and normals from the resident field, as `SculptKernel.refine` makes them.
    static func refine(_ v: SculptVolume, _ verts: [SDFNetVertex], iterations: Int) -> [(Vec3, Vec3)]? {
        guard let c = SculptGPUContext.shared, let field = v.store.gpu as? MTLBuffer, !verts.isEmpty else { return nil }
        func buf(_ a: [SIMD4<Float>]) -> MTLBuffer? {
            a.withUnsafeBytes { c.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        }
        let pos = verts.map { SIMD4<Float>(Float($0.position.x), Float($0.position.y), Float($0.position.z), 0) }
        let cmin = verts.map { SIMD4<Float>(Float($0.cellMin.x), Float($0.cellMin.y), Float($0.cellMin.z), 0) }
        struct R { var origin: SIMD4<Float>; var nx: UInt32, ny: UInt32, nz: UInt32, count: UInt32; var cell: Float, h: Float; var iterations: UInt32 }
        var r = R(origin: SIMD4(Float(v.origin.x), Float(v.origin.y), Float(v.origin.z), Float(v.cell)), nx: UInt32(v.samples.x),
                  ny: UInt32(v.samples.y), nz: UInt32(v.samples.z), count: UInt32(verts.count), cell: Float(v.cell),
                  h: Float(v.cell * 0.25), iterations: UInt32(iterations))
        guard let pb = buf(pos), let cb2 = buf(cmin), let nb = c.device.makeBuffer(length: verts.count * 16, options: .storageModeShared),
              let cb = c.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(c.refinePipe)
        enc.setBuffer(pb, offset: 0, index: 0); enc.setBuffer(nb, offset: 0, index: 1); enc.setBuffer(cb2, offset: 0, index: 2)
        enc.setBuffer(field, offset: 0, index: 3); enc.setBytes(&r, length: MemoryLayout<R>.stride, index: 4)
        enc.dispatchThreads(MTLSize(width: verts.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        let P = UnsafeBufferPointer(start: pb.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: verts.count)
        let N = UnsafeBufferPointer(start: nb.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: verts.count)
        return (0..<verts.count).map { (Vec3(Double(P[$0].x), Double(P[$0].y), Double(P[$0].z)),
                                        Vec3(Double(N[$0].x), Double(N[$0].y), Double(N[$0].z))) }
    }
}
#else
enum SculptGPU {
    static var available: Bool { false }
    static func wrap(_ s: SculptStore) -> AnyObject? { nil }
    static func wait(_ v: SculptVolume) {}
    static func apply(_ P: SculptParams, to v: SculptVolume) -> Bool { false }
    static func upsample(from a: SculptStore, _ n: SIMD3<Int>, to b: SculptStore, _ m: SIMD3<Int>) -> Bool { false }
    static func refine(_ v: SculptVolume, _ verts: [SDFNetVertex], iterations: Int) -> [(Vec3, Vec3)]? { nil }
}
#endif
