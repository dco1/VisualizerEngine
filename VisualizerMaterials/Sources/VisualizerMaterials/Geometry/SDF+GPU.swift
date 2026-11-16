//  SDF+GPU.swift — Metal evaluation + vertex refinement of an `SDF` tree (ticket item C).
//
//  An SDF made only of primitives, placements, booleans (smooth or hard), offset, shell, negate (and so `carved`) is
//  flattened to a small stack program (`SDFInst`, 96 B each) that one compute kernel runs per grid sample
//  (`sdf_grid`) and per mesh vertex (`sdf_refine`: projection + gradient normal). The CPU path in `SDF.swift` is the
//  oracle: the tests hold the GPU grid and mesh to it within Float tolerance.
//  Not on the GPU (the program is nil, meshing falls back to the CPU): `.mesh`, `.custom`, `.twist`, `.bend`, and
//  trees deeper than the kernel's stacks. Only available where Metal is; the MSL is compiled from source at first
//  use (this target is a Swift package with no resources), and `SDFGPU.shared` is nil when there is no device.

import Foundation
import simd

/// One instruction; the layout is `SDFInst` in the MSL below.
struct SDFInst {
    var op: UInt32 = 0
    var k: Float = 0
    var s: Float = 1
    var mode: UInt32 = 0
    var a = SIMD4<Float>(repeating: 0)
    var b = SIMD4<Float>(repeating: 0)
    var r0 = SIMD4<Float>(repeating: 0)
    var r1 = SIMD4<Float>(repeating: 0)
    var r2 = SIMD4<Float>(repeating: 0)
}

public struct SDFProgram: @unchecked Sendable {
    let insts: [SDFInst]
    public var count: Int { insts.count }
}

public extension SDF {
    /// The GPU program for this tree, or nil when it holds something the kernel can't run.
    func gpuProgram() -> SDFProgram? {
        var out: [SDFInst] = []
        var valueDepth = 0, maxValues = 0, pointDepth = 0, maxPoints = 0
        func f(_ v: Vec3, _ w: Double = 0) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), Float(w)) }
        func push(_ i: SDFInst) { valueDepth += 1; maxValues = Swift.max(maxValues, valueDepth); out.append(i) }
        func emit(_ n: Node) -> Bool {
            switch n {
            case .sphere(let r): push(SDFInst(op: 0, a: f(.zero, r)))
            case .box(let h): push(SDFInst(op: 1, a: f(h)))
            case .roundedBox(let h, let r): push(SDFInst(op: 2, a: f(h), b: SIMD4(Float(r), 0, 0, 0)))
            case .capsule(let a, let b, let r): push(SDFInst(op: 3, a: f(a, r), b: f(b)))
            case .torus(let R, let r): push(SDFInst(op: 4, a: SIMD4(Float(R), Float(r), 0, 0)))
            case .cone(let h, let r1, let r2): push(SDFInst(op: 5, a: SIMD4(Float(h), Float(r1), Float(r2), 0)))
            case .plane(let nrm, let off): push(SDFInst(op: 6, a: f(nrm, off)))
            case .mesh, .custom, .twist, .bend: return false
            case .place(let pl, let c):
                let m = simd_double3x3(pl.rotation.inverse)       // rows of R⁻¹ are the columns' transpose
                let rows = (0..<3).map { r in SIMD4(Float(m[0][r]), Float(m[1][r]), Float(m[2][r]), 0) }
                out.append(SDFInst(op: 7, s: Float(pl.scale), a: f(pl.translation), r0: rows[0], r1: rows[1], r2: rows[2]))
                pointDepth += 1; maxPoints = Swift.max(maxPoints, pointDepth)
                guard emit(c) else { return false }
                out.append(SDFInst(op: 8, s: Float(pl.scale)))
                pointDepth -= 1
            case .union(let a, let b, let k): guard emit(a), emit(b) else { return false }
                out.append(SDFInst(op: 9, k: Float(k))); valueDepth -= 1
            case .subtract(let a, let b, let k): guard emit(a), emit(b) else { return false }
                out.append(SDFInst(op: 10, k: Float(k))); valueDepth -= 1
            case .intersect(let a, let b, let k): guard emit(a), emit(b) else { return false }
                out.append(SDFInst(op: 11, k: Float(k))); valueDepth -= 1
            case .negate(let c): guard emit(c) else { return false }
                out.append(SDFInst(op: 12))
            case .offset(let c, let r): guard emit(c) else { return false }
                out.append(SDFInst(op: 13, k: Float(r)))
            case .shell(let c, let t, let side): guard emit(c) else { return false }
                out.append(SDFInst(op: 14, k: Float(t), mode: side.rawValue))
            }
            return true
        }
        guard emit(node), maxValues <= 24, maxPoints <= 12 else { return nil }
        return SDFProgram(insts: out)
    }
}

#if canImport(Metal)
import Metal

/// The device, queue and pipelines, made once. nil without a Metal device.
final class SDFGPU: @unchecked Sendable {
    static let shared: SDFGPU? = SDFGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private let gridPipeline: MTLComputePipelineState, refinePipeline: MTLComputePipelineState

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: SDFGPU.source, options: nil),
              let gf = lib.makeFunction(name: "sdf_grid"), let rf = lib.makeFunction(name: "sdf_refine"),
              let gp = try? d.makeComputePipelineState(function: gf), let rp = try? d.makeComputePipelineState(function: rf)
        else { return nil }
        (device, queue, gridPipeline, refinePipeline) = (d, q, gp, rp)
    }

    private struct GridParams { var origin: SIMD4<Float>; var nx: UInt32, ny: UInt32, nz: UInt32, count: UInt32 }
    private struct RefineParams { var cell: Float, h: Float; var iterations: UInt32, count: UInt32, ninst: UInt32 }

    private func buffer<T>(_ a: [T]) -> MTLBuffer? {
        a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: Swift.max($0.count, 16), options: .storageModeShared) }
    }

    func grid(_ p: SDFProgram, lo: Vec3, cell: Double, cells: SIMD3<Int>) -> [Float]? {
        let nx = cells.x + 1, ny = cells.y + 1, nz = cells.z + 1
        guard let prog = buffer(p.insts), let out = device.makeBuffer(length: nx * ny * nz * 4, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var params = GridParams(origin: SIMD4(Float(lo.x), Float(lo.y), Float(lo.z), Float(cell)),
                                nx: UInt32(nx), ny: UInt32(ny), nz: UInt32(nz), count: UInt32(p.count))
        enc.setComputePipelineState(gridPipeline)
        enc.setBuffer(out, offset: 0, index: 0); enc.setBuffer(prog, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<GridParams>.stride, index: 2)
        enc.dispatchThreads(MTLSize(width: nx, height: ny, depth: nz), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: nx * ny * nz))
    }

    /// Positions (xyz, cell-clamped projection) and normals for each vertex, as the CPU `cpuRefine` makes them.
    func refine(_ p: SDFProgram, _ verts: [SDFNetVertex], cell: Double, iterations: Int) -> [(Vec3, Vec3)]? {
        guard !verts.isEmpty else { return [] }
        let pos = verts.map { SIMD4<Float>(Float($0.position.x), Float($0.position.y), Float($0.position.z), 0) }
        let cmin = verts.map { SIMD4<Float>(Float($0.cellMin.x), Float($0.cellMin.y), Float($0.cellMin.z), 0) }
        guard let prog = buffer(p.insts), let pb = buffer(pos), let cb2 = buffer(cmin),
              let nb = device.makeBuffer(length: verts.count * 16, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var params = RefineParams(cell: Float(cell), h: Float(cell * 0.25), iterations: UInt32(iterations),
                                  count: UInt32(verts.count), ninst: UInt32(p.count))
        enc.setComputePipelineState(refinePipeline)
        enc.setBuffer(pb, offset: 0, index: 0); enc.setBuffer(nb, offset: 0, index: 1)
        enc.setBuffer(cb2, offset: 0, index: 2); enc.setBuffer(prog, offset: 0, index: 3)
        enc.setBytes(&params, length: MemoryLayout<RefineParams>.stride, index: 4)
        enc.dispatchThreads(MTLSize(width: verts.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        let P = UnsafeBufferPointer(start: pb.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: verts.count)
        let N = UnsafeBufferPointer(start: nb.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: verts.count)
        return (0..<verts.count).map { (Vec3(Double(P[$0].x), Double(P[$0].y), Double(P[$0].z)),
                                        Vec3(Double(N[$0].x), Double(N[$0].y), Double(N[$0].z))) }
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct SDFInst { uint op; float k; float s; uint mode; float4 a; float4 b; float4 r0; float4 r1; float4 r2; };
    struct GridParams { float4 origin; uint nx, ny, nz, count; };
    struct RefineParams { float cell, h; uint iterations, count, ninst; };

    static float sdf_smin(float a, float b, float k) {
        if (k <= 0) return min(a, b);
        float h = max(k - fabs(a - b), 0.0f) / k;
        return min(a, b) - h * h * k * 0.25f;
    }

    static float sdf_eval(constant SDFInst* prog, uint n, float3 p) {
        float vs[24]; int vt = 0;
        float3 ps[12]; int pt = 0;
        for (uint i = 0; i < n; i++) {
            constant SDFInst& q = prog[i];
            switch (q.op) {
            case 0: vs[vt++] = length(p) - q.a.w; break;
            case 1: { float3 d = fabs(p) - q.a.xyz;
                vs[vt++] = length(max(d, 0.0f)) + min(max(d.x, max(d.y, d.z)), 0.0f); break; }
            case 2: { float3 d = fabs(p) - (q.a.xyz - q.b.x);
                vs[vt++] = length(max(d, 0.0f)) + min(max(d.x, max(d.y, d.z)), 0.0f) - q.b.x; break; }
            case 3: { float3 pa = p - q.a.xyz, ba = q.b.xyz - q.a.xyz;
                float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-30f), 0.0f, 1.0f);
                vs[vt++] = length(pa - ba * t) - q.a.w; break; }
            case 4: { float2 d = float2(length(float2(p.x, p.z)) - q.a.x, p.y); vs[vt++] = length(d) - q.a.y; break; }
            case 5: {
                float h = q.a.x, r1 = q.a.y, r2 = q.a.z;
                float2 qq = float2(length(float2(p.x, p.z)), p.y);
                float2 k1 = float2(r2, h), k2 = float2(r2 - r1, 2.0f * h);
                float2 ca = float2(qq.x - min(qq.x, qq.y < 0 ? r1 : r2), fabs(qq.y) - h);
                float t = clamp(dot(k1 - qq, k2) / dot(k2, k2), 0.0f, 1.0f);
                float2 cb = qq - k1 + k2 * t;
                float s = (cb.x < 0 && ca.y < 0) ? -1.0f : 1.0f;
                vs[vt++] = s * sqrt(min(dot(ca, ca), dot(cb, cb))); break; }
            case 6: vs[vt++] = dot(p, q.a.xyz) - q.a.w; break;
            case 7: ps[pt++] = p;
                p = float3(dot(q.r0.xyz, p - q.a.xyz), dot(q.r1.xyz, p - q.a.xyz), dot(q.r2.xyz, p - q.a.xyz)) / q.s; break;
            case 8: p = ps[--pt]; vs[vt - 1] *= q.s; break;
            case 9: { float b = vs[--vt]; vs[vt - 1] = sdf_smin(vs[vt - 1], b, q.k); break; }
            case 10: { float b = vs[--vt]; vs[vt - 1] = -sdf_smin(-vs[vt - 1], b, q.k); break; }
            case 11: { float b = vs[--vt]; vs[vt - 1] = -sdf_smin(-vs[vt - 1], -b, q.k); break; }
            case 12: vs[vt - 1] = -vs[vt - 1]; break;
            case 13: vs[vt - 1] -= q.k; break;
            default: { float d = vs[vt - 1];
                vs[vt - 1] = q.mode == 0 ? max(d, -(d + q.k)) : (q.mode == 1 ? fabs(d) - q.k * 0.5f : max(d - q.k, -d)); }
            }
        }
        return vs[0];
    }

    kernel void sdf_grid(device float* out [[buffer(0)]], constant SDFInst* prog [[buffer(1)]],
                         constant GridParams& g [[buffer(2)]], uint3 id [[thread_position_in_grid]]) {
        if (id.x >= g.nx || id.y >= g.ny || id.z >= g.nz) return;
        float3 p = g.origin.xyz + float3(id) * g.origin.w;
        out[(id.z * g.ny + id.y) * g.nx + id.x] = sdf_eval(prog, g.count, p);
    }

    static float3 sdf_grad(constant SDFInst* prog, uint n, float3 p, float h) {
        return float3(sdf_eval(prog, n, p + float3(h, 0, 0)) - sdf_eval(prog, n, p - float3(h, 0, 0)),
                      sdf_eval(prog, n, p + float3(0, h, 0)) - sdf_eval(prog, n, p - float3(0, h, 0)),
                      sdf_eval(prog, n, p + float3(0, 0, h)) - sdf_eval(prog, n, p - float3(0, 0, h))) / (2 * h);
    }

    kernel void sdf_refine(device float4* pos [[buffer(0)]], device float4* nrm [[buffer(1)]],
                           device const float4* cmin [[buffer(2)]], constant SDFInst* prog [[buffer(3)]],
                           constant RefineParams& r [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= r.count) return;
        float3 p = pos[i].xyz, lo = cmin[i].xyz, hi = lo + r.cell;
        for (uint it = 0; it < r.iterations; it++) {
            float d = sdf_eval(prog, r.ninst, p);
            float3 g = sdf_grad(prog, r.ninst, p, r.h);
            float g2 = dot(g, g);
            if (g2 > 1e-12f) p = clamp(p - g * (d / g2), lo, hi);
        }
        float3 g = sdf_grad(prog, r.ninst, p, r.h);
        float l = length(g);
        pos[i] = float4(p, 0);
        nrm[i] = float4(l > 1e-6f ? g / l : float3(0, 1, 0), 0);
    }
    """
}
#else
final class SDFGPU: @unchecked Sendable {
    static let shared: SDFGPU? = nil
    func grid(_ p: SDFProgram, lo: Vec3, cell: Double, cells: SIMD3<Int>) -> [Float]? { nil }
    func refine(_ p: SDFProgram, _ verts: [SDFNetVertex], cell: Double, iterations: Int) -> [(Vec3, Vec3)]? { nil }
}
#endif
