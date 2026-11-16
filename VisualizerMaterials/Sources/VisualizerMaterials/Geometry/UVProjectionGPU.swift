import Foundation
import simd

//  UVProjectionGPU.swift — the four UV projections as one Metal kernel (ticket item M3).
//
//  Guarantees: `cornerUVs(of:backend:)` returns what `UVProjection.cornerUVs(of:)` returns (3 per triangle, same
//  order, same seam/pole rules: a straddling triangle's longitudes lifted past 1, a pole taking its neighbours' mean)
//  to Float precision. One thread per triangle; the frame, scale and rotation are folded on the CPU into 40 bytes
//  of parameters. Doesn't: split vertices — `Mesh3.withCornerUVs` (CPU, a hash over corners) still does that.
//  Pole test is 1e-6 relative in Float where the CPU uses 1e-9 in Double.

public extension UVProjection {
    func cornerUVs(of mesh: Mesh3, backend: UVBackend) -> [Vec2] {
        #if canImport(Metal)
        if backend.useGPU(work: mesh.triangleCount), let g = UVGPU.shared, let r = gpuCornerUVs(mesh, g) { return r }
        #endif
        return cornerUVs(of: mesh)
    }
}

public extension Mesh3 {
    /// The mesh with `projection`'s UVs, computed on `backend`.
    func projectedUVs(_ projection: UVProjection, backend: UVBackend) -> Mesh3 {
        withCornerUVs(projection.cornerUVs(of: self, backend: backend))
    }
}

#if canImport(Metal)
import Metal

extension UVProjection {
    struct GPUParams {
        var origin: SIMD4<Float>, fx: SIMD4<Float>, fy: SIMD4<Float>, fn: SIMD4<Float>
        var scale: SIMD2<Float>, offset: SIMD2<Float>
        var cosr: Float, sinr: Float, kind: UInt32, tris: UInt32
    }

    func gpuCornerUVs(_ mesh: Mesh3, _ g: UVGPU) -> [Vec2]? {
        let f = frame(), a = rotationDegrees * .pi / 180
        func v4(_ v: Vec3) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), 0) }
        var params = GPUParams(origin: v4(origin), fx: v4(f.x), fy: v4(f.y), fn: v4(f.n),
                               scale: SIMD2(Float(scale.x), Float(scale.y)), offset: SIMD2(Float(offset.x), Float(offset.y)),
                               cosr: Float(cos(a)), sinr: Float(sin(a)),
                               kind: { switch kind { case .planar: 0; case .box: 1; case .cylinder: 2; case .sphere: 3 } }(),
                               tris: UInt32(mesh.triangleCount))
        let nTri = mesh.triangleCount
        let pos = g.buffer(UVGPU.float4s(mesh.positions)), idx = g.buffer(mesh.indices), out = g.empty(nTri * 3 * 8)
        guard g.run({ enc in
            enc.setBuffer(pos, offset: 0, index: 0); enc.setBuffer(idx, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
            enc.setBytes(&params, length: MemoryLayout<GPUParams>.stride, index: 3)
            g.dispatch(enc, "uv_project", nTri)
        }) else { return nil }
        return UVGPU.read(out, SIMD2<Float>.self, count: nTri * 3).map { Vec2(Double($0.x), Double($0.y)) }
    }

    static let kernelSource = """
    struct ProjParams { float4 origin, fx, fy, fn; float2 scale, offset; float cosr, sinr; uint kind, tris; };

    kernel void uv_project(device const float4* pos [[buffer(0)]], device const uint* idx [[buffer(1)]],
                           device float2* out [[buffer(2)]], constant ProjParams& P [[buffer(3)]], uint t [[thread_position_in_grid]]) {
        if (t >= P.tris) return;
        float3 l[3]; float2 raw[3];
        for (int c = 0; c < 3; c++) {
            float3 q = pos[idx[3 * t + c]].xyz - P.origin.xyz;
            l[c] = float3(dot(q, P.fx.xyz), dot(q, P.fy.xyz), dot(q, P.fn.xyz));
        }
        if (P.kind == 0) {
            for (int c = 0; c < 3; c++) raw[c] = l[c].xy;
        } else if (P.kind == 1) {
            float3 nn = cross(l[1] - l[0], l[2] - l[0]);
            int a = (fabs(nn.x) >= fabs(nn.y) && fabs(nn.x) >= fabs(nn.z)) ? 0 : (fabs(nn.y) >= fabs(nn.z) ? 1 : 2);
            float nv = a == 0 ? nn.x : (a == 1 ? nn.y : nn.z);
            float flip = nv < 0 ? -1.0f : 1.0f;
            for (int c = 0; c < 3; c++) {
                float3 q = l[c];
                raw[c] = a == 0 ? float2(flip * q.y, q.z) : (a == 1 ? float2(flip * q.z, q.x) : float2(flip * q.x, q.y));
            }
        } else {
            float lon[3]; bool pole[3];
            for (int c = 0; c < 3; c++) {
                lon[c] = atan2(l[c].y, l[c].x) / (2 * M_PI_F) + 0.5f;
                pole[c] = P.kind == 3 && length(l[c].xy) < 1e-6f * max(1.0f, length(l[c]));
            }
            float lo = INFINITY, hi = -INFINITY; int others = 0;
            for (int c = 0; c < 3; c++) if (!pole[c]) { lo = min(lo, lon[c]); hi = max(hi, lon[c]); others++; }
            if (others > 0 && hi - lo > 0.5f) { for (int c = 0; c < 3; c++) if (!pole[c] && lon[c] < 0.5f) lon[c] += 1; }
            float mean = 0.5f;
            if (others > 0) { float s = 0; for (int c = 0; c < 3; c++) if (!pole[c]) s += lon[c]; mean = s / float(others); }
            for (int c = 0; c < 3; c++) {
                if (pole[c]) lon[c] = mean;
                raw[c] = P.kind == 2 ? float2(lon[c], l[c].z)
                                     : float2(lon[c], atan2(l[c].z, length(l[c].xy)) / M_PI_F + 0.5f);
            }
        }
        for (int c = 0; c < 3; c++) {
            float2 s = raw[c] * P.scale;
            out[3 * t + c] = float2(P.cosr * s.x - P.sinr * s.y, P.sinr * s.x + P.cosr * s.y) + P.offset;
        }
    }
    """
}
#endif

enum UVProjectionGPU {
    static var source: String {
        #if canImport(Metal)
        UVProjection.kernelSource
        #else
        ""
        #endif
    }
}
