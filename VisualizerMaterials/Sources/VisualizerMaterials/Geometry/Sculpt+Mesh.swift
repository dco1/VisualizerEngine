//  Sculpt+Mesh.swift — meshing a `SculptVolume` (ticket item I), through item C's surface-nets topology.
//
//  Topology (one vertex per sign-changing cell, one quad per sign-changing edge) is `SDF.netTopology`, CPU as in item C;
//  the vertex pass (projection onto the trilinear field + gradient normal) runs on the GPU straight from the resident
//  field buffer (`sculpt_refine`), with a CPU twin (`SculptKernel.refine`) when `useGPU` is off or there is no device.
//  Guarantees: closed, edge-manifold, outward wound, shared vertices (the topology's), so a result is watertight.

import Foundation
import simd

public extension SculptVolume {
    func mesh(uvs: SDFMeshOptions.UVs = .none, projectIterations: Int = 1) -> Mesh3 {
        sync()
        let cells = samples &- 1
        let raw = Array(UnsafeBufferPointer(start: store.ptr, count: count))
        let topo = SDF.netTopology(raw, cells: cells, lo: origin, cell: cell)
        var out: [(Vec3, Vec3)]? = nil
        if useGPU { out = SculptGPU.refine(self, topo.vertices, iterations: projectIterations) }
        let refined = out ?? topo.vertices.map { v -> (Vec3, Vec3) in
            let (p, n) = SculptKernel.refine(position: SIMD3<Float>(v.position), cellMin: SIMD3<Float>(v.cellMin), field: store.ptr,
                                             n: samples, origin: SIMD3<Float>(origin), cell: Float(cell), iterations: projectIterations)
            return (Vec3(Double(p.x), Double(p.y), Double(p.z)), Vec3(Double(n.x), Double(n.y), Double(n.z)))
        }
        var m = Mesh3()
        m.positions = refined.map { $0.0 }
        m.normals = refined.map { $0.1 }
        m.indices = topo.indices
        switch uvs {
        case .none: m.uvs = [Vec2](repeating: .zero, count: m.positions.count)
        case .boxProjected(let s):
            m.uvs = zip(m.positions, m.normals).map { p, n in
                let a = simd_abs(n)
                if a.x >= a.y && a.x >= a.z { return Vec2(p.z, p.y) / s }
                return a.y >= a.z ? Vec2(p.x, p.z) / s : Vec2(p.x, p.y) / s
            }
        }
        return m
    }
}
