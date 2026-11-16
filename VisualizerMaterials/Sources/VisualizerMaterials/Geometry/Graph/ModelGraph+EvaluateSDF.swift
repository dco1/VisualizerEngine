//  ModelGraph+EvaluateSDF.swift — the SDF half of evaluation, and the one bridge back to meshes.
//
//  Every SDF node only builds a bigger `SDF` tree (and a conservative bounding box) — nothing is sampled until a
//  `sdfToMesh` node meshes the WHOLE subtree once with the surface-nets mesher, on the GPU when a Metal device
//  exists and the tree has a GPU form (`SDFMeshOptions.Backend.auto`, SDF+GPU.swift), CPU otherwise. So a long
//  chain of unions, shells and carves is one dispatch, not one mesh per step. `meshToSDF` wraps a closed mesh as a
//  field (SDFMeshField): that subtree is CPU-only and slow per sample; use it sparingly.
//
//  Guarantees: bounds contain the surface (offsets, shells, smooth blends grow them by what they can move it).
//  Not guaranteed: bounds are boxes of boxes (loose under rotation); `sdfToMesh` rounds sharp edges by about a cell.

import Foundation
import simd

extension GraphEvaluator {
    static func computeSDF(_ n: GraphNode, _ a: GraphArgs, _ ins: [String: GraphValue]) throws -> GraphValue {
        func make(_ s: SDF, _ half: Vec3) -> GraphValue { .sdf(SDFValue(sdf: s, lo: -half, hi: half)) }
        switch n.kind {
        case .sdfBox:
            let s = a.vec3("size", Vec3(1, 1, 1)); return make(.box(size: s), s / 2)
        case .sdfSphere:
            let r = a.double("radius", 0.5); return make(.sphere(radius: r), Vec3(repeating: r))
        case .sdfCylinder:
            let r = a.double("radius", 0.5), h = a.double("height", 1)
            return make(.cone(height: h, bottomRadius: r, topRadius: r), Vec3(r, h / 2, r))
        case .sdfCone:
            let r = a.double("radius", 0.5), rt = a.double("topRadius", 0), h = a.double("height", 1), m = Swift.max(r, rt)
            return make(.cone(height: h, bottomRadius: r, topRadius: rt), Vec3(m, h / 2, m))
        case .sdfTorus:
            let big = a.double("major", 0.5), small = a.double("minor", 0.15)
            return make(.torus(major: big, minor: small), Vec3(big + small, small, big + small))
        case .sdfTransform:
            return .sdf(try sdf(n, ins, "sdf").placed(transform(ins, a)))
        case .sdfCombine:
            let x = try sdf(n, ins, "a"), y = try sdf(n, ins, "b"), k = a.double("smooth", 0)
            let pad = Vec3(repeating: Swift.max(0, k) * 0.25)
            switch a.string("op", "union") {
            case "union": return .sdf(SDFValue(sdf: x.sdf.union(y.sdf, smooth: k), lo: simd_min(x.lo, y.lo) - pad, hi: simd_max(x.hi, y.hi) + pad))
            case "subtract": return .sdf(SDFValue(sdf: x.sdf.subtracting(y.sdf, smooth: k), lo: x.lo - pad, hi: x.hi + pad))
            case "intersect":
                var lo = simd_max(x.lo, y.lo), hi = simd_min(x.hi, y.hi)
                if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z { lo = x.lo; hi = x.hi }
                return .sdf(SDFValue(sdf: x.sdf.intersecting(y.sdf, smooth: k), lo: lo - pad, hi: hi + pad))
            case let o: throw GraphError.badParameter(node: n.id, message: "sdfCombine op '\(o)' (union, subtract, intersect)")
            }
        case .sdfOffset:
            let x = try sdf(n, ins, "sdf"), r = a.double("amount", 0), g = Vec3(repeating: Swift.max(0, r))
            return .sdf(SDFValue(sdf: x.sdf.offset(by: r), lo: x.lo - g, hi: x.hi + g))
        case .sdfShell:
            let x = try sdf(n, ins, "sdf"), t = a.double("thickness", 0.02)
            let side: SDF.ShellSide
            switch a.string("side", "inside") { case "centered": side = .centered; case "outside": side = .outside; default: side = .inside }
            let g = Vec3(repeating: side == .outside ? t : side == .centered ? t / 2 : 0)
            return .sdf(SDFValue(sdf: x.sdf.shell(thickness: t, side: side), lo: x.lo - g, hi: x.hi + g))
        case .sdfCarve:
            let x = try sdf(n, ins, "sdf"), tool = try sdf(n, ins, "tool")
            return .sdf(SDFValue(sdf: x.sdf.carved(tool: tool.sdf, depth: a.double("depth", 0.01), falloff: a.double("falloff", 0)), lo: x.lo, hi: x.hi))
        case .meshToSDF:
            let m = try mesh(n, ins, "mesh")
            guard let b = m.bounds else { throw GraphError.badParameter(node: n.id, message: "meshToSDF of an empty mesh") }
            return .sdf(SDFValue(sdf: .mesh(m), lo: b.min, hi: b.max))
        default: return try toMesh(n, a, try sdf(n, ins, "sdf"))
        }
    }

    /// `sdfToMesh`: mesh the tree inside its bounds (or `boundsMin` / `boundsMax`), `cell` metres a cell.
    static func toMesh(_ n: GraphNode, _ a: GraphArgs, _ v: SDFValue) throws -> GraphValue {
        var lo = v.lo, hi = v.hi
        if a.values["boundsMin"] != nil { lo = a.vec3("boundsMin", lo) }
        if a.values["boundsMax"] != nil { hi = a.vec3("boundsMax", hi) }
        let extent = hi - lo
        let cell = a.double("cell", Swift.max(extent.x, Swift.max(extent.y, extent.z)) / 64)
        guard cell > 0 else { throw GraphError.badParameter(node: n.id, message: "sdfToMesh cell must be > 0") }
        let pad = Vec3(repeating: cell * a.double("padding", 2))
        lo -= pad; hi += pad
        let cells = (hi - lo) / cell
        guard cells.x < 600, cells.y < 600, cells.z < 600 else {
            throw GraphError.badParameter(node: n.id, message: "sdfToMesh would need \(Int(cells.x))×\(Int(cells.y))×\(Int(cells.z)) cells; raise 'cell'")
        }
        var o = SDFMeshOptions()
        switch a.string("backend", "auto") { case "cpu": o.backend = .cpu; case "gpu": o.backend = .gpu; default: o.backend = .auto }
        let uv = a.double("uvMetersPerUnit", 0)
        if uv > 0 { o.uvs = .boxProjected(metersPerUnit: uv) }
        return .mesh(v.sdf.mesh(lo: lo, hi: hi, cell: cell, options: o))
    }
}
