//  NodeKind.swift — every kind of graph node, with its socket signature (`NodeSpec`).
//
//  The signature is the whole type system: `ModelGraph.validate` checks each connection against it. The kinds
//  that make an `sdf` only ever take `sdf` inputs (plus scalars), so a whole SDF subtree stays an SDF tree until a
//  `sdfToMesh` node meshes it ONCE — on the GPU where a device exists (SDF+GPU.swift).

import Foundation

public enum NodeKind: String, Codable, Sendable, CaseIterable {
    // constants / maths
    case scalar, vec3, math, makeTransform
    // mesh primitives
    case box, cylinder, sphere, cone, torus
    // SDF primitives
    case sdfBox, sdfSphere, sdfCylinder, sdfCone, sdfTorus
    // mesh operators
    case meshTransform, mirror, array, radialArray, boolean, modifier, bevel, subdivide, decimate, join, setUVs, importMesh
    // SDF operators
    case sdfTransform, sdfCombine, sdfOffset, sdfShell, sdfCarve, meshToSDF
    // bridges and scattering
    case sdfToMesh, distribute, instance

    public struct Input: Sendable { public var name: String; public var type: SocketType; public var required: Bool }
    public struct Spec: Sendable {
        public var inputs: [Input]
        public var output: SocketType
        /// Inputs named `in0`, `in1`, … of this type are accepted too (join).
        public var variadic: SocketType?
    }

    public var spec: Spec {
        func req(_ n: String, _ t: SocketType) -> Input { Input(name: n, type: t, required: true) }
        func opt(_ n: String, _ t: SocketType) -> Input { Input(name: n, type: t, required: false) }
        func s(_ i: [Input], _ o: SocketType, variadic: SocketType? = nil) -> Spec { Spec(inputs: i, output: o, variadic: variadic) }
        let paramSockets: [Input] = []
        switch self {
        case .scalar: return s(paramSockets, .scalar)
        case .vec3: return s(paramSockets, .vec3)
        case .math: return s([opt("a", .scalar), opt("b", .scalar)], .scalar)
        case .makeTransform: return s([opt("translate", .vec3), opt("rotate", .vec3), opt("scale", .vec3)], .transform)
        case .box: return s([opt("size", .vec3)], .mesh)
        case .cylinder, .sphere, .cone: return s([opt("radius", .scalar), opt("height", .scalar)], .mesh)
        case .torus: return s([opt("major", .scalar), opt("minor", .scalar)], .mesh)
        case .sdfTorus: return s([opt("major", .scalar), opt("minor", .scalar)], .sdf)
        case .sdfBox: return s([opt("size", .vec3)], .sdf)
        case .sdfSphere, .sdfCylinder, .sdfCone: return s([opt("radius", .scalar), opt("height", .scalar)], .sdf)
        case .meshTransform: return s([req("mesh", .mesh), opt("transform", .transform), opt("translate", .vec3),
                                       opt("rotate", .vec3), opt("scale", .vec3)], .mesh)
        case .mirror, .array, .radialArray, .modifier, .setUVs: return s([req("mesh", .mesh)], .mesh)
        case .bevel: return s([req("mesh", .mesh), opt("radius", .scalar)], .mesh)
        case .subdivide: return s([req("mesh", .mesh), opt("levels", .scalar)], .mesh)
        case .decimate: return s([req("mesh", .mesh), opt("ratio", .scalar)], .mesh)
        case .boolean: return s([req("a", .mesh), req("b", .mesh)], .mesh)
        case .join: return s([], .mesh, variadic: .mesh)
        case .importMesh: return s([], .mesh)
        case .sdfTransform: return s([req("sdf", .sdf), opt("transform", .transform), opt("translate", .vec3),
                                      opt("rotate", .vec3), opt("scale", .vec3)], .sdf)
        case .sdfCombine: return s([req("a", .sdf), req("b", .sdf), opt("smooth", .scalar)], .sdf)
        case .sdfOffset: return s([req("sdf", .sdf), opt("amount", .scalar)], .sdf)
        case .sdfShell: return s([req("sdf", .sdf), opt("thickness", .scalar)], .sdf)
        case .sdfCarve: return s([req("sdf", .sdf), req("tool", .sdf), opt("depth", .scalar), opt("falloff", .scalar)], .sdf)
        case .meshToSDF: return s([req("mesh", .mesh)], .sdf)
        case .sdfToMesh: return s([req("sdf", .sdf), opt("cell", .scalar)], .mesh)
        case .distribute: return s([req("mesh", .mesh), opt("count", .scalar)], .points)
        case .instance: return s([req("mesh", .mesh), req("points", .points)], .mesh)
        }
    }

    /// The socket type an input of this kind has, or nil if it has no such input.
    func inputType(_ name: String) -> SocketType? {
        if let i = spec.inputs.first(where: { $0.name == name }) { return i.type }
        if let v = spec.variadic, name.hasPrefix("in"), Int(name.dropFirst(2)) != nil { return v }
        return nil
    }
}
