//  GraphBuilder.swift — a Swift DSL over `ModelGraph`: typed references, fluent operators, no ids by hand.
//
//      let g = GraphBuilder()
//      let part = g.box(size: Vec3(2, 2, 2)).subtracting(g.cylinder(radius: 0.5, height: 4))
//      let graph = g.build(output: part)
//
//  Each node-making call returns a `Ref` typed by what the node outputs (`MeshRef`, `SDFRef`, …), so connecting a
//  mesh where an SDF belongs does not compile. `bind(_:to:)` wires a scalar / vec3 node into a same-named parameter.
//
//  Guarantees: a built graph is an ordinary `ModelGraph` (Codable, validated at evaluate). Not guaranteed:
//  the builder is a reference type for fluent chaining and is not thread-safe; build on one thread.

import Foundation

public enum GraphKind { public enum Mesh {}; public enum SDF {}; public enum Points {}; public enum Scalar {}; public enum Vec3 {}; public enum Transform {} }

public struct Ref<K> {
    public let id: Int
    let builder: GraphBuilder
    /// Feed `value` (a scalar or vec3 node) into this node's input / parameter called `name`.
    @discardableResult public func bind<V>(_ name: String, to value: Ref<V>) -> Ref<K> { builder.graph.connect(value.id, to: id, input: name); return self }
    @discardableResult public func set(_ key: String, _ v: Double) -> Ref<K> { builder.graph.setParam(id, key, v); return self }
    @discardableResult public func set(_ key: String, _ v: Param) -> Ref<K> { builder.graph.setParam(id, key, v); return self }
}
public typealias MeshRef = Ref<GraphKind.Mesh>
public typealias SDFRef = Ref<GraphKind.SDF>
public typealias PointsRef = Ref<GraphKind.Points>
public typealias ScalarRef = Ref<GraphKind.Scalar>
public typealias Vec3Ref = Ref<GraphKind.Vec3>
public typealias TransformRef = Ref<GraphKind.Transform>

public final class GraphBuilder {
    public fileprivate(set) var graph = ModelGraph()
    public init() {}

    public func build(output: (any GraphRef)? = nil) -> ModelGraph {
        var g = graph
        if let o = output { g.output = o.nodeID }
        return g
    }

    fileprivate func node<K>(_ kind: NodeKind, _ p: [String: Param] = [:], _ ins: [String: Int] = [:]) -> Ref<K> {
        Ref(id: graph.add(kind, params: p, inputs: ins), builder: self)
    }
    private func n(_ v: Double) -> Param { .number(v) }

    // constants
    public func scalar(_ v: Double) -> ScalarRef { node(.scalar, ["value": .number(v)]) }
    public func vec3(_ v: Vec3) -> Vec3Ref { node(.vec3, ["value": .vec3(v)]) }
    public func math(_ op: String, _ a: ScalarRef, _ b: ScalarRef) -> ScalarRef { node(.math, ["op": .string(op)], ["a": a.id, "b": b.id]) }
    public func transform(translate: Vec3 = .zero, rotate: Vec3 = .zero, scale: Vec3 = Vec3(1, 1, 1)) -> TransformRef {
        node(.makeTransform, ["translate": .vec3(translate), "rotate": .vec3(rotate), "scale": .vec3(scale)])
    }

    // mesh primitives
    public func box(size: Vec3 = Vec3(1, 1, 1)) -> MeshRef { node(.box, ["size": .vec3(size)]) }
    public func cylinder(radius: Double = 0.5, height: Double = 1, segments: Int = 32) -> MeshRef {
        node(.cylinder, ["radius": n(radius), "height": n(height), "segments": .int(segments)])
    }
    public func sphere(radius: Double = 0.5, segments: Int = 32) -> MeshRef { node(.sphere, ["radius": n(radius), "segments": .int(segments)]) }
    public func cone(radius: Double = 0.5, topRadius: Double = 0, height: Double = 1, segments: Int = 32) -> MeshRef {
        node(.cone, ["radius": n(radius), "topRadius": n(topRadius), "height": n(height), "segments": .int(segments)])
    }
    public func torus(major: Double = 0.5, minor: Double = 0.15, segments: Int = 32, tubeSegments: Int = 16) -> MeshRef {
        node(.torus, ["major": n(major), "minor": n(minor), "segments": .int(segments), "tubeSegments": .int(tubeSegments)])
    }
    public func importMesh(_ m: Mesh3) -> MeshRef { node(.importMesh, ["data": .data(MeshBlob.encode(m))]) }
    public func join(_ parts: [MeshRef]) -> MeshRef {
        var ins: [String: Int] = [:]
        for (i, p) in parts.enumerated() { ins["in\(i)"] = p.id }
        return node(.join, [:], ins)
    }

    // SDF primitives
    public func sdfBox(size: Vec3 = Vec3(1, 1, 1)) -> SDFRef { node(.sdfBox, ["size": .vec3(size)]) }
    public func sdfSphere(radius: Double = 0.5) -> SDFRef { node(.sdfSphere, ["radius": n(radius)]) }
    public func sdfCylinder(radius: Double = 0.5, height: Double = 1) -> SDFRef { node(.sdfCylinder, ["radius": n(radius), "height": n(height)]) }
    public func sdfCone(radius: Double = 0.5, topRadius: Double = 0, height: Double = 1) -> SDFRef {
        node(.sdfCone, ["radius": n(radius), "topRadius": n(topRadius), "height": n(height)])
    }
    public func sdfTorus(major: Double = 0.5, minor: Double = 0.15) -> SDFRef { node(.sdfTorus, ["major": n(major), "minor": n(minor)]) }

    // scattering
    public func distribute(on m: MeshRef, count: Int, seed: Int = 0) -> PointsRef { node(.distribute, ["count": .int(count), "seed": .int(seed)], ["mesh": m.id]) }
    public func instance(_ m: MeshRef, on pts: PointsRef, scale: ClosedRange<Double> = 1 ... 1, rotateJitter: Vec3 = .zero,
                         alignToNormal: Bool = false, seed: Int = 0) -> MeshRef {
        node(.instance, ["scaleMin": n(scale.lowerBound), "scaleMax": n(scale.upperBound), "rotateJitter": .vec3(rotateJitter),
                         "alignToNormal": .bool(alignToNormal), "seed": .int(seed)], ["mesh": m.id, "points": pts.id])
    }
}

/// Anything with a node id (every `Ref`).
public protocol GraphRef { var nodeID: Int { get } }
extension Ref: GraphRef { public var nodeID: Int { id } }

public extension Ref where K == GraphKind.Mesh {
    private func make(_ kind: NodeKind, _ p: [String: Param] = [:], _ ins: [String: Int] = [:]) -> MeshRef { builder.node(kind, p, ins) }
    private func opName(_ o: BooleanOp) -> String { o == .union ? "union" : o == .difference ? "difference" : "intersection" }

    func transformed(translate: Vec3 = .zero, rotate: Vec3 = .zero, scale: Vec3 = Vec3(1, 1, 1)) -> MeshRef {
        make(.meshTransform, ["translate": .vec3(translate), "rotate": .vec3(rotate), "scale": .vec3(scale)], ["mesh": id])
    }
    func transformed(by t: TransformRef) -> MeshRef { make(.meshTransform, [:], ["mesh": id, "transform": t.id]) }
    func mirrored(_ axis: String = "x", at plane: Double = 0, keepOriginal: Bool = true) -> MeshRef {
        make(.mirror, ["axis": .string(axis), "plane": .number(plane), "keepOriginal": .bool(keepOriginal)], ["mesh": id])
    }
    func arrayed(count: Int, offset: Vec3) -> MeshRef { make(.array, ["count": .int(count), "offset": .vec3(offset)], ["mesh": id]) }
    func radialArrayed(count: Int, around centre: Vec3 = .zero, startAngle: Double = 0) -> MeshRef {
        make(.radialArray, ["count": .int(count), "centre": .vec3(centre), "startAngle": .number(startAngle)], ["mesh": id])
    }
    func boolean(_ op: BooleanOp, _ other: MeshRef, engine: String = "robust") -> MeshRef {
        make(.boolean, ["op": .string(opName(op)), "engine": .string(engine)], ["a": id, "b": other.id])
    }
    func union(_ o: MeshRef, engine: String = "robust") -> MeshRef { boolean(.union, o, engine: engine) }
    func subtracting(_ o: MeshRef, engine: String = "robust") -> MeshRef { boolean(.difference, o, engine: engine) }
    func intersecting(_ o: MeshRef, engine: String = "robust") -> MeshRef { boolean(.intersection, o, engine: engine) }
    /// Any modifier from the stack by name (`GraphModifierRegistry.names`).
    func modified(_ name: String, _ args: [String: Param] = [:]) -> MeshRef { make(.modifier, ["name": .string(name), "args": .table(args)], ["mesh": id]) }
    func beveled(radius: Double, crease: Double = 30, segments: Int = 3) -> MeshRef {
        make(.bevel, ["radius": .number(radius), "crease": .number(crease), "segments": .int(segments)], ["mesh": id])
    }
    func subdivided(levels: Int = 1, crease: Double = 35) -> MeshRef { make(.subdivide, ["levels": .int(levels), "crease": .number(crease)], ["mesh": id]) }
    func decimated(ratio: Double) -> MeshRef { make(.decimate, ["ratio": .number(ratio)], ["mesh": id]) }
    func decimated(targetTriangles: Int) -> MeshRef { make(.decimate, ["targetTriangles": .int(targetTriangles)], ["mesh": id]) }
    func uvs(_ projection: String = "box", scale: Double = 1, origin: Vec3 = .zero, axis: Vec3 = Vec3(0, 0, 1)) -> MeshRef {
        make(.setUVs, ["projection": .string(projection), "scale": .number(scale), "origin": .vec3(origin), "axis": .vec3(axis)], ["mesh": id])
    }
    func toSDF() -> SDFRef { builder.node(.meshToSDF, [:], ["mesh": id]) }
}

public extension Ref where K == GraphKind.SDF {
    private func make(_ kind: NodeKind, _ p: [String: Param] = [:], _ ins: [String: Int] = [:]) -> SDFRef { builder.node(kind, p, ins) }
    private func combine(_ op: String, _ o: SDFRef, _ k: Double) -> SDFRef {
        make(.sdfCombine, ["op": .string(op), "smooth": .number(k)], ["a": id, "b": o.id])
    }
    func transformed(translate: Vec3 = .zero, rotate: Vec3 = .zero, scale: Double = 1) -> SDFRef {
        make(.sdfTransform, ["translate": .vec3(translate), "rotate": .vec3(rotate), "scale": .vec3(Vec3(repeating: scale))], ["sdf": id])
    }
    func union(_ o: SDFRef, smooth k: Double = 0) -> SDFRef { combine("union", o, k) }
    func subtracting(_ o: SDFRef, smooth k: Double = 0) -> SDFRef { combine("subtract", o, k) }
    func intersecting(_ o: SDFRef, smooth k: Double = 0) -> SDFRef { combine("intersect", o, k) }
    func offset(by r: Double) -> SDFRef { make(.sdfOffset, ["amount": .number(r)], ["sdf": id]) }
    func shell(thickness: Double, side: String = "inside") -> SDFRef { make(.sdfShell, ["thickness": .number(thickness), "side": .string(side)], ["sdf": id]) }
    func carved(tool: SDFRef, depth: Double, falloff: Double = 0) -> SDFRef {
        make(.sdfCarve, ["depth": .number(depth), "falloff": .number(falloff)], ["sdf": id, "tool": tool.id])
    }
    /// Mesh the whole subtree once (GPU surface nets where available).
    func meshed(cell: Double? = nil, backend: String = "auto", uvMetersPerUnit: Double = 0) -> MeshRef {
        var p: [String: Param] = ["backend": .string(backend), "uvMetersPerUnit": .number(uvMetersPerUnit)]
        if let c = cell { p["cell"] = .number(c) }
        return builder.node(.sdfToMesh, p, ["sdf": id])
    }
}
