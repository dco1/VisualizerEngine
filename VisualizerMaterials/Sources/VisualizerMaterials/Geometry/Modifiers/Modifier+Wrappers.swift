//  Thin Modifier wrappers over the existing Mesh3 operations: Mirror, Array, Bevel, Subdivide, Boolean.
//  Each only forwards to `mirrored` / `arrayed` / `radialArray` / `beveled` / `subdivided` / the boolean, so it
//  inherits their guarantees and limits (see those files).
//
//  Boolean takes its engine as a closure (`BooleanModifier.Engine`). The default is the BSP CSG
//  (`union` / `subtracting` / `intersection`), which is not watertight-guaranteed; to use the robust boolean, pass
//  `engine: { a, b, op in RobustBoolean.run(a, b, op) }` (or whatever it is called) — nothing else changes.

import simd

public struct MirrorModifier: Modifier {
    public var axis: Mesh3.Axis
    public var plane: Double
    /// Keep the original beside its reflection (Blender's default); false returns only the reflection.
    public var keepOriginal: Bool
    public init(_ axis: Mesh3.Axis, at plane: Double = 0, keepOriginal: Bool = true) {
        self.axis = axis; self.plane = plane; self.keepOriginal = keepOriginal
    }
    public var summary: String { "Mirror(\(axis) at:\(plane) keep:\(keepOriginal))" }
    public func apply(_ mesh: Mesh3) -> Mesh3 {
        var out = keepOriginal ? mesh : Mesh3()
        out.append(mesh.mirrored(axis, at: plane))
        return out
    }
}

public struct ArrayModifier: Modifier {
    public enum Layout: Sendable { case linear(offset: Vec3), radial(centre: Vec3, startAngle: Double) }
    public var count: Int
    public var layout: Layout
    public init(count: Int, offset: Vec3) { self.count = count; self.layout = .linear(offset: offset) }
    public init(radialCount count: Int, around centre: Vec3 = .zero, startAngle: Double = 0) {
        self.count = count; self.layout = .radial(centre: centre, startAngle: startAngle)
    }
    public var summary: String { "Array(\(count) \(layout))" }
    public func apply(_ mesh: Mesh3) -> Mesh3 {
        switch layout {
        case .linear(let o): return mesh.arrayed(count: count, offset: o)
        case .radial(let c, let a): return mesh.radialArray(count: count, around: c, startAngle: a)
        }
    }
}

public struct BevelModifier: Modifier {
    public var radius: Double, creaseDegrees: Double, segments: Int
    public init(radius: Double, creaseDegrees: Double = 30, segments: Int = 3) {
        self.radius = radius; self.creaseDegrees = creaseDegrees; self.segments = segments
    }
    public var summary: String { "Bevel(r:\(radius) crease:\(creaseDegrees) seg:\(segments))" }
    public func apply(_ mesh: Mesh3) -> Mesh3 { mesh.beveled(radius: radius, creaseDegrees: creaseDegrees, segments: segments) }
}

public struct SubdivideModifier: Modifier {
    public var levels: Int, creaseDegrees: Double
    public init(levels: Int = 1, creaseDegrees: Double = 35) { self.levels = levels; self.creaseDegrees = creaseDegrees }
    public var summary: String { "Subdivide(levels:\(levels) crease:\(creaseDegrees))" }
    public func apply(_ mesh: Mesh3) -> Mesh3 { mesh.subdivided(levels: levels, creaseDegrees: creaseDegrees) }
}

public struct BooleanModifier: Modifier {
    public enum Op: Sendable { case union, difference, intersection }
    public typealias Engine = @Sendable (_ base: Mesh3, _ operand: Mesh3, _ op: Op) -> Mesh3

    /// The existing BSP CSG.
    public static let bsp: Engine = { a, b, op in
        switch op {
        case .union: return a.union(b)
        case .difference: return a.subtracting(b)
        case .intersection: return a.intersection(b)
        }
    }

    public var operand: Mesh3
    public var op: Op
    public var engine: Engine
    public init(_ op: Op, with operand: Mesh3, engine: @escaping Engine = BooleanModifier.bsp) {
        self.op = op; self.operand = operand; self.engine = engine
    }
    public var summary: String { "Boolean(\(op) operand:\(operand.triangleCount)t)" }
    public func apply(_ mesh: Mesh3) -> Mesh3 { engine(mesh, operand, op) }
}
