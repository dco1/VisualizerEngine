//  ModelGraph+Evaluate.swift — evaluation with a per-node cache keyed by structure.
//
//  A node's KEY is a 64-bit hash of its kind, its parameters and the KEYS of the nodes feeding it (a Merkle hash),
//  so it changes exactly when something upstream of it changed. `GraphCache` maps key → value: change one
//  parameter and only that node and everything downstream of it recompute; the rest are cache hits.
//  `GraphCache.computed` logs every node actually computed (tests count it).
//
//  Guarantees: deterministic (same graph, same result; random nodes are seeded by parameters); cycles throw
//  `GraphError.cycle` before anything runs. Not guaranteed: a 64-bit key can collide in principle (~2⁻⁶⁴ per pair);
//  the cache is not pruned unless asked (`trim(keeping:)` / `evaluate(prune: true)`).

import Foundation
import simd

public final class GraphCache: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [UInt64: GraphValue] = [:]
    private var canonical: [UInt64: GraphValue] = [:]
    private var log: [(id: Int, kind: NodeKind)] = []

    public init() {}
    /// Every node computed (not served from the cache) since the last `resetCounters()`, in order.
    public var computed: [(id: Int, kind: NodeKind)] { lock.withLock { log } }
    public var computeCount: Int { lock.withLock { log.count } }
    public var entryCount: Int { lock.withLock { entries.count } }
    public func resetCounters() { lock.withLock { log.removeAll() } }
    public func removeAll() { lock.withLock { entries.removeAll(); canonical.removeAll() } }
    func value(for key: UInt64) -> GraphValue? { lock.withLock { entries[key] } }
    func store(_ v: GraphValue, key: UInt64, id: Int, kind: NodeKind) { lock.withLock { entries[key] = v; log.append((id, kind)) } }
    /// The canonical (byte-stable) form of a result, made once per key (see GraphCanonical.swift).
    func canonicalValue(for key: UInt64, make: () -> GraphValue) -> GraphValue {
        if let v = lock.withLock({ canonical[key] }) { return v }
        let v = make()
        lock.withLock { canonical[key] = v }
        return v
    }
    public func trim(keeping keys: Set<UInt64>) {
        lock.withLock { entries = entries.filter { keys.contains($0.key) }; canonical = canonical.filter { keys.contains($0.key) } }
    }
}

public extension ModelGraph {
    /// Structural keys of `id` and everything it reads (throws on a cycle or a missing node).
    func keys(for id: Int) throws -> [Int: UInt64] {
        let index = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        var out: [Int: UInt64] = [:]
        var stack: [Int] = []
        func key(_ id: Int) throws -> UInt64 {
            if let k = out[id] { return k }
            if let at = stack.firstIndex(of: id) { throw GraphError.cycle(Array(stack[at...]) + [id]) }
            guard let n = index[id] else { throw GraphError.missingNode(id) }
            stack.append(id)
            var h = GraphHasher()
            h.add(n.kind.rawValue)
            for k in n.params.keys.sorted() { h.add(k); h.add(n.params[k]!) }
            for (name, src) in n.inputs.sorted(by: { $0.key < $1.key }) { h.add(name); h.add(try key(src)) }
            stack.removeLast()
            out[id] = h.state
            return h.state
        }
        _ = try key(id)
        return out
    }

    /// Evaluate `id` (default: `output`). Validates first, so type and cycle errors come before any work.
    func evaluate(_ id: Int? = nil, cache: GraphCache = GraphCache(), prune: Bool = false) throws -> GraphValue {
        guard let target = id ?? output else { throw GraphError.noOutput }
        try validate()
        let keys = try keys(for: target)
        let index = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        var done: [Int: GraphValue] = [:]
        func run(_ id: Int) throws -> GraphValue {
            if let v = done[id] { return v }
            let n = index[id]!
            let k = keys[id]!
            let v: GraphValue
            if let hit = cache.value(for: k) { v = hit } else {
                var ins: [String: GraphValue] = [:]
                for (name, src) in n.inputs { ins[name] = try run(src) }
                v = try GraphEvaluator.compute(n, inputs: ins)
                cache.store(v, key: k, id: id, kind: n.kind)
            }
            done[id] = v
            return v
        }
        var v = try run(target)
        // Nodes hand each other their RAW results (exact coordinates, whatever order the operator built them in):
        // canonicalising between nodes snapped vertices to a grid, and that moved a boolean's cut vertices into
        // sub-tolerance slivers its next boolean could not retriangulate (open edges). Only what leaves the graph
        // is canonical, so it is still byte-stable.
        if case .mesh(let m) = v, GraphEvaluator.needsCanonical(target, index: index) {
            v = cache.canonicalValue(for: keys[target]!) { .mesh(m.canonicalized()) }
        }
        if prune { cache.trim(keeping: Set(keys.values)) }
        return v
    }

    func evaluateMesh(_ id: Int? = nil, cache: GraphCache = GraphCache()) throws -> Mesh3 {
        let v = try evaluate(id, cache: cache)
        guard let m = v.mesh else { throw GraphError.typeMismatch(node: id ?? output ?? 0, input: "output", expected: .mesh, got: v.type) }
        return m
    }
}

enum GraphEvaluator {
    /// Parameters with every scalar / vec3 input merged over the same-named parameter.
    static func args(_ n: GraphNode, _ ins: [String: GraphValue]) -> GraphArgs {
        var p = n.params
        for (name, v) in ins {
            switch v {
            case .scalar(let s): p[name] = .number(s)
            case .vec3(let s): p[name] = .vec3(s)
            default: break
            }
        }
        return GraphArgs(p)
    }

    static func mesh(_ n: GraphNode, _ ins: [String: GraphValue], _ name: String) throws -> Mesh3 {
        guard let m = ins[name]?.mesh else { throw GraphError.missingInput(node: n.id, kind: n.kind.rawValue, input: name) }
        return m
    }
    static func sdf(_ n: GraphNode, _ ins: [String: GraphValue], _ name: String) throws -> SDFValue {
        guard let m = ins[name]?.sdf else { throw GraphError.missingInput(node: n.id, kind: n.kind.rawValue, input: name) }
        return m
    }
    static func transform(_ ins: [String: GraphValue], _ a: GraphArgs) -> GraphTransform {
        if case .transform(let t)? = ins["transform"] { return t }
        return GraphTransform(a)
    }

    /// Kinds whose engine operator orders its output by dictionary iteration (see GraphCanonical.swift).
    static let unordered: Set<NodeKind> = [.modifier, .subdivide, .decimate, .bevel, .boolean]

    /// Whether `id`'s mesh is, or is built from, an operator whose vertex / triangle order is not stable.
    static func needsCanonical(_ id: Int, index: [Int: GraphNode]) -> Bool {
        var seen = Set<Int>(), stack = [id]
        while let i = stack.popLast() {
            guard seen.insert(i).inserted, let n = index[i] else { continue }
            if unordered.contains(n.kind) { return true }
            stack.append(contentsOf: n.inputs.values)
        }
        return false
    }

    static func compute(_ n: GraphNode, inputs ins: [String: GraphValue]) throws -> GraphValue {
        try computeRaw(n, inputs: ins)
    }

    private static func computeRaw(_ n: GraphNode, inputs ins: [String: GraphValue]) throws -> GraphValue {
        let a = args(n, ins)
        switch n.kind {
        case .scalar: return .scalar(a.double("value", 0))
        case .vec3: return .vec3(a.vec3("value", .zero))
        case .math:
            let x = a.double("a", 0), y = a.double("b", 0)
            switch a.string("op", "add") {
            case "sub": return .scalar(x - y)
            case "mul": return .scalar(x * y)
            case "div": return .scalar(y == 0 ? 0 : x / y)
            case "min": return .scalar(Swift.min(x, y))
            case "max": return .scalar(Swift.max(x, y))
            case "add": return .scalar(x + y)
            case let o: throw GraphError.badParameter(node: n.id, message: "math op '\(o)' (add, sub, mul, div, min, max)")
            }
        case .makeTransform: return .transform(GraphTransform(a))

        case .box: return .mesh(GraphPrimitives.box(size: a.vec3("size", Vec3(1, 1, 1))))
        case .cylinder: return .mesh(GraphPrimitives.cylinder(radius: a.double("radius", 0.5), height: a.double("height", 1), segments: a.int("segments", 32)))
        case .sphere: return .mesh(GraphPrimitives.sphere(radius: a.double("radius", 0.5), segments: a.int("segments", 32)))
        case .cone:
            return .mesh(GraphPrimitives.cone(radius: a.double("radius", 0.5), topRadius: a.double("topRadius", 0), height: a.double("height", 1), segments: a.int("segments", 32)))
        case .torus:
            return .mesh(GraphPrimitives.torus(major: a.double("major", 0.5), minor: a.double("minor", 0.15), segments: a.int("segments", 32), tube: a.int("tubeSegments", 16)))

        case .meshTransform: return .mesh(try mesh(n, ins, "mesh").placed(transform(ins, a)))
        case .mirror:
            let axis: Mesh3.Axis
            switch a.string("axis", "x") { case "y": axis = .y; case "z": axis = .z; default: axis = .x }
            return .mesh(MirrorModifier(axis, at: a.double("plane", 0), keepOriginal: a.bool("keepOriginal", true)).apply(try mesh(n, ins, "mesh")))
        case .array:
            return .mesh(try mesh(n, ins, "mesh").arrayed(count: a.int("count", 2), offset: a.vec3("offset", Vec3(1, 0, 0))))
        case .radialArray:
            return .mesh(try mesh(n, ins, "mesh").radialArray(count: a.int("count", 6), around: a.vec3("centre", .zero), startAngle: a.double("startAngle", 0)))
        case .boolean: return .mesh(try boolean(n, a, try mesh(n, ins, "a"), try mesh(n, ins, "b")))
        case .modifier:
            let m = try GraphModifierRegistry.make(a.string("name", ""), a.table("args"), node: n.id)
            return .mesh(m.apply(try mesh(n, ins, "mesh")))
        case .bevel:
            return .mesh(try mesh(n, ins, "mesh").beveled(radius: a.double("radius", 0.01), creaseDegrees: a.double("crease", 30), segments: a.int("segments", 3)))
        case .subdivide:
            return .mesh(try mesh(n, ins, "mesh").subdivided(levels: a.int("levels", 1), creaseDegrees: a.double("crease", 35)))
        case .decimate:
            let m = try mesh(n, ins, "mesh")
            let target = a.int("targetTriangles", 0)
            return .mesh(target > 0 ? m.decimated(targetTriangles: target, creaseDegrees: a.double("crease", 40))
                                    : m.decimated(ratio: a.double("ratio", 0.5), creaseDegrees: a.double("crease", 40)))
        case .join:
            var out = Mesh3()
            for k in ins.keys.sorted(by: { (Int($0.dropFirst(2)) ?? 0) < (Int($1.dropFirst(2)) ?? 0) }) { if let m = ins[k]?.mesh { out.append(m) } }
            return .mesh(out)
        case .setUVs: return .mesh(try mesh(n, ins, "mesh").projectedUVs(try projection(n, a)))
        case .importMesh:
            guard let d = a.data("data"), let m = MeshBlob.decode(d) else { throw GraphError.badParameter(node: n.id, message: "importMesh has no valid 'data'") }
            return .mesh(m)

        case .distribute, .instance: return try GraphScatter.compute(n, a, ins)
        case .sdfBox, .sdfSphere, .sdfCylinder, .sdfCone, .sdfTorus, .sdfTransform, .sdfCombine, .sdfOffset, .sdfShell, .sdfCarve, .meshToSDF, .sdfToMesh:
            return try computeSDF(n, a, ins)
        }
    }

    static func boolean(_ n: GraphNode, _ a: GraphArgs, _ x: Mesh3, _ y: Mesh3) throws -> Mesh3 {
        let op: BooleanOp, old: BooleanModifier.Op
        switch a.string("op", "difference") {
        case "union": op = .union; old = .union
        case "intersection", "intersect": op = .intersection; old = .intersection
        case "difference", "subtract": op = .difference; old = .difference
        case let o: throw GraphError.badParameter(node: n.id, message: "boolean op '\(o)' (union, difference, intersection)")
        }
        switch a.string("engine", "robust") {
        case "robust":
            var o = BooleanOptions()
            if a.bool("keepCreases", false) { o.weld = .keepCreases }
            return x.boolean(op, y, options: o).mesh
        case "bsp": return BooleanModifier.bsp(x, y, old)
        case let e: throw GraphError.badParameter(node: n.id, message: "boolean engine '\(e)' (robust, bsp)")
        }
    }

    static func projection(_ n: GraphNode, _ a: GraphArgs) throws -> UVProjection {
        let kind: UVProjection.Kind
        switch a.string("projection", "box") {
        case "planar": kind = .planar
        case "box": kind = .box
        case "cylinder": kind = .cylinder
        case "sphere": kind = .sphere
        case let k: throw GraphError.badParameter(node: n.id, message: "projection '\(k)' (planar, box, cylinder, sphere)")
        }
        let s = a.double("scale", 1)
        return UVProjection(kind, origin: a.vec3("origin", .zero), axis: a.vec3("axis", Vec3(0, 0, 1)),
                            scale: Vec2(s, s), rotationDegrees: a.double("rotationDegrees", 0))
    }
}
