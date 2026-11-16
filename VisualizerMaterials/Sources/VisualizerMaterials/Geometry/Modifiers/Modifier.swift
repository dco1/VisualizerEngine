//  Modifier stack (modelling kit, item E) — the protocol, the ordered stack and the helpers modifiers share.
//
//  A `Modifier` is a pure function Mesh3 -> Mesh3. A `ModifierStack` is an ordered list of them with an enable flag
//  each; `evaluate(_:)` re-runs the whole chain from the untouched base mesh, so editing a parameter, toggling or
//  reordering an entry and evaluating again is all "re-evaluation" there is (no hidden state, nothing cached).
//
//  Guarantees: evaluation is deterministic and never mutates the input. Modifiers that move vertices keep the
//  triangle list and re-derive normals by turning each corner's old normal with its face's turn (flat stays flat,
//  smooth stays smooth, creases stay creased). Not guaranteed: a modifier does not repair a bad input (a deform
//  can fold a mesh through itself), and no modifier makes UVs for faces it creates (rims get zero UVs).
//
//  GPU: the per-vertex modifiers (Displace, SimpleDeform, Lattice, Shrinkwrap) are independent per vertex and
//  would map onto a Metal kernel directly, but they take Swift closures / a target BVH, and at modelling sizes
//  (tens of thousands of vertices) the CPU path is milliseconds, so they stay CPU. A kernel for the procedural
//  noise + the analytic deforms is the natural follow-up if a scene needs millions of vertices.

import simd

public protocol Modifier: Sendable {
    func apply(_ mesh: Mesh3) -> Mesh3
    /// One-line description of the modifier and its parameters; the stack compares these for equality.
    var summary: String { get }
}

public extension Modifier {
    var summary: String { String(describing: self) }
}

public struct ModifierStack: Sendable, Equatable, CustomStringConvertible, Modifier {
    public struct Entry: Sendable, Equatable, Identifiable {
        public let id: Int
        public var name: String
        public var isEnabled: Bool
        public var modifier: any Modifier
        public static func == (a: Entry, b: Entry) -> Bool {
            a.id == b.id && a.name == b.name && a.isEnabled == b.isEnabled && a.modifier.summary == b.modifier.summary
        }
    }

    public private(set) var entries: [Entry] = []
    private var nextID = 0

    public init() {}

    /// Add a modifier at the end (or at `index`). Returns its entry id.
    @discardableResult
    public mutating func add(_ modifier: any Modifier, name: String? = nil, enabled: Bool = true, at index: Int? = nil) -> Int {
        let e = Entry(id: nextID, name: name ?? String(describing: type(of: modifier)), isEnabled: enabled, modifier: modifier)
        nextID += 1
        entries.insert(e, at: Swift.min(Swift.max(0, index ?? entries.count), entries.count))
        return e.id
    }

    public mutating func remove(id: Int) { entries.removeAll { $0.id == id } }

    public mutating func setEnabled(_ on: Bool, id: Int) {
        if let i = entries.firstIndex(where: { $0.id == id }) { entries[i].isEnabled = on }
    }

    public mutating func replace(id: Int, with modifier: any Modifier) {
        if let i = entries.firstIndex(where: { $0.id == id }) { entries[i].modifier = modifier }
    }

    /// Move entry `from` to position `to` (indices into `entries`).
    public mutating func move(from: Int, to: Int) {
        guard entries.indices.contains(from) else { return }
        let e = entries.remove(at: from)
        entries.insert(e, at: Swift.min(Swift.max(0, to), entries.count))
    }

    /// Run every enabled modifier in order over `base`. `upTo` (exclusive, counting all entries, enabled or not)
    /// stops early — the "show result up to this modifier" of an editor.
    public func evaluate(_ base: Mesh3, upTo: Int? = nil) -> Mesh3 {
        var m = base
        for e in entries.prefix(upTo ?? entries.count) where e.isEnabled { m = e.modifier.apply(m) }
        return m
    }

    public func apply(_ mesh: Mesh3) -> Mesh3 { evaluate(mesh) }

    /// One line per entry, `+`/`-` for enabled/disabled: the Equatable-ish description of what the stack does.
    public var description: String {
        entries.map { "\($0.isEnabled ? "+" : "-") \($0.name): \($0.modifier.summary)" }.joined(separator: "\n")
    }
    public var summary: String { entries.map { "\($0.isEnabled ? "+" : "-")\($0.name):\($0.modifier.summary)" }.joined(separator: " | ") }

    public static func == (a: ModifierStack, b: ModifierStack) -> Bool { a.entries == b.entries }
}

// MARK: - Shared helpers

/// A mesh with its vertices welded by position (1 µm), plus angle-weighted smooth vertex normals per welded vertex.
/// Mesh3 is usually flat-shaded (three vertices a triangle), so anything that moves vertices along a normal must
/// work on the welded point or it tears the surface along every hard edge.
struct WeldedMesh {
    var points: [Vec3] = []
    var normals: [Vec3] = []
    var tris: [SIMD3<Int>] = []
    var cornerID: [Int] = []          // mesh vertex index -> welded id

    init(_ m: Mesh3) {
        func key(_ p: Vec3) -> SIMD3<Int64> {
            SIMD3(Int64((p.x * 1e6).rounded()), Int64((p.y * 1e6).rounded()), Int64((p.z * 1e6).rounded()))
        }
        var ids: [SIMD3<Int64>: Int] = [:]
        cornerID.reserveCapacity(m.positions.count)
        for p in m.positions {
            let k = key(p)
            if let i = ids[k] { cornerID.append(i) } else {
                ids[k] = points.count; cornerID.append(points.count); points.append(p)
            }
        }
        normals = [Vec3](repeating: .zero, count: points.count)
        var t = 0
        while t + 2 < m.indices.count {
            let i = (Int(m.indices[t]), Int(m.indices[t + 1]), Int(m.indices[t + 2]))
            t += 3
            guard i.0 < m.positions.count, i.1 < m.positions.count, i.2 < m.positions.count else { continue }
            let a = cornerID[i.0], b = cornerID[i.1], c = cornerID[i.2]
            guard a != b, b != c, a != c else { continue }
            tris.append(SIMD3(a, b, c))
            let pa = points[a], pb = points[b], pc = points[c]
            let fn = cross3(pb - pa, pc - pa)
            guard len3(fn) > 1e-14 else { continue }
            let n = normalize3(fn)
            normals[a] += n * WeldedMesh.angle(pa, pb, pc)
            normals[b] += n * WeldedMesh.angle(pb, pc, pa)
            normals[c] += n * WeldedMesh.angle(pc, pa, pb)
        }
        for i in normals.indices { normals[i] = len3(normals[i]) > 1e-12 ? normalize3(normals[i]) : Vec3(0, 1, 0) }
    }

    static func angle(_ a: Vec3, _ b: Vec3, _ c: Vec3) -> Double {
        let u = b - a, v = c - a, lu = len3(u), lv = len3(v)
        guard lu > 1e-12, lv > 1e-12 else { return 0 }
        return acos(Swift.max(-1, Swift.min(1, dot3(u, v) / (lu * lv))))
    }
}

extension Mesh3 {
    /// The same triangles with `newPositions`; each corner's normal is turned by the rotation that takes its old
    /// face normal to the new one, so flat stays flat, smooth stays smooth, hard edges stay hard.
    func moved(to newPositions: [Vec3]) -> Mesh3 {
        var out = Mesh3(positions: newPositions, normals: normals, uvs: uvs, indices: indices)
        guard normals.count == positions.count else { return out }
        var done = [Bool](repeating: false, count: positions.count)
        var t = 0
        while t + 2 < indices.count {
            let i = (Int(indices[t]), Int(indices[t + 1]), Int(indices[t + 2]))
            t += 3
            let old = cross3(positions[i.1] - positions[i.0], positions[i.2] - positions[i.0])
            let new = cross3(newPositions[i.1] - newPositions[i.0], newPositions[i.2] - newPositions[i.0])
            guard len3(old) > 1e-14, len3(new) > 1e-14 else { continue }
            let from = normalize3(old), to = normalize3(new)
            for k in [i.0, i.1, i.2] where !done[k] {
                done[k] = true
                out.normals[k] = normalize3(Mesh3.turn(normals[k], from: from, to: to))
            }
        }
        return out
    }

    /// Rotate `v` by the shortest rotation taking unit `from` to unit `to` (Rodrigues).
    static func turn(_ v: Vec3, from: Vec3, to: Vec3) -> Vec3 {
        let axis = cross3(from, to), s = len3(axis), c = dot3(from, to)
        if s < 1e-9 { return v }               // parallel (or opposite: leave, no unique axis)
        let k = axis / s
        return v * c + cross3(k, v) * s + k * (dot3(k, v) * (1 - c))
    }
}

/// Orthonormal frame with `w` as its third axis.
func frame(for w: Vec3) -> (u: Vec3, v: Vec3, w: Vec3) {
    let w = normalize3(w)
    let helper = abs(w.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
    let u = normalize3(cross3(helper, w))
    return (u, cross3(w, u), w)
}
