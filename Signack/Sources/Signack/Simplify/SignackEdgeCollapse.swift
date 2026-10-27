import simd

/// Simplifies a CONNECTED surface (a trunk, a sofa) by quadric half-edge collapse.
///
/// **Half-edge, not full-edge:** a collapse u → v moves u onto v's existing position instead of an
/// optimal new one. Slightly worse error per triangle, but no new vertices — every level of the DAG
/// indexes the SAME vertex buffer as level 0, which is what lets a host register one buffer for all
/// LODs (S0's per-rung meshes paid for their vertices once per rung).
///
/// Locked (never moved): the group boundary (`lockedPositions` — the crack-free contract), every
/// edge that is not shared by exactly two of the group's triangles (open borders, non-manifold
/// fins), and attribute seams (two vertices at one position — moving one would tear the seam).
/// Rejected: collapses that turn a face past `minNormalDot` or fold one flat, and collapses that
/// would make the surface non-manifold (the link condition).
///
/// `collapseOpenBorders` (opt-in, off by default) frees an open border — a leaf's margin, a slit, a
/// hole — instead of locking it: a border vertex may slide along its own border onto a border
/// neighbour, and every border edge adds a heavily weighted plane through it, perpendicular to its
/// face, so the outline keeps its shape while it loses vertices. Without it a sheet whose every
/// vertex is near an edge (a cut leaf) cannot simplify at all. Corners where a border does not
/// pass straight through (≠ 2 border edges) stay locked.
public struct SignackEdgeCollapse: SignackSimplifier, Equatable {
    /// Minimum cosine between a face's normal before and after a collapse.
    public var minNormalDot: Float
    /// See the type doc. Off = every open edge locked (the trees' behaviour, unchanged).
    public var collapseOpenBorders: Bool
    /// How many times over a border plane's squared distance counts against the surface's own.
    public static let borderPenalty: Double = 40

    public init(minNormalDot: Float = 0.2, collapseOpenBorders: Bool = false) {
        self.minNormalDot = minNormalDot; self.collapseOpenBorders = collapseOpenBorders
    }

    public func simplify(_ g: SignackGroupInput, mesh: SignackMeshView) -> SignackGroupOutput? {
        var s = State(g, mesh, openBorders: collapseOpenBorders)
        s.run(target: g.targetTriangles, minNormalDot: minNormalDot)
        let out = s.output()
        guard out.count < g.triangles.count else { return nil }
        return SignackGroupOutput(triangles: out, atoms: nil, error: Float(s.maxError.squareRoot()))
    }

    /// The working mesh of one group, in local position ids.
    struct State {
        var pos: [SIMD3<Float>] = []
        var tv: [Int32] = []          // 3 local position ids per triangle
        var ta: [UInt32] = []         // 3 vertex ids per triangle (what the output indexes)
        var alive: [Bool]
        var vt: [[Int32]]             // live triangles around each local position
        var locked: [Bool]
        var border: [Bool]           // on a free open border (`collapseOpenBorders`)
        var dead: [Bool]
        var q: [SignackQuadric]
        var aliveCount: Int
        var maxError: Double = 0
        var heap = SignackMinHeap<UInt64>()

        init(_ g: SignackGroupInput, _ mesh: SignackMeshView, openBorders: Bool = false) {
            var localOf: [UInt32: Int32] = [:]
            var global: [UInt32] = []
            for t in g.triangles {
                for v in [t.x, t.y, t.z] {
                    let p = mesh.positionID[Int(v)]
                    let l: Int32
                    if let e = localOf[p] { l = e } else {
                        l = Int32(global.count); localOf[p] = l; global.append(p)
                        pos.append(mesh.positions[Int(v)])
                    }
                    tv.append(l); ta.append(v)
                }
            }
            let nT = g.triangles.count, nV = global.count
            alive = [Bool](repeating: true, count: nT)
            aliveCount = nT
            vt = [[Int32]](repeating: [], count: nV)
            for t in 0..<nT { for k in 0..<3 { vt[Int(tv[3 * t + k])].append(Int32(t)) } }
            locked = global.map { mesh.seam[Int($0)] || g.lockedPositions.contains($0) }
            dead = [Bool](repeating: false, count: nV)
            var valence: [UInt64: Int32] = [:]
            for t in 0..<nT {
                for k in 0..<3 {
                    valence[SignackEdges.key(UInt32(tv[3 * t + k]), UInt32(tv[3 * t + (k + 1) % 3])), default: 0] += 1
                }
            }
            border = [Bool](repeating: false, count: nV)
            var borderEdges = [Int](repeating: 0, count: nV)
            // A group's own cut line also looks like an open edge here; it is locked already
            // (`lockedPositions`), so only an edge with neither end locked is a free border.
            let preLocked = locked
            var freeBorder = Set<UInt64>()
            for (e, c) in valence where c != 2 {
                if openBorders && c == 1 && !preLocked[Int(e >> 32)] && !preLocked[Int(e & 0xFFFF_FFFF)] {
                    freeBorder.insert(e)
                    borderEdges[Int(e >> 32)] += 1; borderEdges[Int(e & 0xFFFF_FFFF)] += 1
                } else {
                    locked[Int(e >> 32)] = true; locked[Int(e & 0xFFFF_FFFF)] = true
                }
            }
            for v in 0..<nV where borderEdges[v] > 0 {
                if borderEdges[v] == 2 { border[v] = true } else { locked[v] = true }
            }
            q = [SignackQuadric](repeating: SignackQuadric(), count: nV)
            for t in 0..<nT {
                let a = Int(tv[3 * t]), b = Int(tv[3 * t + 1]), c = Int(tv[3 * t + 2])
                let qt = SignackQuadric(a: pos[a], b: pos[b], c: pos[c])
                q[a] += qt; q[b] += qt; q[c] += qt
                guard openBorders else { continue }
                // Each border edge's constraint plane: through the edge, perpendicular to its face.
                let fn = simd_cross(SIMD3<Double>(pos[b] - pos[a]), SIMD3<Double>(pos[c] - pos[a]))
                guard simd_length(fn) > 1e-18 else { continue }
                for k in 0..<3 {
                    let i = Int(tv[3 * t + k]), j = Int(tv[3 * t + (k + 1) % 3])
                    guard freeBorder.contains(SignackEdges.key(UInt32(i), UInt32(j))) else { continue }
                    let e = SIMD3<Double>(pos[j] - pos[i])
                    let pn = simd_cross(e, fn)
                    guard simd_length(pn) > 1e-18 else { continue }
                    let bq = SignackQuadric(planeNormal: simd_normalize(pn), through: SIMD3<Double>(pos[i]),
                                            weight: simd_length_squared(e), penalty: SignackEdgeCollapse.borderPenalty)
                    q[i] += bq; q[j] += bq
                }
            }
            for e in valence.keys.sorted() {
                let a = Int32(e >> 32), b = Int32(e & 0xFFFF_FFFF)
                push(a, b); push(b, a)
            }
        }

        @inline(__always) func cost(_ u: Int32, _ v: Int32) -> Double {
            (q[Int(u)] + q[Int(v)]).error(at: pos[Int(v)])
        }

        mutating func push(_ u: Int32, _ v: Int32) {
            guard !locked[Int(u)] else { return }
            heap.push(cost(u, v), (UInt64(UInt32(u)) << 32) | UInt64(UInt32(v)))
        }

        mutating func run(target: Int, minNormalDot: Float) {
            while aliveCount > target, let (c, payload) = heap.pop() {
                let u = Int32(payload >> 32), v = Int32(payload & 0xFFFF_FFFF)
                guard !dead[Int(u)], !dead[Int(v)], !locked[Int(u)] else { continue }
                let now = cost(u, v)
                if abs(now - c) > 1e-9 * max(now, 1e-12) { heap.push(now, payload); continue }
                if collapse(u, v, minNormalDot: minNormalDot) { maxError = max(maxError, now) }
            }
        }

        /// Collapse u onto v if it is legal; true when it happened.
        mutating func collapse(_ u: Int32, _ v: Int32, minNormalDot: Float) -> Bool {
            let around = vt[Int(u)].filter { alive[Int($0)] }
            var shared: [Int32] = [], moved: [Int32] = []
            for t in around {
                let b = 3 * Int(t)
                if tv[b] == v || tv[b + 1] == v || tv[b + 2] == v { shared.append(t) } else { moved.append(t) }
            }
            guard !shared.isEmpty else { return false }
            // A border vertex only slides along its own border: onto a neighbour across a border edge.
            if border[Int(u)] && (shared.count != 1 || !border[Int(v)]) { return false }
            // The vertex at v this side of the seam (if v is a seam) — every shared face must agree.
            var target: UInt32?
            var opposite = Set<Int32>()
            for t in shared {
                let b = 3 * Int(t)
                for k in 0..<3 where tv[b + k] == v {
                    if let x = target, x != ta[b + k] { return false }
                    target = ta[b + k]
                }
                for k in 0..<3 where tv[b + k] != u && tv[b + k] != v { opposite.insert(tv[b + k]) }
            }
            // Link condition: u and v may share no neighbours but the shared faces' third corners.
            var nu = Set<Int32>(), nv = Set<Int32>()
            for t in around { for k in 0..<3 { nu.insert(tv[3 * Int(t) + k]) } }
            for t in vt[Int(v)] where alive[Int(t)] { for k in 0..<3 { nv.insert(tv[3 * Int(t) + k]) } }
            nu.remove(u); nu.remove(v); nv.remove(u); nv.remove(v)
            guard nu.intersection(nv).count == opposite.count else { return false }
            // No face may flip or fold flat.
            for t in moved {
                let b = 3 * Int(t)
                let p = (0..<3).map { pos[Int(tv[b + $0])] }
                let n0 = simd_cross(p[1] - p[0], p[2] - p[0])
                let r = (0..<3).map { tv[b + $0] == u ? pos[Int(v)] : p[$0] }
                let n1 = simd_cross(r[1] - r[0], r[2] - r[0])
                let l0 = simd_length(n0), l1 = simd_length(n1)
                guard l1 > 1e-12 * max(1, l0) else { return false }
                if l0 > 1e-20, simd_dot(n0 / l0, n1 / l1) < minNormalDot { return false }
            }
            guard let target else { return false }
            for t in shared { alive[Int(t)] = false; aliveCount -= 1 }
            for t in moved {
                let b = 3 * Int(t)
                for k in 0..<3 where tv[b + k] == u { tv[b + k] = v; ta[b + k] = target }
                vt[Int(v)].append(t)
            }
            q[Int(v)] += q[Int(u)]
            dead[Int(u)] = true
            vt[Int(u)] = []
            vt[Int(v)].removeAll { !alive[Int($0)] }
            var ring = Set<Int32>()
            for t in vt[Int(v)] { for k in 0..<3 where tv[3 * Int(t) + k] != v { ring.insert(tv[3 * Int(t) + k]) } }
            for w in ring.sorted() { push(w, v); push(v, w) }
            return true
        }

        func output() -> [SIMD3<UInt32>] {
            alive.indices.filter { alive[$0] }.map { SIMD3(ta[3 * $0], ta[3 * $0 + 1], ta[3 * $0 + 2]) }
        }
    }
}
