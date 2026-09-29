import Foundation
import simd

// ── CoinCompound ─────────────────────────────────────────────────────────────
//
// CPU-side registration math for CoinDEMSolver's COMPOUND bodies (shape tag 6): one
// rigid body made of up to 128 oriented boxes — a worker's C-shaped hand, or the Digital
// Clock's toy tower crane, whose open-truss jib, counter-jib and tower head are ~110
// chords, diagonals and battens, each its own exact box (the truss's triangles are open,
// so a bar can fall through one). Done ONCE per compound shape, never per frame:
//
//   1. Mass: each box is a uniform solid of relative `density`, so its mass share is
//      density·volume / Σ density·volume (overlapping boxes count their overlap twice —
//      give non-overlapping boxes, or lower the density of an embedded one).
//   2. COM = the mass-weighted box centres; inertia about the COM = Σ each box's own
//      solid-box tensor rotated into the compound frame (R·diag·Rᵀ) plus the
//      PARALLEL-AXIS term m·(|d|²·1 − d·dᵀ) for its offset d from the COM.
//   3. Diagonalize (CoinHullMath.jacobiEigen) and re-express the child boxes in the
//      PRINCIPAL frame with the COM at the origin — the GPU models local inertia as a
//      diagonal, exactly as for hulls. The frame change is returned so the render mesh
//      (the same boxes) can be drawn in the frame the solver simulates in.
//
//   4. A compound of MORE than `flatChildren` (16) children also gets CULLING CLUSTERS:
//      its children grouped (≤ 16 each) along the principal axis of its longest extent —
//      the children longer than a cluster's share of that extent in clusters of their own —
//      each with the AABB of its children in the principal frame. The kernels test a
//      cluster's box against a static box / a partner's bounding sphere / a plane before any
//      of its children, and skip it whole when even its box is beyond the speculative margin
//      (no child of it could have touched: the contact set is unchanged, only the candidate
//      pairs that could never produce a contact are not listed). A child keeps its input
//      index everywhere (the collider read-back, the contacts' feature ids): a cluster lists
//      its children by index. A compound of ≤ 16 children has no clusters and runs exactly
//      as before.
//
// Everything is computed in Double; the result is exact for the union of the boxes as
// given (no voxelisation, no bounding-box approximation).

/// One box of a compound, in the caller's design frame.
public struct CoinCompoundBox: Sendable, Hashable {
    public var center: SIMD3<Float>
    public var halfExtents: SIMD3<Float>
    /// Box-local → design-frame rotation.
    public var orientation: simd_quatf
    /// Relative density (mass share ∝ density × volume). 1 = uniform.
    public var density: Float

    public init(center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                orientation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), density: Float = 1) {
        self.center = center
        self.halfExtents = halfExtents
        self.orientation = orientation
        self.density = density
    }

    public static func == (a: CoinCompoundBox, b: CoinCompoundBox) -> Bool {
        a.center == b.center && a.halfExtents == b.halfExtents && a.density == b.density
            && a.orientation.vector == b.orientation.vector
    }
    public func hash(into h: inout Hasher) {
        h.combine(center); h.combine(halfExtents); h.combine(orientation.vector); h.combine(density)
    }
}

enum CoinCompoundMath {

    /// GPU cap (CoinDEMNarrowphase.h CD_CMAX): children per compound. The pair list keeps
    /// 8 bits per piece and a contact's feature id 7 (cdPieceFeat), so 128 is the ceiling;
    /// a compound of ≤ 16 children runs exactly as it did under the old cap of 16.
    static let maxChildren = 128

    /// Above this many children a compound is culled cluster by cluster (CoinDEMNarrowphase.h
    /// CD_CFLAT); at or below it, child by child, exactly as before clusters existed.
    static let flatChildren = 16

    struct Child: Sendable {
        var center: SIMD3<Float>          // principal frame, COM at the origin
        var halfExtents: SIMD3<Float>
        var orientation: simd_quatf       // child-local → principal frame
    }

    /// A culling cluster: the AABB (principal frame) of its children, and their input indices.
    struct Cluster: Sendable {
        var center: SIMD3<Float>
        var halfExtents: SIMD3<Float>
        var members: [Int]
    }

    struct Prepared: Sendable {
        var children: [Child]
        /// Per-unit-mass INVERSE inertia diagonal in the principal frame (I⁻¹ = invMass · k).
        var invInertiaK: SIMD3<Float>
        var boundingRadius: Float
        var minHalfExtent: Float
        /// COM in the caller's design frame.
        var comOffset: SIMD3<Float>
        /// Design frame (COM-shifted) → principal frame: stored = R⁻¹ · (design − comOffset).
        var principalRotation: simd_quatf
        /// Each box's share of the total mass (sums to 1), in input order.
        var massFractions: [Float]
        /// Per-unit-mass inertia tensor about the COM in the DESIGN frame (for tests).
        var inertiaPerMassDesign: simd_double3x3
        /// Culling clusters (empty for ≤ `flatChildren` children).
        var clusters: [Cluster] = []
    }

    /// Group `children` (principal frame) into culling clusters of ≤ `flatChildren`: sorted along
    /// the axis of the union's longest extent; a child whose extent along it exceeds a cluster's
    /// share (the extent / cluster count) goes first, in clusters of its own, so one chord that
    /// spans the whole body does not stretch every cluster it would otherwise land in.
    static func clusters(of children: [Child]) -> [Cluster] {
        let n = children.count
        guard n > flatChildren else { return [] }
        func aabb(_ c: Child) -> (lo: SIMD3<Double>, hi: SIMD3<Double>) {
            let R = simd_double3x3(simd_quatd(ix: Double(c.orientation.imag.x), iy: Double(c.orientation.imag.y),
                                              iz: Double(c.orientation.imag.z), r: Double(c.orientation.real)).normalized)
            let he = SIMD3<Double>(Double(c.halfExtents.x), Double(c.halfExtents.y), Double(c.halfExtents.z))
            let e = simd_abs(R.columns.0) * he.x + simd_abs(R.columns.1) * he.y + simd_abs(R.columns.2) * he.z
            let ctr = SIMD3<Double>(Double(c.center.x), Double(c.center.y), Double(c.center.z))
            return (ctr - e, ctr + e)
        }
        let boxes = children.map(aabb)
        var lo = SIMD3<Double>(repeating: .greatestFiniteMagnitude), hi = -lo
        for b in boxes { lo = simd_min(lo, b.lo); hi = simd_max(hi, b.hi) }
        let span = hi - lo
        let axis = span.x >= span.y && span.x >= span.z ? 0 : (span.y >= span.z ? 1 : 2)
        let groups = (n + flatChildren - 1) / flatChildren
        let longer = span[axis] / Double(groups)
        func mid(_ i: Int) -> Double { (boxes[i].lo[axis] + boxes[i].hi[axis]) / 2 }
        let order = Array(0..<n).sorted { a, b in (mid(a), a) < (mid(b), b) }
        let long = order.filter { boxes[$0].hi[axis] - boxes[$0].lo[axis] > longer }
        let short = order.filter { boxes[$0].hi[axis] - boxes[$0].lo[axis] <= longer }
        var out: [Cluster] = []
        for list in [long, short] {
            var k = 0
            while k < list.count {
                let members = Array(list[k..<min(k + flatChildren, list.count)])
                var cl = SIMD3<Double>(repeating: .greatestFiniteMagnitude), ch = -cl
                for m in members { cl = simd_min(cl, boxes[m].lo); ch = simd_max(ch, boxes[m].hi) }
                out.append(Cluster(center: SIMD3<Float>((cl + ch) / 2), halfExtents: SIMD3<Float>((ch - cl) / 2),
                                   members: members))
                k += flatChildren
            }
        }
        return out
    }

    static func prepare(_ boxes: [CoinCompoundBox]) -> Prepared? {
        guard !boxes.isEmpty, boxes.count <= maxChildren else { return nil }
        var masses: [Double] = []
        var total = 0.0
        for b in boxes {
            let he = SIMD3<Double>(Double(b.halfExtents.x), Double(b.halfExtents.y), Double(b.halfExtents.z))
            guard he.x > 0, he.y > 0, he.z > 0, b.density > 0,
                  b.center.x.isFinite, b.center.y.isFinite, b.center.z.isFinite else { return nil }
            let m = Double(b.density) * 8 * he.x * he.y * he.z
            masses.append(m); total += m
        }
        guard total > 0 else { return nil }
        func d3(_ v: SIMD3<Float>) -> SIMD3<Double> { SIMD3(Double(v.x), Double(v.y), Double(v.z)) }
        func qd(_ q: simd_quatf) -> simd_quatd {
            simd_quatd(ix: Double(q.imag.x), iy: Double(q.imag.y), iz: Double(q.imag.z), r: Double(q.real)).normalized
        }
        var com = SIMD3<Double>.zero
        for (b, m) in zip(boxes, masses) { com += d3(b.center) * m }
        com /= total

        // Inertia about the COM, per unit (total) mass.
        var I = simd_double3x3(0)
        for (b, m) in zip(boxes, masses) {
            let w = m / total
            let he = d3(b.halfExtents)
            let own = SIMD3<Double>(he.y * he.y + he.z * he.z, he.x * he.x + he.z * he.z, he.x * he.x + he.y * he.y) / 3
            let R = simd_double3x3(qd(b.orientation))
            I += (R * simd_double3x3(diagonal: own) * R.transpose) * w
            let d = d3(b.center) - com
            let par = simd_double3x3(diagonal: SIMD3(repeating: simd_dot(d, d)))
                    - simd_double3x3(columns: (d * d.x, d * d.y, d * d.z))
            I += par * w
        }
        let (eig, vecs) = CoinHullMath.jacobiEigen(I)
        var Rp = vecs
        if simd_determinant(Rp) < 0 { Rp.columns.2 = -Rp.columns.2 }
        let qp = simd_quatd(Rp).normalized
        let qpInv = qp.inverse

        var children: [Child] = []
        var boundR = 0.0
        var minHE = Double.greatestFiniteMagnitude
        for b in boxes {
            let c = simd_act(qpInv, d3(b.center) - com)
            let q = (qpInv * qd(b.orientation)).normalized
            let he = d3(b.halfExtents)
            for i in 0..<8 {
                let corner = SIMD3<Double>((i & 1) != 0 ? he.x : -he.x, (i & 2) != 0 ? he.y : -he.y, (i & 4) != 0 ? he.z : -he.z)
                boundR = max(boundR, simd_length(c + simd_act(q, corner)))
            }
            minHE = min(minHE, min(he.x, min(he.y, he.z)))
            children.append(Child(center: SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)),
                                  halfExtents: b.halfExtents,
                                  orientation: simd_quatf(ix: Float(q.imag.x), iy: Float(q.imag.y),
                                                          iz: Float(q.imag.z), r: Float(q.real)).normalized))
        }
        // Relative floor, as for hulls (a real 3-D compound's moments are all ~L²).
        let L = 2 * boundR
        let floorEV = 1e-12 * L * L
        let ev = SIMD3<Double>(max(eig.x, floorEV), max(eig.y, floorEV), max(eig.z, floorEV))
        return Prepared(
            children: children,
            invInertiaK: SIMD3<Float>(Float(1 / ev.x), Float(1 / ev.y), Float(1 / ev.z)),
            boundingRadius: Float(boundR),
            minHalfExtent: Float(minHE),
            comOffset: SIMD3<Float>(Float(com.x), Float(com.y), Float(com.z)),
            principalRotation: simd_quatf(ix: Float(qp.imag.x), iy: Float(qp.imag.y),
                                          iz: Float(qp.imag.z), r: Float(qp.real)).normalized,
            massFractions: masses.map { Float($0 / total) },
            inertiaPerMassDesign: I,
            clusters: clusters(of: children))
    }
}
