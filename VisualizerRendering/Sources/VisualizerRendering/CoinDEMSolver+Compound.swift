import Foundation
import Metal
import simd

// ── CoinDEMSolver + Compound ────────────────────────────────────────────────
//
// COMPOUND bodies (shape tag 6): one rigid body made of up to 128 oriented boxes
// (CoinCompound.swift has the mass maths). Constraint path only, like hulls and eggs.
// Each child box collides through the exact polytope narrowphase (CoinDEMNarrowphase.h):
// box–box / box–hull separating-axis test with a clipped face manifold, swept (sphere /
// capsule / egg) vs box exact closest features, SAT + clipping against a disc's 16-gon
// prism (caps exact, rim inscribed: ≤ 1.9 % R — the one shape approximated), planes /
// tubes / the pusher on the child's corners, and SAT + clipping against static boxes. The children of one compound never collide
// with each other; sleeping, waking and the static bounding reject treat the compound
// as one body (its bounding sphere is the union's).
//
// Registration writes the children into the shared shape table (`hullVertexBuffer`,
// 3 float4 per child: centre, half-extents, orientation — principal frame, COM at the
// origin), so every kernel that already binds the hull table sees them. A compound of more
// than 16 children follows them with its culling clusters (CoinCompound.swift, step 4): a
// header float4 (x = the cluster count G, as uint bits), G × 2 float4 (the cluster's AABB
// centre with (first | count << 16) in w, its half-extents), then the member list — every
// child index, uint16, grouped by cluster, eight per float4.

extension CoinDEMSolver {

    /// A registered compound: its index in the shape table and the frame change
    /// (stored = principalRotation⁻¹ · (design − comOffset)), so the render mesh can be
    /// drawn in the frame the solver simulates in — exactly HullHandle's convention.
    public struct CompoundHandle: Sendable {
        public let index: Int
        /// The child boxes in the PRINCIPAL frame (COM at the origin).
        public let children: [(center: SIMD3<Float>, halfExtents: SIMD3<Float>, orientation: simd_quatf)]
        public let comOffset: SIMD3<Float>
        public let principalRotation: simd_quatf
        public let boundingRadius: Float
        /// Each input box's share of the body's mass (input order).
        public let massFractions: [Float]
        /// Per-unit-mass inverse principal inertia (I⁻¹ = invMass · this).
        public let invInertiaK: SIMD3<Float>
    }

    /// Register a compound shape: 1…128 boxes (`CoinCompoundMath.maxChildren`) in the
    /// caller's design frame. nil when the input is degenerate (a non-positive extent or
    /// density), has more boxes than the cap, or the shape table is full.
    public func registerCompound(boxes: [CoinCompoundBox]) -> CompoundHandle? {
        guard shapeTable.count < Self.maxHulls,
              let prep = CoinCompoundMath.prepare(boxes) else { return nil }
        let n = prep.children.count
        let clusters = compoundClustersForTesting ? prep.clusters : []
        let g = clusters.count
        // Above the flat cap a header always follows the children (G = 0: walk every child).
        let clusterSlots = n > CoinCompoundMath.flatChildren ? 1 + 2 * g + (g > 0 ? (n + 7) / 8 : 0) : 0
        guard hullVertexCursor + 3 * n + clusterSlots <= Self.maxHullVertices else { return nil }
        let index = shapeTable.count
        let vp = hullVertexBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: Self.maxHullVertices)
        for (i, c) in prep.children.enumerated() {
            vp[hullVertexCursor + 3 * i + 0] = SIMD4(c.center, 0)
            vp[hullVertexCursor + 3 * i + 1] = SIMD4(c.halfExtents, 0)
            vp[hullVertexCursor + 3 * i + 2] = SIMD4(c.orientation.imag, c.orientation.real)
        }
        if clusterSlots > 0 {
            // Integer fields are stored as raw 32-bit words (the kernels read them with as_type).
            let t = hullVertexCursor + 3 * n
            let raw = UnsafeMutableRawPointer(vp)
            func word(_ slot: Int, _ lane: Int, _ v: UInt32) {
                raw.storeBytes(of: v, toByteOffset: (slot * 4 + lane) * MemoryLayout<UInt32>.stride, as: UInt32.self)
            }
            vp[t] = .zero
            word(t, 0, UInt32(g))
            var list: [UInt16] = []
            for (k, cl) in clusters.enumerated() {
                vp[t + 1 + 2 * k] = SIMD4(cl.center, 0)
                word(t + 1 + 2 * k, 3, UInt32(list.count) | (UInt32(cl.members.count) << 16))
                vp[t + 2 + 2 * k] = SIMD4(cl.halfExtents, 0)
                list += cl.members.map { UInt16($0) }
            }
            list += [UInt16](repeating: 0, count: (8 - list.count % 8) % 8)
            for q in 0..<(list.count / 8) {
                for j in 0..<4 {
                    word(t + 1 + 2 * g + q, j, UInt32(list[8 * q + 2 * j]) | (UInt32(list[8 * q + 2 * j + 1]) << 16))
                }
            }
        }
        hullRangeBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: Self.maxHulls)[index] =
            SIMD2(UInt32(hullVertexCursor), UInt32(n))
        hullVertexCursor += 3 * n + clusterSlots
        shapeTable.append(.compound(prep))
        return CompoundHandle(index: index,
                              children: prep.children.map { ($0.center, $0.halfExtents, $0.orientation) },
                              comOffset: prep.comOffset, principalRotation: prep.principalRotation,
                              boundingRadius: prep.boundingRadius, massFractions: prep.massFractions,
                              invInertiaK: prep.invInertiaK)
    }

    /// Activate a COMPOUND body. `position` is the COM and `orient` the principal
    /// frame's orientation; to place the design at (R_d, t_d) — as with hulls —
    /// pass orient = R_d ⊗ principalRotation and position = t_d + R_d·comOffset
    /// (or use `spawnCompound(design:…)`). Constraint path only.
    @discardableResult
    public func spawnCompound(at position: SIMD3<Float>,
                              compound: CompoundHandle,
                              velocity: SIMD3<Float> = .zero,
                              orient: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                              tumble: SIMD3<Float> = .zero,
                              mass: Float = 1,
                              friction: Float? = nil,
                              restitution: Float? = nil,
                              type: UInt32 = 0) -> Int? {
        assert(solverMode == .constraint,
               "spawnCompound requires solverMode == .constraint (legacy has no compound narrowphase)")
        guard compound.index >= 0, compound.index < shapeTable.count,
              case .compound(let prep) = shapeTable[compound.index],
              let slot = claimBodySlot() else { return nil }
        let ptr = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)
        let invMass: Float = mass > 1e-6 ? 1.0 / mass : 1.0
        ptr[slot] = CoinBody(position: position, invMass: invMass, velocity: velocity,
                             orient: orient, angVel: tumble,
                             shapeExtents: SIMD4(Float(prep.children.count), 0, 0, 6),   // w = 6 → compound
                             hullRef: SIMD4(Float(compound.index),
                                            prep.invInertiaK.x, prep.invInertiaK.y, prep.invInertiaK.z))
        ptr[slot].prevPos.w = prep.boundingRadius
        ptr[slot].vel.w     = prep.minHalfExtent
        finishSpawn(slot, friction: friction, restitution: restitution, type: type)
        noteBodyBound(prep.boundingRadius)
        return slot
    }

    /// Spawn a compound at its DESIGN pose (R_d, t_d): the boxes land exactly where
    /// `registerCompound` was given them, transformed by (R_d, t_d).
    @discardableResult
    public func spawnCompound(design rd: simd_quatf, _ td: SIMD3<Float>,
                              compound: CompoundHandle,
                              velocity: SIMD3<Float> = .zero, tumble: SIMD3<Float> = .zero,
                              mass: Float = 1, friction: Float? = nil, restitution: Float? = nil,
                              type: UInt32 = 0) -> Int? {
        let q = (rd * compound.principalRotation).normalized
        return spawnCompound(at: td + simd_act(rd, compound.comOffset), compound: compound,
                             velocity: velocity, orient: SIMD4(q.imag, q.real), tumble: tumble,
                             mass: mass, friction: friction, restitution: restitution, type: type)
    }

    /// World-space child boxes of a live compound body, read back from the GPU's own
    /// shape table and the body's current pose (centre, half-extents, orientation) — the
    /// COLLIDER exactly as the kernels see it.
    public func compoundColliderBoxes(of slot: Int) -> [(center: SIMD3<Float>, halfExtents: SIMD3<Float>, orientation: simd_quatf)]? {
        guard slot >= 0, slot < highWater else { return nil }
        let b = coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)[slot]
        guard b.posInvMass.w != 0, b.shapeExtents.w > 5.5, b.shapeExtents.w < 6.5 else { return nil }
        let idx = Int(b.hullRef.x + 0.5)
        guard idx >= 0, idx < shapeTable.count else { return nil }
        let r = hullRangeBuffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: Self.maxHulls)[idx]
        let vp = hullVertexBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: Self.maxHullVertices)
        let qb = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        let x = SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z)
        return (0..<Int(r.y)).map { i in
            let c = vp[Int(r.x) + 3 * i], he = vp[Int(r.x) + 3 * i + 1], q = vp[Int(r.x) + 3 * i + 2]
            let qc = simd_quatf(ix: q.x, iy: q.y, iz: q.z, r: q.w)
            return (x + simd_act(qb, SIMD3(c.x, c.y, c.z)), SIMD3(he.x, he.y, he.z), (qb * qc).normalized)
        }
    }
}
