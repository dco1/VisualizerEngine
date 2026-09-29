import Foundation
import simd

// ── Pooled DISTANCE joints ─────────────────────────────────────────────────────────────────
//
// plausibility: real — the rigid rod `addDistanceJoint` makes (two anchor points held a fixed
// distance apart, free to swing), written into a slot RESERVED by `CoinJointPool`, so a scene can
// switch one on and off without `wakeAll` — the pool's ball / hinge / prismatic / weld already
// could (CoinDEMSolver+Actuation.swift, E5); the rod could not, and add/removeJoint wake every
// sleeping body in the world. Additive and opt-in: nothing calls it unless a scene does, so every
// existing world is untouched.
//
// Why a scene wants it: a rod to a far world point constrains its anchor ALONG the rod only —
// two of them at right angles pin a point in the horizontal plane while leaving it free to rise
// and sink and the body free to rotate about it (a guided plunger's rubber tip gripping a table:
// the Digital Clock's workers turning in place). A ball joint cannot do that: its vertical row
// is rigid, and beside a spring leg and a unilateral contact the vertical load split becomes
// indeterminate (measured: the base picked up 100–157 % of the toy's weight and 9–10 of 18
// turns stalled).

@MainActor
extension CoinDEMSolver {

    /// Enable a pooled joint slot (one `CoinJointPool.takeSlot()` handed out) as a DISTANCE joint:
    /// `worldAnchorA` on body A and `worldAnchorB` on body B — or that fixed world point when
    /// `bodyB` is nil — are held `restLength` apart (default: their separation now). The same lanes
    /// `addDistanceJoint` writes (swing friction off, `anchorB.w = 0`); the pool's `give` /
    /// `disableJoint(slot:)` switch it off again. Wakes A and B only. No-op on a slot outside the
    /// table or a dead body.
    public func enableDistance(slot: Int, bodyA: Int, bodyB: Int?, worldAnchorA: SIMD3<Float>,
                               worldAnchorB: SIMD3<Float>, restLength: Float? = nil,
                               collideConnected: Bool = false) {
        guard slot >= 0, slot < jointCount, pooledDistanceBodyIsLive(bodyA) else { return }
        if let b = bodyB, b == bodyA || !pooledDistanceBodyIsLive(b) { return }
        let rest = max(restLength ?? simd_length(worldAnchorB - worldAnchorA), 0)
        let world: UInt32 = 0xFFFF_FFFF
        let jp = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        jp[slot] = CoinJoint(
            meta: SIMD4(2, UInt32(bodyA), bodyB.map { UInt32($0) } ?? world, 1 | (collideConnected ? 2 : 0)),
            anchorA: SIMD4(pooledDistanceLocal(bodyA, worldAnchorA), rest),
            anchorB: SIMD4(bodyB.map { pooledDistanceLocal($0, worldAnchorB) } ?? worldAnchorB, 0),
            axisA: SIMD4(0, 1, 0, 0), axisB: SIMD4(0, 1, 0, 0),
            ref: SIMD4(0, 0, 0, 1))
        wake(bodyB.map { [bodyA, $0] } ?? [bodyA])
    }

    private func pooledDistanceBody(_ slot: Int) -> CoinBody {
        coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)[slot]
    }

    private func pooledDistanceBodyIsLive(_ slot: Int) -> Bool {
        slot >= 0 && slot < highWater && pooledDistanceBody(slot).posInvMass.w != 0
    }

    /// Body-local coordinates of a world point (the body's current pose).
    private func pooledDistanceLocal(_ slot: Int, _ world: SIMD3<Float>) -> SIMD3<Float> {
        let b = pooledDistanceBody(slot)
        let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
        return simd_act(q.inverse, world - SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z))
    }
}
