import Foundation
import Metal
import simd

// ── CoinDEMSolver + SheaveFriction ───────────────────────────────────────────
//
// plausibility: real — Coulomb friction in the SHEAVES of a rope fall (VZ-0168, stage B2). A
// crane's hook block hangs on falls of rope that run over sheaves in the trolley and the block.
// When the block swings in the sheaves' plane by θ, the rope rolls θ round every sheave (rope
// moves from one fall to its neighbour across the sheave), so every sheave turns on its pin,
// and the pin's Coulomb friction resists with a moment μ_pin·r_pin·N_pin — N_pin ≈ 2T for a
// rope wrapped 180°. That is what stops a real toy crane's hook swinging within a few seconds;
// the engine's rigid distance "falls" were frictionless rods, so the §C1 block swung as a
// perfect pendulum (≈ 1.3 mm at its 0.6 s period for the whole 10 s idle of the fence world)
// and the crane never slept (VZ-0168).
//
// THE MODEL. Per fall, the friction moment its end supports exert against its swing is
// M = c·T (T the fall's tension, c the friction ARM in metres). On the distance joint that is a
// force pair ⟂ the fall at its two anchors of at most M / L = (c/L)·T: two friction rows on the
// anchors' relative velocity ⟂ the fall, their ACCUMULATED impulse clamped to a disc of radius
// (c/L)·|Λ_axial| — the contact friction law, with the rope's own tension as the normal load
// (CoinDEM.metal, "Swing friction of a DISTANCE joint"). A block swinging with amplitude x on
// falls of length L then loses 4c of amplitude per cycle and comes to rest within ±c of plumb
// (the restoring force m·g·x/L falls below the friction (c/L)·m·g) — Coulomb damping: no
// viscous term, nothing that depends on a frame rate, and with c = 0 (the default of every
// distance joint) exactly the frictionless rod it always was.
//
// THE ROPE. The falls stay inextensible (rigid distance rows), so they have no axial bounce
// mode to damp; the rope's bending hysteresis over the sheaves is rate-independent, like pin
// friction, and is lumped into the same arm.
//
// CALL RULE (as CoinDEMSolver+Actuation): @MainActor, writes the shared joint table — call only
// while no command buffer that encodes this solver is in flight.

@MainActor
extension CoinDEMSolver {

    /// The friction arm (m) of one rope fall that runs over a sheave at each end whose rotation
    /// the swing drives: each sheave's pin carries ≈ 2T (180° wrap) and is shared by the two falls
    /// that leave it, so each end contributes μ_pin·r_pin per unit of this fall's tension.
    ///   • `pinFriction` — μ of the sheave bore SLIDING on its pin. Measured, dry: pure PA6 on
    ///     smooth steel 0.28 at 2.5 MPa and 0.23 at 8 MPa (Voyer et al., "Static and Dynamic
    ///     Friction of Pure and Friction-Modified PA6 Polymers in Contact with Steel Surfaces",
    ///     Lubricants 2019, 7, 17 — Fig. 11a), which also cites 0.25–0.35 sliding and 0.35–0.50
    ///     static for PA6 / PA6.6 on carbon steel and 0.3–0.4 sliding on bearing steel; nylon on
    ///     nylon 0.15–0.25 and nylon on steel 0.4, static (Engineering ToolBox, "Friction and
    ///     Friction Coefficients for various Materials").
    ///   • `pinRadius` — the pin's radius (a toy crane's 1.5 mm steel axle: 0.75 mm).
    ///   • `sheaves` — sheaves along the fall that turn as it swings (2: one at the trolley, one
    ///     in the block).
    public static func sheaveFallFrictionArm(pinFriction: Float, pinRadius: Float, sheaves: Int = 2) -> Float {
        max(pinFriction, 0) * max(pinRadius, 0) * Float(max(sheaves, 0))
    }

    /// The Digital Clock crane's falls (§C1): toy nylon sheaves on 1.5 mm steel pins, a sheave
    /// at each end, μ 0.25 — dry PA6 sliding on smooth steel (0.23–0.28 measured, above) and the
    /// low end of the published nylon-on-steel span, so the settle time it gives is if anything
    /// long (0.3–0.4 settles sooner). c = 2·0.25·0.75 mm = 0.375 mm: a swing loses 1.5 mm of
    /// amplitude per cycle and rests within 0.375 mm of plumb.
    public static let toyCraneFallFrictionArm: Float = sheaveFallFrictionArm(pinFriction: 0.25, pinRadius: 0.00075)

    /// Give a DISTANCE joint (a rope fall) Coulomb swing friction with arm `arm` (m; see the
    /// header — `sheaveFallFrictionArm` computes it from the sheaves). 0 removes it: the fall is a
    /// frictionless rod. Ignored for any other joint type. Lane: anchorB.w (unused by a distance
    /// joint, which `addDistanceJoint` writes as 0).
    public func setDistanceSwingFriction(_ joint: Int, arm: Float) {
        guard joint >= 0, joint < Self.maxJoints else { return }
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        guard j[joint].meta.x == 2, (j[joint].meta.w & 1) != 0 else { return }
        j[joint].anchorB.w = arm.isFinite ? max(arm, 0) : 0
    }

    /// A distance joint's swing-friction arm (m), or nil if `joint` is not a live distance joint.
    public func distanceSwingFriction(_ joint: Int) -> Float? {
        guard joint >= 0, joint < Self.maxJoints else { return nil }
        let j = jointBuffer.contents().bindMemory(to: CoinJoint.self, capacity: Self.maxJoints)
        guard j[joint].meta.x == 2, (j[joint].meta.w & 1) != 0 else { return nil }
        return j[joint].anchorB.w
    }
}
