import Foundation
import Metal
import simd

// ── CoinDEMSolver + Torsion ──────────────────────────────────────────────────
//
// plausibility: real — TORSIONAL (point) friction, engine plan item 5a (stage B2). A real
// contact is a small patch, not a point: pressed with normal force N, Coulomb friction across
// the patch resists spin ABOUT THE CONTACT NORMAL with a moment of order μ·N times the patch's
// radius. The narrowphase emits a point, whose friction rows have no lever about the normal, so
// a round-bottomed body given a yaw spin (a Weeble worker landing a turning hop) kept > 70 % of
// it for a whole second (CoinDEMActuationTests #8) — the scene had to stop it by hand. With a
// patch radius set, every contact of the body gets one more row: the relative spin about n
// driven to 0, its ACCUMULATED impulse over the substep bounded by μ·r_patch·Λn (Λn the
// contact's own accumulated normal impulse), exactly like the Coulomb rows beside it
// (Shaders/CoinDEM.metal, cdSolveContactTorsion).
//
// WHAT r IS. The effective moment arm that multiplies μ·N — not the geometric radius: a
// uniformly loaded flat disc of radius a gives (2/3)·a, a Hertzian patch (3π/16)·a ≈ 0.59·a.
// Worked for the Digital Clock's §B1 worker (14.6 g, fat radius 11.5 mm, ABS shell E ≈ 2.2 GPa
// on an oak top loaded across the grain, E ≈ 1–2 GPa → E* ≈ 1 GPa): a = (3·F·R / 4E*)^⅓ ≈
// 0.11 mm, so r ≈ 0.065 mm and a 5 rad/s spin stops in ≈ 0.6 s — a hard round bottom barely
// grips in torsion. A soft (rubber / felt) foot pad, or a flat base, has a patch of millimetres.
// A contact takes the larger of its two bodies' values (a static collider has none: the curved
// body's patch is the contact's). A grouped manifold (`manifoldSolve`) shares the term 1/n per
// point — a flat face resists spin mostly through the tangential friction of its spread points.
//
// OPT-IN. Every body starts at 0 (a point contact); while no body has a patch the solver's
// CD_FLAG_TORSION bit stays clear and nothing new runs — every existing scene and Daydream
// Home step exactly as before. A reused slot never inherits a patch (spawn / despawn / cull /
// clearAll clear it).
//
// SPRING FEET. A planted foot's pad has the same law with its own normal load (the leg force),
// so a worker standing on its shell AND a preloaded pad has μ·r·(N_shell + N_leg) of torsion in
// all — each patch counted once. While any patch is set, planted feet solve their pad rows every
// velocity iteration (CoinDEMSolver+Foot.swift), so the yaw motor and the shell's friction
// converge together instead of the pad getting the last word.

@MainActor
extension CoinDEMSolver {

    /// Give `slot`'s contacts a torsional friction patch of effective radius `radius` (m): each
    /// contact then resists spin about its normal with up to μ·radius·N (see the header for what
    /// `radius` means). 0 (the default) is a point contact: no torsion row. Constraint path only.
    public func setPatchRadius(_ slot: Int, _ radius: Float) {
        guard slot >= 0, slot < maxCoins else { return }
        let r = radius.isFinite ? max(radius, 0) : 0
        // Only a LIVE body takes a patch: despawn / cull / clearAll clear a freed slot's, and
        // the next spawn into a free slot relies on that (it does not reset the patch itself).
        if r > 0 {
            guard slot < highWater,
                  coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: maxCoins)[slot].posInvMass.w != 0
            else { return }
        }
        patchRadiusBuffer.contents().bindMemory(to: Float.self, capacity: maxCoins)[slot] = r
        if r > 0 { torsionPatchSlots.insert(slot) } else { torsionPatchSlots.remove(slot) }
    }

    /// The patch radius set on `slot` (0 = a point contact).
    public func patchRadius(of slot: Int) -> Float {
        guard slot >= 0, slot < maxCoins else { return 0 }
        return patchRadiusBuffer.contents().bindMemory(to: Float.self, capacity: maxCoins)[slot]
    }

    /// Whether any body has a patch (the solver then runs the torsion rows).
    public var torsionEnabled: Bool { !torsionPatchSlots.isEmpty }

    /// Slot bookkeeping (spawn / despawn / cull / clearAll): back to a point contact.
    func clearPatchRadius(_ slot: Int) {
        guard slot >= 0, slot < maxCoins else { return }
        guard torsionPatchSlots.contains(slot) || patchRadius(of: slot) != 0 else { return }
        patchRadiusBuffer.contents().bindMemory(to: Float.self, capacity: maxCoins)[slot] = 0
        torsionPatchSlots.remove(slot)
    }
}
