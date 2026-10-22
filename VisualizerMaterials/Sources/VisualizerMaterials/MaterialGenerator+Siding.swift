import simd
import Foundation

/// Which way a painted siding's boards run — it decides the grain direction and whether the
/// course line is baked.
public enum SidingStyle: String, Sendable, CaseIterable {
    /// Horizontal lap (clapboard / bevel) courses, grain along U.
    case lap
    /// Vertical boards with battens over the joints, grain along V.
    case boardAndBatten
}

extension MaterialGenerator {
    /// **Painted wood siding** — the SKIN a sided house wears, in any colour.
    ///
    /// The relief that makes siding read as siding — the shadow line under every lap course, the
    /// battens standing off the boards, the corner boards — is real GEOMETRY in the host
    /// (`CladdingMesh` in Daydream Home): at a 22 mm butt a course is half a tenth of the SSAO
    /// kernel, and a normal map cannot cast the sun's shadow line at all. So this material is only
    /// what geometry cannot carry: the board grain telegraphing through the paint, the satin
    /// paint's uneven sheen, and — for lap — a faint occlusion band at the top of each course's
    /// exposure, where the butt of the course above overhangs it. That band is what keeps the
    /// course rhythm when the house is far enough away that the 22 mm step is under a pixel.
    ///
    /// `courses` is how many lap exposures one tile holds; the host sets the tile's UV period to
    /// `courses × exposure` and starts its geometry courses at world-height multiples of the
    /// exposure plus `coursePhase` of one (a butt at world height `(m + coursePhase) × exposure`),
    /// so the baked band lands under the real butt rather than beside it. The default half-course
    /// phase keeps every course edge OFF the tile wrap: a butt line on the wrap is a hard tone step
    /// exactly where TextureAudit (rightly) reads a seam.
    ///
    /// Channels are `.wood` (COHERENT for anti-tiling): the course period and the grain are
    /// directional patterns, and the hex de-repeat would overlay offset copies of the band.
    public static func paintedSiding(size: Int = MaterialGenerator.bakeSize,
                                     color: Vec3 = Vec3(0.34, 0.40, 0.40),
                                     style: SidingStyle = .lap, courses: Int = 4,
                                     coursePhase: Double = 0.5, seed: UInt64 = 31) -> MaterialChannels {
        var ch = MaterialChannels(size: size, category: .wood)
        let n = max(1, courses)
        let lap = style == .lap
        for y in 0..<size {
            for x in 0..<size {
                let u = Double(x) / Double(size), v = Double(y) / Double(size)
                // Grain telegraphing through two coats: long streaks along the board, 40:1.
                let grain = lap
                    ? Noise.fbmTiledAniso(u, v, cellsX: 3, cellsY: 120, octaves: 3, seed: seed)
                    : Noise.fbmTiledAniso(u, v, cellsX: 120, cellsY: 3, octaves: 3, seed: seed)
                let drift = Noise.fbmTiled(u, v, baseCells: 3, octaves: 2, seed: seed ^ 0x2B) - 0.5
                var band = 0.0, board = 0.0
                if lap {
                    // c: 0 at a course's butt, 1 at the next butt up. The course above overhangs
                    // the top of this exposure — a soft occlusion band just under it.
                    let cv = v * Double(n) + 1 - coursePhase   // c = 0 on a butt
                    let c = cv - floor(cv)
                    // It lets go just short of the butt line rather than ON it, so the step back
                    // to full tone is a soft one inside the course, not a hard edge that lands on
                    // the tile wrap (TextureAudit's seam tell).
                    band = smoothstep(0.75, 0.96, c) * (1 - smoothstep(0.975, 1.0, c))
                    // Each course a whisper different (boards are cut from different stock).
                    let k = UInt64(Int(floor(cv)) % n)   // (floor(cv) ∈ 0…n)
                    board = Noise.valueTiled(Double(k) / Double(n), 0.5, cells: n, seed: seed ^ 0x77) - 0.5
                } else {
                    board = Noise.fbmTiledAniso(u, v, cellsX: 4, cellsY: 1, octaves: 1, seed: seed ^ 0x77) - 0.5
                }
                let tone = 1.0 + (lap ? 0.030 : 0.075) * (grain - 0.5) + 0.010 * drift + (lap ? 0.016 : 0.045) * board - 0.085 * band
                ch.albedo[ch.idx(x, y)] = clampBand(color * tone)
                ch.height[ch.idx(x, y)] = clamp01(0.5 + 0.6 * (grain - 0.5) - 0.25 * band)
                // Satin exterior paint: the grain ridges hold a little more sheen than the valleys.
                ch.roughness[ch.idx(x, y)] = clamp01(0.58 - 0.10 * (grain - 0.5) + 0.12 * drift + 0.06 * band)
            }
        }
        // The board's grain direction (TextureAudit's wood tell #7): along U for lap, V for battens.
        ch.grainTangent = [Vec2](repeating: lap ? Vec2(1, 0) : Vec2(0, 1), count: size * size)
        ch.clearcoat = 0
        ch.deriveNormals(strength: 1.4)
        addMicroDetail(&ch, seed: seed ^ 0xD4, baseCells: 96, strength: 0.30,
                       grainAspect: 6, grainAlongU: lap)
        return ch
    }
}
