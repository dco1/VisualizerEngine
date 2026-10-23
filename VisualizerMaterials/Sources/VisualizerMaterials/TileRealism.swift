import Foundation

/// **What makes a tile read as a fired, laid object and not a flat colour grid** — the four things
/// the bake used to leave out, written once for `tile` and `tileLattice` (the same vocabulary the
/// two bakers already share).
///
///  1. **An eased rim.** A real tile's arris is never a razor 90°; a ~1 mm radius catches a thin
///     highlight along every edge. The `.square` edge used to bake the face flat to the joint.
///  2. **Gritty, sponged grout.** Grout is a sand-and-cement paste: grainy, stained unevenly,
///     struck concave. It used to be one flat colour at one flat roughness.
///  3. **Hue-shifted glaze per tile.** Glaze batches drift in HUE as well as value; the per-unit
///     tone was a brightness scalar only, so every tile was the same colour made lighter or darker.
///  4. **Glaze defects.** Pinholes, a few crazed tiles, and the odd chipped rim, scaled by how
///     glossy the finish is (a gloss glaze shows every one, a matte one hides most).
///
/// Everything is pure and keyed on texel coordinates, tile keys and tiled noise, so the bake still
/// wraps. `MaterialGenerator.tileRealismEnabled` switches the whole pass off for an A/B.
struct TileRealism {
    let enabled: Bool
    /// Metres per bake texel — the rim ease and chips are sized in metres but cannot be finer
    /// than the bake can hold.
    let texelMeters: Double
    let jointHalfMeters: Double
    let seed: UInt64
    let size: Int
    /// 0…1: how much of the glaze defect set this finish shows.
    let defectGain: Double
    let toneVariation: Double
    let cells: Int

    // MARK: taste dials (single named levers — Danny to retune)

    /// Rim radius of an eased square edge, metres. Floored at 2 texels so it survives the bake.
    static let rimRadiusMeters = 0.0012
    /// How deep the eased rim falls away, in height-field units (the joint recess is 0.7).
    static let rimEaseDepth = 0.11
    /// Extra recess at the centre of a joint — a sponged, concave grout profile.
    static let groutConcavity = 0.07
    /// Largest hue shift between tiles, as a fraction of channel value, at full tone variation.
    static let maxHueShift = 0.045
    /// Probability that a tile is crazed / has a chipped corner run.
    static let crazedShare = 0.16
    static let chippedShare = 0.10
    /// Pinhole density per texel on a full-gloss glaze.
    static let pinholeDensity = 0.0005
    /// The bisque a chip exposes — unglazed fired clay, warmer and lighter than the glaze.
    static let bisque = Vec3(0.80, 0.72, 0.60)

    init(params: TileParams, lattice: UnitLattice, size: Int) {
        enabled = MaterialGenerator.tileRealismEnabled
        texelMeters = lattice.layout.runMeters / Double(Swift.max(1, size))
        jointHalfMeters = lattice.jointHalfMeters
        seed = params.pattern.seed
        self.size = size
        toneVariation = params.pattern.toneVariation
        cells = Swift.max(lattice.layout.cols, 2)
        switch params.finish {
        case .gloss: defectGain = 1.0
        case .satin: defectGain = 0.6
        case .matte: defectGain = 0.2
        }
    }

    /// Distance from this texel to the unit's arris, metres (0 inside the joint).
    func arrisDistance(_ c: UnitLattice.Cell) -> Double { Swift.max(0, c.rimMeters - jointHalfMeters) }

    // MARK: 1 — eased rim

    /// Height to subtract at this distance from the arris. Only the `.square` edge gets it:
    /// `.bevelled` is a deliberate chamfer and `.handmade` already cushions its rim.
    func rimEase(_ c: UnitLattice.Cell, edge: TileEdge) -> Double {
        guard enabled, edge == .square, c.joint < 1 else { return 0 }
        let r = Swift.max(Self.rimRadiusMeters, 2.0 * texelMeters)
        let t = arrisDistance(c) / r
        let e = 1 - smoothstep(0, 1, t)
        return e * e * Self.rimEaseDepth
    }

    // MARK: 2 — grout

    struct Grout { var albedoMul: Double; var roughAdd: Double; var heightAdd: Double }

    /// The grout's own surface, applied in proportion to `joint`. `u`/`v` are the bake UV.
    func grout(_ c: UnitLattice.Cell, u: Double, v: Double, x: Int, y: Int) -> Grout {
        guard enabled, c.joint > 0 else { return Grout(albedoMul: 1, roughAdd: 0, heightAdd: 0) }
        // Sand: near-texel grain, one octave, so it reads as aggregate rather than blur.
        let grain = Noise.valueTiled(u, v, cells: Swift.max(size / 2, 8), seed: seed ^ 0x6A1) - 0.5
        // Stain: dirt and uneven water-take in sparse patches across whole joint runs. Patches, not a
        // blanket, so the grout's AUTHORED colour stays its average — a white grout stays white.
        let dirt = smoothstep(0.52, 0.85, Noise.fbmTiled(u, v, baseCells: cells * 2, octaves: 3, seed: seed ^ 0x6A2))
        // Darkens a touch toward the tile edge, where grout sits deepest and collects dirt.
        let deep = 1 - smoothstep(0, 1, c.edgeN)
        let mul = 1 - 0.12 * Swift.max(0, grain + 0.1) - 0.15 * dirt - 0.03 * deep
        let concave = (1 - smoothstep(0, 1, c.edgeN))   // 1 at the joint's centre
        return Grout(albedoMul: Swift.max(0.55, mul),
                     roughAdd: grain * 0.16,
                     heightAdd: -Self.groutConcavity * concave + grain * 0.05)
    }

    // MARK: 3 — hue

    /// A per-tile chroma multiplier: warm (more red, less blue) or cool, never a brightness change.
    func hue(unitHash h: UInt64) -> Vec3 {
        guard enabled else { return Vec3(1, 1, 1) }
        let s = (Noise.unit(Noise.mix(h ^ 0x4E1)) - 0.5) * 2
        let k = s * Self.maxHueShift * (0.35 + 0.65 * Swift.min(1, toneVariation / 0.14))
        return Vec3(1 + 0.6 * k, 1 + 0.1 * k, 1 - 0.8 * k)
    }

    // MARK: 4 — defects

    struct Defect { var albedoMul: Double; var albedoToBisque: Double; var roughAdd: Double; var heightAdd: Double }

    func defects(_ c: UnitLattice.Cell, unitHash h: UInt64, u: Double, v: Double, x: Int, y: Int) -> Defect {
        var d = Defect(albedoMul: 1, albedoToBisque: 0, roughAdd: 0, heightAdd: 0)
        guard enabled, c.joint < 1 else { return d }

        // Pinholes: single-texel dark specks where a gas bubble burst in the glaze.
        // Keyed on the SAMPLING texel (u, v), not (x, y), so a vertical run stays an exact transpose.
        let ix = Int(u * Double(size)), iy = Int(v * Double(size))
        let p = Noise.unit(Noise.hash2(ix, iy, seed ^ 0x91))
        if p < Self.pinholeDensity * defectGain {
            d.albedoMul *= 0.62; d.roughAdd += 0.25; d.heightAdd -= 0.05
        }

        // Crazing: a fine hairline network, on a minority of tiles only.
        if defectGain > 0.3, Noise.unit(Noise.mix(h ^ 0xC2A)) < Self.crazedShare {
            // Two crossing contour sets of fine noise: a thin polygonal network, not wandering loops.
            let cc = Swift.max(size / 22, 6)
            let n1 = Noise.valueTiled(u, v, cells: cc, seed: seed ^ 0xC2B)
            let n2 = Noise.valueTiled(u, v, cells: cc, seed: seed ^ 0xC2C)
            let dist = Swift.min(abs(n1 - 0.5), abs(n2 - 0.5))
            let line = (1 - smoothstep(0, 0.012, dist)) * defectGain
            d.albedoMul *= 1 - 0.07 * line; d.roughAdd += 0.05 * line; d.heightAdd -= 0.015 * line
        }

        // Chips: a few specks along the rim of a minority of tiles, exposing the bisque.
        if Noise.unit(Noise.mix(h ^ 0xC41)) < Self.chippedShare {
            let band = 1 - smoothstep(0, Swift.max(0.0016, 2.0 * texelMeters), arrisDistance(c))
            let n = Noise.valueTiled(u, v, cells: Swift.max(size / 3, 8), seed: seed ^ 0xC42)
            let chip = band * smoothstep(0.70, 0.76, n)
            d.albedoToBisque = chip; d.roughAdd += 0.5 * chip; d.heightAdd -= 0.10 * chip
        }
        return d
    }
}

public extension MaterialGenerator {
    /// **A/B lever for the tile realism pass** (`TileRealism`). ON by default; OFF bakes tile exactly
    /// as it did before the pass, so a before/after render differs by this switch alone.
    nonisolated(unsafe) static var tileRealismEnabled = true
}
