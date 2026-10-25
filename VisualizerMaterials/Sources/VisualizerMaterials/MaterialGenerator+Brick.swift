import simd
import Foundation

// MARK: - Brick: fired-clay facing brick in a mortar bed (DH-1497)

/// **A clay facing brick's customization** — the shared laid-unit spine (`UnitPatternParams`:
/// size, bond, run, body and mortar colour, joint, per-brick spread, seed) and nothing else.
/// Brick is the same object as tile and pavers; what makes it brick is its FACE
/// (`MaterialGenerator.brick`), not a second set of settings.
///
/// **The standard, stated once:** the metric work size of BS EN 771-1 / the UK standard brick —
/// a **215 × 65 mm face** (102.5 mm bed depth) laid with **10 mm joints**, i.e. a 225 × 75 mm
/// coordinating module. That module is EXACTLY 3:1, which is why the shape is `.plank` and why
/// the US modular brick (7⅝ × 2¼″ face + ⅜″ joint = an 8 × 2⅔″ module) is the same pattern at a
/// 203 mm module. `unitSizeMeters` is the MODULE (face + one joint), because that is what the
/// shared layout lays on its grid: the joint is drawn inside the module's boundary, half a joint
/// on each side.
public struct BrickParams: Equatable, Hashable, Sendable, Codable {
    public var pattern: UnitPatternParams

    /// Face length / height and joint width of the standard brick, metres.
    public static let faceLengthMeters = 0.215
    public static let faceHeightMeters = 0.065
    public static let jointMeters = 0.010
    /// The coordinating module the grid is laid on: face + one joint (225 mm).
    public static let moduleMeters = faceLengthMeters + jointMeters

    /// SINGLE NAMED LEVER: a red clay facing brick, linear. sRGB ≈ (160, 97, 79) — the warm,
    /// slightly brown red of a common machine-made facing brick, not a paint red. Mean Y ≈ 0.16,
    /// inside the 0.12–0.25 LRV band published for red clay facing bricks.
    public static let defaultBody = Vec3(0.35, 0.12, 0.078)
    /// SINGLE NAMED LEVER: a cement–lime–sand mortar, linear. sRGB ≈ (167, 162, 150) — a
    /// believable grey-tan, deliberately NOT the near-white grout tile defaults to.
    public static let defaultMortar = Vec3(0.39, 0.36, 0.31)

    public init(pattern: UnitPatternParams) { self.pattern = pattern }

    public init(bodyColor: Vec3 = BrickParams.defaultBody,
                mortarColor: Vec3 = BrickParams.defaultMortar,
                bond: UnitBond = .running,
                toneVariation: Double = 0.5,
                seed: UInt64 = 11) {
        pattern = UnitPatternParams(shape: .plank, unitSizeMeters: Self.moduleMeters, bond: bond,
                                    bodyColor: bodyColor, jointColor: mortarColor,
                                    jointFraction: Self.jointMeters / Self.moduleMeters,
                                    toneVariation: toneVariation, seed: seed)
    }

    /// Tolerant decode — missing fields take the standard brick.
    public init(from decoder: Decoder) throws {
        let d = try decoder.container(keyedBy: CodingKeys.self)
        self.init(pattern: try d.decodeIfPresent(UnitPatternParams.self, forKey: .pattern) ?? BrickParams().pattern)
    }

    /// The grid this brick lays on a surface (the shared tile layout — any size stays exact on an
    /// adjustable wall, quantised on a mesh whose UVs were baked earlier).
    public func layout(on tiling: SurfaceTiling) -> TileLayout { TileLayout.of(pattern, on: tiling) }

    /// The clay colours a brick shop sells, as presets. Each is a body colour; the per-brick
    /// spread, the firing flash and the speckle are the generator's and apply to all of them.
    public static let presets: [(name: String, params: BrickParams)] = [
        ("Red", BrickParams()),
        // A yellow/buff stock brick (London stock, Chicago common): pale, warm, sandy.
        ("Buff", BrickParams(bodyColor: Vec3(0.46, 0.33, 0.19), mortarColor: Vec3(0.42, 0.40, 0.36))),
        // A dark brown / "blue" engineering-style brick with a darker mortar.
        ("Brown", BrickParams(bodyColor: Vec3(0.17, 0.085, 0.060), mortarColor: Vec3(0.30, 0.29, 0.27),
                              toneVariation: 0.4)),
    ]
}

public extension MaterialGenerator {

    /// **Depth range of the brick's height field**, metres: the face's crown sits at the top of
    /// it, a struck mortar bed ~4 mm below. The normal is derived from height at this PHYSICAL
    /// scale (`brick` turns it into a `deriveNormals` strength from the bake's texel size), so a
    /// joint reads as the groove it is under a raking sun rather than as a printed line.
    static let brickReliefMeters = 0.008

    /// **Brick** — fired-clay facing brick laid in mortar, on the shared laid-unit lattice.
    ///
    /// The grid (bond, run, joint, per-unit identity) is `UnitLattice`, exactly as tile's is; what
    /// is brick here is the face and the joint:
    ///  - **per-brick clay**: each brick draws a value, a hue shift toward orange or toward brown,
    ///    and a small chance of being a dark FLASHED (kiln-reduced, purple-brown) or a pale
    ///    under-fired brick — a fired-clay batch varies far more than a glaze, and that variety is
    ///    what a brick wall looks like from across the street;
    ///  - **firing flash**: a soft darkening across part of each face, in a per-brick direction;
    ///  - **face texture**: low-frequency mottle, dark iron specks, and a sanded tooth;
    ///  - **the arris**: a ~2.5 mm roundover with per-brick CHIPS knocked out of it, exposing a
    ///    slightly fresher clay;
    ///  - **the joint**: a recessed, concave (bucket-handle) mortar bed ~4 mm behind the face,
    ///    sandy grey-tan, darkened toward the brick edges where the cavity occludes it.
    ///
    /// Roughness varies with all of it (flashed faces are glassier, the mortar is the roughest
    /// thing on the wall) and the diffuse roughness is 1.0 — OpenPBR's own brick example.
    ///
    /// Per-brick tone de-repeat: a horizontal running bond declares its cells with a NEGATIVE x
    /// count (`MaterialChannels.patternCells`), which the engine reads as "odd courses offset half
    /// a unit", so every PHYSICAL brick across a whole wall gets its own tone draw rather than the
    /// bake's 1.35 m repeat printing the same pattern of dark and light bricks.
    static func brick(size: Int = MaterialGenerator.bakeSize,
                      params: BrickParams = BrickParams(),
                      tiling: SurfaceTiling = .librarySwatch) -> MaterialChannels {
        let p = params.pattern
        let layout = params.layout(on: tiling)
        let lattice = UnitLattice(pattern: p, layout: layout, size: size)
        var ch = MaterialChannels(size: size, category: .brick)

        // Per-physical-brick tone de-repeat (see the doc above). Third-running and vertical runs
        // vary per course only (the bonded axis ships 0, as tile's does); lattices declare nothing.
        let cols = layout.cols, rows = layout.rows
        switch (p.bond, p.run) {
        case (.stack, .horizontal):   ch.patternCells = Vec2(Double(cols), Double(rows))
        case (.stack, .vertical):     ch.patternCells = Vec2(Double(rows), Double(cols))
        case (.running, .horizontal): ch.patternCells = Vec2(-Double(cols), Double(rows))
        case (_, .horizontal) where !p.bond.isLattice: ch.patternCells = Vec2(0, Double(rows))
        case (_, .vertical) where !p.bond.isLattice:   ch.patternCells = Vec2(Double(rows), 0)
        default:                      ch.patternCells = .zero
        }
        ch.patternJitter = ch.patternCells == .zero ? 0 : 0.07

        let tv = p.toneVariation
        let texelM = layout.runMeters / Double(size)
        let jointHalfM = lattice.jointHalfMeters
        let relief = brickReliefMeters
        // Speckle lattice: ~8 mm cells, a whole number per bake so it wraps.
        let speckCells = Swift.max(8, Int((layout.runMeters / 0.008).rounded()))
        // Chip lattice along the arris: ~22 mm cells.
        let chipCells = Swift.max(4, Int((layout.runMeters / 0.022).rounded()))
        let mottleCells = Swift.max(4, Int((layout.runMeters / 0.015).rounded()))
        let patchCells = Swift.max(4, Int((layout.runMeters / 0.05).rounded()))
        let mottleOct = Noise.bandLimitedOctaves(3, baseCells: mottleCells, size: size)
        let mortarCells = Swift.max(8, Int((layout.runMeters / 0.012).rounded()))
        // Sand / sanded tooth: two-texel value noise — the finest the bake can draw without the
        // per-texel sparkle white noise turns into once a normal is derived from it.
        let fineCells = Swift.max(8, size / 2)
        // Sand-face pits: ~8 mm cells, each pit ~1 texel across.
        let pitCells = Swift.max(8, Int((layout.runMeters / 0.008).rounded()))

        for y in 0..<size {
            for x in 0..<size {
                let (u, v) = lattice.uv(x: x, y: y)
                let c = lattice.cell(u: u, v: v)
                // Signed distance from the brick's arris, metres: + into the face, − into the joint.
                let e = c.rimMeters - jointHalfM
                // Coverage of brick vs mortar, antialiased over one texel.
                let w = smoothstep(-0.5 * texelM, 0.5 * texelM, e)

                // ── the brick ────────────────────────────────────────────────────────────
                let h0 = Noise.hash2(c.keyI, c.keyJ, p.seed)
                let r1 = Noise.unit(h0), r2 = Noise.unit(Noise.mix(h0 ^ 0xB1)), r3 = Noise.unit(Noise.mix(h0 ^ 0xC2))
                let r4 = Noise.unit(Noise.mix(h0 ^ 0xD3)), r5 = Noise.unit(Noise.mix(h0 ^ 0xE4))
                var body = p.bodyColor * (1 + (r1 - 0.5) * 0.8 * tv)
                let hue = r2 * 2 - 1
                body = hue > 0 ? mix(body, body * Vec3(1.10, 1.00, 0.80), hue * tv * 1.4)
                               : mix(body, body * Vec3(0.84, 0.81, 0.88), -hue * tv * 1.4)
                // A few bricks per hundred come out of the kiln reduced (flashed purple-brown) or
                // under-fired (pale salmon). Kept rare: a distinctive brick is a LANDMARK, and the
                // bake repeats every ~1.35 m, so a common one prints a lattice of dark bricks.
                let flashedBrick = r3 < 0.05 * Swift.min(1, tv * 2)
                let paleBrick = !flashedBrick && r3 > 1 - 0.05 * Swift.min(1, tv * 2)
                if flashedBrick { body = body * Vec3(0.68, 0.57, 0.62) }
                if paleBrick    { body = body * Vec3(1.12, 1.17, 1.15) }
                // Firing flash: a soft dark band across part of the face, per-brick direction —
                // the end that sat nearest the fire.
                let ang = r4 * 2 * Double.pi
                let g = (c.fx - 0.5) * cos(ang) + (c.fy - 0.5) * sin(ang) * 0.6
                let flash = (0.06 + 0.30 * r5 * r5) * tv * 2 * smoothstep(-0.05, 0.45, g)
                body = mix(body, body * Vec3(0.66, 0.56, 0.62), Swift.min(1, flash))
                // Face texture: patchy clay (a few blotches per brick), fine mottle, iron specks.
                let patch = Noise.fbmTiled(u, v, baseCells: patchCells, octaves: 2, seed: p.seed ^ 0x4C)
                let mottle = Noise.fbmTiled(u, v, baseCells: mottleCells, octaves: mottleOct, seed: p.seed ^ 0x51)
                body = body * (0.86 + 0.20 * patch) * (0.95 + 0.10 * mottle)
                // Specks: sparse, each its own size and darkness — not a polka-dot screen.
                let sp = Noise.voronoiTiled(u, v, cells: speckCells, jitter: 1, seed: p.seed ^ 0x77)
                let sh = Noise.mix(sp.cellId)
                let speckR = 0.08 + 0.14 * Noise.unit(sh)
                let speck = (Noise.unit(sp.cellId) < 0.16)
                    ? (1 - smoothstep(speckR * 0.6, speckR, sp.f1)) * (0.4 + 0.5 * Noise.unit(Noise.mix(sh)))
                    : 0
                body = mix(body, body * Vec3(0.42, 0.38, 0.40), speck)
                let grain = Noise.valueTiled(u, v, cells: fineCells, seed: p.seed ^ 0x33)
                body = body * (0.95 + 0.10 * grain)
                // Arris chips: a ragged bite out of the roundover where the chip field peaks.
                let chipN = Noise.fbmTiled(u, v, baseCells: chipCells, octaves: 2, seed: p.seed ^ 0x9A)
                let chipBite = smoothstep(0.62, 0.80, chipN) * 0.006   // up to 6 mm into the face
                let chipped = e >= 0 && e < chipBite ? 1.0 : 0.0
                body = mix(body, body * Vec3(1.10, 1.04, 0.96), chipped * 0.8)

                // ── the mortar ───────────────────────────────────────────────────────────
                let mt = Noise.fbmTiled(u, v, baseCells: mortarCells, octaves: 3, seed: p.seed ^ 0x2D)
                let sand = Noise.valueTiled(u, v, cells: fineCells, seed: p.seed ^ 0x5E)
                // 0 at the joint's centre line … 1 at the brick edge.
                let t = clamp01(c.rimMeters / Swift.max(jointHalfM, 1e-6))
                let cavity = 1 - 0.24 * smoothstep(0.40, 1.0, t)
                let mortar = p.jointColor * (0.90 + 0.16 * mt) * (0.94 + 0.12 * sand) * cavity

                ch.albedo[ch.idx(x, y)] = clampBand(mix(mortar, body, w))

                // ── roughness ────────────────────────────────────────────────────────────
                let brickRough = 0.86 + 0.08 * (mottle - 0.5) - 0.10 * Swift.min(1, flash)
                    + 0.05 * speck + 0.04 * (grain - 0.5) + 0.04 * chipped
                let mortarRough = 0.93 + 0.06 * (mt - 0.5) + 0.03 * (sand - 0.5)
                ch.roughness[ch.idx(x, y)] = clamp01(mix(mortarRough, brickRough, w))

                // ── height, in metres then normalised to `brickReliefMeters` ───────────
                let r = 0.0025                                 // arris roundover radius
                let ee = Swift.max(0, e)
                let round = ee < r ? r - (r * r - (r - ee) * (r - ee)).squareRoot() : 0
                // Every brick sits a little differently in its bed: proud or set back by a fraction
                // of a millimetre and tilted across its face — so under a raking sun neighbouring
                // bricks catch the light differently, the strongest tell that the wall was LAID.
                let r6 = Noise.unit(Noise.mix(h0 ^ 0xF5)), r7 = Noise.unit(Noise.mix(h0 ^ 0x17))
                let seat = 0.0006 * (r6 - 0.5)
                    + 0.0007 * (r7 - 0.5) * (c.fx - 0.5) * 2
                    + 0.0004 * (Noise.unit(Noise.mix(h0 ^ 0x29)) - 0.5) * (c.fy - 0.5) * 2
                // Sand-faced pits: small dents in the face, slightly darker (they hold shadow).
                let pit = Noise.voronoiTiled(u, v, cells: pitCells, jitter: 1, seed: p.seed ^ 0x3B)
                let pitDepth = Noise.unit(pit.cellId) < 0.35 ? (1 - smoothstep(0.12, 0.30, pit.f1)) : 0
                let faceZ = relief - 0.0012 - round - (chipped > 0 ? 0.5 * (chipBite - ee) : 0) + seat
                    + 0.0004 * (mottle - 0.5) + 0.00025 * (grain - 0.5) - 0.0005 * pitDepth
                let mortarZ = 0.0035 - 0.0012 * (1 - t * t) + 0.0003 * (mt - 0.5)
                ch.height[ch.idx(x, y)] = clamp01(mix(mortarZ, faceZ, w) / relief)
            }
        }
        ch.clearcoat = 0
        ch.diffuseRoughness = 1.0          // DH-1301 — OpenPBR's brick example
        // The height field is metric, so the slope is too: Δh over two texels × relief ÷ 2 texels.
        ch.deriveNormals(strength: relief / (2 * texelM))
        // Fine masonry tooth below the bake's resolution — sand in the face and the mortar.
        addMicroDetail(&ch, seed: p.seed ^ 0xB7, baseCells: 96, strength: 0.55)
        return ch
    }
}
