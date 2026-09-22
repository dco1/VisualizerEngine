import Foundation

/// The moulded profile of a GLASS BLOCK — the ONE description of the block module and its
/// surface shape in the codebase.
///
/// **Why this exists.** A glass block is used in two places that look like two features:
///
///  * as **unit masonry** — a laid-up partition/backsplash, drawn as an opaque wall finish
///    (`MaterialRegistry` id `glass-block`, category `.brick`); and
///  * as **glazing** — real blocks laid up inside a window opening, which must actually
///    TRANSMIT light (`GlassMaterial.block`, registry id `glass-block-glazing`,
///    category `.glass`).
///
/// Both are the same physical object, so both read their module and their face shape from
/// here. `MaterialGenerator.glassBlock` / `.glassBlockGlazing` bake their swatches from this
/// profile, and `HouseMesh.glassBlockPane` builds the real pane geometry from it — nothing
/// downstream restates a dimension.
///
/// **The glazing pattern is SHAPE, not paint.** A glazing draw group never gets an albedo
/// slice: the bridge routes it into the engine's dielectric pass, where the only thing that
/// survives is the mesh's own surface. So a glass-block *window* has to carry its pattern as
/// real geometry — a hollow block per module (two thin shells, pillowed outside, rippled
/// inside) set in an OPAQUE mortar lattice that the bridge draws as ordinary structure —
/// exactly as fluted glass carries its reeds (`FlutedGlassProfile`). Nothing here raises the
/// glass's roughness: a rough dielectric is a per-frame re-seeded stochastic ray estimator,
/// which is the "live noise" that modelling obscuring glass as shape exists to avoid.
public enum GlassBlockProfile {

    // MARK: – the masonry module (shared by the wall finish and the pane)

    /// Nominal glass-block face: the standard 240 × 240 × 80 mm block.
    ///
    /// 240 (not the other standard, 190) is chosen because its 250 mm module divides the 2 m
    /// wall tile a whole 8 times AND divides the 512² bake exactly (64 px per block), so the
    /// tile wrap lands dead on a mortar joint. At 10 modules the wrap fell 0.8 px off-joint
    /// and the seam score tripled (0.14 → 2.03) — a real, measurable repeat edge.
    public static let faceMeters = 0.240
    /// The mortar joint between two blocks (10 mm is the standard bed).
    public static let mortarMeters = 0.010
    /// Centre-to-centre spacing of the laid-up grid — DERIVED, never typed.
    public static var moduleMeters: Double { faceMeters + mortarMeters }
    /// Block depth through the wall — the third dimension of the 240 × 240 × 80 block. This
    /// is the pane thickness a glass-block window is built at (a glass block is 4× a window
    /// pane, which is exactly why it reads as masonry rather than glazing).
    public static let depthMeters = 0.080

    /// How much real wall ONE baked tile spans. This is not a free choice: a wall finish is
    /// sampled at the app's `ElementUVScale.roomSurface`, so the tile must be authored for
    /// exactly that span or the blocks render at the wrong SIZE — which is what the first
    /// real-Metal capture showed (4 blocks per tile put ~460 mm "glass blocks" on the wall,
    /// 2.3× life size). `RenderBridgeMaterialTests` pins the two together so they can't drift.
    public static let wallTileMeters = 2.0
    /// Blocks across one baked tile — DERIVED, so the module stays life-size on the wall.
    public static var perTile: Int { Int((wallTileMeters / moduleMeters).rounded()) }
    /// The real span one baked tile covers at the derived block count (== `wallTileMeters`).
    public static var tileMeters: Double { moduleMeters * Double(perTile) }

    /// Half the mortar joint, as a fraction of one module — the shared coordinate the bake
    /// and the pane both use to place the joint.
    public static var jointHalfFraction: Double { (mortarMeters / moduleMeters) / 2 }

    /// Moulded waves across one block face — the wall bake's normal-map flutes AND the pane's
    /// inside-face ripple (`ripple(_:waves:)`), so the finish and the window are one block.
    public static let flutesPerBlock = 7.0

    // MARK: – the pane's moulded face

    /// Peak relief of a block face above the mortar valley, per side, in metres. A real block
    /// face is proud of its bed joint by a few millimetres and chamfered back into it; 7 mm
    /// gives a lens strong enough to bend the scene behind it perceptibly at room distance.
    public static let pillowMeters = 0.007
    /// Width of the chamfer that runs the face back down into the joint, as a fraction of the
    /// module. Together with `jointHalfFraction` this is the whole cross-section of a joint:
    /// flat valley out to `jointHalfFraction`, then a shoulder, then the face plateau.
    public static let shoulderFraction = 0.16
    // MARK: – the pane: a HOLLOW block in a mortar lattice

    /// Whole blocks across a `span` metres of opening. ROUNDED, and the caller stretches the
    /// module to fit the real span — so a window never ends on a sliced-off partial block,
    /// which is also how glass block is really laid (the opening is sized to the module).
    @inlinable public static func blockCount(forSpan span: Double) -> Int {
        max(1, Int((span / moduleMeters).rounded()))
    }

    /// Wall thickness of each pressed face shell. A glass block is two pressed halves fused
    /// round a sealed air cavity — NOT 80 mm of solid glass. The difference is the whole look:
    /// light crossing two thin moulded skins is warped and softened, while 80 mm of solid glass
    /// at any angle shifts the scene sideways by centimetres and traps rays by internal
    /// reflection, so each block showed its own displaced, hard-edged chunk of the world
    /// behind it (DH-0713's "mosaic").
    public static let shellMeters = 0.009
    /// Peak-to-trough depth of the moulded ripple on the INSIDE face of each shell. The ripple
    /// is what makes a block obscure the view (streaks of colour, no detail) — it lives inside,
    /// where a real block's pattern is pressed, so the outer faces stay smooth and glossy. The
    /// two shells ripple at right angles (vertical reeds front, horizontal back), which is the
    /// classic cross-hatched "wave" block.
    public static let rippleMeters = 0.0012
    /// Samples across a block face along the axis that carries no ripple — the pillow dome is
    /// gentle and smooth-normalled, so a handful is enough.
    public static let pillowSegmentsPerBlock = 6
    /// How far the mortar's face sits back from the blocks' edges — a tooled joint. The joint
    /// is OPAQUE and it is the strongest glass-block cue there is: without it the grid never
    /// reads, because the glass pass draws no colour of its own.
    public static let mortarRecessMeters = 0.003
    /// Air left between a block's glass edge and the mortar around it, so the two never share
    /// a coplanar face (which would z-fight in raster and make RT hits order-dependent).
    public static let edgeClearanceMeters = 0.0005
    /// Hard ceiling on one pane's GLASS triangles, so a wall-sized glass-block window can never
    /// blow the structural triangle budget: the pane builder steps its pressing's detail down
    /// until it fits (DaydreamCore `GlassBlockStyle.detailLevels`). Deterministic — a pure
    /// function of the block count and the style.
    public static let maxPaneTriangles = 24_000

    /// The inside-face ripple at `u` ∈ [0, 1] across a face, as a 0…1 fraction of
    /// `rippleMeters` (0 at both edges, so every shell's rim is a plain rectangle).
    @inlinable public static func ripple(_ u: Double, waves: Int) -> Double {
        guard waves > 0 else { return 0 }
        return 0.5 - 0.5 * cos(2 * Double.pi * Double(waves) * u)
    }

    /// The block-face relief at block-local coordinates `(fx, fy)` ∈ [0,1)², as a 0…1
    /// fraction of `pillowMeters`. ZERO from the joint centre-line out to the face edge
    /// (`jointHalfFraction`), so a block's outer face comes back to its shell's rim exactly;
    /// 1 at the crown of the face.
    ///
    /// Two terms, each doing one job: the **plateau** (a chamfered flat face, which is what
    /// makes the joint read as a crisp grid line) and the **pillow** (a shallow dome over it,
    /// which is what makes each block act as a lens and bend the room behind it).
    @inlinable public static func relief(_ fx: Double, _ fy: Double) -> Double {
        let j = jointHalfFraction, s = shoulderFraction
        func plateau(_ t: Double) -> Double {
            smoothstep(j, j + s, min(t, 1 - t))
        }
        let flat = plateau(fx) * plateau(fy)
        let dome = sin(Double.pi * fx) * sin(Double.pi * fy)
        return flat * (0.55 + 0.45 * dome)
    }

    /// Per-block amplitude jitter (0.92…1.08) from ONE hash of the block index — the same
    /// mechanism the wall bake uses to vary flute direction, so no two neighbouring blocks
    /// are clones. Because `relief` is zero at every joint, varying the amplitude per block
    /// cannot open a crack between them.
    @inlinable public static func amplitudeScale(block bi: Int, _ bj: Int,
                                                 seed: UInt64 = 331) -> Double {
        0.92 + 0.16 * Noise.unit(Noise.hash2(bi, bj, seed ^ 0x51ED_2701))
    }
}
