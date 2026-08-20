import Foundation

/// **How big one texture repeat is on a surface, and whether the material gets a say.**
///
/// This replaces a bare `uvScale: Double` that had been quietly meaning two different things.
/// A surface's UVs are baked into its mesh at conversion time (`HouseSceneBuilder.convert`
/// divides world position by the scale), so:
///
///  - On the **structural** path — walls, room floors, room ceilings — the mesh is converted
///    *with the resolved material in hand*, so the material can choose the period. A tile can
///    therefore be exactly the size the user typed.
///  - On every **instanced placeable** — millwork, furniture, stairs, decks, frames — the mesh
///    was registered once, earlier, against a dummy material at a period of 1 m. Whatever scale
///    the emit site passes is *inert*: it changes the cache key and the bake, but not one UV.
///    A material that assumes it can set the period there is writing a control that does nothing.
///
/// Conflating those is why a tile on a kitchen counter renders ~1.5× too small today: the emit
/// site declares 1.5 m, the bake believes it, and the mesh renders 1 m. Carrying `isAdjustable`
/// alongside the number makes the distinction impossible to forget — a material can ask, and
/// gets told the truth.
public struct SurfaceTiling: Equatable, Hashable, Sendable {
    /// World metres spanned by one texture repeat.
    public var metresPerRepeat: Double
    /// Whether the consuming mesh takes its UV period from the resolved material (true), or
    /// baked it earlier and will ignore anything we choose (false).
    public var isAdjustable: Bool

    public init(_ metresPerRepeat: Double, isAdjustable: Bool) {
        self.metresPerRepeat = Swift.max(1e-3, metresPerRepeat)
        self.isAdjustable = isAdjustable
    }

    /// **The mesh is converted with this material in hand**, so this number IS the period its UVs
    /// bake at — and a size-carrying material may choose it. The structural wall/floor/ceiling/site
    /// path and the per-sub-part vanity path are the surfaces that work this way.
    public static func baked(_ metresPerRepeat: Double) -> SurfaceTiling {
        SurfaceTiling(metresPerRepeat, isAdjustable: true)
    }

    /// **The mesh baked its UVs earlier**, against a dummy material at `metresPerRepeat`. Nothing
    /// this material chooses can change them, so a size-carrying material must quantise to fit
    /// rather than render a size it was never going to get.
    public static func prebaked(_ metresPerRepeat: Double) -> SurfaceTiling {
        SurfaceTiling(metresPerRepeat, isAdjustable: false)
    }

    /// The period a bare library bake / a picker swatch is quoted at. Adjustable, because a
    /// swatch is rendered from the channels alone — nothing has pre-baked UVs to disagree with.
    /// **A swatch must resolve at the same period as the surface it previews**, or the picker
    /// shows a different tile from the one you get (it did: the well baked at 1.0 m while a wall
    /// baked at 2.0 m).
    public static let librarySwatch = SurfaceTiling(2.0, isAdjustable: true)
}
