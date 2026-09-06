import simd

/// Painting a plant's foliage colour and its backlight flag onto baked vertices.
///
/// Every plant a host renders arrives from its mesh generator with no colour: the CPU geometry
/// layer (`VisualizerVegetation`) emits shape, and the leaf green is a *render* decision — it
/// depends on the species' material, a per-instance shade variation, and whether this particular
/// plant is lit from behind. So each host stamped it on itself, and the stamp was rewritten once
/// per plant kind: four near-identical copies in Daydream Home alone, each carrying its own
/// re-declared SplitMix64 to derive the same ±7% jitter, differing only in which alpha they wrote.
///
/// That divergence is the whole hazard. The alpha is not decoration — it is the **thin-sheet
/// subsurface flag** the G-buffer reads, and getting it wrong is visible: an interior potted plant
/// stamped translucent out-glows the room it stands in under a low sun. Four hand-written copies of
/// a rule that subtle is four chances to write the wrong constant. Here it is a named case.
public enum FoliageVertexStamp {

    /// What the vertex-colour alpha means to the shading pass.
    ///
    /// This is a real product decision per plant, not a default to be normalized away — see the
    /// two cases. Naming it forces a host to choose rather than copy whichever literal was nearby.
    public enum Backlight: Sendable {
        /// `alpha = 0` — the blade is a thin sheet and takes the backlight: light passing through
        /// it from behind scatters forward and the leaf glows. Right for anything outdoors, where
        /// the sun can get behind the foliage — grass, trees, garden plants, planter beds.
        case translucent
        /// `alpha = 1` — opaque; the blade is lit only from the front. Right for an INTERIOR plant,
        /// which has no sun behind it and which, stamped translucent, out-glows the room under a
        /// low sun.
        case opaque

        @inlinable public var alpha: Float {
            switch self {
            case .translucent: return 0
            case .opaque:      return 1
            }
        }
    }

    /// A deterministic per-plant shade offset in ±`spread`, derived from the plant's own seed.
    ///
    /// Two plants of the same species standing side by side reading as the exact same green is one
    /// of the louder "this is a copy-paste" tells, so every foliage stamp shifts its green a little.
    /// Derived here rather than by the caller so the *same* seed keeps producing the *same* shade
    /// after this moved out of the app — SplitMix64 is spelled out below for that reason, and must
    /// not be swapped for another generator.
    ///
    /// Note this deliberately does NOT use `hashValue`: Swift re-seeds hashing every process, so a
    /// hash-derived shade would reshuffle at every launch.
    public static func shadeJitter(seed: UInt64, spread: Double = 0.07) -> Float {
        // SplitMix64, one draw, reproducing the app-side generator EXACTLY — including the
        // `seed * 0xD1B5… + 1` pre-mix that was spelled at the call sites, the zero-guard, and
        // the golden-ratio advance BEFORE the mix rather than after. Every one of those changes
        // the stream, and the stream is what fixes each plant's shade; drift here re-shades every
        // plant in every document.
        var state = seed &* 0xD1B5_4A32_D192_ED03 &+ 1
        if state == 0 { state = 0x9E37_79B9_7F4A_7C15 }
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        let unit = Double(z >> 11) * (1.0 / 9_007_199_254_740_992.0)   // [0, 1)
        // `spread` is DOUBLE deliberately. Typed `Float`, `0.07` widens to 0.07000000029802322 and
        // the result lands one Float ULP off the app's `unit * 0.14 - 0.07` for ~5% of seeds —
        // invisible (7e-09 on a [0,1] channel) but enough to make "bit-exact" a false claim, and a
        // false claim is what stops the next person checking. Doubling is exact in binary, so
        // `unit * spread * 2 - spread` and `unit * 0.14 - 0.07` agree to the last bit.
        return Float(unit * spread * 2 - spread)
    }

    /// Stamp one flat colour + backlight flag over every vertex of a baked foliage group.
    ///
    /// `jitter` is added to each channel and clamped at zero — pass `shadeJitter(seed:)` for the
    /// per-plant variation, or `0` for a colour that is itself the point (a bloom's petal, whose
    /// vivid hue should not be shifted toward its neighbours').
    public static func stamp(_ vertices: inout [IlluminatoramaVertex],
                             color: SIMD3<Float>,
                             backlight: Backlight,
                             jitter: Float = 0) {
        let c = SIMD4<Float>(max(0, color.x + jitter),
                             max(0, color.y + jitter),
                             max(0, color.z + jitter),
                             backlight.alpha)
        for i in vertices.indices { vertices[i].color = c }
    }
}
