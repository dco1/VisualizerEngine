/// Per-scene hardware-RT opt-in for the global Illuminatorama overlay.
///
/// When a `SceneDescriptor` carries one of these, `AppModel` enables the
/// extracted-scene hardware-RT path (soft shadows + 1-bounce GI, optionally
/// glossy reflections) the moment that scene is shown through the overlay —
/// the *permanent* replacement for the headless-only `VIZ_ILLUMI_RT` env hook
/// (which still works as a dev override). `nil` on a descriptor = no RT.
///
/// ⚠️ **Only flag scenes with STABLE TOPOLOGY and CPU-READABLE geometry.**
/// Unlike the env hook (which built the acceleration structure once at snapshot
/// time), this enables RT in the live 60 Hz tick loop. A scene whose geometry
/// is GPU-fed (compute-written `MTLBuffer`-backed `SCNGeometry` — no readable
/// CPU triangles) or whose mesh set/topology changes every frame defeats the
/// TLAS *refit* optimisation: the RT path then does a full TLAS + per-mesh BLAS
/// **rebuild every frame** (each with a `waitUntilCompleted`), which **hangs the
/// main thread** and traces a faulted AS (→ magenta). `forest` is the worked
/// example of what NOT to flag (LeafField billboards + BarkRenderer + swaying
/// branches → per-frame rebuild → hang). Verify any newly-flagged scene
/// sustains RT in the interactive overlay before shipping the flag.
public struct IlluminatoramaRTOptions: Hashable, Sendable {
    /// Enable the extracted-scene RT pass (soft sun shadows + 1-bounce GI).
    public var enabled: Bool
    /// Also trace glossy reflection rays against the same acceleration
    /// structure. Costs extra rays; only meaningful when `enabled`.
    public var reflections: Bool
    /// Enable the Lumen-style on-surface radiance cache (multi-bounce GI) for
    /// this scene (Phase 4.38). The extractor auto-builds per-triangle cards
    /// from the scene's geometry. Routes the scene through the soup RT path the
    /// cache rides (so it's for light, mostly-static scenes); requires `enabled`.
    public var surfaceCache: Bool
    /// Desaturate the sky equirect when baking the diffuse irradiance cube
    /// (0 = no change, 1 = fully greyscale). Use for scenes whose sky is
    /// near-monochromatic (broilers, vivid sunsets) — the baked irradiance will
    /// be more neutral so albedo × irradiance doesn't flood everything with one
    /// hue. Visual sky is unchanged; only the baked IBL is affected.
    public var iblBakeDesaturation: Float
    /// Override the auto-exposure target EV for the overlay on this scene.
    /// `nil` keeps the renderer's global default (-4.0). Use for bright scenes
    /// (broilers, sunlit days) where the default under-exposes.
    public var autoExposureTargetEV: Float?
    /// When `false`, disable auto-exposure for this scene and use a fixed
    /// `exposure = 1.0` instead. Useful for scenes (broilers, extreme colour
    /// balance) where the log-luminance measurement gives a misleading reading
    /// and the renderer's default EMA over-darkens the image.
    public var autoExposureEnabled: Bool

    public init(enabled: Bool = true, reflections: Bool = false,
                surfaceCache: Bool = false, iblBakeDesaturation: Float = 0.0,
                autoExposureTargetEV: Float? = nil, autoExposureEnabled: Bool = true) {
        self.enabled = enabled
        self.reflections = reflections
        self.surfaceCache = surfaceCache
        self.iblBakeDesaturation = iblBakeDesaturation
        self.autoExposureTargetEV = autoExposureTargetEV
        self.autoExposureEnabled = autoExposureEnabled
    }
}

/// Which sidebar section a scene belongs to.
///
/// `main` is the curated headline list. `retired` collects classic base scenes
/// that have been superseded by a `+` / Ultra variant but are kept around for
/// reference. `test` collects developer/QA scenes (renderer labs, asset dumps).
/// The sidebar renders one section per case, in `allCases` order, under a
/// labelled divider — see `SceneListView`.
public enum SceneCategory: String, CaseIterable, Sendable, Hashable {
    case main
    case retired
    case test

    /// Section header shown above the group in the sidebar. `nil` for `main`,
    /// which renders without a divider as the headline list.
    public var sectionTitle: String? {
        switch self {
        case .main: return nil
        case .retired: return "Retired Scenes"
        case .test: return "Test Scenes"
        }
    }
}

/// Metadata + factory for one visualizer scene.
///
/// `SceneManifest.all` in the app target holds one descriptor per active
/// scene. To disable a broken scene without touching any other file, remove
/// its `import SceneXxx` line and its descriptor from that array.
public struct SceneDescriptor: Identifiable, Hashable {
    public let id: String
    public let displayName: String
    public let systemImage: String
    public let blurb: String
    /// Which sidebar section this scene lives under. Defaults to `.main`.
    public let category: SceneCategory
    /// Async factory that builds the controller on the main actor. Both sync
    /// `init()` controllers and `async make()` factory controllers satisfy
    /// this because an async context can await a synchronous call.
    public let make: @MainActor () async -> any SceneController
    /// Per-scene hardware-RT opt-in for the global overlay (`nil` = off).
    /// Replaces the `VIZ_ILLUMI_RT` env hook for interactive use.
    public let illuminatoramaRT: IlluminatoramaRTOptions?

    public init(
        id: String,
        displayName: String,
        systemImage: String,
        blurb: String,
        category: SceneCategory = .main,
        illuminatoramaRT: IlluminatoramaRTOptions? = nil,
        make: @escaping @MainActor () async -> any SceneController
    ) {
        self.id = id
        self.displayName = displayName
        self.systemImage = systemImage
        self.blurb = blurb
        self.category = category
        self.illuminatoramaRT = illuminatoramaRT
        self.make = make
    }

    // Equality and hashing are keyed on `id` (a String) — no actor isolation
    // needed for these conformances.
    public static func == (lhs: SceneDescriptor, rhs: SceneDescriptor) -> Bool {
        lhs.id == rhs.id
    }
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
