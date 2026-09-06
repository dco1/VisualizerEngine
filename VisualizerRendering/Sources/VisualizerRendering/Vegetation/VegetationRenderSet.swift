import simd

/// The wind a vegetation system is asked to respond to.
///
/// One value rather than three loose parameters because every system that moves reads the same
/// three things, and the two hosts had already drifted on what "wind off" means — one zeroed the
/// amplitude, the other stopped stepping entirely, which are different (a stopped solver holds its
/// last bend; a zero-amplitude one relaxes upright).
public struct VegetationWind: Sendable, Equatable {
    /// Wind speed in m/s. A system scales its own response from this — a grass blade and a tree
    /// crown do not move by the same factor, and that factor is the system's business, not the
    /// caller's.
    public var speed: Float
    /// Compass heading the wind blows toward, in radians.
    public var headingRadians: Float
    /// When false the system relaxes to rest rather than freezing — see the note above.
    public var enabled: Bool

    public init(speed: Float, headingRadians: Float, enabled: Bool = true) {
        self.speed = speed
        self.headingRadians = headingRadians
        self.enabled = enabled
    }

    /// Still air.
    public static let calm = VegetationWind(speed: 0, headingRadians: 0, enabled: false)

    /// The wind direction as a ground-plane unit vector.
    public var direction: SIMD2<Float> { SIMD2(cos(headingRadians), sin(headingRadians)) }
}

/// One live vegetation system — grass, a canopy, a terrain cover — that owns its whole render
/// lifecycle: it builds and registers itself at construction, advances itself each frame, and
/// releases its GPU resources on teardown.
///
/// The point is what is NOT here. Building the geometry, registering the mesh with the renderer,
/// and encoding the per-frame solver work are all things a host used to write itself, once per
/// system, in its scene controller — and every host wrote them slightly differently. Registration
/// in particular is public engine API, so there was never a reason for it to be app code.
///
/// What stays with the host is the part that genuinely cannot move: which plants exist and where
/// (that is document state), the UI that edits them, and the two trigger calls — "the model
/// changed, rebuild" and "a frame happened, step".
///
/// `@MainActor` because the solvers and renderers it drives are: `PBDFieldSolver`,
/// `GrassRibbonRenderer` and `IlluminatoramaRenderer` are all main-actor types, and a render set
/// exists to own exactly those. This is a statement of where the work already happens, not a new
/// constraint.
@MainActor
public protocol VegetationRenderSet: AnyObject {
    /// Advance one frame. `dt` is wall-clock seconds since the previous step, already clamped by
    /// the caller against a stall.
    func step(dt: Float, wind: VegetationWind, camera: SIMD3<Float>)

    /// Release GPU resources and unregister from the renderer.
    func teardown()
}

// MARK: - On the absence of a `VegetationScene` umbrella
//
// The 2026-06 architecture note (docs/VEGETATION_SHARED_ARCHITECTURE.md) specified an umbrella
// holding `[VegetationRenderSet]` and fanning one `step(dt:wind:camera:)` out to all of them, so a
// host's per-frame residue collapses to one call. It is deliberately NOT here, and this note is so
// nobody adds it back without the second system that would justify it.
//
// **There is only one system that steps.** Daydream Home's trees are static CPU bakes with no
// per-frame work at all, and Visualizer's Forest superseded the `LeafField` path that note assumed
// both apps drove — so the umbrella would fan out to exactly one member. TREES_TECH_SCOPE §4a warns
// against precisely that ("without inventing a single-consumer abstraction"), and it was written
// about this very lift.
//
// **And it would not be free.** A scene holding its sets strongly makes a SECOND owner of a live
// GPU field, and `HouseRenderBridgeGPUTests_Hero.testWhoRetainsThePreviousGrassField` exists
// because stranding the previous field on rebuild is a real, measured failure mode here. Paying
// that risk to wrap a one-element array is a bad trade today.
//
// Add it when a second system actually steps — a swaying canopy, a terrain cover. At that point the
// host's frame loop is about to grow its second `if solver != nil` branch, which is the moment the
// umbrella starts earning its keep.
