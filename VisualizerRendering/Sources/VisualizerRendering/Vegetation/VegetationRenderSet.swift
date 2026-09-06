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

/// Holds the live vegetation systems and fans one `step` out to all of them.
///
/// **Honest note on its current weight:** Daydream Home has exactly one system that steps — grass.
/// Its trees are static CPU bakes with no per-frame work at all, and Visualizer's Forest superseded
/// the `LeafField` path the original architecture note assumed both apps drove. So today this
/// umbrella fans out to one member, and on its own that would be a single-consumer abstraction not
/// worth having.
///
/// It earns its place on the *other* side: it is what makes adding the second system a change to
/// nothing but the system itself. A host's frame loop calls `step` once and never grows a second
/// `if solver != nil` branch — which is the shape the per-system glue kept re-growing in both apps.
/// Keep it thin; if it ever needs to know what KIND of system it holds, that is the tell that
/// something belongs on the protocol instead.
@MainActor
public final class VegetationScene {

    public private(set) var sets: [any VegetationRenderSet] = []

    public init() {}

    public func add(_ set: any VegetationRenderSet) { sets.append(set) }

    /// Remove one system and tear it down. Identity comparison — a host holds its own reference.
    public func remove(_ set: any VegetationRenderSet) {
        sets.removeAll { $0 === set }
        set.teardown()
    }

    /// Advance every live system.
    public func step(dt: Float, wind: VegetationWind, camera: SIMD3<Float>) {
        for set in sets { set.step(dt: dt, wind: wind, camera: camera) }
    }

    /// Tear every system down and empty the scene. For a document swap.
    public func teardown() {
        for set in sets { set.teardown() }
        sets.removeAll()
    }
}
