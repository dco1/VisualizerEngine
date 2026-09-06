import Metal
import simd

/// A simulated grass field: XPBD blade chains, GPU ribbon expansion, and the per-frame step —
/// built, registered and driven from one place.
///
/// The solver (`PBDFieldSolver`) and the expander (`GrassRibbonRenderer`) have been shared engine
/// code for a long time. The *glue* around them was not: constructing the solver, configuring the
/// chains, wiring the ribbon, registering the GPU mesh, tuning nine wind and spring constants,
/// priming the field so the first frame shows real blades rather than a flat mat, and encoding two
/// command buffers per frame — all of that was written out longhand in each host's scene
/// controller, twice, in two apps, from the same original. That is the gap
/// `docs/VEGETATION_SHARED_ARCHITECTURE.md` named: *"pull the build/step glue down beside its
/// already-shared solver."*
///
/// ## What the host still owns
///
/// **The scatter.** Where blades go is a question about the document — the lot's shape, the house
/// footprint, paving, decks, a driveway, canopy thinning — and reading those needs the host's own
/// types. So this takes finished `restPositions` and an optional pre-built colour buffer. That is
/// deliberately the low-risk seam: everything downstream of "here are the blades" is identical
/// between hosts, and everything upstream is not.
@MainActor
public final class GrassRenderSet: VegetationRenderSet {

    /// Solver constants. The defaults are the values Daydream Home's lawn shipped with; a host that
    /// has tuned its own passes them in rather than mutating the solver afterwards, so there is one
    /// place to read what a field is set to.
    public struct Tuning: Sendable {
        public var gravity: Float = 0
        public var damping: Float = 0.985
        /// Peak wind acceleration, m/s². Constant unless `windResponse` is set — see `step`.
        public var windAmp: Float = 1.4
        public var windFreq: Float = 0.35
        public var windScroll: Float = 0.7
        public var windDir: SIMD2<Float> = SIMD2(0.8, 0.6)
        public var springK: Float = 9.0
        public var springDamp: Float = 0.02
        public var constraintIterations: Int = 4
        public var bendStiffness: Float = 0.5

        public init() {}
    }

    /// How a field converts a `VegetationWind` into solver amplitude, if at all.
    ///
    /// **Nil means the field ignores the wind argument entirely** and keeps the constant breeze in
    /// `Tuning`. That is not a placeholder — it is what Daydream Home's lawn does today, and making
    /// wind live by default would have silently changed how every lawn moves the moment this glue
    /// moved into the engine. A host that wants wind-reactive grass opts in.
    public struct WindResponse: Sendable {
        /// Solver `windAmp` per m/s of wind speed.
        public var ampPerMetrePerSecond: Float
        public init(ampPerMetrePerSecond: Float) { self.ampPerMetrePerSecond = ampPerMetrePerSecond }
    }

    // MARK: - Public state

    /// The registered GPU mesh. The host builds its own draw instance against this handle's kind —
    /// instance albedo and roughness are material decisions, not field decisions.
    public let meshHandle: IlluminatoramaMeshHandle
    public let bladeCount: Int
    /// Set to make the field respond to `step`'s wind argument. Nil keeps the constant breeze.
    public var windResponse: WindResponse?

    /// The field's XPBD solver. Readable because a host's own gates reach for it — Daydream
    /// Home's yard-wind A/B zeroes `windAmp` to render a still arm, and its leak gate holds a
    /// `weak` reference to prove a rebuild released the previous field. *Tuning* still belongs in
    /// `Tuning` / `retune` so there is one place to read what a field is set to.
    public let solver: PBDFieldSolver
    /// The GPU ribbon expander, readable for the same reason as `solver`.
    public let ribbon: GrassRibbonRenderer

    /// The vertex layout a field of this shape will have — two ribbon vertices per particle,
    /// which is `GrassRibbonRenderer`'s layout.
    ///
    /// A host that builds its own per-vertex colour buffer needs the layout *before* the set
    /// exists, because the ribbon that reports it is constructed inside `init`. Same numbers as
    /// `ribbon.vertexLayout` on the built field.
    public static func vertexLayout(bladeCount: Int, particlesPerChain: Int)
        -> (vertexCount: Int, bladeCount: Int, particlesPerBlade: Int) {
        (vertexCount: bladeCount * particlesPerChain * 2,
         bladeCount: bladeCount,
         particlesPerBlade: particlesPerChain)
    }

    // MARK: - Private

    private let engine: SimEngine
    private let colorBuffer: MTLBuffer?
    private weak var renderer: IlluminatoramaRenderer?
    private var tuning: Tuning
    private var isTornDown = false

    // MARK: - Build

    /// Build the field, register it, and prime it.
    ///
    /// Returns nil if the solver, the ribbon or the mesh registration fails — a host treats that as
    /// "no grass this build" rather than a crash, because it is usually a transient allocation
    /// failure on a very large lot.
    ///
    /// - Parameters:
    ///   - restPositions: the blade chains' rest pose, `bladeCount × particlesPerChain` entries.
    ///   - bladeWidthScales: optional per-blade width multiplier. A flower head is bulkier than a
    ///     grass blade and vanishes at yard distance without it.
    ///   - colorBuffer: optional per-vertex colour. Its alpha is the foliage thin-sheet flag, so a
    ///     field built without one is opaque and will not take the backlight.
    ///   - primeFrames: how many steps to run before the first real frame. Zero shows a flat
    ///     unbent mat on frame one, which reads as a bug; the shipped value is 8.
    public init?(renderer: IlluminatoramaRenderer,
                 engine: SimEngine = .shared,
                 bladeCount: Int,
                 particlesPerChain: Int,
                 restPositions: [SIMD3<Float>],
                 baseHalfWidth: Float,
                 tipHalfWidthFrac: Float,
                 bladeWidthScales: [Float]? = nil,
                 colorBuffer: MTLBuffer? = nil,
                 tuning: Tuning = Tuning(),
                 primeFrames: Int = 8) {
        guard bladeCount > 0,
              let solver = PBDFieldSolver(engine: engine,
                                          maxChains: bladeCount,
                                          particlesPerChain: particlesPerChain)
        else { return nil }

        solver.configureChains(chainCount: bladeCount,
                               particlesPerChain: particlesPerChain,
                               restPositions: restPositions,
                               bendStiffness: tuning.bendStiffness)

        guard let ribbon = GrassRibbonRenderer(solver: solver,
                                               baseHalfWidth: baseHalfWidth,
                                               tipHalfWidthFrac: tipHalfWidthFrac)
        else { return nil }

        if let scales = bladeWidthScales { ribbon.setBladeWidthScales(scales) }

        guard let handle = renderer.registerGPUMesh(
            ribbon.illuminatoramaDescriptor(colorBuffer: colorBuffer))
        else { return nil }

        self.engine = engine
        self.solver = solver
        self.ribbon = ribbon
        self.colorBuffer = colorBuffer
        self.renderer = renderer
        self.meshHandle = handle
        self.bladeCount = bladeCount
        self.tuning = tuning

        apply(tuning)

        // Prime so the first presented frame shows bent, settled blades.
        ribbon.cameraPosition = renderer.camera.position
        for _ in 0 ..< max(0, primeFrames) {
            guard let cb = engine.commandQueue.makeCommandBuffer() else { break }
            cb.label = "Vegetation.grass.prime"
            solver.encode(to: cb, wallDt: 1.0 / 60.0)
            ribbon.encodeExpand(to: cb)
            cb.commit()
        }
    }

    private func apply(_ t: Tuning) {
        solver.gravity = t.gravity
        solver.damping = t.damping
        solver.windAmp = t.windAmp
        solver.windFreq = t.windFreq
        solver.windScroll = t.windScroll
        solver.windDir = t.windDir
        solver.springK = t.springK
        solver.springDamp = t.springDamp
        solver.constraintIterations = t.constraintIterations
    }

    /// Re-tune a live field without rebuilding it — a settings slider, not a document edit.
    public func retune(_ t: Tuning) {
        tuning = t
        apply(t)
    }

    // MARK: - VegetationRenderSet

    /// Advance the field one frame: one solver substep and one ribbon expansion, on the sim queue.
    ///
    /// `camera` matters — the ribbon orients each blade's quad toward the viewer, so a field stepped
    /// with a stale camera turns edge-on as you orbit and the lawn appears to thin out.
    public func step(dt: Float, wind: VegetationWind, camera: SIMD3<Float>) {
        guard !isTornDown else { return }
        if let response = windResponse {
            solver.windAmp = wind.enabled ? wind.speed * response.ampPerMetrePerSecond : 0
            solver.windDir = wind.direction
        }
        guard let cb = engine.commandQueue.makeCommandBuffer() else { return }
        cb.label = "Vegetation.grass.step"
        ribbon.cameraPosition = camera
        solver.encode(to: cb, wallDt: max(0, dt))
        ribbon.encodeExpand(to: cb)
        cb.commit()
    }

    public func teardown() {
        guard !isTornDown else { return }
        isTornDown = true
        renderer?.removeMesh(meshHandle.kind)
        renderer = nil
    }
}
