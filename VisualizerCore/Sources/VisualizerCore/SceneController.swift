import Metal
import MetalKit
import SceneKit
import SwiftUI
import simd

/// The contract every scene's controller must satisfy.
///
/// `AppModel` stores a `[String: any SceneController]` dictionary keyed by
/// scene ID instead of 21 typed optional properties, so adding or removing a
/// scene requires only a one-line change in `SceneManifest.swift`. If a scene
/// package fails to compile, remove its `import` + descriptor from
/// `SceneManifest.swift` and the rest of the app builds unaffected.
///
/// Default implementations are provided for optional capabilities: `technique`
/// (nil), `sceneTimeline` (nil), `settingsExportText` (""). Conforming types
/// only override what they actually use.
@MainActor
public protocol SceneController: AnyObject {
    // ── Rendering ────────────────────────────────────────────────────
    var scene: SCNScene { get }
    /// The camera node to use as `SCNView.pointOfView`.
    var cameraNode: SCNNode { get }
    /// An optional `SCNTechnique` post-effect to install on the `SCNView`.
    /// Defaults to `nil`; only pizza-plus-tier scenes override this.
    var technique: SCNTechnique? { get }
    /// When non-nil, `ContentView` hosts this view directly instead of an
    /// `SCNView`, eliminating the redundant SceneKit render pass. Scenes that
    /// own their Metal pipeline entirely (native Illuminatorama scenes) create
    /// and manage the view; the host only embeds it in the SwiftUI hierarchy.
    /// Defaults to `nil` — SCNView path is used.
    var metalView: MTKView? { get }

    // ── Timeline ─────────────────────────────────────────────────────
    /// The scene's keyframe timeline, if it has one. Named `sceneTimeline`
    /// rather than `timeline` to avoid colliding with controllers' stored
    /// `let timeline: SceneTimeline` property (stored vs. optional types
    /// can't share a name in Swift).
    var sceneTimeline: SceneTimeline? { get }

    // ── Pause ────────────────────────────────────────────────────────
    /// Whether the scene's tick loop is paused. Routed through `AppModel`
    /// so the toolbar Play/Pause button has a single binding to write to.
    var isPaused: Bool { get set }

    // ── Settings export ──────────────────────────────────────────────
    var settingsExportText: String { get }

    // ── Lifecycle ────────────────────────────────────────────────────
    /// Activate or deactivate the scene's tick loop. Called by `AppModel`
    /// whenever the active selection changes.
    func setActive(_ active: Bool)

    /// Apply perf-profiling overrides before a measurement run. Each entry is
    /// a settings key → string value, sourced from `VIZ_PERF_<key>=<value>`
    /// env vars set by `Scripts/perf-profile.sh`. Scenes that support profiling
    /// parse the keys they recognise and ignore unknown ones. Default is no-op,
    /// which is correct for scenes that haven't added profiling support yet.
    func applyPerfOverrides(_ overrides: [String: String])

    // ── Headless multi-angle capture ─────────────────────────────────
    /// For scenes that render through their OWN off-SceneKit camera (native
    /// Illuminatorama scenes paint `scene.background.contents = renderer
    /// .outputTexture`, so changing the SCN `pointOfView` is a no-op), re-aim
    /// that camera to a review angle, render a fully-accumulated frame, and
    /// return the freshly-rendered output texture.
    ///
    /// `yawDeg` is an azimuth OFFSET from the scene's hero framing; `pitchDeg`
    /// is the ABSOLUTE elevation (0 = ground-grazing, 85 = top-down). The
    /// implementation must stop fighting its own tick loop (the harness calls
    /// `setActive(false)` first), pump enough blocking frames to converge TAA /
    /// RT temporal history after the camera cut, and restore its hero camera
    /// before returning. Returns `nil` (default) for scenes that don't own an
    /// off-SceneKit camera — the harness then falls back to the SCN sweep.
    func captureReviewAngle(yawDeg: Float, pitchDeg: Float,
                            size: SIMD2<Int>) -> MTLTexture?

    /// For native off-SceneKit renderers (Illuminatorama): settle the HERO frame
    /// at the controller's CURRENT live framing (no camera re-aim) by running the
    /// same TAA flush + accumulate the review angles use, then return the converged
    /// output texture. The headless hero path otherwise reads the live texture mid-
    /// accumulation, leaving warm reprojection smear on grazing-angle foreground.
    /// Returns `nil` (default) for scenes that don't own an off-SceneKit camera.
    func captureHeroSettled(size: SIMD2<Int>) -> MTLTexture?

    // ── Penetration probing ──────────────────────────────────────────
    /// The scene's GPU/solver-level interpenetration probers, if any. A scene
    /// whose bodies live in MTLBuffers (CoinDEM piles, PBD chains) returns its
    /// solver(s) here so `PenetrationProbe` can measure overlap from the real
    /// simulation state — the code answer to "these objects are interpenetrating,"
    /// covering the Metal-simulated scenes the SceneKit `contactTest` path can't
    /// see. Defaults to `[]`; SceneKit-physics and hand-placed / Illuminatorama
    /// scenes are measured by the probe's own physics-body / geometry-OBB passes.
    func penetrationProbers() -> [any PenetrationProbing]

    /// The scene's curve/rail centerlines, densely sampled in world space, for
    /// the spline-continuity check. A rail-network scene (Eggs) returns its
    /// tracks here so kinks are caught as an exact location. Defaults to `[]`.
    func splinePolylines() -> [SplinePolyline]

    // ── FPS reporting ────────────────────────────────────────────────
    /// Set by `AppModel` after load. The controller calls it every ~0.5 s
    /// with its measured tick rate (Hz) so the toolbar can display the real
    /// frame rate for scenes whose rendering bypasses the SceneKit render
    /// thread. SCNView-based scenes leave this at the default (no-op) and
    /// report FPS via the SCNView render-delegate path instead.
    var fpsReporter: (@MainActor (Double) -> Void)? { get set }

    // ── UI factory ───────────────────────────────────────────────────
    /// Build the settings panel for this scene. The controller captures
    /// `self` (its concrete type) in the closure so `@Bindable` still works
    /// in the returned view — no type erasure is needed inside the view.
    /// `blurb` is the scene's one-paragraph description from `SceneDescriptor`,
    /// injected here so the settings view can display it without depending on
    /// the app's `SceneManifest`.
    func makeSettingsView(blurb: String) -> AnyView
}

// MARK: - Native Metal scenes

/// A scene that renders entirely through its own Metal pipeline (native
/// Illuminatorama scenes) and hosts its own `MTKView`. `ContentView` embeds
/// that view directly via `c as? NativeMetalScene`, skipping the SCNView pass.
///
/// WHY THIS EXISTS AS A REFINING PROTOCOL WITH A *NON-OPTIONAL* REQUIREMENT:
/// `SceneController.metalView` is `MTKView?` with a `nil` default. A controller's
/// stored `let metalView: MTKView` (NON-optional) does **not** witness that
/// optional requirement when a default exists — through `any SceneController` you
/// get the default `nil`, not the stored view (the same `stored let` vs
/// `optional var` trap the `timeline`/`sceneTimeline` split already dodged).
/// That silently routed EVERY native scene to the SCNView `background.contents`
/// path, which stretches the fixed-canvas texture to the window (object-fit:
/// fill) — the long-standing "aspect ratio" bug. The cover-correct
/// `IlluminatoramaRenderer.present(to:)` path was never reached for the live
/// view. This protocol's requirement is non-optional, so the existing
/// `let metalView: MTKView` witnesses it correctly and the MTKView present path
/// (with object-fit:cover) actually drives display. Conformances are declared
/// centrally in `SceneManifest.swift`, so scene controllers need no change.
@MainActor
public protocol NativeMetalScene: SceneController {
    var metalView: MTKView { get }

    /// Install (or clear, with `nil`) a per-frame tap used by the video
    /// recorder. Native scenes drive their own `MTKView` present, so there's no
    /// SceneKit render delegate for `SceneRecorder` to hook — instead the
    /// controller forwards `tap` to its renderer, which calls it each presented
    /// frame with the final frame texture + a command queue ordered behind the
    /// render. Default no-op for native scenes without a tap-capable renderer.
    func setRecorderFrameTap(_ tap: ((_ texture: MTLTexture, _ queue: MTLCommandQueue) -> Void)?)
}

public extension NativeMetalScene {
    func setRecorderFrameTap(_ tap: ((MTLTexture, MTLCommandQueue) -> Void)?) {}
}

// MARK: - Default implementations

public extension SceneController {
    /// The native render-canvas resolution in pixels for Illuminatorama scenes
    /// that paint their fixed-size output texture into `scene.background.contents`
    /// (the snapshot seam every native scene binds at init — the texture is
    /// allocated at the renderer's `outputWidth × outputHeight`). Derived from
    /// that texture's dimensions so every native scene reports its canvas
    /// generically, with no per-scene wiring.
    ///
    /// `nil` for plain SCNView scenes — whose background is a colour / HDR image,
    /// not an `MTLTexture` — which render straight into the live drawable and have
    /// no fixed canvas to up/downscale from. Consumed by the render-scale HUD to
    /// flag when the fixed canvas is being upscaled to fill a larger window (the
    /// reason the live view can look softer than a headless capture).
    var renderCanvasSize: SIMD2<Int>? {
        guard let tex = scene.background.contents as? MTLTexture else { return nil }
        return SIMD2(tex.width, tex.height)
    }

    var technique: SCNTechnique? { nil }
    var metalView: MTKView? { nil }
    var sceneTimeline: SceneTimeline? { nil }
    var settingsExportText: String { "" }
    /// SCNView-based scenes don't need this — their FPS comes from the SceneKit
    /// render-delegate path. The no-op default makes the generic AppModel wiring
    /// (`ctrl.fpsReporter = ...`) a safe call for every scene type.
    var fpsReporter: (@MainActor (Double) -> Void)? {
        get { nil }
        set { }
    }
    func applyPerfOverrides(_ overrides: [String: String]) {}

    // ── Shared activation (GPU conservation) ─────────────────────────
    /// The single entry point `AppModel` uses to (de)activate a scene when the
    /// sidebar selection changes. Do NOT call `setActive` directly from the app.
    ///
    /// This centralises the one piece of lifecycle every scene must get right
    /// but historically hand-rolled (and several forgot): pausing a native
    /// `MTKView`'s vsync present so a *backgrounded* scene stops driving the GPU.
    /// Only the active scene's view is hosted in the SwiftUI hierarchy, but a
    /// retained `NativeMetalScene` whose `MTKView.isPaused` is left `false` keeps
    /// its display link presenting — burning GPU the active scene needs. Pausing
    /// it here, for every `NativeMetalScene` uniformly, fixes that in one place.
    ///
    /// After the shared MTKView pause, this forwards to the scene's own
    /// `setActive`, which owns the scene-specific tick/spawn/render-timer control.
    /// Scenes therefore no longer need to touch `metalView.isPaused` themselves.
    func applyActivation(_ active: Bool) {
        // `NativeMetalScene.metalView` is non-optional and correctly witnessed;
        // the `SceneController.metalView` requirement is nil for these (the
        // stored-`let` vs optional-requirement trap documented on NativeMetalScene),
        // so route through the refining protocol to reach the real view.
        (self as? NativeMetalScene)?.metalView.isPaused = !active
        setActive(active)
    }

    func captureReviewAngle(yawDeg: Float, pitchDeg: Float,
                            size: SIMD2<Int>) -> MTLTexture? { nil }
    func captureHeroSettled(size: SIMD2<Int>) -> MTLTexture? { nil }
    func penetrationProbers() -> [any PenetrationProbing] { [] }
    func splinePolylines() -> [SplinePolyline] { [] }
}
