import Foundation
import Metal
import Observation
import OSLog
import QuartzCore
import SceneKit
import VisualizerCore

// ── ILLUMINATORAMA OVERLAY (Phase 2.6) ───────────────────────────────────────
//
// The "render any scene through Illuminatorama" wrapper. The app's global
// "Use Illuminatorama Renderer" toggle attaches one of these to the active
// `SceneController`; the overlay drives a 60 Hz tick that:
//
//   1. Walks the controller's `SCNScene` via `IlluminatoramaSceneExtractor`,
//      pushing instances + lights into a `IlluminatoramaRenderer`.
//   2. Pulls the camera state from the controller's `cameraNode` so the
//      Illuminatorama image follows whatever the user does in the host UI.
//   3. Calls `renderer.render()` which writes its `outputTexture`.
//
// The detail view binds `outputTexture` as the SCNScene background of a
// minimal "display scene" so the visible pixels are Illuminatorama's, not
// SceneKit's.
//
// Lifecycle is bound to the AppModel: `attach(scene:cameraNode:)` starts the
// tick loop, `detach()` invalidates the Timer and tears the renderer down.
// Switching scenes detaches and re-attaches.
//
// What doesn't render (carried over from the extractor doc):
//   • SCNParticleSystem (particles)
//   • SCNMaterial.shaderModifiers
//   • Spot lights, area lights
//   • Skinned / morphed geometry (only static buffers are read)
//   • SceneKit post-FX (the renderer's own bloom/SSAO/SSR/TAA stack applies
//     instead).

@MainActor
@Observable
public final class IlluminatoramaOverlay {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                     category: "illuminatoramaOverlay")

    /// Identifies the controller this overlay is currently attached to.
    /// The host uses this to decide whether a scene-selection change needs
    /// to tear down the overlay and rebuild for the new scene.
    public private(set) var attachedSceneID: String? = nil

    /// SceneKit scene whose `background.contents` is the renderer's output
    /// texture. Bind this scene to the host `SCNView` so SceneKit composites
    /// the Illuminatorama image onto its drawable without competing with
    /// the original scene's render-to-screen.
    public let displayScene: SCNScene

    /// Camera node parented under `displayScene.rootNode`. `scene.background.contents`
    /// renders the Illuminatorama output regardless of the camera, but the
    /// host SCNView and the headless snapshot harness both expect a non-nil
    /// `pointOfView`. Exposed so the renderer wrapper view can pass it
    /// explicitly to `SceneRenderer.pointOfView`.
    public let displayPointOfView: SCNNode

    /// The final LDR texture the overlay writes each tick. Same identity
    /// as `displayScene.background.contents`; exposed so consumers can
    /// read it directly (e.g. `HeadlessSnapshot` bypasses SCNRenderer for
    /// overlay scenes because `SCNRenderer.snapshot()` doesn't pull
    /// `scene.background.contents` from an offscreen render).
    public var displayTexture: MTLTexture { renderer.outputTexture }

    /// One-shot render at an arbitrary output size for headless snapshots
    /// (or any consumer that needs a fixed-aspect frame independent of the
    /// live SCNView's drawable). Resizes the renderer to the requested
    /// output dims, runs a fresh extract+render pass, and returns the
    /// resulting `outputTexture`. The next regular tick will resize back
    /// to whatever `viewportSizeProvider` reports — interactive use
    /// resumes at the SCNView's native aspect without further plumbing.
    ///
    /// Important: returns the renderer's `outputTexture` directly. Reading
    /// from it via `getBytes` only works on `.shared` storage — the
    /// caller must blit into a staging texture first (see
    /// `HeadlessSnapshot.renderIlluminatoramaHero`).
    public func renderForSnapshot(width: Int, height: Int) -> MTLTexture {
        // Pull the latest scene state in too — the overlay's regular tick
        // is what drives extraction, and a snapshot fired between ticks
        // would otherwise capture stale instances.
        offlineUpdater.update(atTime: CACurrentMediaTime())
        renderer.resize(width: max(1, width), height: max(1, height))
        // Apply env-driven debug/feature flags BEFORE extraction — some (RT
        // from extracted scene) gate what the extractor builds, not just the
        // final render. The render-only knobs (debug term, exposure) are
        // unaffected by the earlier ordering.
        applySnapshotDebugEnv()
        extractor.extractFrame(into: renderer)
        // Blocking render: the export path must NOT drop the frame under GPU
        // back-pressure (the interactive tick's default), or the snapshot reads
        // a stale / never-rendered pool buffer and captures magenta.
        renderer.render(blocking: true)
        // `renderer.render()` commits its command buffer but doesn't wait.
        // A caller that immediately reads `outputTexture` (e.g. via a blit
        // on a *different* command queue, as `HeadlessSnapshot` does)
        // would race the GPU — the blit could schedule before the render
        // finishes writing. Sync barrier here on the renderer's own queue:
        // an empty command buffer guarantees that, by the time it
        // completes, every earlier-enqueued buffer on the same queue is
        // also complete. Cheap (no GPU work) and avoids leaking the queue.
        //
        // The wait is fine here because `renderForSnapshot` is the headless
        // PNG-export path — it runs only when `HeadlessSnapshot` needs a
        // sync read of `outputTexture`, NEVER on the per-tick interactive
        // render. The CPU stall it introduces is bounded to the exact
        // moment a snapshot is captured.
        if let cb = renderer.commandQueue.makeCommandBuffer() {
            cb.label = "Illuminatorama.snapshotBarrier"
            cb.commit()
            cb.waitUntilCompleted() // gpu-ok: snapshot export path, not per-frame
        }
        // The frame we just rendered is now complete; promote it so the
        // returned `outputTexture` is THIS frame, not the previous presented
        // one. (The render-loop promote runs at the *start* of render(), so
        // without this the snapshot would lag by one frame.)
        renderer.promoteCompletedBuffer()
        return renderer.outputTexture
    }

    /// Headless-only diagnostic hook (no effect on interactive ticks, which
    /// never call `renderForSnapshot`). Lets the per-term split-render and a
    /// static-exposure A/B be driven from the render script without a rebuild:
    ///   VIZ_ILLUMI_DEBUG_TERM   0..8 → `renderer.debugTerm` (see DebugTerm;
    ///                           8 = surface-cache isolation, RT + cache only)
    ///   VIZ_ILLUMI_STATIC_EXPOSURE  "1" → auto-exposure off, exposure = 1.0
    ///                               (or a float value → that exposure)
    private func applySnapshotDebugEnv() {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["VIZ_ILLUMI_DEBUG_TERM"], let v = UInt32(raw),
           let term = IlluminatoramaRenderer.DebugTerm(rawValue: v) {
            renderer.debugTerm = term
            // Also mirror into the shared setting: `render()`'s per-frame
            // `applySharedLensFX` copies `shared.debugView → renderer.debugTerm`,
            // which would otherwise clobber this env value back to `.composite`.
            if let dv = IlluminatoramaDebugView(rawValue: Int(v)) {
                IlluminatoramaSharedSettings.shared.debugView = dv
            }
        }
        if let raw = env["VIZ_ILLUMI_STATIC_EXPOSURE"] {
            renderer.autoExposureEnabled = false
            renderer.exposure = Float(raw) ?? 1.0
        }
        // Drive the "Exposure" compensation knob headless WITHOUT pinning
        // auto-exposure (unlike STATIC_EXPOSURE). Set BOTH the shared setting and
        // `renderer.exposure` directly: this hook runs AFTER the tick's
        // `applySharedSettings` already copied `shared.exposure → renderer.exposure`
        // for this frame, so the shared write alone would land too late. Auto-
        // exposure stays on, so the tonemap uses `autoBase · renderer.exposure` —
        // exercising the same compensation path the interactive slider drives.
        if let raw = env["VIZ_ILLUMI_EXPOSURE"], let v = Double(raw) {
            IlluminatoramaSharedSettings.shared.exposure = v
            renderer.exposure = Float(v)
        }
        if let raw = env["VIZ_ILLUMI_IBL_DESAT"], let v = Float(raw) {
            renderer.iblDiffuseDesaturation = v
        }
        // VIZ_ILLUMI_RT=1 → hardware-RT GI / soft shadows built from the
        // EXTRACTED scene geometry (generalises the hand-authored-room path).
        // The extractor bakes a world-space AS and points the RT sun at the
        // scene's key light. VIZ_ILLUMI_RT_REFL=1 adds RT glossy reflections.
        // This is the headless/dev OVERRIDE; the permanent per-scene opt-in is
        // `SceneDescriptor.illuminatoramaRT` → `setExtractedSceneRT` (Phase
        // 4.36). NOTE this env path enables RT only here, inside the snapshot
        // render — so it builds the AS ONCE; the descriptor path enables RT in
        // the live tick loop, which only suits stable-topology / CPU-clean
        // scenes (see `IlluminatoramaRTOptions`).
        if env["VIZ_ILLUMI_RT"] == "1" {
            renderer.rtEnabled = true
            renderer.buildRTFromExtractedScene = true
            if env["VIZ_ILLUMI_RT_REFL"] == "1" {
                renderer.rtReflectionsEnabled = true
            }
            // VIZ_ILLUMI_SURFCACHE=1 → multi-bounce surface cache through the
            // TLAS path (P1c). Dev/headless A/B override for the descriptor's
            // `surfaceCache` flag; requires RT.
            if env["VIZ_ILLUMI_SURFCACHE"] == "1" {
                renderer.surfaceCacheEnabled = true
            }
            // Incremental invalidation (per-triangle TLAS path) is default-ON now;
            // VIZ_SURFCACHE_INCREMENTAL=0 is the kill-switch (for an A/B against the
            // full-reset behaviour). Same form as configureForScene so order is moot.
            renderer.surfaceCacheIncremental =
                renderer.surfaceCacheEnabled && env["VIZ_SURFCACHE_INCREMENTAL"] != "0"
        }
    }

    /// FIXED render canvas. The overlay is built at this size ONCE and never
    /// resized at runtime — `IlluminatoramaRenderer.resize` reallocates every
    /// GPU target and resets all temporal accumulation (TAA/SVGF/AO/SSR/GI),
    /// flashing noise on every window drag. `tick()` instead cover-crops this
    /// canvas to the live viewport aspect via the display background's UV
    /// transform: fills the window, never stretches, never letterboxes. 16:9 at
    /// retina resolution.
    public static let renderWidth: Int = 2880
    public static let renderHeight: Int = 1620

    /// Closure the host (AppModel) sets so the overlay can poll the
    /// SCNView's drawable size each tick. When the reported size differs
    /// from the renderer's current size, the renderer is resized to
    /// match — which also updates `IlluminatoramaCamera.aspect`, so the
    /// image stays undistorted across window-aspect changes.
    ///
    /// Pattern lifted directly from `FireworksUltraController`'s
    /// cover-fill compositor resize (commit `feafe40`). `nil` means the
    /// overlay stays at its init size (lab / tests / headless use).
    public var viewportSizeProvider: (@MainActor () -> SIMD2<Int>?)? = nil

    /// Closure the host (AppModel) sets to receive the overlay's actual
    /// per-tick frame rate. SCN's render-thread tick rate (the toolbar's
    /// default reading) overstates the perceived FPS for overlay-driven
    /// scenes — SCN keeps drawing the same `outputTexture` between
    /// Illuminatorama updates, which can run at any rate ≤ the Timer's
    /// 60 Hz cap depending on SSAA scale and lighting cost.
    ///
    /// Mirrors `FireworksUltraController.fpsReporter` (commit `87f4f54`).
    /// Emitted ~twice per second so the toolbar's numeric display stays
    /// readable.
    public var fpsReporter: (@MainActor (Double) -> Void)? = nil

    /// Rolling FPS sample state. Reset on detach so the next attach
    /// starts a fresh window rather than reporting stale dt.
    private var fpsAccumulatedTime: CFTimeInterval = 0
    private var fpsTickCount: Int = 0
    private var lastTickTime: CFTimeInterval = CACurrentMediaTime()

    private let renderer: IlluminatoramaRenderer
    private let extractor: IlluminatoramaSceneExtractor

    /// Identity of the renderer output texture currently bound to
    /// `displayScene.background.contents`. The renderer rotates through a pool
    /// of output buffers (race fix), so we rebind whenever this changes.
    private var boundOutputTexture: ObjectIdentifier?

    /// Off-screen SCNRenderer used purely to drive the host scene's
    /// `update(atTime:)`. SCNView's render loop is what normally ticks
    /// `SCNScene.physicsWorld`, `SCNAction` queues, particle emitters, and
    /// per-node animations; when Illuminatorama takes over the visible
    /// view, the host scene loses that driver and physics-heavy scenes
    /// (HotdogDrop, Eggs, etc.) freeze in place. Calling
    /// `update(atTime:)` on a non-view-owned `SCNRenderer` is Apple's
    /// documented way to advance scene state in the absence of a visible
    /// renderer — see `SCNRenderer` "Offline rendering and updates" in
    /// the SceneKit reference.
    ///
    /// We never call this renderer's `render(...)` — it's purely a tick
    /// driver. No drawable, no command-buffer cost.
    private let offlineUpdater: SCNRenderer

    private var tickTimer: Timer? = nil

    public init(engine: SimEngine = .shared) throws {
        self.displayScene = SCNScene()

        let cam = IlluminatoramaCamera(
            position: SIMD3(0, 3, 7),
            target: SIMD3(0, 0.5, 0),
            up: SIMD3(0, 1, 0),
            fovYRadians: .pi / 3,
            aspect: Float(Self.renderWidth) / Float(Self.renderHeight),
            zNear: 0.1,
            zFar: 200
        )
        self.renderer = try IlluminatoramaRenderer(
            engine: engine,
            width: Self.renderWidth,
            height: Self.renderHeight,
            camera: cam
        )
        self.extractor = IlluminatoramaSceneExtractor(device: engine.device)
        self.offlineUpdater = SCNRenderer(device: engine.device, options: nil)
        self.displayScene.background.contents = renderer.outputTexture
        self.boundOutputTexture = ObjectIdentifier(renderer.outputTexture)
        // Give the display scene a dummy POV. `scene.background.contents`
        // draws the renderer's output texture full-frame regardless of
        // the camera, but consumers that walk the scene (HeadlessSnapshot,
        // future SCNTechnique-based effects) expect a non-nil `pointOfView`
        // and crash / no-op without one.
        let dummyCamera = SCNCamera()
        let dummyCameraNode = SCNNode()
        dummyCameraNode.camera = dummyCamera
        dummyCameraNode.name = "IlluminatoramaOverlay.displayPOV"
        self.displayScene.rootNode.addChildNode(dummyCameraNode)
        self.displayPointOfView = dummyCameraNode
    }

    /// Attach to a controller. Starts the 60 Hz extract+render loop.
    /// Calling `attach` while already attached re-points at the new scene
    /// without tearing the renderer down (avoids a hitch on scene switch).
    public func attach(scene: SCNScene, cameraNode: SCNNode, sceneID: String) {
        if attachedSceneID == sceneID, extractor.scene === scene { return }
        // Different scene → flush mesh cache. SCNGeometry pointers from
        // the previous scene are stale; keeping their entries would leak
        // VRAM until next scene-switch.
        if attachedSceneID != sceneID {
            extractor.resetCache()
        }
        extractor.scene = scene
        extractor.cameraNode = cameraNode
        offlineUpdater.scene = scene
        offlineUpdater.pointOfView = cameraNode
        attachedSceneID = sceneID
        if tickTimer == nil { startTickLoop() }
        // Fresh FPS window for the new attachment so the first sample
        // reflects this scene's actual rate rather than including a long
        // gap since the last detach.
        lastTickTime = CACurrentMediaTime()
        fpsAccumulatedTime = 0
        fpsTickCount = 0
        Self.log.info("Illuminatorama overlay attached to \(sceneID, privacy: .public)")
    }

    /// Detach and stop ticking. The renderer + displayScene stay alive so
    /// the next attach is fast; only the per-scene state is reset.
    public func detach() {
        tickTimer?.invalidate()
        tickTimer = nil
        extractor.scene = nil
        extractor.cameraNode = nil
        offlineUpdater.scene = nil
        offlineUpdater.pointOfView = nil
        attachedSceneID = nil
        Self.log.info("Illuminatorama overlay detached")
    }

    // ── Tick loop ─────────────────────────────────────────────────────────────

    private func startTickLoop() {
        let t = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.tick()
            }
        }
        RunLoop.main.add(t, forMode: .common)
        tickTimer = t
    }

    private func tick() {
        // No scene → no work. Detach was probably mid-flight.
        guard extractor.scene != nil else { return }
        // object-fit: cover. The canvas is FIXED (see `renderWidth/Height`); we
        // never resize the renderer to the drawable — that reallocates every GPU
        // target and resets all temporal accumulation, flashing noise on every
        // window drag. Crop the fixed canvas to the live viewport aspect via the
        // display background's UV transform: fills the window, never stretches,
        // never letterboxes. Identity when the window already matches 16:9.
        if let size = viewportSizeProvider?(), size.x > 0, size.y > 0 {
            let uv = renderer.coverUVRect(viewportWidth: size.x, viewportHeight: size.y)
            var xform = SCNMatrix4Identity
            xform.m11 = CGFloat(uv.x); xform.m22 = CGFloat(uv.y)
            xform.m41 = CGFloat(uv.z); xform.m42 = CGFloat(uv.w)
            displayScene.background.contentsTransform = xform
        }
        // Tick the host scene's physics + actions + particle emitters
        // before reading its node graph. Without this, scenes that rely
        // on SceneKit's render loop to advance their simulation (which
        // is most of them — physics, SCNAction, particles) would freeze
        // while Illuminatorama is the visible renderer.
        offlineUpdater.update(atTime: CACurrentMediaTime())
        extractor.extractFrame(into: renderer)
        applySharedSettings()
        let rendered = renderer.render()
        // The renderer triple-buffers its output and promotes a freshly
        // *completed* buffer to `outputTexture` each tick (race fix — see
        // `IlluminatoramaRenderer.ldrPool`). Its identity therefore changes
        // frame-to-frame, so rebind the display background whenever it does;
        // otherwise SceneKit keeps sampling the first buffer (which the
        // renderer stops writing to once it rotates past it).
        if ObjectIdentifier(renderer.outputTexture) != boundOutputTexture {
            displayScene.background.contents = renderer.outputTexture
            boundOutputTexture = ObjectIdentifier(renderer.outputTexture)
        }

        // FPS measurement — wall-clock dt between tick completions, which
        // reflects what Illuminatorama is actually producing rather than
        // SCN's render-thread cadence (which keeps drawing the cached
        // `outputTexture` between updates and overstates perceived FPS,
        // especially at higher SSAA scales where the renderer drops
        // below the Timer's 60 Hz cap). Mirrors `FireworksUltraController`.
        // Only RENDERED frames count: `render()` drops a frame (returns false)
        // when the GPU hasn't drained the previous frames, so counting every
        // tick would report the fixed 60 Hz Timer cadence, not real throughput.
        let now = CACurrentMediaTime()
        let dt = max(0, min(0.1, now - lastTickTime))
        lastTickTime = now
        if dt > 0 {
            fpsAccumulatedTime += dt
            if rendered { fpsTickCount += 1 }
            if fpsAccumulatedTime >= 0.5 {
                let hz = Double(fpsTickCount) / fpsAccumulatedTime
                fpsReporter?(hz)
                fpsAccumulatedTime = 0
                fpsTickCount = 0
            }
        }
    }

    /// Push every knob from `IlluminatoramaSharedSettings.shared` into the
    /// renderer. Called each tick so changes made in the modal sheet take
    /// effect on the next frame.
    private func applySharedSettings() {
        let s = IlluminatoramaSharedSettings.shared

        // Post-FX (shared lens/colour-grade group — see applySharedPostFX)
        renderer.exposure = Float(s.exposure)
        renderer.applySharedPostFX(s)

        // SSAO + SSR
        renderer.ssaoIntensity = Float(s.ssaoIntensity)
        renderer.ssaoRadius = Float(s.ssaoRadius)
        renderer.ssrIntensity = Float(s.ssrIntensity)
        renderer.ssrMaxDistance = Float(s.ssrMaxDistance)
        renderer.ssrThickness = Float(s.ssrThickness)
        renderer.ssrMaxSteps = UInt32(max(8, min(128, s.ssrMaxSteps)))

        // TAA + denoiser
        renderer.taaEnabled = s.taaEnabled
        renderer.taaHistoryBlend = Float(s.taaHistoryBlend)
        renderer.taaJitterPixels = Float(s.taaJitterPixels)
        renderer.ssaoDenoiseEnabled = s.denoiserEnabled
        renderer.ssrDenoiseEnabled = s.denoiserEnabled
        renderer.debandDitherEnabled = s.debandDitherEnabled

        // IBL
        renderer.iblEnabled = s.iblEnabled
        renderer.iblIntensity = Float(s.iblIntensity)
        renderer.dfgLUTEnabled = s.dfgLUTEnabled

        // DDGI
        renderer.ddgiEnabled = s.ddgiEnabled
        renderer.ddgiIrradianceScale = Float(s.ddgiIrradianceScale)
        renderer.ddgiHysteresis = Float(s.ddgiHysteresis)
        renderer.ddgiTwoBounceEnabled = s.ddgiTwoBounceEnabled

        // Shadows
        renderer.shadowsEnabled = s.shadowsEnabled
        renderer.shadowBias = Float(s.shadowBias)
        renderer.shadowSlopeBias = Float(s.shadowSlopeBias)
        renderer.shadowPcfRadius = UInt32(max(0, min(2, s.shadowPcfRadius)))
        renderer.shadowMaxDistance = Float(s.shadowMaxDistance)

        // Internal render scale — resize() is a no-op when nothing changes.
        // When it does change it swaps `outputTexture`; the rebind right
        // after `renderer.render()` (see `tick()`) catches that for free.
        if abs(renderer.internalRenderScale - Float(s.internalRenderScale)) > 1e-3 {
            renderer.internalRenderScale = Float(s.internalRenderScale)
            renderer.resize(width: renderer.outputWidth, height: renderer.outputHeight)
            displayScene.background.contents = renderer.outputTexture
            boundOutputTexture = ObjectIdentifier(renderer.outputTexture)
        }
    }

    // ── Tunables (public so AppModel / UI can wire these to settings) ────────

    public var iblEnabled: Bool {
        get { renderer.iblEnabled }
        set { renderer.iblEnabled = newValue }
    }

    public var ddgiEnabled: Bool {
        get { renderer.ddgiEnabled }
        set { renderer.ddgiEnabled = newValue }
    }

    public var taaEnabled: Bool {
        get { renderer.taaEnabled }
        set { renderer.taaEnabled = newValue }
    }

    public var exposure: Float {
        get { renderer.exposure }
        set { renderer.exposure = newValue }
    }

    /// Phase 4.10 — flip on to render per-spot shadow maps for the first
    /// N spots in the active scene (N = atlas capacity, currently 8).
    /// Costs one depth render pass per shadowed spot per frame; default
    /// off so scenes with many emissive rail spots (Eggs has 20+) don't
    /// pay the cost unless asked.
    public var spotShadowsEnabled: Bool {
        get { renderer.spotShadowsEnabled }
        set { renderer.spotShadowsEnabled = newValue }
    }

    /// Phase 4.36 — permanent per-scene hardware-RT opt-in for the global
    /// overlay (replaces the headless-only `VIZ_ILLUMI_RT` env hook for
    /// interactive use). `AppModel` calls this on every attach with the active
    /// `SceneDescriptor.illuminatoramaRT`. Enables the extracted-scene RT pass
    /// (soft sun shadows + 1-bounce GI built from the scene's geometry; on RT
    /// hardware this runs through the TLAS + per-instance-refit path) and,
    /// optionally, glossy RT reflections.
    ///
    /// The `VIZ_ILLUMI_RT` / `VIZ_ILLUMI_RT_REFL` env vars still force RT on
    /// regardless of the descriptor (dev/headless override), so passing `nil`
    /// for a scene that the env hook enabled won't switch it back off.
    public func setExtractedSceneRT(_ options: IlluminatoramaRTOptions?) {
        let env = ProcessInfo.processInfo.environment
        let envForce = env["VIZ_ILLUMI_RT"] == "1"
        let envRefl = env["VIZ_ILLUMI_RT_REFL"] == "1"
        let wantRT = (options?.enabled ?? false) || envForce
        let wantRefl = ((options?.enabled ?? false) && (options?.reflections ?? false))
            || (envForce && envRefl)
        renderer.rtEnabled = wantRT
        renderer.buildRTFromExtractedScene = wantRT
        renderer.rtReflectionsEnabled = wantRefl
        // Phase 4.38 — surface cache (multi-bounce GI) opt-in. Requires RT; the
        // extractor auto-builds per-triangle cards and the scene routes through
        // the soup RT path the cache rides (see updateRTAccel).
        let envSurf = env["VIZ_ILLUMI_SURFCACHE"] == "1"
        renderer.surfaceCacheEnabled = wantRT && ((options?.surfaceCache ?? false) || envSurf)
        // Incremental invalidation (Phase 4.38d → production). DEFAULT ON for any
        // surface-cache scene; it self-gates to the per-triangle TLAS path and is a
        // no-op on chart/soup/static scenes, so it only helps animated TLAS scenes
        // (e.g. cityStreet's car). `VIZ_SURFCACHE_INCREMENTAL=0` kills it for an A/B
        // against the full-reset behaviour. Set here (not just headless) so the
        // moved-card re-framing runs in the LIVE tick loop. See
        // docs/illuminatorama/surface-cache-incremental-invalidation.md.
        renderer.surfaceCacheIncremental =
            renderer.surfaceCacheEnabled && env["VIZ_SURFCACHE_INCREMENTAL"] != "0"
        // Per-scene IBL bake desaturation (for monochromatic-sky scenes like
        // the pizza broiler). Marks IBL dirty so it re-bakes with the new value.
        renderer.iblBakeDesaturation = options?.iblBakeDesaturation ?? 0.0
        // Always reset exposure knobs so switching away from an overriding scene
        // doesn't leave the renderer in a non-default state.
        renderer.autoExposureTargetEV = options?.autoExposureTargetEV ?? -4.0
        renderer.autoExposureEnabled  = options?.autoExposureEnabled  ?? true
        // Give the freshly-shown scene a clean chance at RT — clear any
        // auto-disable latched by a previous (thrashing/heavy) scene.
        renderer.resetRTGuard()
    }
}
