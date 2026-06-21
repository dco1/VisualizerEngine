import simd
import VisualizerCore

public extension IlluminatoramaRenderer {
    /// The **universal lens-aberration knobs** from `IlluminatoramaSharedSettings`:
    /// transverse chromatic aberration, spherical aberration, axial fringing, and
    /// the post-FX easing time-constant.
    ///
    /// These are special because NO scene anywhere overrides them — every scene
    /// leaves them to the Illuminatorama Settings panel (all four are off-by-default
    /// no-ops). That makes them safe to apply for EVERY Illuminatorama scene from
    /// one place: `render()` calls this each frame (gated by `appliesSharedLensFX`),
    /// so every scene — including the three that never wired post-FX, and every
    /// future scene — inherits them with zero per-scene code. This is what stops the
    /// "added a panel knob, forgot to wire it into scene N" failure mode (how
    /// spherical aberration first shipped looking dead on Room / Lab / Soft Serve).
    func applySharedLensFX(_ s: IlluminatoramaSharedSettings = .shared) {
        chromaticAberration = Float(s.chromaticAberrationEnabled ? s.chromaticAberration : 0)
        fringe              = Float(s.fringeEnabled ? s.fringe : 0)
        fringeTint          = SIMD3(Float(s.fringeTintR), Float(s.fringeTintG), Float(s.fringeTintB))
        sphericalAberration = Float(s.sphericalAberrationEnabled ? s.sphericalAberration : 0)
        vignetteStrength    = Float(s.vignetteEnabled ? s.vignette : 0)
        vignetteExtent      = Float(s.vignetteExtent)
        filmGrainStrength   = Float(s.filmGrainEnabled ? s.filmGrain : 0)
        filmGrainSize       = Float(s.filmGrainSize)
        motionBlurStrength  = Float(s.motionBlurEnabled ? s.motionBlur : 0)
        motionBlurMaxPx     = Float(s.motionBlurMaxPx)
        postFXEasingTau     = s.postFXEasing.tau
    }

    /// The shared lens FX (above) **plus the panel's bloom**.
    ///
    /// Bloom is OPT-IN per scene — unlike the aberration cluster, many scenes
    /// art-direct their own bloom (EggsUltra 2.2 for HDR emissives, HotdogPress 0.55
    /// for wet specular, CoinPusher's dark-arcade neon, …) instead of taking the
    /// panel value. So bloom is deliberately NOT auto-applied in `render()` (that
    /// runs after the scene's tick and would clobber those choices). A scene that
    /// *does* want the panel to drive its bloom calls this from its tick.
    ///
    /// Deliberately still NOT handled — each scene curates these itself:
    ///   • `exposure` / `autoExposureEnabled` — House fixes its own; Room/Lab pass
    ///     the shared value through. There's no single correct policy.
    ///   • SSAO / SSR / TAA / DDGI / shadows / IBL / internal render-scale — some
    ///     scenes own a subset (Room's RT path drives SSR/shadows/IBL; Lab installs
    ///     a custom DDGI probe grid).
    func applySharedPostFX(_ s: IlluminatoramaSharedSettings = .shared) {
        bloomThreshold = Float(s.bloomThreshold)
        bloomIntensity = Float(s.bloomIntensity)
        applySharedLensFX(s)
    }
}
