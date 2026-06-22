import Foundation
import Observation

/// Renderer-level knobs that apply to *any* scene rendered through the
/// Illuminatorama pipeline — both the global overlay (when "Use Illuminatorama
/// Renderer" is on) and the natively-Illuminatorama scenes (Lab, Room,
/// InfiniteSoftServe).
///
/// These used to live duplicated on each native scene's Settings struct. They
/// now live once, on `IlluminatoramaSharedSettings.shared`, and the
/// Illuminatorama Settings modal sheet edits them from anywhere in the app —
/// including scenes whose own settings panel doesn't expose them.
///
/// Per-scene knobs (camera, lighting authoring, RT opt-in, scene-specific
/// dust/DoF/coil/etc.) stay on the per-scene Settings.
@MainActor
@Observable
public final class IlluminatoramaSharedSettings {
    /// Process-wide singleton. Native controllers + the IlluminatoramaOverlay
    /// pull from this each tick; the modal edits it directly.
    public static let shared = IlluminatoramaSharedSettings()

    // ── Post-FX ───────────────────────────────────────────────────────
    /// How a Post-FX slider change is applied: instantly, or eased toward the new
    /// value over a time constant. The renderer reads `.tau` and exponentially
    /// smooths the Post-FX knobs (exposure, bloom, chromatic aberration, fringe)
    /// toward their targets each frame, so dragging a slider glides.
    public var postFXEasing: IlluminatoramaEasing = .smooth
    public var exposure: Double = 1.0
    public var bloomThreshold: Double = 1.0
    public var bloomIntensity: Double = 0.6
    /// Lens-style transverse chromatic aberration (R/B radial split that grows
    /// toward the frame edges). OFF by default → an exact shader no-op, so every
    /// scene is unaffected unless the user opts in from the Illuminatorama
    /// settings panel. `chromaticAberration` is the strength multiplier (only
    /// meaningful when enabled).
    public var chromaticAberrationEnabled: Bool = false
    public var chromaticAberration: Double = 1.0
    /// Longitudinal / axial chromatic aberration — "purple fringing": a coloured
    /// halo on high-contrast edges (frame-uniform, unlike the radial lateral CA
    /// above). The tint sits on the dark side of a bright edge; its complement
    /// (≈ green for a violet tint) lands on the bright side, mimicking how a real
    /// lens focuses wavelengths at slightly different depths. OFF by default → an
    /// exact shader no-op. `fringe` is the strength; `fringeTintR/G/B` is the
    /// dark-side tint in sRGB (default a violet).
    public var fringeEnabled: Bool = false
    public var fringe: Double = 1.0
    public var fringeTintR: Double = 0.62
    public var fringeTintG: Double = 0.12
    public var fringeTintB: Double = 0.92
    /// Spherical aberration: a radial blur that grows quadratically from the
    /// frame centre toward the edges, mimicking how a real lens's outer zones
    /// focus at a slightly different plane than the paraxial rays. OFF by
    /// default → exact shader no-op. `sphericalAberration` is the strength.
    public var sphericalAberrationEnabled: Bool = false
    public var sphericalAberration: Double = 1.0
    /// Lens vignette — darkens the frame corners (an optical falloff that draws the
    /// eye to the centre). OFF by default → an exact shader no-op. `vignette` is the
    /// corner-darkening strength; `vignetteExtent` is the fraction of the frame
    /// (from the centre) kept fully bright before the falloff begins.
    public var vignetteEnabled: Bool = false
    public var vignette: Double = 0.35
    public var vignetteExtent: Double = 0.55
    /// Film-stock grain — per-frame, luminance-masked noise over the final image
    /// (peaks in the mids, fades in highlights/blacks like real film). OFF by
    /// default → an exact shader no-op. `filmGrain` is the amplitude; `filmGrainSize`
    /// is the grain cell size in pixels.
    public var filmGrainEnabled: Bool = false
    public var filmGrain: Double = 0.06
    public var filmGrainSize: Double = 1.5
    /// Velocity-buffer motion blur — streaks moving objects + camera pans along their
    /// screen motion (a camera-shutter look). OFF by default → exact shader no-op.
    /// `motionBlur` is the strength (≈ shutter fraction; ~0.5 ≈ a natural 180° look);
    /// `motionBlurMaxPx` caps the streak length in pixels so a fast pan can't smear
    /// the whole frame.
    public var motionBlurEnabled: Bool = false
    public var motionBlur: Double = 0.5
    public var motionBlurMaxPx: Double = 48
    /// Colour-grade LUT look (issue #65). `.none` → exact no-op; any other look
    /// bakes a procedural 3D LUT the renderer blends by `colorLUTAmount`. A scene
    /// can override with a bespoke `.cube` via `IlluminatoramaRenderer.loadCubeLUT`.
    public var colorGradeLook: IlluminatoramaColorGradeLook = .none
    public var colorLUTAmount: Double = 1.0

    // ── SSAO ──────────────────────────────────────────────────────────
    public var ssaoIntensity: Double = 0.85
    public var ssaoRadius: Double = 0.4

    // ── SSR ───────────────────────────────────────────────────────────
    public var ssrIntensity: Double = 0.7
    public var ssrMaxDistance: Double = 18.0
    public var ssrThickness: Double = 0.6
    public var ssrMaxSteps: Int = 48

    // ── TAA ───────────────────────────────────────────────────────────
    // Off by default: the velocity-reprojected resolve ghosts too much on
    // moving geometry. Opt in per-scene via the Illuminatorama settings panel.
    public var taaEnabled: Bool = false
    public var taaHistoryBlend: Double = 0.05
    public var taaJitterPixels: Double = 0.0

    // ── Spatiotemporal denoiser + deband dither ───────────────────────
    public var denoiserEnabled: Bool = true
    public var debandDitherEnabled: Bool = true
    /// Temporal accumulation of the RT diffuse (1-bounce GI) term before its
    /// spatial denoise — velocity-reprojected exponential history that keeps the
    /// GI converging under camera motion (kills the wall "crawl" the main TAA
    /// only resolves when static). Renderer-level; a no-op in non-RT scenes.
    public var rtGITemporalEnabled: Bool = true
    /// Current-frame weight in steady state (≈ 1/window). 0.06 ≈ a 16-frame EMA.
    public var rtGITemporalBlend: Double = 0.06

    // ── IBL ───────────────────────────────────────────────────────────
    public var iblEnabled: Bool = true
    public var iblIntensity: Double = 1.0
    public var dfgLUTEnabled: Bool = true

    // ── DDGI (probe GI) ───────────────────────────────────────────────
    public var ddgiEnabled: Bool = true
    public var ddgiIrradianceScale: Double = 1.0
    public var ddgiHysteresis: Double = 0.99
    public var ddgiTwoBounceEnabled: Bool = false

    // ── Cascaded shadow maps ──────────────────────────────────────────
    public var shadowsEnabled: Bool = true
    public var shadowBias: Double = 0.0003
    public var shadowSlopeBias: Double = 0.0
    public var shadowPcfRadius: Int = 1
    public var shadowMaxDistance: Double = 50.0

    // ── Screen-space contact shadows (issue #65) ──────────────────────
    /// A short screen-space ray march toward the primary directional sun in the
    /// deferred lighting pass, catching the fine contact occlusion the cascaded
    /// shadow maps + RT soft shadows miss at object-base scale (a chip on felt,
    /// an egg on a floor, a prop on a table). OFF by default → an exact shader
    /// no-op. `contactShadowStrength` is the direct-sun attenuation amount (0..1);
    /// `contactShadowLengthCm` the march reach and `contactShadowThicknessCm` the
    /// occluder-depth window — both in CENTIMETRES (scenes are metre-scaled, so
    /// the renderer converts ÷100 to world units); `contactShadowSteps` the
    /// ray-march sample count (8–16 is plenty for the short reach).
    public var contactShadowEnabled: Bool = false
    public var contactShadowStrength: Double = 0.7
    public var contactShadowLengthCm: Double = 5.0
    public var contactShadowThicknessCm: Double = 2.0
    public var contactShadowSteps: Int = 12
    // ── Subsurface scattering (issue #65) ─────────────────────────────
    /// Jimenez-style separable screen-space SSS for skin / wax / marble / food.
    /// OFF by default → an exact pipeline no-op. `sssStrength` blends the blurred
    /// diffuse over the sharp diffuse; `sssRadiusMm` is the diffusion mean-free-
    /// path in MILLIMETRES; `sssTintR/G/B` scale the per-channel scatter distance
    /// (default reddish skin profile → warm bleed; marble ≈ 1,1,1; wax ≈ 1,0.7,0.45).
    /// A surface opts in by tagging its mesh vertices with colour alpha ∈ [0.90,0.98].
    public var sssEnabled: Bool = false
    public var sssStrength: Double = 0.8
    public var sssRadiusMm: Double = 8.0
    public var sssTintR: Double = 1.0
    public var sssTintG: Double = 0.4
    public var sssTintB: Double = 0.25

    // ── Internal render scale (SSAA) ──────────────────────────────────
    public var internalRenderScale: Double = 2.0

    public init() {}

    public func exportText() -> String {
        let fmt = SettingsExportFormat.fmt
        return """
        # Illuminatorama shared (renderer) settings
        # postFXEasing = \(postFXEasing.rawValue)   (enum — informational; Save-defaults skips it)
        exposure = \(fmt(exposure))
        bloomThreshold = \(fmt(bloomThreshold))
        bloomIntensity = \(fmt(bloomIntensity))
        chromaticAberrationEnabled = \(chromaticAberrationEnabled)
        chromaticAberration = \(fmt(chromaticAberration))
        fringeEnabled = \(fringeEnabled)
        fringe = \(fmt(fringe))
        fringeTintR = \(fmt(fringeTintR))
        fringeTintG = \(fmt(fringeTintG))
        fringeTintB = \(fmt(fringeTintB))
        sphericalAberrationEnabled = \(sphericalAberrationEnabled)
        sphericalAberration = \(fmt(sphericalAberration))
        vignetteEnabled = \(vignetteEnabled)
        vignette = \(fmt(vignette))
        vignetteExtent = \(fmt(vignetteExtent))
        filmGrainEnabled = \(filmGrainEnabled)
        filmGrain = \(fmt(filmGrain))
        filmGrainSize = \(fmt(filmGrainSize))
        motionBlurEnabled = \(motionBlurEnabled)
        motionBlur = \(fmt(motionBlur))
        motionBlurMaxPx = \(fmt(motionBlurMaxPx))
        # colorGradeLook = \(colorGradeLook.rawValue)   (enum — informational; Save-defaults skips it)
        colorLUTAmount = \(fmt(colorLUTAmount))
        ssaoIntensity = \(fmt(ssaoIntensity))
        ssaoRadius = \(fmt(ssaoRadius))
        ssrIntensity = \(fmt(ssrIntensity))
        ssrMaxDistance = \(fmt(ssrMaxDistance))
        ssrThickness = \(fmt(ssrThickness))
        ssrMaxSteps = \(ssrMaxSteps)
        taaEnabled = \(taaEnabled)
        taaHistoryBlend = \(fmt(taaHistoryBlend))
        taaJitterPixels = \(fmt(taaJitterPixels))
        denoiserEnabled = \(denoiserEnabled)
        debandDitherEnabled = \(debandDitherEnabled)
        rtGITemporalEnabled = \(rtGITemporalEnabled)
        rtGITemporalBlend = \(fmt(rtGITemporalBlend))
        iblEnabled = \(iblEnabled)
        iblIntensity = \(fmt(iblIntensity))
        dfgLUTEnabled = \(dfgLUTEnabled)
        ddgiEnabled = \(ddgiEnabled)
        ddgiIrradianceScale = \(fmt(ddgiIrradianceScale))
        ddgiHysteresis = \(fmt(ddgiHysteresis))
        ddgiTwoBounceEnabled = \(ddgiTwoBounceEnabled)
        shadowsEnabled = \(shadowsEnabled)
        shadowBias = \(fmt(shadowBias))
        shadowSlopeBias = \(fmt(shadowSlopeBias))
        shadowPcfRadius = \(shadowPcfRadius)
        shadowMaxDistance = \(fmt(shadowMaxDistance))
        contactShadowEnabled = \(contactShadowEnabled)
        contactShadowStrength = \(fmt(contactShadowStrength))
        contactShadowLengthCm = \(fmt(contactShadowLengthCm))
        contactShadowThicknessCm = \(fmt(contactShadowThicknessCm))
        contactShadowSteps = \(contactShadowSteps)
        sssEnabled = \(sssEnabled)
        sssStrength = \(fmt(sssStrength))
        sssRadiusMm = \(fmt(sssRadiusMm))
        sssTintR = \(fmt(sssTintR))
        sssTintG = \(fmt(sssTintG))
        sssTintB = \(fmt(sssTintB))
        internalRenderScale = \(fmt(internalRenderScale))
        """
    }
}

extension IlluminatoramaSharedSettings: DefaultsExportableSettings {
    public static let sourceFilePath: String = #filePath
}
