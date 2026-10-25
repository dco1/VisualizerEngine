import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - GlassKnobs

/// Stored state for the renderer's `glass` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `GlassKnobs` (`kGlass`) and forwards each name to it from
/// `IlluminatoramaRenderer+GlassKnobs.swift`; callers see no difference.
@MainActor
final class GlassKnobs {
    typealias GlassMeshGroup = IlluminatoramaRenderer.GlassMeshGroup
    typealias MeshKind = IlluminatoramaRenderer.MeshKind

    // ── AAA ray-traced glass (#60) ────────────────────────────────────
    /// Host sets these to render glass. `glassPaneKind` is a registered mesh and
    /// `glassPaneInstance` is one glass instance (transform + material). Nil → no
    /// glass. When ray tracing is available the glass is rendered by
    /// `illumi_glass_rt_fs` — true entry+exit refraction, reflection, Beer–Lambert
    /// absorption, and dispersion traced against the scene TLAS (the glass meshes
    /// are added to that TLAS so glass refracts/reflects real geometry and other
    /// glass). Without RT it falls back to `illumi_glass_fallback_fs` (single-
    /// surface Fresnel + sky). Drawn into the HDR composite after lighting,
    /// depth-tested but not written, so opaque geometry in front occludes it.
    var glassPaneKind: MeshKind? = nil
    var glassPaneInstance: IlluminatoramaGlassInstance? = nil
    /// Multiple glass instances drawn in ONE instanced draw, all sharing
    /// `glassPaneKind`. Takes precedence over `glassPaneInstance`. Used by Floating
    /// Lenses for an orbit of tumbling concave lenses. Sort back-to-front (far →
    /// near): the pass doesn't write depth, so draw order resolves overlaps (the RT
    /// path returns opaque alpha, so the nearest pane — drawn last — wins, and its
    /// trace already composited everything behind it).
    var glassInstances: [IlluminatoramaGlassInstance] = []
    var glassMeshGroups: [GlassMeshGroup] = []
    /// Sample the albedo atlas at the hit's barycentric UV. Off ⇒ per-instance mean albedo.
    var glassHitTexturingEnabled: Bool = true
    /// Accumulate point + spot lights at the hit. Off ⇒ sun + sky only, so a lamp-lit
    /// room goes dark through a window.
    var glassHitLocalLightsEnabled: Bool = true
    /// Apply the host's interior IBL/ambient separation at the hit. Off ⇒ an interior
    /// surface gets exterior-strength fill, several times short.
    var glassHitInteriorFillEnabled: Bool = true
    /// DH-0718 — light every secondary hit (a reflection, a GI bounce, the world through a
    /// pane) with the frame's rectangular AREA lights too, by the same form factor the
    /// deferred kernel uses. Off ⇒ the pre-DH-0718 behaviour: a secondary hit saw sun + sky
    /// + point/spot lights but no area light — which in Daydream Home is every window
    /// portal, i.e. a room's dominant daytime source, so a reflected room was lit by a
    /// different light than the room itself and cast none of that light's shadows.
    var secondaryAreaLightsEnabled: Bool = false   // DH-0718 WIP: default flips to true when it lands
    /// DH-0718 — visibility rays per shadow-casting area light at a secondary hit (toward
    /// jittered points on the rectangle, like `rtAreaShadowRays` on the primary). 0 ⇒ lit
    /// but unoccluded. 1 by default: an accumulating still converges it like the sun cone.
    var secondaryAreaShadowRays: Int = 1
    /// CHEAP semi-transparent glass (no TLAS, no per-frank BLAS). Off (0) keeps the
    /// plain Fresnel+sky fallback unchanged for every other scene. A host opts in
    /// when it wants "semi-transparent glass" rather than "hardware-RT refraction"
    /// and can't afford a BLAS-per-deforming-body TLAS (Hot Dog Press: #64). Drawn
    /// through `illumi_glass_fallback_fs`, gated on `u.cheapGlassMode`:
    ///   1 = FRESNEL  — synthetic tinted translucent sheet + Fresnel edge + a key
    ///                  specular glint, alpha-blended over the lit scene. Cheapest;
    ///                  the panes read as a coloured glass case but don't lens/tint
    ///                  the bodies behind multiplicatively (no scene read).
    ///   2 = SCREEN-SPACE — samples the pre-glass composite behind the pane
    ///                  (`glassBackdropTexture`), Beer–Lambert-tints + screen-space-
    ///                  refracts it, adds Fresnel + key glint. True "look through
    ///                  tinted glass" read; one extra full-frame blit + one texture
    ///                  sample per glass fragment, still no ray tracing.
    /// Only consulted on the NON-RT path (`rtGlassEnabled == false`); RT glass wins.
    /// Mode 2 reads the per-pane screen-space refraction strength (pixels at grazing)
    /// from each glass instance's `rdrf.w` (the reserved "fresnelPower" slot).
    var glassCheapMode: Int = 0
    /// THIN-FILM IRIDESCENCE for cheap-glass mode 2 (soap bubbles — Bubble Lab).
    /// Master strength of the interference coat added on top of the screen-space
    /// refraction. 0 (default) → an EXACT no-op, so every other cheap-glass scene
    /// is unaffected. The coat colour is computed in-shader from the optical path
    /// difference `2·n·d·cosθ` at R/G/B wavelengths — real two-beam interference,
    /// not a scrolled texture. Per-bubble thickness variation rides in each glass
    /// instance's `dispersionPad.y`.
    var thinFilmStrength: Float = 0
    /// Base soap-film thickness in nanometres at the equator (interference order).
    var thinFilmThicknessNm: Float = 320
    /// Refractive index of the film itself (water + surfactant ≈ 1.33).
    var thinFilmIOR: Float = 1.33
    /// OSCILLATION-MODE surface undulation for the bubble shells (the glass VS
    /// displaces each vertex radially by a sum of beating angular modes — real
    /// per-vertex lobing an affine transform can't produce). 0 (default) → no
    /// displacement, so every other glass scene is unaffected. Radial amplitude
    /// in object-space units (the unit sphere has r = 1).
    var bubbleWobbleAmp: Float = 0
    /// Global rate multiplier for the beating oscillation modes.
    var bubbleWobbleFreq: Float = 1
    // ── Glass caustics (#60) ───────────────────────────────────────────
    /// Photon-traced caustics for the RT glass, splatted into the surface-cache
    /// card atlas (so they land on ANY receiver) and read back per primary pixel.
    /// Opt-in AND requires the surface cache on (that's the card store). Off → no
    /// cost. See `docs/illuminatorama/glass.md`.
    var causticsEnabled: Bool = false
    /// Photons traced per frame. Energy accumulates across frames via `causticDecay`,
    /// so fewer photons/frame just converge a touch slower — 400k is a good
    /// perf/quality balance (1M was ~3× the cost for marginal gain).
    var causticPhotonCount: Int = 200_000
    /// Composite strength of the caustic term added to the lit receiver.
    var causticStrength: Float = 1.0
    /// Per-frame EMA decay of the caustic atlas. Higher = more frames accumulated
    /// = less photon speckle (at the cost of slower response to motion). 0.94 ≈ a
    /// ~17-frame running average — smooth pools on a static-ish scene.
    var causticDecay: Float = 0.94
    /// Per-photon flux scale (normalisation). Tuned for ~400k photons over a small
    /// scene; scale inversely if you change `causticPhotonCount` a lot (brightness
    /// ≈ energy × count). The receiver store is the surface-cache card atlas, so
    /// caustics land at GI-card resolution — focused pools, not razor filaments.
    var causticPhotonEnergy: Float = 0.06
}
