import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - ShadowKnobs

/// Stored state for the renderer's `shadow` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `ShadowKnobs` (`kShadow`) and forwards each name to it from
/// `IlluminatoramaRenderer+ShadowKnobs.swift`; callers see no difference.
@MainActor
final class ShadowKnobs {
    typealias MeshKind = IlluminatoramaRenderer.MeshKind
    typealias ShadowRasterBias = IlluminatoramaRenderer.ShadowRasterBias

    /// A/B lever for the per-slice reuse above (default on). Off = every slice re-renders whenever
    /// GPU-fed geometry is live, the pre-DH-0631 behaviour.
    var spotShadowGPUGeometryOptOutEnabled: Bool = true
    /// Daydream DH-1064 — A/B lever for STICKY spot-atlas slices (default on). On, a caster whose
    /// light matrix already sits in some slice keeps that slice when the set of casters changes,
    /// so a membership change re-renders only the slices that genuinely changed. Off = slices are
    /// handed out compactly in caster order every frame (the pre-DH-1064 rule), where one portal
    /// joining the set shifted every later slice and re-rendered them all.
    var spotShadowStickySlicesEnabled: Bool = true
    /// Daydream DH-1064 — test lever: false re-renders every spot/area slice every frame (no
    /// static reuse at all), so a gate can diff a reused atlas against a freshly rasterised one.
    var spotShadowReuseEnabled: Bool = true
    /// Daydream DH-1590 — A/B lever for PER-SLICE caster dirtying (default on). On, a caster whose
    /// shadow shape changed (moved, swapped mesh, re-swung) re-renders only the slices whose light
    /// frustum its old or new bounds reach. Off = any shape change re-renders every slice.
    var spotShadowPerSliceDirtyEnabled: Bool = true
    // ── Phase 2.5 shadow knobs ───────────────────────────────────────
    /// Toggle the cascaded-shadow pass + sampling. Off = direct sun lights
    /// every pixel, regardless of occlusion (Phase 2 behaviour).
    var shadowsEnabled: Bool = true
    /// Mesh kinds excluded from the DIRECTIONAL (cascade/sun) shadow pass: their
    /// geometry is not drawn as a sun-shadow occluder, so it casts no shadow from
    /// the sun. Empty by default → no behaviour change. The host opts a kind in
    /// when a self-illuminated object shouldn't blot its own light pool with a
    /// hard sun shadow (e.g. a lamp: its body casting a crisp shadow across the
    /// circular pool it throws reads as a lopsided pool). Spot-cone shadows
    /// (`encodeSpotShadowPasses`) are unaffected — the object still traps its own
    /// light; only the sun stops shadowing it.
    var directionalShadowExcludedKinds: Set<MeshKind> = []
    /// Mesh kinds drawn ONLY into the shadow maps (sun cascades, spot atlas, point
    /// cubes) and the RT TLAS — never the G-buffer. Empty by default → no behaviour
    /// change. A "lighting-only" occluder: the host registers geometry that must
    /// BLOCK light without being visible — e.g. a dollhouse ceiling that keeps the
    /// noon sun out of interior rooms while the camera still looks freely down into
    /// them. The instances are uploaded normally (they need slots in the instance
    /// buffer for the shadow passes); only the G-buffer raster skips them, so they
    /// write no albedo/depth/velocity/layer and are invisible + unpickable on screen.
    var shadowOnlyMeshKinds: Set<MeshKind> = []
    // DEFAULTS LESSON (PR #33 fix): the initial Phase 2.5 defaults were
    // shadowBias = 0.0008 and shadowSlopeBias = 0.005 — both too aggressive.
    // The shadow pass uses front-face culling, which already shifts stored
    // depths onto the back of casters and naturally separates lit receivers
    // from self-shadowing. Stacking a generous bias on top lifted shadows
    // off their contact line ("peter-panning") and the user caught it
    // immediately. When adding similar bias/tolerance defaults elsewhere
    // (e.g. SDF GI thickness, voxel cone tracing self-occlusion), default
    // to "off / minimum that does something" — let the user dial up.
    /// Constant depth bias subtracted from light-NDC z before the PCF compare.
    /// Tiny by default because the shadow pass uses front-face culling, which
    /// stores the back-face depth of casters and already separates lit
    /// receivers from self-shadowing — so we only need a hair of slop to
    /// soak up floating-point noise. Raise this only if you see acne.
    var shadowBias: Float = 0.0003
    /// Bias added in proportion to `(1 - NdotL_sun)`. Defaults to zero
    /// because the front-face-culling bias above handles grazing surfaces
    /// too. Raise this only if grazing-angle acne shows up that the constant
    /// bias can't fix — too much causes "peter-panning" (shadows visibly
    /// detached from their caster's contact line with the receiver).
    var shadowSlopeBias: Float = 0.0
    /// **TEST LEVER — production must leave this false.** Makes the three shadow passes
    /// pick their cull mode from `IlluminatoramaMesh.doubleSided` instead of the narrow
    /// `shadowCastsBothFaces` opt-in. This is the "obvious" version of the open-shell
    /// shadow fix, and it exists only so an A/B can MEASURE what it costs: `doubleSided`
    /// is a shading property that closed solids set freely (Daydream's walls, floors,
    /// millwork and stairs all do), so routing the shadow cull through it removes the
    /// second-depth acne defence from the entire structural shell. Keep it reachable so
    /// the acne is a number in a gate rather than a claim in a comment.
    var shadowCullFollowsDoubleSidedForTest: Bool = false
    /// **Which face of a closed solid writes the sun's shadow depth.**
    ///
    /// `false` (default) culls FRONT faces, so the map stores each occluder's BACK face —
    /// second-depth shadow mapping, which is a strong acne defence because the stored depth sits
    /// a whole solid's thickness behind any lit surface. Its price is at CONTACTS: where a
    /// receiver touches the occluder's back face, stored and receiver depth are equal, the
    /// comparison ties, and the tie resolves to LIT. A floor meeting a wall is exactly that
    /// geometry, so the sun draws a bright line along the junction and the PCF kernel spreads it
    /// into a band (Danny, 2026-08-18/19, the Pantry wall-foot leak).
    ///
    /// `true` stores the FRONT face instead. At a wall foot the stored depth is then the wall's
    /// OUTER face — a full wall thickness in front of the receiver — so the junction is
    /// unambiguously shadowed and no PCF tap inside the wall's footprint can escape. Acne is
    /// then carried by `ShadowRasterBias` — the slope-scaled raster bias below, which this
    /// property selects automatically — plus the normal-offset bias in `cascadeVisibility`.
    var shadowStoresFrontFaces: Bool = false
    /// Overrides the profile `shadowStoresFrontFaces` would otherwise pick. `nil` derives it, so
    /// a host that flips the convention gets the matching acne defence without a second call —
    /// the two are one decision. Set it to sweep values in a gate.
    var shadowRasterBiasOverride: ShadowRasterBias?
    /// Mesh kinds that are **substrate** — the ground a street, a driveway or a court line is
    /// laid flush on, whose separation from them is millimetres of deliberate lift rather than
    /// real space. These are biased AWAY from the camera so that anything resting on them wins
    /// the depth test at any distance.
    ///
    /// **Why the substrate recedes instead of the decal advancing.** Both close the same gap,
    /// and the gap is real: the camera projects into a NON-reversed `depth32Float` buffer where
    /// the smallest resolvable separation grows with the square of the distance —
    /// `≈ ulp(1.0) · z² / zNear`, which at `zNear = 0.05` is 0.5 mm at 20 m but **7.6 mm at
    /// 80 m and 17 mm at 120 m**. Any fixed lift therefore has a range past which the surface
    /// underneath starts winning pixels and the decal dithers into it. Measured in Daydream
    /// Home, 2026-08-28: a street with a 10 mm lift is clean to ~90 m and eaten by dirt beyond.
    ///
    /// Pushing the DECAL forward closes it too, but it spends the whole bias in the direction
    /// where something legitimately sits above the decal — a pavement stands on a 150 mm kerb,
    /// and a bias big enough to fix the road at 150 m is big enough to lift the road over that
    /// kerb. Pushing the SUBSTRATE back spends it in the direction where nothing is: there is,
    /// by definition, nothing under the ground. So the bias can be generous without inverting
    /// anything, and every surface laid on the ground is fixed at once rather than one by one.
    ///
    /// Empty by default — a host that declares no substrate gets byte-identical frames.
    var substrateMeshKinds: Set<MeshKind> = []
    /// How far the substrate is pushed away from the camera, in depth-buffer ULPs (positive =
    /// further, since this buffer is not reversed).
    ///
    /// **Why 32.** Swept on Daydream Home's flat-lot street at a fixed camera, measuring the
    /// share of two windows lying wholly ON the roadway — one a few tens of metres out, one
    /// past where a 10 mm lift stops resolving — that read as bare ground rather than asphalt:
    ///
    /// | bias | near roadway eaten | far roadway eaten |
    /// |---|---|---|
    /// | 0 | 0.06 | **0.77** |
    /// | 16 | 0.00 | 0.00 |
    /// | 64 | 0.00 | 0.00 |
    /// | 128 | 0.00 | 0.00 |
    ///
    /// 16 already clears it completely; 32 is one doubling of margin for scenes whose substrate
    /// sits at a different depth range. It is not pushed higher for free: the G-buffer depth is
    /// what SSAO and contact shadows sample, and a bias is a real error in that texture — 32
    /// ULPs is ~1.5 cm of world-equivalent depth error on ground 20 m away and ~38 cm at 100 m,
    /// which is under the AO radius near the camera and irrelevant at the far end, where ground
    /// is a handful of pixels. Doubling again would start eating into the near figure.
    var substrateDepthBias: Float = 32
    /// PCF kernel radius in shadow-map texels: 0 = single tap, 1 = 3×3, 2 = 5×5.
    var shadowPcfRadius: UInt32 = 1
    /// How far from the camera the outermost cascade extends, in metres. Past
    /// this distance, surfaces are treated as lit (no shadows). Cascades are
    /// distributed in [zNear, shadowMaxDistance] using a practical
    /// log/uniform split.
    var shadowMaxDistance: Float = 50.0
    // ── Phase 4.10 — spot light shadow maps ──────────────────────────
    /// Toggle per-spot shadow casting. Off by default — scenes with
    /// dozens of spot lights (Eggs' rail spotlights) would otherwise pay
    /// N additional depth render passes per frame. Flip on per-scene
    /// when the visual gain justifies the cost.
    var spotShadowsEnabled: Bool = false
    /// Constant depth bias subtracted from the spot light-space depth
    /// before the PCF compare. Combats acne on lit surfaces.
    var spotShadowBias: Float = 0.0005
    // ── Point-light cubemap shadows (opt-in, additive) ────────────────
    /// Master enable for the point-light depth-cubemap shadow pass. Default
    /// OFF — no shadow textures are allocated, no depth is rendered, and the
    /// lighting kernel takes the exact pre-change point-light path, so every
    /// scene that never flips this (Visualizer) is byte-identical. Turn on
    /// per-scene (Daydream) and additionally set `castsShadow` on the point
    /// lights you want shadowed (default off per light too).
    ///
    /// Mechanism: for each shadow-casting point light the renderer renders SIX
    /// 90°-perspective depth maps (the faces of a cube centred on the bulb) into
    /// a shared depth atlas, then the deferred lighting kernel projects each
    /// fragment through the dominant cube face and PCF-compares — so the bulb is
    /// blocked by geometry, PASSES through openings (doorways), and casts
    /// intra-room object shadows. Because it's a REAL depth occlusion, a shadowed
    /// light no longer needs (and must not use) the cheap position-based room
    /// layer mask — see `pointShadowSupersedesLayerMask`.
    var pointShadowsEnabled: Bool = false
    /// Constant depth bias subtracted from the point light-space depth before the
    /// PCF compare (same role as `spotShadowBias`). Combats acne on lit surfaces.
    var pointShadowBias: Float = 0.0006
    /// Screen-space contact shadows (issue #65). A short screen-space ray march
    /// toward the PRIMARY directional sun in the deferred lighting pass, catching
    /// the fine contact occlusion the cascaded shadow maps + RT soft shadows miss
    /// at object-base scale (a chip on felt, an egg on a floor). 0 strength =
    /// OFF → an exact shader no-op (the march is skipped). `contactShadowLength`
    /// is the march reach in WORLD units; `contactShadowSteps` the sample count;
    /// `contactShadowThickness` the occluder-depth window (acne guard). Driven
    /// from the panel via `applySharedLensFX` (universal off-by-default knob).
    var contactShadowStrength: Float = 0
    var contactShadowLength: Float = 0.05
    var contactShadowSteps: UInt32 = 12
    var contactShadowThickness: Float = 0.02
    var spotLightProfileRowCount: Int = 0
}
