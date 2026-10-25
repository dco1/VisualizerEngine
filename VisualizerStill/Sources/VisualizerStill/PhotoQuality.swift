/// **The one table that decides how hard a STILL works.**
///
/// `RenderQuality` sizes the *interactive* frame, where the budget is one vsync. A photo has
/// no such budget: nothing is moving, the user has asked for a picture, and a few hundred
/// extra milliseconds are invisible. This is the table for that case — the settings Export
/// Photo turns on that the live canvas cannot afford.
///
/// ## What a still gets, and why only these
///
/// Every candidate was measured on the REAL Metal path (a furnished great room, 10 × 7 m,
/// mid-morning sun) before it was let in or kept out — see
/// `HouseRenderBridgeGPUTests_PhotoQuality`:
///
/// | candidate | measured | verdict |
/// |---|---|---|
/// | supersample 2× | detail resolved rises steeply 1.0 → 1.5 → 2.0 (high-frequency energy 3.92 → 5.41 → 7.97) | **IN** |
/// | supersample 2.5× / 3× | 7.26 / 7.64 — noise around the same plateau, for +22 %/+50 % frame cost | out: 2.0 is the knee |
/// | ray-traced glass | true refraction/reflection in the panes; ~3 ms flat | **IN** — forced on even for the `performance` tier |
/// | screen-space contact shadows | high-frequency energy 7.97 → 8.00, mean luma 159.95 → 159.92 — no visible change | out: the march is toward the PRIMARY SUN only, and the 4096² cascade already owns that shadow at object-base scale |
/// | ray-traced reflections | no visible change | out: `rtReflRoughnessCutoff` is 0.6 and instance roughness is floored at 0.45, so essentially no house material qualifies |
/// | ray-traced GI | re-measured 2026-08-14: auto-disabled on the real document, +4.89 % near-uniform lift, 9.0 % unconverged at 32 frames, ×17.4 the cost — **re-decided 2026-09-06 (S4.4)**: the still builds its TLAS under `rtStillCapsRelaxed` (the glass-arm caps, so a furnished document no longer auto-disables it), and the PhotoRTGI gate that was fail-closed toward convergence FIRED: gain-matched ringing is 2.5 % (junctions) / 2.8 % (falloff) at 32 frames, under the 5 % bar. With the flat interior bands scaled to 0.35 while it is live (so bounce is not counted twice) the picture gains the corner wrap, the niche gradients and the colour bleed a flat fill cannot produce — measured on the gallery still: GI+RTAO 65/50, GI-only 55/40, RTAO-only 40/28 (SCORE/LOOK) | **IN** (`photoGIRequested`, default on; an A/B lever for gates) |
/// | surface-radiance cache — multi-bounce GI (DH-0622) | 4000 Sunset Primary Bedroom, walkthrough, photo lane, auto-exposure on: +15.2 % mean luma as the ceiling, soffit and shaded sides fill with second-hand light (+21.4 % before DH-0849's back-face guard removed light leaking along the creases); frame GPU 600 → 696 ms (+16 %, almost all of it the cache update — RT lighting is unchanged, since a cached hit shades its own sun and lamps exactly as an uncached one does), plus ~1 s of warm-up once per export (`photoSurfaceCacheWarmIterations`; a cold 32-frame still otherwise carried +2.7 % of the light). Past the engine's 60 k-triangle default the cache is skipped, so the still raises the budget (`photoSurfaceCacheMaxTriangles`). Energy not yet judged against a reference render (DH-0642) | **IN for Export Quality ▸ High only** (`PhotoGIQuality.multiBounce`; this rung was named Maximum until 2026-10-01); Standard stays one bounce |
/// | ray-traced ambient occlusion (DH-0528, S4.4) | `illumi_rtao_tlas` in place of the GTAO march when a TLAS is live: 8 cosine rays/px/frame, 0.6 m world reach, same denoise chain; converged by the still's accumulator | **IN** (`rtaoRequested`), photo lane only — a single raw-lane frame is noisier than the march (`testRTAONearField` measures it), which is why it never runs interactively |
/// | ray-traced portal visibility (S4.3) | every shadow-casting area light traced with 3 rays/px/frame toward its rectangle instead of one 150° PCF map — a niche beside the window shadows under its shelves | **IN** whenever the traced sun is on |
/// | TAA jitter accumulation (S4.2) | ringing fell 5.60 % → 0.50 % of pixels once the engine stopped folding the jitter into its motion vectors and stopped sharpening its own accumulator | **IN** — 32 jittered samples on top of the 2× supersample. Full write-up on `jitterPixels` |
/// | ray-traced soft sun shadows (S4.1) | cone-sampled shadow rays at the sun's PHYSICAL angular radius (0.00465 rad) replace the cascade-map visibility in the deferred kernel — a penumbra that widens with occluder distance, which a fixed PCF footprint cannot produce. Phase-2 A/B pending (`s41-softsun-*`) | **IN when the user turns on "Soft sun shadows"; OFF by default** — the default and the softness rung are Danny's dials. Rides the S3.4 big-caps TLAS (`rtTLASForReflections`, the arm proven NOT to auto-disable on the real document or regress RT glass), and only the still's 32-frame accumulator converges the per-frame Monte-Carlo — which is why it is photo-lane only, never interactive. Escapes the RT-GI rejection on all three of its grounds: it runs on the real document (glass-arm caps, not the strict live-RT pair), it is transport the picture can SEE (a shadow edge, not a near-uniform lift), and the accumulator it needs is the one S4.2 already ships |
///
/// ## The RT-GI row, re-decided on new evidence (2026-08-14)
///
/// **Both of the reasons this row originally gave are dead, and it is still a no.** The old
/// text condemned RT GI for "RT hits are UNTEXTURED" — already false when the 2026-08-05 A/B
/// ran, since S3.1's `shadeSecondarySurface` / `secondaryAlbedo` tap the albedo atlas through
/// per-triangle mesh UVs — and for "with TAA and SVGF both off nothing denoises 4 GI rays",
/// which predates S3.1b (the denoise chain left the soup-only branch) and S4.2 (a still now
/// accumulates 32 jittered frames, exactly the accumulator that sentence said was missing).
/// The three causes that actually washed the interiors out are in S3.1b and are all fixed.
///
/// So the A/B was **re-run at photo quality on the Debug Tester rig**
/// (`HouseRenderBridgeGPUTests_PhotoRTGI`, reports and frames in
/// `DaydreamCore/build/render-checks/photortgi-*`), and it came back with three NEW reasons,
/// none of which is the old one:
///
/// 1. **It cannot run on the user's own house.** The engine's live-RT hang guard caps a scene
///    at 128 mesh groups / 150 k triangles. Measured on `4000 Sunset.daydreamhome`:
///    **176 mesh groups, 490 567 instance-expanded triangles → the TLAS goes dark and the
///    renderer reports `.sceneTooHeavy`.** Only the group cap has an override, documented
///    in-engine as an instrument rather than a shipped setting, and the triangle cap has none.
///    A photo setting that silently does nothing on a real document is the inert-feature
///    defect class, not a feature. (The Debug Tester scene itself is 75 groups / 47 560 tris
///    and *does* stay live — which is why the A/B below could be taken at all.)
/// 2. **What it buys is a near-uniform lift, not transport.** The falloff station is one
///    material, one window, probes at 0.5 / 1 / 2 / 4 / 8 m; normalised to each arm's own frame
///    mean the ladder reads 1.2041 / 1.2478 / 1.2916 / 0.8790 / 0.8399 with RT off and
///    1.1883 / 1.2309 / 1.2734 / 0.8806 / 0.8268 at GI 1.0 — **it does not steepen**. Frame
///    mean moves +4.89 % (junctions) / +1.95 % (falloff); 32×32 block SD 21.57 → 21.43 and
///    25.41 → 26.14. The diff frames show the one genuinely correct effect in the set: a smooth
///    lift on the walls, brightest right beside the sun patch on the floor (real floor→wall
///    one-bounce), plus a hairline darkening at the wall/floor arrises. It is real light. It is
///    just not enough light to change the picture.
/// 3. **32 accumulated frames do not converge it.** Two identical GI stills differ by
///    **103 199 px — 9.0 % of the frame** — where two identical RT-off stills differ by
///    5 268 px (0.46 %). With the exposure difference divided out, 34.9 % (junctions) / 14.6 %
///    (falloff) of pixels land outside their own neighbourhood's range in the RT-off still,
///    against a 0.016 % / 0.088 % drift floor. The residual is visible in the diff frames as a
///    floor-wide speckle field.
///
/// And the price: **11.2 s against 0.6 s for the same still, ×17.4.**
///
/// The frames agree with the numbers and are published as a labelled set (scene-ids
/// `rtgi-ab-junctions-A/B/C` and `rtgi-ab-falloff-A/B/C`): no wash-out, no dark blotches —
/// the old failure modes really are gone — and no visible improvement either.
///
/// **What would change this verdict is a SECOND bounce, not a bigger `rtGIStrength`**, which is
/// the same conclusion S3.1b and S3.5 reached from opposite directions. Both gates in
/// `HouseRenderBridgeGPUTests_PhotoRTGI` were therefore fail-closed *toward* the fix — one fires
/// when RT stops auto-disabling on the real document, the other when the still stops being
/// visibly unconverged. **Both fired on 2026-09-06** (S4.4: relaxed still caps, and the ringing
/// measured 2.5 % / 2.8 %), and the row above is the re-decision; the gates now pin the new
/// state (converged, live on the document) as regression guards.
///
/// The rule the table encodes: a photo is the user's design **at higher fidelity**, never a
/// different picture. Nothing here touches the grade, the exposure, the sun or the look —
/// those are the user's, and a photo that came back warmer or brighter than the screen would
/// be a bug, not a feature.
public enum PhotoQuality {

    /// Supersample factor a still aims for. 2× linear = 4 samples per output pixel.
    ///
    /// 2.0 is where the measured returns stop: 2.5× and 3.0× resolve no more detail and cost
    /// 22 % and 50 % more per frame. This number was picked while the still was supersample-
    /// ONLY and stands unchanged now that `jitterPixels` also accumulates — the two compose
    /// (4 spatial samples per output pixel, each of the 32 accumulated frames offset within
    /// its own pixel), and the knee measured here is a property of the spatial ladder.
    public static let targetScale: Float = 2.0

    /// Ceiling on internal pixels for a still, in megapixels.
    ///
    /// The engine allocates its whole internal target set at the internal size — ~194 bytes
    /// per internal pixel (see `RenderQuality`). The shipping `quality` tier already
    /// allocates 8.3 MPix (≈1.6 GB) on the reference 2560 × 1440 display, so this cap sits
    /// modestly above what is already proven, and only binds on displays large enough that
    /// a flat 2× would ask for several gigabytes.
    ///
    /// **Raised 12.0 → 24.0 alongside `outputSize` (DH-0378, 2026-08-27).** Once a still's
    /// OUTPUT is bumped past the canvas (see `outputSize`/`targetLongEdge`), the old cap binds
    /// much harder: on Danny's measured window (2560×1316, DH-0378) the output alone grows
    /// 3.37 → 7.58 MPix, and 12.0 would drop the supersample from the previously-measured
    /// ~1.9× to 1.26× — undoing most of the sampling quality this table already earned. 24.0
    /// keeps that same canvas at 1.78×, inside a few percent of the old number, for
    /// ≈+2.3 GB (≈4.66 GB total) of ONE transient capture-time allocation — not something
    /// paid every frame, or even every export at full size (`scale` still floors at the same
    /// 12.0-cap behaviour did: reduce the multiplier, never refuse the photo).
    public static let maxInternalMegapixels: Double = 24.0

    /// The internal render scale a still should use for a given output size: `targetScale`,
    /// reduced only far enough to respect `maxInternalMegapixels`, and never below 1.0
    /// (a photo must never be softer than the screen it was taken from). Clamped to the
    /// engine's own `internalRenderScale` range by the property itself.
    ///
    /// Pure — a test can check the whole policy with no GPU at all.
    public static func scale(outputPixels: Double) -> Float {
        guard outputPixels > 0 else { return targetScale }
        let capPixels = maxInternalMegapixels * 1_000_000
        let capScale = (capPixels / outputPixels).squareRoot()
        return Float(max(1.0, min(Double(targetScale), capScale)))
    }

    // MARK: spot-shadow slices (DH-0134)

    /// How many spot cones may carry a shadow map in a STILL.
    ///
    /// The live canvas shadow-maps at most `ShadowSlotBudget.engineSpotShadowCapacity` (8) cones a
    /// frame — each slice is a full-scene depth pass (~0.5 ms), so on a vsync budget the cap is
    /// real, and a cone past it renders unshadowed and leaks through the wall it stands against.
    /// A still has no such budget (Danny, 2026-08-31: the highest-quality preset renders one frame,
    /// so it should have NO cap on shadowed spots), so the photo lane asks the engine for every
    /// slice it can allocate and every casting cone in the document is shadowed. This mirrors the
    /// engine's `IlluminatoramaRenderer.spotShadowCapacityHardMax`; the two modules cannot see each
    /// other, so `HouseRenderBridgeGPUTests` asserts the mirror. The engine grows its atlas lazily
    /// toward the request (only as far as the casters need) and shrinks back when the request drops,
    /// so the live lane pays nothing for this number existing.
    public static let spotShadowCapacity = 64

    // MARK: output resolution (DH-0378)

    /// Long edge, in pixels, a still's OUTPUT targets — independent of `scale`, which only
    /// governs internal SAMPLING. "4K" by convention (UHD's horizontal edge).
    ///
    /// **Export Photo used to read back the canvas at whatever size the app window happened
    /// to be** (DH-0378) — measured on a real photo-lane frame, the AA quality itself was
    /// already excellent (a clean edge deviated 0.084 px RMS from straight over 75 rows, and
    /// 99.9 % of strong edges carried at least one intermediate pixel), but the window was
    /// only 2560×1316 = 3.4 MPix, which is not many pixels for a picture someone crops into
    /// or prints. Danny, 2026-08-27: bump the default, no picker — every still targets at
    /// least this long edge, silently, with no export-panel control.
    public static let targetLongEdge = 3840

    /// The output size a still should render at, given the live canvas it is captured from:
    /// `targetLongEdge` on the canvas's own longer axis, the shorter axis scaled by the same
    /// factor so the aspect ratio is preserved exactly, and never SMALLER than the canvas
    /// itself — the same floor `scale` enforces for supersample (a photo must never come back
    /// with fewer pixels than the screen already had). Degenerate input passes through
    /// unchanged rather than producing a zero or negative size. Pure — a test can check the
    /// whole policy with no GPU.
    public static func outputSize(canvasWidth: Int, canvasHeight: Int) -> (width: Int, height: Int) {
        guard canvasWidth > 0, canvasHeight > 0 else { return (canvasWidth, canvasHeight) }
        let longEdge = Double(max(canvasWidth, canvasHeight))
        let factor = max(1.0, Double(targetLongEdge) / longEdge)
        let w = Int((Double(canvasWidth) * factor).rounded())
        let h = Int((Double(canvasHeight) * factor).rounded())
        return (max(1, w), max(1, h))
    }

    /// Frames to render before reading the photo back.
    ///
    /// Raising the internal scale RESIZES the engine's targets, and a resize destroys every
    /// history texture — `taaNeedsFirstFrame`, `aoNeedsFirstFrame`, `ssrNeedsFirstFrame` and
    /// `rtGINeedsFirstFrame` all go true, so SSAO restarts its temporal accumulation from
    /// nothing (blend 0.90 ⇒ ~20–30 frames to converge). The old 8-frame settle was sized for
    /// a capture that changed no render settings at all; at photo quality it would read a
    /// half-accumulated AO term.
    ///
    /// A resize does **not** re-seed the auto-exposure EMA (only `resetTemporalHistory()`
    /// does), so the photo keeps the exposure the screen had settled at — verified: mean luma
    /// moved 160.15 → 159.95 across the whole 1.0 → 3.0 scale ladder.
    public static let settleFrames = 32

    // MARK: the angle sheet's settle

    /// Frames to settle **per angle** inside a photo-quality sweep (`captureAngles`).
    ///
    /// A camera cut is not a resize — the history textures survive it — but every temporal
    /// accumulator (SSAO, SSR, TAA) reprojects through the velocity buffer, and a jump to a
    /// completely different yaw disoccludes essentially the whole frame. What comes back is
    /// therefore a *cold* AO term again, and the old 4-frame per-angle warm-up read it
    /// half-accumulated. This is the same reason `settleFrames` is 32 for a single still.
    ///
    /// The one thing the sweep must NOT do per angle is re-enter photo quality: the resize is
    /// what costs the ~44 ms reallocation and destroys the histories outright, so
    /// `captureAngles` scales ONCE for the whole sweep and then only pays this per shot.
    ///
    /// **Measured** (`HouseRenderBridgeGPUTests_PhotoSheetQuality`, each settle run from an
    /// identical cold start so no sweep inherits the previous one's warm history — mean |Δ|
    /// per channel against a 200-frame settle of the same angle):
    ///
    /// | per-angle settle | 4 | 12 | 24 | 48 | 96 |
    /// |---|---|---|---|---|---|
    /// | residual, interactive exposure EMA | 4.16 | 3.30 | 2.34 | 1.16 | 0.23 |
    /// | …of which is a flat mean-luma offset | 3.52 | 2.80 | 1.99 | 0.99 | 0.20 |
    /// | residual, EMA snapped (shipped) | 1.12 | 0.048 | **0.004** | 0.002 | 0.000 |
    ///
    /// The second row is the finding, and it was not what S4.3 was written down as being about:
    /// what a camera cut leaves unconverged is **overwhelmingly the auto-exposure EMA**, not
    /// the AO — 85 % of the residual at every settle length is a flat brightness offset. A
    /// top-down aerial and a ¾ corner see very different amounts of sky, so the estimator
    /// genuinely re-meters, and at the interactive half-life that takes ~100 frames.
    /// `sweepExposureHalfLife` collapses it (third row), which is what lets this number stay at
    /// 24 — enough for the AO alone — instead of ~96 per angle across nine angles.
    ///
    /// ⚠️ **THE TABLE ABOVE PREDATES TEMPORAL ACCUMULATION, measured 2026-08-23.** It was recorded
    /// in `b559346` (2026-08-09, "S4.2 jitter measured, blocked, backed out") — on a build that
    /// emitted DETERMINISTIC frames. Four days later `ef5b609` turned photo-quality accumulation
    /// ON (`jitterPixels`, `accumulationBlend = 1/32`), and it reaches the sweep, because
    /// `stillAccumulates` derives from `capturingPhoto`. Nobody re-took the ladder.
    ///
    /// So the "24 → 0.004" row no longer describes this renderer. **24 frames at a 1/32 blend is
    /// 1 − (31/32)²⁴ = 53 % converged**, against a 200-frame reference at 99.8 %, and the residual
    /// measures **0.197 (σ 0.004 over 16 runs, no trend)**. That residual is not the exposure DC
    /// the third row is about — the EMA snap still works, collapsing a 1.50-level offset to ~0.08.
    /// It is the un-averaged half of the jitter sequence: 83.8 % of channel samples are
    /// bit-identical to the reference and 76 % of the rest are ±1 LSB, concentrated on hard edges
    /// (0.8 % of pixels carrying 12 % of the residual at 3.17 levels) — exactly where sub-pixel
    /// accumulation pays off.
    ///
    /// **Raised 24 → 96, Danny's call 2026-08-31 (DH-0204).** The standing policy: when the only
    /// cost of a quality improvement is render time/frames, pay it in Photo Export — never on the
    /// live canvas. `angleSettleFrames` runs ONLY during angle-sheet photo export, so there is no
    /// interactive-fps cost to weigh, and the sheet should get the best the lane can settle to.
    ///
    /// Post-RT-glass the lane carries a Monte-Carlo term that converges ~1/√N toward a nonzero
    /// floor, so the convergence curve is (DH-0381, measured 2026-08-24, vs a deep settle):
    ///
    /// | per-angle settle | 4 | 24 | 48 | 96 | 200 (floor) |
    /// |---|---|---|---|---|---|
    /// | residual, levels | 1.51 | 0.175 | 0.137 | **0.100** | 0.041 |
    ///
    /// On that machine 96 is where the sheet reaches the `< 0.10` bar the gate wanted, so 96 is
    /// where it settles. The ~4× export-time cost across the sheet's angles is expected and
    /// accepted; export speed is not the metric being optimised.
    ///
    /// (⚠️ the ~0.041 floor above is machine-dependent: on the maintenance host, 2026-08-31, two
    /// independent 200-frame settles disagree by ~0.35 — an ~8× higher Monte-Carlo floor — so
    /// `< 0.10` is not reachable there at any settle count. The gate stays floor-relative for
    /// that reason; see `testAngleSheetIsPhotoQualityAndItsFramesAreConverged` and DH-0204.)
    public static let angleSettleFrames = 96

    /// Auto-exposure EMA half-life, in seconds, while an angle SWEEP settles.
    ///
    /// The 0.25 s the engine ships is a smoothing constant for a live camera: it exists so a
    /// pan across a window doesn't pump the whole picture. A batch of stills has no such
    /// requirement — nobody sees the intermediate frames — and paying ~100 frames per angle to
    /// let a smoothing filter catch up is pure waste. Snapping it makes each shot land on the
    /// exposure that angle actually meters to, which also makes the sheet **order-independent**:
    /// with the interactive half-life, a shot's exposure depends on which angle preceded it.
    ///
    /// Deliberately NOT applied to a single Export Photo: that still must keep the exposure the
    /// screen had settled at (`settleFrames` documents why, and a GPU test asserts it to within
    /// a few percent). The difference is real — the sheet visits angles the screen never had.
    public static let sweepExposureHalfLife: Float = 0.02

    // MARK: temporal accumulation (S4.2)

    /// Sub-pixel Halton(2,3) jitter for a still, in pixels — the engine's `taaJitterPixels`.
    ///
    /// **Turned on (at 0.5) once the engine landed the two fixes this was blocked on.** A still has no
    /// vsync budget, so it ought to converge like an offline renderer, and the engine has the
    /// machinery (Halton jitter + a TAA resolve). Turning it on used to make the picture
    /// WORSE, for two reasons that were both inside the renderer and are both now fixed:
    ///
    /// 1. **The motion vectors carried the jitter, so the jitter was TRACKED, not accumulated.**
    ///    Velocity was `currNDC − prevNDC` from two JITTERED matrices, so the resolve
    ///    reprojected the history by exactly the jitter delta and realigned it onto the
    ///    *current* sub-pixel sample — successive frames never contributed a *different*
    ///    sample to a fixed output pixel, which is the whole mechanism of jitter supersampling.
    ///    The engine now subtracts the frame-to-frame jitter delta back out
    ///    (`FrameUniforms.taaJitterDelta`), which is the standard formulation.
    /// 2. **The resolve's acutance sharpen compounded**, because the sharpened value WAS the
    ///    next frame's history. It is now applied only to the frame handed downstream; the
    ///    accumulator keeps the unsharpened blend.
    ///
    /// **Measured, same scene and rig as before** (1100 × 780 at 2.00× internal, 32 settle
    /// frames, so the only variable is the temporal pass). The metric is pixels landing
    /// OUTSIDE their own neighbourhood's range in the un-accumulated render — better
    /// anti-aliasing can only move a pixel *toward* the middle of its neighbourhood, so
    /// anything outside is ringing or a firefly:
    ///
    /// | still policy | before the engine fix | after |
    /// |---|---|---|
    /// | shipped, re-run (drift floor) | 0.000 % | 0.000 % |
    /// | + TAA accumulation, jitter 0.5 px | **5.60 %** | **0.50 %** |
    ///
    /// An 11× drop, against a renderer that is otherwise dead still. The frames agree:
    /// `photoq-accum-jitter05.png` next to `photoq-accum-shipped.png` shows cleaner plank
    /// lines, pot rims and chair arms with none of the halo/speckle the old build had.
    ///
    /// **1.0 since DH-0733 (2026-09-18) — the value is the jitter's full WIDTH, not a radius.**
    /// The engine offsets each frame by `(halton − 0.5) × taaJitterPixels` internal pixels, so
    /// 0.5 spread the 32 samples over only the middle ±¼ px of each internal pixel: a quarter of
    /// its area, the corners never sampled. Nothing above could see that — every gate compared
    /// arms with EACH OTHER. `testStillAccumulationConvergesTowardAGroundTruthSupersample`
    /// compares them with the ANSWER (the same frame at 3× the size on the supersample-only
    /// lane, box-filtered: 36 samples per output pixel, no temporal pass), and edge RMSE
    /// against it, in 8-bit luma, reads:
    ///
    /// | still policy | edges | whole frame |
    /// |---|---|---|
    /// | supersample only (and its re-run) | 4.878 / 4.866 | 2.346 |
    /// | TAA, jitter 0 | 5.114 | 2.287 |
    /// | TAA, jitter 0.5 (the old shipped value) | 4.42–4.52 | 2.11–2.18 |
    /// | **TAA, jitter 1.0** | **4.059** | **2.030** |
    /// | TAA, jitter 1.5 | 4.012 | 2.041 |
    ///
    /// So the still WAS supersampling, at half strength; 1.0 (±½ px — the "standard" span the
    /// engine's own doc means) roughly doubles what the accumulation buys at edges, and 1.5
    /// buys nothing more while flat surfaces start to soften. The no-jitter resolve is WORSE
    /// than no temporal pass at edges — the resolve's downstream acutance sharpen, applied to
    /// one frame's samples on the way out — which is why jitter, not the resolve, is the lever.
    public static let jitterPixels: Float = 1.0

    /// Current-frame weight for the still's TAA resolve (the engine's `taaHistoryBlend`, whose
    /// name is a misnomer — it is the CURRENT sample's weight). 1/32 is the matching weight for
    /// a `settleFrames`-long accumulation. Reached today only through the A/B lever
    /// (`HouseRenderBridge.photoJitterOverride`), since the shipped policy accumulates nothing.
    public static let accumulationBlend: Float = 1.0 / 32.0

    /// The still's blend for its k-th accumulated frame: a CUMULATIVE AVERAGE, `1/k`, down to
    /// the resolve kernel's own floor. A fixed 1/32 is an EMA — after 96 frames its effective
    /// sample count is ~57 and the seed frame still carries 5 %; `1/k` weights all 96 equally
    /// (effective count 96, seed 1 %) for the same 96 renders. The floor is the TAA kernel's
    /// `clamp(alpha, 0.01, 1)`, so past 100 frames the average becomes a long EMA rather than
    /// a divide the kernel would refuse. The live canvas keeps `accumulationBlend`: its
    /// converge-and-hold (DH-0064) is calibrated on the fixed blend.
    public static let accumulationFloor: Float = 0.01
    public static func accumulationBlend(frame k: Int) -> Float {
        StillAccumulation.blend(frame: k, floor: accumulationFloor)
    }

    /// Whether a still turns TAA accumulation on at all. Derived from `jitterPixels` (never a
    /// second hand-typed switch): accumulation without jitter averages identical frames, so the
    /// resolve can only ever subtract from them — measured above as the WORST of the three.
    public static var accumulates: Bool { jitterPixels > 0 }
}
