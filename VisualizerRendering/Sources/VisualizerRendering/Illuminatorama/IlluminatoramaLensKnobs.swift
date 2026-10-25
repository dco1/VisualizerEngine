import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - LensKnobs

/// Stored state for the renderer's `lens` public knobs. Swift extensions cannot hold stored
/// properties, so the renderer holds one `LensKnobs` (`kLens`) and forwards each name to it from
/// `IlluminatoramaRenderer+LensKnobs.swift`; callers see no difference.
@MainActor
final class LensKnobs {
    typealias DOFParams = IlluminatoramaRenderer.DOFParams

    // ── Depth of field ───────────────────────────────────────────────
    /// Lens-accurate DOF on the resolved HDR before bloom. Off = sharp everywhere.
    var dofEnabled: Bool = false
    /// Focus plane distance from the camera, in metres.
    var dofFocusDistance: Float = 3.0
    /// **The whole lens, as one number.** The circle-of-confusion DIAMETER in pixels
    /// per unit of `|z − z_f| / z`, which is the thin-lens relation with everything
    /// that does not vary per pixel — focal length, f-number, focus distance, sensor
    /// size, buffer height — already folded in. The host computes it (in Daydream
    /// Home, `ThinLens.cocCoefficientPixels`); the renderer never re-derives optics.
    ///
    /// Because the host scales it by the buffer height, the defocus is automatically
    /// the same fraction of the frame at every render scale and export size.
    var dofCoCCoefficient: Float = 0
    /// Clamp on the CoC radius in pixels. A performance and sanity bound on the
    /// gather's reach, not part of the optics — leave it generous.
    var dofMaxRadius: Float = 32.0
    /// Iris blade count. `0` (or < 3) is a perfectly circular aperture; 5–9 are the
    /// common real values, and give bokeh balls their polygonal edge.
    var dofBlades: Int = 0
    /// Orientation of the blade polygon, in radians.
    var dofBladeRotation: Float = 0
    /// How wide each gather tap's colour is prefiltered, as a multiple of that tap's own share
    /// of the confusion disc. 1 = exactly its share (the intended value); 0 = point taps, i.e.
    /// the pre-DH-0728 behaviour, kept as an A/B lever rather than as a default.
    ///
    /// This was `DOFParams._pad0` — a field named as a pad on the Swift side and as a control
    /// on the Metal side — so the prefilter pyramid DH-0728 added to the shader was fed 0 and
    /// the gather point-sampled every tap. See DH-0883.
    var dofPrefilterScale: Float = 1.0
    /// **Diffraction**, as the Airy disc's CoC RADIUS in pixels of the internal buffer — a
    /// floor under the circle of confusion at every distance, the focus plane included, since
    /// a diffraction-limited image is soft everywhere rather than soft off the focus plane.
    /// The host derives it from the aperture (`ThinLens.diffractionCoCPixels`); 0 = off.
    ///
    /// Stopped down this is not a subtlety: at 550 nm the Airy diameter is 2.44·λ·N, so f/22
    /// puts it at 0.029 mm — exactly the acceptable CoC on full frame, which is why f/22 is
    /// where a 35 mm lens stops resolving. On 4 × 5, f/22 is a normal working aperture.
    var dofDiffractionCoCPixels: Float = 0
    /// Optical vignetting, 0…1 — how hard the lens barrel clips the aperture toward
    /// the frame corners, turning bokeh discs into cat's-eye lemons. 0 = none.
    var dofCatsEye: Float = 0
    /// **Sharp subjects in front of a defocused background** (opt-in; off = byte-identical).
    /// Two gather fixes for an in-focus silhouette against a far, blurred field: the in-focus
    /// pixel's OWN confusion disc is counted explicitly (a spiral whose reach a defocused
    /// neighbour pushed out to 8+ px never reliably hits a ≲ 1 px disc, so the subject's edge went
    /// see-through, a band as wide as the background's CoC), and a background tap reads the
    /// prefilter pyramid only where its footprint holds no sharper layer (an HDR-bright subject
    /// swamps the pyramid's CoC floor weight and was carried out a whole background-CoC wide as a
    /// halo). Measured on the Digital Clock's toys (VZ-0188).
    var dofSharpSubjects: Bool = false
    /// **Partial occlusion — the LAYERED COMPOSITE** (opt-in, with `dofSharpSubjects`; off =
    /// byte-identical). What the thin lens does at a pixel whose own surface has CoC c: the share of
    /// its rays that a NEARER defocused surface stops is that surface's absolute coverage, composited
    /// over the rest; the rest cross the pixel's depth inside its own disc (radius |c|), where the frame
    /// shows either its own layer or — through a HOLE, past an edge or a gap — whatever lies farther.
    /// So the pixel is `α_F·front + (1 − α_F)·(its own disc's average, holes included)`.
    /// Replaces the sharp-subject gather's behind-tap rule (a farther tap at |c_p|/|c_s| of its weight,
    /// within |c_p| + 1) and its normalised averaging of nearer taps, which — measured against an
    /// aperture-sampled reference of the Digital Clock (`dofReferenceAccumulation`) — kept a blurred
    /// wall's edge 97 % wall where the lens has 50 % (a hard step at every defocused silhouette),
    /// left a defocused FOREGROUND member opaque to its own edge (a crane mast's lattice cut out sharp
    /// inside its soft halo; its falls dark and crisp), and gave a foreground over a sharp subject a
    /// third of the pixel at the silhouette where the lens gives half. Keeps VZ-0188's invariant (an
    /// in-focus subject stays opaque, the wall behind it clean). Taps: the own disc and the near-field
    /// disc round the pixel, one spiral or two; prefiltered reads only where a tap's footprint belongs
    /// to its term (the min-CoC pyramid carries the signed-CoC range for that, rgba16Float while on).
    /// The fronts' coverage is sampled finely enough not to print as a dotted line along every
    /// defocused edge: a NEAR-FIELD coverage pyramid (two more half-res mip chains while on) lets a
    /// near-field front count as the coverage of each tap's footprint instead of a hit-or-miss texel,
    /// and a front behind the focus plane much smaller than the pixel's own disc gets a spiral fitted
    /// to it. Held to the exact layered thin-lens integral in `IlluminatoramaDOFPartialOcclusionTests`.
    var dofPartialOcclusion: Bool = false
    /// **Fast gather** (opt-in, with `dofSharpSubjects`; off = byte-identical). The gather's cost is
    /// taps × pixels and the taps followed the dilated tile max, so every in-focus pixel within a
    /// max-radius of a blurred wall paid the wall's 100+ taps (Digital Clock: 65–73 ms of a 145 ms
    /// frame at 2880×1620). ON: (1) the exact reach — in sharp-subject mode a tap outside the pixel's
    /// own blur and beyond the near field's widest disc can only weigh 0, so it is never taken;
    /// (2) 3 prefiltered taps per px of reach (8…64) instead of 6 (24…128); (3) pixels at least
    /// `dofHalfResMinCoC` px out of focus take a half-resolution gather where the four half-res
    /// texels they read are one layer. Measured against the full gather in
    /// `IlluminatoramaDOFSubjectTests` (VZ-0191).
    var dofFastGather: Bool = false
    /// Fast gather: the |CoC| (px, radius) from which a pixel may take the half-res result.
    var dofHalfResMinCoC: Float = 3
    /// **Quarter-resolution background** (opt-in, with `dofFastGather`; 0 = off = byte-identical).
    /// Pixels at least this many px out of focus (radius, internal pixels; clamped to ≥ 6) take a
    /// bilinear read of a gather made once per 4 × 4 block, where that block is one layer, and the
    /// half-res pass skips every texel whose pixels all do — a macro shot's far wall at the CoC
    /// clamp then costs a quarter of the half-res gathers. A ≥ 12 px disc averages > 450 px; its
    /// 4-px bilinear reconstruction (tent σ 1.63 px) widens it by ≤ 3.6 % in σ at 12 px, ~1 % at 32.
    /// Measured against the full gather in `IlluminatoramaDOFSubjectTests` (VZ-0197).
    var dofQuarterResMinCoC: Float = 0
    // ── Depth of field: the GROUND TRUTH (opt-in diagnostics; off = exact no-op) ─────────
    /// **Depth of field by aperture sampling** — the reference the gather is measured against.
    /// An offline renderer makes defocus by integrating over the lens: the host renders N pinhole
    /// frames of a frozen instant, frame k with the eye moved to pupil point k and the frustum
    /// sheared by `projectionShiftNDC` so the focus plane stays on the same pixels, and this
    /// averages them (`illumi_dof_reference_accumulate`). While on, the DOF pass does not gather:
    /// it adds this frame's HDR to a float accumulator (per-pixel cat's-eye weight for
    /// `dofReferenceApertureSample`) and hands the running MEAN to local adaptation, bloom,
    /// halation and the tonemap — so the finished frame is the lens's own integral through the same
    /// post chain the gather's output takes. The host turns `dofEnabled` off for it (every pixel
    /// shaded and traced at full rate — no defocus shading rate, no reduced glass trace) and keeps
    /// `dofCatsEye` / `naturalVignetteK` as the lens has them. See the kernel for the optics.
    var dofReferenceAccumulation: Bool = false
    /// This frame's pupil point, in units of the iris's circumradius, oriented like the gather's
    /// footprint (x right, y DOWN in pixels). Only its cat's-eye clip is read here — the host has
    /// already moved the eye and sheared the frustum for it.
    var dofReferenceApertureSample: SIMD2<Float> = .zero
    /// false ⇒ this frame is not added: the stored mean is re-presented (settle frames, so the
    /// temporal post state — local adaptation — converges on the finished integral).
    var dofReferenceAddsFrame: Bool = true
    /// Frames added since the last reset (counted when encoded).
    var dofReferenceFrameCount: Int = 0
}
