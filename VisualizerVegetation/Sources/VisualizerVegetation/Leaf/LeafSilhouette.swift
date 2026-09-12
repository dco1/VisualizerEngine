import simd
import VisualizerMaterials

/// The outline of ONE blade — a leaf, a petal, a leaflet, a pad — as **data**.
///
/// This used to be a closed `enum Silhouette` with a `switch` returning literals, which meant
/// every new species was a new case in an enum plus a new arm in a switch plus (on the tree side)
/// a new arm in a *second* switch mapping species onto it. Fifteen SoCal natives filed against
/// that shape would have been fifteen edits to a growing switch in a shared file.
///
/// As a value it is just a table someone writes down: a half-margin, mirrored at emit time. The
/// shipped outlines are `static let`s below — a catalog, not a language construct — so a new
/// species is a new constant (or a literal built at the call site), and nothing that already
/// compiles has to change to admit it.
///
/// **The half-margin convention.** `v` runs 0 at the blade base → 1 at the tip; `u` is the
/// half-width as a fraction of `width / 2`. The first control is the base (`u = 0`) and the last
/// is the tip (`u = 0`); everything between is one side of the outline, mirrored to the other.
public struct LeafSilhouette: Sendable, Hashable {

    /// One control point on the half-margin.
    public struct Control: Sendable, Hashable {
        public var v: Double
        public var u: Double
        public init(_ v: Double, _ u: Double) { self.v = v; self.u = u }
    }

    /// Detail tier. A lobed outline can afford ~7 controls in the foreground and ~5 at mid
    /// distance; a simple entire blade carries one outline at every tier and ignores this.
    public enum Tier: Sendable, Hashable { case hero, mid }

    /// The foreground outline.
    public var hero: [Control]
    /// The reduced outline. Nil means tier-invariant — `mid` resolves to `hero`, which is the
    /// right answer for any outline simple enough that dropping controls would only cost shape.
    public var mid: [Control]?
    /// A human-readable name, for diagnostics and census labelling.
    public var name: String

    public init(name: String, hero: [Control], mid: [Control]? = nil) {
        self.name = name
        self.hero = hero
        self.mid = mid
    }

    /// Convenience initializer taking the raw `(v, u)` pairs the catalog is written in.
    public init(name: String, hero: [(Double, Double)], mid: [(Double, Double)]? = nil) {
        self.init(name: name,
                  hero: hero.map { Control($0.0, $0.1) },
                  mid: mid.map { $0.map { Control($0.0, $0.1) } })
    }

    /// The authored control points for a tier, before smoothing.
    public func controls(tier: Tier = .hero) -> [Control] {
        switch tier {
        case .hero: return hero
        case .mid:  return mid ?? hero
        }
    }

    /// The resolved half-margin: controls for `tier`, rounded by `subdivisions` steps per segment.
    ///
    /// `subdivisions <= 1` returns the authored controls unchanged. That is not a micro-optimization
    /// — it is the byte-identity escape hatch the yard-tree bake depends on, and its exact triangle
    /// counts are pinned by `ForestTreeValidityTests`.
    public func margin(tier: Tier = .hero, subdivisions: Int = 1) -> [Control] {
        Self.smooth(controls(tier: tier), subdivisions: subdivisions)
    }

    /// Round a straight-segment half-margin into a smooth curve by Catmull-Rom subdivision.
    ///
    /// The authored margins are 4–10 controls joined by STRAIGHT edges. On a small tree leaf that
    /// reads fine; blown up to a big houseplant leaf the straight runs and hard corners are exactly
    /// the "chaotic pile of sharp star-shaped shards" DH-0628 caught, and the same root cause
    /// DH-0626 / DH-0629 / DH-0647 each rediscovered independently. Rounding it HERE, once, is the
    /// point of having one constructor.
    ///
    /// Two details are load-bearing and easy to get wrong:
    ///  - `v` interpolates LINEARLY so the midrib parameter stays strictly increasing. Splining it
    ///    too lets an overshoot reverse the parameter and invert a strip triangle.
    ///  - `u` is clamped `>= 0` so an overshoot near an endpoint cannot fold the margin across the
    ///    midrib and turn the blade inside out.
    ///
    /// The base and tip controls are preserved exactly.
    public static func smooth(_ pts: [Control], subdivisions: Int) -> [Control] {
        guard subdivisions > 1, pts.count >= 3 else { return pts }
        func u(_ i: Int) -> Double { pts[min(max(i, 0), pts.count - 1)].u }
        var out: [Control] = []
        out.reserveCapacity((pts.count - 1) * subdivisions + 1)
        for i in 0 ..< (pts.count - 1) {
            let u0 = u(i - 1), u1 = pts[i].u, u2 = pts[i + 1].u, u3 = u(i + 2)
            let v0 = pts[i].v, v1 = pts[i + 1].v
            for s in 0 ..< subdivisions {
                let t = Double(s) / Double(subdivisions)
                let t2 = t * t, t3 = t2 * t
                // Uniform Catmull-Rom on the width profile.
                let uu = 0.5 * ((2 * u1)
                    + (-u0 + u2) * t
                    + (2 * u0 - 5 * u1 + 4 * u2 - u3) * t2
                    + (-u0 + 3 * u1 - 3 * u2 + u3) * t3)
                out.append(Control(v0 + (v1 - v0) * t, max(0, uu)))
            }
        }
        out.append(pts[pts.count - 1])   // exact tip
        return out
    }
}

// MARK: - The shipped catalog
//
// Outlines, not species. A species picks one of these (or authors its own) — see
// `VegetationSpecies`. Every table here is carried over UNCHANGED from
// `DaydreamCore.LeafCardGeometry.margin`, digit for digit, because the yard tree's baked triangle
// counts are pinned to them by `ForestTreeValidityTests.testLegacyLeverReproducesThePreFixTriangleCounts`
// (exact totals) and `testDecimationRemovesWholeLeafCardsNotTriangleShards` (a hero card is exactly
// 20 triangles). If you change one of these numbers, that gate moves with it: re-derive it, don't
// patch it.

extension LeafSilhouette {

    /// Obovate, pinnately lobed: narrow rounded base, broad rounded lobes past the middle, blunt
    /// tip; sinuses (u→small) cut between the lobes.
    public static let oak = LeafSilhouette(
        name: "oak",
        hero: [(0.00, 0.00), (0.16, 0.46), (0.30, 0.30),
               (0.48, 0.78), (0.62, 0.42), (0.80, 0.66), (1.00, 0.00)],
        mid:  [(0.00, 0.00), (0.34, 0.62), (0.58, 0.40),
               (0.80, 0.66), (1.00, 0.00)])

    /// Triangular-ovate, doubly serrated: broad base → drawn-out point, saw teeth on the edge.
    public static let birch = LeafSilhouette(
        name: "birch",
        hero: [(0.00, 0.00), (0.10, 0.66), (0.24, 0.50),
               (0.40, 0.62), (0.56, 0.44), (0.78, 0.30), (1.00, 0.00)],
        mid:  [(0.00, 0.00), (0.14, 0.66), (0.42, 0.54),
               (0.72, 0.36), (1.00, 0.00)])

    /// Palmate: pointed lobes radiate from the petiole, deep sinuses between, widest mid-blade.
    public static let maple = LeafSilhouette(
        name: "maple",
        hero: [(0.00, 0.00), (0.14, 0.84), (0.30, 0.28),
               (0.46, 0.92), (0.62, 0.26), (0.80, 0.50), (1.00, 0.00)],
        mid:  [(0.00, 0.00), (0.16, 0.82), (0.42, 0.30),
               (0.66, 0.62), (1.00, 0.00)])

    /// A small, entire (un-lobed) elliptic-ovate leaf drawn out to a soft point — the glossy citrus
    /// blade of an orange/lemon. Widest just past the base, then a clean taper: NO lobes or
    /// serrations (those read as maple/fern up close and are wrong for a citrus).
    public static let citrus = LeafSilhouette(
        name: "citrus",
        hero: [(0.00, 0.00), (0.12, 0.46), (0.34, 0.60),
               (0.60, 0.55), (0.82, 0.36), (1.00, 0.00)])

    /// A tall strap that stays wide most of its length then tapers to a soft point — a snake-plant
    /// sword. No lobes (they'd read as fern pinnae here).
    public static let strapBlade = LeafSilhouette(
        name: "strapBlade",
        hero: [(0.00, 0.00), (0.08, 0.85), (0.35, 1.00),
               (0.70, 0.95), (0.90, 0.60), (1.00, 0.00)])

    /// Rounded-obovate teardrop: widens quickly, rounds over, tapers to a blunt tip. A petal, not
    /// a leaf — no lobes.
    public static let petal = LeafSilhouette(
        name: "petal",
        hero: [(0.00, 0.00), (0.12, 0.55), (0.34, 0.92),
               (0.60, 1.00), (0.82, 0.78), (1.00, 0.00)])

    /// A big SIMPLE entire leaf, violin/obovate: narrow base, a gentle waist, then the broadest
    /// point high on the blade rounding over to a blunt tip. NO lobes or teeth — the fiddle-leaf
    /// fig's whole read is one large glossy sheet.
    ///
    /// The "gentle waist" this doc promised was never actually IN the curve: the old control
    /// points (0.10,0.30)→(0.26,0.40)→(0.42,0.40) only widen then plateau — u never dips, so
    /// there's no pinch for `(0.60, 0.60)`'s later widen to read AGAINST. A realism pass caught
    /// this as a real, consistent complaint ("generic oval, lacks the waisted violin shape") across
    /// every run, independent of colour/material/lighting changes — this is the actual geometric
    /// fix for it. `(0.40, 0.30)` now sits BELOW its neighbours on both sides (`0.26→0.38` up,
    /// `0.58→0.34` back down before the real flare), a genuine pinch, and the broad shoulder moved
    /// later/wider (`0.80, 0.72`) so the leaf's iconic top-heavy flare reads more dramatically.
    ///
    /// The shoulder-to-tip run originally dropped `0.80→0.93` in ONE step (swing `-0.24`) then
    /// `0.93→1.00` in another (`-0.48`, the universal tip taper) — two aggressive pull-ins back to
    /// back read as a sharp beak, not the soft rounded-over tip of a real fiddle-leaf-fig blade
    /// (the same "spiky, not lobed" mechanism the monstera fix above documents: a swing this size
    /// still turns a real corner AT the control point no matter how much subdivision smooths the
    /// path between two of them). Splitting it into `0.80→0.87→0.93` (swings `-0.10`/`-0.14`, both
    /// inside the ~0.12-0.20 range that reads rounded) gives the tip one more station to round
    /// through before the expected final taper to the point.
    public static let fiddleLeafFig = LeafSilhouette(
        name: "fiddleLeafFig",
        hero: [(0.00, 0.00), (0.12, 0.26), (0.26, 0.38), (0.40, 0.30),
               (0.58, 0.34), (0.68, 0.55), (0.80, 0.72), (0.87, 0.62),
               (0.93, 0.48), (1.00, 0.00)])

    /// A broad blade with a few gentle rounded lobes — the stylized stand-in for the monstera's
    /// signature fenestration.
    ///
    /// The PREVIOUS curve swung u by 0.44-0.48 between adjacent control points (0.86 → 0.42 → 0.88
    /// → 0.40 → 0.82 → 0.40) — Catmull-Rom subdivision smooths the PATH between two points, but it
    /// threads THROUGH every control point, so a swing that sharp still turns a genuine corner AT
    /// each one; "smoothing" the segments between two spikes doesn't blunt the spikes themselves.
    /// The result rendered as a spiky star/urchin, not a lobed tropical leaf, confirmed both by a
    /// realism pass and by direct visual inspection (DH-0775). Halving the swing to ~0.12-0.20 per
    /// step is what actually reads as ROUNDED lobes with shallow scalloped valleys between them —
    /// still a wavy, lobed margin (not a plain oval), just without the sharp reversal at each peak.
    public static let monstera = LeafSilhouette(
        name: "monstera",
        hero: [(0.00, 0.00), (0.10, 0.52), (0.22, 0.68), (0.34, 0.56),
               (0.46, 0.74), (0.58, 0.58), (0.70, 0.66), (0.82, 0.50),
               (0.92, 0.32), (1.00, 0.00)])

    /// A short, fat, very rounded fleshy paddle (jade / echeveria): widens fast to a broad rounded
    /// middle, then rounds over to a blunt tip. Entire margin.
    public static let succulentPad = LeafSilhouette(
        name: "succulentPad",
        hero: [(0.00, 0.00), (0.14, 0.58), (0.36, 0.84),
               (0.60, 0.86), (0.82, 0.64), (1.00, 0.00)])

    /// ONE small lanceolate pinnule of a compound frond: widest just past the base, then a clean
    /// taper to a soft point. Kept to the FEWEST controls that still round nicely — a leaflet is
    /// tiny on screen and there are many per frond, so every extra segment is multiplied by the
    /// leaflet count (the tri-budget lever).
    public static let fernLeaflet = LeafSilhouette(
        name: "fernLeaflet",
        hero: [(0.00, 0.00), (0.20, 0.62), (0.56, 0.52), (1.00, 0.00)])

    /// ONE conifer needle: a long, very narrow lens — widest through the middle, tapering to a point
    /// at BOTH the sheath end and the tip. Unlike every other outline here it is symmetric about
    /// mid-length, because a needle has no base-to-tip shape story to tell; its whole read at any
    /// distance a house is viewed from is "a thin green line with soft ends".
    ///
    /// **Four controls, and that is a budget decision, not a shape one.** A needle is the one blade
    /// that gets multiplied twice over — by the fascicle's `count`, then by the shoot's fascicle
    /// count, then by the tree's shoot count — so every control here is paid for perhaps 100 000
    /// times per conifer. Four controls is three segments, which is 12 triangles per needle
    /// single-sided (see `NeedleFascicle`); a fifth control to give the lens a true single centre
    /// peak would be a third more, for a difference nothing can resolve. Catmull-Rom rounding
    /// (`subdivisions: 2`) is the better lever if a foreground needle ever needs it.
    ///
    /// Widths are `u` fractions of the half-width, and `NeedleFascicle.defaultAspect` sets the
    /// absolute width — the peak at `u = 0.92` rather than `1.00` is what lets the lens round over
    /// through the middle instead of running straight between two corners.
    public static let needle = LeafSilhouette(
        name: "needle",
        hero: [(0.00, 0.00), (0.30, 0.92), (0.70, 0.92), (1.00, 0.00)])
}
