import Foundation
import simd

/// Engine-level **interpenetration measurement** — the code answer to the
/// recurring "these objects are interpenetrating / aren't colliding naturally"
/// note. Instead of an LLM eyeballing a render, a solver measures overlap from
/// the SAME state it simulates on, so the number is exact and consistent with
/// what the contact solver itself sees.
///
/// A type conforms when it can report how badly its bodies overlap in their
/// CURRENT state. Implementations:
///   • `CoinDEMSolver` — a GPU kernel (`coinMeasurePenetration`) reuses the
///     solver's spatial-hash broadphase + the exact disk-vs-disk SAT it
///     de-penetrates with, reducing to a tiny readback. Covers coin / jewel /
///     frankfurter piles — the biggest interpenetration surface in the app.
///   • Illuminatorama-native / hand-placed scenes are measured at the geometry
///     level by `PenetrationProbe` (app side) walking the SCN scene's drawn
///     bodies as oriented boxes — those scenes expose no per-body solver.
///
/// One scene can return several probers (a mixed pile + a separate chain).
///
/// Main-actor isolated: solvers and controllers in this project are `@MainActor`
/// (see CLAUDE.md conventions), and the probe is driven from the main actor.
@MainActor
public protocol PenetrationProbing: AnyObject {
    /// A representative body radius, in world metres. The probe multiplies this
    /// by a relative fraction to get a scale-appropriate absolute threshold —
    /// so a 1.5 m stylized chip and a 6 cm egg are judged by the SAME relative
    /// bar, instead of an absolute centimetre figure that's strict on one and
    /// meaningless on the other.
    var characteristicBodyScale: Float { get }

    /// Measure interpenetration in the current state. `threshold` is the depth,
    /// in world metres, below which an overlap is treated as a normal resting
    /// contact rather than a defect (the caller usually derives it from
    /// `characteristicBodyScale`). Cheap enough to call once after the sim
    /// settles; not intended for the per-frame hot path.
    func measurePenetration(threshold: Float) -> PenetrationStats
}

/// One named overlapping pair, for the probes that can attribute overlap to
/// specific bodies (the geometry/instance OBB path). The GPU reduction reports
/// counts + max depth only, so it leaves `worst` empty.
public struct PenetrationPair: Sendable, Equatable {
    public var a: String
    public var b: String
    public var depth: Float   // metres
    public init(a: String, b: String, depth: Float) {
        self.a = a; self.b = b; self.depth = depth
    }
}

/// Result of one penetration measurement.
public struct PenetrationStats: Sendable {
    /// Which backend produced this (e.g. "CoinDEM", "Illuminatorama-instances").
    public var source: String
    /// Bodies considered (active solver bodies, or drawn instances).
    public var bodyCount: Int
    /// Distinct body pairs overlapping by more than `threshold`.
    public var penetratingPairs: Int
    /// Deepest overlap seen, in metres (0 if none).
    public var maxPenetration: Float
    /// The threshold this measurement used.
    public var threshold: Float
    /// Deepest offenders, named, sorted deep→shallow. May be empty when the
    /// backend reports aggregate counts only (the GPU reduction).
    public var worst: [PenetrationPair]

    public init(source: String,
                bodyCount: Int,
                penetratingPairs: Int,
                maxPenetration: Float,
                threshold: Float,
                worst: [PenetrationPair] = []) {
        self.source = source
        self.bodyCount = bodyCount
        self.penetratingPairs = penetratingPairs
        self.maxPenetration = maxPenetration
        self.threshold = threshold
        self.worst = worst
    }

    /// Did this measurement clear the bar?
    public var isClean: Bool { penetratingPairs == 0 }
}
