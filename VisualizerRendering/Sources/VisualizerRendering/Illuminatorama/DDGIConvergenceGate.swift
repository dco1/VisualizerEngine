import Foundation

/// **View-independent settle gate for the DDGI probe trace, as a pure value** —
/// no Metal, no renderer, unit-testable on a machine with no GPU at all (this
/// package's `swift test` produces no metallib and cannot construct a renderer).
///
/// **Why it exists.** `encodeDDGIFrame` re-traces every probe every frame while
/// `ddgiEnabled` is true, and that trace + the two atlas-update kernels are the
/// whole GPU cost of DDGI. But the probe field is **world-space and
/// camera-independent** — the irradiance a probe sees does not change when the
/// user merely orbits the camera. So in a document editor, where a camera-only
/// orbit is by far the common frame, the per-frame re-trace is pure waste: it
/// recomputes a field that is byte-for-byte the same and makes orbiting the whole
/// scene stutter — exactly the "heavy placeable makes the camera stutter" failure
/// the perf gate exists to prevent.
///
/// **What it does.** The host feeds it, each frame, a hash of everything the probe
/// field actually depends on — the geometry epoch, the directional light, the grid,
/// the emitters, the ray/hysteresis knobs — and a convergence budget. When the hash
/// changes (a wall moved, the sun rotated, a proxy set was rebuilt) the gate re-arms
/// for `convergenceFrames` frames so the EMA-hysteresis field can settle to its new
/// answer; once those frames elapse with the hash held still, it stops tracing and
/// the last converged atlas is reused. A camera-only orbit never changes the hash, so
/// after the initial settle it costs nothing.
///
/// **Legacy behaviour is the default.** `convergenceFrames <= 0` means *always
/// trace* — bit-identical to the pre-gate renderer — so a host that does not opt in
/// (every Visualizer scene) is completely unaffected. Only a host that sets a
/// positive budget (Daydream Home's perf-safe house DDGI) gets the skip.
public struct DDGIConvergenceGate: Equatable, Sendable {

    /// The last input hash the gate re-armed on. `nil` until the first query.
    private var lastHash: UInt64? = nil
    /// Frames still to trace in the current convergence window. 0 ⇒ converged
    /// (skip) unless the budget is <= 0 (always trace).
    private var framesRemaining: Int = 0

    /// Whether the most recent `shouldTrace(_:)` returned true — test/instrument
    /// observable, so a perf gate can assert "orbiting N settled frames traced 0".
    public private(set) var didTraceLastQuery: Bool = false

    public init() {}

    /// Decide whether to run the DDGI trace + atlas-update kernels this frame.
    ///
    /// - `inputHash`: a digest of every probe-field input (see the type doc). A
    ///   changed value re-arms the convergence window.
    /// - `convergenceFrames`: how many frames to trace after an input change before
    ///   the field is considered settled. **<= 0 ⇒ always trace** (legacy behaviour).
    @discardableResult
    public mutating func shouldTrace(inputHash: UInt64, convergenceFrames: Int) -> Bool {
        // Opt-out path: no budget ⇒ the pre-gate every-frame trace, exactly.
        guard convergenceFrames > 0 else {
            lastHash = inputHash
            framesRemaining = 0
            didTraceLastQuery = true
            return true
        }
        if inputHash != lastHash {
            lastHash = inputHash
            framesRemaining = convergenceFrames
        }
        let trace = framesRemaining > 0
        if trace { framesRemaining -= 1 }
        didTraceLastQuery = trace
        return trace
    }

    /// Frames left in the current convergence window (0 once settled). Readable for
    /// tests and a host HUD.
    public var framesUntilConverged: Int { framesRemaining }

    /// Force the next `shouldTrace` to re-arm regardless of hash — used when the
    /// atlases were reallocated (grid-dim change) so a stale field is never reused.
    public mutating func reset() {
        lastHash = nil
        framesRemaining = 0
        didTraceLastQuery = false
    }
}
