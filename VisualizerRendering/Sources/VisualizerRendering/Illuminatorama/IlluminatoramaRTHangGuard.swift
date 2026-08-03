import Foundation

/// Why `IlluminatoramaRenderer` currently has the ray-traced TLAS path switched
/// off by itself. Exposed (via `IlluminatoramaRenderer.rtGuardState`) because a
/// silently auto-disabled TLAS changes what a rendered frame MEANS — no RT
/// glass, no RT GI — and a host that can't see the flag will chase the
/// appearance change as an art bug.
public enum IlluminatoramaRTGuardTrip: String, Sendable {
    /// The scene is past the instance / mesh-group / triangle caps. Re-checked
    /// every frame against the live scene, so it clears by itself if the scene
    /// gets lighter (a floor hidden, city context switched off).
    case sceneTooHeavy
    /// The TLAS topology changed for more consecutive frames than the rebuild
    /// budget allows. Clears after the topology holds still (see
    /// `IlluminatoramaRTHangGuard.recoveryStableFrames`).
    case topologyThrashing
}

/// The RT hang-guard's state machine, as a pure value — no Metal, no renderer,
/// unit-testable on a machine with no GPU at all (this package's `swift test`
/// produces no metallib and cannot construct a renderer).
///
/// **Why it exists at all:** a scene whose geometry is regenerated every frame
/// makes the TLAS topology hash differ every frame, which sends `updateRTAccel`
/// down the full BLAS+TLAS *rebuild* branch every frame — each rebuild carries
/// `waitUntilCompleted` — and the main thread hangs while a faulted acceleration
/// structure traces to magenta. The guard's job is to notice that and fall back
/// to non-RT.
///
/// **Why it recovers:** the guard used to LATCH until the next scene attach.
/// In a document editor that is the wrong shape: the model mutates on every tick
/// of a drag (push/pull a wall ⇒ new wall meshes ⇒ a new topology hash), so a
/// few frames into the gesture the guard tripped, RT glass switched off, and it
/// never came back — a perf spike followed by a permanent visual pop with no
/// user-reachable way to restore it short of reopening the document. The
/// condition the guard actually cares about ("rebuilding this TLAS would thrash")
/// stops being true the moment the scene settles, so the trip is recoverable:
/// feed it the topology hash each frame and it re-arms once the hash has held
/// still. A genuinely thrashing scene (GPU-fed foliage: a different hash every
/// single frame) never satisfies that and stays disabled exactly as before —
/// which is the property the hang-proofing depends on.
struct IlluminatoramaRTHangGuard: Equatable {
    /// Consecutive frames of an UNCHANGED topology hash required before a
    /// tripped guard re-arms RT. 30 frames ≈ 0.4 s at 75 Hz / 0.5 s at 60 Hz —
    /// deliberately inside the product's "photoreal resolves ~0.3–0.5 s after
    /// the pointer settles" window, so RT returns as part of the settle rather
    /// than as a second, later pop. Long enough that a pause mid-drag (hovering
    /// between two wall positions) doesn't re-arm RT only to re-trip it a
    /// handful of frames later.
    static let recoveryStableFrames = 30

    private(set) var autoDisabled = false
    private(set) var trip: IlluminatoramaRTGuardTrip?
    /// Monotonic since the last `reset()`. A host that sees this climbing during
    /// interaction knows the scene is churning RT topology — a modelling/bridge
    /// problem, not a renderer one.
    private(set) var tripCount = 0
    /// Consecutive frames the topology hash has held while disabled.
    private(set) var settledFrames = 0
    private var settledHash: Int?

    /// Frames of settled topology still owed before RT re-arms; `nil` when
    /// nothing is disabled. Resets to the full window on any frame the scene
    /// keeps churning, so a host watching this during a drag correctly sees it
    /// never reach 0.
    var framesUntilRetry: Int? {
        autoDisabled ? max(0, Self.recoveryStableFrames - settledFrames) : nil
    }

    /// Disable RT for `reason`. Returns `true` only when this call CHANGED the
    /// guard's state — a new trip, or a different reason for an existing one —
    /// so the caller logs once per transition. Both trip sites are re-evaluated
    /// every frame, and a per-frame warning would be a log flood during a drag.
    ///
    /// Always clears the settle window: a fresh trip starts its recovery
    /// countdown from zero, never from a stale partial count.
    @discardableResult
    mutating func recordTrip(_ reason: IlluminatoramaRTGuardTrip) -> Bool {
        settledFrames = 0
        settledHash = nil
        guard !autoDisabled || trip != reason else { return false }
        autoDisabled = true
        trip = reason
        tripCount += 1
        return true
    }

    /// Feed the frame's TLAS topology hash. Returns `true` when RT may run this
    /// frame — either it was never disabled, or the topology has now held still
    /// long enough, in which case the guard re-arms itself (the caller's next
    /// step is the single rebuild that brings the TLAS back).
    ///
    /// Counting: the first call with a new hash establishes the baseline at
    /// `settledFrames == 0`, so re-arming happens on the call
    /// `recoveryStableFrames` frames after that baseline.
    mutating func admitsRetry(topologyHash: Int) -> Bool {
        guard autoDisabled else { return true }
        if settledHash == topologyHash {
            settledFrames += 1
        } else {
            settledHash = topologyHash
            settledFrames = 0
        }
        guard settledFrames >= Self.recoveryStableFrames else { return false }
        // Re-armed. `tripCount` deliberately survives — it is the running
        // history a host uses to tell "this settled once" from "this has been
        // flapping all session".
        autoDisabled = false
        trip = nil
        settledFrames = 0
        settledHash = nil
        return true
    }

    /// Full reset — a freshly attached scene gets a clean chance at RT
    /// regardless of what the previous scene did, trip history included.
    mutating func reset() { self = Self() }
}
