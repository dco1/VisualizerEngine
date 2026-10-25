import Foundation

/// **"Render smooth once, then hold it."** — the settled-canvas convergence state machine.
///
/// `DraftLatch` decides *how* a frame is drawn while a gesture is live (draft) vs once the
/// view settles (photoreal). This decides what a *settled* frame does over TIME: a still image
/// has no vsync budget, so instead of re-rendering the same picture every vsync forever, the
/// canvas may **converge** — accumulate a fixed number of sub-pixel-jittered frames the way the
/// photo-export lane does, supersampling the still smooth — and then **hold**, skipping
/// `render()`/`present()` entirely so the GPU idles until something changes (Danny, DH-0064,
/// 2026-09-05: option A — freeze animated elements while held; the literal "hold it").
///
/// Like `DraftLatch` this is deliberately PURE — no Metal, no clock, no AppKit — so the
/// behaviour that guards `main` (a canvas that must never freeze on an edit) is asserted as a
/// number rather than eyeballed. The host feeds it three facts per frame and it returns the
/// phase for that frame.
///
/// ## The one rule that keeps it safe
///
/// A **content change re-arms convergence to `.live` for that frame**, and convergence only
/// *begins* on a frame whose content is provably UNCHANGED from the frame before it. This has
/// two consequences that matter:
///
/// - The frame that first shows an edit is always `.live` — the fresh picture reaches the
///   screen immediately, never blended into a stale accumulator (the "second arm rendered the
///   first arm's converged image" bug the prior attempt hit).
/// - A transient animation (a swinging door, the sun easing to a new hour) is given a frame to
///   register as `animating` before anything holds, so nothing freezes mid-motion.
///
/// The `epoch` the host passes is a hash of *every* render input (`frameContentEpoch` on the
/// canvas coordinator); an input left out of that hash is exactly the freeze-on-edit hazard, so
/// it is covered by a fail-closed completeness gate (`ConvergenceEpochCompletenessTests`).
/// Under-inclusive is the bug; over-inclusive merely means the canvas re-renders a frame it
/// could have held, which is only ever the current behaviour. So the epoch is built as a
/// superset, and this latch trusts it.
public struct ConvergenceLatch: Sendable, Equatable {

    /// What a settled frame does this tick.
    public enum Phase: Sendable, Equatable {
        /// Render live at the settled tier — a gesture is active, the content just changed, or
        /// a transient animation is still in flight. Animated elements run; nothing accumulates.
        case live
        /// Accumulate one more sub-pixel-jittered sample into the converging still. Animated
        /// elements are frozen (the host pins the frame clock) so the accumulation stays sharp.
        case converging
        /// The ramp has converged; keep accumulating with the Export Photo's LIGHTING (traced
        /// sun + window shadows, one-bounce GI) so the canvas refines in place toward the
        /// export (Danny, 2026-09-23). Same frozen clock, same re-arm rules — any change drops
        /// straight back to `.live`. Skipped entirely when `refineFrames` is 0.
        case refining
        /// The still has converged. Skip `render()`/`present()`; the GPU idles and the last
        /// presented frame stays on screen until the content epoch moves.
        case held
    }

    /// How many jittered samples to accumulate before holding. Matches the photo lane's
    /// `PhotoQuality.settleFrames` so the settled canvas converges to the same still the export
    /// produces — the visual-acceptance bar for DH-0064.
    public var convergenceFrames: Int
    /// How many export-lit frames to refine for after the ramp, before holding. 0 = no refine
    /// phase (the pre-2026-09-23 behaviour, and the default for a bare latch).
    public var refineFrames: Int

    /// The last content epoch observed. `nil` before the first frame, so the first frame is
    /// always a content change (→ `.live`) and the latch never claims to have converged at
    /// startup.
    private var epoch: Int?
    /// Samples accumulated into the current still. Reset to 0 on every re-arm.
    public private(set) var framesConverged = 0
    /// Refine frames rendered into the current still. Reset to 0 on every re-arm.
    public private(set) var framesRefined = 0
    /// The phase returned last tick — for the host and for the regression gate.
    public private(set) var phase: Phase = .live

    public init(convergenceFrames: Int = 32, refineFrames: Int = 0) {
        self.convergenceFrames = max(1, convergenceFrames)
        self.refineFrames = max(0, refineFrames)
    }

    /// Advance one frame.
    ///
    /// - Parameters:
    ///   - epoch: hash of every render input this frame (see `frameContentEpoch`).
    ///   - settled: `true` when `DraftLatch` is NOT drafting (no gesture in flight).
    ///   - animating: `true` while a transient, non-gesture animation is still easing toward its
    ///     target (the sun's time-of-day ease, a door/oven swing) or a perpetual animation the
    ///     frame clock cannot freeze is running (falling rain). Perpetual animations that the
    ///     clock DOES freeze (grass, sway, TV flicker) are NOT `animating` — option A freezes
    ///     them, so they must not keep the canvas live.
    /// - Returns: the phase to render this frame at.
    @discardableResult
    public mutating func update(epoch: Int, settled: Bool, animating: Bool) -> Phase {
        let contentChanged = epoch != self.epoch
        self.epoch = epoch
        // Re-arm on ANY of: a live gesture, a transient still easing, or a content change. The
        // re-arm frame renders live so the new picture shows at once and animations get a frame
        // to start.
        if !settled || animating || contentChanged {
            framesConverged = 0
            framesRefined = 0
            phase = .live
            return phase
        }
        // Provably static, settled, nothing animating: accumulate, then hold.
        if framesConverged < convergenceFrames {
            framesConverged += 1
            phase = .converging
        } else if framesRefined < refineFrames {
            framesRefined += 1
            phase = .refining
        } else {
            phase = .held
        }
        return phase
    }

    /// The host did NOT render this tick's frame (the GPU still had its frames in flight and the
    /// renderer dropped it). A dropped tick is not a sample: un-count it, so `convergenceFrames`
    /// and `refineFrames` mean RENDERED frames. Without this a 90 ms converge frame at 60 Hz
    /// turned the 32-frame ramp into ~6 real samples (2026-09-23).
    public mutating func frameDropped() {
        switch phase {
        case .converging: framesConverged = max(0, framesConverged - 1)
        case .refining:   framesRefined = max(0, framesRefined - 1)
        case .live, .held: break
        }
    }

    /// Force back to `.live` with no accumulated history — for a hard context change (a new
    /// document, the canvas losing its window) where the host applies the live state itself.
    public mutating func reset() {
        epoch = nil
        framesConverged = 0
        framesRefined = 0
        phase = .live
    }
}
