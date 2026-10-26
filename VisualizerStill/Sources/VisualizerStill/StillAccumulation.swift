/// The accumulation schedule of a still, in one place: frame `k` (1-based) blends in at `1/k` — a
/// CUMULATIVE AVERAGE, every frame weighing the same — down to the resolve kernel's own floor
/// (`clamp(alpha, 0.01, 1)`), past which it is a long EMA rather than a divide the kernel would refuse.
/// `PhotoQuality.accumulationBlend(frame:)` is this with the table's floor; the engine's still session
/// uses it directly.
public enum StillAccumulation {
    public static func blend(frame k: Int, floor: Float) -> Float {
        max(1 / Float(max(k, 1)), floor)
    }

    /// What the renderer's temporal knobs should be for ONE frame. The single vocabulary every lane speaks —
    /// the engine's still session, the host's settled canvas, an export — so the knob writes exist once
    /// (`IlluminatoramaRenderer.applyAccumulation`).
    public enum Step: Equatable, Sendable {
        /// No accumulation: no jitter, the temporal blend is left alone.
        case off
        /// A FIXED current-frame weight (an EMA) with sub-pixel jitter — the live canvas's converge ramp and
        /// the A/B lever that pins the accumulator.
        case fixed(jitter: Float, blend: Float)
        /// The k-th frame of a cumulative average (`1/k` down to `floor`). `aoDial` (the host's
        /// noise-reduction value for the AO accumulator), when given, opens the AO floor to the same
        /// `1 − floor` so AO is averaged too, not smoothed by the dial's short EMA.
        case average(jitter: Float, frame: Int, floor: Float, aoDial: Float?)
        /// Print the running mean and add nothing: jitter on, blend 0.
        case hold(jitter: Float)
        /// A frame held OUT of every accumulator: no jitter, blend 1 (history is replaced, not averaged).
        case heldOut
    }
}

/// The running state behind `StillAccumulation.blend`: how many frames the current average holds, where a
/// settled canvas's REFINE phase is, and which view the average was taken from. Pure value state, so a
/// host that must undo a dropped frame just keeps a copy (`let before = accumulator … accumulator = before`).
///
/// Three lanes drive it, with one vocabulary: an export still (`advance()`), the settled canvas's converge
/// ramp and refine (`advanceConverge`), and a frozen camera whose average restarts when the view changes
/// (`restartOnCut`). The blend each frame gets is then `StillAccumulation.blend(frame:floor:)` — the engine's
/// `applyAccumulation(.average(frame:…))`.
public struct StillAccumulator: Equatable, Sendable {
    /// Frames the current average holds (the count the next blend is `1/(frames + prior)` of).
    public private(set) var frames = 0
    /// Frames rendered in the current refine phase; the host zeroes it on entering/leaving refine.
    public var refineFrames = 0
    /// This frame was one of the refine's warm-up frames (kept OUT of the average).
    public private(set) var warming = false
    private var signature: [Float]?

    public init() {}

    /// The average starts over (its history is gone, or a new still begins).
    public mutating func restart() { frames = 0 }

    /// One accumulated frame of an export still; returns its 1-based index.
    public mutating func advance() -> Int { frames += 1; return frames }

    /// A frozen camera's average restarts when what it averages changes (camera + internal size).
    /// `signature` is whatever identifies the view; nil/`accumulating == false` forgets it.
    public mutating func restartOnCut(accumulating: Bool, signature new: [Float]) {
        if accumulating, let last = signature, last != new { frames = 0 }
        signature = accumulating ? new : nil
    }

    public enum Converge: Equatable, Sendable {
        /// A warm-up frame of the refine: add nothing (blend 0).
        case hold
        /// Blend in as frame `frame` of the running average (already includes `prior`);
        /// `resetGI` is true on the first frame after the warm-up, when the GI running mean must be cleared.
        case average(frame: Int, resetGI: Bool)
    }

    /// One frame of the settled canvas. Refine restarts the average on its first frame (the converged settle
    /// counts as `prior` frames, the export-lit frames dissolve in over it); its first `warmupFrames` frames
    /// shade against a structure still being built and add nothing.
    public mutating func advanceConverge(refining: Bool, warmupFrames: Int, prior: Int) -> Converge {
        warming = false
        if refining {
            refineFrames += 1
            if refineFrames == 1 { frames = 0 }
            warming = refineFrames <= warmupFrames
        }
        if warming { return .hold }
        let resetGI = refining && refineFrames == warmupFrames + 1
        frames += 1
        return .average(frame: frames + prior, resetGI: resetGI)
    }
}
