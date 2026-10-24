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
