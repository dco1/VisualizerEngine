/// The accumulation schedule of a still, in one place: frame `k` (1-based) blends in at `1/k` — a
/// CUMULATIVE AVERAGE, every frame weighing the same — down to the resolve kernel's own floor
/// (`clamp(alpha, 0.01, 1)`), past which it is a long EMA rather than a divide the kernel would refuse.
/// `PhotoQuality.accumulationBlend(frame:)` is this with the table's floor; the engine's still session
/// uses it directly.
public enum StillAccumulation {
    public static func blend(frame k: Int, floor: Float) -> Float {
        max(1 / Float(max(k, 1)), floor)
    }
}
