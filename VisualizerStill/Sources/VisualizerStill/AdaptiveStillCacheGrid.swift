/// The per-batch cache-cell sequence: batch 0 the shipped cell, then golden-ratio steps over [1, 1.5)×.
public enum AdaptiveStillCacheGrid {
    public static let spread: Float = 0.5
    public static func cell(base: Float, batch: Int, jitter: Bool) -> Float {
        guard jitter, base > 0 else { return base }
        let u = (Double(batch) * 0.6180339887498949).truncatingRemainder(dividingBy: 1)
        return base * (1 + spread * Float(u))
    }
}
