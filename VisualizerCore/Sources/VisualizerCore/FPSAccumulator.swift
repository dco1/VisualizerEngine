import CoreFoundation

/// Accumulates per-tick deltas and fires `report` every half-second with the
/// measured Hz. Replaces the repeated fpsAccumulatedTime/fpsTickCount pair
/// that every native Illuminatorama scene carried.
///
/// `rendered` is the crux of measuring honest FPS: the wall-clock `elapsed`
/// time base advances on EVERY tick (it's real time passing), but a tick only
/// counts as a frame when it actually produced one. A native scene's tick loop
/// is driven by a fixed 60 Hz `Timer`, and `IlluminatoramaRenderer.render()`
/// silently DROPS frames when the GPU is behind (its non-blocking in-flight
/// semaphore). Counting every tick reports the Timer cadence (~60) regardless
/// of what the GPU finished; counting only rendered ticks reports the true
/// throughput the viewer sees. Always pass `render()`'s return value here.
public struct FPSAccumulator {
    private var elapsed: CFTimeInterval = 0
    private var count: Int = 0

    public init() {}

    @MainActor
    public mutating func tick(dt: CFTimeInterval, rendered: Bool = true,
                              report: (@MainActor (Double) -> Void)?) {
        guard dt > 0 else { return }
        if rendered { count += 1 }
        elapsed += dt
        guard elapsed >= 0.5 else { return }
        report?(Double(count) / elapsed)
        elapsed = 0
        count = 0
    }
}
