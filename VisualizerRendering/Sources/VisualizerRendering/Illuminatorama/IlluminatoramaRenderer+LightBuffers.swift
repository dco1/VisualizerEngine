import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - LightBufferState

/// Per-frame light and frame-uniform ring buffers.
/// Owned by the renderer as `lts`; every member keeps the name it had on the renderer.
@MainActor
final class LightBufferState {
    var areaLightRing: [MTLBuffer]!

    var areaLightCapacity: Int!
    var extraDirectionalRing: [MTLBuffer]!

    var interiorApertureRing: [MTLBuffer]!

    var interiorApertureCapacity: Int = 0
    var extraDirectionalCapacity: Int!
    var frameUniformRing: [MTLBuffer]!

    /// Which ring slot the CURRENT frame writes + binds. Advanced by exactly one
    /// per ACCEPTED `render()` (after the in-flight semaphore admits the frame),
    /// so two consecutive in-flight frames always use different slots. Sized to
    /// `maxFramesInFlight`; the GPU's max concurrency is bounded by the same
    /// semaphore, so a slot is never both written (CPU, frame N+1) and read
    /// (GPU, frame N) at once.
    var frameRingIndex: Int = 0
}
