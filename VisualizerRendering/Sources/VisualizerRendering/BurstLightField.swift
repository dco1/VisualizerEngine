import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── BURST LIGHT FIELD ───────────────────────────────────────────────────────
//
// Shared GPU-resident record of every active firework burst's contribution to
// scene illumination. One `SimBuffer<BurstLight>` holds up to N concurrent
// bursts (default 48); each burst registers itself on detonate, ages out over
// its lifespan, and stays addressable to every consumer that wants to "be
// lit by the show":
//
//   • Ocean overlay glow (oceanGlowPaint kernel)
//   • Smoke particle render (smoke kernels)
//   • Star washout near bright bursts (starApplyBurstWashout kernel)
//   • Anything else that reads the buffer + the shared `sampleBurstField`
//     function in BurstLightField.metal
//
// The buffer is layout-locked to the Metal `BurstLight` struct in
// BurstLightField.metal. Per the project ALIGNMENT RULE all fields are float4.
//
// Lifecycle:
//   1. Controller calls `register(center:color:intensity:radius:lifespan:)`
//      on burst-fire.
//   2. Per tick the controller calls `tick(dt:)` to age every slot's
//      `colorAge.a` and compact dead slots out the back.
//   3. Consumers bind `buffer` + `activeCount` to their kernels.
//
// The CPU mirror keeps the array in-order (oldest-first), which lets us
// O(1) drop dead slots from the front and append new bursts at the back.

@MainActor
public final class BurstLightField {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "BurstLightField")

    public let device: MTLDevice
    public let buffer: SimBuffer<BurstLight>
    public let capacity: Int

    /// Live count — GPU consumers should clamp dispatch to this.
    public private(set) var activeCount: Int = 0

    /// CPU mirror. Oldest-first so we can pop from the front as bursts expire.
    private var mirror: [BurstLight] = []

    public init?(device: MTLDevice, capacity: Int = 48) {
        guard let buf = SimBuffer<BurstLight>(
            device: device,
            capacity: capacity,
            label: "Fireworks.burstLights"
        ) else {
            Self.log.error("BurstLightField buffer allocation failed")
            return nil
        }
        self.device = device
        self.buffer = buf
        self.capacity = capacity
        // Pre-zero the GPU buffer so kernels iterating activeCount=0 see safe
        // values even before the first register call.
        let zero = BurstLight()
        var seed = Array(repeating: zero, count: capacity)
        seed.withUnsafeBufferPointer { p in
            buf.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<BurstLight>.stride * capacity
            )
        }
    }

    /// Register a fresh burst. Drops the oldest entry if at capacity.
    public func register(
        center: SIMD3<Float>,
        color: SIMD3<Float>,
        intensity: Float,
        radius: Float,
        lifespan: Float
    ) {
        let light = BurstLight(
            positionIntensity: SIMD4(center.x, center.y, center.z, intensity),
            colorAge:          SIMD4(color.x,  color.y,  color.z,  0),
            radiusLifePad:     SIMD4(radius,   lifespan, 0,        0)
        )
        if mirror.count >= capacity {
            mirror.removeFirst()
        }
        mirror.append(light)
        uploadMirror()
    }

    /// Advance every burst's age by dt; drop expired entries (age ≥ lifespan).
    public func tick(dt: Float) {
        guard !mirror.isEmpty else { return }
        var changed = false
        for i in 0..<mirror.count {
            mirror[i].colorAge.w += dt
        }
        // Trim from the front while the oldest is dead.
        while let first = mirror.first, first.colorAge.w >= first.radiusLifePad.y {
            mirror.removeFirst()
            changed = true
        }
        _ = changed
        uploadMirror()
    }

    /// Drop everything (scene clear / pause-reset).
    public func clear() {
        mirror.removeAll(keepingCapacity: true)
        activeCount = 0
        let zero = BurstLight()
        var seed = Array(repeating: zero, count: capacity)
        seed.withUnsafeBufferPointer { p in
            buffer.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<BurstLight>.stride * capacity
            )
        }
    }

    private func uploadMirror() {
        activeCount = min(mirror.count, capacity)
        guard activeCount > 0 else { return }
        let ptr = buffer.buffer.contents().bindMemory(to: BurstLight.self, capacity: capacity)
        for i in 0..<activeCount {
            ptr[i] = mirror[i]
        }
    }
}

// ── Shared struct ───────────────────────────────────────────────────────────
//
// 48 bytes, all float4 per the project ALIGNMENT RULE. Mirror of the metal
// `BurstLight` struct in BurstLightField.metal.

public struct BurstLight {
    public var positionIntensity: SIMD4<Float>   // xyz pos, w intensity (linear)
    public var colorAge:          SIMD4<Float>   // rgb base color, a age (s)
    public var radiusLifePad:     SIMD4<Float>   // x max radius (m), y lifespan (s)

    public init(
        positionIntensity: SIMD4<Float> = SIMD4(0, 0, 0, 0),
        colorAge:          SIMD4<Float> = SIMD4(0, 0, 0, 0),
        radiusLifePad:     SIMD4<Float> = SIMD4(0, 0, 0, 0)
    ) {
        self.positionIntensity = positionIntensity
        self.colorAge          = colorAge
        self.radiusLifePad     = radiusLifePad
    }
}
