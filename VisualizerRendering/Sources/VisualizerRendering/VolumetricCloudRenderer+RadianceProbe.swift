import Foundation
import Metal
import os
import simd

// ── SKY RADIANCE PROBES (opt-in) ─────────────────────────────────────────────
//
// A host that lets its scene's light come from the sky it SHOWS — a window's portal area light,
// a room's fill — needs that sky's radiance as a number, in the dome's own units, the moment the
// dome changes. Hand-fitting a curve against the sun's elevation instead makes a second source of
// truth that drifts whenever the atmosphere model changes (Digital Clock's twilight portal was
// ~15× the dome it stood for between −6° and −7.9° once the single-scatter dome went black there).
//
// `radianceProbes` names up to 8 planes. Every dome bake (`render(params:)`'s dome pass,
// `renderNow`) then encodes `skyRadianceProbe` (SkyRadianceProbe.metal) right after the dome in
// the same command buffer: per plane, the cosine-weighted mean of the dome over the hemisphere
// around its normal — E(n)/π, the radiance of a Lambertian surface passing the flux the plane
// receives — 4096 dome samples, one threadgroup. A completion handler copies the few float4s out
// of a small shared buffer; `latestRadianceProbeReading` returns the newest completed one. No
// CPU wait, no readback of the dome itself: the host gets its numbers a frame or so after the
// bake, in step with the sky it is looking at.
//
// Purely additive: with `radianceProbes` empty (the default) nothing is encoded and no pipeline is
// built, so every existing host is byte-identical.

extension VolumetricCloudRenderer {

    /// At most this many planes per bake (one threadgroup each).
    public nonisolated static let maxRadianceProbes = 8

    /// One plane whose sky radiance the dome bakes measure.
    public struct RadianceProbe: Sendable, Equatable {
        /// Unit normal of the plane: the probe integrates the dome over the hemisphere around it.
        /// A window's probe uses its OUTWARD normal (the sky it looks at).
        public var normal: SIMD3<Float>
        /// Optional cone left out of the integral — e.g. the dome's stylised sun disc and halo,
        /// whose light a host already carries as its directional (nil = none).
        public var excludeDirection: SIMD3<Float>?
        /// Half-angle of the excluded cone (radians).
        public var excludeHalfAngle: Float

        public init(normal: SIMD3<Float>, excludeDirection: SIMD3<Float>? = nil, excludeHalfAngle: Float = 0) {
            self.normal = normal
            self.excludeDirection = excludeDirection
            self.excludeHalfAngle = excludeHalfAngle
        }
    }

    /// One completed bake's measurements, in `radianceProbes` order at the time of the bake.
    public struct RadianceProbeReading: Sendable, Equatable {
        /// The probes this reading measured.
        public var probes: [RadianceProbe]
        /// Per probe: E(n)/π in the dome's linear radiance units (RGB).
        public var radiance: [SIMD3<Float>]
        /// Per probe: the fraction of the 4096 samples kept (1 − the excluded cone's share).
        public var keptFraction: [Float]
        /// Increments with every completed bake that carried probes.
        public var generation: UInt64
    }

    /// The newest completed reading (nil until the first probed bake completes on the GPU).
    public var latestRadianceProbeReading: RadianceProbeReading? { radianceProbeStore.latest }

    /// The `skyRadianceProbe` kernel's `buffer(0)`: per probe, float4(normal, 0) then
    /// float4(excluded direction, cos half-angle) — w = 2 (> 1) when nothing is excluded.
    nonisolated static func radianceProbeGPUData(_ probes: [RadianceProbe]) -> [SIMD4<Float>] {
        var gpu: [SIMD4<Float>] = []
        gpu.reserveCapacity(probes.count * 2)
        for p in probes {
            let n = simd_length(p.normal) > 0 ? simd_normalize(p.normal) : SIMD3<Float>(0, 1, 0)
            gpu.append(SIMD4(n, 0))
            if let e = p.excludeDirection, simd_length(e) > 0, p.excludeHalfAngle > 0 {
                gpu.append(SIMD4(simd_normalize(e), cos(p.excludeHalfAngle)))
            } else {
                gpu.append(SIMD4(0, 0, 0, 2))
            }
        }
        return gpu
    }

    /// Encode the probe pass after the dome in `cmd` (no-op when no probes are set).
    func encodeRadianceProbes(into cmd: MTLCommandBuffer) {
        let probes = radianceProbes
        guard !probes.isEmpty,
              let pso = pipelineCache.pipelineState(name: "skyRadianceProbe", device: device) else { return }
        if radianceProbeRing.isEmpty {
            // Four slots: the dome's in-flight cap is two, so a slot is never rewritten while its
            // completion handler may still be reading it.
            for i in 0..<4 {
                guard let b = device.makeBuffer(length: Self.maxRadianceProbes * MemoryLayout<SIMD4<Float>>.stride,
                                                options: .storageModeShared) else { return }
                b.label = "VolumetricCloudRenderer.radianceProbe\(i)"
                radianceProbeRing.append(b)
            }
        }
        let out = radianceProbeRing[radianceProbeRingIndex]
        radianceProbeRingIndex = (radianceProbeRingIndex + 1) % radianceProbeRing.count
        let gpu = Self.radianceProbeGPUData(probes)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.label = "skyRadianceProbe"
        enc.setComputePipelineState(pso)
        enc.setTexture(outputTexture, index: 0)
        gpu.withUnsafeBytes { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 0)
        }
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: probes.count, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        radianceProbeGeneration &+= 1
        let generation = radianceProbeGeneration
        let store = radianceProbeStore
        let count = probes.count
        cmd.addCompletedHandler { cb in
            guard cb.status == .completed else { return }
            let ptr = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
            var radiance: [SIMD3<Float>] = []
            var kept: [Float] = []
            for i in 0..<count {
                let v = ptr[i]
                radiance.append(SIMD3(v.x, v.y, v.z))
                kept.append(v.w)
            }
            store.publish(RadianceProbeReading(probes: probes, radiance: radiance,
                                               keptFraction: kept, generation: generation))
        }
    }
}

/// Lock-protected newest reading: written from a Metal completion handler, read on the main actor.
/// Keeps only the highest generation, so a bake completing out of order never rolls it back.
final class RadianceProbeStore: Sendable {
    private let state = OSAllocatedUnfairLock<VolumetricCloudRenderer.RadianceProbeReading?>(initialState: nil)

    var latest: VolumetricCloudRenderer.RadianceProbeReading? { state.withLock { $0 } }

    func publish(_ r: VolumetricCloudRenderer.RadianceProbeReading) {
        state.withLock { current in
            if current.map({ $0.generation < r.generation }) ?? true { current = r }
        }
    }
}
