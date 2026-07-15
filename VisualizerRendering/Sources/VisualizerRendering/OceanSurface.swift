#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#else
import UIKit
#endif
import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore
import simd

// ── OCEAN SURFACE ───────────────────────────────────────────────────────────
//
// A GPU-driven Gerstner-wave heightfield ocean. One resolution² shared
// MTLBuffer holds (position, normal) per grid vertex; `Ocean.metal` rewrites
// every vertex each tick from a stack of 4–8 traveling sine waves. SceneKit
// reads positions and normals straight out of the buffer via standard
// `SCNGeometrySource`s — same zero-copy pattern as `FireworkParticleRenderer`
// and `DynamicMesh`. No CPU snapshot per frame.
//
// USAGE
//
//   let ocean = OceanSurface(
//       engine: SimEngine.shared,
//       resolution: 192,
//       worldSize: 240,
//       waves: OceanSurface.defaultWaves(scale: 1.0, primaryDirection: SIMD2(1, 0.2))
//   )
//   scene.rootNode.addChildNode(ocean.node)
//
//   // every tick, in the SAME command buffer as other GPU work for this scene:
//   ocean.encodeStep(into: cb, dt: dt, amplitudeScale: 1.0)
//
// SCOPE
//
// Single shared heightfield, no tiling — the surface is one big square plane
// centred at the configured origin. The sweet spot at default settings is
// ~240 m square at 192² verts ≈ 37k verts / 73k tris, which fits comfortably
// inside the per-tick compute budget while still resolving 4 m wavelengths.
// Camera-relative tiling and FFT-based wavelength stacks are the natural v2.
//
// MATERIAL
//
// Constructed inside `init` and returned via `node.geometry?.firstMaterial`.
// PBR water tuned for the project's HDR + bloom pipeline:
//   • diffuse  — deep midnight blue (almost black) so the IBL specular reads
//                as the dominant surface response
//   • metalness — moderate, to push the response toward dielectric-water
//                specular without the silvery look of pure metal
//   • roughness — low, so the IBL highlights are crisp pinpoint glints
//                rather than a smeared sheen
//   • emission — black; the surface gets all its light from the IBL +
//                whatever scene lights are above
//   • normal source — analytic from the Gerstner kernel, no normal map needed
//
// PlanarRT reflections of the firework bursts can be added on top as a future
// pass — wire the reflection MTLTexture into a shader modifier that perturbs
// the lookup by the surface normal. Out of scope for this initial pass; the
// IBL + bloom + low roughness combination already gives the night-water look
// when fireworks are above.

@MainActor
public final class OceanSurface {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "OceanSurface")

    public let node: SCNNode
    public let geometry: SCNGeometry
    public let material: SCNMaterial
    public let engine: SimEngine

    public let resolution: Int
    public let worldSize: Float
    public let origin: SIMD2<Float>

    /// Wall-clock time fed to the kernel. Advanced by `encodeStep(dt:)`.
    public private(set) var time: Float = 0

    /// Bound at init; can be reconfigured via `setWaves(_:)`. Caller owns the
    /// distribution; `defaultWaves(...)` gives a reasonable starter set.
    public private(set) var waves: [OceanWave]

    private let device: MTLDevice
    private let pipeline: MTLComputePipelineState
    private let vertexBuffer: SimBuffer<OceanVertex>
    private let indexBuffer: MTLBuffer
    private let waveBuffer: MTLBuffer
    private let uniformBuffer: MTLBuffer

    /// Optional shared burst-light field driving per-vertex specular speckle.
    /// Set via `attachBurstField(_:)`; when nil, the kernel writes zero
    /// emission and the additive overlay contributes nothing.
    public private(set) var burstField: BurstLightField?
    /// "Dummy" buffer bound at buffer-slot 3 when no burst field is attached.
    /// Metal requires every shader argument to be backed by something; a
    /// 1-element zero-stride buffer keeps the kernel happy without branching.
    private let burstFallbackBuffer: MTLBuffer
    /// Additive emission overlay geometry — shares the vertex buffer with the
    /// PBR water geometry, but uses position + per-vertex color sources to
    /// composite a constant-lit additive pass on top. The kernel writes the
    /// per-vertex emission into the vertex buffer; SceneKit picks it up via
    /// the .color semantic source.
    public let emissionNode: SCNNode

    // ── Construction ─────────────────────────────────────────────

    public init?(
        engine: SimEngine,
        resolution: Int = 192,
        worldSize: Float = 240,
        origin: SIMD2<Float> = SIMD2(0, 0),
        waves: [OceanWave]? = nil
    ) {
        precondition(resolution >= 2 && resolution <= 1024, "Ocean resolution out of sane range")

        let device = engine.device
        guard let pipeline = engine.pipeline("oceanBuild") else {
            Self.log.error("oceanBuild pipeline lookup failed — is Ocean.metal in VisualizerRendering/Shaders/?")
            return nil
        }

        let vertCount = resolution * resolution
        let quadCount = (resolution - 1) * (resolution - 1)
        let triCount  = quadCount * 2
        let indexCount = triCount * 3

        guard
            let vBuf = SimBuffer<OceanVertex>(device: device,
                                              capacity: vertCount,
                                              label: "Ocean.vertices"),
            let iBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * indexCount,
                                          options: .storageModeShared),
            let wBuf = device.makeBuffer(length: MemoryLayout<OceanWave>.stride * Self.maxWaves,
                                          options: .storageModeShared),
            let uBuf = device.makeBuffer(length: MemoryLayout<OceanUniforms>.stride,
                                          options: .storageModeShared),
            // 48-byte zeroed fallback so the kernel can read buffer(3) even
            // when no burst field is attached (burstCount uniform = 0 makes
            // the inner loop a no-op, but the binding must exist).
            let fBuf = device.makeBuffer(length: 48,
                                         options: .storageModeShared)
        else {
            Self.log.error("OceanSurface buffer allocation failed")
            return nil
        }
        iBuf.label = "Ocean.indices"
        wBuf.label = "Ocean.waves"
        uBuf.label = "Ocean.uniforms"
        fBuf.label = "Ocean.burstFallback"

        // Seed the vertex buffer with a flat grid so the geometry has a valid
        // shape before the first kernel run (otherwise zero-initialised float4
        // positions would collapse every vertex to the origin).
        let rowStride = (Float(worldSize)) / Float(resolution - 1)
        var seed = [OceanVertex](repeating: OceanVertex(), count: vertCount)
        for iz in 0..<resolution {
            for ix in 0..<resolution {
                let x = origin.x + (Float(ix) - Float(resolution - 1) * 0.5) * rowStride
                let z = origin.y + (Float(iz) - Float(resolution - 1) * 0.5) * rowStride
                seed[iz * resolution + ix] = OceanVertex(
                    position: SIMD4(x, 0, z, 1),
                    normal:   SIMD4(0, 1, 0, 0)
                )
            }
        }
        seed.withUnsafeBufferPointer { p in
            vBuf.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<OceanVertex>.stride * vertCount
            )
        }

        // Pre-fill triangle indices. CW winding to match SceneKit's custom-
        // geometry front-face convention (see [[scn_geometry_winding.md]]):
        // per quad the two triangles are (a, c, b) + (a, d, c) where
        // a=(ix,iz), b=(ix+1,iz), c=(ix+1,iz+1), d=(ix,iz+1).
        let idxPtr = iBuf.contents().bindMemory(to: UInt32.self, capacity: indexCount)
        var dst = 0
        for iz in 0..<(resolution - 1) {
            for ix in 0..<(resolution - 1) {
                let a = UInt32(iz * resolution + ix)
                let b = a + 1
                let c = a + UInt32(resolution) + 1
                let d = a + UInt32(resolution)
                idxPtr[dst + 0] = a
                idxPtr[dst + 1] = c
                idxPtr[dst + 2] = b
                idxPtr[dst + 3] = a
                idxPtr[dst + 4] = d
                idxPtr[dst + 5] = c
                dst += 6
            }
        }

        let vStride = MemoryLayout<OceanVertex>.stride
        let posSource = SCNGeometrySource(
            buffer: vBuf.buffer,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: vertCount,
            dataOffset: 0,
            dataStride: vStride
        )
        let nrmSource = SCNGeometrySource(
            buffer: vBuf.buffer,
            vertexFormat: .float3,
            semantic: .normal,
            vertexCount: vertCount,
            dataOffset: 16,
            dataStride: vStride
        )

        let element = SCNGeometryElement(
            buffer: iBuf,
            primitiveType: .triangles,
            primitiveCount: triCount,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )

        let geom = SCNGeometry(sources: [posSource, nrmSource], elements: [element])
        let mat = Self.makeWaterMaterial()
        geom.materials = [mat]

        // Additive EMISSION OVERLAY — shares the same vertex buffer + index
        // buffer as the PBR water; reads positions (same offset 0) and the
        // per-vertex emission color the kernel writes at byte 32. Constant-
        // lit additive material is exactly the "the burst is lighting up
        // each wave facet that happens to mirror it toward the camera" look.
        // Because the geometry rides on the WAVE-displaced positions (not
        // a flat plane), the speckle follows the actual wave silhouettes
        // — the missing piece that made the prior overlay look like an oil
        // puddle.
        let emissionColSource = SCNGeometrySource(
            buffer: vBuf.buffer,
            vertexFormat: .float3,
            semantic: .color,
            vertexCount: vertCount,
            dataOffset: 32,
            dataStride: vStride
        )
        let emissionElement = SCNGeometryElement(
            buffer: iBuf,
            primitiveType: .triangles,
            primitiveCount: triCount,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )
        let emissionGeom = SCNGeometry(
            sources: [posSource, emissionColSource],
            elements: [emissionElement]
        )
        let emissionMat = SCNMaterial()
        emissionMat.lightingModel = .constant
        emissionMat.diffuse.contents = PlatformColor.white   // per-vertex color modulates
        emissionMat.blendMode = .add
        emissionMat.writesToDepthBuffer = false
        emissionMat.readsFromDepthBuffer = true
        emissionMat.isDoubleSided = true
        emissionGeom.materials = [emissionMat]
        let emissionNode = SCNNode(geometry: emissionGeom)
        emissionNode.name = "OceanSurface.burstEmission"
        emissionNode.castsShadow = false
        emissionNode.renderingOrder = 5  // draw after the PBR water

        let node = SCNNode(geometry: geom)
        node.name = "OceanSurface"
        node.castsShadow = false
        node.addChildNode(emissionNode)
        // The Gerstner displacement adds up to ~4× the largest single amplitude
        // in extremes. Pad generously so the bbox doesn't have to be touched
        // when waves are tuned bigger.
        let half = worldSize * 0.5 + 8
        let yMax: Float = 12
        node.boundingBox = (
            min: SCNVector3(CGFloat(origin.x - half), CGFloat(-yMax), CGFloat(origin.y - half)),
            max: SCNVector3(CGFloat(origin.x + half), CGFloat( yMax), CGFloat(origin.y + half))
        )

        self.engine               = engine
        self.device               = device
        self.pipeline             = pipeline
        self.vertexBuffer         = vBuf
        self.indexBuffer          = iBuf
        self.waveBuffer           = wBuf
        self.uniformBuffer        = uBuf
        self.burstFallbackBuffer  = fBuf
        self.emissionNode         = emissionNode
        self.resolution           = resolution
        self.worldSize            = worldSize
        self.origin               = origin
        self.geometry             = geom
        self.material             = mat
        self.node                 = node
        self.waves                = []

        let chosen = waves ?? Self.defaultWaves(scale: 1.0, primaryDirection: SIMD2(1, 0.2))
        setWaves(chosen)
    }

    // ── Wave configuration ───────────────────────────────────────

    /// Maximum number of Gerstner components packed into the wave MTLBuffer.
    /// Tradeoff: each wave is N more sin/cos pairs per vertex per frame. 8 is
    /// plenty for the "ocean at night with fireworks" look — two long swell
    /// waves + four mid waves + two ripple waves.
    public static let maxWaves: Int = 8

    public func setWaves(_ newWaves: [OceanWave]) {
        let clamped = Array(newWaves.prefix(Self.maxWaves))
        var copy = clamped
        while copy.count < Self.maxWaves {
            // Pad with zero-amplitude waves so the kernel doesn't read garbage
            // past the active count. The kernel respects `waveCount` and won't
            // touch these, but keeping them zero is defensive.
            copy.append(OceanWave(direction: SIMD2(1, 0),
                                  amplitude: 0,
                                  wavelength: 1,
                                  speed: 1,
                                  steepness: 0))
        }
        copy.withUnsafeBufferPointer { p in
            waveBuffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<OceanWave>.stride * Self.maxWaves
            )
        }
        self.waves = clamped
    }

    /// A serviceable default ocean: two long low-frequency swells crossing at
    /// ~30°, a few mid-frequency chop waves, and small ripples. All flowing
    /// roughly in `primaryDirection` (xz unit vector), with the cross-waves
    /// offset by ±20–40° so the surface doesn't read as a 1-D corduroy.
    public static func defaultWaves(
        scale: Float = 1.0,
        primaryDirection: SIMD2<Float> = SIMD2(1, 0.2)
    ) -> [OceanWave] {
        let primary = simd_normalize(primaryDirection)
        // Build an orthonormal helper so cross-waves stay in the xz plane.
        let perp = SIMD2<Float>(-primary.y, primary.x)
        func rot(_ d: SIMD2<Float>, _ deg: Float) -> SIMD2<Float> {
            let r = deg * .pi / 180
            let c = cos(r), s = sin(r)
            return SIMD2(c * d.x - s * d.y, s * d.x + c * d.y)
        }
        _ = perp

        // Wavelengths chosen so the largest swell sits a touch under 1/4 of a
        // 240 m surface — long enough to feel oceanic, short enough that we
        // still see ~3 crests across the frame at default camera distance.
        let s = scale
        return [
            // Long swells (low frequency, moderate amplitude).
            OceanWave(direction: rot(primary, -8),  amplitude: 0.55 * s, wavelength: 38, speed: 4.5, steepness: 0.55),
            OceanWave(direction: rot(primary,  22), amplitude: 0.40 * s, wavelength: 26, speed: 3.8, steepness: 0.55),
            // Mid chop.
            OceanWave(direction: rot(primary, -38), amplitude: 0.22 * s, wavelength: 14, speed: 2.8, steepness: 0.5),
            OceanWave(direction: rot(primary,  46), amplitude: 0.18 * s, wavelength: 10.5, speed: 2.4, steepness: 0.5),
            OceanWave(direction: rot(primary,   6), amplitude: 0.14 * s, wavelength: 7.5, speed: 2.0, steepness: 0.45),
            // Small ripples — keep steepness low so they don't pinch sharply.
            OceanWave(direction: rot(primary,  78), amplitude: 0.07 * s, wavelength: 3.6, speed: 1.5, steepness: 0.3),
            OceanWave(direction: rot(primary, -64), amplitude: 0.06 * s, wavelength: 2.6, speed: 1.3, steepness: 0.3),
            OceanWave(direction: rot(primary,  18), amplitude: 0.04 * s, wavelength: 1.8, speed: 1.05, steepness: 0.25),
        ]
    }

    /// Attach a shared `BurstLightField` so the ocean kernel can light each
    /// wave vertex from the active bursts via per-vertex Blinn-Phong
    /// specular. Pass nil (or skip the call) for a non-burst-lit ocean.
    public func attachBurstField(_ field: BurstLightField?) {
        self.burstField = field
    }

    // ── Per-tick step ────────────────────────────────────────────

    /// Encode one ocean rebuild pass. Advances internal time, writes the
    /// uniform block, dispatches `oceanBuild` over the resolution² grid. The
    /// caller commits the command buffer (typically batched with the firework
    /// integrate + streak passes).
    public func encodeStep(
        into cb: MTLCommandBuffer,
        dt: Float,
        amplitudeScale: Float = 1.0,
        cameraPosition: SIMD3<Float> = SIMD3(0, 0, 0)
    ) {
        time += dt
        let res = UInt32(resolution)
        let waveCount = UInt32(waves.count)
        let burstCount = UInt32(burstField?.activeCount ?? 0)
        let resAsFloat   = Float(bitPattern: res)
        let countAsFloat = Float(bitPattern: waveCount)
        let burstAsFloat = Float(bitPattern: burstCount)

        var u = OceanUniforms(
            originSizeRes: SIMD4(origin.x, origin.y, worldSize, resAsFloat),
            timeCount:     SIMD4(time, countAsFloat, amplitudeScale, burstAsFloat),
            cameraPos:     SIMD4(cameraPosition.x, cameraPosition.y, cameraPosition.z, 0)
        )
        uniformBuffer.contents()
            .copyMemory(from: &u, byteCount: MemoryLayout<OceanUniforms>.stride)

        let total = resolution * resolution
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Ocean.build"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(vertexBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(waveBuffer,          offset: 0, index: 1)
        enc.setBuffer(uniformBuffer,       offset: 0, index: 2)
        // Burst-light buffer at slot 3. Use the fallback (1-element zeroed
        // buffer) when no field is attached so the kernel arg binding is
        // always populated; the kernel skips the inner loop when burstCount=0.
        if let burstField = burstField, burstField.activeCount > 0 {
            enc.setBuffer(burstField.buffer.buffer, offset: 0, index: 3)
        } else {
            enc.setBuffer(burstFallbackBuffer, offset: 0, index: 3)
        }
        let tg = min(total, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
    }

    // ── Default water material ───────────────────────────────────

    private static func makeWaterMaterial() -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        // Almost-black base — at night, the diffuse contribution is supposed
        // to be a sliver; the IBL specular off the wave normals is what
        // carries the surface.
        m.diffuse.contents  = PlatformColor(calibratedRed: 0.005, green: 0.012, blue: 0.025, alpha: 1)
        m.metalness.contents = 0.04
        m.roughness.contents = 0.06
        m.isDoubleSided      = false
        // Light-purplish glint colour for the specular tint — matches the
        // existing night-sky IBL's cool cast so highlights don't feel
        // disconnected from the environment.
        m.fresnelExponent    = 1.4
        return m
    }
}

// ── Shared structs ──────────────────────────────────────────────────────────

/// Mirror of `OceanVertex` in Ocean.metal. 48 bytes, all float4 per the
/// project alignment rule. Layout:
///   bytes  0..<12  = position xyz (SCN vertex source)
///   bytes 16..<28  = normal xyz   (SCN normal source)
///   bytes 32..<44  = emission rgb (SCN color source on the additive overlay
///                                  geometry — per-vertex burst-light specular
///                                  contribution written by the kernel)
public struct OceanVertex {
    public var position:      SIMD4<Float>
    public var normal:        SIMD4<Float>
    public var emissionColor: SIMD4<Float>

    public init(position: SIMD4<Float> = SIMD4(0, 0, 0, 1),
                normal:   SIMD4<Float> = SIMD4(0, 1, 0, 0),
                emissionColor: SIMD4<Float> = SIMD4(0, 0, 0, 0)) {
        self.position      = position
        self.normal        = normal
        self.emissionColor = emissionColor
    }
}

/// One Gerstner wave component. Direction is a unit xz vector (will be
/// re-normalised in the initializer). Steepness Q controls how peaked the
/// crests are — 0 = pure cosine bobbing, 1 = sharply peaked (and visible
/// trochoid loops if you push it too far). Keep below ~0.7 for clean,
/// non-self-intersecting surfaces.
public struct OceanWave {
    public var dirAmpWavelen: SIMD4<Float>
    public var speedSteepPad: SIMD4<Float>

    public init(direction: SIMD2<Float>,
                amplitude: Float,
                wavelength: Float,
                speed: Float,
                steepness: Float) {
        let n = simd_length(direction) > 1e-5 ? simd_normalize(direction) : SIMD2(1, 0)
        self.dirAmpWavelen = SIMD4(n.x, n.y, amplitude, wavelength)
        self.speedSteepPad = SIMD4(speed, max(0, min(1, steepness)), 0, 0)
    }
}

struct OceanUniforms {
    var originSizeRes: SIMD4<Float>
    var timeCount:     SIMD4<Float>     // w now = burst count
    var cameraPos:     SIMD4<Float>
}
