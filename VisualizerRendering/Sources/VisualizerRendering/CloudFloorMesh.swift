import AppKit
import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore
import simd

// ── CLOUD FLOOR MESH (Option B — real 3D cumulus geometry) ──────────────────
//
// GPU-driven heightfield cumulus deck sitting beneath the firework burst
// zone. Real 3D triangles in world space — shells fly through actual
// geometry with SceneKit's depth buffer handling occlusion natively, and
// bursts above light the cloud TOPS via per-vertex Lambert N·L (no
// triangulation artifact because diffuse lighting is broad — unlike the
// ocean's specular response that revealed the mesh density as visible
// triangles).
//
// ── ARCHITECTURE: how this got here ─────────────────────────────────────────
//
// Three options were considered for "fireworks coming through clouds":
//
// • **Option A (deleted)** — screen-space raymarched overlay quad.
//   `VolumetricCloudFloorRenderer` + `Shaders/CloudFloor.metal`. Camera-
//   attached overlay plane sampled a per-pixel raymarched cumulus density.
//   Cheap to write but reads as a flat 2D sheet because that's literally
//   what it is, and never depth-tests against scene geometry (shells
//   "behind" the cloud overlay still composited over). Removed in favour
//   of this geometry-based approach.
//
// • **Option B (THIS FILE)** — GPU-driven heightfield mesh in world space.
//   One thread per grid vertex generates a cumulus height + normal + per-
//   vertex burst-light emission. SceneKit renders the cloud as opaque-ish
//   PBR triangles with depth testing; the additive emission overlay
//   carries the burst-driven cloud-top illumination. Shells visibly fly
//   through the mesh, bursts above tint the cloud tops in their colour,
//   and the per-vertex Lambert smooths over the tessellation (no "matrix
//   triangulation" artifact). This is the right tool for SceneKit's
//   opaque-geometry pipeline.
//
// • **Option C (TODO)** — true volumetric with depth integration. Would
//   require an `SCNRenderer` delegate that captures the scene's depth
//   buffer and feeds it to a fresh raymarch kernel that terminates each
//   march at scene depth. Cloud kernels then properly occlude with scene
//   geometry, and shells inside the cloud get genuine volumetric
//   absorption + scatter on the right side of opaque elements. Days of
//   work — needs custom render passes, careful synchronisation with
//   SceneKit's frame timing, and a swap-chain-aware depth read.
//   Implement when the project graduates beyond SceneKit's render pipeline
//   (or when it really, really matters for a single hero shot).
//
// ── IMPLEMENTATION ─────────────────────────────────────────────────────────
//
// Two SCNGeometries share the same vertex MTLBuffer (same pattern the ocean
// uses for its burst-emission overlay):
//
//   1. **Cloud body** — PBR-ish material. White-grey diffuse, high
//      roughness, low metalness. Position + normal sources. Renders during
//      the normal opaque pass with depth writes ON.
//   2. **Emission overlay** — constant-lit additive material. Position +
//      `.color` sources reading the per-vertex emission the kernel writes.
//      Renders AFTER the body so it composites bright burst-tint on top.
//
// The build kernel runs every tick. It's cheap enough (256² verts × few ops
// per vertex × few active bursts) that there's no need to split static vs
// dynamic passes — full rebuild each tick keeps the wind-drifted cloud
// shape live and the burst lighting current.

@MainActor
public final class CloudFloorMesh {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "CloudFloorMesh")

    public let device: MTLDevice
    public let engine: SimEngine
    public let burstField: BurstLightField

    /// Parent SCNNode — add this to the scene root. Contains both the
    /// cloud body geometry and the additive emission overlay as children.
    public let node: SCNNode

    /// Cloud-body geometry (PBR-ish). Position + normal off the shared
    /// vertex buffer. Renders during the opaque pass.
    public let bodyGeometry: SCNGeometry
    public let bodyMaterial: SCNMaterial

    /// Additive emission overlay — same vertex buffer + same triangles,
    /// reads the per-vertex emission color via `.color` semantic. Renders
    /// AFTER the body so burst tints composite on top of the cloud surface.
    public let emissionGeometry: SCNGeometry
    public let emissionMaterial: SCNMaterial

    public let resolution: Int
    public let worldSize: Float
    public let origin: SIMD2<Float>

    private let pipeline: MTLComputePipelineState
    private let noiseVolume: NoiseVolume
    private let vertexBuffer: SimBuffer<CloudFloorVertex>
    private let indexBuffer: MTLBuffer
    private let uniformBuffer: MTLBuffer

    private var time: Float = 0

    // ── Cloud parameters (live-tunable) ─────────────────────────────────────
    public var baseY: Float = 4
    public var thickness: Float = 3.5
    /// Maps world XZ → noise UV. Lower = bigger cloud features. 1.5 gives
    /// noise period ~85m, with FBM's dominant octave at ~42m and detail
    /// down to ~10m — at this scale cumulus mounds read as rounded forms
    /// rather than fractal-noise spikes.
    public var horizontalScale: Float = 1.5
    /// Fraction of sky covered with cloud. 1.0 = total overcast, 0.0 = no
    /// cloud at all (height=0 everywhere).
    public var coverage: Float = 0.50
    /// Shape curve exponent. Pow > 1 rounds peaks DOWN (smoother, bumpier
    /// cumulus); pow < 1 makes them cone-like. 1.3 = rounded cauliflower
    /// shape — the classic cumulus silhouette.
    public var shapeCurve: Float = 1.3
    /// XZ wind direction the cloud field drifts along. Drift is slow
    /// (`0.05` multiplier in the kernel) — clouds barely move over a
    /// minute-scale show.
    public var windDirection: SIMD2<Float> = SIMD2(0.85, 0.52)
    /// Burst-light contribution gain. KEEP LOW — with intensity 8 ×
    /// Lambert N·L ≈ 1 × falloff ≈ 0.8, even a single burst contributes
    /// ~6 of HDR brightness at gain 1.0; bloom amplifies that to pure
    /// white. 0.15 lets bursts tint the cloud noticeably without blowing
    /// out, and multiple stacked bursts still stay readable.
    public var burstLightingGain: Float = 0.15

    public init?(
        engine: SimEngine,
        burstField: BurstLightField,
        resolution: Int = 256,
        worldSize: Float = 280,
        origin: SIMD2<Float> = SIMD2(0, 0)
    ) {
        precondition(resolution >= 4 && resolution <= 1024, "CloudFloorMesh: resolution out of range")
        guard let pipeline = engine.pipeline("cloudFloorMeshBuild") else {
            Self.log.error("cloudFloorMeshBuild pipeline lookup failed — is CloudFloorMesh.metal in Shaders/?")
            return nil
        }
        let device = engine.device

        let vertCount = resolution * resolution
        let quadCount = (resolution - 1) * (resolution - 1)
        let triCount = quadCount * 2
        let indexCount = triCount * 3

        guard
            let vBuf = SimBuffer<CloudFloorVertex>(
                device: device, capacity: vertCount, label: "CloudFloorMesh.vertices"
            ),
            let iBuf = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride * indexCount,
                options: .storageModeShared
            ),
            let uBuf = device.makeBuffer(
                length: MemoryLayout<CloudMeshUniforms>.stride,
                options: .storageModeShared
            )
        else {
            Self.log.error("CloudFloorMesh buffer allocation failed")
            return nil
        }
        iBuf.label = "CloudFloorMesh.indices"
        uBuf.label = "CloudFloorMesh.uniforms"

        // Seed vertex buffer with flat plane (will be overwritten by the
        // first kernel dispatch, but matters for the very first frame in
        // case rendering happens before the kernel runs).
        let stride = Float(worldSize) / Float(resolution - 1)
        var seed = [CloudFloorVertex](repeating: CloudFloorVertex(), count: vertCount)
        for iz in 0..<resolution {
            for ix in 0..<resolution {
                let x = origin.x + (Float(ix) - Float(resolution - 1) * 0.5) * stride
                let z = origin.y + (Float(iz) - Float(resolution - 1) * 0.5) * stride
                seed[iz * resolution + ix] = CloudFloorVertex(
                    position: SIMD4(x, 5, z, 1),
                    normal: SIMD4(0, 1, 0, 0),
                    emission: SIMD4(0, 0, 0, 0)
                )
            }
        }
        seed.withUnsafeBufferPointer { p in
            vBuf.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<CloudFloorVertex>.stride * vertCount
            )
        }

        // Pre-fill triangle indices. CW winding to match SceneKit's
        // custom-geometry convention (per [scn_geometry_winding.md]).
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

        let vStride = MemoryLayout<CloudFloorVertex>.stride
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
        let emiSource = SCNGeometrySource(
            buffer: vBuf.buffer,
            vertexFormat: .float3,
            semantic: .color,
            vertexCount: vertCount,
            dataOffset: 32,
            dataStride: vStride
        )

        let element = SCNGeometryElement(
            buffer: iBuf,
            primitiveType: .triangles,
            primitiveCount: triCount,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )

        // Body geometry — PBR cloud. Diffuse a neutral cloud grey, high
        // roughness so any IBL/scene-light bounce is soft.
        let bodyGeom = SCNGeometry(sources: [posSource, nrmSource], elements: [element])
        let bodyMat = SCNMaterial()
        bodyMat.lightingModel = .physicallyBased
        bodyMat.diffuse.contents = NSColor(calibratedWhite: 0.68, alpha: 1)
        bodyMat.metalness.contents = 0.0
        bodyMat.roughness.contents = 1.0
        bodyMat.isDoubleSided = true
        // Slight emission so the cloud is visible at night even with no
        // direct light — same role the ambient-tint had in the screen-
        // space overlay.
        bodyMat.emission.contents = NSColor(calibratedRed: 0.10, green: 0.12, blue: 0.16, alpha: 1)
        bodyGeom.materials = [bodyMat]

        let bodyNode = SCNNode(geometry: bodyGeom)
        bodyNode.name = "CloudFloorMesh.body"
        bodyNode.castsShadow = false

        // Emission overlay — same vertex buffer, samples color source for
        // additive burst-light. Renders AFTER body so the additive contrib
        // composites over the PBR shading.
        let emissionElement = SCNGeometryElement(
            buffer: iBuf,
            primitiveType: .triangles,
            primitiveCount: triCount,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )
        let emissionGeom = SCNGeometry(
            sources: [posSource, emiSource],
            elements: [emissionElement]
        )
        let emissionMat = SCNMaterial()
        emissionMat.lightingModel = .constant
        emissionMat.diffuse.contents = NSColor.white
        emissionMat.blendMode = .add
        emissionMat.writesToDepthBuffer = false
        emissionMat.readsFromDepthBuffer = true
        emissionMat.isDoubleSided = true
        emissionGeom.materials = [emissionMat]

        let emissionNode = SCNNode(geometry: emissionGeom)
        emissionNode.name = "CloudFloorMesh.emission"
        emissionNode.castsShadow = false
        emissionNode.renderingOrder = 5

        let parent = SCNNode()
        parent.name = "CloudFloorMesh"
        parent.castsShadow = false
        parent.addChildNode(bodyNode)
        parent.addChildNode(emissionNode)

        // Bounding box covers the full extent of the heightfield plus the
        // worst-case cloud thickness lift.
        let half = worldSize * 0.5 + 8
        parent.boundingBox = (
            min: SCNVector3(CGFloat(origin.x - half), -2, CGFloat(origin.y - half)),
            max: SCNVector3(CGFloat(origin.x + half), 20, CGFloat(origin.y + half))
        )

        self.device = device
        self.engine = engine
        self.burstField = burstField
        self.resolution = resolution
        self.worldSize = worldSize
        self.origin = origin
        self.pipeline = pipeline
        self.noiseVolume = NoiseVolume(engine: engine)
        self.vertexBuffer = vBuf
        self.indexBuffer = iBuf
        self.uniformBuffer = uBuf
        self.bodyGeometry = bodyGeom
        self.bodyMaterial = bodyMat
        self.emissionGeometry = emissionGeom
        self.emissionMaterial = emissionMat
        self.node = parent
    }

    public func encodeStep(into cb: MTLCommandBuffer, dt: Float) {
        time += dt
        let res = UInt32(resolution)
        let burstCount = UInt32(burstField.activeCount)
        let resAsFloat = Float(bitPattern: res)
        let burstAsFloat = Float(bitPattern: burstCount)
        let windDir = simd_normalize(windDirection)

        var u = CloudMeshUniforms(
            originSizeRes: SIMD4(origin.x, origin.y, worldSize, resAsFloat),
            baseShapeWind: SIMD4(baseY, thickness, horizontalScale, time),
            coverageGain: SIMD4(coverage, shapeCurve, burstLightingGain, burstAsFloat),
            windDir: SIMD4(windDir.x, windDir.y, 0, 0)
        )
        uniformBuffer.contents().copyMemory(
            from: &u, byteCount: MemoryLayout<CloudMeshUniforms>.stride
        )

        let total = resolution * resolution
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "CloudFloorMesh.build"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(vertexBuffer.buffer,      offset: 0, index: 0)
        enc.setBuffer(uniformBuffer,            offset: 0, index: 1)
        enc.setBuffer(burstField.buffer.buffer, offset: 0, index: 2)
        enc.setTexture(noiseVolume.texture,     index: 0)
        let tg = min(total, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
        )
        enc.endEncoding()
    }
}

// ── Shared structs ──────────────────────────────────────────────────────────

/// Mirror of `CloudFloorVertex` in CloudFloorMesh.metal. 48 bytes, all
/// float4 per the project ALIGNMENT RULE. SceneKit reads:
///   bytes  0..<12 = position xyz (.vertex)
///   bytes 16..<28 = normal xyz   (.normal)
///   bytes 32..<44 = emission rgb (.color, on the emission overlay geometry)
public struct CloudFloorVertex {
    public var position: SIMD4<Float>
    public var normal:   SIMD4<Float>
    public var emission: SIMD4<Float>

    public init(
        position: SIMD4<Float> = SIMD4(0, 0, 0, 1),
        normal:   SIMD4<Float> = SIMD4(0, 1, 0, 0),
        emission: SIMD4<Float> = SIMD4(0, 0, 0, 0)
    ) {
        self.position = position
        self.normal = normal
        self.emission = emission
    }
}

struct CloudMeshUniforms {
    var originSizeRes: SIMD4<Float>
    var baseShapeWind: SIMD4<Float>
    var coverageGain:  SIMD4<Float>
    var windDir:       SIMD4<Float>
}
