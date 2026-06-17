import AppKit
import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore
import simd

// ── VOLUMETRIC SMOKE RENDERER ───────────────────────────────────────────────
//
// Real raymarched smoke for the Fireworks+ scene. Replaces the prior
// camera-facing billboard smoke (flat sprites, no through-volume light
// transport). Uses the same toolkit as `VolumetricCloudRenderer`:
//   • Shared `NoiseVolume` (tileable 3D RGBA16F FBM+erosion) for fluffy shape
//   • Beer-Lambert ray accumulation through analytic puff density
//   • Short light-march per step for self-shadow + silver-lining transmittance
//   • HG phase function so smoke catches a brighter silver lining viewing
//     toward bright bursts
//
// The output goes into a screen-space MTLTexture composited via a
// camera-attached overlay quad (alpha-blended over the scene). Half-res
// (768×432) by default — the bilinear upscale softens the per-pixel noise
// into a "fluffy at distance" feel and keeps the per-frame budget under
// ~1.5 ms on an M-series GPU.
//
// LIFECYCLE
//
//   let smoke = VolumetricSmokeRenderer(engine: .shared,
//                                       burstField: burstField,
//                                       cameraNode: cameraNode)
//   scene.rootNode.addChildNode(smoke.overlayContainer)  // hosts the overlay
//
//   // Per burst:
//   smoke.spawnBurstResidue(at: burstCenter, radius: 4)
//   // Per tick on rising shells:
//   smoke.spawnShellContrail(at: shellPos)
//   // Per tick (in the same command buffer as the firework integrate):
//   smoke.encodeStep(into: cb, dt: dt, time: simTime)

@MainActor
public final class VolumetricSmokeRenderer {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "VolumetricSmokeRenderer")

    public let device: MTLDevice
    public let engine: SimEngine
    public let burstField: BurstLightField
    public let cameraNode: SCNNode

    /// The screen-space smoke buffer. Reference-stable; the overlay's
    /// material samples this each frame.
    public let outputTexture: MTLTexture

    /// Camera-child node containing the overlay quad. Add this (not the
    /// individual nodes) to the scene root so the overlay automatically
    /// follows the camera.
    public let overlayContainer: SCNNode

    /// MTLBuffer of currently-live smoke puffs. CPU-side mirror is kept so
    /// we can age + recompact in O(N) per tick.
    private let puffBuffer: SimBuffer<SmokePuff>
    private var puffMirror: [SmokePuff] = []
    public let capacity: Int

    private let pipeline: MTLComputePipelineState
    private let noiseVolume: NoiseVolume
    private let uniformBuffer: MTLBuffer
    private let outputWidth: Int
    private let outputHeight: Int
    /// Throttle counter — kernel runs every Nth frame to keep GPU budget
    /// safe even with many puffs. The output texture is reference-stable so
    /// in between encodes the overlay shows the previously-rendered smoke.
    private var encodeFrameCount: UInt = 0
    private let encodeEveryNFrames: UInt = 2

    public init?(
        engine: SimEngine,
        burstField: BurstLightField,
        cameraNode: SCNNode,
        capacity: Int = 32,
        outputWidth: Int = 512,
        outputHeight: Int = 288
    ) {
        guard let pipeline = engine.pipeline("smokeVolumeMarch") else {
            Self.log.error("smokeVolumeMarch pipeline lookup failed — is SmokeVolume.metal in Shaders/?")
            return nil
        }
        let device = engine.device

        // Output texture: rgba16Float so the HDR scattered colour doesn't
        // 8-bit-clip. Premultiplied alpha, sampled by the overlay material.
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: outputWidth,
            height: outputHeight,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else {
            Self.log.error("VolumetricSmokeRenderer texture allocation failed")
            return nil
        }
        tex.label = "VolumetricSmoke.output"

        guard let pBuf = SimBuffer<SmokePuff>(
            device: device, capacity: capacity, label: "VolumetricSmoke.puffs"
        ),
              let uBuf = device.makeBuffer(
                length: MemoryLayout<SmokeVolumeUniforms>.stride,
                options: .storageModeShared
              ) else {
            Self.log.error("VolumetricSmokeRenderer buffer allocation failed")
            return nil
        }
        uBuf.label = "VolumetricSmoke.uniforms"

        // Zero-seed the puff buffer.
        let zero = SmokePuff()
        var seed = Array(repeating: zero, count: capacity)
        seed.withUnsafeBufferPointer { p in
            pBuf.buffer.contents().copyMemory(
                from: p.baseAddress!,
                byteCount: MemoryLayout<SmokePuff>.stride * capacity
            )
        }

        // ── Camera-attached overlay quad ────────────────────────────────
        //
        // A flat plane sized to exactly cover the camera's viewport at a
        // fixed distance just past the near plane. The texture is sampled
        // with standard UVs, alpha-blended over the existing scene. The
        // plane is parented to the camera via an SCNNode child of
        // `overlayContainer`, which itself is positioned by the controller
        // to track the camera each tick. (We can't make it a child of the
        // camera directly because the camera is set up before this
        // renderer exists; the container approach keeps the wiring simple.)
        //
        // Sizing: at z = -0.2 from the camera, plane height = 2 ×
        // tan(fov/2) × 0.2. We over-size by 5% just to be safe against
        // float rounding showing a pixel of sky at the edge.
        let overlayPlane = SCNPlane(width: 2.0, height: 2.0)
        overlayPlane.cornerRadius = 0
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.diffuse.contents = tex
        // Premultiplied alpha — the kernel writes (rgb * alpha, alpha), so
        // alpha-over blending composes correctly.
        mat.blendMode = .alpha
        mat.transparencyMode = .aOne
        mat.diffuse.magnificationFilter = .linear
        mat.diffuse.minificationFilter = .linear
        mat.diffuse.wrapS = .clamp
        mat.diffuse.wrapT = .clamp
        mat.writesToDepthBuffer = false
        mat.readsFromDepthBuffer = false
        // Render LAST so it composites over everything. SceneKit's default
        // renderingOrder is 0; pick a high value.
        mat.isDoubleSided = true
        overlayPlane.materials = [mat]
        let planeNode = SCNNode(geometry: overlayPlane)
        planeNode.name = "VolumetricSmoke.overlay"
        planeNode.renderingOrder = 1000  // composite over scene + ocean + everything

        let container = SCNNode()
        container.name = "VolumetricSmoke.overlayContainer"
        container.addChildNode(planeNode)

        self.device          = device
        self.engine          = engine
        self.burstField      = burstField
        self.cameraNode      = cameraNode
        self.outputTexture   = tex
        self.overlayContainer = container
        self.puffBuffer      = pBuf
        self.capacity        = capacity
        self.pipeline        = pipeline
        self.noiseVolume     = NoiseVolume(engine: engine)
        self.uniformBuffer   = uBuf
        self.outputWidth     = outputWidth
        self.outputHeight    = outputHeight
    }

    // ── Spawning ────────────────────────────────────────────────────────────

    /// Spawn a residue cloud at a burst centre. ~6-10 puffs scattered within
    /// the burst's radius, each ~1.5..3 m, growing to ~4-6 m by end of life.
    public func spawnBurstResidue(at center: SIMD3<Float>, burstSize: Float) {
        let n = Int.random(in: 3...5)
        for _ in 0..<n {
            let theta = Float.random(in: 0..<(2 * .pi))
            let phi   = Float.random(in: -.pi/3 ..< .pi/3)
            let r     = burstSize * Float.random(in: 0.3...1.4)
            let pos = SIMD3<Float>(
                center.x + r * cos(phi) * cos(theta),
                center.y + r * sin(phi) * 0.5,
                center.z + r * cos(phi) * sin(theta)
            )
            let start = Float.random(in: 1.2...2.0)
            appendPuff(SmokePuff(
                positionAge: SIMD4(pos.x, pos.y, pos.z, 0),
                paramsLife:  SIMD4(start, start * 2.4,
                                   Float.random(in: 0.50...0.85),
                                   Float.random(in: 5.0...7.5))
            ))
        }
    }

    /// Spawn a contrail puff along a rising shell's current world position.
    public func spawnShellContrail(at position: SIMD3<Float>) {
        let pos = SIMD3<Float>(
            position.x + Float.random(in: -0.05...0.05),
            position.y - 0.08,
            position.z + Float.random(in: -0.05...0.05)
        )
        let start = Float.random(in: 0.30...0.55)
        appendPuff(SmokePuff(
            positionAge: SIMD4(pos.x, pos.y, pos.z, 0),
            paramsLife:  SIMD4(start, start * 1.8,
                               Float.random(in: 0.18...0.30),
                               Float.random(in: 2.5...4.0))
        ))
    }

    private func appendPuff(_ puff: SmokePuff) {
        if puffMirror.count >= capacity {
            puffMirror.removeFirst()
        }
        puffMirror.append(puff)
    }

    public func clearAll() {
        puffMirror.removeAll()
    }

    /// Update the camera-attached overlay quad's world transform + size.
    /// Pulled out so encodeStep can do it every frame even when the kernel
    /// dispatch is throttled.
    private func updateOverlayTransform() {
        overlayContainer.simdWorldTransform = cameraNode.simdWorldTransform
        guard let planeNode = overlayContainer.childNodes.first else { return }
        let fovDeg = cameraNode.camera?.fieldOfView ?? 60
        let fovRad = Float(fovDeg) * .pi / 180
        let z: Float = 0.25
        let halfH = tanf(fovRad * 0.5) * z
        let aspect = Float(outputWidth) / Float(outputHeight)
        let halfW = halfH * aspect
        planeNode.simdPosition = SIMD3(0, 0, -z)
        planeNode.scale = SCNVector3(CGFloat(halfW), CGFloat(halfH), 1)
    }

    // ── Per-tick step ───────────────────────────────────────────────────────

    public func encodeStep(into cb: MTLCommandBuffer, dt: Float, time: Float) {
        // Age + reap (every frame — cheap CPU op).
        for i in 0..<puffMirror.count {
            puffMirror[i].positionAge.w += dt
        }
        puffMirror.removeAll { p in p.positionAge.w >= p.paramsLife.w }

        // Update the overlay quad to track the camera EVERY frame, even
        // when we're throttling the kernel dispatch — otherwise the
        // overlay desyncs from the head/yaw between dispatches.
        updateOverlayTransform()

        // Throttle the expensive raymarch dispatch — every Nth frame is
        // sufficient for drifting smoke, and the GPU budget per frame
        // stays well under saturation. Output texture persists between
        // encodes; the overlay material samples the most recent result.
        encodeFrameCount &+= 1
        if encodeFrameCount % encodeEveryNFrames != 0 { return }

        // (overlay transform is updated by updateOverlayTransform() above)

        // Upload puff buffer.
        let puffCount = puffMirror.count
        if puffCount > 0 {
            let ptr = puffBuffer.buffer.contents().bindMemory(
                to: SmokePuff.self, capacity: capacity
            )
            for i in 0..<puffCount {
                ptr[i] = puffMirror[i]
            }
        }

        // Camera basis for the kernel. SCNNode worldTransform has -Z forward
        // (SCN convention); right = +X, up = +Y in the camera's local space.
        let xf = cameraNode.simdWorldTransform
        let right   = simd_make_float3(xf.columns.0)
        let up      = simd_make_float3(xf.columns.1)
        let forward = -simd_make_float3(xf.columns.2)  // SCN: camera looks down -Z
        let pos     = simd_make_float3(xf.columns.3)
        let fovDeg = cameraNode.camera?.fieldOfView ?? 60
        let fovRad = Float(fovDeg) * .pi / 180
        let aspect = Float(outputWidth) / Float(outputHeight)
        let halfFovY = tanf(fovRad * 0.5)
        let halfFovX = halfFovY * aspect

        let puffAsFloat  = Float(bitPattern: UInt32(puffCount))
        let burstAsFloat = Float(bitPattern: UInt32(burstField.activeCount))

        var u = SmokeVolumeUniforms(
            cameraPos:     SIMD4(pos.x, pos.y, pos.z, 0),
            cameraRight:   SIMD4(right.x, right.y, right.z, halfFovX),
            cameraUp:      SIMD4(up.x, up.y, up.z, halfFovY),
            cameraForward: SIMD4(forward.x, forward.y, forward.z, aspect),
            puffsBursts:   SIMD4(puffAsFloat, burstAsFloat, 0.3, 90.0),
            ambient:       SIMD4(0.04, 0.05, 0.07, 0.85),
            marchParams:   SIMD4(20.0, 2.4, 1.0, 0.07),  // step count, step len (m), light steps (unused now), noise scale
            timeAndPhase:  SIMD4(time, 0.45, 0, 0)
        )
        uniformBuffer.contents().copyMemory(
            from: &u, byteCount: MemoryLayout<SmokeVolumeUniforms>.stride
        )

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "VolumetricSmoke.march"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(outputTexture, index: 0)
        enc.setTexture(noiseVolume.texture, index: 1)
        enc.setBuffer(puffBuffer.buffer,         offset: 0, index: 0)
        enc.setBuffer(burstField.buffer.buffer,  offset: 0, index: 1)
        enc.setBuffer(uniformBuffer,             offset: 0, index: 2)

        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: outputWidth, height: outputHeight, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

public struct SmokePuff {
    public var positionAge: SIMD4<Float>   // xyz = world centre, w = age (s)
    public var paramsLife:  SIMD4<Float>   // x = startR, y = endR, z = density, w = lifespan

    public init() {
        self.positionAge = SIMD4(0, 0, 0, 0)
        self.paramsLife  = SIMD4(0, 0, 0, 0)
    }

    public init(positionAge: SIMD4<Float>, paramsLife: SIMD4<Float>) {
        self.positionAge = positionAge
        self.paramsLife  = paramsLife
    }
}

struct SmokeVolumeUniforms {
    var cameraPos:     SIMD4<Float>
    var cameraRight:   SIMD4<Float>   // .w = tan(halfFovX)
    var cameraUp:      SIMD4<Float>   // .w = tan(halfFovY)
    var cameraForward: SIMD4<Float>   // .w = aspect
    var puffsBursts:   SIMD4<Float>   // x = puff count, y = burst count, z = near, w = far
    var ambient:       SIMD4<Float>   // rgb = ambient sky tint, a = density gain
    var marchParams:   SIMD4<Float>   // x = steps, y = step len, z = light steps, w = noise scale
    var timeAndPhase:  SIMD4<Float>   // x = time, y = hg forward
}
