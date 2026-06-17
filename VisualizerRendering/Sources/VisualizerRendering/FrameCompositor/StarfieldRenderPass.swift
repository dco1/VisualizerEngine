import AppKit
import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── STARFIELD RENDER PASS ───────────────────────────────────────────────────
//
// Renders a list of star points (position + colour) as tiny additive
// billboards through a Metal render pipeline, writing into the compositor's
// HDR colour attachment. The stars sit at a very large world-space radius
// (typically 250+ m) so they read as the at-infinity sky; the cloud's
// raymarch sees them at "deep" depth and correctly occludes them when
// the cloud is in front.
//
// REUSABLE for any scene that wants a sparse point-light star field in
// world space with proper depth interaction. Smaller stars look like
// point lights, larger ones look like a "twinkle." Could be repurposed
// for distant city lights, runway markers, anything that's "small bright
// point at world position with depth."

/// Per-star vertex written to the GPU buffer each frame. Position +
/// colour + size. 32 bytes float4-aligned.
public struct StarVertex {
    public var positionSize: SIMD4<Float>  // xyz = world position, w = billboard half-size (m)
    public var color: SIMD4<Float>         // rgb = HDR colour, a = corner index 0..3

    public init() {
        self.positionSize = SIMD4(0, 0, 0, 0)
        self.color = SIMD4(0, 0, 0, 0)
    }

    public init(positionSize: SIMD4<Float>, color: SIMD4<Float>) {
        self.positionSize = positionSize
        self.color = color
    }
}

/// One star the host wants drawn this frame.
public struct StarRenderItem {
    public var position: SIMD3<Float>
    public var color: SIMD3<Float>
    public var size: Float

    public init(position: SIMD3<Float>, color: SIMD3<Float>, size: Float = 0.25) {
        self.position = position
        self.color = color
        self.size = size
    }
}

@MainActor
public final class StarfieldRenderPass {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "StarfieldRenderPass")

    public let device: MTLDevice
    public let engine: SimEngine
    public let capacity: Int

    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer

    public init?(engine: SimEngine, capacity: Int = 4096) {
        let device = engine.device
        self.device = device
        self.engine = engine
        self.capacity = capacity

        let vertexBytes = capacity * 4 * MemoryLayout<StarVertex>.stride
        let indexBytes  = capacity * 6 * MemoryLayout<UInt32>.stride
        guard
            let vBuf = device.makeBuffer(length: vertexBytes, options: .storageModeShared),
            let iBuf = device.makeBuffer(length: indexBytes,  options: .storageModeShared)
        else {
            Self.log.error("StarfieldRenderPass buffer alloc failed")
            return nil
        }
        vBuf.label = "StarfieldRenderPass.vertices"
        iBuf.label = "StarfieldRenderPass.indices"

        // Pre-fill 6-index pattern per star (2 triangles per billboard quad).
        let idxPtr = iBuf.contents().bindMemory(to: UInt32.self, capacity: capacity * 6)
        for s in 0..<capacity {
            let base = UInt32(s * 4)
            let dst = s * 6
            idxPtr[dst + 0] = base + 0
            idxPtr[dst + 1] = base + 1
            idxPtr[dst + 2] = base + 2
            idxPtr[dst + 3] = base + 0
            idxPtr[dst + 4] = base + 2
            idxPtr[dst + 5] = base + 3
        }
        self.vertexBuffer = vBuf
        self.indexBuffer = iBuf

        // Vertex descriptor
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float4
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float4
        vd.attributes[1].offset = 16
        vd.attributes[1].bufferIndex = 0
        vd.layouts[0].stride = 32
        vd.layouts[0].stepFunction = .perVertex

        guard let library = engine.library,
              let vfn = library.makeFunction(name: "starfieldVS"),
              let ffn = library.makeFunction(name: "starfieldFS") else {
            Self.log.error("StarfieldRenderPass: functions not found")
            return nil
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "StarfieldRenderPass"
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.vertexDescriptor = vd
        let colorAtt = desc.colorAttachments[0]!
        colorAtt.pixelFormat = .rgba16Float
        colorAtt.isBlendingEnabled = true
        colorAtt.rgbBlendOperation = .add
        colorAtt.alphaBlendOperation = .add
        colorAtt.sourceRGBBlendFactor = .one
        colorAtt.sourceAlphaBlendFactor = .one
        colorAtt.destinationRGBBlendFactor = .one
        colorAtt.destinationAlphaBlendFactor = .one
        desc.depthAttachmentPixelFormat = .depth32Float

        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else {
            Self.log.error("StarfieldRenderPass: pipeline build failed")
            return nil
        }
        self.pipeline = pipeline

        // Depth state: TEST + WRITE. Stars sit at far depth (large radius
        // from camera) and write that depth, so the cloud raymarch sees
        // them as "behind cloud" when the cloud is in front of them and
        // correctly occludes.
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .lessEqual
        depthDesc.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depthDesc) else {
            Self.log.error("StarfieldRenderPass: depth state failed")
            return nil
        }
        self.depthState = depthState
    }

    public func encode(
        into cb: MTLCommandBuffer,
        target: FrameCompositor,
        viewProjection: float4x4,
        cameraRight: SIMD3<Float>,
        cameraUp: SIMD3<Float>,
        stars: [StarRenderItem]
    ) {
        let n = min(stars.count, capacity)
        guard n > 0 else { return }

        let vptr = vertexBuffer.contents().bindMemory(to: StarVertex.self, capacity: capacity * 4)
        for i in 0..<n {
            let s = stars[i]
            let base = i * 4
            let posS = SIMD4(s.position.x, s.position.y, s.position.z, s.size * 0.5)
            for c in 0..<4 {
                var color = SIMD4(s.color.x, s.color.y, s.color.z, Float(c))
                _ = color  // silence unused-warning if any
                vptr[base + c] = StarVertex(
                    positionSize: posS,
                    color: SIMD4(s.color.x, s.color.y, s.color.z, Float(c))
                )
            }
        }

        let pass = target.makeRenderPassDescriptor()
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "StarfieldRenderPass"
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.none)

        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        var vp = viewProjection
        var right = SIMD4(cameraRight.x, cameraRight.y, cameraRight.z, 0)
        var up = SIMD4(cameraUp.x, cameraUp.y, cameraUp.z, 0)
        enc.setVertexBytes(&vp, length: MemoryLayout<float4x4>.size, index: 1)
        enc.setVertexBytes(&right, length: MemoryLayout<SIMD4<Float>>.size, index: 2)
        enc.setVertexBytes(&up, length: MemoryLayout<SIMD4<Float>>.size, index: 3)

        enc.drawIndexedPrimitives(
            type: .triangle,
            indexCount: n * 6,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )
        enc.endEncoding()
    }
}
