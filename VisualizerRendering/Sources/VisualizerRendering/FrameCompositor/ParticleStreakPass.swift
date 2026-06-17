import AppKit
import Foundation
import Metal
import MetalKit
import OSLog
import simd
import VisualizerCore

// ── PARTICLE STREAK PASS ────────────────────────────────────────────────────
//
// Renders a velocity-aligned billboard particle buffer (the same FWStreakVertex
// layout the FireworkParticleSolver writes) directly via a Metal render
// pipeline, instead of going through SceneKit's geometry abstraction.
//
// REUSABLE across forward-additive particle scenes — pass in the vertex
// buffer, index buffer, UV buffer, and the soft-sprite texture, and you
// get HDR additive particles composited into the FrameCompositor's HDR
// target with proper depth interaction (depth-read-only so the particles
// can be occluded by world geometry like clouds, but they don't occlude
// each other — they're additive).
//
// USAGE
//
//   let pass = ParticleStreakPass(engine: .shared,
//                                 spriteTexture: FireworkParticleRenderer.sparkSprite)
//   ...
//   // per tick (after the compositor's beginFrame):
//   pass.encode(into: cb,
//               target: compositor,
//               viewProjection: cameraVP,
//               positionBuffer: solver.streakBuffer.buffer,
//               uvBuffer: solver.streakUVBuffer.buffer,
//               indexBuffer: indexBuffer,
//               indexCount: particleCount * 6)
//
// Vertex layout matches `FWStreakVertex` exactly:
//   bytes 0..<12  = float3 position
//   bytes 16..<28 = float3 color
// UV layout is a parallel float2 buffer (`streakUVBuffer` from solver).

/// Per-fragment tunables for the streak shader. Mirrors the
/// `ParticleStreakParams` struct in `ParticleStreakPass.metal` byte-for-byte.
///
///   • `intensity` — overall HDR multiplier. The fixed value used previously
///     was ~3.5; expose so the user can pull this up for a brighter pop or
///     down to taste.
///   • `saturationPow` — exponent applied to the colour after normalising
///     to its brightest channel. `1.0` is neutral; higher values squeeze
///     the secondary channels toward zero so a red-ish particle reads as
///     bold pure red instead of pink. Practical range ≈ 1.0..3.5.
public struct ParticleStreakParams {
    public var intensity: Float
    public var saturationPow: Float

    public init(intensity: Float = 3.5, saturationPow: Float = 2.0) {
        self.intensity = intensity
        self.saturationPow = saturationPow
    }
}

@MainActor
public final class ParticleStreakPass {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "ParticleStreakPass")

    public let device: MTLDevice
    public let engine: SimEngine

    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let spriteTexture: MTLTexture

    /// Build a particle streak pass. `spriteTexture` is the soft-sparkle
    /// sprite each particle billboard samples — typically
    /// `FireworkParticleRenderer.sparkSprite` converted to MTLTexture.
    public init?(engine: SimEngine, spriteTexture: MTLTexture) {
        let device = engine.device
        self.device = device
        self.engine = engine
        self.spriteTexture = spriteTexture

        // Vertex descriptor matching FWStreakVertex layout. Position at
        // offset 0 (float3), UVs from a separate buffer at attribute 1,
        // color at offset 16 of the position buffer (float3).
        let vd = MTLVertexDescriptor()
        // Buffer 0: FWStreakVertex (position + color interleaved)
        vd.attributes[0].format = .float3
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        // Attribute 1: color as float4. The first 3 floats are RGB; the
        // 4th (alpha) carries the per-particle STRETCH FACTOR (`vTile`)
        // written by `fwBuildStreaks` in Fireworks.metal — the streak's
        // length-to-width ratio. The fragment shader needs this to build
        // the anisotropic capsule SDF for cinematic motion blur (the
        // capsule stretches by `stretchFactor` along velocity).
        //
        // Previously this was a float3 (RGB only); changing to float4
        // costs zero extra bytes in the buffer (the alpha was always
        // there in `FWStreakVertex.color: float4`) — we just weren't
        // reading it on this side.
        vd.attributes[1].format = .float4
        vd.attributes[1].offset = 16
        vd.attributes[1].bufferIndex = 0
        vd.layouts[0].stride = 32  // sizeof FWStreakVertex
        vd.layouts[0].stepRate = 1
        vd.layouts[0].stepFunction = .perVertex
        // Buffer 1: float2 UVs (parallel buffer)
        vd.attributes[2].format = .float2
        vd.attributes[2].offset = 0
        vd.attributes[2].bufferIndex = 1
        vd.layouts[1].stride = 8
        vd.layouts[1].stepRate = 1
        vd.layouts[1].stepFunction = .perVertex

        guard let library = engine.library else {
            Self.log.error("ParticleStreakPass: no Metal library on engine")
            return nil
        }
        guard let vfn = library.makeFunction(name: "particleStreakVS"),
              let ffn = library.makeFunction(name: "particleStreakFS") else {
            Self.log.error("ParticleStreakPass: vertex/fragment functions not found in metallib")
            return nil
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "ParticleStreakPass"
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.vertexDescriptor = vd

        // HDR rgba16Float colour attachment to match FrameCompositor.
        let colorAtt = desc.colorAttachments[0]!
        colorAtt.pixelFormat = .rgba16Float
        // Additive blend — particles add their light into the framebuffer
        // without occluding what's behind. RGB additive, alpha left alone.
        colorAtt.isBlendingEnabled = true
        colorAtt.rgbBlendOperation = .add
        colorAtt.alphaBlendOperation = .add
        colorAtt.sourceRGBBlendFactor = .one
        colorAtt.sourceAlphaBlendFactor = .one
        colorAtt.destinationRGBBlendFactor = .one
        colorAtt.destinationAlphaBlendFactor = .one

        desc.depthAttachmentPixelFormat = .depth32Float

        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else {
            Self.log.error("ParticleStreakPass: failed to build pipeline state")
            return nil
        }
        self.pipeline = pipeline

        // Depth state: TEST against scene depth (so particles behind solid
        // geometry like clouds are occluded), but DO NOT WRITE depth so
        // additive particles don't block one another and don't make
        // subsequent passes see them as solid occluders.
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .lessEqual
        depthDesc.isDepthWriteEnabled = false
        guard let depthState = device.makeDepthStencilState(descriptor: depthDesc) else {
            Self.log.error("ParticleStreakPass: failed to build depth state")
            return nil
        }
        self.depthState = depthState
    }

    /// Encode the particle draw into the given command buffer, writing to
    /// the compositor's HDR + depth attachments. `vertexCount` is the
    /// total number of vertices in `positionBuffer` (typically
    /// `particleCount × 4`). The render pass loads the existing
    /// attachments (doesn't clear) so it composes additively over earlier
    /// passes in the frame.
    public func encode(
        into cb: MTLCommandBuffer,
        target: FrameCompositor,
        viewProjection: float4x4,
        positionBuffer: MTLBuffer,
        uvBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        indexCount: Int,
        params: ParticleStreakParams = ParticleStreakParams()
    ) {
        guard indexCount > 0 else { return }
        let pass = target.makeRenderPassDescriptor()
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "ParticleStreakPass"
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(depthState)
        enc.setCullMode(.none)

        enc.setVertexBuffer(positionBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(uvBuffer,       offset: 0, index: 1)
        var vp = viewProjection
        enc.setVertexBytes(&vp, length: MemoryLayout<float4x4>.size, index: 2)

        enc.setFragmentTexture(spriteTexture, index: 0)
        var p = params
        enc.setFragmentBytes(&p, length: MemoryLayout<ParticleStreakParams>.size, index: 0)

        enc.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )
        enc.endEncoding()
    }
}
