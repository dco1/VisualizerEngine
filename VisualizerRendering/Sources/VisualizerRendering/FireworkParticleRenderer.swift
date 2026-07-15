#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#else
import UIKit
#endif
import Foundation
import Metal
import SceneKit
import simd
import VisualizerCore

// ── FIREWORK PARTICLE RENDERER ──────────────────────────────────────────────
//
// Bridges a `FireworkParticleSolver` into an SCNGeometry rendered as velocity-
// aligned billboard quads (one quad = two triangles per particle). SceneKit
// reads position + color straight out of the solver's per-frame
// `streakBuffer` — no CPU snapshot, no allocation in the steady state.
//
// The streak quads themselves are written by `fwBuildStreaks` on the GPU; this
// renderer is purely a wiring shim: a 4-vertex / 6-index per-particle topology
// hanging off the shared MTLBuffer, plus an additive constant-lit material.
//
// HISTORY — v1 used SceneKit's `.point` primitive with one vertex per particle
// and a fixed screen-space size. That couldn't carry per-spark motion-blur
// streaks (the point primitive has no per-vertex size, and shader modifiers
// silently break on it — see memory [[shader_modifiers_break_point_primitives.md]]).
// Switching to four billboard verts per particle solves both: the streak is
// real geometry, and per-vertex color carries the head→tail falloff that sells
// the motion blur.
//
// USAGE
//
//   let renderer = FireworkParticleRenderer(solver: solver)
//   scene.rootNode.addChildNode(renderer.node)
//   // every tick, in the SAME command buffer as solver.encodeStep:
//   solver.encodeBuildStreaks(into: cb,
//                             cameraPosition: cameraNode.simdWorldPosition,
//                             streakDuration: 0.06,
//                             streakWidth:    0.06,
//                             minLength:      0.06)

@MainActor
public final class FireworkParticleRenderer {

    public let node: SCNNode
    public let solver: FireworkParticleSolver

    /// Exposed for use by `ParticleStreakPass` (custom Metal renderer
    /// pipeline). Same triangle topology as the SCN geometry uses, so the
    /// new pass can read from the same buffer.
    public let indexBuffer: MTLBuffer

    public init?(solver: FireworkParticleSolver) {
        let device = solver.device
        let cap = solver.particleBuffer.capacity
        let vertCount = cap * 4
        let triCount  = cap * 2

        // Pre-fill the index buffer: per particle, two triangles in CW order
        // (matches the v1/v2/v3 corner layout in the build kernel — see the
        // diagram in Fireworks.metal). 32-bit indices because vertCount can
        // exceed 2¹⁶ at the default capacity (32k particles × 4 verts = 128k).
        let idxStride = MemoryLayout<UInt32>.stride
        let idxCount = triCount * 3
        guard let idxBuf = device.makeBuffer(length: idxStride * idxCount,
                                             options: .storageModeShared)
        else { return nil }
        idxBuf.label = "Fireworks.streakIndices"
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: idxCount)
        for p in 0..<cap {
            let base = UInt32(p * 4)
            let dst  = p * 6
            idxPtr[dst + 0] = base + 0
            idxPtr[dst + 1] = base + 1
            idxPtr[dst + 2] = base + 2
            idxPtr[dst + 3] = base + 0
            idxPtr[dst + 4] = base + 2
            idxPtr[dst + 5] = base + 3
        }

        let vertStride = MemoryLayout<FWStreakVertex>.stride
        let uvStride = MemoryLayout<SIMD2<Float>>.stride

        // Position: first 12 bytes of each FWStreakVertex.
        let posSource = SCNGeometrySource(
            buffer: solver.streakBuffer.buffer,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: vertCount,
            dataOffset: 0,
            dataStride: vertStride
        )

        // Color: bytes 16..<28 (positionXYZw=16, then color.rgb).
        let colSource = SCNGeometrySource(
            buffer: solver.streakBuffer.buffer,
            vertexFormat: .float3,
            semantic: .color,
            vertexCount: vertCount,
            dataOffset: 16,
            dataStride: vertStride
        )

        // Dynamic UV source: kernel writes V per-frame from 0 (tail) to
        // `streakLen / streakWidth` (head). With wrap mode = .repeat the
        // Gaussian sprite tiles along the streak as a chain of round dots,
        // each one sprite-width long. When the streak collapses to the
        // sprite (dying ember) the tile count is 1 → single soft dot.
        let uvSource = SCNGeometrySource(
            buffer: solver.streakUVBuffer.buffer,
            vertexFormat: .float2,
            semantic: .texcoord,
            vertexCount: vertCount,
            dataOffset: 0,
            dataStride: uvStride
        )

        let element = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .triangles,
            primitiveCount: triCount,
            bytesPerIndex: idxStride
        )

        let geom = SCNGeometry(
            sources: [posSource, colSource, uvSource],
            elements: [element]
        )

        // Constant-lit + additive: the soft sparkle sprite encodes its falloff
        // in RGB (since `.add` blend ignores alpha) — sampled white at the
        // bright core, fading to black at the corners. Per-vertex color
        // multiplies the sample, so the spark's hue × hot-start brightness
        // × tail-fade × sprite-falloff all stack into the final emissive
        // colour the bloom shader integrates over. Without the sprite each
        // quad reads as a hard 22 cm confetti square; with it, the same quad
        // is a soft round point that motion-blurs into a streak when the
        // velocity-driven length stretches the texture.
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.diffuse.contents = Self.sparkSprite
        mat.diffuse.wrapS = .clamp
        // V wraps so the kernel can repeat the sprite N times along the
        // streak. With the dynamic UV buffer setting head verts to
        // V = streakLen / width, the texture sampler tiles the Gaussian dot
        // sprite once per sprite-width along the velocity vector — what reads
        // as "a collection of dying light particles" instead of one extended
        // line.
        mat.diffuse.wrapT = .repeat
        mat.diffuse.magnificationFilter = .linear
        mat.diffuse.minificationFilter = .linear
        mat.diffuse.mipFilter = .linear
        mat.blendMode = .add
        mat.writesToDepthBuffer = false
        mat.readsFromDepthBuffer = false
        // Per [[scn_geometry_winding.md]] SceneKit uses CW front-face for
        // custom geometry. Streaks are billboards with no inside, and bloom-
        // dominated additive blending hides any winding mismatch — double-
        // sided is the safe choice.
        mat.isDoubleSided = true
        geom.materials = [mat]

        let n = SCNNode(geometry: geom)
        n.name = "FireworkStreaks"
        n.castsShadow = false
        // The seeded streak buffer parks every vertex at y = −10000 until
        // the first build kernel runs, so SCN's auto bbox would frustum-cull
        // the geometry forever. Set an explicit world-space box covering the
        // burst domain so it keeps drawing every frame.
        n.boundingBox = (
            min: SCNVector3(-200, -10, -200),
            max: SCNVector3( 200, 200,  200)
        )

        self.solver = solver
        self.indexBuffer = idxBuf
        self.node = n
    }

    // ── Illuminatorama particle-field registration ───────────────────────────
    //
    // Publishes the per-particle buffer (NOT the expanded streak buffer) into
    // the shared `SimEngine.particleFields` registry so the bursts show up
    // through the overlay. `FireworkParticle` is 6 × `SIMD4<Float>` = 24-float
    // stride; `positionAge.xyz` leads at offset 0 and `displayColor.rgb` sits
    // 8 floats in (the 3rd SIMD4). The colour-offset support lets ONE buffer
    // feed both position and colour, so no separate contiguous colour buffer
    // is needed (unlike FoamRenderer, whose particle struct has no colour).
    //
    // The overlay draws these as additive HDR points — no velocity-aligned
    // streaks (that needs a textured-quad path, a later phase), but the
    // bloom-amplified points read as a real burst instead of the nothing
    // FireworksUltra/Plus showed through the overlay before. Dead particles are
    // parked at y = −10000 by the solver, so they project off-screen and clip.
    //
    // `ownerScene` scopes the source to the controller's SCNScene so the
    // overlay only draws the active scene's fields (AppModel caches controllers
    // across scene switches). The SCN streak node's own `.add` triangle
    // geometry still draws on SceneKit's forward pass.
    public func registerParticleField(scene: SCNScene,
                                      pointSize: Float = 3.0,
                                      colorScale: Float = 1.6) {
        let buf = solver.particleBuffer.buffer
        SimEngine.particleFields.register(ParticleFieldSource(
            positionBuffer:       buf,
            colorBuffer:          buf,                     // same interleaved buffer
            vertexCount:          solver.particleBuffer.capacity,
            ownerScene:           ObjectIdentifier(scene),
            positionStrideFloats: 24,                      // FireworkParticle stride / 4
            colorStrideFloats:    24,
            positionOffsetFloats: 0,                       // positionAge.xyz
            colorOffsetFloats:    8,                        // displayColor.rgb (3rd SIMD4)
            pointSize:            pointSize,
            colorScale:           colorScale
        ))
    }

    // ── Soft sparkle sprite ─────────────────────────────────────────────────
    //
    // Procedural 128×128 sprite consumed by every streak quad. Combines a
    // tight bright core (small Gaussian, σ low) with a wider soft halo
    // (larger Gaussian, σ high) so the spark has a hot pinpoint surrounded
    // by a smooth glow — what a real spark looks like through a camera's
    // exposure response.
    //
    // Falloff lives in RGB because the material's `.add` blend mode ignores
    // alpha. Alpha is set to the same falloff so any future blend-mode swap
    // (e.g. premultiplied alpha) keeps working without a regen.
    public static let sparkSprite: PlatformImage = makeSparkSprite()

    private static func makeSparkSprite() -> PlatformImage {
        let size = 128
        let bytesPerPixel = 4
        let bytesPerRow = size * bytesPerPixel
        var data = [UInt8](repeating: 0, count: bytesPerRow * size)
        let center = Double(size - 1) * 0.5
        // Normalise so r=1 lands at the quad corner (corner is √2 × halfSize
        // from centre; we use full halfSize so the halo barely reaches the
        // edge, leaving a clean transparent border that prevents the next
        // mip level from bleeding bright pixels into neighbouring streaks).
        let invHalf = 1.0 / center
        for y in 0..<size {
            for x in 0..<size {
                let dx = (Double(x) - center) * invHalf
                let dy = (Double(y) - center) * invHalf
                let r2 = dx * dx + dy * dy
                let core = exp(-r2 * 22.0)              // tight bright pinpoint
                let halo = exp(-r2 * 3.2)  * 0.55       // soft outer glow
                let edge = max(0.0, 1.0 - sqrt(r2))     // hard mask to clamp at the unit disc
                let a = min(1.0, (core + halo) * edge)
                let v = UInt8((a * 255.0).rounded())
                let i = (y * size + x) * bytesPerPixel
                data[i + 0] = v   // R — encodes falloff for `.add` blend
                data[i + 1] = v   // G
                data[i + 2] = v   // B
                data[i + 3] = v   // A — kept in sync for future blend-mode swaps
            }
        }
        let provider = CGDataProvider(data: Data(data) as CFData)!
        let cg = CGImage(
            width: size,
            height: size,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
        return PlatformImage(cgImage: cg, size: CGSize(width: size, height: size))
    }
}
