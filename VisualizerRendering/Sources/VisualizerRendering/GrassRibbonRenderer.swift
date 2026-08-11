import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── GRASS RIBBON RENDERER ────────────────────────────────────────────────────
//
// Bridges PBDFieldSolver → SCNGeometry for a field of grass blades. One
// SCNGeometry, one MTLBuffer of vertex positions, one MTLBuffer of normals,
// one index buffer — shared across every blade. Per frame:
//
//   1. PBDFieldSolver writes new particle positions into its particle buffer.
//   2. `pbdGrassExpand` reads those positions and writes vertex+normal
//      positions into the geometry's MTLBuffer (1 thread per particle, 2
//      verts per particle).
//   3. SceneKit reads the same MTLBuffer when it draws.
//
// No CPU round-trip, no SCNMorpher, no per-blade SCNNode. Adding the 10000th
// blade adds ~144 bytes of vertex storage and zero draw calls.
//
// CAMERA-FACING RIBBON
// The expand kernel needs the camera position to orient each blade. Call
// `cameraPosition = …` from the controller every frame (cheap struct write
// into shared memory). Without this, blades collapse to a line because the
// expand axis is `cross(tangent, viewDir)` which degenerates when viewDir is
// stale and aligned with the blade tangent.

@MainActor
public final class GrassRibbonRenderer {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "GrassRibbonRenderer")

    public let solver: PBDFieldSolver

    // ── SceneKit wrappers: BUILT ON DEMAND, and that is a memory fix ──────────────
    //
    // A SceneKit host adds `node` to its scene graph; a native Illuminatorama host takes
    // `illuminatoramaDescriptor()` instead and never touches either of these. Building them
    // eagerly in `init` therefore cost the native hosts an `SCNGeometry` they never drew —
    // and, measured 2026-08-11 on Daydream Home's lawn rebuild, that geometry **is never
    // released**: a field's `SCNGeometry` and `SCNNode` were still alive after the
    // `GrassRibbonRenderer` that owned them had deinited, pinning the position, normal and
    // index buffers they were constructed over (~29 MB per re-scatter at 60k blades, linear
    // and unbounded — a `weak` probe across successive rebuilds named them as the only two
    // survivors). Nothing in the host holds them, so the retention is inside SceneKit, which
    // registers a geometry built over MTLBuffers with its own resource machinery; the only
    // reliable way not to pay it is not to create the object.
    //
    // Building on first access keeps every SceneKit host byte-identical (the first `node` read
    // builds exactly what `init` used to) while a native host allocates nothing. Written as an
    // explicit backing store rather than `lazy` so `hasSceneKitWrappers` can report whether they
    // exist WITHOUT creating them — a `lazy` property cannot be inspected without forcing it, and
    // a gate that has to touch the thing it is checking for measures its own probe.
    private var _geometry: SCNGeometry?
    private var _node: SCNNode?

    public var geometry: SCNGeometry {
        if let g = _geometry { return g }
        let g = makeGeometry()
        _geometry = g
        return g
    }

    public var node: SCNNode {
        if let n = _node { return n }
        let n = SCNNode(geometry: geometry)
        _node = n
        return n
    }

    /// TEST-OBSERVABLE: have the SceneKit wrappers been materialised? A native Illuminatorama
    /// host must answer `false` for the lifetime of the field — see `geometry` for what it costs
    /// when they exist.
    public var hasSceneKitWrappers: Bool { _geometry != nil || _node != nil }

    private let expandPipeline: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer
    private let positionBuffer: SimBuffer<SIMD3<Float>>
    private let normalBuffer:   SimBuffer<SIMD3<Float>>
    // Retained so native (off-SceneKit) Illuminatorama scenes can register the
    // blade field as a GPU mesh via `illuminatoramaDescriptor` — the same
    // compute-written position/normal buffers the SceneKit geometry reads, so
    // the grass is RT-traced + deferred-lit in the soup pipeline with zero CPU
    // round-trip (mirrors the DynamicMesh bridge).
    private let indexBuffer: MTLBuffer
    private let indexCount: Int
    private let vertexCount: Int

    public let bladeCount: Int
    public let particlesPerBlade: Int

    /// Half-width of the blade at the root in metres. The kernel tapers to
    /// `baseHalfWidth * tipHalfWidthFrac` at the tip. Reasonable defaults:
    /// 0.012 m base / 0.10 tip-frac for "fescue" grass.
    public var baseHalfWidth: Float
    public var tipHalfWidthFrac: Float

    /// World-space camera position. The expand kernel uses this every
    /// frame to keep blades facing the camera. Update from the controller's
    /// per-tick before encoding.
    public var cameraPosition: SIMD3<Float> = SIMD3(0, 1, 5)

    /// Per-blade width multiplier (×`baseHalfWidth`, same root→tip taper).
    /// Defaults to 1.0 for every blade; hosts can widen a subset so flower
    /// heads / reeds stand out from the grass without a second field.
    private let widthScaleBuffer: MTLBuffer

    // ── Init ───────────────────────────────────────────────────────────

    public init?(solver: PBDFieldSolver,
                 baseHalfWidth: Float = 0.012,
                 tipHalfWidthFrac: Float = 0.10) {
        let device = solver.device
        guard let expand = solver.engine.pipelineCache.pipelineState(
                  name: "pbdGrassExpand", device: device)
        else {
            Self.log.error("pbdGrassExpand pipeline missing — check PBDField.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        guard let uBuf = device.makeBuffer(length: MemoryLayout<GrassExpandUniforms>.stride,
                                           options: .storageModeShared)
        else {
            Self.log.error("Grass uniform buffer alloc failed")
            return nil
        }
        uBuf.label = "Grass.uniforms"

        let blades  = solver.chainCount
        let perBld  = solver.particlesPerChain
        let verts   = blades * perBld * 2
        precondition(verts > 0, "GrassRibbonRenderer: solver must be configured (chainCount × particlesPerChain > 0)")

        guard
            let posBuf  = SimBuffer<SIMD3<Float>>(device: device, capacity: verts,
                                                  label: "Grass.positions"),
            let normBuf = SimBuffer<SIMD3<Float>>(device: device, capacity: verts,
                                                  label: "Grass.normals")
        else { return nil }

        // ── Index buffer ───────────────────────────────────────────────
        //
        // Per blade with M particles → 2M verts, 2(M-1) triangles. With N
        // blades total vertex count can exceed UInt16.max easily (e.g. 5000
        // blades × 6 particles × 2 = 60 000 — comfortably within UInt16,
        // but 6000 × 6 × 2 = 72 000 isn't). Use UInt32 indices unconditionally
        // so we don't have to special-case at threshold.
        //
        // Triangle winding: blades are double-sided so we don't sweat the CW
        // vs CCW debate documented in the project memory (the inspector trick
        // doesn't apply here — wrong winding would just show as a one-side-
        // only blade, masked by isDoubleSided).
        let M = perBld
        var indices = [UInt32]()
        indices.reserveCapacity(blades * (M - 1) * 6)
        for b in 0..<blades {
            let base = UInt32(b * M * 2)
            for i in 0..<(M - 1) {
                let bl = base + UInt32(i * 2)
                let br = bl + 1
                let tl = bl + 2
                let tr = bl + 3
                // Two triangles per quad. CW from "blade front."
                indices.append(contentsOf: [bl, tl, br,  br, tl, tr])
            }
        }
        let idxByteLen = MemoryLayout<UInt32>.stride * indices.count
        guard let idxBuf = indices.withUnsafeBufferPointer({ ptr -> MTLBuffer? in
            device.makeBuffer(bytes: ptr.baseAddress!, length: idxByteLen,
                              options: .storageModeShared)
        }) else {
            Self.log.error("Grass index buffer alloc failed")
            return nil
        }
        idxBuf.label = "Grass.indices"

        // Per-blade width scales, all 1.0 until a host overrides a subset.
        let ones = [Float](repeating: 1, count: blades)
        guard let wsBuf = ones.withUnsafeBufferPointer({ ptr -> MTLBuffer? in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: MemoryLayout<Float>.stride * blades,
                              options: .storageModeShared)
        }) else {
            Self.log.error("Grass width-scale buffer alloc failed")
            return nil
        }
        wsBuf.label = "Grass.widthScales"

        self.solver         = solver
        self.expandPipeline = expand
        self.uniformBuffer  = uBuf
        self.positionBuffer = posBuf
        self.normalBuffer   = normBuf
        self.indexBuffer    = idxBuf
        self.widthScaleBuffer = wsBuf
        self.indexCount     = indices.count
        self.vertexCount    = verts
        self.bladeCount     = blades
        self.particlesPerBlade = perBld
        self.baseHalfWidth  = baseHalfWidth
        self.tipHalfWidthFrac = tipHalfWidthFrac
    }

    /// The SceneKit geometry over the compute-written blade buffers — exactly what `init` used
    /// to build eagerly, now built on first access to `geometry` / `node` (see those properties
    /// for why). Sources reference the live MTLBuffers, so the geometry animates with the field
    /// without a CPU round-trip, as before.
    private func makeGeometry() -> SCNGeometry {
        let stride12 = MemoryLayout<Float>.stride * 3  // packed_float3
        let posSource = SCNGeometrySource(
            buffer: positionBuffer.buffer,
            vertexFormat: .float3, semantic: .vertex,
            vertexCount: vertexCount, dataOffset: 0, dataStride: stride12
        )
        let normSource = SCNGeometrySource(
            buffer: normalBuffer.buffer,
            vertexFormat: .float3, semantic: .normal,
            vertexCount: vertexCount, dataOffset: 0, dataStride: stride12
        )
        let element = SCNGeometryElement(
            buffer: indexBuffer,
            primitiveType: .triangles,
            primitiveCount: indexCount / 3,
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )

        let geom = SCNGeometry(sources: [posSource, normSource], elements: [element]) // winding-ok: blades are isDoubleSided + normals are GPU-computed by pbdGrassExpand
        // Grass material: PBR-ish but driven by simple parameters. The
        // controller can override `geom.firstMaterial` if it wants something
        // fancier (vertex-colour gradient, etc.).
        let mat = SCNMaterial()
        mat.lightingModel = .physicallyBased
        mat.diffuse.contents = PlatformColor(deviceRed: 0.30, green: 0.55, blue: 0.18, alpha: 1)
        mat.roughness.contents = 0.85
        mat.metalness.contents = 0.0
        mat.isDoubleSided = true
        geom.firstMaterial = mat
        return geom
    }

    /// Bridge descriptor for native-Illuminatorama scenes. Hand this to
    /// `IlluminatoramaRenderer.registerGPUMesh(_:)` and add an identity-transform
    /// instance referencing the returned handle's `kind`. The renderer repacks
    /// the compute-written position/normal buffers into its interleaved vertex
    /// layout each frame and refits the blade BLAS — so the XPBD grass animates
    /// in the deferred G-buffer AND the RT shadow/GI pass. Blades are
    /// double-sided (the ribbon winding is front-only, masked by no-cull).
    ///
    /// `colorBuffer` (optional, stride-16 RGBA `float4`, one per vertex in the
    /// renderer's `vertexCount` order) is multiplied into albedo at shade time —
    /// pass a root-dark → tip-light gradient (+ per-blade jitter) so the field
    /// doesn't read as a flat single-green sheet (issue #58 grass item #7).
    public func illuminatoramaDescriptor(colorBuffer: MTLBuffer? = nil)
        -> IlluminatoramaGPUMeshDescriptor {
        IlluminatoramaGPUMeshDescriptor(
            positionBuffer: positionBuffer.buffer,
            normalBuffer: normalBuffer.buffer,
            positionStride: MemoryLayout<Float>.stride * 3,   // packed_float3
            normalStride: MemoryLayout<Float>.stride * 3,
            vertexCount: vertexCount,
            bodyIndexBuffer: indexBuffer,
            bodyIndexCount: indexCount,
            bodyIndexType: .uint32,
            colorBuffer: colorBuffer,
            doubleSided: true,
            // A blade ribbon is an open shell — no back face for the shadow pass's
            // second-depth (`cull .front`) trick, so front-face culling discards the lit
            // face and the blade casts nothing. See `IlluminatoramaMesh.shadowCastsBothFaces`.
            shadowCastsBothFaces: true)
    }

    /// Override the per-blade width multipliers (blade order = solver chain
    /// order, the same order `configureChains` received). Counts beyond
    /// `bladeCount` are ignored; missing entries keep their previous scale.
    /// A scale multiplies `baseHalfWidth` with the same root→tip taper —
    /// use it to widen flower-head / reed blades within one field.
    public func setBladeWidthScales(_ scales: [Float]) {
        let n = min(scales.count, bladeCount)
        guard n > 0 else { return }
        widthScaleBuffer.contents()
            .bindMemory(to: Float.self, capacity: bladeCount)
            .update(from: scales, count: n)
    }

    /// The renderer-order vertex layout this geometry uses, so a caller can build
    /// a matching per-vertex `colorBuffer`: vertex index = (blade·M + particle)·2
    /// + side, with `f = particle / (M-1)` the root→tip fraction (0 at root).
    public var vertexLayout: (vertexCount: Int, bladeCount: Int, particlesPerBlade: Int) {
        (vertexCount, bladeCount, particlesPerBlade)
    }

    // ── Per-frame expand ───────────────────────────────────────────────

    public func encodeExpand(to cb: MTLCommandBuffer) {
        let total = bladeCount * particlesPerBlade
        guard total > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Grass.expand"

        let uPtr = uniformBuffer.contents().bindMemory(to: GrassExpandUniforms.self, capacity: 1)
        uPtr.pointee = GrassExpandUniforms(
            bladeCount: UInt32(bladeCount),
            particlesPerBlade: UInt32(particlesPerBlade),
            baseHalfWidth: baseHalfWidth,
            tipHalfWidthFrac: tipHalfWidthFrac,
            cameraPos: SIMD4(cameraPosition, 0)
        )

        enc.setComputePipelineState(expandPipeline)
        enc.setBuffer(solver.particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(positionBuffer.buffer,         offset: 0, index: 1)
        enc.setBuffer(uniformBuffer,                  offset: 0, index: 2)
        enc.setBuffer(normalBuffer.buffer,            offset: 0, index: 3)
        enc.setBuffer(widthScaleBuffer,               offset: 0, index: 4)

        let w = min(total, expandPipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }
}

// ── Grass-expand uniform mirror ──────────────────────────────────────────────
//
// Keep field order in lockstep with `GrassExpandUniforms` in PBDField.metal.
struct GrassExpandUniforms {
    var bladeCount:        UInt32
    var particlesPerBlade: UInt32
    var baseHalfWidth:     Float
    var tipHalfWidthFrac:  Float
    var cameraPos:         SIMD4<Float>
}
