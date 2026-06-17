import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

/// ── CAUSTICS RT (revertable) ────────────────────────────────────
/// Photon-traced caustics for an MLS-MPM water surface.
///
/// Pairs with a `MarchingCubesBridge` (read the live water mesh
/// directly from its position/normal buffers — no per-frame water AS
/// build) and a static rectangular receiver (typically the pool
/// bottom). The receiver acceleration structure is built once at
/// init; the per-frame work is:
///
///   1. Clear the compact-tri counter + the accumulation buffer.
///   2. Compact: scan MC's per-cell vertex slots, append every
///      non-degenerate triangle (its base vertex offset + area) to a
///      compact list, atomic-counter for the length.
///   3. Emit + trace + splat (one fused kernel): for each photon,
///      sample a triangle uniformly from the compact list, sample a
///      barycentric point + interpolated normal, refract the light
///      direction via Snell's law, fire the refracted ray at the
///      receiver primitive AS via `intersector<triangle_data>`,
///      atomic-splat colour into a 2D world-XZ accumulator.
///   4. Resolve: 5×5 box blur + Reinhard tone map + gamma →
///      `causticTexture` (RGBA8).
///
/// Apply the texture to the receiver material via a `.surface`
/// shader modifier (see `surfaceModifierSource`). The modifier maps
/// world XZ across the receiver rectangle and additively blends the
/// caustic colour into the material's emission.
///
/// Kernels live in `Visualizer/Rendering/CausticsRT.metal` (app
/// target). Revert: delete that file + this one + the
/// `// ── CAUSTICS RT (revertable) ──` blocks in
/// FluidTestController / Settings / SettingsView.
@MainActor
public final class CausticsRT {

    static let log = Logger(subsystem: AppLog.subsystem, category: "causticsRT")

    // ── Receiver geometry (world space) ──────────────────────────

    /// Rectangle the caustic texture is painted across. UV (0,0) maps
    /// to (minX, minZ), UV (1,1) maps to (minX + extentX, minZ + extentZ).
    public struct Receiver {
        public var y: Float
        public var minX: Float
        public var minZ: Float
        public var extentX: Float
        public var extentZ: Float

        public init(y: Float, minX: Float, minZ: Float,
                    extentX: Float, extentZ: Float) {
            self.y = y; self.minX = minX; self.minZ = minZ
            self.extentX = extentX; self.extentZ = extentZ
        }
    }

    // ── Tuning ───────────────────────────────────────────────────

    /// Light direction in the same space the MC surface vertices live
    /// in (sim-space for FluidTest). Should point FROM the light INTO
    /// the scene — i.e. the same convention as a directional light's
    /// emission direction. Defaults to slightly tilted overhead so
    /// caustics get a directional anchor instead of bouncing straight
    /// down.
    public var lightDirection: SIMD3<Float> = normalize(SIMD3(-0.30, -1.0, -0.20))

    public var lightColor: SIMD3<Float> = SIMD3(1.0, 0.95, 0.85)
    public var lightIntensity: Float = 1.0

    /// Water IOR. 1.33 is the textbook value; tweak slightly higher
    /// (1.4–1.5) to exaggerate the focus pattern.
    public var ior: Float = 1.33

    /// Per-frame photon budget. 64K is a sweet spot — enough to fill
    /// the floor on a 256² accumulator after a 5×5 blur, cheap to
    /// dispatch.
    public var photonCount: Int = 65_536

    /// Triangles with area smaller than this are dropped at compact
    /// time. MC cell-size is `solver.cellSize` (≈ 0.1 m) so genuine
    /// triangles span the millimetre-squared range and up; degenerate
    /// slots are zero-area. 1e-6 catches both.
    public var surfaceAreaEpsilon: Float = 1.0e-6

    // ── Public outputs ───────────────────────────────────────────

    public let causticTexture: MTLTexture
    public let textureSize: Int
    public let receiver: Receiver

    // ── Metal plumbing ───────────────────────────────────────────

    private let device: MTLDevice
    private let clearCounterPipeline: MTLComputePipelineState
    private let clearAccumPipeline:   MTLComputePipelineState
    private let compactPipeline:      MTLComputePipelineState
    private let emitTraceSplatPipeline: MTLComputePipelineState
    private let resolvePipeline:      MTLComputePipelineState

    // GPU buffers driven each frame.
    private let accumBuffer: MTLBuffer            // atomic float, texSize² × 4
    private let compactBuffer: MTLBuffer          // CompactTri × maxTris
    private let counterBuffer: MTLBuffer          // atomic_uint × 1
    private let uniformBuffer: MTLBuffer          // CausticsUniforms × 1
    private let receiverVertexBuffer: MTLBuffer   // 4 packed_float3
    private let receiverIndexBuffer: MTLBuffer    // 6 uint32

    // Static receiver primitive AS (2 triangles, never rebuilt).
    private var receiverAS: MTLAccelerationStructure!

    // Surface vertex buffers (borrowed from the MC bridge — we don't own them).
    private weak var surfaceBridge: MarchingCubesBridge?
    private let surfaceVertexCount: Int

    // ── Init ─────────────────────────────────────────────────────

    /// Initialise a caustics renderer that reads from `bridge`'s live
    /// surface mesh and paints onto `receiver`. Returns nil if RT isn't
    /// supported or any Metal resource fails to allocate.
    public init?(bridge: MarchingCubesBridge,
                 receiver: Receiver,
                 textureSize: Int = 512) {
        guard GPUCapabilities.supportsRaytracing,
              let device = GPUCapabilities.device,
              device.supportsRaytracing else {
            Self.log.notice("ray tracing not supported on this GPU; skipping init")
            return nil
        }

        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .main)
        } catch {
            Self.log.error("default Metal library missing: \(error.localizedDescription)")
            return nil
        }

        func makePipeline(_ name: String) -> MTLComputePipelineState? {
            guard let fn = library.makeFunction(name: name) else {
                Self.log.error("Metal function `\(name)` not found in default library")
                return nil
            }
            do {
                return try device.makeComputePipelineState(function: fn)
            } catch {
                Self.log.error("pipeline `\(name)` failed: \(error.localizedDescription)")
                return nil
            }
        }
        guard
            let pClearCounter = makePipeline("caustic_clear_counter"),
            let pClearAccum   = makePipeline("caustic_clear_accum"),
            let pCompact      = makePipeline("caustic_compact"),
            let pEmit         = makePipeline("caustic_emit_trace_splat"),
            let pResolve      = makePipeline("caustic_resolve")
        else {
            return nil
        }

        // Output texture — RGBA8 is plenty for tone-mapped, gamma'd caustics.
        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: textureSize, height: textureSize,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead, .shaderWrite]
        texDesc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: texDesc) else {
            Self.log.error("could not allocate caustic texture")
            return nil
        }
        tex.label = "Caustics.texture"

        // Accumulation buffer: texSize² × 4 floats (atomic). The 4th channel
        // is unused (kept for 16-byte alignment and future weight tracking).
        let accumCount = textureSize * textureSize * 4
        guard let accum = device.makeBuffer(
            length: MemoryLayout<Float>.stride * accumCount,
            options: [.storageModePrivate]
        ) else {
            Self.log.error("could not allocate accumulation buffer")
            return nil
        }
        accum.label = "Caustics.accum"

        // Compact-tri list. Sized to the MC bridge's max vertex count / 3
        // (every slot _could_ be non-degenerate in principle; in practice
        // the count is ~10× smaller because most cells are empty).
        let maxTris = bridge.vertexCount / 3
        guard let compact = device.makeBuffer(
            length: MemoryLayout<CompactTri>.stride * max(maxTris, 1),
            options: [.storageModePrivate]
        ) else {
            Self.log.error("could not allocate compact triangle buffer")
            return nil
        }
        compact.label = "Caustics.compactTris"

        guard let counter = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModePrivate]
        ) else {
            Self.log.error("could not allocate counter buffer")
            return nil
        }
        counter.label = "Caustics.counter"

        guard let uniforms = device.makeBuffer(
            length: MemoryLayout<CausticsUniforms>.stride,
            options: [.storageModeShared]
        ) else {
            Self.log.error("could not allocate uniform buffer")
            return nil
        }
        uniforms.label = "Caustics.uniforms"

        // Receiver geometry — two triangles in a quad, written into shared
        // buffers so we can build the AS from them. Local-space matches
        // world-space here (the receiver lives in the same coord frame the
        // photons are computed in — the caller is responsible for ensuring
        // the MC surface and receiver share that frame).
        let y = receiver.y
        let x0 = receiver.minX, x1 = receiver.minX + receiver.extentX
        let z0 = receiver.minZ, z1 = receiver.minZ + receiver.extentZ
        let verts: [SIMD3<Float>] = [
            SIMD3(x0, y, z0),
            SIMD3(x1, y, z0),
            SIMD3(x1, y, z1),
            SIMD3(x0, y, z1)
        ]
        let indices: [UInt32] = [0, 1, 2, 0, 2, 3]
        guard
            let vbuf = device.makeBuffer(
                bytes: verts,
                length: MemoryLayout<SIMD3<Float>>.stride * verts.count,
                options: [.storageModeShared]
            ),
            let ibuf = device.makeBuffer(
                bytes: indices,
                length: MemoryLayout<UInt32>.stride * indices.count,
                options: [.storageModeShared]
            )
        else {
            Self.log.error("could not allocate receiver mesh buffers")
            return nil
        }
        vbuf.label = "Caustics.receiverVerts"
        ibuf.label = "Caustics.receiverIdx"

        self.device = device
        self.clearCounterPipeline = pClearCounter
        self.clearAccumPipeline   = pClearAccum
        self.compactPipeline      = pCompact
        self.emitTraceSplatPipeline = pEmit
        self.resolvePipeline      = pResolve
        self.causticTexture       = tex
        self.textureSize          = textureSize
        self.accumBuffer          = accum
        self.compactBuffer        = compact
        self.counterBuffer        = counter
        self.uniformBuffer        = uniforms
        self.receiverVertexBuffer = vbuf
        self.receiverIndexBuffer  = ibuf
        self.receiver             = receiver
        self.surfaceBridge        = bridge
        self.surfaceVertexCount   = bridge.vertexCount

        // Receiver AS — build once, on its own command buffer so we can
        // wait for it here before any encode call needs it.
        guard buildReceiverAS() else { return nil }

        Self.log.notice("""
            Caustics initialised \
            tex=\(textureSize)×\(textureSize) \
            photons=\(self.photonCount) \
            maxTris=\(maxTris) \
            receiver=(\(receiver.minX), \(receiver.y), \(receiver.minZ)) \
            ext=(\(receiver.extentX), \(receiver.extentZ))
            """)
    }

    // ── Receiver AS (one-shot) ───────────────────────────────────

    private func buildReceiverAS() -> Bool {
        let geomDesc = MTLAccelerationStructureTriangleGeometryDescriptor()
        geomDesc.vertexBuffer = receiverVertexBuffer
        geomDesc.vertexBufferOffset = 0
        geomDesc.vertexStride = MemoryLayout<SIMD3<Float>>.stride
        geomDesc.vertexFormat = .float3
        geomDesc.indexBuffer = receiverIndexBuffer
        geomDesc.indexBufferOffset = 0
        geomDesc.indexType = .uint32
        geomDesc.triangleCount = 2

        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [geomDesc]
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)

        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(
                length: max(sizes.buildScratchBufferSize, 16),
                options: [.storageModePrivate]
            ),
            let queue = device.makeCommandQueue(),
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("could not build receiver AS")
            return false
        }
        storage.label = "Caustics.receiverAS"
        enc.build(accelerationStructure: storage,
                  descriptor: asDesc,
                  scratchBuffer: scratch,
                  scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        self.receiverAS = storage
        return true
    }

    // ── Per-frame encode ─────────────────────────────────────────

    /// Encode the full caustic pipeline into `cb`. Must be called AFTER
    /// the MC bridge has encoded its extract on the same command
    /// buffer, since we read its position/normal buffers.
    public func encode(to cb: MTLCommandBuffer, time: Float) {
        guard let bridge = surfaceBridge else { return }

        // Write uniforms.
        let uPtr = uniformBuffer.contents().bindMemory(to: CausticsUniforms.self,
                                                       capacity: 1)
        // Per-photon weight — energy budget / photons. Boosted from 30
        // → 250 because the resolve kernel's Reinhard tone-map was
        // crushing peaks: with 65k photons across a 512² texel grid the
        // density is ~0.25 photons/texel, Reinhard clamps these to
        // small values, and the texture lit the floor only faintly.
        // Removing Reinhard (see CausticsRT.metal) plus the higher
        // weight gives bright caustic peaks that read as the dancing
        // light signature of refracted-light convergence.
        // The accumulator now persists 80% per frame (temporal EMA),
        // so steady-state intensity is 1/(1-0.8) = 5× what a single
        // frame's contribution would be. Compensate by halving the
        // per-photon weight ~5×: 900 → 180 keeps the on-screen
        // brightness comparable to the old hard-clear pipeline while
        // eliminating the flicker.
        let weight: Float = 180.0 / Float(photonCount)
        uPtr.pointee = CausticsUniforms(
            surfaceVertexCount: UInt32(surfaceVertexCount),
            photonCount: UInt32(photonCount),
            texSize: UInt32(textureSize),
            surfaceAreaEpsilon: surfaceAreaEpsilon,
            lightDirection: (lightDirection.x, lightDirection.y, lightDirection.z),
            ior: ior,
            lightColor: (lightColor.x, lightColor.y, lightColor.z),
            lightIntensity: lightIntensity,
            receiverY: receiver.y,
            receiverMinX: receiver.minX,
            receiverMinZ: receiver.minZ,
            receiverExtentX: receiver.extentX,
            receiverExtentZ: receiver.extentZ,
            photonWeight: weight,
            time: time,
            _pad0: 0
        )

        // 1. Clear the compact-tri counter (single thread).
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Caustics.clearCounter"
            enc.setComputePipelineState(clearCounterPipeline)
            enc.setBuffer(counterBuffer, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2. Decay-rather-than-clear the accumulation buffer. The kernel
        // multiplies each slot by `decay` so we keep an exponential
        // moving average of caustic intensity. Eliminates the
        // frame-to-frame flicker that randomised photon sampling
        // produces — even with a perfectly stationary water surface,
        // each frame's photons fall at slightly different texels, so
        // a hard-clear → re-accumulate cycle strobes; this decay loop
        // smooths it out.
        let accumCount = textureSize * textureSize * 4
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Caustics.decayAccum"
            enc.setComputePipelineState(clearAccumPipeline)
            enc.setBuffer(accumBuffer, offset: 0, index: 0)
            var cnt: UInt32 = UInt32(accumCount)
            enc.setBytes(&cnt, length: MemoryLayout<UInt32>.stride, index: 1)
            var decay: Float = 0.80   // 20% new / 80% prev → ~10-frame EMA
            enc.setBytes(&decay, length: MemoryLayout<Float>.stride, index: 2)
            let tg = clearAccumPipeline.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: accumCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 3. Compact non-degenerate MC triangles.
        let triSlotCount = surfaceVertexCount / 3
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Caustics.compact"
            enc.setComputePipelineState(compactPipeline)
            enc.setBuffer(bridge.positionBuffer, offset: 0, index: 0)
            enc.setBuffer(compactBuffer,         offset: 0, index: 1)
            enc.setBuffer(counterBuffer,         offset: 0, index: 2)
            enc.setBuffer(uniformBuffer,         offset: 0, index: 3)
            let tg = min(compactPipeline.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreads(MTLSize(width: triSlotCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 4. Emit + trace + splat.
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Caustics.emitTraceSplat"
            enc.setComputePipelineState(emitTraceSplatPipeline)
            enc.setBuffer(bridge.positionBuffer, offset: 0, index: 0)
            enc.setBuffer(bridge.normalBuffer,   offset: 0, index: 1)
            enc.setBuffer(compactBuffer,         offset: 0, index: 2)
            // Counter is read as `const device uint&` — same buffer.
            enc.setBuffer(counterBuffer,         offset: 0, index: 3)
            enc.setBuffer(uniformBuffer,         offset: 0, index: 4)
            enc.setBuffer(accumBuffer,           offset: 0, index: 5)
            enc.setAccelerationStructure(receiverAS, bufferIndex: 6)
            let tg = min(emitTraceSplatPipeline.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreads(MTLSize(width: photonCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 5. Resolve (blur + tone map → output texture).
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Caustics.resolve"
            enc.setComputePipelineState(resolvePipeline)
            enc.setBuffer(accumBuffer, offset: 0, index: 0)
            enc.setTexture(causticTexture, index: 0)
            let w = resolvePipeline.threadExecutionWidth
            let h = resolvePipeline.maxTotalThreadsPerThreadgroup / w
            enc.dispatchThreads(MTLSize(width: textureSize, height: textureSize, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
            enc.endEncoding()
        }
    }
}

// ── Shader modifier helper ───────────────────────────────────────

extension CausticsRT {
    /// `.surface` shader modifier source that maps the caustic texture
    /// across the receiver rectangle by world XZ and adds the result
    /// to `_surface.emission`. Written in MSL on purpose — SceneKit's
    /// macOS 15 modifier translator doesn't translate GLSL texture
    /// sampling reliably (the material falls back to magenta error).
    ///
    /// The material must have these uniforms set before the modifier
    /// is attached:
    ///   - `causticTex`: SCNMaterialProperty wrapping the MTLTexture
    ///   - `causticBlend`: NSNumber Float — 0 disables, 1 is full
    ///   - `causticMinX`, `causticMinZ`: NSNumber Float — receiver origin
    ///   - `causticExtentX`, `causticExtentZ`: NSNumber Float — receiver size
    public static func surfaceModifierSource() -> String {
        return """
        #pragma arguments
        texture2d<float> causticTex;
        float causticBlend;
        float causticMinX;
        float causticMinZ;
        float causticExtentX;
        float causticExtentZ;

        #pragma body
        if (causticBlend > 0.0001) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            // _surface.position is view-space; transform back to world.
            float4 worldPos = u_inverseViewTransform * float4(_surface.position, 1.0);
            float u = (worldPos.x - causticMinX) / causticExtentX;
            float v = (worldPos.z - causticMinZ) / causticExtentZ;
            float2 uv = clamp(float2(u, v), float2(0.001), float2(0.999));
            float4 c = causticTex.sample(s, uv);
            _surface.emission.rgb += c.rgb * causticBlend;
        }
        """
    }
}

// ── Layout-compatible mirrors ────────────────────────────────────

// MUST match struct CausticsUniforms in CausticsRT.metal. Tuple-of-Float
// (12 bytes / 4-aligned) matches packed_float3 layout.
private struct CausticsUniforms {
    var surfaceVertexCount: UInt32
    var photonCount: UInt32
    var texSize: UInt32
    var surfaceAreaEpsilon: Float
    var lightDirection: (Float, Float, Float)
    var ior: Float
    var lightColor: (Float, Float, Float)
    var lightIntensity: Float
    var receiverY: Float
    var receiverMinX: Float
    var receiverMinZ: Float
    var receiverExtentX: Float
    var receiverExtentZ: Float
    var photonWeight: Float
    var time: Float
    var _pad0: Float
}

// MUST match struct CompactTri in CausticsRT.metal.
private struct CompactTri {
    var base: UInt32
    var area: Float
    var cumulativeArea: Float
    var _pad0: Float
}
