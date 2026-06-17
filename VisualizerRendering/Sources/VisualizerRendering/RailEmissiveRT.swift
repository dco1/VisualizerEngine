import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

/// ── RAIL EMISSIVE RT (revertable) ──────────────────────────────
/// Hardware-ray-traced area-light integration for the Egg Roller
/// scene's rail-glow tube + marquee-bulb merged meshes onto the eggs.
///
/// SceneKit's stock lighting only takes contributions from `SCNLight`
/// objects — emissive surfaces glow visually but don't cast onto other
/// geometry. This class steps outside SceneKit to fix that:
///
/// 1. At init, build a primitive acceleration structure from the two
///    merged meshes the rail-light system already constructs (rail-glow
///    tube + bulb sphere grid). Both meshes carry a per-vertex `uvX`
///    that indexes into the `BulbLUTWriter` texture, so the kernel can
///    read the actual emissive colour at any hit point.
/// 2. Per tick, the controller hands us an array of per-egg world
///    positions plus a reference to the live LUT. The compute kernel
///    fires N rays per egg uniformly into the unit sphere, intersects
///    with the AS, and integrates the LUT colour at each hit.
/// 3. The kernel's output is a per-egg `float4`: `rgb` = mean emissive
///    colour over the hits, `a` = fraction of rays that found
///    emissive surface. The controller multiplies the two to drive
///    each egg's fill-light intensity, and uses the colour directly.
///
/// **Where this lives in the hardware budget.** ~30 eggs × N networks
/// × ~16 rays = a couple thousand rays per tick against an AS of
/// ~100k triangles. Hardware RT eats this in microseconds on Apple
/// silicon. The Swift-side cost is one short command-buffer commit
/// per network, with the per-egg output buffer in shared storage so
/// we read it back without an explicit blit.
///
/// **Why not literal-RT-from-the-egg-surface.** A per-fragment RT
/// integration on the egg shell would be the most physically correct
/// answer (each shaded fragment fires its own hemisphere rays), but
/// it'd require a SceneKit shader modifier on the egg material with
/// access to a Metal AS — and shader modifiers in SceneKit have
/// well-documented compatibility hazards (see
/// `docs/known-issues/shader-modifier-incompatible-surfaces.md`).
/// Integrating per-egg-centre instead, then driving the existing per-
/// egg fill `SCNLight` with the result, lands the colour and
/// intensity in the regular PBR pipeline with no shader-modifier risk.
/// The egg shell still gets diffuse + specular shading from the fill
/// light; that shading just happens to be coloured by the actual rail
/// emission the egg can "see".
///
/// Revert: delete this file + `RailEmissiveRT.metal` + every
/// `// ── RT EMISSIVE (revertable) ──` block in `EggsController` /
/// `EggsRailLights`.
@MainActor
public final class RailEmissiveRT {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "railEmissiveRT")

    // ── Public surface ───────────────────────────────────────────

    /// The live LUT texture sampled at each ray hit. Held by reference
    /// — the rail-light system mutates this every tick.
    private weak var lutTextureSource: BulbLUTWriterRef?

    // ── Metal plumbing ───────────────────────────────────────────

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    /// Primitive AS over the rail-glow + bulb merged meshes. Built once
    /// — the geometry is static for the network's lifetime, only the
    /// LUT texture's contents change.
    private var primitiveAS: MTLAccelerationStructure?
    private var primitiveScratch: MTLBuffer?
    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let uvXBuffer: MTLBuffer
    private let triangleCount: Int

    /// Egg position input. Grown as needed.
    private var eggPositionBuffer: MTLBuffer?
    private var eggPositionCapacity: Int = 0

    /// Per-egg output (float4 — rgb = mean colour, a = hit fraction).
    /// `.shared` storage so the CPU read-back path is just a pointer.
    private var outputBuffer: MTLBuffer?
    private var outputCapacity: Int = 0

    /// Uniforms — small fixed struct, allocated once.
    private let uniformBuffer: MTLBuffer

    /// Rays per egg per tick. 16 is enough to get a smooth integrated
    /// colour without being noisy; bump to 32 if banding shows up in a
    /// future pattern (e.g. very sparse rail layouts where one or two
    /// hits dominate).
    public let raysPerEgg: Int

    /// Max ray distance. Rails span the full network (~190 m); 80 m is
    /// well past the longest sight-line from any egg to the farthest
    /// emissive surface that could plausibly contribute. Beyond this we
    /// don't bother — the inverse-square falloff has long since killed
    /// any meaningful contribution.
    public let rayMaxDistance: Float

    /// Counter that gets mixed into the per-ray seed so consecutive
    /// frames sample different directions — without this the same rays
    /// hit the same triangles every frame and the result is biased.
    private var seedFrame: UInt32 = 0

    // ── Init ─────────────────────────────────────────────────────

    /// Returns nil on hardware without RT support, if the kernel is
    /// missing from the default Metal library, or if the merged-mesh
    /// buffers fail to allocate. A nil return is handled gracefully by
    /// the controller — the analytical CPU integration is the fallback.
    /// Number of triangles in the rail-glow sub-mesh — the bulb mesh
    /// occupies `[railGlowTriCount, totalTri)`. The kernel uses this
    /// to honour the per-tick visibility toggles passed to `dispatch`.
    public let railGlowTriCount: Int

    public init?(
        meshPositions: [SIMD3<Float>],
        meshIndices: [UInt32],
        meshUVX: [Float],
        railGlowTriCount: Int,
        raysPerEgg: Int = 64,
        rayMaxDistance: Float = 80.0
    ) {
        guard
            GPUCapabilities.supportsRaytracing,
            let device = GPUCapabilities.device,
            device.supportsRaytracing
        else {
            Self.log.notice("ray tracing unavailable; skipping RailEmissiveRT init")
            return nil
        }
        guard let queue = device.makeCommandQueue() else {
            Self.log.warning("could not create command queue")
            return nil
        }
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .main)
        } catch {
            Self.log.error("default Metal library missing: \(error.localizedDescription)")
            return nil
        }
        guard let function = library.makeFunction(name: "railEmissiveIntegrate") else {
            Self.log.error("kernel `railEmissiveIntegrate` not found")
            return nil
        }
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            Self.log.error("compute pipeline build failed: \(error.localizedDescription)")
            return nil
        }

        guard meshIndices.count % 3 == 0, !meshIndices.isEmpty,
              meshUVX.count == meshPositions.count else {
            Self.log.error("merged mesh buffers malformed (vCount=\(meshPositions.count) iCount=\(meshIndices.count) uvCount=\(meshUVX.count))")
            return nil
        }

        guard
            let vb = device.makeBuffer(
                bytes: meshPositions,
                length: MemoryLayout<SIMD3<Float>>.stride * meshPositions.count,
                options: [.storageModeShared]
            ),
            let ib = device.makeBuffer(
                bytes: meshIndices,
                length: MemoryLayout<UInt32>.stride * meshIndices.count,
                options: [.storageModeShared]
            ),
            let ub = device.makeBuffer(
                bytes: meshUVX,
                length: MemoryLayout<Float>.stride * meshUVX.count,
                options: [.storageModeShared]
            )
        else {
            Self.log.error("could not allocate merged mesh Metal buffers")
            return nil
        }
        vb.label = "RailEmissiveRT.positions"
        ib.label = "RailEmissiveRT.indices"
        ub.label = "RailEmissiveRT.uvX"

        guard let uniforms = device.makeBuffer(
            length: MemoryLayout<RailEmissiveUniforms>.stride,
            options: [.storageModeShared]
        ) else {
            Self.log.error("could not allocate uniform buffer")
            return nil
        }
        uniforms.label = "RailEmissiveRT.uniforms"

        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.uvXBuffer = ub
        self.triangleCount = meshIndices.count / 3
        self.railGlowTriCount = railGlowTriCount
        self.uniformBuffer = uniforms
        self.raysPerEgg = raysPerEgg
        self.rayMaxDistance = rayMaxDistance

        do {
            try buildPrimitiveAS()
        } catch {
            Self.log.error("primitive AS build failed: \(error.localizedDescription)")
            return nil
        }

        Self.log.notice("""
            RailEmissiveRT initialised \
            tri=\(self.triangleCount) \
            verts=\(meshPositions.count) \
            raysPerEgg=\(raysPerEgg) \
            maxDist=\(rayMaxDistance)
            """)
    }

    // ── Per-tick ─────────────────────────────────────────────────

    /// Dispatch one integration pass for the given egg positions, with
    /// the live LUT texture as the emissive source. After the command
    /// buffer commits, `readResults(count:)` returns the per-egg
    /// `float4` array (`rgb` mean colour, `a` hit fraction).
    public func dispatch(
        eggPositions: [SIMD3<Float>],
        lutTexture: MTLTexture,
        includeRailGlow: Bool,
        includeBulbs: Bool
    ) {
        guard !eggPositions.isEmpty, primitiveAS != nil else { return }
        ensureCapacity(forEggCount: eggPositions.count)

        // Write egg positions. Stored as packed_float3 (12 bytes each),
        // but Swift's SIMD3<Float> is also 16-byte stride — pack
        // explicitly to match Metal's `packed_float3`.
        if let buf = eggPositionBuffer {
            let ptr = buf.contents().bindMemory(
                to: Self.PackedFloat3.self, capacity: eggPositions.count
            )
            for (i, p) in eggPositions.enumerated() {
                ptr[i] = Self.PackedFloat3(x: p.x, y: p.y, z: p.z)
            }
        }

        // Write uniforms.
        seedFrame &+= 1
        let u = RailEmissiveUniforms(
            eggCount: UInt32(eggPositions.count),
            rayCount: UInt32(raysPerEgg),
            seedBase: seedFrame,
            rayMaxDistance: rayMaxDistance,
            railGlowTriCount: UInt32(railGlowTriCount),
            includeRailGlow: includeRailGlow ? 1 : 0,
            includeBulbs: includeBulbs ? 1 : 0
        )
        uniformBuffer.contents().assumingMemoryBound(to: RailEmissiveUniforms.self).pointee = u

        guard
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeComputeCommandEncoder(),
            let as_ = primitiveAS,
            let posBuf = eggPositionBuffer,
            let outBuf = outputBuffer
        else { return }
        cmd.label = "RailEmissiveRT.frame"
        enc.label = "RailEmissiveRT.integrate"

        enc.setComputePipelineState(pipeline)
        enc.setAccelerationStructure(as_, bufferIndex: 0)
        enc.setBuffer(indexBuffer, offset: 0, index: 1)
        enc.setBuffer(uvXBuffer, offset: 0, index: 2)
        enc.setBuffer(posBuf, offset: 0, index: 3)
        enc.setBuffer(uniformBuffer, offset: 0, index: 4)
        enc.setBuffer(outBuf, offset: 0, index: 5)
        enc.setTexture(lutTexture, index: 0)
        // The kernel reads triangle vertex positions through the AS, but
        // Metal needs explicit useResource calls for indirect-AS reads
        // it can't statically analyse — the index buffer is bound above,
        // so the vertex buffer needs a use-residency hint.
        enc.useResource(vertexBuffer, usage: .read)

        let threadgroup = MTLSize(
            width: min(pipeline.maxTotalThreadsPerThreadgroup, 64),
            height: 1, depth: 1
        )
        let grid = MTLSize(width: eggPositions.count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
        enc.endEncoding()
        cmd.commit()
        // Block until the GPU finishes so the CPU read-back below is
        // current. A 2026-05-31 perf audit replaced this with a double-
        // buffered async read-back (+ DispatchSemaphore(2) bound) and
        // measured NO improvement (p95 at density 1.0/bulbs-on held
        // ~68-73 ms vs this version's ~55); the per-tick wait is NOT the
        // bottleneck (per-egg KVC SH-uniform loop + draw-call count are),
        // and async only added queue-saturation surface. Kept synchronous.
        cmd.waitUntilCompleted() // gpu-ok: per-egg readback; async variant measured no faster (2026-05-31 audit), only added saturation risk
    }

    /// Read back the per-egg results from the shared output buffer.
    /// Caller is responsible for sizing — pass the same `count` you
    /// used in `dispatch`.
    public func readResults(count: Int) -> [EggIrradianceResult] {
        guard let outBuf = outputBuffer, count > 0,
              count <= outputCapacity else { return [] }
        let ptr = outBuf.contents().bindMemory(
            to: EggIrradianceResult.self, capacity: count
        )
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // ── AS build ─────────────────────────────────────────────────

    private func buildPrimitiveAS() throws {
        let geomDesc = MTLAccelerationStructureTriangleGeometryDescriptor()
        geomDesc.vertexBuffer = vertexBuffer
        geomDesc.vertexBufferOffset = 0
        geomDesc.vertexStride = MemoryLayout<SIMD3<Float>>.stride
        geomDesc.vertexFormat = .float3
        geomDesc.indexBuffer = indexBuffer
        geomDesc.indexBufferOffset = 0
        geomDesc.indexType = .uint32
        geomDesc.triangleCount = triangleCount

        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [geomDesc]

        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(
                length: max(sizes.buildScratchBufferSize, 16),
                options: [.storageModePrivate]
            ),
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            throw NSError(
                domain: "RailEmissiveRT", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "AS storage / scratch alloc failed"]
            )
        }
        storage.label = "RailEmissiveRT.primitiveAS"
        enc.build(
            accelerationStructure: storage,
            descriptor: asDesc,
            scratchBuffer: scratch,
            scratchBufferOffset: 0
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        self.primitiveAS = storage
        self.primitiveScratch = scratch
    }

    // ── Capacity management ──────────────────────────────────────

    private func ensureCapacity(forEggCount n: Int) {
        if n > eggPositionCapacity {
            // Round up to reduce realloc churn when egg counts wobble.
            let target = max(n, eggPositionCapacity * 2, 32)
            eggPositionBuffer = device.makeBuffer(
                length: MemoryLayout<PackedFloat3>.stride * target,
                options: [.storageModeShared]
            )
            eggPositionBuffer?.label = "RailEmissiveRT.eggPositions"
            eggPositionCapacity = target
        }
        if n > outputCapacity {
            let target = max(n, outputCapacity * 2, 32)
            outputBuffer = device.makeBuffer(
                length: MemoryLayout<EggIrradianceResult>.stride * target,
                options: [.storageModeShared]
            )
            outputBuffer?.label = "RailEmissiveRT.output"
            outputCapacity = target
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    /// 12-byte alignment match for Metal's `packed_float3` (vs Swift's
    /// 16-byte-stride `SIMD3<Float>`).
    private struct PackedFloat3 {
        var x: Float
        var y: Float
        var z: Float
    }
}

/// Mirror of the Metal-side uniforms struct.
private struct RailEmissiveUniforms {
    var eggCount: UInt32
    var rayCount: UInt32
    var seedBase: UInt32
    var rayMaxDistance: Float
    /// Number of triangles in the rail-glow mesh — the bulb mesh
    /// occupies the rest of the index buffer. The kernel uses this
    /// to mask hits on whichever sub-mesh is currently hidden.
    var railGlowTriCount: UInt32
    var includeRailGlow: UInt32
    var includeBulbs: UInt32
}

/// Per-egg output of the `railEmissiveIntegrate` kernel. Layout must
/// match Metal's `EggEmissiveResult` (four `float4`s back-to-back =
/// 64 bytes).
///
///   • `shR / shG / shB`: L1 spherical-harmonic coefficients per RGB
///     channel for the incoming radiance hemisphere — `(Y₀, Y₁ᵧ, Y₁ᵤ,
///     Y₁ₓ)`. The egg material's shader modifier evaluates these at
///     each fragment's world-space normal direction to compute the
///     surface's response to the rail emission.
///   • `meta.x` = hit fraction (0 = nothing emissive in view, 1 =
///     surrounded). Useful for falloff/gating; not needed for the
///     shading itself since the SH coefficients already carry total
///     magnitude.
public struct EggIrradianceResult {
    public var shR: SIMD4<Float>
    public var shG: SIMD4<Float>
    public var shB: SIMD4<Float>
    public var meta: SIMD4<Float>
}

/// Marker so we don't accidentally retain the BulbLUTWriter through
/// this class (the writer is owned by `EggsRailLights.Built`).
/// Unused for now — we just take the texture by argument at dispatch
/// time. Kept as scaffolding in case a future refactor needs a strong
/// reference back to the writer.
private protocol BulbLUTWriterRef: AnyObject {}
