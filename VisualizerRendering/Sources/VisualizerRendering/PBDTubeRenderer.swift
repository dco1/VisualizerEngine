import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore

// ── PBD TUBE RENDERER ────────────────────────────────────────────────────────
//
// Wires PBDSolver → DynamicMesh for a cylindrical tube (floppy hot dog, rope,
// tentacle). Every frame it appends a tube-expand compute pass to the same
// command buffer as the physics step, so the GPU sees a coherent spine before
// expanding it into a full tube mesh. Zero CPU round-trip.
//
// INTEGRATION PATTERN (add to your scene controller)
// ────────────────────────────────────────────────────
//   guard
//       let device = GPUCapabilities.device,
//       let solver = PBDSolver(device: device, maxParticles: 12, maxConstraints: 11),
//       let tubeRenderer = PBDTubeRenderer(solver: solver, spineCount: 12,
//                                          ringSegments: 16, radius: 0.06)
//   else { return }
//   solver.configureChain(count: 12, segLength: 0.08)
//   let node = SCNNode()
//   node.geometry = tubeRenderer.mesh.geometry
//   node.geometry?.firstMaterial = yourSausageMaterial
//   scene.rootNode.addChildNode(node)
//   // each frame: tubeRenderer.tick(wallDt: Float(dt))
//
// WHERE TO GO NEXT
// ─────────────────
// • Normals (Phase 2): the tube-expand kernel already computes the per-vertex
//   radial direction. Add a second output buffer in PBD.metal and write
//   normalize(ringVert - spinePos) into it. Wire into DynamicMesh.normalBuffer.
//
// • Multiple chains per frame (Phase 2): batch all hot-dog particles into one
//   larger particleBuffer. Add a chainOffset uniform so one tube-expand dispatch
//   handles chain N's slice. Avoids creating one solver per hot dog.
//
// • Grass (separate simpler path): each blade is a 2-D bend parameterised by
//   two floats. A vertex shader reads a wind-noise texture and displaces the
//   blade tip. No compute needed — cheaper and simpler than PBD for grass.
//
// • Fire (SCNParticleSystem path): SCNParticleSystem + custom fragment shader
//   modifier handles stylised fire without new infrastructure. Add a 2-D
//   heat/smoke advection texture (128×128 RG16Float) for colour/opacity.
//
// • Liquid foam (SPH, Phase 5): replace distance constraints with SPH pressure +
//   viscosity kernels. The spatial hash from Phase 4 provides neighbour queries.
//
// • MLS-MPM water (Phase 5+): hybrid particle↔grid. Requires MTLTexture3D,
//   P2G scatter with atomics, G2P gather, and a pressure-projection pass.
//   Validate SPH foam first before reaching for MLS-MPM.

@MainActor
public final class PBDTubeRenderer {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "PBDTubeRenderer")

    public let solver: PBDSolver
    public let mesh: DynamicMesh

    private let expandPipeline: MTLComputePipelineState
    private let expandUniformBuffer: MTLBuffer
    private let buildCollidersPipeline: MTLComputePipelineState
    private let buildCollidersUniformBuffer: MTLBuffer

    public let ringSegments: Int
    public let spineCount: Int
    public var radius: Float

    // Catmull-Rom sub-samples per spine segment. 4 turns 4 physics particles
    // into 13 visual rings — smooth curves without extra simulation cost.
    // Increase for very low spine counts or high-quality renders; keep ≤8 to
    // stay under the UInt16 vertex-count limit with large ringSegments.
    public let subSegments: Int

    // Optional fixed reference up-axis for the tube-expand frame. nil = use
    // the kernel's default (world-Y with a Z fallback near vertical tangents).
    // A non-nil value pins the ring-frame seed to that vector, eliminating
    // the chevron pucker that the default's mid-curve flip produces for
    // planar curves (e.g. Hotdog Font glyphs in the XY plane → fixedUp = Z).
    public var fixedUpAxis: SIMD3<Float>?

    // Tip-taper fraction: 0 = perfect cylinder; 0.25 = tips 25% narrower than
    // the body centre. Forwarded to TubeExpandUniforms each frame.
    public var taperFactor: Float = 0

    /// Straight-line render mode (HotdogDropUltra). When true, the tube-expand
    /// kernel renders the tube as a perfect straight capsule from spine[0] to
    /// spine[N-1] instead of following the Catmull-Rom spline. Eliminates silhouette
    /// lobes at knot bends (silhouette bulge at outer radius of a curved tube) and
    /// frame-seam artifacts. Physics capsules are unaffected. Default false.
    public var useStraightLine: Bool = false

    /// Rotation-minimizing frame. When true (and `useStraightLine` is false), the
    /// tube-expand kernel propagates ONE coherent cross-section frame along the
    /// Catmull-Rom centerline instead of rebuilding it independently per ring.
    /// This eliminates the pinched waist a BENT tube gets when `frameFromTangent`
    /// discontinuously switches its up-axis as the centerline tangent sweeps near
    /// vertical — while keeping the organic bend (unlike `useStraightLine`, which
    /// kills the bend). Use for tubes that bend through arbitrary orientations
    /// (e.g. franks stuffed/jammed in a tube). Default false (legacy per-ring frame).
    public var useRotationMinimizingFrame: Bool = false

    /// Dip-coat waterline (HotdogDropUltra mustard bath). When `dipStrength`
    /// > 0, tube vertices below `dipY` (within `dipRadius` of the world Y
    /// axis) blend their vertex RGB toward `dipColorMul` — a per-channel
    /// albedo MULTIPLIER (fluid colour ÷ body colour) so the coated zone
    /// renders as the fluid. Carries the sub-voxel "wet film" the coarse MC
    /// fluid grid cannot mesh; driven by the sim's measured surface height.
    public var dipY: Float = 0
    public var dipRadius: Float = 0
    public var dipStrength: Float = 0
    public var dipColorMul: SIMD3<Float> = SIMD3(1, 1, 1)

    /// Per-frank BODY colour multiplier written into the tube's vertex-colour
    /// stream (multiplied into albedo at shading). Default (1,1,1) = white
    /// identity, so per-instance scenes are unchanged. The BATCHED path sets this
    /// to the frank's per-frank albedo so a single white InstanceRef + the shared
    /// colour buffer reproduce per-frank tone in one draw. A dip-coat blends this
    /// body colour → `dipColorMul` below the waterline.
    public var bodyColorMul: SIMD3<Float> = SIMD3(1, 1, 1)

    /// External-output redirection (batched-tube-mesh path). When set, the
    /// tube-expand kernel writes THIS tube's vertices into a slice of a SHARED
    /// set of buffers (owned by `BatchedTubeMesh`) instead of the renderer's
    /// own per-tube `mesh.*` buffers — so many franks land in one registered
    /// mesh and draw in ONE call. `vertexBase` is the tube's slot start (in
    /// vertices); the buffers are bound at `vertexBase * stride`. The kernel's
    /// own writes are vertex-relative (verts[id*ringSegments + r], cap poles at
    /// totalRings*ringSegments [+1]), so binding at the slot offset places every
    /// vertex into [vertexBase, vertexBase + vertsPerTube). `nil` → unchanged
    /// (writes into `mesh.*` at offset 0).
    public struct ExternalTubeOutput {
        public let position: MTLBuffer   // packed_float3, 12-byte stride
        public let normal: MTLBuffer     // packed_float3, 12-byte stride
        public let color: MTLBuffer      // float4, 16-byte stride
        public let vertexBase: Int       // slot start, in vertices
        public init(position: MTLBuffer, normal: MTLBuffer, color: MTLBuffer, vertexBase: Int) {
            self.position = position; self.normal = normal
            self.color = color; self.vertexBase = vertexBase
        }
    }
    public var externalOutput: ExternalTubeOutput? = nil

    // Derived: total visual rings written by the tube-expand kernel each frame.
    public var visualRingCount: Int { (spineCount - 1) * subSegments + 1 }

    // ── Init ─────────────────────────────────────────────────────────────────

    public init?(solver: PBDSolver, spineCount: Int, ringSegments: Int,
                 radius: Float, subSegments: Int = 4,
                 fixedUpAxis: SIMD3<Float>? = nil) {
        let device = solver.device

        // PERF: share the cached pipeline with every other PBDTubeRenderer.
        // First call on a device pays the build cost; everyone after is free.
        guard let cached = SimPipelineCache.shared.pbdPipelines(for: device) else {
            Self.log.error("pbdTubeExpand pipeline cache failed — check PBD.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        let pipeline = cached.tubeExpand

        guard let uBuf = device.makeBuffer(length: MemoryLayout<TubeExpandUniforms>.stride,
                                           options: .storageModeShared),
              let bcBuf = device.makeBuffer(length: MemoryLayout<BuildCollidersUniforms>.stride,
                                            options: .storageModeShared)
        else {
            Self.log.error("PBDTube uniform-buffer allocation failed")
            return nil
        }
        uBuf.label = "PBDTube.uniforms"
        bcBuf.label = "PBDTube.buildCollidersUniforms"

        let visualRings = (spineCount - 1) * subSegments + 1
        guard let dynamicMesh = DynamicMesh.makeTube(device: device,
                                                     ringCount: visualRings,
                                                     ringSegments: ringSegments) else {
            Self.log.error("DynamicMesh.makeTube failed (vertexCount probably > 65535)")
            return nil
        }

        self.solver                       = solver
        self.expandPipeline               = pipeline
        self.expandUniformBuffer          = uBuf
        self.buildCollidersPipeline       = cached.buildColliders
        self.buildCollidersUniformBuffer  = bcBuf
        self.mesh                         = dynamicMesh
        self.ringSegments                 = ringSegments
        self.spineCount                   = spineCount
        self.radius                       = radius
        self.subSegments                  = subSegments
        self.fixedUpAxis                  = fixedUpAxis
    }

    // ── Per-frame tick ────────────────────────────────────────────────────────

    // Encode one fixed physics step + tube-expand into an externally-owned
    // command buffer. The caller is responsible for the substep accumulator
    // and for committing the buffer. Use this from a batched controller tick
    // (one CB shared across all tubes per substep) to avoid N_tubes separate
    // command buffer submissions per frame.
    public func encode(to commandBuffer: MTLCommandBuffer, dt: Float) {
        solver.encode(to: commandBuffer, dt: dt)
        encodeTubeExpand(to: commandBuffer)
    }

    // Stand-alone tick for single-tube use (creates its own CB per step).
    public func tick(wallDt: Float) {
        let fixedDt: Float = 1.0 / 120.0
        solver.accumulator += wallDt
        var steps = 0
        while solver.accumulator >= fixedDt && steps < 4 {
            guard let cb = solver.commandQueue.makeCommandBuffer() else { break }
            cb.label = "PBDTube.step"
            solver.encode(to: cb, dt: fixedDt)
            encodeTubeExpand(to: cb)
            cb.commit()
            solver.accumulator -= fixedDt
            steps += 1
        }
    }

    // ── GPU collider rebuild ─────────────────────────────────────────────────
    //
    // Writes one capsule per spine segment of THIS tube into the caller's
    // shared collider buffer at the given offset. Use between the integrate
    // and constraint phases so SDF reads post-integrate capsules.
    //
    // The dispatch is `spineCount - 1` threads — one per capsule — and each
    // thread writes to `target[targetOffset + threadID]`. The caller is
    // responsible for offset arithmetic (sum of `spineCount - 1` across
    // earlier tubes in the per-substep collider layout).
    public func encodeBuildColliders(to commandBuffer: MTLCommandBuffer,
                                     target: MTLBuffer,
                                     targetOffset: Int,
                                     ownerID: UInt32) {
        let capsuleCount = spineCount - 1
        guard capsuleCount > 0,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBDTube.buildColliders"
        encoder.setComputePipelineState(buildCollidersPipeline)

        encoder.setBuffer(solver.particleBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(target,                       offset: 0, index: 1)

        let uPtr = buildCollidersUniformBuffer.contents()
            .bindMemory(to: BuildCollidersUniforms.self, capacity: 1)
        uPtr.pointee = BuildCollidersUniforms(
            particleCount: UInt32(spineCount),
            targetOffset:  UInt32(targetOffset),
            ownerID:       ownerID,
            radius:        radius
        )
        encoder.setBuffer(buildCollidersUniformBuffer, offset: 0, index: 2)

        let w = min(capsuleCount, buildCollidersPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: capsuleCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    // ── Tube-expand pass ─────────────────────────────────────────────────────

    // ── Batched-dispatch primitives (one encoder shared across many tubes) ──
    //
    // Mirror of the `PBDSolver.dispatch*` family: bind + dispatch into a
    // caller-owned encoder so a multi-tube controller can run one
    // MTLComputeCommandEncoder per phase across all bodies, instead of one
    // encoder per body per phase. See `PBDSolver.swift` for the rationale and
    // `HotdogDropPlusController.tickPBD` for the calling pattern.

    public func dispatchBuildColliders(into encoder: MTLComputeCommandEncoder,
                                       target: MTLBuffer,
                                       targetOffset: Int,
                                       ownerID: UInt32) {
        let capsuleCount = spineCount - 1
        guard capsuleCount > 0 else { return }
        encoder.setComputePipelineState(buildCollidersPipeline)
        encoder.setBuffer(solver.particleBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(target,                       offset: 0, index: 1)

        let uPtr = buildCollidersUniformBuffer.contents()
            .bindMemory(to: BuildCollidersUniforms.self, capacity: 1)
        uPtr.pointee = BuildCollidersUniforms(
            particleCount: UInt32(spineCount),
            targetOffset:  UInt32(targetOffset),
            ownerID:       ownerID,
            radius:        radius
        )
        encoder.setBuffer(buildCollidersUniformBuffer, offset: 0, index: 2)

        let w = min(capsuleCount, buildCollidersPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: capsuleCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    public func dispatchTubeExpand(into encoder: MTLComputeCommandEncoder) {
        let rings = visualRingCount
        guard rings > 0 else { return }
        encoder.setComputePipelineState(expandPipeline)
        encoder.setBuffer(solver.particleBuffer.buffer, offset: 0, index: 0)
        // Batched path: redirect this tube's vertex writes into its slot of the
        // shared buffers (bound at the slot byte-offset; the kernel's own
        // vertex-relative indexing fills [vertexBase, vertexBase+vertsPerTube)).
        if let ext = externalOutput {
            encoder.setBuffer(ext.position, offset: ext.vertexBase * 12, index: 1) // packed_float3
            encoder.setBuffer(ext.normal,   offset: ext.vertexBase * 12, index: 3)
            encoder.setBuffer(ext.color,    offset: ext.vertexBase * 16, index: 4) // float4
        } else {
            encoder.setBuffer(mesh.positionBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(mesh.normalBuffer.buffer,   offset: 0, index: 3)
            encoder.setBuffer(mesh.colorBuffer.buffer,    offset: 0, index: 4)
        }

        let up = fixedUpAxis ?? SIMD3<Float>(0, 0, 0)
        let uPtr = expandUniformBuffer.contents().bindMemory(to: TubeExpandUniforms.self,
                                                              capacity: 1)
        uPtr.pointee = TubeExpandUniforms(
            ringSegments:    UInt32(ringSegments),
            spineCount:      UInt32(spineCount),
            radius:          radius,
            subSegments:     UInt32(subSegments),
            useFixedUp:      fixedUpAxis == nil ? 0 : 1,
            fixedUpX:        up.x,
            fixedUpY:        up.y,
            fixedUpZ:        up.z,
            spineWrap:       solver.ringCount > 0 ? 1 : 0,
            spineOffset:     UInt32(solver.ringHead),
            taperFactor:     taperFactor,
            useStraightLine: useStraightLine ? 1 : 0,
            dipY:            dipY,
            dipRadius:       dipRadius,
            dipStrength:     dipStrength,
            dipR:            dipColorMul.x,
            dipG:            dipColorMul.y,
            dipB:            dipColorMul.z,
            bodyR:           bodyColorMul.x,
            bodyG:           bodyColorMul.y,
            bodyB:           bodyColorMul.z,
            useRMF:          useRotationMinimizingFrame ? 1 : 0
        )
        encoder.setBuffer(expandUniformBuffer, offset: 0, index: 2)

        let w = min(rings, expandPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: rings, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
    }

    public func encodeTubeExpand(to commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "PBDTube.expand"
        encoder.setComputePipelineState(expandPipeline)

        encoder.setBuffer(solver.particleBuffer.buffer, offset: 0, index: 0)
        if let ext = externalOutput {
            encoder.setBuffer(ext.position, offset: ext.vertexBase * 12, index: 1)
            encoder.setBuffer(ext.normal,   offset: ext.vertexBase * 12, index: 3)
            encoder.setBuffer(ext.color,    offset: ext.vertexBase * 16, index: 4)
        } else {
            encoder.setBuffer(mesh.positionBuffer.buffer, offset: 0, index: 1)
            encoder.setBuffer(mesh.normalBuffer.buffer,   offset: 0, index: 3)
            encoder.setBuffer(mesh.colorBuffer.buffer,    offset: 0, index: 4)
        }

        let up = fixedUpAxis ?? SIMD3<Float>(0, 0, 0)
        let uPtr = expandUniformBuffer.contents().bindMemory(to: TubeExpandUniforms.self,
                                                              capacity: 1)
        uPtr.pointee = TubeExpandUniforms(
            ringSegments:    UInt32(ringSegments),
            spineCount:      UInt32(spineCount),
            radius:          radius,
            subSegments:     UInt32(subSegments),
            useFixedUp:      fixedUpAxis == nil ? 0 : 1,
            fixedUpX:        up.x,
            fixedUpY:        up.y,
            fixedUpZ:        up.z,
            spineWrap:       solver.ringCount > 0 ? 1 : 0,
            spineOffset:     UInt32(solver.ringHead),
            taperFactor:     taperFactor,
            useStraightLine: useStraightLine ? 1 : 0,
            dipY:            dipY,
            dipRadius:       dipRadius,
            dipStrength:     dipStrength,
            dipR:            dipColorMul.x,
            dipG:            dipColorMul.y,
            dipB:            dipColorMul.z,
            bodyR:           bodyColorMul.x,
            bodyG:           bodyColorMul.y,
            bodyB:           bodyColorMul.z,
            useRMF:          useRotationMinimizingFrame ? 1 : 0
        )
        encoder.setBuffer(expandUniformBuffer, offset: 0, index: 2)

        let rings = visualRingCount
        let w = min(rings, expandPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: rings, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }
}
