import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── SHAPE-MATCH SOFT BODY (batch) ────────────────────────────────────────────
//
// GPU multi-body soft-body runtime: N highly-compressible plush bodies, each an
// OVERLAPPING shape-matching cluster set (see ShapeMatchSoftBody.metal for the
// math + APD rotation extraction). Built for the Teddy Bear subject of Hot Dog
// Press, but the runtime is subject-agnostic — a `SoftBodyTemplate` (rest
// particles, clusters, skinned mesh) is all it needs.
//
// Plugs into the shared runtime exactly like the other solvers (CLAUDE.md: plug
// in, don't duplicate): one `SimEngine` (device + queue), pipelines via
// `SimPipelineCache`, GPU-resident skinned mesh via `registerGPUMesh` (no CPU
// readback), `PBDCollider` boxes reused verbatim for the tube walls + floor.
//
// All N bodies share ONE flat particle buffer; body in slot s occupies the slice
// [s·P, (s+1)·P). Slots are a fixed pool with a free-list; an inactive slot is
// pinned + parked far below the void and simply not drawn. The rest pose
// (restLocal / clusters / skin weights) is template data shared by every slot.
//
// The scene manager (HotdogPressTeddies) owns the spawn / stuffer-drive / cull /
// glass-clamp logic on the shared particle buffer between `step()`s — exactly the
// HotdogPressFranksPBD ↔ PBDTubeBatch split.

// ── Shared Swift↔Metal structs (alignment rule: float4 / SIMD4 only) ─────────

public struct SMParticle {
    public var pos:  SIMD4<Float>   // xyz world, w = invMass (0 = pinned)
    public var prev: SIMD4<Float>   // xyz prev world, w = bodyID (float)
    public init(pos: SIMD4<Float>, prev: SIMD4<Float>) { self.pos = pos; self.prev = prev }
}

struct SMClusterState {
    var centroid: SIMD4<Float>
    var quat:     SIMD4<Float>
}

struct SMBodyParams {
    // x = beta (shape-match stiffness), y = restScale, z = gravityScale, w = unused
    var p: SIMD4<Float>
}

struct SMUniforms {
    var particleCount:    UInt32
    var particlesPerBody: UInt32
    var clusterCount:     UInt32
    var bodyCount:        UInt32
    var dt:               Float
    var damping:          Float
    var gravity:          Float
    var floorY:           Float
    var colliderCount:    UInt32
    var collideStiffness: Float
    var collideFriction:  Float
    var _pad0:            Float = 0
}

struct SMHashUniforms {
    var particleCount:    UInt32
    var tableSize:        UInt32
    var cellSize:         Float
    var radius:           Float
    var particlesPerBody: UInt32
    var _pad0: UInt32 = 0
    var _pad1: UInt32 = 0
    var _pad2: UInt32 = 0
}

struct SMSkinUniforms {
    var vertexCount:  UInt32
    var clusterCount: UInt32
    var bodyBase:     UInt32
    var K:            UInt32
}

// ── Template ─────────────────────────────────────────────────────────────────

/// The rest definition of one soft body. Built once (e.g. by `TeddyBearGeometry`)
/// and shared by every instance. All positions are in body-local space, centred
/// on the body's rest centroid.
public struct SoftBodyTemplate {
    public var restLocal:        [SIMD3<Float>]   // P particle rest positions
    public var clusterMembers:   [[Int]]          // C clusters → local particle indices
    // Skinned surface mesh.
    public var vertexRest:       [SIMD3<Float>]   // V
    public var vertexRestNormal: [SIMD3<Float>]   // V
    public var indices:          [UInt32]
    public var uvs:              [SIMD2<Float>]   // V
    public var skinClusters:     [[Int]]          // V → K cluster indices
    public var skinWeights:      [[Float]]        // V → K weights
    public var flagAlpha:        Float            // vertex-color alpha (plush gbuffer flag)
    public var K:                Int

    public init(restLocal: [SIMD3<Float>], clusterMembers: [[Int]],
                vertexRest: [SIMD3<Float>], vertexRestNormal: [SIMD3<Float>],
                indices: [UInt32], uvs: [SIMD2<Float>],
                skinClusters: [[Int]], skinWeights: [[Float]],
                flagAlpha: Float, K: Int) {
        self.restLocal = restLocal; self.clusterMembers = clusterMembers
        self.vertexRest = vertexRest; self.vertexRestNormal = vertexRestNormal
        self.indices = indices; self.uvs = uvs
        self.skinClusters = skinClusters; self.skinWeights = skinWeights
        self.flagAlpha = flagAlpha; self.K = K
    }

    public var particlesPerBody: Int { restLocal.count }
    public var clusterCount: Int { clusterMembers.count }
    public var vertexCount: Int { vertexRest.count }
}

@MainActor
public final class ShapeMatchSoftBody {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "ShapeMatchSoftBody")

    public let engine: SimEngine
    public var device: MTLDevice { engine.device }
    public var commandQueue: MTLCommandQueue { engine.commandQueue }

    private let template: SoftBodyTemplate
    public let maxBodies: Int
    public let P: Int          // particles per body
    public let C: Int          // clusters per body
    public let V: Int          // vertices per body

    // ── Pipelines ──
    private let integratePipeline:  MTLComputePipelineState
    private let clusterFitPipeline: MTLComputePipelineState
    private let applyGoalsPipeline: MTLComputePipelineState
    private let collidePipeline:    MTLComputePipelineState
    private let hashClearPipeline:  MTLComputePipelineState
    private let hashCountPipeline:  MTLComputePipelineState
    private let hashScanPipeline:   MTLComputePipelineState
    private let hashScatterPipeline:MTLComputePipelineState
    private let selfCollidePipeline:MTLComputePipelineState
    private let skinWritePipeline:  MTLComputePipelineState

    // ── Flat per-body dynamic state ──
    public let particleBuffer: SimBuffer<SMParticle>     // maxBodies · P
    private let bodyParamsBuffer: SimBuffer<SMBodyParams> // maxBodies
    private let clusterStateBuffer: MTLBuffer            // maxBodies · C · SMClusterState
    private let colliderBuffer: SimBuffer<PBDCollider>
    private var colliderCount = 0
    private let uniformBuffer: MTLBuffer

    // ── Template (shared) buffers ──
    private let restLocalBuffer:   MTLBuffer   // packed_float3 [P]
    private let clusterRestBuffer: MTLBuffer   // packed_float3 [C]
    private let memberStartBuffer: MTLBuffer   // uint [C]
    private let memberCountBuffer: MTLBuffer   // uint [C]
    private let membersBuffer:     MTLBuffer   // uint flat
    private let pcStartBuffer:     MTLBuffer   // uint [P]
    private let pcCountBuffer:     MTLBuffer   // uint [P]
    private let pcListBuffer:      MTLBuffer   // uint flat
    private let vRestBuffer:       MTLBuffer   // packed_float3 [V]
    private let vRestNormBuffer:   MTLBuffer   // packed_float3 [V]
    private let skinIdxBuffer:     MTLBuffer   // uint [V*K]
    private let skinWBuffer:       MTLBuffer   // float [V*K]
    private let skinUniformBuffer: MTLBuffer

    // ── Self-collision hash ──
    private static let hashTableSize = 1 << 14
    private let cellCountsBuffer:  MTLBuffer
    private let cellOffsetsBuffer: MTLBuffer
    private let sortedBuffer:      MTLBuffer
    private let hashUniformBuffer: MTLBuffer

    // ── Per-slot render output + mesh ──
    private(set) public var positionBuffers: [MTLBuffer] = []  // [maxBodies] packed_float3 [V]
    private(set) public var normalBuffers:   [MTLBuffer] = []
    private var colorBuffer: MTLBuffer!                         // shared [V] float4, alpha = flag
    private var indexBuffer: MTLBuffer!
    private var uvBuffer: MTLBuffer!
    private var meshHandles: [IlluminatoramaMeshHandle?] = []
    private weak var renderer: IlluminatoramaRenderer?

    // ── Slot pool ──
    private var slotAlive: [Bool]
    public private(set) var bodyCount = 0   // highest alive slot + 1 (kernels dispatch over this · P)

    // ── Tunables ──
    public var gravity: Float = 9.8
    public var damping: Float = 0.985
    public var floorY: Float = -50
    public var collideStiffness: Float = 1.0
    public var collideFriction: Float = 0.30
    public var shapeMatchIterations = 4
    public var selfRadius: Float = 0.05
    public var selfCellSize: Float = 0.12
    public var selfCollisionEnabled = true
    private let fixedDt: Float = 1.0 / 120.0
    private var accumulator: Float = 0

    // ── Init ──

    public init?(engine: SimEngine, template: SoftBodyTemplate,
                 maxBodies: Int, renderer: IlluminatoramaRenderer) {
        guard template.particlesPerBody > 0, template.clusterCount > 0,
              template.vertexCount > 0, maxBodies >= 1 else { return nil }
        let cache = engine.pipelineCache
        let dev = engine.device
        guard
            let integ   = cache.pipelineState(name: "sm_integrate",   device: dev),
            let fit     = cache.pipelineState(name: "sm_cluster_fit", device: dev),
            let goals   = cache.pipelineState(name: "sm_apply_goals", device: dev),
            let coll    = cache.pipelineState(name: "sm_collide",     device: dev),
            let hClear  = cache.pipelineState(name: "sm_hashClear",   device: dev),
            let hCount  = cache.pipelineState(name: "sm_hashCount",   device: dev),
            let hScan   = cache.pipelineState(name: "sm_hashScan",    device: dev),
            let hScatter = cache.pipelineState(name: "sm_hashScatter", device: dev),
            let selfC   = cache.pipelineState(name: "sm_selfCollide", device: dev),
            let skin    = cache.pipelineState(name: "sm_skin_write",  device: dev)
        else {
            Self.log.error("ShapeMatchSoftBody pipeline cache failed — check ShapeMatchSoftBody.metal ships in Shaders/")
            return nil
        }
        self.engine = engine
        self.template = template
        self.maxBodies = maxBodies
        self.renderer = renderer
        self.P = template.particlesPerBody
        self.C = template.clusterCount
        self.V = template.vertexCount
        self.integratePipeline = integ
        self.clusterFitPipeline = fit
        self.applyGoalsPipeline = goals
        self.collidePipeline = coll
        self.hashClearPipeline = hClear
        self.hashCountPipeline = hCount
        self.hashScanPipeline = hScan
        self.hashScatterPipeline = hScatter
        self.selfCollidePipeline = selfC
        self.skinWritePipeline = skin

        let maxParticles = maxBodies * P
        guard
            let pBuf = SimBuffer<SMParticle>(device: dev, capacity: maxParticles, label: "SoftBody.particles"),
            let bpBuf = SimBuffer<SMBodyParams>(device: dev, capacity: maxBodies, label: "SoftBody.bodyParams"),
            let csBuf = dev.makeBuffer(length: maxBodies * C * MemoryLayout<SMClusterState>.stride, options: .storageModeShared),
            let colBuf = SimBuffer<PBDCollider>(device: dev, capacity: 8, label: "SoftBody.colliders"),
            let uBuf = dev.makeBuffer(length: MemoryLayout<SMUniforms>.stride, options: .storageModeShared),
            let ccBuf = dev.makeBuffer(length: Self.hashTableSize * 4, options: .storageModePrivate),
            let coBuf = dev.makeBuffer(length: Self.hashTableSize * 4, options: .storageModePrivate),
            let srBuf = dev.makeBuffer(length: max(1, maxParticles) * 4, options: .storageModePrivate),
            let huBuf = dev.makeBuffer(length: MemoryLayout<SMHashUniforms>.stride, options: .storageModeShared),
            let skuBuf = dev.makeBuffer(length: MemoryLayout<SMSkinUniforms>.stride, options: .storageModeShared)
        else {
            Self.log.error("ShapeMatchSoftBody buffer allocation failed")
            return nil
        }
        self.particleBuffer = pBuf
        self.bodyParamsBuffer = bpBuf
        self.clusterStateBuffer = csBuf
        self.colliderBuffer = colBuf
        self.uniformBuffer = uBuf
        self.cellCountsBuffer = ccBuf
        self.cellOffsetsBuffer = coBuf
        self.sortedBuffer = srBuf
        self.hashUniformBuffer = huBuf
        self.skinUniformBuffer = skuBuf

        // ── Template buffers ──
        func packed3(_ a: [SIMD3<Float>]) -> MTLBuffer {
            var flat = [Float](); flat.reserveCapacity(a.count * 3)
            for v in a { flat.append(v.x); flat.append(v.y); flat.append(v.z) }
            return flat.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        func uints(_ a: [UInt32]) -> MTLBuffer {
            a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: max(4, $0.count), options: .storageModeShared)! }
        }
        func floats(_ a: [Float]) -> MTLBuffer {
            a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: max(4, $0.count), options: .storageModeShared)! }
        }

        // Cluster rest centroids + flat membership.
        var clusterRest: [SIMD3<Float>] = []
        var memberStart: [UInt32] = [], memberCount: [UInt32] = [], membersFlat: [UInt32] = []
        for cluster in template.clusterMembers {
            var cen = SIMD3<Float>(repeating: 0)
            for li in cluster { cen += template.restLocal[li] }
            cen /= Float(max(1, cluster.count))
            clusterRest.append(cen)
            memberStart.append(UInt32(membersFlat.count))
            memberCount.append(UInt32(cluster.count))
            membersFlat.append(contentsOf: cluster.map { UInt32($0) })
        }
        // Inverse: per-particle cluster list.
        var perParticle = [[Int]](repeating: [], count: P)
        for (ci, cluster) in template.clusterMembers.enumerated() {
            for li in cluster { perParticle[li].append(ci) }
        }
        var pcStart: [UInt32] = [], pcCount: [UInt32] = [], pcFlat: [UInt32] = []
        for list in perParticle {
            pcStart.append(UInt32(pcFlat.count))
            pcCount.append(UInt32(list.count))
            pcFlat.append(contentsOf: list.map { UInt32($0) })
        }
        // Skin flat (V*K).
        var skinIdx: [UInt32] = [], skinW: [Float] = []
        for v in 0..<V {
            for k in 0..<template.K {
                skinIdx.append(UInt32(template.skinClusters[v][k]))
                skinW.append(template.skinWeights[v][k])
            }
        }

        self.restLocalBuffer   = packed3(template.restLocal)
        self.clusterRestBuffer = packed3(clusterRest)
        self.memberStartBuffer = uints(memberStart)
        self.memberCountBuffer = uints(memberCount)
        self.membersBuffer     = uints(membersFlat)
        self.pcStartBuffer     = uints(pcStart)
        self.pcCountBuffer     = uints(pcCount)
        self.pcListBuffer      = uints(pcFlat)
        self.vRestBuffer       = packed3(template.vertexRest)
        self.vRestNormBuffer   = packed3(template.vertexRestNormal)
        self.skinIdxBuffer     = uints(skinIdx)
        self.skinWBuffer       = floats(skinW)

        self.slotAlive = [Bool](repeating: false, count: maxBodies)

        // ── Render buffers + mesh registration ──
        buildSharedMeshBuffers()
        for s in 0..<maxBodies {
            let pb = dev.makeBuffer(length: V * 12, options: .storageModeShared)!
            let nb = dev.makeBuffer(length: V * 12, options: .storageModeShared)!
            pb.label = "SoftBody.pos[\(s)]"; nb.label = "SoftBody.norm[\(s)]"
            positionBuffers.append(pb); normalBuffers.append(nb)
            // Park the slot's mesh at the rest pose (off-screen) so frame 0 is valid.
            seedRestMesh(slot: s)
            let desc = IlluminatoramaGPUMeshDescriptor(
                positionBuffer: pb, normalBuffer: nb,
                vertexCount: V,
                bodyIndexBuffer: indexBuffer, bodyIndexCount: template.indices.count,
                bodyIndexType: .uint32,
                uvBuffer: uvBuffer, colorBuffer: colorBuffer, doubleSided: false)
            meshHandles.append(renderer.registerGPUMesh(desc))
        }

        // Clear cluster state to identity quats so APD warm-starts cleanly.
        let cs = clusterStateBuffer.contents().bindMemory(to: SMClusterState.self, capacity: maxBodies * C)
        for i in 0..<(maxBodies * C) {
            cs[i] = SMClusterState(centroid: .zero, quat: SIMD4(0, 0, 0, 1))
        }
    }

    private func buildSharedMeshBuffers() {
        let dev = device
        indexBuffer = template.indices.withUnsafeBytes {
            dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        indexBuffer.label = "SoftBody.indices"
        var uvFlat = [Float](); uvFlat.reserveCapacity(V * 2)
        for uv in template.uvs { uvFlat.append(uv.x); uvFlat.append(uv.y) }
        uvBuffer = uvFlat.withUnsafeBytes {
            dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        uvBuffer.label = "SoftBody.uv"
        // Per-vertex colour: rgb = 1 (albedo untouched; the instance albedo drives
        // colour), alpha = the plush gbuffer flag so the shader switches to the
        // plush BRDF + SSS for these vertices only.
        var colFlat = [Float](); colFlat.reserveCapacity(V * 4)
        for _ in 0..<V { colFlat.append(1); colFlat.append(1); colFlat.append(1); colFlat.append(template.flagAlpha) }
        colorBuffer = colFlat.withUnsafeBytes {
            dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        colorBuffer.label = "SoftBody.color"
    }

    /// CPU-seed slot `s`'s render buffers from the rest pose so the registered
    /// mesh renders before the first GPU skin pass (parked off-screen until spawn).
    private func seedRestMesh(slot s: Int) {
        let pp = positionBuffers[s].contents().bindMemory(to: Float.self, capacity: V * 3)
        let np = normalBuffers[s].contents().bindMemory(to: Float.self, capacity: V * 3)
        for v in 0..<V {
            let p = template.vertexRest[v] + SIMD3<Float>(0, -1000, 0)   // parked
            let n = template.vertexRestNormal[v]
            pp[3*v] = p.x; pp[3*v+1] = p.y; pp[3*v+2] = p.z
            np[3*v] = n.x; np[3*v+1] = n.y; np[3*v+2] = n.z
        }
    }

    // ── Colliders ──

    public func setColliders(_ colliders: [PBDCollider]) {
        let clipped = colliders.count > colliderBuffer.capacity
            ? Array(colliders.prefix(colliderBuffer.capacity)) : colliders
        colliderBuffer.write(clipped)
        colliderCount = clipped.count
    }

    // ── Spawn / despawn ──

    /// Claim a free slot, place the template rest particles into world via `xform`
    /// (rotation+translation), tag bodyID, and set initial body params. Returns the
    /// slot index, or nil if the pool is full.
    @discardableResult
    public func spawn(transform xform: simd_float4x4,
                      beta: Float, restScale: Float, gravityScale: Float) -> Int? {
        guard let slot = (0..<maxBodies).first(where: { !slotAlive[$0] }) else { return nil }
        slotAlive[slot] = true
        let base = slot * P
        let ptr = particleBuffer.contents
        // bodyID = slot (unique per live body; self-collision skips same-body pairs).
        for i in 0..<P {
            let rl = template.restLocal[i]
            let w4 = xform * SIMD4<Float>(rl, 1)
            let wp = SIMD3<Float>(w4.x, w4.y, w4.z)
            ptr[base + i] = SMParticle(pos: SIMD4(wp, 1), prev: SIMD4(wp, Float(slot)))
        }
        bodyParamsBuffer.contents[slot] = SMBodyParams(p: SIMD4(beta, restScale, gravityScale, 0))
        // Reset this slot's cluster warm-start quats to identity.
        let cs = clusterStateBuffer.contents().bindMemory(to: SMClusterState.self, capacity: maxBodies * C)
        for c in 0..<C { cs[slot * C + c] = SMClusterState(centroid: SIMD4(0, 0, 0, 0), quat: SIMD4(0, 0, 0, 1)) }
        bodyCount = max(bodyCount, slot + 1)
        return slot
    }

    public func despawn(slot: Int) {
        guard slot >= 0, slot < maxBodies, slotAlive[slot] else { return }
        slotAlive[slot] = false
        // Park + pin so it stays inert in the over-dispatched kernels and off-screen.
        let base = slot * P
        let ptr = particleBuffer.contents
        for i in 0..<P {
            let p = SIMD3<Float>(0, -1000, 0)
            ptr[base + i] = SMParticle(pos: SIMD4(p, 0), prev: SIMD4(p, 0))
        }
        seedRestMesh(slot: slot)
        // Shrink bodyCount if trailing slots are now dead.
        while bodyCount > 0 && !slotAlive[bodyCount - 1] { bodyCount -= 1 }
    }

    public func isAlive(_ slot: Int) -> Bool { slot >= 0 && slot < maxBodies && slotAlive[slot] }
    public var aliveSlots: [Int] { (0..<maxBodies).filter { slotAlive[$0] } }

    /// Per-frame body params (the controller drives the compress→inflate ease).
    public func setBodyParams(slot: Int, beta: Float, restScale: Float, gravityScale: Float) {
        guard slot >= 0, slot < maxBodies else { return }
        bodyParamsBuffer.contents[slot] = SMBodyParams(p: SIMD4(beta, restScale, gravityScale, 0))
    }

    /// World-space centroid of a body's particles (CPU read; valid after the last
    /// committed step). Used by the manager for cull / state classification.
    public func centroid(slot: Int) -> SIMD3<Float> {
        let base = slot * P
        let ptr = particleBuffer.contents
        var c = SIMD3<Float>(repeating: 0)
        for i in 0..<P { let p = ptr[base + i].pos; c += SIMD3(p.x, p.y, p.z) }
        return c / Float(P)
    }

    /// Mesh handle for a slot (for the instance list). nil if registration failed.
    public func meshHandle(slot: Int) -> IlluminatoramaMeshHandle? { meshHandles[slot] }

    // ── Step ──

    /// Advance all alive bodies and re-skin their meshes. Encodes into one command
    /// buffer on the shared queue and commits (non-blocking).
    public func step(dt: Float) {
        guard bodyCount > 0 else { return }
        accumulator += max(0, min(dt, 1.0 / 30.0))
        var steps = 0
        guard let cb = commandQueue.makeCommandBuffer() else { return }
        cb.label = "SoftBody.step"
        while accumulator >= fixedDt && steps < 4 {
            encodeSubstep(cb, dt: fixedDt)
            // Inter-body contact PER SUBSTEP (not once/frame) so the next substep's
            // shape-match goals don't simply re-overlap packed neighbours — the
            // separation is re-enforced after every integrate+match+wall pass.
            if selfCollisionEnabled { encodeSelfCollide(cb) }
            accumulator -= fixedDt
            steps += 1
        }
        encodeSkin(cb)
        cb.commit()
    }

    /// Run `count` settle substeps synchronously (init pre-settle, no live frame).
    public func presettle(steps count: Int) {
        guard bodyCount > 0, let cb = commandQueue.makeCommandBuffer() else { return }
        for _ in 0..<count {
            encodeSubstep(cb, dt: fixedDt)
            if selfCollisionEnabled { encodeSelfCollide(cb) }
        }
        encodeSkin(cb)
        cb.commit()
        cb.waitUntilCompleted()   // gpu-ok: one-shot init settle, not per-frame
    }

    private func writeUniforms(dt: Float) {
        let u = uniformBuffer.contents().bindMemory(to: SMUniforms.self, capacity: 1)
        u.pointee = SMUniforms(
            particleCount: UInt32(bodyCount * P),
            particlesPerBody: UInt32(P),
            clusterCount: UInt32(C),
            bodyCount: UInt32(bodyCount),
            dt: dt, damping: damping, gravity: gravity, floorY: floorY,
            colliderCount: UInt32(colliderCount),
            collideStiffness: collideStiffness, collideFriction: collideFriction)
    }

    private func encodeSubstep(_ cb: MTLCommandBuffer, dt: Float) {
        writeUniforms(dt: dt)
        let nP = bodyCount * P
        let nC = bodyCount * C

        encode(cb, integratePipeline, count: nP, label: "SoftBody.integrate") { e in
            e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 0)
            e.setBuffer(self.bodyParamsBuffer.buffer, offset: 0, index: 1)
            e.setBuffer(self.uniformBuffer, offset: 0, index: 2)
        }
        for _ in 0..<shapeMatchIterations {
            encode(cb, clusterFitPipeline, count: nC, label: "SoftBody.fit") { e in
                e.setBuffer(self.clusterStateBuffer, offset: 0, index: 0)
                e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 1)
                e.setBuffer(self.memberStartBuffer, offset: 0, index: 2)
                e.setBuffer(self.memberCountBuffer, offset: 0, index: 3)
                e.setBuffer(self.membersBuffer, offset: 0, index: 4)
                e.setBuffer(self.restLocalBuffer, offset: 0, index: 5)
                e.setBuffer(self.clusterRestBuffer, offset: 0, index: 6)
                e.setBuffer(self.uniformBuffer, offset: 0, index: 7)
            }
            encode(cb, applyGoalsPipeline, count: nP, label: "SoftBody.goals") { e in
                e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 0)
                e.setBuffer(self.clusterStateBuffer, offset: 0, index: 1)
                e.setBuffer(self.bodyParamsBuffer.buffer, offset: 0, index: 2)
                e.setBuffer(self.pcStartBuffer, offset: 0, index: 3)
                e.setBuffer(self.pcCountBuffer, offset: 0, index: 4)
                e.setBuffer(self.pcListBuffer, offset: 0, index: 5)
                e.setBuffer(self.restLocalBuffer, offset: 0, index: 6)
                e.setBuffer(self.clusterRestBuffer, offset: 0, index: 7)
                e.setBuffer(self.uniformBuffer, offset: 0, index: 8)
            }
        }
        if colliderCount > 0 {
            encode(cb, collidePipeline, count: nP, label: "SoftBody.collide") { e in
                e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 0)
                e.setBuffer(self.colliderBuffer.buffer, offset: 0, index: 1)
                e.setBuffer(self.uniformBuffer, offset: 0, index: 2)
            }
        }
    }

    private func encodeSelfCollide(_ cb: MTLCommandBuffer) {
        let n = bodyCount * P
        guard n > 0 else { return }
        let hu = hashUniformBuffer.contents().bindMemory(to: SMHashUniforms.self, capacity: 1)
        hu.pointee = SMHashUniforms(particleCount: UInt32(n), tableSize: UInt32(Self.hashTableSize),
                                    cellSize: selfCellSize, radius: selfRadius,
                                    particlesPerBody: UInt32(P))
        var T = UInt32(Self.hashTableSize)
        encode(cb, hashClearPipeline, count: Self.hashTableSize, label: "SoftBody.hashClear") { e in
            e.setBuffer(self.cellCountsBuffer, offset: 0, index: 0)
            e.setBytes(&T, length: 4, index: 1)
        }
        encode(cb, hashCountPipeline, count: n, label: "SoftBody.hashCount") { e in
            e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 0)
            e.setBuffer(self.cellCountsBuffer, offset: 0, index: 1)
            e.setBuffer(self.hashUniformBuffer, offset: 0, index: 2)
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "SoftBody.hashScan"; e.setComputePipelineState(hashScanPipeline)
            e.setBuffer(cellCountsBuffer, offset: 0, index: 0)
            e.setBuffer(cellOffsetsBuffer, offset: 0, index: 1)
            e.setBytes(&T, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            e.endEncoding()
        }
        encode(cb, hashScatterPipeline, count: n, label: "SoftBody.hashScatter") { e in
            e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 0)
            e.setBuffer(self.cellCountsBuffer, offset: 0, index: 1)
            e.setBuffer(self.cellOffsetsBuffer, offset: 0, index: 2)
            e.setBuffer(self.sortedBuffer, offset: 0, index: 3)
            e.setBuffer(self.hashUniformBuffer, offset: 0, index: 4)
        }
        for _ in 0..<6 {
            encode(cb, selfCollidePipeline, count: n, label: "SoftBody.selfCollide") { e in
                e.setBuffer(self.particleBuffer.buffer, offset: 0, index: 0)
                e.setBuffer(self.cellOffsetsBuffer, offset: 0, index: 1)
                e.setBuffer(self.cellCountsBuffer, offset: 0, index: 2)
                e.setBuffer(self.sortedBuffer, offset: 0, index: 3)
                e.setBuffer(self.hashUniformBuffer, offset: 0, index: 4)
            }
        }
    }

    private func encodeSkin(_ cb: MTLCommandBuffer) {
        for s in 0..<bodyCount where slotAlive[s] {
            let su = skinUniformBuffer.contents().bindMemory(to: SMSkinUniforms.self, capacity: 1)
            su.pointee = SMSkinUniforms(vertexCount: UInt32(V), clusterCount: UInt32(C),
                                        bodyBase: UInt32(s * C), K: UInt32(template.K))
            guard let e = cb.makeComputeCommandEncoder() else { continue }
            e.label = "SoftBody.skin[\(s)]"
            e.setComputePipelineState(skinWritePipeline)
            e.setBuffer(clusterStateBuffer, offset: 0, index: 0)
            e.setBuffer(vRestBuffer, offset: 0, index: 1)
            e.setBuffer(vRestNormBuffer, offset: 0, index: 2)
            e.setBuffer(skinIdxBuffer, offset: 0, index: 3)
            e.setBuffer(skinWBuffer, offset: 0, index: 4)
            e.setBuffer(clusterRestBuffer, offset: 0, index: 5)
            e.setBuffer(positionBuffers[s], offset: 0, index: 6)
            e.setBuffer(normalBuffers[s], offset: 0, index: 7)
            // Skin uniforms are per-slot; bind a fresh copy via setBytes to avoid
            // racing the single shared buffer across the per-slot dispatches.
            var local = su.pointee
            e.setBytes(&local, length: MemoryLayout<SMSkinUniforms>.stride, index: 8)
            dispatch1D(e, pipeline: skinWritePipeline, count: V)
            e.endEncoding()
        }
    }

    private func encode(_ cb: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                        count: Int, label: String, _ bind: (MTLComputeCommandEncoder) -> Void) {
        guard count > 0, let e = cb.makeComputeCommandEncoder() else { return }
        e.label = label
        e.setComputePipelineState(pipeline)
        bind(e)
        dispatch1D(e, pipeline: pipeline, count: count)
        e.endEncoding()
    }

    private func dispatch1D(_ e: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
    }
}
