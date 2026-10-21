import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore


// MARK: - ParticleState

/// Particles, streaks, and GPU mesh repack / normal-synthesis pipelines.
/// Owned by the renderer as `fxs`; every member keeps the name it had on the renderer.
@MainActor
final class ParticleState {
    // Phase 4.11 — particle pipelines. The compute step integrates
    // positions/velocities/life; the render pipeline draws survivors as
    // additive HDR point sprites into the composite.
    var particleStepPipeline: MTLComputePipelineState!
    var particleDrawPipeline: MTLRenderPipelineState!
    /// Phase 4.23 — host-buffer point-sprite renderer. Stride-aware so
    /// FireworksUltra's `SIMD4<Float>` star buffers + `packed_float3`
    /// burst buffers both bind without a copy.
    var extParticleDrawPipeline: MTLRenderPipelineState!
    // Phase 4.13a — repack kernel for the DynamicMesh bridge.
    var repackPosNormPipeline: MTLComputePipelineState!

    // Phase 4.21 — one-shot GPU mesh normal/tangent synthesis (replaces the
    // CPU passes that ran on the main thread inside IlluminatoramaMesh.from).
    var meshBuildAdjacencyPipeline: MTLComputePipelineState!
    var meshSynthPipeline: MTLComputePipelineState!

    /// Depth-stencil that READS the G-buffer depth without writing.
    /// Opaque geometry occludes particles behind it; particles don't
    /// occlude each other (additive HDR needs every drawn fragment to
    /// add to the composite).
    var particleDepthState: MTLDepthStencilState!

    /// Active particle emitters. Hosts register via `addEmitter` and the
    /// renderer ticks + draws each enabled one per frame.
    var particleEmitters: [IlluminatoramaParticleEmitter] = []
    /// Particle fields published through `SimEngine.particleFields` for the
    /// active scene, refreshed each frame by the extractor via
    /// `setParticleFields`. Each is drawn as additive HDR point sprites into
    /// the same composite the lighting pass writes; bloom + tonemap process
    /// them naturally. (Replaces the Phase 4.23 SCNGeometry-associated-object
    /// bridge with the shared SimEngine registry — see `ParticleFieldSource`.)
    var particleFields: [ParticleFieldSource] = []
    /// Velocity-aligned streak sources refreshed each frame by the host.
    /// Rendered after point-sprite fields in `encodePostResolveFX` so both
    /// participate in the same HDR bloom + tonemap pass.
    var particleStreaks: [IlluminatoramaRenderer.StreakSource] = []
    /// Pipeline for velocity-aligned billboard streak quads (`particleStreakVS/FS`).
    /// `nil` when the shaders are absent from the library — streak draws are
    /// silently skipped in that case (logged once at init).
    var streakPipeline: MTLRenderPipelineState?

    /// Wall-clock of the previous `render()` call, used to derive a real
    /// `dt` for particle integration even when the host's frame rate
    /// varies.
    var lastParticleTickTime: CFTimeInterval = CACurrentMediaTime()

    var gpuRepackTasks: [IlluminatoramaRenderer.GPURepackTask] = []
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    /// Matches the Metal `ExtParticleParams` struct byte-for-byte.
    /// Set via `setVertexBytes` per emitter draw.
    struct ExternalPointParams {
        var positionStrideFloats: UInt32
        var colorStrideFloats: UInt32
        var positionOffsetFloats: UInt32
        var colorOffsetFloats: UInt32
        var pointSize: Float
        var colorScale: Float
    }

    // ── Phase 4.21 — one-shot GPU mesh normal/tangent synthesis ────────
    //
    // Replaces the CPU `synthesiseNormals` / `synthesiseTangents` passes that
    // used to run on the MAIN THREAD inside `IlluminatoramaMesh.from`. On a
    // cold-cache scene the extractor converts every geometry in one synchronous
    // `extractFrame` tick; the CPU tangent scatter-add parked the run loop for
    // seconds (observed ~9 s hang in `accumulateTangents`). This runs the same
    // math on the GPU instead.
    //
    // Builds a bounded per-vertex triangle-adjacency list with INTEGER atomics
    // (the counting-sort idiom; the project avoids float-atomic scatter — see
    // MLSMPM.metal), then gathers per vertex. Three GPU steps on one command
    // buffer: blit-clear the count buffer, `illumi_mesh_build_adjacency` (per
    // triangle), `illumi_mesh_synth` (per vertex). The adjacency and synth
    // dispatches live in SEPARATE compute encoders so the second observes the
    // first's writes; the blit precedes both.
    //
    // The command buffer is committed but NOT waited on: it shares the
    // renderer's single command queue with `render()`, whose command buffer is
    // committed later the same frame, so FIFO queue ordering guarantees the
    // synthesis completes before the G-buffer pass reads the vertex buffer — no
    // main-thread stall, no fence. Metal keeps the bound buffers (vertex/index +
    // the transient scratch) alive until the command buffer completes.
    static let meshSynthMaxValence: UInt32 = 24

    public func synthesiseMeshGeometry(_ mesh: IlluminatoramaMesh) {
        guard mesh.pendingNormalSynth || mesh.pendingTangentSynth else { return }
        let vertCount = mesh.vertexCount
        let triCount = mesh.indexCount / 3
        guard vertCount > 0, triCount > 0 else {
            mesh.pendingNormalSynth = false
            mesh.pendingTangentSynth = false
            return
        }
        let maxValence = Self.meshSynthMaxValence

        // Transient scratch: per-vertex atomic counter + bounded adjacency list.
        // Freed when the command buffer completes (Metal retains them until then).
        let countBytes = MemoryLayout<UInt32>.stride * vertCount
        let listBytes = MemoryLayout<UInt32>.stride * vertCount * Int(maxValence)
        guard let vertTriCount = device.makeBuffer(length: countBytes,
                                                   options: .storageModePrivate),
              let vertTriList = device.makeBuffer(length: listBytes,
                                                  options: .storageModePrivate),
              let cb = commandQueue.makeCommandBuffer() else {
            // Couldn't allocate / encode — leave the placeholder normals and
            // zero tangents rather than crashing; clear the flags so we don't
            // retry every frame.
            Self.log.warning("synthesiseMeshGeometry: scratch alloc / cb failed; mesh ships unsynthesised")
            mesh.pendingNormalSynth = false
            mesh.pendingTangentSynth = false
            return
        }
        vertTriCount.label = "Illuminatorama.meshSynth.count"
        vertTriList.label = "Illuminatorama.meshSynth.list"
        cb.label = "Illuminatorama.meshSynth"

        var isU32: UInt32 = (mesh.indexType == .uint32) ? 1 : 0
        var triCountU = UInt32(triCount)
        var vertCountU = UInt32(vertCount)
        var maxValenceU = maxValence
        var doNormals: UInt32 = mesh.pendingNormalSynth ? 1 : 0
        var doTangents: UInt32 = mesh.pendingTangentSynth ? 1 : 0

        // 1. Zero the per-vertex counter (private buffers are not zero-inited).
        if let blit = cb.makeBlitCommandEncoder() {
            blit.fill(buffer: vertTriCount, range: 0..<countBytes, value: 0)
            blit.endEncoding()
        }

        // 2. Build adjacency — one thread per triangle.
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Illuminatorama.meshSynth.adjacency"
            enc.setComputePipelineState(fxs.meshBuildAdjacencyPipeline)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 0)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 1)
            enc.setBytes(&isU32, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&triCountU, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&maxValenceU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBuffer(vertTriCount, offset: 0, index: 5)
            enc.setBuffer(vertTriList, offset: 0, index: 6)
            enc.setBytes(&vertCountU, length: MemoryLayout<UInt32>.stride, index: 7)
            dispatch1D(enc, pipeline: fxs.meshBuildAdjacencyPipeline, count: triCount)
            enc.endEncoding()
        }

        // 3. Synthesise normals/tangents — one thread per vertex. Separate
        //    encoder so it observes the adjacency writes from step 2.
        if let enc = cb.makeComputeCommandEncoder() {
            enc.label = "Illuminatorama.meshSynth.gather"
            enc.setComputePipelineState(fxs.meshSynthPipeline)
            enc.setBuffer(mesh.vertexBuffer, offset: 0, index: 0)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 1)
            enc.setBuffer(mesh.indexBuffer, offset: 0, index: 2)
            enc.setBytes(&isU32, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&vertCountU, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&maxValenceU, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBuffer(vertTriCount, offset: 0, index: 6)
            enc.setBuffer(vertTriList, offset: 0, index: 7)
            enc.setBytes(&doNormals, length: MemoryLayout<UInt32>.stride, index: 8)
            enc.setBytes(&doTangents, length: MemoryLayout<UInt32>.stride, index: 9)
            dispatch1D(enc, pipeline: fxs.meshSynthPipeline, count: vertCount)
            enc.endEncoding()
        }

        cb.commit()   // gpu-ok: committed, not waited — render()'s later cb on
                      // the same queue runs after this by FIFO ordering.
        mesh.pendingNormalSynth = false
        mesh.pendingTangentSynth = false
    }

    // ── Phase 4.13a — DynamicMesh bridge registration ────────────────

    /// A single per-frame GPU repack: take separate
    /// position + normal buffers (compute-fed by `DynamicMesh`) and
    /// interleave into an `IlluminatoramaVertex` buffer the G-buffer
    /// pass reads. Lives on the renderer for the lifetime of the
    /// corresponding `IlluminatoramaMeshHandle`.
    struct GPURepackTask {
        let kind: MeshKind
        let positionBuffer: MTLBuffer
        let normalBuffer: MTLBuffer
        let vertexBuffer: MTLBuffer       // the output Illuminatorama-format buffer
        let vertexCount: Int
        /// Phase 4.17 — optional UV buffer (packed_float2, stride 8).
        /// `nil` → kernel writes synthetic `(0,0)` UV (the prior
        /// behaviour, fine for non-textured surfaces). Non-nil →
        /// per-vertex UV read from this buffer, enabling textured
        /// compute-fed geometry like LeafField sprite cards.
        let uvBuffer: MTLBuffer?
        /// Optional per-vertex RGBA color stream (stride-16 float4) → albedo.
        let colorBuffer: MTLBuffer?
        /// Issue #65 — side copy of LAST frame's `positionBuffer` (packed_float3 ×
        /// vertexCount), blitted each frame AFTER the G-buffer draw. The G-buffer
        /// GPU-mesh draw binds this at vertex buffer(5) so the kUsePrevVerts
        /// `illumi_vs` variant can reconstruct per-vertex velocity (curr − prev)
        /// for TAA / motion blur. NOT `positionBuffer` ping-ponged — that buffer is
        /// consumed by the RT BLAS/TLAS refit + shadows + G-buffer, so a swap would
        /// desync the acceleration structure. A dedicated side buffer is isolated.
        let prevPositionBuffer: MTLBuffer
        /// False until the first frame primes `prevPositionBuffer = positionBuffer`
        /// (so frame 1 reads prev == curr → zero velocity, no spurious streak).
        var primed: Bool
        /// DIRECT mesh: a kernel writes `vertexBuffer` itself, so there is nothing to repack and no
        /// previous-position copy. The task still exists so every "GPU geometry is live" check holds.
        var direct: Bool = false
    }

    /// Register a compute-fed geometry described by an
    /// `IlluminatoramaGPUMeshDescriptor`. Allocates the Illuminatorama-
    /// format vertex buffer + combined index buffer once, schedules a
    /// per-frame repack on the renderer's queue, and returns a regular
    /// `IlluminatoramaMeshHandle`. Subsequent frames just dispatch the
    /// repack kernel — the host doesn't have to touch this again.
    @discardableResult
    public func registerGPUMesh(_ descriptor: IlluminatoramaGPUMeshDescriptor)
        -> IlluminatoramaMeshHandle?
    {
        let vertexBytes = MemoryLayout<IlluminatoramaVertex>.stride * descriptor.vertexCount
        let direct = descriptor.directVertexBuffer != nil
        guard let vertexBuffer = descriptor.directVertexBuffer
                ?? device.makeBuffer(length: vertexBytes, options: .storageModePrivate) else {
            return nil
        }
        vertexBuffer.label = "Illuminatorama.gpuMesh.vertices"

        // Build a combined index buffer when the descriptor has both
        // body + caps. Each element is a triangle list with shared
        // vertices — concatenation is correct as long as indices are
        // already absolute (they are, in DynamicMesh's cache). The
        // resulting mesh has one element that draws both halves in a
        // single call.
        let combined = buildCombinedIndexBuffer(descriptor: descriptor)
        guard let combined = combined else { return nil }

        let mesh = IlluminatoramaMesh(
            vertexBuffer: vertexBuffer,
            indexBuffer: combined.buffer,
            indexCount: combined.count,
            indexType: combined.type,
            label: "Illuminatorama.gpuMesh"
        )
        mesh.doubleSided = descriptor.doubleSided
        mesh.shadowCastsBothFaces = descriptor.shadowCastsBothFaces
        mesh.indirectDrawBuffer = descriptor.indirectDrawBuffer
        let kind = MeshKind.custom("gpuMesh#\(UUID().uuidString)")
        meshes[kind] = mesh
        // Issue #65 — side buffer for last frame's positions. Matches the
        // host position buffer's PACKED float3 layout (12-byte stride, NOT
        // SIMD3's 16) so the per-frame blit is a straight byte copy and the
        // `packed_float3*` shader read indexes correctly. See gpuMeshPrevStride.
        let prevPosBytes = direct ? 16 : Self.gpuMeshPrevStride * descriptor.vertexCount
        guard let prevPositionBuffer = device.makeBuffer(
                length: max(prevPosBytes, 16), options: .storageModePrivate) else {
            return nil
        }
        prevPositionBuffer.label = "Illuminatorama.gpuMesh.prevPositions"
        fxs.gpuRepackTasks.append(GPURepackTask(
            kind: kind,
            positionBuffer: descriptor.positionBuffer,
            normalBuffer: descriptor.normalBuffer,
            vertexBuffer: vertexBuffer,
            vertexCount: descriptor.vertexCount,
            uvBuffer: descriptor.uvBuffer,
            colorBuffer: descriptor.colorBuffer,
            prevPositionBuffer: prevPositionBuffer,
            primed: false,
            direct: direct
        ))
        return IlluminatoramaMeshHandle(kind: kind, mesh: mesh, renderer: self)
    }

    /// Concatenate body + (optional) caps index buffers into one buffer
    /// the IlluminatoramaMesh can hand to a single draw call. Both
    /// elements share the same vertex buffer, so absolute indices stay
    /// valid across the merge. Returns nil if buffer allocation fails.
    func buildCombinedIndexBuffer(descriptor: IlluminatoramaGPUMeshDescriptor)
        -> (buffer: MTLBuffer, count: Int, type: MTLIndexType)?
    {
        let stride = descriptor.bodyIndexType == .uint16 ? 2 : 4
        let bodyBytes = descriptor.bodyIndexCount * stride
        let capBytes  = descriptor.capIndexCount  * stride
        let totalBytes = bodyBytes + capBytes
        guard totalBytes > 0 else { return nil }
        guard let combined = device.makeBuffer(length: totalBytes,
                                                options: .storageModeShared) else {
            return nil
        }
        combined.label = "Illuminatorama.gpuMesh.indices"
        // Copy via blit so we can handle both .private and .shared source
        // buffers uniformly without `memcpy(...buffer.contents()...)`.
        // Reuse the renderer's shared SimEngine queue — no per-operation queue.
        guard let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: descriptor.bodyIndexBuffer, sourceOffset: 0,
                  to: combined, destinationOffset: 0, size: bodyBytes)
        if let capBuf = descriptor.capIndexBuffer, descriptor.capIndexCount > 0 {
            blit.copy(from: capBuf, sourceOffset: 0,
                      to: combined, destinationOffset: bodyBytes,
                      size: capBytes)
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted() // gpu-ok: one-shot setup, not per-frame
        let totalCount = descriptor.bodyIndexCount + descriptor.capIndexCount
        return (buffer: combined, count: totalCount, type: descriptor.bodyIndexType)
    }

    /// Per-frame repack pass — runs the `illumi_repack_pos_norm` kernel
    /// for every registered GPU mesh. Encodes one compute pass with all
    /// tasks dispatched back-to-back. Lives BEFORE the shadow / G-buffer
    /// passes so downstream reads see fresh interleaved vertices.
    func encodeGPURepacks(_ cb: MTLCommandBuffer) {
        guard fxs.gpuRepackTasks.contains(where: { !$0.direct }) else { return }
        guard let enc = timedComputeEncoder(cb, "gpuMesh.repack") else { return }
        enc.label = "Illuminatorama.gpuMesh.repack"
        enc.setComputePipelineState(fxs.repackPosNormPipeline)
        for task in fxs.gpuRepackTasks where !task.direct {
            var count = UInt32(task.vertexCount)
            enc.setBuffer(task.positionBuffer, offset: 0, index: 0)
            enc.setBuffer(task.normalBuffer,   offset: 0, index: 1)
            enc.setBuffer(task.vertexBuffer,   offset: 0, index: 2)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
            // Phase 4.17 — UV stream + flag. When the caller supplied a
            // UV buffer, bind it and set `hasUV = 1`; otherwise bind the
            // position buffer as a one-byte dummy (Metal won't fault on
            // an unread bound buffer) and set `hasUV = 0`. The kernel
            // gates on the flag and writes a synthetic `(0,0)` UV in
            // the fallback path.
            var hasUV: UInt32 = task.uvBuffer != nil ? 1 : 0
            enc.setBuffer(task.uvBuffer ?? task.positionBuffer,
                          offset: 0, index: 4)
            enc.setBytes(&hasUV, length: MemoryLayout<UInt32>.stride, index: 5)
            // Optional per-vertex color stream → albedo (coin DEBUG tints).
            var hasColor: UInt32 = task.colorBuffer != nil ? 1 : 0
            enc.setBuffer(task.colorBuffer ?? task.positionBuffer,
                          offset: 0, index: 6)
            enc.setBytes(&hasColor, length: MemoryLayout<UInt32>.stride, index: 7)
            let tg = min(fxs.repackPosNormPipeline.maxTotalThreadsPerThreadgroup,
                         fxs.repackPosNormPipeline.threadExecutionWidth * 8)
            let groups = MTLSize(width: (task.vertexCount + tg - 1) / tg,
                                  height: 1, depth: 1)
            let threadgroup = MTLSize(width: tg, height: 1, depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
        }
        enc.endEncoding()
    }

    /// Issue #65 — capture the current GPU-mesh positions into each task's
    /// `prevPositionBuffer` so the NEXT frame's G-buffer draw can reconstruct a
    /// per-vertex motion vector (curr − prev).
    ///
    /// Two phases, both pure blits (no ping-pong of `positionBuffer`, which the RT
    /// refit / shadows / G-buffer all consume):
    ///   • `prime: true`  — run BEFORE the G-buffer pass, only for not-yet-primed
    ///     meshes: seeds `prevPositionBuffer = positionBuffer` so frame 1 reads
    ///     prev == curr → zero velocity (no spurious first-frame streak).
    ///   • `prime: false` — run AFTER the G-buffer pass (which read last frame's
    ///     prev): copies THIS frame's positions forward for next frame. Metal's
    ///     automatic hazard tracking serializes this blit after the pass's reads.
    func encodeGPUMeshPrevCapture(_ cb: MTLCommandBuffer, prime: Bool) {
        let work = prime ? fxs.gpuRepackTasks.contains { !$0.primed && !$0.direct }
                         : fxs.gpuRepackTasks.contains { !$0.direct }
        guard work else { return }
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.label = prime ? "Illuminatorama.gpuMesh.mvPrime"
                           : "Illuminatorama.gpuMesh.mvCapture"
        for i in fxs.gpuRepackTasks.indices {
            if prime && fxs.gpuRepackTasks[i].primed { continue }
            if fxs.gpuRepackTasks[i].direct { continue }
            let task = fxs.gpuRepackTasks[i]
            let bytes = Self.gpuMeshPrevStride * task.vertexCount
            guard bytes > 0,
                  task.positionBuffer.length >= bytes,
                  task.prevPositionBuffer.length >= bytes else { continue }
            blit.copy(from: task.positionBuffer, sourceOffset: 0,
                      to: task.prevPositionBuffer, destinationOffset: 0,
                      size: bytes)
            if prime { fxs.gpuRepackTasks[i].primed = true }
        }
        blit.endEncoding()
    }

    // ── Phase 4.11 — particle emitter registration ───────────────────
    //
    // Scenes that want Metal-native particles (per the "always Metal"
    // rule) create an `IlluminatoramaParticleEmitter` and register it.
    // Per-frame: the renderer ticks every enabled emitter's compute step
    // and draws survivors as additive HDR point sprites into the
    // composite. The same MTLBuffer the host writes into with `emit(...)`
    // is what the GPU reads.

    public func addEmitter(_ emitter: IlluminatoramaParticleEmitter) {
        if !fxs.particleEmitters.contains(where: { $0 === emitter }) {
            fxs.particleEmitters.append(emitter)
        }
    }

    public func removeEmitter(_ emitter: IlluminatoramaParticleEmitter) {
        fxs.particleEmitters.removeAll { $0 === emitter }
    }

    // ── Particle fields (SimEngine registry) ──────────────────────────
    //
    // The extractor refreshes this each frame with the active scene's
    // `ParticleFieldSource`s (`SimEngine.particleFields.sources(forScene:)`).
    // Scenes that drive their own GPU particle simulation register a source
    // from their renderer-shim — keeps simulation ownership with the host
    // and renders through the overlay for free, with no SceneKit round-trip.
    public func setParticleFields(_ fields: [ParticleFieldSource]) {
        fxs.particleFields = fields
    }

    /// Additive HDR draw of every active-scene particle field (e.g. the
    /// Illuminatorama Room dust) into `hdrTexture`. Called only from
    /// `encodePostResolveFX` (Phase 4.43), AFTER the TAA resolve has seeded
    /// `hdrTexture` with the resolved opaque image — keeping these motion-
    /// vectorless moving points out of the temporal history so they can't
    /// trail. (Pre-4.43 this ran before the SSR gather so sparks reflected in
    /// nearby metals; that placement is what produced the dust trails.)
    func encodeExternalPointDraw(_ cb: MTLCommandBuffer) {
        guard !fxs.particleFields.isEmpty else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = gbs.hdrTexture
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = gbs.depthTexture
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.extParticles.draw"
        enc.setRenderPipelineState(fxs.extParticleDrawPipeline)
        enc.setDepthStencilState(fxs.particleDepthState)
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 2)
        for field in fxs.particleFields {
            enc.setVertexBuffer(field.positionBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(field.colorBuffer,    offset: 0, index: 1)
            // Matches the Metal-side `ExtParticleParams` struct: four
            // `uint` strides/offsets + two `float` scalars = 24-byte cluster.
            var params = ExternalPointParams(
                positionStrideFloats: field.positionStrideFloats,
                colorStrideFloats:    field.colorStrideFloats,
                positionOffsetFloats: field.positionOffsetFloats,
                colorOffsetFloats:    field.colorOffsetFloats,
                pointSize:            field.pointSize,
                colorScale:           field.colorScale
            )
            enc.setVertexBytes(&params,
                                length: MemoryLayout<ExternalPointParams>.stride,
                                index: 3)
            enc.drawPrimitives(type: .point,
                               vertexStart: 0,
                               vertexCount: field.vertexCount)
        }
        enc.endEncoding()
    }

    // ── Particle streak path (Phase 4.44b) ──────────────────────────────────
    //
    // Velocity-aligned billboard quad renderer for fireworks / spark streaks.
    // Hosts build FWStreakVertex buffers via their solver's `encodeBuildStreaks`
    // kernel and pass the pre-built Metal buffers here so the HDR additive draw
    // lands AFTER the TAA resolve — same timing as the point-sprite pass above,
    // so streak particles have no temporal trailing.
    //
    // API: call `setParticleStreaks([StreakSource])` from the host's tick();
    // the renderer calls `encodeStreakDraw(cb)` inside `encodePostResolveFX`.

    /// Host-supplied velocity-aligned streak source for `setParticleStreaks`.
    /// One source → one draw call per frame. Build the vertex / UV / index
    /// buffers via the solver's `encodeBuildStreaks` kernel (4 vertices + 6
    /// indices per particle) and pass them here already on the GPU.
    public struct StreakSource {
        /// `FWStreakVertex` buffer (stride 32): float4 position + float4 color,
        /// where `color.a` carries the per-particle stretch factor.
        public var positionColorBuffer: MTLBuffer
        /// Parallel `SIMD2<Float>` UV buffer (stride 8).
        public var uvBuffer: MTLBuffer
        /// UInt32 triangle index buffer — 6 indices per particle (2 triangles).
        public var indexBuffer: MTLBuffer
        /// Number of indices to draw. Typically `particleCapacity × 6`.
        public var indexCount: Int
        /// Soft Gaussian sparkle sprite (128×128, mipmapped). Sampled by the
        /// streak fragment shader for the sphere-imposter surface shading.
        public var spriteTexture: MTLTexture
        /// Per-draw brightness (`intensity`) and saturation (`saturationPow`)
        /// knobs forwarded to `ParticleStreakParams` in the shader.
        public var params: ParticleStreakParams

        public init(positionColorBuffer: MTLBuffer,
                    uvBuffer:            MTLBuffer,
                    indexBuffer:         MTLBuffer,
                    indexCount:          Int,
                    spriteTexture:       MTLTexture,
                    params:              ParticleStreakParams = ParticleStreakParams()) {
            self.positionColorBuffer = positionColorBuffer
            self.uvBuffer            = uvBuffer
            self.indexBuffer         = indexBuffer
            self.indexCount          = indexCount
            self.spriteTexture       = spriteTexture
            self.params              = params
        }
    }

    /// Replace the active streak-source list for this frame. Call from the
    /// host's tick loop after `encodeBuildStreaks` has updated the vertex buffer.
    public func setParticleStreaks(_ sources: [StreakSource]) {
        fxs.particleStreaks = sources
    }

    /// Render every `StreakSource` into `hdrTexture` using `particleStreakVS/FS`
    /// (velocity-aligned capsule impostors, additive HDR blend, depth-test no-write).
    /// One render encoder for all sources; buffers / textures rebind per draw.
    func encodeStreakDraw(_ cb: MTLCommandBuffer) {
        guard !fxs.particleStreaks.isEmpty, let pipeline = fxs.streakPipeline else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture     = gbs.hdrTexture
        pass.colorAttachments[0].loadAction  = .load
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture     = gbs.depthTexture
        pass.depthAttachment.loadAction  = .load
        pass.depthAttachment.storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.streaks"
        enc.setRenderPipelineState(pipeline)
        enc.setDepthStencilState(fxs.particleDepthState)   // depth-test, no depth-write
        enc.setCullMode(.none)
        // `frameUniformBuffer` starts with `viewProjection` (simd_float4x4,
        // offset 0) — matches `constant float4x4& viewProjection [[ buffer(2) ]]`
        // in `particleStreakVS`. Bound once; shared across all draws in this pass.
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 2)
        for source in fxs.particleStreaks where source.indexCount > 0 {
            enc.setVertexBuffer(source.positionColorBuffer, offset: 0, index: 0)
            enc.setVertexBuffer(source.uvBuffer,            offset: 0, index: 1)
            var p = source.params
            enc.setFragmentBytes(&p,
                                 length: MemoryLayout<ParticleStreakParams>.stride,
                                 index: 0)
            enc.setFragmentTexture(source.spriteTexture, index: 0)
            enc.drawIndexedPrimitives(type: .triangle,
                                      indexCount:        source.indexCount,
                                      indexType:         .uint32,
                                      indexBuffer:       source.indexBuffer,
                                      indexBufferOffset: 0)
        }
        enc.endEncoding()
    }

    /// Phase 4.11 — per-emitter integration step. One compute dispatch per
    /// emitter, covering its full capacity (dead particles early-return
    /// inside the kernel so the cost is bounded by capacity, not by
    /// alive count).
    func encodeParticleStep(_ cb: MTLCommandBuffer) {
        guard !fxs.particleEmitters.isEmpty else { return }
        let now = CACurrentMediaTime()
        let dt = Float(max(0, min(0.05, fixedFrameDelta ?? (now - fxs.lastParticleTickTime))))
        fxs.lastParticleTickTime = now
        guard dt > 0 else { return }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "Illuminatorama.particles.step"
        enc.setComputePipelineState(fxs.particleStepPipeline)
        for emitter in fxs.particleEmitters where emitter.enabled {
            var pf = IlluminatoramaParticleFrameUniforms(
                dt: dt,
                gravity: emitter.gravity,
                drag: emitter.drag,
                capacity: UInt32(emitter.capacity)
            )
            enc.setBuffer(emitter.particleBuffer, offset: 0, index: 0)
            enc.setBytes(&pf, length: MemoryLayout<IlluminatoramaParticleFrameUniforms>.stride,
                         index: 1)
            // 1D dispatch — the kernel declares `uint gid
            // [[thread_position_in_grid]]` (scalar), so threadgroup MUST
            // also be 1D (height & depth = 1). The shared `dispatch()`
            // helper builds a 2D threadgroup matching the lighting
            // kernel's signature, which crashes Metal validation here
            // with `validateBuiltinArguments` (the scalar gid can't
            // accept a 2D dispatch). Build a dedicated 1D dispatch.
            let tg = min(fxs.particleStepPipeline.maxTotalThreadsPerThreadgroup,
                         fxs.particleStepPipeline.threadExecutionWidth * 8)
            let groups = MTLSize(width: (emitter.capacity + tg - 1) / tg,
                                  height: 1, depth: 1)
            let threadgroup = MTLSize(width: tg, height: 1, depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threadgroup)
        }
        enc.endEncoding()
    }

    /// Phase 4.11 — additive HDR forward draw of surviving emitter sprites
    /// into `hdrTexture`; the texture carries `.renderTarget` usage (see
    /// `makeTargets`) so this pass can target it directly. Called only from
    /// `encodePostResolveFX` (Phase 4.43), AFTER the TAA resolve, so these
    /// additive sparks don't enter the temporal history. The velocity-aligned
    /// billboard shape gives them their own motion blur, so unlike the dust
    /// they self-masked most history smear — but they belong in the same
    /// post-resolve overlay for consistency and to keep history strictly
    /// opaque-only. (Pre-4.43 this ran before the SSR gather so sparks could
    /// reflect in nearby metals via SSR.)
    func encodeParticleDraw(_ cb: MTLCommandBuffer) {
        guard !fxs.particleEmitters.isEmpty else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = gbs.hdrTexture
        pass.colorAttachments[0].loadAction = .load    // additive: keep prior lighting
        pass.colorAttachments[0].storeAction = .store
        // Bind the G-buffer depth read-only so opaque geometry occludes
        // particles behind it. `.load + .store` keeps the existing
        // depth values intact for any later passes (TAA, etc.).
        pass.depthAttachment.texture = gbs.depthTexture
        pass.depthAttachment.loadAction = .load
        pass.depthAttachment.storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.label = "Illuminatorama.particles.draw"
        enc.setRenderPipelineState(fxs.particleDrawPipeline)
        enc.setDepthStencilState(fxs.particleDepthState)
        enc.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        for emitter in fxs.particleEmitters where emitter.enabled {
            enc.setVertexBuffer(emitter.particleBuffer, offset: 0, index: 0)
            // Phase 4.11 round 4 — velocity-aligned billboard quads. Six
            // vertices per particle = two triangles sharing the
            // diagonal; the vertex shader picks the corner via
            // `vid % 6` and the particle via `vid / 6`.
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: emitter.capacity * 6)
        }
        enc.endEncoding()
    }
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Particle step, repack, mesh-synth, particle/point-sprite/streak draw pipelines and the particle depth state.
    @MainActor static func buildParticlePipelines(into s: ParticleState, cache: SimPipelineCache, device: MTLDevice, library: MTLLibrary) throws {
        // ── Phase 4.11 — particle pipelines ─────────────────────────────
        guard let partStep = cache.pipelineState(name: "illumi_particles_step",
                                                  device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_particles_step")
        }
        s.particleStepPipeline = partStep

        // ── Phase 4.13a — DynamicMesh repack kernel ──────────────────
        guard let repack = cache.pipelineState(name: "illumi_repack_pos_norm",
                                                device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_repack_pos_norm")
        }
        s.repackPosNormPipeline = repack

        // ── Phase 4.21 — GPU mesh normal/tangent synthesis ───────────
        guard let meshAdj = cache.pipelineState(name: "illumi_mesh_build_adjacency",
                                                 device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_mesh_build_adjacency")
        }
        s.meshBuildAdjacencyPipeline = meshAdj
        guard let meshSynth = cache.pipelineState(name: "illumi_mesh_synth",
                                                  device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_mesh_synth")
        }
        s.meshSynthPipeline = meshSynth

        let partDrawDesc = MTLRenderPipelineDescriptor()
        partDrawDesc.label = "Illuminatorama.particles.draw"
        partDrawDesc.vertexFunction   = library.makeFunction(name: "illumi_particles_vs")
        partDrawDesc.fragmentFunction = library.makeFunction(name: "illumi_particles_fs")
        partDrawDesc.colorAttachments[0].pixelFormat = .rgba16Float
        // HDR additive blend. `one + one` lets bright particles accumulate
        // into the composite; bloom & tonemap then process them with the
        // same pipeline as direct + indirect lighting.
        partDrawDesc.colorAttachments[0].isBlendingEnabled = true
        partDrawDesc.colorAttachments[0].rgbBlendOperation = .add
        partDrawDesc.colorAttachments[0].alphaBlendOperation = .add
        partDrawDesc.colorAttachments[0].sourceRGBBlendFactor = .one
        partDrawDesc.colorAttachments[0].destinationRGBBlendFactor = .one
        partDrawDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        partDrawDesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        // Depth attachment must match the G-buffer's format so we can
        // bind that texture as a read-only depth target and have opaque
        // scene geometry occlude particles behind it.
        partDrawDesc.depthAttachmentPixelFormat = .depth32Float
        s.particleDrawPipeline = try device.makeRenderPipelineState(descriptor: partDrawDesc)

        // Phase 4.23 — host-owned point-sprite pipeline. Mirrors the
        // particles pipeline's HDR additive blend but uses MTLPrimitiveType
        // .point + a stride-parameterised vertex shader, so callers can
        // bind their own MTLBuffer of (position, color) without first
        // copying into Illuminatorama's particle layout.
        let extPartDesc = MTLRenderPipelineDescriptor()
        extPartDesc.label = "Illuminatorama.extParticles.draw"
        extPartDesc.vertexFunction   = library.makeFunction(name: "illumi_extparticle_vs")
        extPartDesc.fragmentFunction = library.makeFunction(name: "illumi_extparticle_fs")
        extPartDesc.colorAttachments[0].pixelFormat = .rgba16Float
        extPartDesc.colorAttachments[0].isBlendingEnabled = true
        extPartDesc.colorAttachments[0].rgbBlendOperation = .add
        extPartDesc.colorAttachments[0].alphaBlendOperation = .add
        extPartDesc.colorAttachments[0].sourceRGBBlendFactor = .one
        extPartDesc.colorAttachments[0].destinationRGBBlendFactor = .one
        extPartDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
        extPartDesc.colorAttachments[0].destinationAlphaBlendFactor = .one
        extPartDesc.depthAttachmentPixelFormat = .depth32Float
        s.extParticleDrawPipeline = try device.makeRenderPipelineState(descriptor: extPartDesc)

        // Phase 4.44b — velocity-aligned streak pipeline. Uses the same
        // `particleStreakVS/FS` shaders as `ParticleStreakPass` but targets
        // the HDR composite directly so streaks participate in bloom + tonemap.
        // Optional: if the shaders are absent the pipeline stays nil and streak
        // draws are silently skipped — logged once here, not per-frame.
        if let svs = library.makeFunction(name: "particleStreakVS"),
           let sfs = library.makeFunction(name: "particleStreakFS") {
            let streakVD = MTLVertexDescriptor()
            // Buffer 0: FWStreakVertex (stride 32) — float3 position @ offset 0,
            //           float4 color (w = stretchFactor) @ offset 16.
            streakVD.attributes[0].format      = .float3
            streakVD.attributes[0].offset      = 0
            streakVD.attributes[0].bufferIndex = 0
            streakVD.attributes[1].format      = .float4
            streakVD.attributes[1].offset      = 16
            streakVD.attributes[1].bufferIndex = 0
            streakVD.layouts[0].stride         = 32
            streakVD.layouts[0].stepFunction   = .perVertex
            // Buffer 1: float2 UV (stride 8).
            streakVD.attributes[2].format      = .float2
            streakVD.attributes[2].offset      = 0
            streakVD.attributes[2].bufferIndex = 1
            streakVD.layouts[1].stride         = 8
            streakVD.layouts[1].stepFunction   = .perVertex
            let sd = MTLRenderPipelineDescriptor()
            sd.label                                                  = "Illuminatorama.streaks"
            sd.vertexFunction                                         = svs
            sd.fragmentFunction                                       = sfs
            sd.vertexDescriptor                                       = streakVD
            sd.colorAttachments[0].pixelFormat                       = .rgba16Float
            sd.colorAttachments[0].isBlendingEnabled                 = true
            sd.colorAttachments[0].rgbBlendOperation                 = .add
            sd.colorAttachments[0].sourceRGBBlendFactor              = .one
            sd.colorAttachments[0].destinationRGBBlendFactor         = .one
            sd.colorAttachments[0].alphaBlendOperation               = .add
            sd.colorAttachments[0].sourceAlphaBlendFactor            = .one
            sd.colorAttachments[0].destinationAlphaBlendFactor       = .one
            sd.depthAttachmentPixelFormat                            = .depth32Float
            s.streakPipeline = try? device.makeRenderPipelineState(descriptor: sd)
            if s.streakPipeline == nil {
                Self.log.error("Illuminatorama: streakPipeline compilation failed — streak draws disabled")
            }
        } else {
            Self.log.warning("Illuminatorama: particleStreakVS/FS not found in library — streak draws disabled")
        }

        // Depth state: less-equal compare, NO write. Lets each particle
        // get tested against G-buffer depth (so the cube/sphere/ground
        // hide particles behind them) while leaving the depth buffer
        // pristine for any subsequent passes.
        let partDepthDesc = MTLDepthStencilDescriptor()
        partDepthDesc.depthCompareFunction = .lessEqual
        partDepthDesc.isDepthWriteEnabled = false
        guard let pds = device.makeDepthStencilState(descriptor: partDepthDesc) else {
            throw IlluminatoramaError.pipelineCreationFailed("particleDepthState")
        }
        s.particleDepthState = pds
    }
}
