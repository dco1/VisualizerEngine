import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// MARK: - RTState

/// Hardware ray tracing: acceleration structures, TLAS/curve pools, GI history + denoise, radiance cache, RT uniform buffers.
/// Owned by the renderer as `rts`; every member keeps the name it had on the renderer.
@MainActor
final class RTState {
    /// The unjittered view-projection of the last frozen-camera resolve (see the camera-cut guard
    /// in `render`); nil while still mode is off.
    var lastStillViewProjection: simd_float4x4?
    var previousRTGIStatic = false
    /// DH-0539 — the RT reflection (specular) denoise chain's textures + knobs; see IlluminatoramaRenderer+SpecularDenoise.swift.
    let spec = SpecDenoiseState()

    var rcFramesSinceReset = 0
    var rcKeysBuffer: MTLBuffer?
    var rcDataBuffer: MTLBuffer?
    var rcDummyKeys: MTLBuffer?
    var rcDummyData: MTLBuffer?

    // RT diffuse (soft shadow + 1-bounce GI) buffer — written by the RT pass,
    // bilateral-filtered into the composite by `encodeRTDenoiseComposite`.
    var rtDiffuseTexture: MTLTexture!

    // RT-GI temporal accumulation: the rtDiffuse term gets its own
    // velocity-reprojected exponential history so the 1-bounce GI keeps
    // converging under camera motion (the main TAA can't). Ping-pong, full-res.
    var rtGIHistoryA: MTLTexture!  // full-res, rgba16Float
    var rtGIHistoryB: MTLTexture!
    var rtGIHistoryToggle: Bool = false

    var rcNeedsClear = true
    var previousRTGITemporalEnabled: Bool = false

    var giSampleCount: MTLTexture!  // full-res
    // SVGF à-trous cascade textures for RT GI (when svgfEnabled).
    // Color ping-pongs {giAtrousA, giAtrousB}; variance ping-pongs {giVariance,
    // giVarianceB}. The two pairs MUST stay disjoint — an à-trous level reads one
    // signal at dilated neighbour taps while writing the other to its own texel, so
    // any shared texture is an in-place read/write race that corrupts colour in a
    // dilated-stride pattern (the issue-#65 masonry). Three buffers can't keep both
    // ping-pongs disjoint; hence the fourth.
    var giVariance: MTLTexture!  // full-res r16Float (variance A)
    var giVarianceB: MTLTexture!  // full-res r16Float (variance B)
    var giAtrousA: MTLTexture!  // full-res rgba16Float
    var giAtrousB: MTLTexture!  // full-res rgba16Float

    var previousGIIrradianceMode = false

    var rtDenoisePipeline: MTLComputePipelineState!
    var rtGITemporalPipeline: MTLComputePipelineState!
    var svgfVariancePipeline: MTLComputePipelineState!  // Phase 4.44 SVGF
    var svgfAtrousPipeline: MTLComputePipelineState!  // Phase 4.44 SVGF

    var rtProxyMeshKinds: Set<IlluminatoramaRenderer.MeshKind> = []

    var rtPipeline: MTLComputePipelineState?
    var rtAccel: MTLAccelerationStructure?
    var rtScratch: MTLBuffer?
    var rtVertexBuffer: MTLBuffer?
    var rtIndexBuffer: MTLBuffer?
    var rtTriAlbedoBuffer: MTLBuffer?
    var rtTriNormalBuffer: MTLBuffer?
    var rtTriangleCount: Int = 0
    var rtUniformBuffer: MTLBuffer!
    var rtDenoiseUniformBuffer: MTLBuffer!
    var rtGITemporalUniformBuffer: MTLBuffer!
    var rtFrameSeed: UInt32 = 0

    var rtTLASPipeline: MTLComputePipelineState?

    var rtMeshCache: [ObjectIdentifier: IlluminatoramaRenderer.RTMeshEntry] = [:]
    /// Whether any cached mesh needs the per-frame refit. Stored rather than derived so the
    /// per-frame static-skip test stays O(1); maintained by `pruneRTMeshCache()`, which runs
    /// on every rebuild — the only occasion the answer can change.
    var rtHasDeformingBLAS = false

    var rtResidentBuffers: [MTLBuffer] = []   // mesh vb/ib the BLASes reference
    var rtBLASList: [MTLAccelerationStructure] = []   // index = accelerationStructureIndex
    var rtObjNormalBuffer: MTLBuffer?  // concatenated object-space per-tri normals (float4)
    var rtObjNormalCount: Int = 0
    /// Concatenated per-triangle MESH UVs (3 × `float2` per triangle), built in
    /// lockstep with `rtObjNormalBuffer` so a hit reads its texture coordinates at
    /// `(normalBase + primitive_id) * 3`. Lets a ray-traced hit take a REAL albedo
    /// atlas sample instead of the instance's mean albedo — the through-glass
    /// "flat slabs" fix. Nil when no mesh exposed CPU-readable vertices.
    var rtObjUVBuffer: MTLBuffer?
    var rtObjUVCount: Int = 0

    // Surface cache (P1c): per-TLAS-instance base offset into the grouped soup
    // triangle list (== the `triCard`/`triUVa`/`triUVc` index space). Built
    // alongside the world-space soup in `rebuildRTAccel` when the cache is on,
    // so a TLAS hit resolves `soupTriBase[instance_id] + primitive_id` → card.
    var rtSoupTriBaseBuffer: MTLBuffer?
    var rtSoupTriBaseCount: Int = 0

    var rtInstanceDataBuffer: MTLBuffer?  // RTInstanceData[], grouped order
    var rtInstanceDescBuffer: MTLBuffer?  // MTLAccelerationStructureInstanceDescriptor[]
    var rtTLAS: MTLAccelerationStructure?
    var rtTLASScratch: MTLBuffer?
    var rtTLASCapacity: Int = 0
    var rtTLASInstanceCount: Int = 0
    var rtTLASTopologyHash: Int = .min
    // ── Curve primitives (#60 item 7, increment 1) ──
    // Sets adopted from `IlluminatoramaCurveRegistry` (synced per frame by
    // version). Control points / radii / segment indices for ALL sets are
    // pooled into three shared buffers (segment index values are ABSOLUTE into
    // the pool, so every curve BLAS and the shading kernels read the same
    // pool with only a per-set segment base). One curve BLAS per set; one TLAS
    // instance per set, appended AFTER the triangle instances — a curve hit's
    // `instance_id - curveInstanceBase` is its set index. Kernels that trace
    // the TLAS switch to a `kRTCurvesEnabled` function-constant variant when
    // curves are present (the intersector's geometry-type contract must match
    // the AS content), so curve-free scenes keep the exact pipelines they had.
    var rtCurveSets: [IlluminatoramaCurveSet] = []
    var rtCurveSyncVersion: Int = 0
    var rtCurveBLASList: [MTLAccelerationStructure] = []
    var rtCurvePoolPoints: MTLBuffer?  // packed float3 (stride 12)
    var rtCurvePoolRadii: MTLBuffer?  // float
    var rtCurvePoolSegments: MTLBuffer?  // uint32, absolute into the pool
    var rtCurvePoolPointCount: Int = 0
    var rtCurveSetSegmentBase: [Int] = [] // per set: first segment in the pool
    var rtCurveSetDataBuffer: MTLBuffer?  // RTCurveSetData per set
    var rtCurveInstanceCount: Int = 0     // sets in the CURRENT TLAS topology
    var rtTLASCurvePipeline: MTLComputePipelineState?

    // ── Curve primitives on the SOUP path (#60 item 7, increment 2) ──
    // Forest et al. are native soup scenes (`setRTGeometry` → primitive AS,
    // `encodeRTLightingPass`), not the TLAS path. The soup AS gains a SECOND
    // geometry descriptor (a curve geometry) alongside the triangle soup; a hit
    // is a curve iff `res.type == curve` (single set, identity-placed, so no
    // instance_id). The pooled rest points are re-displaced per frame by the
    // wind kernel and the AS is refit, so the RT wood sways with the raster tubes.
    var rtSoupCurveSet: IlluminatoramaCurveSet?
    var rtSoupCurveSyncVersion: Int = 0
    var rtSoupCurvesActive: Bool = false
    var rtSoupCurveRestPoints: MTLBuffer?  // packed float3, un-swayed (kernel input)
    var rtSoupCurvePoolPoints: MTLBuffer?  // packed float3, AS-referenced (kernel output / rest copy)
    var rtSoupCurveRadii: MTLBuffer?  // float
    var rtSoupCurveSegments: MTLBuffer?  // uint32 (relative — single geometry descriptor)
    var rtSoupCurveWindAttr: MTLBuffer?  // float4 per point (empty set ⇒ nil ⇒ rigid)
    var rtSoupCurveSetDataBuffer: MTLBuffer?  // single RTCurveSetData
    var rtSoupCurvePointCount: Int = 0
    var rtSoupCurveSegmentCount: Int = 0
    /// SEPARATE curve-only primitive AS (#60 item 7 incr. 2). Metal forbids mixing
    /// triangle + curve geometry descriptors in ONE primitive AS ("must be the
    /// same type"), so curves live in their own AS and the kernel traces BOTH
    /// (any-hit occludes; nearest-hit compares distance). `.refit` usage for sway.
    var rtSoupCurveAccel: MTLAccelerationStructure?
    var rtSoupRefitScratch: MTLBuffer?
    var rtSoupCurvePipeline: MTLComputePipelineState?

    var curveWindDisplacePipeline: MTLComputePipelineState?

    /// Freeze curves at rest pose (A/B for the sway refit). `VIZ_ILLUMI_NO_CURVE_REFIT=1`.
    let noCurveRefit = ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_CURVE_REFIT"] == "1"
    var curveDbgPrinted = false   // VIZ_CURVE_DEBUG one-shot
    var rtInstUniformBuffer: MTLBuffer!
    var rtInstFrameSeed: UInt32 = 0
    /// DH-1588 — the hit sun-cascade proof's audit counters (asked / proven / proven-but-unoccluded) +
    /// the GPU-side `RTSunCascadeCull` block. Made on first TLAS dispatch.
    var hitSunCullStats: MTLBuffer?
    /// DH-0887 — +1 per TLAS lighting dispatch, never by any other pass, so the GI's Sobol index
    /// is contiguous (`rtInstFrameSeed` also advances in the glass pass, and every-other Sobol
    /// index covers only half of the unit square).
    var rtGIProgressiveIndex: UInt32 = 0

    /// True after a successful TLAS (re)build/refit this frame — the RT pass
    /// then traces the instance AS instead of the primitive-AS soup.
    var rtTLASActive: Bool = false

    // ── Phase 4.37a — RT hang-proof guard ────────────────────────────────
    // A scene whose geometry is GPU-fed (regenerated `IlluminatoramaMesh`
    // identities each frame) or whose mesh set/topology otherwise changes
    // every frame makes `rtTopologyHash()` differ every frame → `updateRTAccel`
    // takes the REBUILD branch each frame (new per-mesh BLAS builds, each with
    // a `waitUntilCompleted`, + a TLAS rebuild) → the main thread hangs and a
    // faulted AS traces to magenta. The descriptor-driven per-scene RT opt-in
    // (Phase 4.36/4.37) runs RT in the live tick loop, so any flagged scene
    // could trip this. Two gates make RT fail safe (fall back to non-RT, never
    // hang): an UP-FRONT size cap, and a rebuild-THRASH counter.
    //
    // The guard used to LATCH until the next scene attach, and that was wrong
    // for an editor. In a document app the model mutates on every tick of a
    // drag (push/pull a wall ⇒ new wall meshes ⇒ a new topology hash every
    // frame), so seven frames into the drag the thrash counter tripped, RT
    // glass switched off mid-gesture, and it NEVER came back: a perf spike
    // followed by a permanent visual pop, with no user-reachable way to restore
    // it short of reopening the document. The guard's real job is "don't rebuild
    // a thrashing TLAS", which stops being true the moment the scene settles.
    //
    // So the trip is now RECOVERABLE. While disabled the renderer keeps
    // evaluating cheap evidence each frame — the size caps (a scene can shrink
    // back under them) and the topology hash — and re-arms RT once the topology
    // has held still for `rtRecoveryStableFrames`. A genuinely thrashing scene
    // (Forest: GPU-fed leaf geometry, a different hash every single frame) never
    // satisfies that and stays disabled exactly as before, which is the
    // property the hang-proofing depends on.
    /// The guard's whole state, as one testable value (see
    /// `IlluminatoramaRTHangGuard` for the recovery semantics and why they
    /// changed from a permanent latch).
    var rtGuard = IlluminatoramaRTHangGuard()

    var rtConsecutiveRebuilds: Int = 0
}

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// RT-GI denoise, temporal accumulate and SVGF (variance + a-trous) compute pipelines.
    @MainActor static func buildGIDenoisePipelines(into s: RTState, cache: SimPipelineCache, device: MTLDevice) throws {
        guard let rtDenoisePipelineBuilt = cache.pipelineState(name: "illumi_rt_denoise", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_rt_denoise")
        }
        s.rtDenoisePipeline = rtDenoisePipelineBuilt
        guard let rtGITemporalPipelineBuilt = cache.pipelineState(name: "illumi_rt_gi_temporal", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_rt_gi_temporal")
        }
        s.rtGITemporalPipeline = rtGITemporalPipelineBuilt
        guard let svgfVariancePipelineBuilt = cache.pipelineState(name: "illumi_svgf_variance", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_svgf_variance")
        }
        s.svgfVariancePipeline = svgfVariancePipelineBuilt
        guard let svgfAtrousPipelineBuilt = cache.pipelineState(name: "illumi_svgf_atrous", device: device) else {
            throw IlluminatoramaError.pipelineCreationFailed("illumi_svgf_atrous")
        }
        s.svgfAtrousPipeline = svgfAtrousPipelineBuilt
    }

    /// Hardware RT lighting + instanced-RT (TLAS) pipelines, their uniform buffers and the curve-wind displace kernel. Returns whether RT is supported (the base RT pipeline built).
    @MainActor static func buildRTPipelines(into s: RTState, cache: SimPipelineCache, device: MTLDevice, library: MTLLibrary) throws -> Bool {
        let curvesOff = Self.curvesOffConstants()
        var builtRTPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let rtFn = try? library.makeFunction(name: "illumi_rt_lighting",
                                                constantValues: curvesOff) {
            builtRTPipeline = try? device.makeComputePipelineState(function: rtFn) // gpu-ok: one-time init, RT-capability-gated optional pipeline
        }
        s.rtPipeline = builtRTPipeline
        guard let rtUB = device.makeBuffer(length: MemoryLayout<RTUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtUniforms")
        }
        rtUB.label = "Illuminatorama.rt.uniforms"
        s.rtUniformBuffer = rtUB
        guard let rtDnUB = device.makeBuffer(length: MemoryLayout<RTDenoiseUniforms>.stride,
                                             options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtDenoiseUniforms")
        }
        rtDnUB.label = "Illuminatorama.rt.denoiseUniforms"
        s.rtDenoiseUniformBuffer = rtDnUB
        guard let rtGiUB = device.makeBuffer(length: MemoryLayout<RTGITemporalUniforms>.stride,
                                             options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtGITemporalUniforms")
        }
        rtGiUB.label = "Illuminatorama.rt.giTemporalUniforms"
        s.rtGITemporalUniformBuffer = rtGiUB

        // Curve sway displacement (#60 item 7 incr. 2) — plain compute, no RT
        // dependency; nil leaves curves rigid (no per-frame re-displace).
        s.curveWindDisplacePipeline =
            cache.pipelineState(name: "illumi_curve_wind_displace", device: device)
        // ── Instanced-RT (TLAS) pipeline ─────────────────────────────
        // Same constant-declaring caveat as the surfcache TLAS kernel above:
        // build the explicit kRTCurvesEnabled=false base variant.
        var builtTLASPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let tlasFn = try? library.makeFunction(name: "illumi_rt_lighting_tlas",
                                                  constantValues: curvesOff) {
            builtTLASPipeline = try? device.makeComputePipelineState(function: tlasFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        s.rtTLASPipeline = builtTLASPipeline
        guard let riUB = device.makeBuffer(length: MemoryLayout<RTInstUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("rtInstUniforms")
        }
        riUB.label = "Illuminatorama.rt.instUniforms"
        s.rtInstUniformBuffer = riUB
        return builtRTPipeline != nil
    }

    /// The kRTCurvesEnabled / kSCCurvesEnabled = false base specialisation shared by every curve-declaring kernel.
    static func curvesOffConstants() -> MTLFunctionConstantValues {
        let curvesOff = MTLFunctionConstantValues()
        var curvesOffValue = false
        curvesOff.setConstantValue(&curvesOffValue, type: .bool,
                                   index: Self.curveFunctionConstantIndex)
        return curvesOff
    }
}
