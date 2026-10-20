import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    // ── Phase 3.1: DDGI ──────────────────────────────────────────────────────

    // Swift mirrors of DDGIUniforms and DDGIInstanceData in Illuminatorama.metal.
    // ALIGNMENT: SIMD3<Float>(size=12, align=16) + UInt32(size=4) = 16 bytes
    // per group, matching Metal's float3+uint layout. See IlluminatoramaTypes.swift
    // for the same pattern applied to FrameUniforms.
    struct IlluminatoramaDDGIUniforms {
        var gridOrigin: SIMD3<Float>;          var gridDimsX: UInt32        // 0
        var directionalLightDir: SIMD3<Float>; var gridDimsY: UInt32        // 16
        var directionalLightColor: SIMD3<Float>; var gridDimsZ: UInt32      // 32
        var probeSpacing: Float;               var raysPerProbe: UInt32     // 48
        var hysteresis: Float;                 var depthHysteresis: Float   // 56
        var irradianceScale: Float;            var enabled: UInt32          // 64
        var irrTileSize: UInt32;               var depthTileSize: UInt32    // 72
        var instanceCount: UInt32;             var twoBounceEnabled: UInt32 // 80
        // `rayMask` repurposes the former `_pad2` (S3.2 Ultra): the TLAS
        // transport-ray mask for `illumi_ddgi_trace_tlas`; 0 on the analytic path.
        var emitterCount: UInt32 = 0; var rayMask: UInt32 = 0               // 88..92
        // stride: 96 bytes
    }

    private struct DDGIGPUInstanceData {
        var invModelMatrix: simd_float4x4  // 0
        var normalMatrix:   simd_float4x4  // 64
        var albedo:   SIMD3<Float>; var metallic:   Float  // 128
        var emission: SIMD3<Float>; var roughness:  Float  // 144
        var meshKind: UInt32                               // 160
        var _pad0: UInt32 = 0; var _pad1: UInt32 = 0; var _pad2: UInt32 = 0 // 164..172
        // stride: 176 bytes
    }

    // Mirror of DDGIRayRecord — only used to compute buffer size in Swift.
    private struct DDGIRayRecord {
        var dirAndDist: SIMD4<Float>  // 16
        var radiance:   SIMD4<Float>  // 16
        // stride: 32 bytes
    }

    // Swift mirror of DDGIPointEmitter in Illuminatorama.metal.
    private struct DDGIGPUEmitter {
        var position: SIMD3<Float>; var radius: Float   // 0..15
        var color: SIMD3<Float>;    var _pad: Float = 0 // 16..31
        // stride: 32 bytes
    }

    /// Lazily allocate (or reallocate on grid-dim change) the irradiance atlas,
    /// depth atlas, and ray buffer. No-op when dims are already current.
    private func ensureDDGIResources() {
        let dims = ddgiGridDims
        guard dims != ddgiCurrentGridDims else { return }

        let irrTile = Self.ddgiIrrTileSize
        let depTile = Self.ddgiDepthTileSize
        let irrW = (irrTile + 2) * dims.x * dims.z
        let irrH = (irrTile + 2) * dims.y
        let depW = (depTile + 2) * dims.x * dims.z
        let depH = (depTile + 2) * dims.y
        let probeCount  = dims.x * dims.y * dims.z
        let rayBufLen   = MemoryLayout<DDGIRayRecord>.stride * probeCount * max(1, ddgiRaysPerProbe)

        let irrDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: irrW, height: irrH, mipmapped: false)
        irrDesc.usage       = [.shaderRead, .shaderWrite]
        irrDesc.storageMode = .private

        let depDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float, width: depW, height: depH, mipmapped: false)
        depDesc.usage       = [.shaderRead, .shaderWrite]
        depDesc.storageMode = .private

        // Phase 3.3 — allocate ping-pong atlases (A + B for both irradiance
        // and depth). Both start zero-initialised by Metal; the first frame
        // after a (re)allocation will read zeros for the "previous" atlas,
        // which correctly degenerates to one-bounce GI until the hysteresis
        // EMA builds up the indirect term.
        guard let irrA = device.makeTexture(descriptor: irrDesc),
              let irrB = device.makeTexture(descriptor: irrDesc),
              let depA = device.makeTexture(descriptor: depDesc),
              let depB = device.makeTexture(descriptor: depDesc),
              let rays = device.makeBuffer(length: rayBufLen, options: .storageModePrivate)
        else {
            Self.log.error("DDGI: atlas allocation failed for grid \(dims.x)×\(dims.y)×\(dims.z)")
            return
        }
        irrA.label = "Illuminatorama.ddgi.irrAtlas.A"
        irrB.label = "Illuminatorama.ddgi.irrAtlas.B"
        depA.label = "Illuminatorama.ddgi.depthAtlas.A"
        depB.label = "Illuminatorama.ddgi.depthAtlas.B"
        rays.label = "Illuminatorama.ddgi.rays"

        ddgiIrradianceAtlasA = irrA
        ddgiIrradianceAtlasB = irrB
        ddgiDepthAtlasA      = depA
        ddgiDepthAtlasB      = depB
        ddgiRayBuffer        = rays
        ddgiCurrentGridDims  = dims
        // Reset the ping-pong so the first frame after (re)allocation reads
        // from a definitely-zeroed B atlas, not a partially-filled one.
        ddgiUseAtlasA        = true
    }

    /// Grow (or create) the per-frame DDGI instance-data buffer.
    private func ensureDDGIInstanceDataBuffer(count: Int) {
        guard count > ddgiInstanceDataCapacity else { return }
        let newCap = max(count, ddgiInstanceDataCapacity * 2, 8)
        let len = MemoryLayout<DDGIGPUInstanceData>.stride * newCap
        guard let buf = device.makeBuffer(length: len, options: .storageModeShared) else { return }
        buf.label = "Illuminatorama.ddgi.instanceData"
        ddgiInstanceDataBuffer = buf
        ddgiInstanceDataCapacity = newCap
    }

    /// FNV-1a digest of every DDGI probe-field input EXCEPT the instance / proxy
    /// geometry (folded in separately via `ddgiGeometryEpoch`). A change re-arms the
    /// settle gate — the sun rotating, the grid moving, an emitter lighting up, a
    /// ray/hysteresis knob turned. NOT hashed: the sky texture's *contents* (an
    /// in-place re-bake keeps the same object); a time-of-day change that alters the
    /// sky also rotates `directionalLightDirection`, which IS hashed, so the common
    /// case re-arms anyway.
    private func ddgiScalarInputHash(
        emitterLights: [(position: SIMD3<Float>, color: SIMD3<Float>, radius: Float)]
    ) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ f: Float) { h = (h ^ UInt64(f.bitPattern)) &* 0x0000_0100_0000_01b3 }
        func mix3(_ v: SIMD3<Float>) { mix(v.x); mix(v.y); mix(v.z) }
        func mixI(_ i: Int) { h = (h ^ UInt64(bitPattern: Int64(i))) &* 0x0000_0100_0000_01b3 }
        mix3(simd_normalize(directionalLightDirection))
        mix3(directionalLightColor)
        mix3(ddgiGridOrigin)
        mixI(ddgiGridDims.x); mixI(ddgiGridDims.y); mixI(ddgiGridDims.z)
        mix(ddgiProbeSpacing); mixI(max(1, ddgiRaysPerProbe))
        mix(ddgiHysteresis); mix(ddgiIrradianceScale)
        mixI(ddgiTwoBounceEnabled ? 1 : 0)
        // S3.2 Ultra — the lane itself is an input: flipping between the TLAS
        // and analytic kernels (knob or TLAS availability) must re-arm the
        // settle gate so the field re-converges instead of freezing mid-mix.
        mixI(ddgiRTTraceUsable ? 1 : 0)
        for e in emitterLights { mix3(e.position); mix3(e.color); mix(e.radius) }
        return h
    }

    /// True when this frame's DDGI trace can run against the real TLAS.
    private var ddgiRTTraceUsable: Bool {
        ddgiRTTraceEnabled && rtTLASActive && rtTLAS != nil
            && rtInstanceDataBuffer != nil && rtObjNormalBuffer != nil
            && ddgiTraceTLASPipeline != nil
    }

    func encodeDDGIFrame(_ cb: MTLCommandBuffer) {
        // Phase 3.3 — toggle the ping-pong at the *start* of the DDGI frame
        // (not the end) so accessor semantics stay obvious through the
        // function: `ddgiIrradianceCurrent` is the atlas we write to in
        // this frame AND the atlas the lighting kernel reads after we
        // return. `ddgiIrradiancePrevious` is the atlas the trace kernel
        // reads for the second bounce.
        //
        // First frame after enable / resource (re)allocation:
        //   useAtlasA starts at true → toggles to false → Current = B,
        //   Previous = A (both still empty). Trace reads zero, so two-
        //   bounce contributes zero this frame; one-bounce direct works
        //   normally. Subsequent frames carry real data.
        //
        // S3.2 — the ping-pong toggle MOVED below the settle gate: a frame the
        // gate skips must NOT toggle, or `ddgiIrradianceCurrent` would flip to the
        // atlas we did not write and the lighting pass would read a half-stale
        // field. Only a frame that actually traces advances the ping-pong.

        // S3.2 — the trace intersects the render instances (analytic subset) PLUS
        // the host's non-rendered proxy boxes (a house's walls/floor/ceiling). The
        // uniform instance count, the buffer capacity and the pack loop all cover
        // both.
        let totalDDGIInstances = instances.count + ddgiProxyInstances.count

        // Write the DDGIUniforms buffer regardless of the enabled flag —
        // the lighting kernel always binds it (enabled=0 makes the lookup a no-op).
        let dimsNow = ddgiEnabled ? ddgiGridDims : SIMD3(1, 1, 1)
        var u = IlluminatoramaDDGIUniforms(
            gridOrigin:           ddgiGridOrigin,
            gridDimsX:            UInt32(dimsNow.x),
            directionalLightDir:  simd_normalize(directionalLightDirection),
            gridDimsY:            UInt32(dimsNow.y),
            directionalLightColor: directionalLightColor,
            gridDimsZ:            UInt32(dimsNow.z),
            probeSpacing:         ddgiProbeSpacing,
            raysPerProbe:         UInt32(max(1, ddgiRaysPerProbe)),
            hysteresis:           ddgiHysteresis,
            depthHysteresis:      min(ddgiHysteresis + 0.01, 0.99),
            irradianceScale:      ddgiIrradianceScale,
            enabled:              ddgiEnabled ? 1 : 0,
            irrTileSize:          UInt32(Self.ddgiIrrTileSize),
            depthTileSize:        UInt32(Self.ddgiDepthTileSize),
            instanceCount:        UInt32(totalDDGIInstances),
            twoBounceEnabled:     (ddgiEnabled && ddgiTwoBounceEnabled) ? 1 : 0
        )

        // Collect analytic emitters from registered particle emitters + SimEngine fields.
        var emitterLights: [(position: SIMD3<Float>, color: SIMD3<Float>, radius: Float)] = []
        for e in particleEmitters where e.enabled {
            if let light = e.ddgiLight { emitterLights.append(light) }
        }
        for f in particleFields {
            if let light = f.ddgiLight { emitterLights.append(light) }
        }
        u.emitterCount = UInt32(emitterLights.count)
        // S3.2 Ultra — transport mask for the TLAS trace: opaque + invisible
        // occluder, the same mask GI rays carry (a probe ray must stop at the
        // lighting-only ceiling slabs or interiors flood with sky). 0 on the
        // analytic lane, where the kernel never reads it.
        let rtTrace = ddgiRTTraceUsable
        u.rayMask = rtTrace ? Self.rtTransportRayMask : 0

        memcpy(ddgiUniformBuffer.contents(), &u, MemoryLayout<IlluminatoramaDDGIUniforms>.stride)

        guard ddgiEnabled, totalDDGIInstances > 0 else { return }

        // ── S3.2 view-independent settle gate ────────────────────────────
        // The probe field is world-space; a camera-only orbit does not change it,
        // so re-tracing every frame during an orbit is pure waste. Re-arm the
        // trace only when a probe-field INPUT actually moves: geometry (folded via
        // a monotonic epoch bumped on any unstable-content frame), proxies, the
        // directional light, the grid, the ray/hysteresis knobs, or the emitters.
        // With `ddgiConvergenceFrames == 0` (every scene but Daydream's house) the
        // gate always traces — pre-S3.2 behaviour, bit-identical.
        if instanceStableFrames == 0 || ddgiProxiesDirty { ddgiGeometryEpoch &+= 1 }
        let scalarHash = ddgiScalarInputHash(emitterLights: emitterLights)
        let inputHash = ddgiGeometryEpoch &* 0x100000001b3 ^ scalarHash
        let trace = ddgiConvergenceGate.shouldTrace(
            inputHash: inputHash, convergenceFrames: ddgiConvergenceFrames)
        ddgiDidTraceLastFrame = trace
        // A skipped frame reuses the converged atlas the lighting pass already
        // reads (`ddgiIrradianceCurrent`, untouched since the last traced frame).
        guard trace else { return }

        ddgiUseAtlasA.toggle()
        ensureDDGIResources()
        guard let irrAtlas    = ddgiIrradianceCurrent,
              let depAtlas    = ddgiDepthCurrent,
              let irrAtlasPrv = ddgiIrradiancePrevious,
              let depAtlasPrv = ddgiDepthPrevious,
              let rayBuf      = ddgiRayBuffer else { return }

        // Build the emitter buffer (may be empty; the trace kernel reads emitterCount).
        let emitCount = emitterLights.count
        if emitCount > 0 {
            if emitCount > ddgiEmitterCapacity {
                let newCap = max(emitCount, ddgiEmitterCapacity * 2, 4)
                let len = MemoryLayout<DDGIGPUEmitter>.stride * newCap
                ddgiEmitterBuffer = device.makeBuffer(length: len, options: .storageModeShared)
                ddgiEmitterBuffer?.label = "Illuminatorama.ddgi.emitters"
                ddgiEmitterCapacity = newCap
            }
            if let buf = ddgiEmitterBuffer {
                let ptr = buf.contents().bindMemory(to: DDGIGPUEmitter.self, capacity: emitCount)
                for (i, light) in emitterLights.enumerated() {
                    ptr[i] = DDGIGPUEmitter(position: light.position, radius: light.radius,
                                            color: light.color)
                }
            }
        }

        // Build per-instance DDGI data (invModelMatrix + material) for the trace kernel.
        // PERF (static-scene skip): this loop pays a general 4×4 inverse per instance
        // per frame — with stable content the packed buffer already holds exactly this
        // data, so repack only when the content generation moved (or on first use).
        ensureDDGIInstanceDataBuffer(count: totalDDGIInstances)
        guard let instBuf = ddgiInstanceDataBuffer else { return }
        if instanceStableFrames == 0 || ddgiPackedInstanceCount != totalDDGIInstances
            || ddgiProxiesDirty {
            let instPtr = instBuf.contents().bindMemory(
                to: DDGIGPUInstanceData.self, capacity: totalDDGIInstances)
            // Render instances first (analytic subset intersects; meshKind 3 is
            // skipped in-shader), then the non-rendered proxy boxes appended.
            var i = 0
            for ref in instances {
                // Phase 2.6 — meshKind=3 is "custom mesh, no analytic intersection";
                // the trace kernel skips intersection on anything not in {0,1,2}.
                instPtr[i] = DDGIGPUInstanceData(
                    invModelMatrix: ref.data.modelMatrix.inverse,
                    normalMatrix:   ref.data.normalMatrix,
                    albedo:         ref.data.albedo,
                    metallic:       ref.data.metallic,
                    emission:       ref.data.emission,
                    roughness:      ref.data.roughness,
                    meshKind:       ref.meshKind.gpuMeshKind
                )
                i += 1
            }
            for ref in ddgiProxyInstances {
                instPtr[i] = DDGIGPUInstanceData(
                    invModelMatrix: ref.data.modelMatrix.inverse,
                    normalMatrix:   ref.data.normalMatrix,
                    albedo:         ref.data.albedo,
                    metallic:       ref.data.metallic,
                    emission:       ref.data.emission,
                    roughness:      ref.data.roughness,
                    meshKind:       ref.meshKind.gpuMeshKind
                )
                i += 1
            }
            ddgiPackedInstanceCount = totalDDGIInstances
            ddgiProxiesDirty = false
        }

        let probeCount   = ddgiGridDims.x * ddgiGridDims.y * ddgiGridDims.z
        let raysPerProbe = max(1, ddgiRaysPerProbe)
        let sky = equirectSky ?? dummySkyTexture

        guard let enc = timedComputeEncoder(cb, "ddgi") else { return }
        enc.label = "Illuminatorama.ddgi"

        // ── Trace: (raysPerProbe, probeCount) 2D dispatch ───────────────
        // Phase 3.3 — trace reads the PREVIOUS-frame atlases at texture(1)/(2)
        // for the second-bounce lookup; the update kernels below write to
        // the CURRENT-frame atlases. With ping-pong these are distinct
        // textures so the read-then-write within one encoder doesn't alias.
        //
        // S3.2 Ultra — when the TLAS lane is usable this dispatches
        // `illumi_ddgi_trace_tlas` instead: same ray pattern, same uniforms,
        // same DDGIRayRecord output, but hits come from the real acceleration
        // structure (buffers 4–6). The update kernels below cannot tell the
        // difference; `ddgiScalarInputHash` mixes the lane so a flip re-settles.
        let tracePipeline: MTLComputePipelineState
        if rtTrace, let tlasPipeline = ddgiTraceTLASPipeline,
           let tlas = rtTLAS, let rtInst = rtInstanceDataBuffer, let objN = rtObjNormalBuffer {
            tracePipeline = tlasPipeline
            enc.setComputePipelineState(tlasPipeline)
            enc.setAccelerationStructure(tlas, bufferIndex: 4)
            enc.setBuffer(rtInst, offset: 0, index: 5)
            enc.setBuffer(objN,   offset: 0, index: 6)
            // The TLAS references the BLASes, which reference the mesh
            // vertex/index buffers — all must be resident for the intersector.
            for blas in rtBLASList { enc.useResource(blas, usage: .read) }
            for blas in rtCurveBLASList { enc.useResource(blas, usage: .read) }
            for buf in rtResidentBuffers { enc.useResource(buf, usage: .read) }
        } else {
            tracePipeline = ddgiTracePipeline
            enc.setComputePipelineState(ddgiTracePipeline)
        }
        enc.setBuffer(rayBuf,              offset: 0, index: 0)
        enc.setBuffer(ddgiUniformBuffer,   offset: 0, index: 1)
        enc.setBuffer(instBuf,             offset: 0, index: 2)
        enc.setBuffer(ddgiEmitterBuffer ?? instBuf, offset: 0, index: 3)  // dummy when no emitters
        enc.setTexture(sky,         index: 0)
        enc.setTexture(irrAtlasPrv, index: 1)
        enc.setTexture(depAtlasPrv, index: 2)
        let tgwT = tracePipeline.threadExecutionWidth
        enc.dispatchThreadgroups(
            MTLSize(width: (raysPerProbe + tgwT - 1) / tgwT, height: probeCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgwT, height: 1, depth: 1))

        // ── Update irradiance atlas: (irrTileSize × irrTileSize × probeCount) ─
        enc.setComputePipelineState(ddgiUpdateIrrPipeline)
        enc.setBuffer(rayBuf,             offset: 0, index: 0)
        enc.setBuffer(ddgiUniformBuffer,  offset: 0, index: 1)
        enc.setTexture(irrAtlas, index: 0)
        let irrTile = Self.ddgiIrrTileSize
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: irrTile, depth: probeCount),
            threadsPerThreadgroup: MTLSize(width: irrTile, height: 1, depth: 1))

        // ── Update depth atlas: (depthTileSize × depthTileSize × probeCount) ─
        enc.setComputePipelineState(ddgiUpdateDepthPipeline)
        enc.setBuffer(rayBuf,             offset: 0, index: 0)
        enc.setBuffer(ddgiUniformBuffer,  offset: 0, index: 1)
        enc.setTexture(depAtlas, index: 0)
        let depTile = Self.ddgiDepthTileSize
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: depTile, depth: probeCount),
            threadsPerThreadgroup: MTLSize(width: depTile, height: 1, depth: 1))

        enc.endEncoding()
    }
}
