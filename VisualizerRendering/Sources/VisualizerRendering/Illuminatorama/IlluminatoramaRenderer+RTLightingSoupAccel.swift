import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Build the soup primitive acceleration structure from the stored triangle
    /// buffers, plus — when a soup curve set is active (#60 item 7 incr. 2) — a
    /// SECOND curve geometry descriptor (round Catmull-Rom wood). Rebuilt on a
    /// topology change (new triangle geometry OR a curve registry version bump).
    /// When curves are active the AS uses `.refit` usage so the per-frame sway
    /// kernel + `refitRTSoupAccel` can update the control points in place.
    func buildSoupAccel() {
        guard let vb = rts.rtVertexBuffer, let ib = rts.rtIndexBuffer, rts.rtTriangleCount > 0 else { return }
        // Triangle soup AS — its OWN primitive AS (curves can't share it). Static,
        // so default usage (the room/extractor scenes never refit this).
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = vb
        geom.vertexBufferOffset = 0
        geom.vertexStride = MemoryLayout<SIMD3<Float>>.stride   // 16
        geom.vertexFormat = .float3
        geom.indexBuffer = ib
        geom.indexBufferOffset = 0
        geom.indexType = .uint32
        geom.triangleCount = rts.rtTriangleCount

        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [geom]
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                            options: .storageModePrivate),
            let cmd = commandQueue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("buildSoupAccel: AS build setup failed")
            return
        }
        storage.label = "Illuminatorama.rt.accel"
        enc.build(accelerationStructure: storage, descriptor: asDesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: one-time / topology-change AS build (not per-frame)
        Self.recordASBuildError(cmd, "buildSoupAccel.primAS")
        rts.rtAccel = storage
        rts.rtScratch = scratch
        // Curves live in a SEPARATE primitive AS (Metal forbids mixed-type
        // descriptors in one AS). Built with .refit so the per-frame sway kernel
        // can update the control points + refit in place.
        buildSoupCurveAccel()
    }

    /// Build the curve-only primitive AS (#60 item 7 incr. 2) for the active soup
    /// curve set. Separate from the triangle AS (Metal: one geometry type per
    /// primitive AS). `.refit` usage so `encodeCurveWindDisplaceAndRefit` can sway.
    func buildSoupCurveAccel() {
        rts.rtSoupCurveAccel = nil; rts.rtSoupRefitScratch = nil
        guard rts.rtSoupCurvesActive, let cgeom = makeSoupCurveGeom() else { return }
        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [cgeom]
        asDesc.usage = .refit
        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(length: max(sizes.buildScratchBufferSize, 16),
                                            options: .storageModePrivate),
            let cmd = commandQueue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            Self.log.error("buildSoupCurveAccel: AS build setup failed; dropping curves.")
            rts.rtSoupCurvesActive = false; return
        }
        storage.label = "Illuminatorama.rt.curveAccel"
        enc.build(accelerationStructure: storage, descriptor: asDesc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding(); cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: one-time / topology-change curve AS build
        Self.recordASBuildError(cmd, "buildSoupCurveAccel.curveAS")
        rts.rtSoupCurveAccel = storage
        // Scratch sized for BOTH refit and a full in-place rebuild — the per-frame
        // sway path REBUILDS the curve AS (cheap for the few-segment curve set),
        // confirmed to update the traced control points (the displace→rebuild→trace
        // +2m-offset probe moved the curves' shadows; whole-frame RMSE 0.0064).
        rts.rtSoupRefitScratch = device.makeBuffer(
            length: max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize, 16),
            options: .storageModePrivate)
    }

    /// Build the curve geometry descriptor for the active soup curve set from the
    /// pooled buffers. The control-point buffer is the AS-referenced (swayed) pool;
    /// the kernel writes this each frame and the AS refits to it.
    func makeSoupCurveGeom() -> MTLAccelerationStructureCurveGeometryDescriptor? {
        guard let cpts = rts.rtSoupCurvePoolPoints, let crad = rts.rtSoupCurveRadii,
              let cseg = rts.rtSoupCurveSegments, rts.rtSoupCurvePointCount > 0,
              rts.rtSoupCurveSegmentCount > 0 else { return nil }
        let g = MTLAccelerationStructureCurveGeometryDescriptor()
        g.controlPointBuffer = cpts
        g.controlPointCount = rts.rtSoupCurvePointCount
        g.controlPointStride = 12       // packed float3
        g.controlPointFormat = .float3
        g.radiusBuffer = crad
        g.radiusFormat = .float
        g.radiusStride = MemoryLayout<Float>.stride
        g.indexBuffer = cseg
        g.indexType = .uint32
        g.segmentCount = rts.rtSoupCurveSegmentCount
        g.segmentControlPointCount = 4
        g.curveType = .round            // exact 3D normals (branches)
        g.curveBasis = .catmullRom
        g.curveEndCaps = .sphere
        g.opaque = true
        return g
    }

    /// Adopt the process-wide curve registry for the SOUP path (#60 item 7 incr. 2).
    /// Called from `render()`'s non-TLAS branch before the RT passes. A registry
    /// version bump is a TOPOLOGY change here (the soup AS gains/replaces its curve
    /// geometry descriptor), so it rebuilds `rtAccel`. Requires BOTH curve-variant
    /// pipelines — if either fails to compile the set is dropped (a curve-holding
    /// AS traced by a triangle-contract intersector is UB, never shippable).
    func syncSoupCurveRegistry() {
        // TLAS-path renderers (extracted scenes) handle curves via syncCurveRegistry
        // in updateRTAccel; this is the soup/native path only.
        guard !buildRTFromExtractedScene else { return }
        let reg = IlluminatoramaCurveRegistry.shared
        guard reg.version != rts.rtSoupCurveSyncVersion else { return }
        rts.rtSoupCurveSyncVersion = reg.version
        // The soup path supports ONE curve set (single geometry descriptor). A
        // scene that needs many concatenates them into one set before registering.
        rts.rtSoupCurveSet = device.supportsRaytracing ? reg.sets.first : nil
        if rts.rtSoupCurveSet != nil {
            let cache = engine.pipelineCache
            var on = true
            if rts.rtSoupCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                rts.rtSoupCurvePipeline = cache.pipelineState(
                    name: "illumi_rt_lighting", device: device, constants: cv, variantKey: "soupCurves")
            }
            if scs.surfCacheSoupCurvePipeline == nil {
                let cv = MTLFunctionConstantValues()
                cv.setConstantValue(&on, type: .bool, index: Self.curveFunctionConstantIndex)
                scs.surfCacheSoupCurvePipeline = cache.pipelineState(
                    name: "illumi_surfcache_update", device: device, constants: cv, variantKey: "soupCurves")
            }
            if rts.rtSoupCurvePipeline == nil || scs.surfCacheSoupCurvePipeline == nil {
                Self.log.error("Soup curve RT: kRTCurvesEnabled pipeline variant failed; dropping curve set.")
                rts.rtSoupCurveSet = nil
            }
        }
        buildSoupCurvePool()
        rts.rtSoupCurvesActive = (rts.rtSoupCurveSet != nil)
            && rts.rtSoupCurvePoolPoints != nil && rts.rtSoupCurveRadii != nil
            && rts.rtSoupCurveSegments != nil && rts.rtSoupCurveSetDataBuffer != nil
        buildSoupAccel()   // rebuild the AS with / without the curve geometry descriptor
    }

    /// Pool the active soup curve set's control points (flat `[Float]` → read as
    /// `packed_float3`, stride 12), radii, segment indices (RELATIVE — one geometry
    /// descriptor), per-point wind attribute, and the single `RTCurveSetData`
    /// (identity transform — the soup set is world-space). Keeps a REST copy of the
    /// points so the per-frame wind kernel can re-displace from rest each frame.
    func buildSoupCurvePool() {
        rts.rtSoupCurveRestPoints = nil; rts.rtSoupCurvePoolPoints = nil; rts.rtSoupCurveRadii = nil
        rts.rtSoupCurveSegments = nil; rts.rtSoupCurveWindAttr = nil; rts.rtSoupCurveSetDataBuffer = nil
        rts.rtSoupCurvePointCount = 0; rts.rtSoupCurveSegmentCount = 0
        guard let set = rts.rtSoupCurveSet else { return }
        var pts: [Float] = []; pts.reserveCapacity(set.controlPoints.count * 3)
        for p in set.controlPoints { pts.append(p.x); pts.append(p.y); pts.append(p.z) }
        let data = [RTCurveSetData(
            m0: set.transform.columns.0, m1: set.transform.columns.1,
            m2: set.transform.columns.2, m3: set.transform.columns.3,
            albedoRoughness: SIMD4(set.material.albedo, set.material.roughness),
            emissionPad: SIMD4(set.material.emission, 0),
            meta: SIMD4<UInt32>(0, 0, 0, 0))]   // single set ⇒ segment base 0
        let ptLen = MemoryLayout<Float>.stride * pts.count
        guard ptLen > 0,
              let rest = device.makeBuffer(bytes: pts, length: ptLen, options: .storageModeShared),
              let pool = device.makeBuffer(bytes: pts, length: ptLen, options: .storageModeShared),
              let rb = device.makeBuffer(bytes: set.radii,
                  length: MemoryLayout<Float>.stride * set.radii.count, options: .storageModeShared),
              let sb = device.makeBuffer(bytes: set.segmentIndices,
                  length: MemoryLayout<UInt32>.stride * set.segmentIndices.count, options: .storageModeShared),
              let db = device.makeBuffer(bytes: data,
                  length: MemoryLayout<RTCurveSetData>.stride, options: .storageModeShared)
        else {
            Self.log.error("Soup curve RT: pool allocation failed; dropping curve set.")
            rts.rtSoupCurveSet = nil; return
        }
        rest.label = "Illuminatorama.rt.soupCurve.restPoints"
        pool.label = "Illuminatorama.rt.soupCurve.points"
        rb.label = "Illuminatorama.rt.soupCurve.radii"
        sb.label = "Illuminatorama.rt.soupCurve.segments"
        db.label = "Illuminatorama.rt.soupCurve.setData"
        rts.rtSoupCurveRestPoints = rest; rts.rtSoupCurvePoolPoints = pool
        rts.rtSoupCurveRadii = rb; rts.rtSoupCurveSegments = sb; rts.rtSoupCurveSetDataBuffer = db
        rts.rtSoupCurvePointCount = set.controlPoints.count
        rts.rtSoupCurveSegmentCount = set.segmentIndices.count
        // Per-point wind attribute (sway/phase/flutter) — only when the set opts in.
        if set.windAttr.count == set.controlPoints.count {
            rts.rtSoupCurveWindAttr = device.makeBuffer(bytes: set.windAttr,
                length: MemoryLayout<SIMD4<Float>>.stride * set.windAttr.count,
                options: .storageModeShared)
            rts.rtSoupCurveWindAttr?.label = "Illuminatorama.rt.soupCurve.windAttr"
        }
    }

    /// Per-frame curve sway (#60 item 7 incr. 2): re-displace the rest control
    /// points into the AS-referenced pool with the SAME `applyTreeWind` the vertex
    /// shader uses, then refit the soup AS. No-op for a rigid set (no windAttr) or
    /// under `VIZ_ILLUMI_NO_CURVE_REFIT=1`.
    func encodeCurveWindDisplaceAndRefit(_ cb: MTLCommandBuffer) {
        guard rts.rtSoupCurvesActive, !rts.noCurveRefit,
              let pipe = rts.curveWindDisplacePipeline,
              let rest = rts.rtSoupCurveRestPoints, let pool = rts.rtSoupCurvePoolPoints,
              let wind = rts.rtSoupCurveWindAttr, rts.rtSoupCurvePointCount > 0 else {
            if !rts.curveDbgPrinted, Self.frameEnv("VIZ_CURVE_DEBUG") == "1" {
                rts.curveDbgPrinted = true
                let m = "CURVE displace SKIPPED: active=\(rts.rtSoupCurvesActive) noRefit=\(rts.noCurveRefit) pipe=\(rts.curveWindDisplacePipeline != nil) rest=\(rts.rtSoupCurveRestPoints != nil) pool=\(rts.rtSoupCurvePoolPoints != nil) wind=\(rts.rtSoupCurveWindAttr != nil) pts=\(rts.rtSoupCurvePointCount)\n"
                FileHandle.standardError.write(m.data(using: .utf8)!)
            }
            return
        }
        if !rts.curveDbgPrinted, Self.frameEnv("VIZ_CURVE_DEBUG") == "1" {
            rts.curveDbgPrinted = true
            let m = "CURVE displace RAN: pts=\(rts.rtSoupCurvePointCount) time=\(time) windStrength=\(treeWindStrength) heading=\(treeWindHeading)\n"
            FileHandle.standardError.write(m.data(using: .utf8)!)
        }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(rest, offset: 0, index: 0)
        enc.setBuffer(wind, offset: 0, index: 1)
        enc.setBuffer(pool, offset: 0, index: 2)
        var u = CurveWindUniforms(time: time, windStrength: treeWindStrength,
                                  windHeading: treeWindHeading,
                                  pointCount: UInt32(rts.rtSoupCurvePointCount))
        enc.setBytes(&u, length: MemoryLayout<CurveWindUniforms>.stride, index: 3)
        let w = min(pipe.threadExecutionWidth, rts.rtSoupCurvePointCount)
        enc.dispatchThreads(MTLSize(width: rts.rtSoupCurvePointCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
        enc.endEncoding()
        // Rebuild the CURVE-only AS in place from the displaced control points on
        // the SAME command buffer (Metal sequences the compute write → AS-build
        // read). A full rebuild is cheap for the few-segment curve set and
        // unambiguously re-reads the swayed pool (confirmed: the +2m-offset probe
        // moved the curves' shadows). gpu-ok: per-frame rebuild encoded on the
        // frame cb, no waitUntilCompleted.
        guard let accel = rts.rtSoupCurveAccel, let scratch = rts.rtSoupRefitScratch,
              let cgeom = makeSoupCurveGeom() else { return }
        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [cgeom]
        asDesc.usage = .refit
        guard let renc = cb.makeAccelerationStructureCommandEncoder() else { return }
        renc.build(accelerationStructure: accel, descriptor: asDesc,
                   scratchBuffer: scratch, scratchBufferOffset: 0)
        renc.endEncoding()
    }
}
