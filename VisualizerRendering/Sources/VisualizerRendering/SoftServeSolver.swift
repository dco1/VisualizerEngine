import Metal
import OSLog
import simd
import VisualizerCore

// plausibility: real — GPU offload of main's CURRENT (steady-treadmill, baked-
// flavour) InfiniteSoftServe CPU math; same math, parallelised. Verified A/B vs
// the CPU path (VIZ_SOFTSERVE_GPU=0/1).

// ── SoftServeSolver ─────────────────────────────────────────────────────────
//
// Encodes the GPU compute phases that replace the two dominant CPU hot-paths in
// main's InfiniteSoftServeController:
//
//   encodeCoilExtrude   — replaces updateCoilExtrusion() (~43ms/frame). Writes
//                         POSITION + NORMAL only into the coil mesh vertex
//                         buffer; per-vertex colour stays BAKED (CPU
//                         bakeFlavorColors — a TAA-correctness design).
//   encodeDepthAndCoat  — replaces displaceChocoCoat() (~205ms/frame):
//                         ss_depth_init, ss_depth_scatter, ss_dilate_max×2,
//                         ss_box_blur_float (depth), ss_cover_threshold,
//                         ss_dilate_bin×2, ss_erode_bin, swap, ss_box_blur_float
//                         (filmH→hSmooth), ss_erode_bin (interior), then
//                         ss_coat_displace.
//
// All GPU buffers use .storageModeShared (unified memory on Apple Silicon), so
// the CPU can read the results without a memcpy or a waitUntilCompleted — the
// serial SimEngine queue guarantees the compute CB commits before
// renderer.render() (its own CB on the same queue) runs, so no explicit wait is
// needed. NO per-frame waitUntilCompleted, NO per-subsystem queue/library.

@MainActor
public final class SoftServeSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "SoftServeSolver")
    private let device: MTLDevice

    // ── Coil extrusion GPU buffers (uploaded once per rebuildCoilMesh) ──────
    public private(set) var coilVertexCount: Int = 0
    private var arcTurnsBuf:      MTLBuffer?   // float  × N
    private var origPosBuf:       MTLBuffer?   // float4 × N  (also used by depth-scatter)
    private var origNormBuf:      MTLBuffer?   // float4 × N
    private var straightPosBuf:   MTLBuffer?   // float4 × N
    private var straightNormBuf:  MTLBuffer?   // float4 × N

    // ── Steady-treadmill clip buffers (uploaded once per rebuildCoilMesh) ────
    public private(set) var steadyVertexCount: Int = 0
    private var steadyOrigPosBuf: MTLBuffer?   // float4 × M (baked, never mutated)

    // ── Choco coat GPU buffers (allocated once per setupChocoFilm) ───────────
    public private(set) var filmCols: Int = 0
    public private(set) var filmRows: Int = 0

    // Depth ping-pong (table holds float-as-uint during scatter, plain float after).
    private var depthTableBuf:  MTLBuffer?   // uint/float × cells
    private var depthSmoothBuf: MTLBuffer?   // float × cells
    // Coverage ping-pong pair (uint masks).
    private var coverBufA: MTLBuffer?        // uint × cells
    private var coverBufB: MTLBuffer?        // uint × cells
    // Blurred film-h for hole fill.
    private var filmHSmoothBuf: MTLBuffer?   // float × cells
    // Persistent per-frame inputs (CPU film state copied in each tick — avoids
    // a per-frame makeBuffer allocation).
    private var filmHInBuf:     MTLBuffer?   // float × cells
    private var filmFlavorBuf:  MTLBuffer?   // int32 × cells

    public init(device: MTLDevice) {
        self.device = device
    }

    /// Persistent input buffers for the coat pass (filmH + flavour). Copy the
    /// current-frame CPU arrays into them (shared memory → GPU-visible). Returns
    /// nil until allocateFilm has run.
    public func filmInputBuffers() -> (h: MTLBuffer, flavor: MTLBuffer)? {
        guard let h = filmHInBuf, let f = filmFlavorBuf else { return nil }
        return (h, f)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// Upload coil mesh arrays. Call after every rebuildCoilMesh(). Arrays are
    /// converted to float4 (SIMD4) to satisfy the 16-byte alignment rule.
    public func uploadCoilData(
        arcTurns:     [Float],
        origPos:      [SIMD3<Float>],
        origNorm:     [SIMD3<Float>],
        straightPos:  [SIMD3<Float>],
        straightNorm: [SIMD3<Float>]
    ) {
        let n = arcTurns.count
        guard n > 0,
              origPos.count == n, origNorm.count == n,
              straightPos.count == n, straightNorm.count == n else {
            Self.log.error("uploadCoilData: inconsistent array sizes")
            return
        }
        coilVertexCount = n
        arcTurnsBuf = makeBuffer(arcTurns, label: "ss.arcTurns")
        origPosBuf      = makeBuffer(origPos.map      { SIMD4($0.x, $0.y, $0.z, 0) }, label: "ss.origPos")
        origNormBuf     = makeBuffer(origNorm.map     { SIMD4($0.x, $0.y, $0.z, 0) }, label: "ss.origNorm")
        straightPosBuf  = makeBuffer(straightPos.map  { SIMD4($0.x, $0.y, $0.z, 0) }, label: "ss.straightPos")
        straightNormBuf = makeBuffer(straightNorm.map { SIMD4($0.x, $0.y, $0.z, 0) }, label: "ss.straightNorm")
    }

    /// Upload the steady-treadmill mesh's baked positions (call after every
    /// rebuildCoilMesh). Consumed by `encodeSteadyClip` to collapse the coil
    /// below the cup floor without mutating the source positions.
    public func uploadSteadyData(positions: [SIMD3<Float>]) {
        steadyVertexCount = positions.count
        steadyOrigPosBuf = makeBuffer(positions.map { SIMD4($0.x, $0.y, $0.z, 0) },
                                      label: "ss.steadyOrigPos")
    }

    /// Encode the steady-clip compute pass into `cb`. `outputBuffer` =
    /// coilSteadyMesh.vertexBuffer (IlluminatoramaVertex × M).
    public func encodeSteadyClip(
        into cb: MTLCommandBuffer,
        outputBuffer: MTLBuffer,
        uniforms: SteadyClipUniforms
    ) {
        guard steadyVertexCount > 0, let osp = steadyOrigPosBuf else { return }
        guard let pso = SimPipelineCache.shared.pipelineState(name: "ss_steady_clip", device: device) else {
            Self.log.error("ss_steady_clip pipeline missing"); return
        }
        var u = uniforms
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "ss_steady_clip"
        enc.setComputePipelineState(pso)
        enc.setBytes(&u, length: MemoryLayout<SteadyClipUniforms>.stride, index: 0)
        enc.setBuffer(osp, offset: 0, index: 1)
        enc.setBuffer(outputBuffer, offset: 0, index: 7)
        dispatch1D(enc: enc, pso: pso, count: steadyVertexCount)
        enc.endEncoding()
    }

    /// Allocate coat buffers when the film grid size is set (or changes).
    public func allocateFilm(cols: Int, rows: Int) {
        filmCols = cols
        filmRows = rows
        let cells = cols * rows
        guard cells > 0 else { return }
        let bytesF = cells * MemoryLayout<Float>.stride
        let bytesU = cells * MemoryLayout<UInt32>.stride
        depthTableBuf  = device.makeBuffer(length: bytesU, options: .storageModeShared)
        depthSmoothBuf = device.makeBuffer(length: bytesF, options: .storageModeShared)
        coverBufA      = device.makeBuffer(length: bytesU, options: .storageModeShared)
        coverBufB      = device.makeBuffer(length: bytesU, options: .storageModeShared)
        filmHSmoothBuf = device.makeBuffer(length: bytesF, options: .storageModeShared)
        filmHInBuf     = device.makeBuffer(length: bytesF, options: .storageModeShared)
        filmFlavorBuf  = device.makeBuffer(length: cells * MemoryLayout<Int32>.stride,
                                           options: .storageModeShared)
        depthTableBuf?.label  = "ss.depthTable"
        depthSmoothBuf?.label = "ss.depthSmooth"
        coverBufA?.label      = "ss.coverA"
        coverBufB?.label      = "ss.coverB"
        filmHSmoothBuf?.label = "ss.filmHSmooth"
        filmHInBuf?.label     = "ss.filmHIn"
        filmFlavorBuf?.label  = "ss.filmFlavorIn"
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Encoding helpers (encode into a caller-owned command buffer)
    // ─────────────────────────────────────────────────────────────────────────

    /// Encode the coil-extrude compute pass into `cb`. `outputBuffer` =
    /// coilMesh.vertexBuffer (IlluminatoramaVertex × N).
    public func encodeCoilExtrude(
        into cb: MTLCommandBuffer,
        outputBuffer: MTLBuffer,
        uniforms: CoilUniforms
    ) {
        guard coilVertexCount > 0,
              let arc  = arcTurnsBuf,
              let opb  = origPosBuf,
              let onb  = origNormBuf,
              let spb  = straightPosBuf,
              let snb  = straightNormBuf else { return }
        guard let pso = SimPipelineCache.shared.pipelineState(name: "ss_coil_extrude", device: device) else {
            Self.log.error("ss_coil_extrude pipeline missing"); return
        }
        var u = uniforms
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "ss_coil_extrude"
        enc.setComputePipelineState(pso)
        enc.setBytes(&u, length: MemoryLayout<CoilUniforms>.stride, index: 0)
        enc.setBuffer(arc, offset: 0, index: 1)
        enc.setBuffer(opb, offset: 0, index: 2)
        enc.setBuffer(onb, offset: 0, index: 3)
        enc.setBuffer(spb, offset: 0, index: 4)
        enc.setBuffer(snb, offset: 0, index: 5)
        enc.setBuffer(outputBuffer, offset: 0, index: 7)
        dispatch1D(enc: enc, pso: pso, count: coilVertexCount)
        enc.endEncoding()
    }

    /// Encode the depth-scatter + morphological-close + coat-displace passes.
    /// Mirrors displaceChocoCoat() step-for-step.
    ///   `filmHBuf`      = MTLBuffer wrapping the current-frame chocoFilmH array.
    ///   `filmFlavorBuf` = MTLBuffer wrapping chocoFilmFlavor (as Int32).
    ///   `outputBuffer`  = chocoShellMesh.vertexBuffer.
    public func encodeDepthAndCoat(
        into cb: MTLCommandBuffer,
        filmHBuf: MTLBuffer,
        filmFlavorBuf: MTLBuffer,
        outputBuffer: MTLBuffer,
        uniforms: CoatUniforms,
        depthUniforms: DepthScatterUniforms
    ) {
        guard filmRows > 0, filmCols > 0,
              let dTbl    = depthTableBuf,
              let dSmth   = depthSmoothBuf,
              let cvA     = coverBufA,
              let cvB     = coverBufB,
              let hSmth   = filmHSmoothBuf,
              let origPos = origPosBuf else { return }

        let cells = filmRows * filmCols
        let morphU = MorphUniforms(rows: UInt32(filmRows), cols: UInt32(filmCols),
                                   revealFloor: uniforms.revealFloor, pad: 0)

        // ── Depth build: init → scatter ──────────────────────────────────────
        encode1D(cb, "ss_depth_init", cells) { enc, _ in
            var du = depthUniforms
            enc.setBytes(&du, length: MemoryLayout<DepthScatterUniforms>.stride, index: 0)
            enc.setBuffer(dTbl, offset: 0, index: 1)
        }
        if coilVertexCount > 0 {
            encode1D(cb, "ss_depth_scatter", coilVertexCount) { enc, _ in
                var du = depthUniforms
                enc.setBytes(&du, length: MemoryLayout<DepthScatterUniforms>.stride, index: 0)
                enc.setBuffer(origPos, offset: 0, index: 1)
                enc.setBuffer(dTbl,    offset: 0, index: 2)
            }
        }

        // ── Depth smoothing (CPU: dilate table→smooth, dilate smooth→table,
        //    boxBlur table→smooth → final drape in depthSmooth). ──────────────
        encode2D(cb, "ss_dilate_max", "dilate_depth_1", morphU, src: dTbl,  dst: dSmth)
        encode2D(cb, "ss_dilate_max", "dilate_depth_2", morphU, src: dSmth, dst: dTbl)
        encode2D(cb, "ss_box_blur_float", "blur_depth", morphU, src: dTbl,  dst: dSmth)

        // ── Coverage close. CPU sequence (cover=cvA, tmp=cvB):
        //    threshold→cvA; dilate cvA→cvB; dilate cvB→cvA; erode cvA→cvB;
        //    swap(cover,tmp) ⇒ closed mask now in cvB, junk in cvA;
        //    boxBlur filmH→hSmooth; erode (cover=cvB)→tmp=cvA  ⇒ interior in cvA.
        // After: cvB = CLOSED mask, cvA = INTERIOR mask. ─────────────────────
        encode1D(cb, "ss_cover_threshold", cells) { enc, _ in
            var mu = morphU
            enc.setBytes(&mu, length: MemoryLayout<MorphUniforms>.stride, index: 0)
            enc.setBuffer(filmHBuf, offset: 0, index: 1)
            enc.setBuffer(cvA,      offset: 0, index: 2)
        }
        encode2D(cb, "ss_dilate_bin", "dilate_cov_1", morphU, src: cvA, dst: cvB)
        encode2D(cb, "ss_dilate_bin", "dilate_cov_2", morphU, src: cvB, dst: cvA)
        encode2D(cb, "ss_erode_bin",  "erode_cov_1",  morphU, src: cvA, dst: cvB)
        // (CPU swap(cover,tmp): cover←cvB. We now treat cvB as the closed mask.)
        encode2D(cb, "ss_box_blur_float", "blur_hsmooth", morphU, src: filmHBuf, dst: hSmth)
        encode2D(cb, "ss_erode_bin",  "erode_interior", morphU, src: cvB, dst: cvA)
        // cvB = closed mask, cvA = interior mask.

        // ── Coat displace ────────────────────────────────────────────────────
        if let pso = SimPipelineCache.shared.pipelineState(name: "ss_coat_displace", device: device),
           let enc = cb.makeComputeCommandEncoder() {
            enc.label = "ss_coat_displace"
            enc.setComputePipelineState(pso)
            var cu = uniforms
            enc.setBytes(&cu, length: MemoryLayout<CoatUniforms>.stride, index: 0)
            enc.setBuffer(dSmth,         offset: 0, index: 1)
            enc.setBuffer(filmHBuf,      offset: 0, index: 2)
            enc.setBuffer(hSmth,         offset: 0, index: 3)
            enc.setBuffer(cvB,           offset: 0, index: 4)   // closed mask
            enc.setBuffer(cvA,           offset: 0, index: 5)   // interior mask
            enc.setBuffer(filmFlavorBuf, offset: 0, index: 6)
            enc.setBuffer(outputBuffer,  offset: 0, index: 7)
            dispatch2D(enc: enc, pso: pso, cols: filmCols, rows: filmRows)
            enc.endEncoding()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private func makeBuffer<T>(_ array: [T], label: String) -> MTLBuffer? {
        guard !array.isEmpty else { return nil }
        let bytes = array.count * MemoryLayout<T>.stride
        let buf = array.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: bytes,
                              options: .storageModeShared)
        }
        guard let buf else { return nil }
        buf.label = label
        return buf
    }

    private func encode1D(_ cb: MTLCommandBuffer, _ name: String, _ count: Int,
                          _ setup: (MTLComputeCommandEncoder, MTLComputePipelineState) -> Void) {
        guard let pso = SimPipelineCache.shared.pipelineState(name: name, device: device),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = name
        enc.setComputePipelineState(pso)
        setup(enc, pso)
        dispatch1D(enc: enc, pso: pso, count: count)
        enc.endEncoding()
    }

    private func encode2D(_ cb: MTLCommandBuffer, _ name: String, _ label: String,
                          _ morphU: MorphUniforms, src: MTLBuffer, dst: MTLBuffer) {
        guard let pso = SimPipelineCache.shared.pipelineState(name: name, device: device),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pso)
        var mu = morphU
        enc.setBytes(&mu, length: MemoryLayout<MorphUniforms>.stride, index: 0)
        enc.setBuffer(src, offset: 0, index: 1)
        enc.setBuffer(dst, offset: 0, index: 2)
        dispatch2D(enc: enc, pso: pso, cols: filmCols, rows: filmRows)
        enc.endEncoding()
    }

    private func dispatch1D(enc: MTLComputeCommandEncoder,
                             pso: MTLComputePipelineState, count: Int) {
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    private func dispatch2D(enc: MTLComputeCommandEncoder,
                             pso: MTLComputePipelineState, cols: Int, rows: Int) {
        enc.dispatchThreads(MTLSize(width: cols, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
}

// ── Uniform types — Swift mirrors of Metal structs ────────────────────────────
// ALIGNMENT RULE: every field must match the Metal struct byte-for-byte.
// Use Float (4) and UInt32 (4) only — no SIMD3 in shared structs.

/// Mirrors `CoilUniforms` in SoftServeSolver.metal. 48 bytes.
public struct CoilUniforms {
    public var tipTurns:            Float
    public var coilStraightTurns:   Float
    public var pitch:               Float
    public var dieExitY:            Float
    public var tubeRadius:          Float
    public var coilRadius:          Float
    public var funnelStraightTurns: Float
    public var funnelRampTurns:     Float
    public var liveCupRimWorldY:    Float
    public var coneHeight:          Float
    public var coneFloorY:          Float
    public var vertexCount:         UInt32

    public init(tipTurns: Float, coilStraightTurns: Float, pitch: Float, dieExitY: Float,
                tubeRadius: Float, coilRadius: Float, funnelStraightTurns: Float,
                funnelRampTurns: Float, liveCupRimWorldY: Float, coneHeight: Float,
                coneFloorY: Float, vertexCount: UInt32) {
        self.tipTurns = tipTurns; self.coilStraightTurns = coilStraightTurns
        self.pitch = pitch; self.dieExitY = dieExitY; self.tubeRadius = tubeRadius
        self.coilRadius = coilRadius; self.funnelStraightTurns = funnelStraightTurns
        self.funnelRampTurns = funnelRampTurns; self.liveCupRimWorldY = liveCupRimWorldY
        self.coneHeight = coneHeight; self.coneFloorY = coneFloorY
        self.vertexCount = vertexCount
    }
}

/// Mirrors `CoatUniforms` in SoftServeSolver.metal. 64 bytes.
public struct CoatUniforms {
    public var coilRadius:  Float
    public var tubeRadius:  Float
    public var gap:         Float
    public var rowH:        Float
    public var topY:        Float
    public var maxWedge:    Float
    public var subRow:      Float
    public var dAlpha:      Float
    public var hiddenR:     Float
    public var revealFloor: Float
    public var cols:        UInt32
    public var rows:        UInt32
    public var debug:       UInt32
    public var pad0:        UInt32 = 0
    public var pad1:        UInt32 = 0
    public var pad2:        UInt32 = 0

    public init(coilRadius: Float, tubeRadius: Float, gap: Float, rowH: Float,
                topY: Float, maxWedge: Float, subRow: Float, dAlpha: Float,
                hiddenR: Float, revealFloor: Float, cols: UInt32, rows: UInt32,
                debug: UInt32) {
        self.coilRadius = coilRadius; self.tubeRadius = tubeRadius; self.gap = gap
        self.rowH = rowH; self.topY = topY; self.maxWedge = maxWedge
        self.subRow = subRow; self.dAlpha = dAlpha; self.hiddenR = hiddenR
        self.revealFloor = revealFloor; self.cols = cols; self.rows = rows
        self.debug = debug
    }
}

/// Mirrors `DepthScatterUniforms` in SoftServeSolver.metal. 48 bytes.
public struct DepthScatterUniforms {
    public var phi:            Float
    public var coilRadius:     Float
    public var dieExitY:       Float
    public var rowH:           Float
    public var topY:           Float
    public var maxWedge:       Float
    public var cols:           UInt32
    public var rows:           UInt32
    public var totalCoilVerts: UInt32
    public var pad0:           UInt32 = 0
    public var pad1:           UInt32 = 0
    public var pad2:           UInt32 = 0

    public init(phi: Float, coilRadius: Float, dieExitY: Float, rowH: Float,
                topY: Float, maxWedge: Float, cols: UInt32, rows: UInt32,
                totalCoilVerts: UInt32) {
        self.phi = phi; self.coilRadius = coilRadius; self.dieExitY = dieExitY
        self.rowH = rowH; self.topY = topY; self.maxWedge = maxWedge
        self.cols = cols; self.rows = rows; self.totalCoilVerts = totalCoilVerts
    }
}

/// Mirrors `MorphUniforms` in SoftServeSolver.metal. 16 bytes.
public struct MorphUniforms {
    public var rows:        UInt32
    public var cols:        UInt32
    public var revealFloor: Float
    public var pad:         UInt32 = 0
}

/// Mirrors `SteadyClipUniforms` in SoftServeSolver.metal. 32 bytes.
public struct SteadyClipUniforms {
    public var cutoffLocalY: Float
    public var capX:         Float
    public var capY:         Float
    public var capZ:         Float
    public var vertexCount:  UInt32
    public var pad0:         UInt32 = 0
    public var pad1:         UInt32 = 0
    public var pad2:         UInt32 = 0

    public init(cutoffLocalY: Float, capX: Float, capY: Float, capZ: Float,
                vertexCount: UInt32) {
        self.cutoffLocalY = cutoffLocalY
        self.capX = capX; self.capY = capY; self.capZ = capZ
        self.vertexCount = vertexCount
    }
}
