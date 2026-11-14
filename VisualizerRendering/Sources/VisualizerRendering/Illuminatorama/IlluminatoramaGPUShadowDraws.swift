import Metal
import OSLog
import simd

// GPU-DRIVEN SHADOW DRAW SUBMISSION (Daydream DH-1684) — the state.
//
// The why, the kernel and the order-independence argument are in IlluminatoramaGPUDraws.metal; the
// encode is IlluminatoramaRenderer+GPUShadowDraws.swift. What lives here:
//
//   · `IlluminatoramaGPUShadowRecord` / `IlluminatoramaGPUShadowView` — the kernel's input layouts.
//   · `GPUShadowDrawTable` — one record per drawable mesh group, built when the grouping, the mesh
//     table or a flag the classification reads changes (never on a camera-only frame), with the
//     groups the CPU loop still encodes (`fallback`) and the buffers the commands reach.
//   · `GPUShadowFrameSlot` — the per-frame ICB, view block and census counters. A slot is reused only
//     after its command buffer completed, so the GPU never reads a slot the CPU is rewriting, and it
//     keeps the table (therefore every mesh whose buffer address its commands hold) alive till then.

/// One draw record — 80 bytes, must match `GPUShadowRecord` in IlluminatoramaGPUDraws.metal.
struct IlluminatoramaGPUShadowRecord {
    var vertices: UInt64 = 0
    var indices: UInt64 = 0
    var instanceOffset: UInt32 = 0
    var indexCount: UInt32 = 0
    var instanceCount: UInt32 = 0
    var flags: UInt32 = 0
    var lo: SIMD4<Float> = .zero
    var hi: SIMD4<Float> = .zero
    var triangles: UInt32 = 0
    var pad0: UInt32 = 0, pad1: UInt32 = 0, pad2: UInt32 = 0

    static let uint32Indices: UInt32 = 1, bounded: UInt32 = 2, noSunShadow: UInt32 = 4
}

/// One view — 128 bytes, must match `GPUShadowView` in IlluminatoramaGPUDraws.metal.
struct IlluminatoramaGPUShadowView {
    var planes: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        = (.zero, .zero, .zero, .zero, .zero, .zero)
    var instances: UInt64 = 0
    var lightVP: UInt64 = 0
    var clock: UInt64 = 0
    var commandBase: UInt32 = 0
    var flags: UInt32 = 0

    static let cull: UInt32 = 1, sun: UInt32 = 2

    /// The clip volume of `m` as six world-space planes with unit normals (Metal clip space, the same
    /// six `IlluminatoramaClipVolume` tests), so the kernel's float epsilon is in metres. A
    /// degenerate plane becomes (0, 0, 0, 1): never outside.
    static func planes(_ m: simd_float4x4) -> [SIMD4<Float>] {
        func row(_ i: Int) -> SIMD4<Double> {
            SIMD4(Double(m.columns.0[i]), Double(m.columns.1[i]), Double(m.columns.2[i]), Double(m.columns.3[i]))
        }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        return [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map { p in
            let n = (p.x * p.x + p.y * p.y + p.z * p.z).squareRoot()
            guard n > 1e-12, n.isFinite, p.w.isFinite else { return SIMD4<Float>(0, 0, 0, 1) }
            return SIMD4<Float>(Float(p.x / n), Float(p.y / n), Float(p.z / n), Float(p.w / n))
        }
    }
}

/// What one shadow view asks of the GPU-driven path.
struct GPUShadowViewSpec {
    /// The matrix the vertex stage rasterises with (bound at vertex buffer 3).
    var vp: simd_float4x4
    /// The volume the view culls against (a region pass's own frustum); nil = culling off.
    var cullVP: simd_float4x4?
    /// A sun cascade: honours `directionalShadowExcludedKinds`.
    var sun: Bool
}

/// Everything the classification of a group reads, so a table is rebuilt exactly when it could differ.
struct GPUShadowTableKey: Equatable {
    var groupsVersion: Int
    var meshTableVersion: Int
    var cullBounds: IlluminatoramaRenderer.CullBoundsKey?
    var cullBoundsCount: Int
    var leafCutout: Bool
    var storesFrontFaces: Bool
    var followsDoubleSided: Bool
    var hidden: Set<IlluminatoramaRenderer.MeshKind>
    var noSun: Set<IlluminatoramaRenderer.MeshKind>
    var repacked: [IlluminatoramaRenderer.MeshKind]
}

/// The drawable groups of one grouping, as GPU records bucketed by shadow cull mode.
@MainActor
final class GPUShadowDrawTable {
    let key: GPUShadowTableKey
    let records: MTLBuffer?
    let count: Int
    /// Record ranges per cull mode — one `executeCommandsInBuffer` each.
    let buckets: [(cull: MTLCullMode, range: Range<Int>)]
    /// Group indices the CPU loop still encodes (indirect / batched / deforming / leaf-cutout meshes).
    let fallback: [Int]
    /// Groups hidden from every raster pass: the CPU loop counted each as culled once per view.
    let hiddenGroups: Int
    /// The vertex and index buffers the commands point at (made resident per pass).
    let resources: [MTLResource]
    /// The meshes behind the records — keeps their buffers alive while a frame holds the table, and
    /// is re-checked every frame against `flags` (a host may flip a mesh flag in place).
    let meshes: [IlluminatoramaMesh]
    let flags: [UInt8]

    init(key: GPUShadowTableKey, records: MTLBuffer?, count: Int,
         buckets: [(cull: MTLCullMode, range: Range<Int>)], fallback: [Int], hiddenGroups: Int,
         resources: [MTLResource], meshes: [IlluminatoramaMesh], flags: [UInt8]) {
        self.key = key; self.records = records; self.count = count; self.buckets = buckets
        self.fallback = fallback; self.hiddenGroups = hiddenGroups; self.resources = resources
        self.meshes = meshes; self.flags = flags
    }

    /// The mesh flags a record's classification depends on, packed.
    static func flags(_ m: IlluminatoramaMesh) -> UInt8 {
        var f: UInt8 = 0
        if m.doubleSided { f |= 1 }
        if m.shadowCastsBothFaces { f |= 2 }
        if m.tracedOnly { f |= 4 }
        if m.leafCutout { f |= 8 }
        if m.indirectDrawBuffer != nil { f |= 16 }
        if m.gpuInstanceBuffer != nil { f |= 32 }
        return f
    }

    /// True while every mesh still carries the flags the table was classified with.
    var flagsHeld: Bool {
        for i in meshes.indices where Self.flags(meshes[i]) != flags[i] { return false }
        return true
    }
}

/// One frame's GPU-driven shadow resources (a cascade batch or a spot batch).
final class GPUShadowFrameSlot {
    var cb: MTLCommandBuffer?
    var icb: MTLIndirectCommandBuffer?
    var icbCapacity = 0
    var container: MTLBuffer?
    var views: MTLBuffer?
    var stats: MTLBuffer?
    var countsBytes = SIMD2<UInt32>(0, 0)
    /// Held until `cb` completes: the table (meshes) and the instance ring the commands point into.
    var keepAlive: [AnyObject] = []
    /// The instance ring the commands point into (made resident by every view's encoder).
    var ring: MTLBuffer?
    /// Census owed to `visibilityCullStats` once `cb` completes.
    var pending: (spot: Bool, views: Int, hidden: Int)?

    var isFree: Bool {
        guard let cb else { return true }
        return cb.status == .completed || cb.status == .error
    }
}

/// The renderer's GPU-driven shadow state (`renderer.gds`).
@MainActor
final class GPUShadowDrawState {
    static let log = Logger(subsystem: "VisualizerRendering", category: "gpuShadowDraws")

    /// A/B lever (`VIZ_ILLUMI_NO_GPU_DRAWS=1` starts it off): off = every shadow draw is encoded by
    /// the CPU loop, exactly as before DH-1684.
    var enabled = ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_GPU_DRAWS"] != "1"
    var table: GPUShadowDrawTable?
    var slots: [GPUShadowFrameSlot] = []
    var cullPipeline: MTLComputePipelineState?
    var drawPipeline: MTLRenderPipelineState?
    var argumentEncoder: MTLArgumentEncoder?
    /// Pipelines could not be built on this device / library — stay on the CPU loop.
    var unavailable = false
    /// Diagnostics: tables built, views drawn on the GPU path, groups left to the CPU loop.
    var tablesBuilt = 0
    var viewsDrawn = 0
    var lastFallbackGroups = 0
    var lastRecordCount = 0
}
