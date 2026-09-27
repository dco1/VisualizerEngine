import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore
import simd

// ── ARC TUBE BATCH ───────────────────────────────────────────────────────────
//
// Many "rubber-hose" limbs — tubes of CONSTANT length that bend as one smooth circular arc
// between two points (a cartoon arm or leg) — swept on the GPU into ONE shared mesh.
//
// The host supplies a handful of numbers per tube (the arc: start, chord and bow
// directions, length, half-turn, centre, radius — 64 bytes) and a compute kernel
// (`arcTubeSweep`, ArcTube.metal) writes every vertex from the closed form, one thread per
// vertex, into packed position/normal buffers. Registered with Illuminatorama as a GPU
// mesh, the whole batch is one draw (G-buffer and every shadow cascade), and the
// renderer's previous-position capture gives each vertex a true motion vector. A SceneKit
// twin reads the same buffers (the AssetLab previews).
//
// Arcs travel as inline command-buffer bytes (`setBytes`, ≤ 4 KB → `maxTubes` ≤ 64), so a
// frame's upload can never race the GPU reading the previous one. The output buffers are
// hazard-tracked: a sweep waits for the frame before it to finish reading them.
//
// Slots beyond this dispatch's tube count collapse to zero-area triangles far below the
// world (the draw covers every slot, always — a fixed topology).
@MainActor
public final class ArcTubeBatch {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "ArcTubeBatch")

    /// One tube's centreline: a circular arc of length `chord.w` from `start` (or a straight
    /// run along `chord` when the half-turn `bow.w` is 0). Mirror of `ArcTubeArc`.
    public struct Arc: Sendable {
        /// xyz = root point, w = tube radius.
        public var start: SIMD4<Float>
        /// xyz = unit chord direction (root → tip), w = arc length.
        public var chord: SIMD4<Float>
        /// xyz = unit bow direction (across the chord, toward the bend), w = half the angle
        /// the arc turns through (0 = straight).
        public var bow: SIMD4<Float>
        /// xyz = the arc's centre, w = its radius.
        public var centre: SIMD4<Float>

        public init(start: SIMD3<Float>, chordDirection e: SIMD3<Float>, bowDirection u: SIMD3<Float>,
                    length: Float, halfTurn theta: Float, centre: SIMD3<Float>, arcRadius: Float,
                    tubeRadius: Float) {
            self.start = SIMD4(start, tubeRadius)
            self.chord = SIMD4(e, length)
            self.bow = SIMD4(u, theta)
            self.centre = SIMD4(centre, arcRadius)
        }
    }

    private struct Params {
        var tubeCount: UInt32
        var maxTubes: UInt32
        var rings: UInt32
        var radial: UInt32
    }

    public let maxTubes: Int
    public let rings: Int
    public let radial: Int
    public var vertsPerTube: Int { rings * radial }
    public var vertexCount: Int { maxTubes * vertsPerTube }

    /// packed_float3 (12-byte stride) — written by the kernel.
    public let positionBuffer: MTLBuffer
    public let normalBuffer: MTLBuffer
    /// uint32, every slot's quads (counter-clockwise seen from outside).
    public let indexBuffer: MTLBuffer
    public let indexCount: Int

    private let pipeline: MTLComputePipelineState

    /// `rings` along each tube × `radial` round it. `maxTubes` ≤ 64 (inline arc upload).
    public init?(engine: SimEngine = .shared, maxTubes: Int, rings: Int, radial: Int) {
        guard maxTubes > 0, maxTubes * MemoryLayout<Arc>.stride <= 4096, rings >= 2, radial >= 3 else {
            Self.log.error("ArcTubeBatch: bad shape (tubes \(maxTubes), rings \(rings), radial \(radial))")
            return nil
        }
        guard let pso = engine.pipelineCache.pipelineState(name: "arcTubeSweep", device: engine.device) else {
            Self.log.error("arcTubeSweep pipeline missing — check ArcTube.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        self.pipeline = pso
        self.maxTubes = maxTubes
        self.rings = rings
        self.radial = radial
        let verts = maxTubes * rings * radial
        let dev = engine.device
        guard let pos = dev.makeBuffer(length: verts * 12, options: .storageModeShared),
              let nrm = dev.makeBuffer(length: verts * 12, options: .storageModeShared) else {
            Self.log.error("ArcTubeBatch buffer alloc failed (\(verts) vertices)")
            return nil
        }
        pos.label = "ArcTubeBatch.position"
        nrm.label = "ArcTubeBatch.normal"
        // Every slot starts collapsed (a fixed topology draws them all).
        let pp = pos.contents().bindMemory(to: Float.self, capacity: verts * 3)
        let np = nrm.contents().bindMemory(to: Float.self, capacity: verts * 3)
        for v in 0 ..< verts {
            pp[v * 3] = 0; pp[v * 3 + 1] = -100_000; pp[v * 3 + 2] = 0
            np[v * 3] = 0; np[v * 3 + 1] = 1; np[v * 3 + 2] = 0
        }
        self.positionBuffer = pos
        self.normalBuffer = nrm

        var idx: [UInt32] = []
        idx.reserveCapacity(maxTubes * (rings - 1) * radial * 6)
        for s in 0 ..< maxTubes {
            let base = UInt32(s * rings * radial)
            for i in 0 ..< rings - 1 {
                for j in 0 ..< radial {
                    let j1 = (j + 1) % radial
                    let a = base + UInt32(i * radial + j), b = base + UInt32(i * radial + j1)
                    let c = base + UInt32((i + 1) * radial + j), d = base + UInt32((i + 1) * radial + j1)
                    idx += [a, b, c, b, d, c]
                }
            }
        }
        guard let ib = idx.withUnsafeBufferPointer({
            dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count * 4, options: .storageModeShared)
        }) else {
            Self.log.error("ArcTubeBatch index buffer alloc failed")
            return nil
        }
        ib.label = "ArcTubeBatch.index"
        self.indexBuffer = ib
        self.indexCount = idx.count
    }

    /// For `IlluminatoramaRenderer.registerGPUMesh` — the whole batch as one mesh.
    public var descriptor: IlluminatoramaGPUMeshDescriptor {
        IlluminatoramaGPUMeshDescriptor(positionBuffer: positionBuffer, normalBuffer: normalBuffer,
                                        positionStride: 12, normalStride: 12, vertexCount: vertexCount,
                                        bodyIndexBuffer: indexBuffer, bodyIndexCount: indexCount,
                                        bodyIndexType: .uint32)
    }

    /// Sweep `arcs` (the first `maxTubes`; the remaining slots collapse) into the batch.
    /// Encode before the frame that draws it, on the same queue.
    public func encode(_ arcs: [Arc], into cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "arcTubeSweep"
        let n = min(arcs.count, maxTubes)
        var params = Params(tubeCount: UInt32(n), maxTubes: UInt32(maxTubes), rings: UInt32(rings),
                            radial: UInt32(radial))
        enc.setComputePipelineState(pipeline)
        if n > 0 {
            arcs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: n * MemoryLayout<Arc>.stride, index: 0) }
        } else {
            var dummy = Arc(start: .zero, chordDirection: SIMD3(0, 1, 0), bowDirection: SIMD3(1, 0, 0),
                            length: 1, halfTurn: 0, centre: .zero, arcRadius: 0, tubeRadius: 0)
            enc.setBytes(&dummy, length: MemoryLayout<Arc>.stride, index: 0)
        }
        enc.setBuffer(positionBuffer, offset: 0, index: 1)
        enc.setBuffer(normalBuffer, offset: 0, index: 2)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
        let w = min(pipeline.threadExecutionWidth, 64)
        enc.dispatchThreads(MTLSize(width: vertexCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// A SceneKit geometry reading the batch's own buffers (for previews; SceneKit samples
    /// them in place, so a sweep committed before it draws shows up without a copy).
    public func makeSceneKitGeometry() -> SCNGeometry {
        let pos = SCNGeometrySource(buffer: positionBuffer, vertexFormat: .float3, semantic: .vertex,
                                    vertexCount: vertexCount, dataOffset: 0, dataStride: 12)
        let nrm = SCNGeometrySource(buffer: normalBuffer, vertexFormat: .float3, semantic: .normal,
                                    vertexCount: vertexCount, dataOffset: 0, dataStride: 12)
        let element = SCNGeometryElement(buffer: indexBuffer, primitiveType: .triangles,
                                         primitiveCount: indexCount / 3, bytesPerIndex: 4)
        return SCNGeometry(sources: [pos, nrm], elements: [element]) // winding-ok: GPU-written normals; the kernel's frame is right-handed and the index pattern winds CCW outside
    }
}
