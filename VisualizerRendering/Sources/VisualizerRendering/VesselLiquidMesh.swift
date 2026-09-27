import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore
import simd

// ── VESSEL LIQUID MESH ───────────────────────────────────────────────────────
//
// The liquid in a vessel of revolution (a coffee pot, a jug, a glass) with a MOVING free
// surface — written on the GPU (`vesselLiquidMesh`, VesselLiquid.metal) in the vessel's own
// frame: the free surface out to where it meets the wall, the wall (the vessel's inner surface)
// up to that contact line, and the floor.
//
// The surface is `level + tilt·(x,z) + (mode2·(x,z))·(ρ² − ½)`: the first antisymmetric
// sloshing mode as a tilt and the second as a wobble — both antisymmetric, so the volume holds
// — whose amplitudes come from the HOST's slosh dynamics (a damped oscillator per mode driven by
// the vessel's acceleration and tilt: the standard equivalent-mechanical sloshing model). The
// kernel only turns those few numbers into vertices: per azimuth it solves where the surface
// meets the wall (the fixed point y = η(r(y)·dir) against the inner profile), so the liquid
// always fills the vessel exactly to its own surface. One thread per vertex, packed position +
// normal; registered with Illuminatorama as a GPU mesh (one draw, per-vertex motion vectors);
// a SceneKit twin reads the same buffers (the AssetLab previews).
//
// SIMULATED surface: `encode(density:)` builds the same mesh from a real liquid simulation
// instead — the free surface where an MLS-MPM density field (the solver's grid mass, simulated
// in the vessel's frame against this same inner profile via `MLSMPMSolver.vessel`) falls through
// half its rest density, each azimuth's contact line solved against the wall exactly as above.
@MainActor
public final class VesselLiquidMesh {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "VesselLiquidMesh")

    /// The free surface this frame (vessel frame: +y up the vessel's axis).
    public struct Surface: Sendable {
        /// Height of the surface on the axis.
        public var level: Float
        /// First mode: ∂η/∂x, ∂η/∂z.
        public var tilt: SIMD2<Float> = .zero
        /// Second mode amplitude along x / z (height at the wall ≈ ½·amplitude·R).
        public var mode2: SIMD2<Float> = .zero
        public init(level: Float, tilt: SIMD2<Float> = .zero, mode2: SIMD2<Float> = .zero) {
            self.level = level; self.tilt = tilt; self.mode2 = mode2
        }
    }

    private struct Params {
        var surface: SIMD4<Float>
        var mode2: SIMD4<Float>
        var counts: SIMD4<Float>
        var vertexCount: UInt32
        var pad0: UInt32 = 0, pad1: UInt32 = 0, pad2: UInt32 = 0
    }

    public let segments: Int
    public let surfaceRings: Int
    public let wallRings: Int
    public let vertexCount: Int
    public let positionBuffer: MTLBuffer
    public let normalBuffer: MTLBuffer
    public let indexBuffer: MTLBuffer
    public let indexCount: Int
    /// The inner profile, resampled uniformly in y (≤ 128 samples, inline per dispatch).
    public let radii: [Float]
    public let profileY0: Float
    public let profileDY: Float
    /// Liquid floor height (the vessel's inner bottom).
    public let floorY: Float

    private let pipeline: MTLComputePipelineState
    private let densityPipeline: MTLComputePipelineState?
    private let contactPipeline: MTLComputePipelineState?
    private let blurPipeline: MTLComputePipelineState?
    /// Per-azimuth contact line (height, wall radius) — pass 1 of `encode(density:)`.
    private var contactBuffer: MTLBuffer?
    private var blurA: MTLBuffer?
    private var blurB: MTLBuffer?

    /// A simulated liquid's density on a node grid, in the vessel's frame (+y up its axis, the
    /// axis at x = z = 0) — e.g. `MLSMPMSolver.gridMassBuffer` with its bounds and cell size.
    public struct DensityGrid {
        public var buffer: MTLBuffer
        /// Position of node (0, 0, 0).
        public var origin: SIMD3<Float>
        public var cellSize: Float
        public var resolution: SIMD3<UInt32>
        /// The density the surface is drawn at (half the liquid's rest density).
        public var iso: Float
        public init(buffer: MTLBuffer, origin: SIMD3<Float>, cellSize: Float, resolution: SIMD3<UInt32>, iso: Float) {
            self.buffer = buffer; self.origin = origin; self.cellSize = cellSize
            self.resolution = resolution; self.iso = iso
        }
    }

    /// The simulated liquid's gap from the glass, in nodes — fitted so a resting surface runs
    /// flat to the wall (VesselLiquidMeshTests: 0 drooped the rim 0.8 cm, 0.3 curled it up).
    static let standoffFraction: Float = 0.1

    private struct GridParams {
        var origin: SIMD4<Float>
        var res: SIMD4<UInt32>
        var extra: SIMD4<Float>
        var profile: SIMD4<Float>
        var extra2: SIMD4<Float>
    }

    /// `innerProfile` = (y, r) points of the vessel's inner surface, bottom to top (y rising).
    /// `floorY` = the inner bottom the liquid sits on.
    public init?(engine: SimEngine = .shared, innerProfile: [SIMD2<Float>], floorY: Float,
                 segments: Int = 48, surfaceRings: Int = 10, wallRings: Int = 14, samples: Int = 96) {
        guard innerProfile.count >= 2, segments >= 8, surfaceRings >= 2, wallRings >= 2,
              samples >= 2, samples <= 128 else {
            Self.log.error("VesselLiquidMesh: bad shape")
            return nil
        }
        guard let pso = engine.pipelineCache.pipelineState(name: "vesselLiquidMesh", device: engine.device) else {
            Self.log.error("vesselLiquidMesh pipeline missing — check VesselLiquid.metal is in VisualizerRendering/Shaders/")
            return nil
        }
        self.pipeline = pso
        self.densityPipeline = engine.pipelineCache.pipelineState(name: "vesselLiquidFromDensity", device: engine.device)
        self.contactPipeline = engine.pipelineCache.pipelineState(name: "vesselLiquidContact", device: engine.device)
        self.blurPipeline = engine.pipelineCache.pipelineState(name: "vesselDensityBlur", device: engine.device)
        self.segments = segments
        self.surfaceRings = surfaceRings
        self.wallRings = wallRings
        self.floorY = floorY

        // Resample the profile uniformly in y (the kernel interpolates a table).
        let ys = innerProfile.map(\.x)
        let y0 = ys.first!, y1 = ys.last!
        let dy = (y1 - y0) / Float(samples - 1)
        var rs: [Float] = []
        var k = 0
        for s in 0 ..< samples {
            let y = y0 + dy * Float(s)
            while k < innerProfile.count - 2 && innerProfile[k + 1].x < y { k += 1 }
            let a = innerProfile[k], b = innerProfile[k + 1]
            let t = max(0, min(1, (y - a.x) / max(b.x - a.x, 1e-6)))
            rs.append(a.y + (b.y - a.y) * t)
        }
        self.radii = rs
        self.profileY0 = y0
        self.profileDY = dy

        let surf = 1 + surfaceRings * segments
        let wall = wallRings * segments
        let floor = 1 + segments
        let verts = surf + wall + floor
        self.vertexCount = verts
        let dev = engine.device
        guard let pos = dev.makeBuffer(length: verts * 12, options: .storageModeShared),
              let nrm = dev.makeBuffer(length: verts * 12, options: .storageModeShared) else { return nil }
        pos.label = "VesselLiquidMesh.position"
        nrm.label = "VesselLiquidMesh.normal"
        memset(pos.contents(), 0, pos.length)
        memset(nrm.contents(), 0, nrm.length)
        self.positionBuffer = pos
        self.normalBuffer = nrm

        // Indices — counter-clockwise seen from outside the liquid (surface up, wall out,
        // floor down). Azimuth runs +x → +z, which is clockwise seen from above.
        var idx: [UInt32] = []
        let S = UInt32(segments)
        func sv(_ ring: Int, _ j: Int) -> UInt32 { 1 + UInt32(ring) * S + UInt32(j % segments) }
        for j in 0 ..< segments { idx += [0, sv(0, j + 1), sv(0, j)] }
        for r in 0 ..< surfaceRings - 1 {
            for j in 0 ..< segments {
                let a = sv(r, j), b = sv(r, j + 1), c = sv(r + 1, j), d = sv(r + 1, j + 1)
                idx += [a, b, c, b, d, c]
            }
        }
        let wb = UInt32(surf)
        func wv(_ ring: Int, _ j: Int) -> UInt32 { wb + UInt32(ring) * S + UInt32(j % segments) }
        for r in 0 ..< wallRings - 1 {
            for j in 0 ..< segments {
                let a = wv(r, j), b = wv(r, j + 1), c = wv(r + 1, j), d = wv(r + 1, j + 1)
                idx += [a, c, b, b, c, d]
            }
        }
        let fc = UInt32(surf + wall)
        for j in 0 ..< segments { idx += [fc, fc + 1 + UInt32(j), fc + 1 + UInt32((j + 1) % segments)] }
        guard let ib = idx.withUnsafeBufferPointer({
            dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count * 4, options: .storageModeShared)
        }) else { return nil }
        ib.label = "VesselLiquidMesh.index"
        self.indexBuffer = ib
        self.indexCount = idx.count
    }

    /// For `IlluminatoramaRenderer.registerGPUMesh`.
    public var descriptor: IlluminatoramaGPUMeshDescriptor {
        IlluminatoramaGPUMeshDescriptor(positionBuffer: positionBuffer, normalBuffer: normalBuffer,
                                        positionStride: 12, normalStride: 12, vertexCount: vertexCount,
                                        bodyIndexBuffer: indexBuffer, bodyIndexCount: indexCount,
                                        bodyIndexType: .uint32)
    }

    /// Write the liquid for `surface`. Encode before the frame that draws it, on the same queue.
    public func encode(_ surface: Surface, into cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "vesselLiquidMesh"
        var p = Params(surface: SIMD4(surface.level, surface.tilt.x, surface.tilt.y, floorY),
                       mode2: SIMD4(surface.mode2.x, surface.mode2.y, profileY0, profileDY),
                       counts: SIMD4(Float(radii.count), Float(segments), Float(surfaceRings), Float(wallRings)),
                       vertexCount: UInt32(vertexCount))
        enc.setComputePipelineState(pipeline)
        radii.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        enc.setBuffer(positionBuffer, offset: 0, index: 1)
        enc.setBuffer(normalBuffer, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
        let w = min(pipeline.threadExecutionWidth, 64)
        enc.dispatchThreads(MTLSize(width: vertexCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Write the liquid with the free surface of a SIMULATED liquid (`grid`, in this vessel's
    /// frame), after `smoothing` [1 2 1] blur passes over its density (each pass one node wide
    /// per axis — below the surface's own detail, above the particle ripple; two by default, and
    /// slopes taken over three nodes: a resting surface's normals then scatter ≈ 1°, where one
    /// pass over 1.5 nodes left ≈ 2.4° — a hammered sky reflection on a mirror-dark liquid).
    /// Encode after the simulation's step and before the frame that draws it, on the same queue.
    public func encode(density grid: DensityGrid, smoothing: Int = 2, into cb: MTLCommandBuffer) {
        guard let densityPipeline, let contactPipeline else { return }
        if contactBuffer == nil {
            contactBuffer = cb.device.makeBuffer(length: segments * MemoryLayout<SIMD2<Float>>.stride,
                                                 options: .storageModePrivate)
            contactBuffer?.label = "VesselLiquidMesh.contact"
        }
        guard let contactBuffer else { return }
        let nodes = Int(grid.resolution.x) * Int(grid.resolution.y) * Int(grid.resolution.z)
        // The read's total smoothing: P2G's B-spline (variance ¼ node²), each blur pass (½),
        // the B-spline read (¼) — the σ the glass-fullness correction divides out.
        let sigma = grid.cellSize * (0.5 + 0.5 * Float(max(0, smoothing))).squareRoot()
        var g = GridParams(origin: SIMD4(grid.origin, grid.cellSize), res: SIMD4(grid.resolution, 0),
                           extra: SIMD4(grid.iso, 0.6 * grid.cellSize, 1.5 * grid.cellSize, sigma),
                           profile: SIMD4(profileY0, profileDY, Float(radii.count), floorY),
                           extra2: SIMD4(Self.standoffFraction * grid.cellSize, 0, 0, 0))
        var source = grid.buffer
        if smoothing > 0, let blurPipeline {
            if blurA == nil || blurA!.length < nodes * 4 {
                blurA = cb.device.makeBuffer(length: nodes * 4, options: .storageModePrivate)
                blurB = cb.device.makeBuffer(length: nodes * 4, options: .storageModePrivate)
                blurA?.label = "VesselLiquidMesh.blurA"
                blurB?.label = "VesselLiquidMesh.blurB"
            }
            if let a = blurA, let b = blurB, let enc = cb.makeComputeCommandEncoder() {
                enc.label = "vesselDensityBlur"
                enc.setComputePipelineState(blurPipeline)
                enc.setBytes(&g, length: MemoryLayout<GridParams>.stride, index: 2)
                let size = MTLSize(width: Int(grid.resolution.x), height: Int(grid.resolution.y),
                                   depth: Int(grid.resolution.z))
                let tile = MTLSize(width: 4, height: 4, depth: 4)
                var src = grid.buffer
                for _ in 0 ..< smoothing {
                    for axis in 0 ..< 3 {
                        let dst = (src === a) ? b : a
                        var ax = UInt32(axis)
                        enc.setBuffer(src, offset: 0, index: 0)
                        enc.setBuffer(dst, offset: 0, index: 1)
                        enc.setBytes(&ax, length: 4, index: 3)
                        enc.dispatchThreads(size, threadsPerThreadgroup: tile)
                        enc.memoryBarrier(scope: .buffers)
                        src = dst
                    }
                }
                enc.endEncoding()
                source = src
            }
        }
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "vesselLiquidFromDensity"
        var p = Params(surface: SIMD4(0, 0, 0, floorY),
                       mode2: SIMD4(0, 0, profileY0, profileDY),
                       counts: SIMD4(Float(radii.count), Float(segments), Float(surfaceRings), Float(wallRings)),
                       vertexCount: UInt32(vertexCount))
        // Pass 1: the contact line per azimuth.
        enc.setComputePipelineState(contactPipeline)
        radii.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        enc.setBuffer(contactBuffer, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
        enc.setBuffer(source, offset: 0, index: 4)
        enc.setBytes(&g, length: MemoryLayout<GridParams>.stride, index: 5)
        enc.dispatchThreads(MTLSize(width: segments, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(segments, 64), height: 1, depth: 1))
        // Pass 2: every vertex (serial encoder: pass 1's writes are visible).
        enc.setComputePipelineState(densityPipeline)
        enc.setBuffer(positionBuffer, offset: 0, index: 1)
        enc.setBuffer(normalBuffer, offset: 0, index: 2)
        enc.setBuffer(contactBuffer, offset: 0, index: 6)
        let w = min(densityPipeline.threadExecutionWidth, 64)
        enc.dispatchThreads(MTLSize(width: vertexCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// A SceneKit geometry reading the mesh's own buffers (previews).
    public func makeSceneKitGeometry() -> SCNGeometry {
        let pos = SCNGeometrySource(buffer: positionBuffer, vertexFormat: .float3, semantic: .vertex,
                                    vertexCount: vertexCount, dataOffset: 0, dataStride: 12)
        let nrm = SCNGeometrySource(buffer: normalBuffer, vertexFormat: .float3, semantic: .normal,
                                    vertexCount: vertexCount, dataOffset: 0, dataStride: 12)
        let element = SCNGeometryElement(buffer: indexBuffer, primitiveType: .triangles,
                                         primitiveCount: indexCount / 3, bytesPerIndex: 4)
        return SCNGeometry(sources: [pos, nrm], elements: [element]) // winding-ok: GPU-written normals; the index pattern winds CCW outside (VesselLiquidMeshTests)
    }
}
