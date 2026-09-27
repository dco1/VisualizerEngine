import Foundation
import Metal
import OSLog
import VisualizerCore
import simd

// ── MEMBRANE WAVE MESH ───────────────────────────────────────────────────────
//
// A fluid-filled membrane on the GPU: a damped 2-D wave equation solved over a mesh's
// own rows × cols vertex grid (MembraneWave.metal), each vertex displaced along its rest
// normal by its wave height, with normals rebuilt from the displaced neighbours. The
// output streams feed `IlluminatoramaRenderer.registerGPUMesh(descriptor)` exactly like
// DynamicMesh / LinearBlendSkinMesh — no CPU geometry, no readback of the surface.
//
// Use it for anything that should wobble like a waterbed, a jelly, a drum skin: hosts
// poke it with `impulse(row:col:radius:velocity:)` and/or a uniform inertial force, and
// the waves propagate at `waveSpeed`, reflect off pinned vertices (mask 0) and decay
// with `damping`.
//
// The grid must be the FIRST rows × cols vertices of the mesh, row-major (the way a
// lofted surface is usually emitted); any vertices after it (caps, poles) stay at rest.

@MainActor
public final class MembraneWaveMesh {
    private static let log = Logger(subsystem: AppLog.subsystem, category: "MembraneWaveMesh")

    struct Params {
        var rows: UInt32, cols: UInt32, wrapCols: UInt32, impulseCount: UInt32
        var dt: Float, c2: Float, damping: Float, uniformForce: Float
    }
    public struct Impulse {
        public var gridPos: SIMD2<Float>
        public var radius: Float
        public var velocity: Float
        public init(row: Float, col: Float, radius: Float, velocity: Float) {
            gridPos = SIMD2(row, col); self.radius = radius; self.velocity = velocity
        }
    }

    public let rows: Int
    public let cols: Int
    public let wrapCols: Bool
    public let vertexCount: Int
    public let descriptor: IlluminatoramaGPUMeshDescriptor

    /// Wave speed in mesh units per second, damping (1/s), and a uniform acceleration applied
    /// to every free vertex this frame (e.g. the host body's inertial jolt).
    public var waveSpeed: Float = 50
    public var damping: Float = 1.2
    public var uniformForce: Float = 0

    private let step: MTLComputePipelineState
    private let displace: MTLComputePipelineState
    private let restPos: MTLBuffer, restNrm: MTLBuffer
    private let mask: MTLBuffer, invSpace: MTLBuffer
    private let v: MTLBuffer
    private var h: [MTLBuffer]
    private let outPos: MTLBuffer, outNrm: MTLBuffer
    private var pending: [Impulse] = []
    /// A settled membrane skips its dispatches entirely.
    private var quietTime: Float = 10
    private var current = 0
    /// Finest free grid spacing (mesh units) — sets the CFL-safe substep count.
    private let minSpacing: Float
    private static let maxImpulses = 16

    public init?(engine: SimEngine, rows: Int, cols: Int, wrapCols: Bool,
                 positions: [SIMD3<Float>], normals: [SIMD3<Float>],
                 uvs: [SIMD2<Float>]? = nil, colors: [SIMD4<Float>]? = nil,
                 indices: [UInt32], mask maskValues: [Float],
                 label: String = "MembraneWaveMesh") {
        let n = positions.count
        guard rows > 1, cols > 1, rows * cols <= n, normals.count == n, maskValues.count == rows * cols,
              let st = engine.pipeline("membrane_step"), let ds = engine.pipeline("membrane_displace") else {
            Self.log.error("\(label): invalid input or missing membrane pipelines")
            return nil
        }
        let device = engine.device
        func packed(_ v: [SIMD3<Float>]) -> [Float] { v.flatMap { [$0.x, $0.y, $0.z] } }
        // Per-vertex inverse squared rest spacing to the row / column neighbours.
        var inv = [SIMD2<Float>](repeating: SIMD2(1, 1), count: rows * cols)
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                let i = r * cols + c
                let ru = min(rows - 1, r + 1), rd = max(0, r - 1)
                let cr = c + 1 < cols ? c + 1 : (wrapCols ? 0 : c)
                let cl = c > 0 ? c - 1 : (wrapCols ? cols - 1 : c)
                let dr = simd_distance(positions[ru * cols + c], positions[rd * cols + c]) / Float(max(1, ru - rd))
                let span = (cr == c || cl == c) ? 1 : 2
                let dc = simd_distance(positions[r * cols + cr], positions[r * cols + cl]) / Float(span)
                inv[i] = SIMD2(1 / max(1e-3, dr * dr), 1 / max(1e-3, dc * dc))
            }
        }
        let pp = packed(positions), nn = packed(normals)
        let zeros = [Float](repeating: 0, count: rows * cols)
        guard let rp = device.makeBuffer(bytes: pp, length: pp.count * 4, options: .storageModeShared),
              let rn = device.makeBuffer(bytes: nn, length: nn.count * 4, options: .storageModeShared),
              let mk = device.makeBuffer(bytes: maskValues, length: maskValues.count * 4, options: .storageModeShared),
              let isb = device.makeBuffer(bytes: inv, length: inv.count * MemoryLayout<SIMD2<Float>>.stride,
                                          options: .storageModeShared),
              let vb = device.makeBuffer(bytes: zeros, length: zeros.count * 4, options: .storageModeShared),
              // Heights are shared (not private) so a host can sample one vertex for contact —
              // e.g. something standing on the membrane rides it (`height(row:col:)`).
              let h0 = device.makeBuffer(bytes: zeros, length: zeros.count * 4, options: .storageModeShared),
              let h1 = device.makeBuffer(bytes: zeros, length: zeros.count * 4, options: .storageModeShared),
              let op = device.makeBuffer(bytes: pp, length: pp.count * 4, options: .storageModeShared),
              let on = device.makeBuffer(bytes: nn, length: nn.count * 4, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: indices, length: indices.count * 4, options: .storageModeShared)
        else { return nil }
        op.label = "\(label).pos"; on.label = "\(label).nrm"
        var uvBuf: MTLBuffer? = nil
        if let uvs, uvs.count == n {
            let flat = uvs.flatMap { [$0.x, $0.y] }
            uvBuf = device.makeBuffer(bytes: flat, length: flat.count * 4, options: .storageModeShared)
        }
        var colorBuf: MTLBuffer? = nil
        if let colors, colors.count == n {
            colorBuf = device.makeBuffer(bytes: colors, length: n * MemoryLayout<SIMD4<Float>>.stride,
                                         options: .storageModeShared)
        }
        var maxInv: Float = 1e-6
        for i in 0 ..< rows * cols where maskValues[i] > 0 { maxInv = max(maxInv, max(inv[i].x, inv[i].y)) }
        self.minSpacing = 1 / maxInv.squareRoot()
        self.rows = rows; self.cols = cols; self.wrapCols = wrapCols; self.vertexCount = n
        self.step = st; self.displace = ds
        self.restPos = rp; self.restNrm = rn; self.mask = mk; self.invSpace = isb
        self.v = vb; self.h = [h0, h1]; self.outPos = op; self.outNrm = on
        self.descriptor = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: op, normalBuffer: on, vertexCount: n,
            bodyIndexBuffer: ib, bodyIndexCount: indices.count, bodyIndexType: .uint32,
            uvBuffer: uvBuf, colorBuffer: colorBuf)
    }

    /// Wave height at one grid vertex, from the most recently encoded frame. A single-float
    /// read for contact (a body resting on the membrane rides it) — never the surface.
    public func height(row: Int, col: Int) -> Float {
        guard row >= 0, row < rows, col >= 0, col < cols else { return 0 }
        return h[current].contents().bindMemory(to: Float.self, capacity: rows * cols)[row * cols + col]   // budget-ok: one-float contact probe
    }

    /// Queue a poke for the next `encode`: a Gaussian Δv (mesh units / s) at a grid position.
    public func impulse(_ i: Impulse) {
        if pending.count < Self.maxImpulses { pending.append(i) }
        quietTime = 0
    }

    /// Advance the membrane by `dt` (split into CFL-safe substeps) and write the displaced
    /// surface. Encodes into an open compute encoder; the caller ends it.
    public func encode(into enc: MTLComputeCommandEncoder, dt rawDt: Float) {
        // One encode advances at most 1/20 s: a stalled or batched host (headless warm-up, a
        // hitch) must not hand the explicit solver a step its CFL can't take — the surplus is
        // dropped, the way a real membrane can't "catch up" either.
        let dt = min(rawDt, 1.0 / 20)
        if abs(uniformForce) > 1e-4 { quietTime = 0 }
        quietTime += dt
        // After ~8 damping time-constants with no input the surface is at rest: skip entirely.
        if quietTime > 8 / max(0.05, damping) + 0.5 { pending.removeAll(); return }
        // Explicit 2-D wave step is stable for c·h/Δx < 1/√2; keep a margin on the finest cell.
        let subs = max(1, min(32, Int(ceil(dt * waveSpeed / (0.5 * minSpacing)))))
        let h = dt / Float(subs)
        var imps = pending
        pending.removeAll()
        var params = Params(rows: UInt32(rows), cols: UInt32(cols), wrapCols: wrapCols ? 1 : 0,
                            impulseCount: 0, dt: h, c2: waveSpeed * waveSpeed,
                            damping: damping, uniformForce: uniformForce)
        let tg = MTLSize(width: min(step.maxTotalThreadsPerThreadgroup, 128), height: 1, depth: 1)
        // Invariant bindings once per encode (the validation layer rejects redundant re-binds);
        // per substep only the ping-pong heights and the params change.
        enc.setComputePipelineState(step)
        enc.setBuffer(v, offset: 0, index: 1)
        enc.setBuffer(mask, offset: 0, index: 2)
        enc.setBuffer(invSpace, offset: 0, index: 3)
        if imps.isEmpty { imps = [Impulse(row: 0, col: 0, radius: 1, velocity: 0)] }
        imps.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 5) }
        for s in 0 ..< subs {
            params.impulseCount = s == 0 && imps[0].velocity != 0 ? UInt32(imps.count) : 0
            enc.setBuffer(self.h[current], offset: 0, index: 0)
            enc.setBuffer(self.h[1 - current], offset: 0, index: 6)
            enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 4)
            enc.dispatchThreads(MTLSize(width: rows * cols, height: 1, depth: 1), threadsPerThreadgroup: tg)
            current = 1 - current
        }
        enc.setComputePipelineState(displace)
        enc.setBuffer(restPos, offset: 0, index: 0)
        enc.setBuffer(restNrm, offset: 0, index: 1)
        enc.setBuffer(self.h[current], offset: 0, index: 2)
        enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
        var total = UInt32(vertexCount)
        enc.setBytes(&total, length: 4, index: 4)
        enc.setBuffer(outPos, offset: 0, index: 5)
        enc.setBuffer(outNrm, offset: 0, index: 6)
        enc.dispatchThreads(MTLSize(width: vertexCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
    }
}
