//  Modifier+GPU.swift — GPU evaluation of the modifier stack (modelling kit, item M2).
//
//  API (nothing existing changes): `ModifierBackend` (.cpu / .gpu / .auto), `Modifier.apply(_:backend:)`,
//  `ModifierStack.evaluate(_:backend:upTo:)` and `ModifierGPUSession` (the base mesh uploaded once, re-evaluated for
//  every parameter edit). The CPU modifiers stay as oracle and fallback.
//
//  What runs on the GPU: welded-vertex normals (gather over each welded vertex's incident triangles), Displace with
//  the procedural fBm (`NoiseDisplaceModifier`), SimpleDeform (bend/twist/taper/stretch), Lattice (trilinear and
//  cubic), Shrinkwrap (nearest surface, ray projection; BVH traversal in Metal, see Modifier+GPUBVH.swift), Solidify
//  (offsets, even-thickness scale, both layers and the rim), and the normal turning every vertex-moving modifier
//  does (`Mesh3.moved`). A run of consecutive GPU modifiers is ONE command buffer on buffers that stay resident:
//  no CPU round trip between them, the mesh is read back once at the end of the run.
//
//  What stays on the CPU, and why: closure-based `DisplaceModifier` (a Swift closure can't run in Metal; use
//  `NoiseDisplaceModifier`), Mirror / Array / Bevel / Subdivide / Boolean (topology-building wrappers over other
//  items; the evaluator runs them on the CPU between GPU runs). The per-topology tables (weld by 1 µm hash, the
//  vertex → triangle lists, Solidify's boundary edges) are built on the CPU once per mesh topology and cached in
//  the session: their GPU versions belong to items M1 (weld) and M4 (edge tables). Guarantees: Float precision
//  (about 1e-6 relative to the mesh's size), same triangle list as the CPU. Not guaranteed: bit equality with the
//  CPU, and Solidify skips the CPU's "drop zero-area rim/no-normal triangles" test (they come out zero-area).
//  `.gpu` with no Metal device runs everything on the CPU.

import Foundation
import simd

public enum ModifierBackend: Sendable { case cpu, gpu, auto }

/// What an evaluation actually ran where: each inner array is one resident GPU run (names in order).
public struct ModifierRunReport: Sendable {
    public var gpuRuns: [[String]] = []
    public var cpuModifiers: [String] = []
    public var gpuModifierCount: Int { gpuRuns.reduce(0) { $0 + $1.count } }
}

public extension Modifier {
    func apply(_ mesh: Mesh3, backend: ModifierBackend) -> Mesh3 {
        var s = ModifierStack(); s.add(self)
        return s.evaluate(mesh, backend: backend)
    }
}

public extension ModifierStack {
    func apply(_ mesh: Mesh3, backend: ModifierBackend) -> Mesh3 { evaluate(mesh, backend: backend) }

    func evaluate(_ base: Mesh3, backend: ModifierBackend, upTo: Int? = nil) -> Mesh3 {
        evaluateReport(base, backend: backend, upTo: upTo).mesh
    }

    func evaluateReport(_ base: Mesh3, backend: ModifierBackend, upTo: Int? = nil) -> (mesh: Mesh3, report: ModifierRunReport) {
        ModifierGPUSession(base: base, backend: backend).evaluateReport(self, upTo: upTo)
    }
}

/// The base mesh prepared once (topology tables, buffers) so editing a parameter and re-evaluating only runs kernels.
public final class ModifierGPUSession: @unchecked Sendable {
    public let base: Mesh3
    public let backend: ModifierBackend
    private var prepared: AnyObject?                         // GPUMeshState, built on first GPU run

    public init(base: Mesh3, backend: ModifierBackend = .auto) { self.base = base; self.backend = backend }

    public var gpuAvailable: Bool { ModifierGPUSession.deviceAvailable }

    public func evaluate(_ stack: ModifierStack, upTo: Int? = nil) -> Mesh3 { evaluateReport(stack, upTo: upTo).mesh }

    public func evaluateReport(_ stack: ModifierStack, upTo: Int? = nil) -> (mesh: Mesh3, report: ModifierRunReport) {
        var report = ModifierRunReport()
        var mesh = base
        var fromBase = true
        let useGPU = backend != .cpu && gpuAvailable && (backend == .gpu || base.positions.count >= 2048) && !base.isEmpty
        var pending: [GPUOperation] = []
        func flush() {
            guard !pending.isEmpty else { return }
            let names = pending.map { $0.name }
            if let out = runGPU(pending, on: mesh, fromBase: fromBase) {
                mesh = out; report.gpuRuns.append(names)
            } else {
                for op in pending { mesh = op.modifier.apply(mesh); report.cpuModifiers.append(op.name) }
            }
            pending = []; fromBase = false
        }
        for e in stack.entries.prefix(upTo ?? stack.entries.count) where e.isEnabled {
            if useGPU, let op = GPUOperation(e.modifier) { pending.append(op) } else {
                flush()
                mesh = e.modifier.apply(mesh); fromBase = false
                report.cpuModifiers.append(String(describing: type(of: e.modifier)))
            }
        }
        flush()
        return (mesh, report)
    }
}

#if canImport(Metal)
import Metal

extension ModifierGPUSession {
    static var deviceAvailable: Bool { ModifierGPU.shared != nil }

    fileprivate func runGPU(_ ops: [GPUOperation], on mesh: Mesh3, fromBase: Bool) -> Mesh3? {
        guard let gpu = ModifierGPU.shared else { return nil }
        let state: GPUMeshState
        if fromBase {
            if let s = prepared as? GPUMeshState { state = s } else { state = GPUMeshState(mesh: mesh, gpu: gpu); prepared = state }
        } else { state = GPUMeshState(mesh: mesh, gpu: gpu) }
        return gpu.run(ops, on: state)
    }
}

/// The device, queue and pipelines, made once. nil without a Metal device.
final class ModifierGPU: @unchecked Sendable {
    static let shared: ModifierGPU? = ModifierGPU()
    let device: MTLDevice, queue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue(),
              let lib = try? d.makeLibrary(source: ModifierKernels.common + "\n" + ModifierKernels.shrinkwrap, options: nil)
        else { return nil }
        device = d; queue = q
        for name in ["weld_normals", "solid_scale", "turn_normals", "displace_noise", "simple_deform", "lattice",
                     "solid_tris", "solid_rim", "shrinkwrap"] {
            guard let f = lib.makeFunction(name: name), let p = try? d.makeComputePipelineState(function: f) else { return nil }
            pipelines[name] = p
        }
    }

    func buffer<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes { raw in
            raw.count == 0 ? device.makeBuffer(length: 16, options: .storageModeShared)!
                           : device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)!
        }
    }
    func empty(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: Swift.max(16, bytes), options: .storageModeShared)! }

    func dispatch<P>(_ enc: MTLComputeCommandEncoder, _ name: String, _ n: Int, _ bufs: [MTLBuffer], _ params: P) {
        guard n > 0, let p = pipelines[name] else { return }
        enc.setComputePipelineState(p)
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        var pp = params
        enc.setBytes(&pp, length: MemoryLayout<P>.stride, index: bufs.count)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: Swift.min(64, p.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }

    /// Run `ops` back to back in one command buffer, buffers resident, then read the mesh back once.
    func run(_ ops: [GPUOperation], on start: GPUMeshState) -> Mesh3? {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var state = start
        for op in ops { state = encode(op, state, enc) }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { return nil }
        return state.readBack()
    }
}

// MARK: - Topology (CPU, once per mesh topology)

/// Welded ids by 1 µm position hash (as `WeldedMesh`), and the CSR tables the kernels gather over.
final class GPUTopology: @unchecked Sendable {
    let n: Int, nW: Int
    let cornerID: [UInt32]
    let rep: [UInt32]                    // welded id -> first mesh vertex with it
    let wtris: [UInt32]                  // welded triangles (3 ids each), degenerate-id ones left out
    let wtriSrc: [UInt32]                // welded triangle -> mesh triangle
    let wStart: [UInt32], wEnt: [UInt32] // welded vertex -> (welded tri · 3 + slot)
    let vStart: [UInt32], vTris: [UInt32]// mesh vertex -> mesh triangles, ascending
    let indices: [UInt32]
    private(set) lazy var rimEdges: [UInt32] = computeRim()
    private var cache: [String: MTLBuffer] = [:]

    static func weld(_ p: [Vec3]) -> (ids: [UInt32], count: Int) {
        var map: [SIMD3<Int64>: UInt32] = [:]; map.reserveCapacity(p.count)
        var ids = [UInt32](); ids.reserveCapacity(p.count)
        for q in p {
            let k = SIMD3(Int64((q.x * 1e6).rounded()), Int64((q.y * 1e6).rounded()), Int64((q.z * 1e6).rounded()))
            if let i = map[k] { ids.append(i) } else { let i = UInt32(map.count); map[k] = i; ids.append(i) }
        }
        return (ids, map.count)
    }

    init(cornerID: [UInt32], nW: Int, indices: [UInt32]) {
        n = cornerID.count; self.nW = nW; self.cornerID = cornerID; self.indices = indices
        var rep = [UInt32](repeating: 0, count: nW), seen = [Bool](repeating: false, count: nW)
        for (v, id) in cornerID.enumerated() where !seen[Int(id)] { seen[Int(id)] = true; rep[Int(id)] = UInt32(v) }
        self.rep = rep
        var wt: [UInt32] = [], src: [UInt32] = []
        var vCount = [UInt32](repeating: 0, count: n + 1), wCount = [UInt32](repeating: 0, count: nW + 1)
        var validTri: [Bool] = []
        let T = indices.count / 3
        validTri.reserveCapacity(T)
        for t in 0..<T {
            let i0 = Int(indices[3 * t]), i1 = Int(indices[3 * t + 1]), i2 = Int(indices[3 * t + 2])
            let ok = i0 < cornerID.count && i1 < cornerID.count && i2 < cornerID.count
            validTri.append(ok)
            guard ok else { continue }
            vCount[i0] += 1; vCount[i1] += 1; vCount[i2] += 1
            let a = cornerID[i0], b = cornerID[i1], c = cornerID[i2]
            guard a != b, b != c, a != c else { continue }
            wt += [a, b, c]; src.append(UInt32(t))
            wCount[Int(a)] += 1; wCount[Int(b)] += 1; wCount[Int(c)] += 1
        }
        wtris = wt; wtriSrc = src
        func scan(_ c: inout [UInt32]) -> [UInt32] {
            var out = [UInt32](repeating: 0, count: c.count); var s: UInt32 = 0
            for i in 0..<(c.count - 1) { out[i] = s; s += c[i] }
            out[c.count - 1] = s; return out
        }
        let vS = scan(&vCount), wS = scan(&wCount)
        var vFill = vS, wFill = wS
        var vt = [UInt32](repeating: 0, count: Int(vS.last!)), we = [UInt32](repeating: 0, count: Int(wS.last!))
        for t in 0..<T where validTri[t] {
            for k in 0..<3 { let v = Int(indices[3 * t + k]); vt[Int(vFill[v])] = UInt32(t); vFill[v] += 1 }
        }
        for j in 0..<src.count { for k in 0..<3 {
            let id = Int(wt[3 * j + k]); we[Int(wFill[id])] = UInt32(3 * j + k); wFill[id] += 1
        } }
        vStart = vS; vTris = vt; wStart = wS; wEnt = we
    }

    /// Boundary edges (welded ids a, b as sorted pairs) in the CPU Solidify's order: directed edges sorted by (a, b),
    /// used by exactly one face and whose undirected edge is used once.
    private func computeRim() -> [UInt32] {
        var directed: [UInt64: Int] = [:], undirected: [UInt64: Int] = [:]
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(a) << 32 | UInt64(b) }
        for j in 0..<(wtris.count / 3) {
            let t = [wtris[3 * j], wtris[3 * j + 1], wtris[3 * j + 2]]
            for k in 0..<3 {
                let p = t[k], q = t[(k + 1) % 3]
                directed[key(p, q), default: 0] += 1
                undirected[key(Swift.min(p, q), Swift.max(p, q)), default: 0] += 1
            }
        }
        var out: [UInt32] = []
        for k in directed.keys.sorted() where directed[k] == 1 {
            let a = UInt32(k >> 32), b = UInt32(k & 0xFFFF_FFFF)
            if undirected[key(Swift.min(a, b), Swift.max(a, b))] == 1 { out += [a, b] }
        }
        return out
    }

    func buf(_ name: String, _ gpu: ModifierGPU) -> MTLBuffer {
        if let b = cache[name] { return b }
        let b: MTLBuffer
        switch name {
        case "cid": b = gpu.buffer(cornerID)
        case "rep": b = gpu.buffer(rep)
        case "wtris": b = gpu.buffer(wtris)
        case "wsrc": b = gpu.buffer(wtriSrc)
        case "wStart": b = gpu.buffer(wStart)
        case "wEnt": b = gpu.buffer(wEnt)
        case "vStart": b = gpu.buffer(vStart)
        case "vTris": b = gpu.buffer(vTris)
        case "rim": b = gpu.buffer(rimEdges)
        default: b = gpu.buffer(indices)               // "idx"
        }
        cache[name] = b; return b
    }
}

// MARK: - Resident mesh

final class GPUMeshState {
    let n: Int
    let pos: MTLBuffer, nrm: MTLBuffer, uv: MTLBuffer, idx: MTLBuffer
    let hasN: Bool, hasUV: Bool
    private var topoMade: GPUTopology?
    private let topoMaker: (() -> GPUTopology)?
    /// Built on first use: a Solidify's output topology is only worth building if another modifier follows.
    var topo: GPUTopology {
        if let t = topoMade { return t }
        let t = topoMaker!(); topoMade = t; return t
    }
    let gpu: ModifierGPU
    /// CPU arrays handed back unchanged on read-back (nil after Solidify, whose normals and UVs are new).
    let keepNormals: [Vec3]?, keepUVs: [Vec2]?

    init(mesh m: Mesh3, gpu: ModifierGPU) {
        self.gpu = gpu; n = m.positions.count
        func f4(_ v: Vec3) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), 0) }
        pos = gpu.buffer(m.positions.map(f4))
        hasN = m.normals.count == n; hasUV = m.uvs.count == n
        nrm = hasN ? gpu.buffer(m.normals.map(f4)) : gpu.empty(16)
        uv = hasUV ? gpu.buffer(m.uvs.map { SIMD2<Float>(Float($0.x), Float($0.y)) }) : gpu.empty(8)
        idx = gpu.buffer(m.indices)
        let w = GPUTopology.weld(m.positions)
        topoMade = GPUTopology(cornerID: w.ids, nW: w.count, indices: m.indices); topoMaker = nil
        keepNormals = m.normals; keepUVs = m.uvs
        indicesCPU = m.indices
    }

    /// A state produced by a GPU op: new buffers, same or new topology.
    init(gpu: ModifierGPU, n: Int, pos: MTLBuffer, nrm: MTLBuffer, uv: MTLBuffer, idx: MTLBuffer, hasN: Bool, hasUV: Bool,
         topo: @escaping () -> GPUTopology, keepNormals: [Vec3]?, keepUVs: [Vec2]?, indices: [UInt32]) {
        self.gpu = gpu; self.n = n; self.pos = pos; self.nrm = nrm; self.uv = uv; self.idx = idx
        self.hasN = hasN; self.hasUV = hasUV; self.topoMaker = topo; self.topoMade = nil; self.keepNormals = keepNormals; self.keepUVs = keepUVs
        indicesCPU = indices
    }
    let indicesCPU: [UInt32]

    func readBack() -> Mesh3 {
        func read<T>(_ b: MTLBuffer, _ count: Int, _ t: T.Type) -> UnsafeBufferPointer<T> {
            UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: T.self), count: count)
        }
        let P = read(pos, n, SIMD4<Float>.self)
        var out = Mesh3(positions: P.map { Vec3(Double($0.x), Double($0.y), Double($0.z)) },
                        normals: keepNormals ?? [], uvs: keepUVs ?? [], indices: indicesCPU)
        if hasN {
            let N = read(nrm, n, SIMD4<Float>.self)
            out.normals = N.map { Vec3(Double($0.x), Double($0.y), Double($0.z)) }
        }
        if keepUVs == nil, hasUV {
            let U = read(uv, n, SIMD2<Float>.self)
            out.uvs = U.map { Vec2(Double($0.x), Double($0.y)) }
        }
        return out
    }
}
#else
extension ModifierGPUSession {
    static var deviceAvailable: Bool { false }
    fileprivate func runGPU(_ ops: [GPUOperation], on mesh: Mesh3, fromBase: Bool) -> Mesh3? { nil }
}
#endif
