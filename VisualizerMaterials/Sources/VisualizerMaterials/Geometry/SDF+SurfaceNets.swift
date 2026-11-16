//  SDF+SurfaceNets.swift — naive surface-nets mesher for an `SDF` (ticket item C).
//
//  Pipeline (three stages, each CPU or — for a tree of primitives — GPU, see SDF+GPU.swift):
//    1. grid:     the field at every sample of the box (n+1)³ samples, Float.
//    2. topology: one vertex per sign-changing cell (mean of its edge crossings), one quad per sign-changing grid
//                 edge, split along the shorter diagonal. Always CPU: it is integer bookkeeping.
//    3. refine:   each vertex is projected back onto the surface (clamped to its own cell, so it can't tear) and its
//                 normal is the field's gradient there — smooth normals.
//  Guarantees: closed, edge-manifold (every edge in exactly two triangles, by index), outward-wound, vertices
//  shared. Samples on the box's boundary are forced outside, so a solid that leaves the box is capped flat at the
//  box, not left open. Not guaranteed: features thinner than ~2 cells are lost or merged; sharp edges are rounded
//  by about a cell (use a smaller cell); UVs are projected per VERTEX, so a triangle spanning two projection
//  directions stretches (the mesh stays welded: no per-triangle seams).

import Foundation
import simd

public struct SDFMeshOptions: Sendable {
    public enum Backend: Sendable { case cpu, gpu, auto }
    public enum UVs: Sendable { case none, boxProjected(metersPerUnit: Double) }
    public var backend: Backend = .auto
    public var uvs: UVs = .none
    /// Projections of each vertex onto the surface (0 = leave the mean of the crossings).
    public var projectIterations = 1
    public init(backend: Backend = .auto, uvs: UVs = .none, projectIterations: Int = 1) {
        self.backend = backend; self.uvs = uvs; self.projectIterations = projectIterations
    }
}

/// Vertex pass shared by both backends: position, then normal.
struct SDFNetVertex { var position: Vec3; var cellMin: Vec3 }

public extension SDF {
    /// Mesh the zero set inside `[lo, hi]` with square cells of side `cell`.
    func mesh(lo: Vec3, hi: Vec3, cell: Double, options: SDFMeshOptions = .init()) -> Mesh3 {
        let cells = SIMD3<Int>(Int(((hi.x - lo.x) / cell).rounded(.up)), Int(((hi.y - lo.y) / cell).rounded(.up)),
                               Int(((hi.z - lo.z) / cell).rounded(.up)))
        precondition(cells.x >= 2 && cells.y >= 2 && cells.z >= 2, "SDF.mesh: box smaller than two cells")
        let program = options.backend == .cpu ? nil : gpuProgram()
        var grid: [Float]? = nil
        if let program { grid = SDFGPU.shared?.grid(program, lo: lo, cell: cell, cells: cells) }
        let values = grid ?? cpuGrid(lo: lo, cell: cell, cells: cells)
        let topo = SDF.netTopology(values, cells: cells, lo: lo, cell: cell)
        let verts = topo.vertices
        var refined: [(Vec3, Vec3)]? = nil
        if let program, grid != nil {
            refined = SDFGPU.shared?.refine(program, verts, cell: cell, iterations: options.projectIterations)
        }
        let out = refined ?? cpuRefine(verts, cell: cell, iterations: options.projectIterations)

        var m = Mesh3()
        m.positions = out.map { $0.0 }
        m.normals = out.map { $0.1 }
        m.indices = topo.indices
        switch options.uvs {
        case .none: m.uvs = [Vec2](repeating: .zero, count: m.positions.count)
        case .boxProjected(let s):
            m.uvs = zip(m.positions, m.normals).map { p, n in
                let a = simd_abs(n)
                if a.x >= a.y && a.x >= a.z { return Vec2(p.z, p.y) / s }
                return a.y >= a.z ? Vec2(p.x, p.z) / s : Vec2(p.x, p.y) / s
            }
        }
        return m
    }

    /// Stage 1 on the CPU, in parallel by z-slab. Layout: `(k · ny + j) · nx + i` over (n+1)³ samples.
    internal func cpuGrid(lo: Vec3, cell: Double, cells: SIMD3<Int>) -> [Float] {
        let nx = cells.x + 1, ny = cells.y + 1, nz = cells.z + 1
        let slabs = parallelMap(Array(0..<nz)) { k -> [Float] in
            var s = [Float](repeating: 0, count: nx * ny)
            for j in 0..<ny { for i in 0..<nx {
                s[j * nx + i] = Float(SDF.eval(self.node, lo + Vec3(Double(i), Double(j), Double(k)) * cell))
            } }
            return s
        }
        return slabs.flatMap { $0 }
    }

    /// Stage 3 on the CPU.
    internal func cpuRefine(_ verts: [SDFNetVertex], cell: Double, iterations: Int) -> [(Vec3, Vec3)] {
        let chunks = stride(from: 0, to: verts.count, by: 512).map { $0..<Swift.min($0 + 512, verts.count) }
        let h = cell * 0.25
        let parts = parallelMap(chunks) { r -> [(Vec3, Vec3)] in
            r.map { i in
                var p = verts[i].position
                let lo = verts[i].cellMin, hi = lo + Vec3(repeating: cell)
                for _ in 0..<iterations {
                    let d = SDF.eval(self.node, p)
                    let g = self.gradient(at: p, h: h)
                    let g2 = dot3(g, g)
                    if g2 > 1e-12 { p = simd_clamp(p - g * (d / g2), lo, hi) }
                }
                let g = self.gradient(at: p, h: h)
                let n = len3(g) > 1e-12 ? normalize3(g) : Vec3(0, 1, 0)
                return (p, n)
            }
        }
        return parts.flatMap { $0 }
    }

    /// Stage 2: surface-nets topology from a sample grid.
    internal static func netTopology(_ raw: [Float], cells: SIMD3<Int>, lo: Vec3, cell: Double)
        -> (vertices: [SDFNetVertex], indices: [UInt32]) {
        let nx = cells.x + 1, ny = cells.y + 1, nz = cells.z + 1
        var v = raw
        // Box boundary forced outside → closed surface.
        let eps = Float(cell * 1e-3)
        for k in 0..<nz { for j in 0..<ny { for i in 0..<nx
            where i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
            let s = (k * ny + j) * nx + i
            if v[s] <= 0 { v[s] = eps }
        } } }
        func sample(_ i: Int, _ j: Int, _ k: Int) -> Float { v[(k * ny + j) * nx + i] }
        disambiguate(&v, cells: cells, eps: eps)

        var cellVertex = [Int32](repeating: -1, count: cells.x * cells.y * cells.z)
        func ci(_ i: Int, _ j: Int, _ k: Int) -> Int { (k * cells.y + j) * cells.x + i }
        var verts: [SDFNetVertex] = []
        // The 12 cube edges as corner-index pairs; corner c has bits (x = 1, y = 2, z = 4).
        let edges: [(Int, Int)] = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        for k in 0..<cells.z { for j in 0..<cells.y { for i in 0..<cells.x {
            var c = [Float](repeating: 0, count: 8)
            var neg = 0
            for b in 0..<8 {
                c[b] = sample(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1))
                if c[b] < 0 { neg += 1 }
            }
            guard neg != 0 && neg != 8 else { continue }
            var acc = Vec3.zero, n = 0.0
            for (a, b) in edges where (c[a] < 0) != (c[b] < 0) {
                let t = Double(c[a] / (c[a] - c[b]))
                let pa = Vec3(Double(a & 1), Double((a >> 1) & 1), Double((a >> 2) & 1))
                let pb = Vec3(Double(b & 1), Double((b >> 1) & 1), Double((b >> 2) & 1))
                acc += pa + (pb - pa) * t; n += 1
            }
            let cellMin = lo + Vec3(Double(i), Double(j), Double(k)) * cell
            cellVertex[ci(i, j, k)] = Int32(verts.count)
            verts.append(SDFNetVertex(position: cellMin + acc / n * cell, cellMin: cellMin))
        } } }

        var indices: [UInt32] = []
        // One quad per sign-changing grid edge, from the four cells round it.
        for a in 0..<3 {
            let u = (a + 1) % 3, w = (a + 2) % 3
            var lim = SIMD3(nx, ny, nz); lim[a] -= 1               // edge start s_a in 0..<cells[a]
            for k in 0..<lim.z { for j in 0..<lim.y { for i in 0..<lim.x {
                let s = SIMD3(i, j, k)
                guard s[u] >= 1, s[w] >= 1, s[u] <= cells[u] - 1, s[w] <= cells[w] - 1 else { continue }
                var e = s; e[a] += 1
                let v0 = sample(s.x, s.y, s.z), v1 = sample(e.x, e.y, e.z)
                guard (v0 < 0) != (v1 < 0) else { continue }
                func cellAt(_ du: Int, _ dw: Int) -> Int32 {
                    var q = s; q[u] += du; q[w] += dw
                    return cellVertex[ci(q.x, q.y, q.z)]
                }
                let q = [cellAt(-1, -1), cellAt(0, -1), cellAt(0, 0), cellAt(-1, 0)]
                guard !q.contains(-1) else { continue }
                var o = q.map { UInt32($0) }
                if v0 >= 0 { o.reverse() }                         // inside is the far end → flip
                let p = o.map { verts[Int($0)].position }
                if len3(p[0] - p[2]) <= len3(p[1] - p[3]) {
                    indices += [o[0], o[1], o[2], o[0], o[2], o[3]]
                } else {
                    indices += [o[1], o[2], o[3], o[1], o[3], o[0]]
                }
            } } }
        }
        return (verts, indices)
    }
}

extension SDF {
    /// Make every grid face unambiguous. A face whose four samples alternate in sign (a saddle) would put four quads on one
    /// mesh edge (non-manifold). The sample nearest the surface among the face's interior ones is flipped to the other
    /// side (its value negated), so the surface moves by at most that sample's distance. Repeats until none are left
    /// (a flip can make a neighbouring face ambiguous); box-boundary samples are never flipped, so the surface stays closed.
    static func disambiguate(_ v: inout [Float], cells: SIMD3<Int>, eps: Float) {
        let n = SIMD3(cells.x + 1, cells.y + 1, cells.z + 1)
        func idx(_ s: SIMD3<Int>) -> Int { (s.z * n.y + s.y) * n.x + s.x }
        func interior(_ s: SIMD3<Int>) -> Bool {
            s.x > 0 && s.y > 0 && s.z > 0 && s.x < n.x - 1 && s.y < n.y - 1 && s.z < n.z - 1
        }
        for _ in 0..<16 {
            var flipped = 0
            for a in 0..<3 {
                let u = (a + 1) % 3, w = (a + 2) % 3
                for ia in 0..<n[a] { for iu in 0..<cells[u] { for iw in 0..<cells[w] {
                    var s = SIMD3(0, 0, 0); s[a] = ia; s[u] = iu; s[w] = iw
                    var s10 = s; s10[u] += 1
                    var s11 = s10; s11[w] += 1
                    var s01 = s; s01[w] += 1
                    let i0 = idx(s), i1 = idx(s10), i2 = idx(s11), i3 = idx(s01)
                    let n0 = v[i0] < 0
                    guard n0 == (v[i2] < 0), n0 != (v[i1] < 0), (v[i1] < 0) == (v[i3] < 0) else { continue }
                    var best = -1, bestAbs = Float.infinity
                    if interior(s) && abs(v[i0]) < bestAbs { best = i0; bestAbs = abs(v[i0]) }
                    if interior(s10) && abs(v[i1]) < bestAbs { best = i1; bestAbs = abs(v[i1]) }
                    if interior(s11) && abs(v[i2]) < bestAbs { best = i2; bestAbs = abs(v[i2]) }
                    if interior(s01) && abs(v[i3]) < bestAbs { best = i3; bestAbs = abs(v[i3]) }
                    guard best >= 0 else { continue }
                    let at = best
                    v[at] = v[at] < 0 ? Swift.max(-v[at], eps) : -Swift.max(v[at], eps)
                    flipped += 1
                } } }
            }
            if flipped == 0 { return }
        }
    }
}
