//  HalfEdge+Bridge — bridge two boundary loops with a band of quads; fill holes (n-gon, triangles, or a quad
//  grid); delete faces and fill what is left.
//
//  Bridge: both loops must be boundary loops of equal length (selected boundary edges, or vertex rings given in
//  the direction their owning faces run). The loops are paired in opposite directions (the band joins two
//  holes), starting from the pairing with the least total distance, then rotated by `twist` positions. `cuts`
//  adds that many rings of vertices in between (straight lerps; displace them afterwards to shape the band).
//  Two loops of one tube bridged give a closed torus (genus 1). nil (unchanged) when the loops are not both
//  boundary loops, differ in length, or share a vertex.
//  Fill: `.ngon` one face (the cap of a hole), `.triangles` ear-clipped, `.grid` a fan of quads round a new
//  centre vertex (even loops; a 4-loop is a single quad; odd loops fall back to a triangle fan).
//  `deleteAndFill` removes faces and fills the hole left in their neighbours.

import simd

public enum HEFillMode: Sendable { case ngon, triangles, grid }

public extension HalfEdgeMesh {

    // MARK: bridge

    @discardableResult
    mutating func bridge(_ sel: HESelection, cuts: Int = 0, twist: Int = 0) -> HESelection? {
        let loops = boundaryLoops(within: sel.edges)
        guard loops.count == 2 else { return nil }
        return bridge(loopA: loops[0], loopB: loops[1], cuts: cuts, twist: twist)
    }

    /// Vertex rings (owner direction) of the boundary loops all of whose edges are in `es` (all loops if `es` is empty).
    func boundaryLoops(within es: Set<Int>) -> [[Int]] {
        boundaryLoops().filter { loop in
            es.isEmpty || loop.indices.allSatisfy { i in edge(between: loop[i], loop[(i + 1) % loop.count]).map(es.contains) ?? false }
        }
    }

    @discardableResult
    mutating func bridge(loopA a: [Int], loopB b: [Int], cuts: Int = 0, twist: Int = 0) -> HESelection? {
        let n = a.count
        guard n >= 3, b.count == n, Set(a + b).count == 2 * n, cuts >= 0,
              (0..<n).allSatisfy({ halfedge(from: a[$0], to: a[($0 + 1) % n]).map { heTwin[$0] < 0 } ?? false }),
              (0..<n).allSatisfy({ halfedge(from: b[$0], to: b[($0 + 1) % n]).map { heTwin[$0] < 0 } ?? false }) else { return nil }
        func cost(_ c: Int) -> Double { (0..<n).reduce(0) { $0 + dist3(positions[a[$1]], positions[b[(((c - $1) % n) + n) % n]]) } }
        let c0 = (0..<n).min { cost($0) < cost($1) }!
        let c = c0 + twist
        let backup = self
        var rings: [[Int]] = [a]
        for r in 1...(cuts + 1) {
            let t = Double(r) / Double(cuts + 1)
            if r == cuts + 1 { rings.append((0..<n).map { b[(((c - $0) % n) + n) % n] }); break }
            rings.append((0..<n).map { i in
                let bi = b[(((c - i) % n) + n) % n]
                return addVertex(positions[a[i]] + (positions[bi] - positions[a[i]]) * t)
            })
        }
        var adds: [EditPoly] = []
        for r in 0..<(rings.count - 1) { for i in 0..<n {
            let j = (i + 1) % n
            adds.append(EditPoly(v: [rings[r][j], rings[r][i], rings[r + 1][i], rings[r + 1][j]]))
        } }
        guard let ids = rebuild(removing: [], adding: adds) else { self = backup; return nil }
        return selection(.face, ids)
    }

    // MARK: fill

    @discardableResult
    mutating func fillHoles(_ sel: HESelection, mode: HEFillMode = .ngon) -> HESelection? {
        let loops = boundaryLoops(within: sel.edges)
        guard !loops.isEmpty else { return nil }
        let backup = self
        var out: [Int] = []
        for l in loops {
            guard let ids = fill(loop: l, mode: mode) else { self = backup; return nil }
            out += ids
        }
        return selection(.face, out)
    }

    /// Fill one boundary loop (vertices in the direction their owning faces run).
    mutating func fill(loop: [Int], mode: HEFillMode) -> [Int]? {
        let p = Array(loop.reversed()), n = p.count
        guard n >= 3 else { return nil }
        var adds: [EditPoly] = []
        switch mode {
        case .ngon: adds = [EditPoly(v: p)]
        case .triangles:
            adds = HalfEdgeMesh.triangulatePolygon(p.map { positions[$0] }).map { EditPoly(v: [p[$0.0], p[$0.1], p[$0.2]]) }
        case .grid:
            if n == 4 { adds = [EditPoly(v: p)]; break }
            let c = addVertex(p.reduce(Vec3.zero) { $0 + positions[$1] } / Double(n))
            if n % 2 == 0 {
                for k in stride(from: 0, to: n, by: 2) { adds.append(EditPoly(v: [c, p[k], p[(k + 1) % n], p[(k + 2) % n]])) }
            } else {
                for k in 0..<n { adds.append(EditPoly(v: [c, p[k], p[(k + 1) % n]])) }
            }
        }
        return rebuild(removing: [], adding: adds)
    }

    // MARK: delete and fill

    /// Delete the selected faces and fill each hole their neighbours are left round. nil when no hole closes.
    @discardableResult
    mutating func deleteAndFill(_ sel: HESelection, mode: HEFillMode = .ngon) -> HESelection? {
        let region = selectedFaces(sel)
        guard !region.isEmpty else { return nil }
        let rim = Set(boundaryEdges(ofFaces: Set(region)).filter { faces(ofEdge: $0).count == 2 })
        let backup = self
        deleteFaces(region)
        repairVertexLinks()
        var out: [Int] = []
        for loop in boundaryLoops(within: rim) where loop.count >= 3 {
            guard let ids = fill(loop: loop, mode: mode) else { self = backup; return nil }
            out += ids
        }
        guard !out.isEmpty else { self = backup; return nil }
        return selection(.face, out)
    }
}
