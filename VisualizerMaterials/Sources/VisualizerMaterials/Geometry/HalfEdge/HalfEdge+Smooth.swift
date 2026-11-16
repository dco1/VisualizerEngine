//  HalfEdge+Smooth — Laplacian smoothing of selected vertices.
//
//  Each step moves a vertex by `lambda` of the way to the average of its neighbours (Jacobi: every vertex reads
//  the previous step). Vertices on the mesh boundary stay put when `pinBoundary` (the default), and so does
//  everything not selected (unselected neighbours still pull). Guarantee: topology is untouched, validate()
//  stays clean. Not guaranteed: volume (smoothing shrinks a closed surface). Above `HEEdit.gpuThreshold`
//  vertices the iterations run in one Metal command buffer (`HalfEdgeEditGPU`); `laplacianSmoothCPU` is the
//  oracle and the fallback.

import simd

public extension HalfEdgeMesh {

    @discardableResult
    mutating func smoothVertices(_ sel: HESelection, iterations: Int = 1, lambda: Double = 0.5,
                                 pinBoundary: Bool = true, useGPU: Bool? = nil) -> HESelection {
        let vs = sel.vertices.filter { $0 >= 0 && $0 < positions.count && vertexHE[$0] >= 0 }.sorted()
        guard !vs.isEmpty, iterations > 0 else { return sel }
        var nbrs = [[Int]](repeating: [], count: positions.count)
        var pinned = [Bool](repeating: true, count: positions.count)
        for v in vs where !(pinBoundary && isBoundary(vertex: v)) { nbrs[v] = neighbours(of: v); pinned[v] = false }
        let gpu = useGPU ?? (vs.count > HEEdit.gpuThreshold)
        let flat = flatFaces(around: vs)
        if gpu, let r = HalfEdgeEditGPU.shared?.smooth(positions: positions, neighbours: nbrs, pinned: pinned, lambda: lambda, iterations: iterations) {
            for v in vs { positions[v] = r[v] }
        } else {
            positions = HalfEdgeMesh.laplacianSmoothCPU(positions, neighbours: nbrs, pinned: pinned, lambda: lambda, iterations: iterations)
        }
        reshade(flat)
        return sel
    }

    /// CPU twin of the `he_smooth` kernel.
    static func laplacianSmoothCPU(_ p: [Vec3], neighbours: [[Int]], pinned: [Bool], lambda: Double, iterations: Int) -> [Vec3] {
        var cur = p
        for _ in 0..<iterations {
            var next = cur
            for i in 0..<cur.count where !pinned[i] && !neighbours[i].isEmpty {
                let avg = neighbours[i].reduce(Vec3.zero) { $0 + cur[$1] } / Double(neighbours[i].count)
                next[i] = cur[i] + (avg - cur[i]) * lambda
            }
            cur = next
        }
        return cur
    }
}
