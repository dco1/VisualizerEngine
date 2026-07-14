import Foundation
import simd

// ── CoinHull ──────────────────────────────────────────────────────────────────
//
// CPU-side convex-hull registration math for CoinDEMSolver's hull bodies
// (shape tag 4). Done ONCE per hull shape at registration, never per frame:
//
//   1. Build the actual convex hull of the input point cloud (incremental hull —
//      interior points are dropped, so the GPU support function only ever scans
//      true hull vertices).
//   2. Integrate the EXACT solid COM + inertia tensor over the hull's boundary
//      triangles (divergence theorem / tetrahedra against the centroid) —
//      uniform density, the real thing, not a point-cloud or bounding-box guess.
//   3. Diagonalize the inertia tensor (Jacobi rotations) and re-express the hull
//      vertices in the PRINCIPAL frame with the COM at the origin — the GPU
//      solver models local inertia as a diagonal, so the stored frame must be
//      the one where that is true.
//
// The registration returns the frame change (comOffset + principalRotation) so
// a caller can transform its render mesh into the same frame (or just render
// the returned vertices).

enum CoinHullMath {

    struct Prepared {
        /// Hull vertices in the principal frame (COM at origin).
        var vertices: [SIMD3<Float>]
        /// Per-unit-mass INVERSE inertia diagonal in the principal frame
        /// (I⁻¹ = invMass · k, matching cdBodyInvInertia's convention).
        var invInertiaK: SIMD3<Float>
        /// Bounding radius about the COM (broadphase reach).
        var boundingRadius: Float
        /// Smallest AABB half-extent in the principal frame (floor backstop scale).
        var minHalfExtent: Float
        /// COM of the SOLID hull in the caller's input frame.
        var comOffset: SIMD3<Float>
        /// Rotation from the (COM-shifted) input frame to the principal frame:
        /// stored = principalRotation⁻¹ · (input − comOffset).
        var principalRotation: simd_quatf
    }

    /// nil when fewer than 4 non-degenerate points are given.
    static func prepare(_ points: [SIMD3<Float>]) -> Prepared? {
        guard let faces = convexHullFaces(points), !faces.isEmpty else { return nil }

        // ── Exact solid COM + inertia via boundary tetrahedra ─────────────────
        // Each boundary triangle (a,b,c) forms a tetra with the origin; signed
        // volumes make the sum exact for any interior reference point.
        var volume: Float = 0
        var comAccum = SIMD3<Float>.zero
        for f in faces {
            let (a, b, c) = (points[f.0], points[f.1], points[f.2])
            let v = simd_dot(a, simd_cross(b, c)) / 6           // signed tetra volume
            volume += v
            comAccum += (a + b + c) / 4 * v                     // tetra centroid = (0+a+b+c)/4
        }
        guard volume > 1e-12 else { return nil }
        let com = comAccum / volume

        // Inertia about the COM: canonical covariance integral per tetra
        // (standard closed form; see e.g. Blow & Binstock, "How to find the
        // inertia tensor of a polyhedron").
        var C = simd_float3x3(0)                                // covariance ∫ x xᵀ dV
        for f in faces {
            let a = points[f.0] - com, b = points[f.1] - com, c = points[f.2] - com
            let v = simd_dot(a, simd_cross(b, c)) / 6
            // ∫ xᵀx over the tetra (0,a,b,c): v/20 · (Σᵢ Σⱼ xᵢxⱼᵀ + Σᵢ xᵢxᵢᵀ)
            let verts = [a, b, c]
            var S = simd_float3x3(0)
            for vi in verts {
                for vj in verts {
                    S += outer(vi, vj)
                }
                S += outer(vi, vi)
            }
            C += S * (v / 20)
        }
        // Inertia tensor from covariance: I = tr(C)·1 − C, normalized per unit mass.
        let trC = C.columns.0.x + C.columns.1.y + C.columns.2.z
        var I = simd_float3x3(diagonal: SIMD3(repeating: trC)) - C
        I = I * (1 / volume)                                    // per unit mass

        // ── Principal axes (Jacobi eigen-decomposition of the symmetric I) ────
        let (eigVals, eigVecs) = jacobiEigen(I)
        // Right-handed principal basis.
        var R = eigVecs
        if simd_determinant(R) < 0 { R.columns.2 = -R.columns.2 }
        let q = simd_quatf(R).normalized

        // Re-express hull vertices in the principal frame, COM at origin.
        let qInv = q.inverse
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(points.count)
        var usedIdx = Set<Int>()
        for f in faces { usedIdx.insert(f.0); usedIdx.insert(f.1); usedIdx.insert(f.2) }
        var boundR: Float = 0
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in usedIdx.sorted() {
            let v = simd_act(qInv, points[i] - com)
            out.append(v)
            boundR = max(boundR, simd_length(v))
            mn = simd_min(mn, v); mx = simd_max(mx, v)
        }
        let he = (mx - mn) * 0.5
        let evClamped = SIMD3<Float>(max(eigVals.x, 1e-10), max(eigVals.y, 1e-10), max(eigVals.z, 1e-10))
        return Prepared(
            vertices: out,
            invInertiaK: SIMD3(1 / evClamped.x, 1 / evClamped.y, 1 / evClamped.z),
            boundingRadius: boundR,
            minHalfExtent: max(min(he.x, min(he.y, he.z)), 1e-4),
            comOffset: com,
            principalRotation: q)
    }

    private static func outer(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> simd_float3x3 {
        simd_float3x3(columns: (a * b.x, a * b.y, a * b.z))
    }

    // ── Incremental convex hull (points ≤ ~64; done once at registration) ─────
    // Returns outward-wound triangle index faces, or nil for degenerate input.
    static func convexHullFaces(_ pts: [SIMD3<Float>]) -> [(Int, Int, Int)]? {
        let n = pts.count
        guard n >= 4 else { return nil }

        // Initial non-degenerate tetrahedron.
        var i0 = 0, i1 = -1, i2 = -1, i3 = -1
        for i in 1..<n where simd_length(pts[i] - pts[i0]) > 1e-7 { i1 = i; break }
        guard i1 > 0 else { return nil }
        for i in 1..<n where i != i1 {
            if simd_length(simd_cross(pts[i1] - pts[i0], pts[i] - pts[i0])) > 1e-9 { i2 = i; break }
        }
        guard i2 > 0 else { return nil }
        let nrm0 = simd_cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
        for i in 1..<n where i != i1 && i != i2 {
            if abs(simd_dot(nrm0, pts[i] - pts[i0])) > 1e-9 { i3 = i; break }
        }
        guard i3 > 0 else { return nil }

        var faces: [(Int, Int, Int)] = []
        func addFace(_ a: Int, _ b: Int, _ c: Int, awayFrom p: SIMD3<Float>) {
            // Wind so the normal points AWAY from the interior reference p.
            let nn = simd_cross(pts[b] - pts[a], pts[c] - pts[a])
            if simd_dot(nn, pts[a] - p) >= 0 { faces.append((a, b, c)) }
            else { faces.append((a, c, b)) }
        }
        let centroid0 = (pts[i0] + pts[i1] + pts[i2] + pts[i3]) / 4
        addFace(i0, i1, i2, awayFrom: centroid0)
        addFace(i0, i1, i3, awayFrom: centroid0)
        addFace(i0, i2, i3, awayFrom: centroid0)
        addFace(i1, i2, i3, awayFrom: centroid0)

        // Incrementally insert the remaining points.
        for i in 0..<n where i != i0 && i != i1 && i != i2 && i != i3 {
            let p = pts[i]
            // Faces visible from p.
            var visible: [Int] = []
            for (fi, f) in faces.enumerated() {
                let a = pts[f.0]
                let nn = simd_cross(pts[f.1] - a, pts[f.2] - a)
                if simd_dot(nn, p - a) > 1e-9 * max(simd_length(nn), 1) { visible.append(fi) }
            }
            if visible.isEmpty { continue }         // interior point — drop
            // Horizon edges = edges of visible faces not shared by two visible faces.
            var edgeCount: [String: (Int, Int, Int)] = [:]   // key → (a, b, count)
            func tally(_ a: Int, _ b: Int) {
                let key = "\(min(a, b)):\(max(a, b))"
                if var e = edgeCount[key] { e.2 += 1; edgeCount[key] = e }
                else { edgeCount[key] = (a, b, 1) }
            }
            for fi in visible {
                let f = faces[fi]
                tally(f.0, f.1); tally(f.1, f.2); tally(f.2, f.0)
            }
            // Remove visible faces (descending index).
            for fi in visible.sorted(by: >) { faces.remove(at: fi) }
            // Fan new faces from p over the horizon.
            let interior = centroid0
            for (_, e) in edgeCount where e.2 == 1 {
                addFace(e.0, e.1, i, awayFrom: interior)
            }
        }
        return faces
    }

    // ── Jacobi eigen-decomposition of a symmetric 3×3 ─────────────────────────
    // Returns (eigenvalues, eigenvector columns).
    static func jacobiEigen(_ m: simd_float3x3) -> (SIMD3<Float>, simd_float3x3) {
        var a = m
        var v = matrix_identity_float3x3
        for _ in 0..<24 {
            // Largest off-diagonal element.
            let off01 = abs(a.columns.1.x), off02 = abs(a.columns.2.x), off12 = abs(a.columns.2.y)
            var p = 0, q = 1, apq = a.columns.1.x
            if off02 > abs(apq) { p = 0; q = 2; apq = a.columns.2.x }
            if off12 > abs(apq) { p = 1; q = 2; apq = a.columns.2.y }
            if abs(apq) < 1e-12 { break }
            let app = a[p][p], aqq = a[q][q]
            let theta = (aqq - app) / (2 * apq)
            let t = (theta >= 0 ? 1 : -1) / (abs(theta) + sqrt(theta * theta + 1))
            let c = 1 / sqrt(t * t + 1), s = t * c
            var J = matrix_identity_float3x3
            J[p][p] = c; J[q][q] = c; J[p][q] = s; J[q][p] = -s
            a = J.transpose * a * J
            v = v * J
        }
        return (SIMD3(a.columns.0.x, a.columns.1.y, a.columns.2.z), v)
    }
}
