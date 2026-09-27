import Foundation
import simd

// ── CoinHull ──────────────────────────────────────────────────────────────────
//
// CPU-side convex-hull registration math for CoinDEMSolver's hull bodies
// (shape tag 4). Done ONCE per hull shape at registration, never per frame:
//
//   1. Build the actual convex hull of the input point cloud (quickhull — interior
//      points are dropped, so the GPU support function only ever scans true hull
//      vertices).
//   2. Integrate the EXACT solid COM + inertia tensor over the hull's boundary
//      triangles (divergence theorem / tetrahedra against a reference point) —
//      uniform density, the real thing, not a point-cloud or bounding-box guess.
//   3. Diagonalize the inertia tensor (Jacobi rotations) and re-express the hull
//      vertices in the PRINCIPAL frame with the COM at the origin — the GPU
//      solver models local inertia as a diagonal, so the stored frame must be
//      the one where that is true.
//
// SCALE: everything is computed in Double, and every tolerance is RELATIVE to the
// input's size (VZ-0148). The previous incremental hull tested visibility as
// `dot(nn, p − a) > 1e-9·max(|nn|, 1)` with nn the UNNORMALISED face normal
// (|nn| = 2·area): at millimetre scale |nn| ≪ 1, so the threshold was an absolute
// 1e-9 m³-ish product — a point had to sit ~1e-9/|nn| m (0.3–5 mm for the Digital
// Clock bar's faces) outside a face to "see" it. Small faces never saw their
// neighbours' points: the 25 mm bar kept 29 of 32 vertices with 11 non-manifold
// edges and its COM 1.3 mm off, a 1 mm cube lost all six 0.2 mm bumps — while the
// SAME points ×1000 were exact. Now a point is outside a face iff its distance to
// the face's unit-normal plane exceeds tol = 1e-6·L + 1e-7·M (L = input bounding-box
// diagonal; M = largest coordinate magnitude, covering Float input quantisation), so
// the hull of a shape is the same at 1 µm, 1 mm or 1 km.
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
        let P = points.map { SIMD3<Double>(Double($0.x), Double($0.y), Double($0.z)) }

        var usedIdx = Set<Int>()
        for f in faces { usedIdx.insert(f.0); usedIdx.insert(f.1); usedIdx.insert(f.2) }
        let used = usedIdx.sorted()
        // Reference point for the tetra decomposition: the vertex centroid, so the
        // signed volumes stay well-conditioned however far the input sits from the origin.
        var ref = SIMD3<Double>.zero
        var lo = P[used[0]], hi = P[used[0]]
        for i in used { ref += P[i]; lo = simd_min(lo, P[i]); hi = simd_max(hi, P[i]) }
        ref /= Double(used.count)
        let L = simd_length(hi - lo)

        // ── Exact solid COM + inertia via boundary tetrahedra ─────────────────
        // Each boundary triangle (a,b,c) forms a tetra with `ref`; signed volumes
        // make the sum exact for any reference point.
        var volume = 0.0
        var comAccum = SIMD3<Double>.zero
        for f in faces {
            let a = P[f.0] - ref, b = P[f.1] - ref, c = P[f.2] - ref
            let v = simd_dot(a, simd_cross(b, c)) / 6               // signed tetra volume
            volume += v
            comAccum += (a + b + c) / 4 * v                         // tetra centroid = (ref+a+b+c)/4
        }
        // Relative flatness check (was an absolute 1e-12 m³, which a 0.1 mm hull fails).
        guard volume > 1e-12 * L * L * L else { return nil }
        let com = ref + comAccum / volume

        // Inertia about the COM: canonical covariance integral per tetra
        // (standard closed form; see e.g. Blow & Binstock, "How to find the
        // inertia tensor of a polyhedron").
        var C = simd_double3x3(0)                                   // covariance ∫ x xᵀ dV
        for f in faces {
            let a = P[f.0] - com, b = P[f.1] - com, c = P[f.2] - com
            let v = simd_dot(a, simd_cross(b, c)) / 6
            // ∫ xᵀx over the tetra (0,a,b,c): v/20 · (Σᵢ Σⱼ xᵢxⱼᵀ + Σᵢ xᵢxᵢᵀ)
            let verts = [a, b, c]
            var S = simd_double3x3(0)
            for vi in verts {
                for vj in verts { S += outer(vi, vj) }
                S += outer(vi, vi)
            }
            C += S * (v / 20)
        }
        // Inertia tensor from covariance: I = tr(C)·1 − C, normalized per unit mass.
        let trC = C.columns.0.x + C.columns.1.y + C.columns.2.z
        var I = simd_double3x3(diagonal: SIMD3(repeating: trC)) - C
        I = I * (1 / volume)                                        // per unit mass

        // ── Principal axes (Jacobi eigen-decomposition of the symmetric I) ────
        let (eigVals, eigVecs) = jacobiEigen(I)
        // Right-handed principal basis.
        var R = eigVecs
        if simd_determinant(R) < 0 { R.columns.2 = -R.columns.2 }
        let qd = simd_quatd(R).normalized
        let q = simd_quatf(ix: Float(qd.imag.x), iy: Float(qd.imag.y), iz: Float(qd.imag.z), r: Float(qd.real)).normalized

        // Re-express hull vertices in the principal frame, COM at origin.
        let qInv = qd.inverse
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(used.count)
        var boundR = 0.0
        var mn = SIMD3<Double>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Double>(repeating: -.greatestFiniteMagnitude)
        for i in used {
            let v = simd_act(qInv, P[i] - com)
            out.append(SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z)))
            boundR = max(boundR, simd_length(v))
            mn = simd_min(mn, v); mx = simd_max(mx, v)
        }
        let he = (mx - mn) * 0.5
        // Relative floor (a real 3-D hull's principal moments are all ~L²).
        let evFloor = 1e-12 * L * L
        let ev = SIMD3<Double>(max(eigVals.x, evFloor), max(eigVals.y, evFloor), max(eigVals.z, evFloor))
        return Prepared(
            vertices: out,
            invInertiaK: SIMD3<Float>(Float(1 / ev.x), Float(1 / ev.y), Float(1 / ev.z)),
            boundingRadius: Float(boundR),
            minHalfExtent: max(Float(min(he.x, min(he.y, he.z))), 1e-4),
            comOffset: SIMD3<Float>(Float(com.x), Float(com.y), Float(com.z)),
            principalRotation: q)
    }

    private static func outer(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> simd_double3x3 {
        simd_double3x3(columns: (a * b.x, a * b.y, a * b.z))
    }

    // ── Quickhull (done once at registration) ─────────────────────────────────
    // Returns outward-wound triangle index faces (into `pts`), or nil for
    // degenerate input (< 4 points, or all collinear / coplanar within tolerance).
    //
    // Farthest-point quickhull with per-face conflict ("outside") lists:
    //   • seed: the farthest pair among the six axis-extreme points, the point
    //     farthest from their line, the point farthest from that plane;
    //   • repeatedly take the lowest-index live face with outside points, its
    //     farthest point is the eye; the faces the eye sees are found by a BFS over
    //     edge adjacency from that face (a connected cap), the horizon is every
    //     DIRECTED edge of a visible face whose twin face is not visible, and each
    //     horizon edge u→v gets the new face (u, v, eye) — winding inherited from the
    //     removed face, so orientation stays consistent without an interior test;
    //   • orphaned outside points are re-assigned to the new faces or dropped.
    // Deterministic: no Dictionary/Set iteration decides any output order (the old
    // horizon fan came out in Dictionary order, so the face ARRAY — and the COM in its
    // 7th digit — varied from process to process).
    static func convexHullFaces(_ pts: [SIMD3<Float>]) -> [(Int, Int, Int)]? {
        let n = pts.count
        guard n >= 4 else { return nil }
        let P = pts.map { SIMD3<Double>(Double($0.x), Double($0.y), Double($0.z)) }
        for p in P where !(p.x.isFinite && p.y.isFinite && p.z.isFinite) { return nil }
        var lo = P[0], hi = P[0]
        for p in P { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let L = simd_length(hi - lo)
        guard L > 0 else { return nil }
        let M = max(simd_reduce_max(simd_abs(lo)), simd_reduce_max(simd_abs(hi)))
        // Plane-distance tolerance: scale-relative, plus the Float input's own
        // quantisation (≈ 6e-8·|coordinate|) for a hull sitting far from the origin.
        let tol = 1e-6 * L + 1e-7 * M

        // ── Seed tetrahedron (every test relative to `tol`) ───────────────────
        var ext: [Int] = []
        for axis in 0..<3 {
            var mn = 0, mx = 0
            for i in 1..<n {
                if P[i][axis] < P[mn][axis] { mn = i }
                if P[i][axis] > P[mx][axis] { mx = i }
            }
            ext += [mn, mx]
        }
        var i0 = ext[0], i1 = ext[1]
        var best = -1.0
        for a in 0..<6 {
            for b in (a + 1)..<6 {
                let d = simd_length_squared(P[ext[a]] - P[ext[b]])
                if d > best { best = d; i0 = ext[a]; i1 = ext[b] }
            }
        }
        guard best.squareRoot() > tol else { return nil }
        let dir01 = simd_normalize(P[i1] - P[i0])
        var i2 = -1
        best = tol
        for i in 0..<n {
            let d = simd_length(simd_cross(dir01, P[i] - P[i0]))
            if d > best { best = d; i2 = i }
        }
        guard i2 >= 0 else { return nil }                            // collinear
        let n012 = simd_normalize(simd_cross(P[i1] - P[i0], P[i2] - P[i0]))
        var i3 = -1
        best = tol
        for i in 0..<n {
            let d = abs(simd_dot(n012, P[i] - P[i0]))
            if d > best { best = d; i3 = i }
        }
        guard i3 >= 0 else { return nil }                            // coplanar

        // ── Face store ────────────────────────────────────────────────────────
        struct Face {
            var a: Int, b: Int, c: Int
            var normal: SIMD3<Double>          // unit, outward
            var offset: Double                 // normal · P[a]
            var alive = true
            var outside: [Int] = []
        }
        var faces: [Face] = []
        var edgeFace: [Int: Int] = [:]         // directed edge (u→v) key → owning face
        func key(_ u: Int, _ v: Int) -> Int { u &* n &+ v }
        func dist(_ f: Int, _ p: Int) -> Double { simd_dot(faces[f].normal, P[p]) - faces[f].offset }
        @discardableResult
        func addFace(_ a: Int, _ b: Int, _ c: Int) -> Int {
            let nn = simd_cross(P[b] - P[a], P[c] - P[a])
            let len = simd_length(nn)
            let nh = len > 0 ? nn / len : SIMD3<Double>.zero
            faces.append(Face(a: a, b: b, c: c, normal: nh, offset: simd_dot(nh, P[a])))
            let f = faces.count - 1
            edgeFace[key(a, b)] = f; edgeFace[key(b, c)] = f; edgeFace[key(c, a)] = f
            return f
        }

        // Seed faces, each wound away from the tetra's centroid.
        let centroid = (P[i0] + P[i1] + P[i2] + P[i3]) / 4
        for (a, b, c) in [(i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)] {
            let nn = simd_cross(P[b] - P[a], P[c] - P[a])
            if simd_dot(nn, P[a] - centroid) >= 0 { addFace(a, b, c) } else { addFace(a, c, b) }
        }
        // Initial conflict lists: each remaining point goes to the face it is farthest
        // outside of (ties → lowest face index); points outside no face are interior.
        for p in 0..<n where p != i0 && p != i1 && p != i2 && p != i3 {
            var bf = -1, bd = tol
            for f in 0..<faces.count { let d = dist(f, p); if d > bd { bd = d; bf = f } }
            if bf >= 0 { faces[bf].outside.append(p) }
        }

        // ── Expand ────────────────────────────────────────────────────────────
        // Faces before `cursor` have no outside points and never get new ones
        // (orphans only go to NEW faces, appended at the end).
        var cursor = 0
        while true {
            while cursor < faces.count && (!faces[cursor].alive || faces[cursor].outside.isEmpty) { cursor += 1 }
            if cursor == faces.count { break }
            let seedFace = cursor
            var eye = faces[seedFace].outside[0]
            var eyeD = dist(seedFace, eye)
            for p in faces[seedFace].outside.dropFirst() {
                let d = dist(seedFace, p)
                if d > eyeD { eyeD = d; eye = p }
            }
            // Visible cap: BFS over edge adjacency from the seed face.
            var visible = [seedFace]
            var isVisible = Set<Int>([seedFace])
            var qi = 0
            while qi < visible.count {
                let f = visible[qi]; qi += 1
                let fa = faces[f]
                for (u, v) in [(fa.a, fa.b), (fa.b, fa.c), (fa.c, fa.a)] {
                    guard let g = edgeFace[key(v, u)], faces[g].alive, !isVisible.contains(g) else { continue }
                    if dist(g, eye) > tol { isVisible.insert(g); visible.append(g) }
                }
            }
            // Horizon: directed edges of visible faces whose twin face is not visible.
            var horizon: [(Int, Int)] = []
            for f in visible {
                let fa = faces[f]
                for (u, v) in [(fa.a, fa.b), (fa.b, fa.c), (fa.c, fa.a)] {
                    if let g = edgeFace[key(v, u)], isVisible.contains(g) { continue }
                    horizon.append((u, v))
                }
            }
            // Retire the cap (collect its orphans), then fan the eye over the horizon.
            var orphans: [Int] = []
            for f in visible {
                faces[f].alive = false
                orphans += faces[f].outside
                faces[f].outside = []
                let fa = faces[f]
                for (u, v) in [(fa.a, fa.b), (fa.b, fa.c), (fa.c, fa.a)] where edgeFace[key(u, v)] == f {
                    edgeFace[key(u, v)] = nil
                }
            }
            let firstNew = faces.count
            for (u, v) in horizon { addFace(u, v, eye) }
            for p in orphans where p != eye {
                var bf = -1, bd = tol
                for f in firstNew..<faces.count { let d = dist(f, p); if d > bd { bd = d; bf = f } }
                if bf >= 0 { faces[bf].outside.append(p) }
            }
        }
        return faces.filter { $0.alive }.map { ($0.a, $0.b, $0.c) }
    }

    // ── Jacobi eigen-decomposition of a symmetric 3×3 ─────────────────────────
    // Returns (eigenvalues, eigenvector columns). Double, with a threshold relative
    // to the matrix's own magnitude (the inertia of a millimetre hull is ~1e-6 m²).
    static func jacobiEigen(_ m: simd_double3x3) -> (SIMD3<Double>, simd_double3x3) {
        var a = m
        var v = matrix_identity_double3x3
        let scale = max(abs(m.columns.0.x) + abs(m.columns.1.y) + abs(m.columns.2.z), .leastNormalMagnitude)
        for _ in 0..<50 {
            // Largest off-diagonal element.
            let off01 = abs(a.columns.1.x), off02 = abs(a.columns.2.x), off12 = abs(a.columns.2.y)
            var p = 0, q = 1, apq = a.columns.1.x
            if off02 > off01 { p = 0; q = 2; apq = a.columns.2.x }
            if off12 > max(off01, off02) { p = 1; q = 2; apq = a.columns.2.y }
            if abs(apq) <= 1e-15 * scale { break }
            let app = a[p][p], aqq = a[q][q]
            let theta = (aqq - app) / (2 * apq)
            let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
            let c = 1 / (t * t + 1).squareRoot(), s = t * c
            // simd subscripts are [column][row]: this is the rotation with (row p, col q)
            // = +s and (row q, col p) = −s, the one the θ/t formula above zeroes a_pq
            // for (Jᵀ·a·J). The transpose (s ↔ −s, what this read before) turns the
            // residual the wrong way — doubling instead of zeroing it — so any hull whose
            // inertia is not already diagonal in the input frame never converged (50
            // sweeps, off-diagonal left at ~its input size) and got a wrong principal frame.
            var J = matrix_identity_double3x3
            J[p][p] = c; J[q][q] = c; J[q][p] = s; J[p][q] = -s
            a = J.transpose * a * J
            v = v * J
        }
        return (SIMD3(a.columns.0.x, a.columns.1.y, a.columns.2.z), v)
    }
}
