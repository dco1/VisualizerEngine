import simd
import XCTest
@testable import VisualizerVegetation

/// A `LeafCardSink` that keeps everything, verbatim.
///
/// Verbatim is the point. `LeafCardSink` promises a sink that stores triangles as given will get an
/// order already wound to agree with `outward` — the yard tree's `Float` soup is exactly such a
/// sink, and it has no backing store that could rewind for it. A test double that re-derived the
/// winding (as `Mesh3` legitimately does) would silently pass a constructor that got the order
/// wrong, which is the half of the contract most likely to break.
struct CollectingSink<Scalar: LeafScalar>: LeafCardSink {

    struct Triangle {
        var a: SIMD3<Scalar>
        var b: SIMD3<Scalar>
        var c: SIMD3<Scalar>
        /// The direction the constructor declared this face must point.
        var outward: SIMD3<Scalar>
        var geometricNormal: SIMD3<Scalar>

        /// The face normal implied by the STORED vertex order — not the one that was handed over.
        /// The gap between the two is the winding contract.
        var windingNormal: SIMD3<Scalar> {
            let n = cross3(b - a, c - a)
            let m2 = n.x * n.x + n.y * n.y + n.z * n.z
            guard m2 > Scalar(1e-30) else { return SIMD3(repeating: 0) }
            return n / m2.squareRoot()
        }

        var centroid: SIMD3<Scalar> {
            (a + b + c) / 3
        }

        var vertices: [SIMD3<Scalar>] { [a, b, c] }
    }

    private(set) var triangles: [Triangle] = []

    mutating func addLeafTriangle(_ a: LeafStripVertex<Scalar>,
                                  _ b: LeafStripVertex<Scalar>,
                                  _ c: LeafStripVertex<Scalar>,
                                  outward: SIMD3<Scalar>,
                                  geometricNormal: SIMD3<Scalar>) {
        triangles.append(Triangle(a: a.position, b: b.position, c: c.position,
                                  outward: outward, geometricNormal: geometricNormal))
    }

    var count: Int { triangles.count }

    /// Every position emitted, in order — the byte-identity handle a determinism test needs.
    var positions: [SIMD3<Scalar>] {
        triangles.flatMap { [$0.a, $0.b, $0.c] }
    }

    var allValuesAreFinite: Bool {
        for t in triangles {
            for v in [t.a, t.b, t.c, t.outward, t.geometricNormal] where
                !(v.x.isFinite && v.y.isFinite && v.z.isFinite) {
                return false
            }
        }
        return true
    }
}

// MARK: - Solid audits
//
// The pad's contract is "closed, watertight, consistently oriented", and none of that is visible in
// a triangle count. These are the same two questions `MeshAudit` and `Mesh3.signedVolume` ask of a
// mesh once it reaches the consuming app — asked here, on the primitive, where a failure names the
// constructor instead of the plant.

extension CollectingSink where Scalar == Double {

    /// Positions quantized to a micron, so vertices that MUST be the same vertex compare equal.
    ///
    /// A pad's midline column is shared bit-for-bit between the two mirrored halves, and its pinched
    /// base and tip collapse a whole row of grid points onto one position — but the crown's
    /// `sin(πv)` window evaluates to 1.2e-16 rather than 0 at `v = 1`, so the tip row lands within an
    /// attometre of itself rather than exactly on it. Comparing raw `SIMD3<Double>` would call those
    /// distinct and report a hole that is not there.
    static func key(_ p: SIMD3<Double>) -> SIMD3<Int64> {
        SIMD3(Int64((p.x * 1e6).rounded()),
              Int64((p.y * 1e6).rounded()),
              Int64((p.z * 1e6).rounded()))
    }

    /// How many triangles use each undirected edge. A closed manifold answers 2 for every edge.
    func undirectedEdgeUseCounts() -> [Edge: Int] {
        var counts: [Edge: Int] = [:]
        for t in triangles {
            let k = [Self.key(t.a), Self.key(t.b), Self.key(t.c)]
            for i in 0 ..< 3 {
                counts[Edge(k[i], k[(i + 1) % 3]), default: 0] += 1
            }
        }
        return counts
    }

    /// Divergence-theorem volume of the emitted soup. Positive means every face is wound outward
    /// consistently; a shell that is inside out reports the same magnitude negated, and one that is
    /// open reports a number that means nothing — which is why this is only ever read alongside the
    /// edge census.
    var signedVolume: Double {
        var v = 0.0
        for t in triangles {
            v += simd_dot(t.a, simd_cross(t.b, t.c))
        }
        return v / 6
    }

    var boundingBox: (min: SIMD3<Double>, max: SIMD3<Double>) {
        var lo = SIMD3<Double>(repeating: .infinity)
        var hi = SIMD3<Double>(repeating: -.infinity)
        for p in positions {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        return (lo, hi)
    }

    /// Extent of the soup along `axis`, in metres.
    func extent(along axis: SIMD3<Double>) -> Double {
        var lo = Double.infinity, hi = -Double.infinity
        for p in positions {
            let d = simd_dot(p, axis)
            lo = min(lo, d)
            hi = max(hi, d)
        }
        return hi - lo
    }

    /// An undirected edge between two quantized positions.
    struct Edge: Hashable {
        let lo: SIMD3<Int64>
        let hi: SIMD3<Int64>
        init(_ a: SIMD3<Int64>, _ b: SIMD3<Int64>) {
            // Order-independent, so a triangle and its neighbour agree they share this edge no
            // matter which way each of them walks it.
            if (a.x, a.y, a.z) <= (b.x, b.y, b.z) { lo = a; hi = b } else { lo = b; hi = a }
        }
    }
}

/// Every outline the catalog ships, so a census test cannot silently miss a new one.
///
/// A `static let` catalog has no `allCases`, which is the price of it being a table rather than an
/// enum (and worth paying — a new species must not be an edit to a shared switch). This list is the
/// one place that price is paid, and a new silhouette that is not added here simply is not covered:
/// keep it complete.
let shippedSilhouettes: [LeafSilhouette] = [
    .oak, .birch, .maple, .citrus, .strapBlade, .petal,
    .fiddleLeafFig, .monstera, .succulentPad, .fernLeaflet, .needle,
]

/// A placement in the XY plane presenting +Z, for tests that only care about shape.
func flatPlacement(width: Double = 0.30, height: Double = 0.40) -> LeafConstructor.Placement<Double> {
    LeafConstructor.Placement(position: SIMD3(0.7, 1.3, -0.4),
                              xAxis: SIMD3(1, 0, 0),
                              yAxis: SIMD3(0, 1, 0),
                              bentNormal: SIMD3(0, 0, 1),
                              width: width, height: height)
}
