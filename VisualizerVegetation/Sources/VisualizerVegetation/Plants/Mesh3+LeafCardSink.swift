//  Mesh3 as a receiver for the shared vegetation constructor (DH-0651).
//
//  `VisualizerVegetation.LeafConstructor` stitches ONE blade — leaf, petal, fern pinnule,
//  succulent paddle — and hands each triangle to a `LeafCardSink`. This is the `Mesh3` adapter:
//  the whole conformance is a forward, and that is the point.

import simd
import VisualizerMaterials

/// `Mesh3` receives blades from `LeafConstructor`.
///
/// **It forwards to the existing `Mesh3.addTriangle(_:_:_:outward:)` and nothing else.** That
/// function is the single choke point where winding is *declared* rather than hand-authored
/// (CLAUDE.md § Winding): it rewinds the triangle so its geometric normal agrees with `outward`,
/// derives the stored normal from that winding, and drops degenerates. Routing the constructor's
/// triangles through it is what keeps `MeshAudit`'s contract — zero winding mismatches, zero
/// hand-authored normals — true for plant meshes exactly as it was when the strip-stitch lived
/// here. A conformance that appended positions/normals directly would be a second winding
/// authority, which is the defect this whole module exists to remove.
///
/// **The parametric fields (`v` / `u` / `side`) and `geometricNormal` are deliberately ignored.**
/// They exist for the yard tree's sink, which needs them to evaluate a midrib→margin→tip colour
/// gradient and to blend a shading normal toward the bent normal. `Mesh3` has no colour channel
/// and shades FLAT by construction — `addTriangle` writes one face normal to all three vertices —
/// so there is nothing here for either field to feed. `geometricNormal` in particular is only the
/// cross product `addTriangle` is about to recompute from the same three points; taking it would
/// duplicate the value, not save it. (A plant that wants smooth normals gets them the way every
/// other curved part in this package does: `Mesh3.smoothed(creaseDegrees:)`, per part, after the
/// blades are in.)
extension Mesh3: LeafCardSink {
    public typealias Scalar = Double

    public mutating func addLeafTriangle(_ a: LeafStripVertex<Double>,
                                         _ b: LeafStripVertex<Double>,
                                         _ c: LeafStripVertex<Double>,
                                         outward: Vec3,
                                         geometricNormal: Vec3) {
        addTriangle(a.position, b.position, c.position, outward: outward)
    }
}
