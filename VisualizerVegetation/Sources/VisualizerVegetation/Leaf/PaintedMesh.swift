import Foundation
import simd
import VisualizerMaterials

/// A `Mesh3` whose every vertex also carries its own ALBEDO — the colour channel `Mesh3` does not
/// have.
///
/// **Why the colour is decided here, at emission, and not by the host.** A houseplant used to reach
/// the render bridge as bare geometry, and the bridge stamped ONE green over every leaf vertex
/// (`FoliageVertexStamp.stamp`). One flat colour is the loudest of the "lo-fi" tells: real foliage
/// is a different shade leaf to leaf (the new leaf is lime, the old one nearly black-green), paler
/// underneath than on top, paler along the midrib, banded on a snake plant, blushed at an
/// echeveria's tips. Every one of those is a fact about WHERE on WHICH leaf a vertex sits — and only
/// the generator knows that. By the time a host sees a triangle soup, "this vertex is on the
/// midrib of the youngest leaf" is gone. So the generator paints, and the host just carries the
/// paint into its vertex format.
///
/// **How a colour stays with its vertex.** `Mesh3.addTriangle(_:_:_:outward:uv:)` is the winding
/// authority: it may SWAP the last two corners to face `outward`, and it DROPS a degenerate
/// triangle. A parallel colour array appended blindly would drift out of step on the first swap.
/// So each triangle rides its corner index (0, 1, 2) in the `uv` channel — which `addTriangle`
/// swaps together with the positions — and the colours are read back in the order the mesh actually
/// stored them. The uvs are left holding those small indices; no host reads `Mesh3.uvs` for
/// foliage (the Daydream bridge rebuilds planar UVs from position), and they cost nothing.
///
/// Every operation that keeps vertex ORDER keeps the colours aligned: `append`, `smoothed`, and
/// any in-place move of `mesh.positions`.
public struct PaintedMesh: Sendable, Equatable {
    public var mesh = Mesh3()
    /// One albedo (linear RGB) per `mesh` vertex — `colors.count == mesh.vertexCount`, always.
    public var colors: [Vec3] = []

    public init() {}
    public init(mesh: Mesh3, colors: [Vec3]) {
        precondition(colors.count == mesh.vertexCount, "a painted mesh needs one colour per vertex")
        self.mesh = mesh
        self.colors = colors
    }
    /// Paint an existing mesh one flat colour.
    public init(mesh: Mesh3, color: Vec3) {
        self.mesh = mesh
        self.colors = Array(repeating: color, count: mesh.vertexCount)
    }

    public var isEmpty: Bool { mesh.isEmpty }
    public var triangleCount: Int { mesh.triangleCount }

    /// Append one triangle whose three corners carry their own albedo, wound to face `outward`
    /// (see the type doc for how the colours follow `addTriangle`'s swap).
    public mutating func addTriangle(_ a: Vec3, _ b: Vec3, _ c: Vec3,
                                     colors ca: Vec3, _ cb: Vec3, _ cc: Vec3,
                                     outward: Vec3) {
        guard mesh.addTriangle(a, b, c, outward: outward,
                               uv: (Vec2(0, 0), Vec2(1, 0), Vec2(2, 0))) else { return }
        let corner = [ca, cb, cc]
        for uv in mesh.uvs.suffix(3) { colors.append(corner[Int(uv.x)]) }
    }

    /// Append a quad `a→b→c→d` (in order round the face) as two painted triangles.
    public mutating func addQuad(_ a: Vec3, _ b: Vec3, _ c: Vec3, _ d: Vec3,
                                 colors ca: Vec3, _ cb: Vec3, _ cc: Vec3, _ cd: Vec3,
                                 outward: Vec3) {
        addTriangle(a, b, c, colors: ca, cb, cc, outward: outward)
        addTriangle(a, c, d, colors: ca, cc, cd, outward: outward)
    }

    public mutating func append(_ other: PaintedMesh) {
        mesh.append(other.mesh)
        colors.append(contentsOf: other.colors)
    }

    /// Append plain geometry in one flat colour.
    public mutating func append(_ other: Mesh3, color: Vec3) {
        mesh.append(other)
        colors.append(contentsOf: Array(repeating: color, count: other.vertexCount))
    }

    /// Smooth the normals (`Mesh3.smoothed`) — vertex order, and so the paint, is untouched.
    public func smoothed(creaseDegrees: Double = 40) -> PaintedMesh {
        PaintedMesh(mesh: mesh.smoothed(creaseDegrees: creaseDegrees), colors: colors)
    }
}

/// A `LeafCardSink` that PAINTS: every vertex a constructor hands it (`LeafConstructor`,
/// `PadConstructor`) is coloured by `paint`, from the vertex's own place on its blade (`v` along,
/// `u` across, `side`) and which face it is on (`outward` against `upper`). This is how a thick
/// solid leaf — a snake-plant sword, an echeveria spoon — gets bands and blushes without the
/// constructor knowing colour exists.
public struct PaintingSink: LeafCardSink {
    public typealias Scalar = Double
    public var painted = PaintedMesh()
    /// The blade's upper-face direction — compared against each triangle's `outward` so `paint`
    /// knows the underside from the top (a pad's rim sits between the two).
    public var upper: Vec3
    public var paint: (LeafStripVertex<Double>, _ face: LeafFace) -> Vec3

    public init(upper: Vec3, paint: @escaping (LeafStripVertex<Double>, LeafFace) -> Vec3) {
        self.upper = upper
        self.paint = paint
    }

    public mutating func addLeafTriangle(_ a: LeafStripVertex<Double>,
                                         _ b: LeafStripVertex<Double>,
                                         _ c: LeafStripVertex<Double>,
                                         outward: Vec3,
                                         geometricNormal: Vec3) {
        let d = dot3(normalize3(outward), normalize3(upper))
        let face: LeafFace = d > 0.35 ? .upper : (d < -0.35 ? .lower : .rim)
        painted.addTriangle(a.position, b.position, c.position,
                            colors: paint(a, face), paint(b, face), paint(c, face),
                            outward: outward)
    }
}

/// Which face of a leaf a vertex belongs to — the painter's one piece of non-parametric context.
public enum LeafFace: Sendable, Equatable {
    case upper   // the adaxial face — the one the leaf presents to the light
    case lower   // the abaxial face — paler, greyer, the veins raised on it
    case rim     // a solid leaf's edge band (a pad's thickness)
}
