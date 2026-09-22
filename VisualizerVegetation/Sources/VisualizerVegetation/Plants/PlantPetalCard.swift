import Foundation
import simd
import VisualizerMaterials

/// Petal-card geometry for the `.flowers` bouquet style — a petal is just the shared blade
/// parameterized with the `.petal` silhouette (a rounder, lobe-free margin) + a softer fold. Not a
/// fork of the strip-stitch loop: `emitPetal` delegates to `VisualizerVegetation`'s
/// `LeafConstructor`, the ONE blade constructor, so leaves and petals can never drift apart
/// (DH-0651; CLAUDE.md [[feedback-single-source-of-truth]]).
///
/// A bloom is a small CENTER disc (a contrasting button) ringed by N petals in one or two rings,
/// each petal a rounded curved card cupped toward the bloom's outward normal. Petals + the disc
/// are tagged alpha-0 for foliage-SSS at the render bridge (soft backlit glow), exactly like the
/// leaf cards. Colours (vivid per-bloom petal + contrasting center) are chosen by the caller from
/// a seeded palette and stamped per mesh-group at the bridge.
public enum PlantPetalCard {

    /// Rounded-obovate petal half-margin `(v, u)` — a pass-through to the engine's `.petal`
    /// outline (the single source). Retained for any caller/test that read the point list directly.
    public static var petalMargin: [(v: Double, u: Double)] {
        LeafSilhouette.petal.controls(tier: .hero).map { (v: $0.v, u: $0.u) }
    }

    /// Emit ONE petal — delegates to `LeafConstructor.emitBlade` with the `.petal` silhouette, a gentle
    /// cross `fold` (default 0.24), and a longitudinal `curl` (default 0.30) that bows the petal tip
    /// out of plane so the bloom reads as a CUPPED 3D flower, not a pinwheel of flat cards (DH-0488).
    ///
    /// `subdivisions: 3` matches every other blade in this file family (fig/monstera/succulent all
    /// use 3 for the same reason). The OLD call site never passed this parameter at all, which
    /// silently took `LeafConstructor.emitBlade`'s default of 1 — the "byte-identity escape hatch"
    /// that returns `.petal`'s raw 6-point control polygon completely unrounded. A 6-point straight-
    /// edged polygon is a literal faceted hexagon, which is exactly why every bloom rendered as a
    /// cut-gemstone/diamond shape instead of a soft rounded petal (Danny: "flowers read as faceted
    /// gemstones"). This was a silent omission, not a deliberate choice — nothing in this file or
    /// `emitBloom` ever discussed keeping petals unrounded.
    public static func emitPetal(_ mesh: inout Mesh3,
                          pos: Vec3, xAxis: Vec3, yAxis: Vec3, normal bentNormal: Vec3,
                          w: Double, h: Double, fold: Double = 0.24, curl: Double = 0.30,
                          subdivisions: Int = 2) {
        LeafConstructor.emitBlade(
            into: &mesh,
            placement: .init(position: pos, xAxis: xAxis, yAxis: yAxis,
                             bentNormal: bentNormal, width: w, height: h),
            silhouette: .petal,
            tier: .hero,
            subdivisions: subdivisions,
            fold: fold,
            curl: curl,
            winding: .doubleSidedShell)
    }

    /// Emit a small double-sided center disc (the bloom's button/eye) of radius `r` at `center`,
    /// facing `normal`. A low fan — front + back so it reads from either side. Contrasting colour
    /// is stamped by the caller's mesh group.
    public static func emitCenter(_ mesh: inout Mesh3, center: Vec3, normal: Vec3, r: Double, segments: Int = 10) {
        // Build an in-plane basis from `normal`.
        let n = normalize3(normal)
        let ref = abs(n.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let u = normalize3(cross3(ref, n))
        let v = normalize3(cross3(n, u))
        let twoPi = Double.pi * 2
        func rim(_ i: Int) -> Vec3 {
            let a = twoPi * Double(i) / Double(segments)
            return center + (u * cos(a) + v * sin(a)) * r
        }
        for i in 0 ..< segments {
            let p0 = rim(i), p1 = rim((i + 1) % segments)
            mesh.addTriangle(center, p0, p1, outward: n)
            let back = Vec3(-n.x, -n.y, -n.z)
            mesh.addTriangle(center, p1, p0, outward: back)
        }
    }
}
