import Foundation
import simd
import VisualizerMaterials
import VisualizerRendering
import VisualizerVegetation

// plausibility: real — near plants are the engine's deterministic uncapped scatter of the
// foothill-meadow mix, one real library variant per plant, seated on the host's true ground; the
// mid band is a distance LOD of that SAME scatter (same plants, same positions, same colours —
// thinned with distance and widened by 1/√keep so the covered fraction the eye integrates is
// conserved). Everything here is a function of world coordinates + the field seed, so a cell
// built twice is the same cell.

/// A 16 m square of the field: the unit both the near instancing and the mid cards stream by.
public struct FloraCellKey: Hashable, Sendable, Comparable {
    public var ix: Int, iz: Int
    public init(ix: Int, iz: Int) { self.ix = ix; self.iz = iz }
    public static func < (a: Self, b: Self) -> Bool { (a.ix, a.iz) < (b.ix, b.iz) }
}

/// One plant, seated on the ground, in WORLD metres.
public struct FloraPlacement: Sendable {
    public var slot: Int
    public var position: SIMD3<Double>
    public var heading: Float
    public var scale: Float
    public var tint: Float
}

public enum FloraScatter {
    typealias W = WildflowerPlantingMesh
    static let mix = W.Mix.foothillMeadow
    public static let cellSize = 16

    /// The meadow plants of one cell. `ground` is the host's surface height at world (x, z);
    /// `coverage` is 0…1 and thins the cell where the host has something else (a house slab, a
    /// street, a deck) — 0 grows nothing, 1 the full sowing. Plan y = −world z, the convention
    /// `WildflowerPlantingMesh` seats its plants by.
    static func scatter(_ key: FloraCellKey, density: Double, seed: UInt64, heightScale: Double = 1,
                        ground: (Double, Double) -> Double,
                        coverage: (Double, Double) -> Float) -> [FloraPlacement] {
        let s = Double(cellSize)
        let x0 = Double(key.ix) * s, z0 = Double(key.iz) * s
        // Half-open by a hair so a plant on a shared edge belongs to one cell.
        let e = 1e-7
        let poly = [Vec2(x0, -(z0 + s) + e), Vec2(x0 + s - e, -(z0 + s) + e),
                    Vec2(x0 + s - e, -z0), Vec2(x0, -z0)]
        let plants = W.plants(polygon: poly, density: density, seed: seed, mix: mix, triangleCeiling: nil)
        var out: [FloraPlacement] = []
        out.reserveCapacity(plants.count)
        for p in plants {
            let x = p.position.x, z = -p.position.y
            let cov = coverage(x, z)
            if cov <= 0 { continue }
            if cov < 1, FloraHash.unit(Int(x * 211), Int(z * 197), seed &+ 9) >= Double(cov) { continue }
            let jitter = FloraHash.unit(Int(x * 97), Int(z * 89), seed &+ 5)
            out.append(FloraPlacement(
                slot: FloraLibrary.slot(p.kind, p.variant),
                // Sunk a few cm so a crown on a slope never shows daylight under it.
                position: SIMD3(x, ground(x, z) - 0.035, z),
                heading: Float(p.heading), scale: Float(p.scale * heightScale),
                tint: Float(0.93 + 0.14 * jitter)))
        }
        return out
    }

    // ── Mid band ─────────────────────────────────────────────────────

    /// **Canopy calibration** — the fraction of an impostor's naive (unoccluded, sun-facing) shade
    /// the REAL plants deliver at the same distance. Measured in Superbloom Hills 2026-09-22:
    /// real geometry read scene-linear luminance 0.154 in the 40–80 m band, the cards 0.349 at
    /// the same spot — 0.44, uniform across R/G/B. Re-measure whenever plants, sun or card
    /// shading change.
    static let canopyCalibration: Float = 0.44
    /// The same for the horizontal caps, which only read from above (tops, not flanks).
    static let capCalibration: Float = 0.75

    /// Mid-band LOD levels: the fraction of plants kept (the survivors widen by 1/√keep).
    static let midKeep: [Double] = [0.80, 0.58, 0.42, 0.30, 0.22, 0.16, 0.115, 0.083, 0.06, 0.043, 0.031]

    /// Keep-level for a mid cell at centre distance `d`: keep ≈ (32/d)^1.6, quantised to the nine
    /// gentle steps of `midKeep` — a hard 2× step per 16 m cell prints straight edges.
    static func midLevel(_ d: Double, radius: Double) -> Int? {
        guard d < radius else { return nil }
        let keep = min(0.8, pow(32 / max(d, 1), 1.6))
        return midKeep.firstIndex(where: { $0 <= keep * 1.18 }) ?? (midKeep.count - 1)
    }

    /// One mid-band cell as a single merged mesh of vertical cards, local to the cell corner:
    /// per kept plant, one foliage card (its mound's width × height) and one small card per
    /// flower-head cluster, at the cluster's true position and colour. Each cluster is crossed
    /// vertical cards plus a horizontal cap, two-sided.
    static func buildMid(cell: FloraCellKey, level: Int, placements: [FloraPlacement]) -> FloraLibrary.MeshData {
        let keep = midKeep[min(level, midKeep.count - 1)]
        let widen = Float(1 / keep.squareRoot())
        let x0 = Double(cell.ix * cellSize), z0 = Double(cell.iz * cellSize)
        var out = FloraLibrary.MeshData()
        out.vertices.reserveCapacity(Int(Double(placements.count) * keep * 16))
        let lib = FloraLibrary.plants
        let up = SIMD3<Float>(0, 1, 0)
        /// One quad with explicit front AND back faces, each with its OWN normal leaning the way
        /// its face looks but mostly UP (lit as the top of a canopy, not as a wall). A mesh-level
        /// double-sided flag would negate the normal on the back face — straight down, black.
        func quad(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>, _ d: SIMD3<Float>,
                  facing: SIMD3<Float>, color: SIMD4<Float>, tag: Float) {
            for side: Float in [1, -1] {
                let n = simd_normalize(facing * (0.35 * side) + up * 0.94)
                let base = UInt32(out.vertices.count)
                for p in [a, b, c, d] {
                    out.vertices.append(IlluminatoramaVertex(position: p, normal: n, uv: SIMD2(0.5, 0.5),
                                                             tangent: SIMD4(0, 0, 0, tag), color: color))
                }
                out.indices.append(contentsOf: side > 0 ? [base, base + 1, base + 2, base, base + 2, base + 3]
                                                        : [base, base + 2, base + 1, base, base + 3, base + 2])
            }
        }
        func card(center c: SIMD3<Float>, halfWidth hw: Float, halfHeight hh: Float, yaw: Float,
                  color: SIMD3<Float>, translucent: Bool, bottomAnchored: Bool, petal: Bool) {
            let col = SIMD4<Float>(color * canopyCalibration, translucent ? 0 : 1)
            let tag: Float = petal ? 1 : 0
            let lo = bottomAnchored ? c : c - SIMD3(0, hh, 0)
            let hi = lo + SIMD3(0, hh * 2, 0)
            for a in [yaw, yaw + .pi / 2] {
                let right = SIMD3<Float>(cos(a), 0, -sin(a))
                quad(lo - right * hw, lo + right * hw, hi + right * hw, hi - right * hw,
                     facing: SIMD3(sin(a), 0, cos(a)), color: col, tag: tag)
            }
            let r = SIMD3<Float>(cos(yaw), 0, -sin(yaw)) * hw * 0.8, f = SIMD3<Float>(sin(yaw), 0, cos(yaw)) * hw * 0.8
            let top = bottomAnchored ? hi : c
            let base = UInt32(out.vertices.count)
            let capCol = SIMD4<Float>(color * capCalibration, col.w)
            for p in [top - r + f, top + r + f, top + r - f, top - r - f] {
                out.vertices.append(IlluminatoramaVertex(position: p, normal: up, uv: SIMD2(0.5, 0.5),
                                                         tangent: SIMD4(0, 0, 0, tag), color: capCol))
            }
            // Counter-clockwise seen from above.
            out.indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        for (i, p) in placements.enumerated() {
            let u = FloraHash.unit(i, cell.ix &* 7919 &+ cell.iz, 77)
            guard u < keep, let bp = lib[p.slot] else { continue }
            let local = SIMD3<Float>(Float(p.position.x - x0), Float(p.position.y), Float(p.position.z - z0))
            let yaw = Float(FloraHash.unit(i, 3, UInt64(truncatingIfNeeded: cell.ix &+ 99)) - 0.5) * 1.2
            let c = cos(p.heading) * p.scale, s = sin(p.heading) * p.scale
            if bp.kind != .blueDicks {   // blue dicks' leaves are a few withered blades — no mound
                card(center: local, halfWidth: bp.foliageHalfWidth * p.scale * widen,
                     halfHeight: bp.foliageHeight * p.scale * 0.5, yaw: yaw,
                     color: bp.foliageColor * p.tint, translucent: true, bottomAnchored: true, petal: false)
            }
            for d in bp.dots {
                // The same heading/scale transform the near instance applies.
                let q = SIMD3<Float>(d.center.x * c + d.center.z * s, d.center.y * p.scale,
                                     -d.center.x * s + d.center.z * c)
                let hw = d.size * p.scale * 0.5 * widen
                card(center: local + q, halfWidth: hw, halfHeight: d.size * p.scale * 0.45, yaw: yaw,
                     color: d.color * p.tint, translucent: false, bottomAnchored: false, petal: true)
            }
        }
        return out
    }
}
