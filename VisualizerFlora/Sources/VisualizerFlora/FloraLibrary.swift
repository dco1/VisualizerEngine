import Foundation
import simd
import VisualizerMaterials
import VisualizerRendering
import VisualizerVegetation

// plausibility: real — every near plant IS a `WildflowerPlantingMesh` library variant (the
// engine's sown California mix: real leaflet mounds, stems, cupped poppy petals, lupine racemes,
// blue-dicks umbels, wild-oat panicles), drawn as a GPU instance. The "lite" twin is the SAME
// variant with most of its foliage leaflets dropped (the survivors grown to hold the mound's
// silhouette) — a distance LOD past ~12 m where a leaflet is a pixel, not a different plant: its
// flowers and stems are the full variant's, vertex for vertex, so nothing pops when it swaps.
// The mid-band dot/card impostors are DERIVED from the same variants (each dot is a cluster of a
// variant's real petal triangles, at their real position, in their real colour) and are only
// drawn for plants of the same deterministic scatter — a level-of-detail of this exact field,
// not a separate fake. Wind sway is the engine's per-vertex foliage wind (sway weights packed in
// tangent, gated per instance by `windScale`).

/// The wildflower library, baked once into GPU-ready vertex arrays.
///
/// Every variant of every kind becomes ONE mesh with its colour layers folded into the
/// vertex colour (alpha 0 = thin translucent sheet that takes the backlight, 1 = opaque), so a
/// placed plant is one instance, not one per colour layer. Two LODs per variant (`full`,
/// `lite`) plus the mid-band impostor recipe (`dots`, `foliage`).
public enum FloraLibrary {
    package typealias W = WildflowerPlantingMesh

    /// A baked mesh: vertices + indices, ready for `IlluminatoramaMesh(device:vertices:indices:)`.
    public struct MeshData: @unchecked Sendable {
        public var vertices: [IlluminatoramaVertex] = []
        public var indices: [UInt32] = []
        public var triangleCount: Int { indices.count / 3 }
        public init(vertices: [IlluminatoramaVertex] = [], indices: [UInt32] = []) {
            self.vertices = vertices
            self.indices = indices
        }
    }

    /// One flower-head cluster of a variant, for the mid-band impostor.
    package struct Dot: Sendable {
        var center: SIMD3<Float>
        var color: SIMD3<Float>
        var size: Float
    }

    package struct Plant: @unchecked Sendable {
        var kind: W.Kind
        var variant: Int
        var full: MeshData
        var lite: MeshData
        var dots: [Dot]
        /// The foliage mound as a card: half-width, height, colour.
        var foliageHalfWidth: Float
        var foliageHeight: Float
        var foliageColor: SIMD3<Float>
    }

    /// Stable index of (kind, variant) — also the mesh-kind suffix.
    package static func slot(_ kind: W.Kind, _ variant: Int) -> Int { kind.rawValue * 16 + variant }

    /// Built once per process (the engine library itself is a process-wide static).
    package static let plants: [Int: Plant] = {
        var out: [Int: Plant] = [:]
        for kind in W.Mix.foothillMeadow.kinds {
            let hero = W.heroLibrary[kind] ?? []
            for (v, variant) in (W.library[kind] ?? []).enumerated() {
                out[slot(kind, v)] = bake(kind: kind, index: v, variant: variant,
                                          hero: v < hero.count ? hero[v] : variant)
            }
        }
        return out
    }()

    // ── Tunables ─────────────────────────────────────────────────────
    /// Fraction of foliage leaflets the lite LOD keeps, and how much it grows them.
    static let liteFoliageKeep = 0.30
    static let liteFoliageGrow = 1.45
    /// Tip sway (m) of a 0.55 m head at full gust.
    static let swayAtHeadHeight = 0.085
    /// Mid-band flower clusters: triangles whose centroids fall within this radius merge.
    static let dotClusterRadius = 0.07

    private static func isFoliage(_ layer: W.Layer) -> Bool {
        layer.translucent && (W.foliageGreens.contains(layer.color) || layer.color == W.stemGreen)
    }
    private static func isGreen(_ layer: W.Layer) -> Bool {
        W.foliageGreens.contains(layer.color) || layer.color == W.stemGreen
    }

    /// `hero` is the same variant at `.hero` petal detail (rounded, cupped poppy petals) — the
    /// FULL LOD a camera among the flowers sees; the lite LOD keeps the standard petals.
    private static func bake(kind: W.Kind, index: Int, variant: W.Variant, hero: W.Variant) -> Plant {
        let phaseSeed = Double(slot(kind, index)) * 1.618
        var full = MeshData(), lite = MeshData()
        for layer in hero.layers {
            append(layer.mesh, layer: layer, phaseSeed: phaseSeed, into: &full)
        }
        for layer in variant.layers {
            let liteMesh = isFoliage(layer) ? thinned(layer.mesh, keep: liteFoliageKeep,
                                                       grow: liteFoliageGrow, seed: UInt64(slot(kind, index)))
                                            : layer.mesh
            append(liteMesh, layer: layer, phaseSeed: phaseSeed, into: &lite)
        }

        // Flower-head clusters (area-weighted centroid + colour) for the impostor.
        struct Acc { var c: SIMD3<Double>; var col: SIMD3<Double>; var area: Double }
        var accs: [Acc] = []
        var greens: [SIMD3<Double>] = []
        var greenColor = SIMD3<Double>(repeating: 0), greenArea = 0.0
        for layer in variant.layers {
            let m = layer.mesh
            for t in stride(from: 0, to: m.indices.count, by: 3) {
                let a = m.positions[Int(m.indices[t])], b = m.positions[Int(m.indices[t + 1])]
                let c = m.positions[Int(m.indices[t + 2])]
                let area = simd_length(simd_cross(b - a, c - a)) * 0.5
                let cen = (a + b + c) / 3
                if isGreen(layer) {
                    greens.append(cen); greenColor += layer.color * area; greenArea += area
                    continue
                }
                if let i = accs.firstIndex(where: { simd_distance($0.c / max($0.area, 1e-9), cen) < dotClusterRadius }) {
                    accs[i].c += cen * area; accs[i].col += layer.color * area; accs[i].area += area
                } else {
                    accs.append(Acc(c: cen * area, col: layer.color * area, area: area))
                }
            }
        }
        let dots = accs.filter { $0.area > 1e-6 }.map { a in
            Dot(center: SIMD3<Float>(a.c / a.area), color: SIMD3<Float>(a.col / a.area),
                size: Float(max(0.035, a.area.squareRoot() * 1.25)))
        }
        let radii = greens.map { simd_length(SIMD2($0.x, $0.z)) }.sorted()
        let heights = greens.map(\.y).sorted()
        func pct(_ a: [Double], _ p: Double) -> Double { a.isEmpty ? 0 : a[min(a.count - 1, Int(Double(a.count) * p))] }
        return Plant(kind: kind, variant: index, full: full, lite: lite, dots: dots,
                     foliageHalfWidth: Float(max(0.08, pct(radii, 0.8))),
                     foliageHeight: Float(max(0.08, pct(heights, 0.85))),
                     foliageColor: springColor(greenArea > 0 ? greenColor / greenArea : W.foliageGreens[1]))
    }

    /// One engine plant variant as a single instance mesh with its colours re-mapped by `recolor`
    /// (no petal tag — shared with Napa Valley Ultra, which cures the engine's tuft / wild-oat
    /// grasses to late-summer straw). `lite` thins the foliage exactly as this library's lite LOD.
    public static func bakeRecolored(_ variant: WildflowerPlantingMesh.Variant, lite: Bool, seed: UInt64,
                                     recolor: (Vec3) -> SIMD3<Float>) -> MeshData {
        var out = MeshData()
        for layer in variant.layers {
            let m = lite && isFoliage(layer) ? thinned(layer.mesh, keep: liteFoliageKeep, grow: liteFoliageGrow, seed: seed)
                                             : layer.mesh
            append(m, layer: layer, phaseSeed: Double(seed) * 1.618, into: &out, color: recolor(layer.color), petal: false)
        }
        return out
    }

    private static func append(_ m: Mesh3, layer: W.Layer, phaseSeed: Double, into out: inout MeshData,
                               color: SIMD3<Float>? = nil, petal: Bool? = nil) {
        guard !m.isEmpty else { return }
        let base = UInt32(out.vertices.count)
        let col = SIMD4<Float>(color ?? springColor(layer.color), layer.translucent ? 0 : 1)
        let petalTag: Float = (petal ?? !isGreen(layer)) ? 1 : 0
        out.vertices.reserveCapacity(out.vertices.count + m.positions.count)
        for (i, p) in m.positions.enumerated() {
            // Sway grows with height above the crown: the base is rooted, the head swings.
            let t = min(1.4, max(0, p.y / 0.55))
            let sway = Float(swayAtHeadHeight * t * t)
            // Phase varies across the plant so neighbouring heads don't swing in lock-step.
            let phase = Float(phaseSeed + p.x * 9.0 + p.z * 7.0)
            let flutter = Float(layer.translucent ? 0.30 * t : 0.08 * t)
            // tangent.w = 1 TAGS petal vertices for the engine's hue cycle (flower colour show):
            // blooms change colour, leaves and stems do not.
            out.vertices.append(IlluminatoramaVertex(
                position: SIMD3<Float>(p), normal: SIMD3<Float>(m.normals[i]),
                uv: SIMD2<Float>(0.5, 0.5),
                tangent: SIMD4<Float>(sway, phase, flutter, petalTag), color: col))
        }
        out.indices.append(contentsOf: m.indices.map { $0 + base })
    }

    /// The engine palette's greens are a sun-dried garden olive; a foothill in a wet spring is a
    /// vivid, cooler green. Flowers keep their species colour.
    static func springColor(_ c: Vec3) -> SIMD3<Float> {
        // The engine's palest poppy orange reads yellow-ochre against the photo's deep glowing
        // orange: pull every poppy toward it (green down, blue out) — this scene's bake only.
        if W.poppyOranges.contains(c) { return SIMD3<Float>(Float(c.x), Float(c.y * 0.78), Float(c.z * 0.3)) }
        guard W.foliageGreens.contains(c) || c == W.stemGreen else { return SIMD3<Float>(c) }
        return SIMD3<Float>(Float(c.x * 0.50), Float(c.y * 0.95), Float(c.z * 0.40))
    }

    /// Keep a deterministic fraction of a layer's CONNECTED COMPONENTS (its individual leaflets
    /// and blades), growing each survivor about its own centroid.
    private static func thinned(_ m: Mesh3, keep: Double, grow: Double, seed: UInt64) -> Mesh3 {
        var parent = Array(0 ..< m.positions.count)
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { parent[r] = parent[parent[r]]; r = parent[r] }
            return r
        }
        for t in stride(from: 0, to: m.indices.count, by: 3) {
            let a = find(Int(m.indices[t])), b = find(Int(m.indices[t + 1])), c = find(Int(m.indices[t + 2]))
            parent[b] = a; parent[find(c)] = a
        }
        var rootOrder: [Int: Int] = [:]
        var centroid: [Int: (SIMD3<Double>, Double)] = [:]
        for i in m.positions.indices {
            let r = find(i)
            if rootOrder[r] == nil { rootOrder[r] = rootOrder.count }
            let e = centroid[r] ?? (.zero, 0)
            centroid[r] = (e.0 + m.positions[i], e.1 + 1)
        }
        var out = Mesh3()
        var remap = [Int: UInt32]()
        for t in stride(from: 0, to: m.indices.count, by: 3) {
            let r = find(Int(m.indices[t]))
            let ord = rootOrder[r]!
            guard FloraHash.unit(ord, 7, seed) < keep else { continue }
            let (sum, n) = centroid[r]!
            let cen = sum / n
            for k in 0 ..< 3 {
                let vi = Int(m.indices[t + k])
                if let e = remap[vi] { out.indices.append(e); continue }
                let e = UInt32(out.positions.count)
                remap[vi] = e
                out.positions.append(cen + (m.positions[vi] - cen) * grow)
                out.normals.append(m.normals[vi])
                if !m.uvs.isEmpty { out.uvs.append(m.uvs[vi]) }
                out.indices.append(e)
            }
        }
        return out
    }
}
