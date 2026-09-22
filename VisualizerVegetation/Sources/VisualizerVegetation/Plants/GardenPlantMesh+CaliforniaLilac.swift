import Foundation
import simd
import VisualizerMaterials

/// *Ceanothus* 'Ray Hartman' — California lilac (DH-0796). A WOODY BRANCHING SHRUB in true blue.
///
/// What makes it read as this plant, trait by trait:
///  - **A real branching structure** (`californiaLilacBranches`): several trunks from the crown,
///    each forking three times into a rounded crown. Old wood is the grey-brown `stem` part; the
///    youngest twigs are flexible green (a colour group).
///  - **Glossy dark evergreen leaves, cupped**, crowding every young twig — a high fold on a small
///    oval, enough of them that the crown reads as a leafy mass, not bare sticks.
///  - **The flowers are PUFFBALLS** (`californiaLilacClusters`): a dense lumpy cone at every twig
///    tip and at many forks — a textured body of colour, never a petal.
///  - **A true vivid blue** — and a share of the clusters **spent and browned**, still hanging on.
///  - Clusters ride the blooms part (outdoor thin-sheet backlight, like every flower here).
///
/// Not modelled: the leaf's three-vein network (a few-millimetre relief a leaf card this small cannot
/// carry in geometry) and the pollinators that crowd a flowering shrub.
extension GardenPlantMesh {

    public struct CeanothusBranch: Sendable, Equatable {
        public var parent: Int?
        public var depth: Int
        public var start: Vec3
        public var end: Vec3
        public var rStart: Double
        public var rEnd: Double
    }

    public struct CeanothusCluster: Sendable, Equatable {
        public var branch: Int
        public var base: Vec3
        public var axis: Vec3
        public var length: Double
        public var spent: Bool
    }

    /// Trunks fork this many times; the last generation is green young growth.
    public static let ceanothusMaxDepth = 3
    /// A cluster's width as a fraction of its length — a cone, taller than wide.
    public static let ceanothusClusterAspect = 0.48

    public static let ceanothusBlue      = Vec3(0.20, 0.30, 0.78)
    public static let ceanothusSpent     = Vec3(0.40, 0.30, 0.20)
    public static let ceanothusYoungStem = Vec3(0.22, 0.30, 0.14)

    /// The branch armature — the single source the mesh, the clusters and the tests read.
    public static func californiaLilacBranches(spec: GardenPlantSpec) -> [CeanothusBranch] {
        var rng = speciesRNG(spec, lane: 0xCEA0)
        let size = spec.size, up = Vec3(0, 1, 0)
        let trunks = max(3, min(5, Int((3.5 * Double(spec.foliageDensity)).rounded())))
        var branches: [CeanothusBranch] = []

        func grow(parent: Int?, depth: Int, from: Vec3, dir: Vec3, length: Double, r: Double) {
            let index = branches.count
            let end = from + dir * length
            branches.append(CeanothusBranch(parent: parent, depth: depth, start: from, end: end,
                                            rStart: r, rEnd: r * 0.65))
            guard depth < ceanothusMaxDepth else { return }
            let splay = 0.40 + rng.unit() * 0.25
            let roll = rng.unit() * 2 * Double.pi
            let f = frame(yAxis: dir)
            for k in 0 ..< 2 {
                let a = roll + Double(k) * Double.pi
                let side = f.x * cos(a) + f.z * sin(a)
                // Biased upward, so the crown domes instead of splaying flat.
                let child = normalize3(normalize3(dir * cos(splay) + side * sin(splay)) + up * 0.22)
                grow(parent: index, depth: depth + 1, from: end, dir: child,
                     length: length * (0.74 + rng.unit() * 0.10), r: r * 0.65)
            }
        }

        let start = rng.unit() * 2 * Double.pi
        for i in 0 ..< trunks {
            let az = start + Double(i) * 2 * Double.pi / Double(trunks) + (rng.unit() - 0.5) * 0.5
            let out = Vec3(cos(az), 0, sin(az))
            let lean = 0.20 + rng.unit() * 0.22
            grow(parent: nil, depth: 0, from: out * (size * 0.03),
                 dir: normalize3(up * cos(lean) + out * sin(lean)),
                 length: size * (0.32 + rng.unit() * 0.05), r: size * 0.013)
        }
        return branches
    }

    /// The flower clusters: one at every twig tip, and at a density-dependent share of the forks
    /// below. Every fourth cluster (from a seeded start) is spent.
    public static func californiaLilacClusters(spec: GardenPlantSpec, branches: [CeanothusBranch]) -> [CeanothusCluster] {
        var rng = speciesRNG(spec, lane: 0xCEA1)
        let up = Vec3(0, 1, 0)
        let forkShare = min(1.0, Double(spec.foliageDensity) * 0.5)
        var clusters: [CeanothusCluster] = []
        for (i, b) in branches.enumerated() {
            let dir = normalize3(b.end - b.start)
            let length = spec.size * (0.055 + rng.unit() * 0.02)
            if b.depth == ceanothusMaxDepth {
                clusters.append(CeanothusCluster(branch: i, base: b.end, axis: normalize3(dir + up * 0.6),
                                                 length: length, spent: false))
            } else if b.depth == ceanothusMaxDepth - 1 && rng.unit() < forkShare {
                let f = frame(yAxis: dir)
                let a = rng.unit() * 2 * Double.pi
                clusters.append(CeanothusCluster(branch: i, base: b.end,
                                                 axis: normalize3(f.x * cos(a) + f.z * sin(a) + up * 0.7),
                                                 length: length, spent: false))
            }
        }
        let offset = Int(rng.next() % 4)
        for k in clusters.indices where (k + offset) % 4 == 0 { clusters[k].spent = true }
        return clusters
    }

    public static func californiaLilacParts(spec: GardenPlantSpec) -> Parts {
        let size = spec.size
        let up = Vec3(0, 1, 0)
        var wood = Mesh3(), young = Mesh3(), foliage = Mesh3()
        var groups = SpeciesColorGroups()
        var rng = speciesRNG(spec, lane: 0xCEA2)
        let branches = californiaLilacBranches(spec: spec)
        let twigLeaves = max(4, min(6, Int((5.0 * Double(spec.foliageDensity)).rounded())))

        for b in branches {
            let dir = normalize3(b.end - b.start)
            if b.depth == 0 {
                PottedPlantMesh.appendBentStemTube(&wood, from: b.start, to: b.end, rBase: b.rStart, rTip: b.rEnd,
                                                   bendDir: dir, bendAmount: 0.0, sides: 5)
            } else {
                // Open tubes: every fork starts inside its parent's end, and a shrub has hundreds.
                var tube = Mesh3()
                let twig = b.depth == ceanothusMaxDepth
                if tube.sweep(profile: .circle(radius: b.rStart, segments: twig ? 3 : 5),
                              along: [b.start - dir * b.rStart, b.end], scales: [1, b.rEnd / b.rStart],
                              capStart: false, capEnd: false) {
                    if twig { young.append(tube.smoothed()) } else { wood.append(tube.smoothed()) }
                }
            }

            // Leaves crowd the young growth and dress the fork below it.
            guard b.depth >= ceanothusMaxDepth - 1 else { continue }
            let count = b.depth == ceanothusMaxDepth ? twigLeaves : 3
            let f = frame(yAxis: dir)
            let phase = rng.unit() * 2 * Double.pi
            for k in 0 ..< count {
                let t = 0.15 + 0.80 * (Double(k) + 0.5) / Double(count)
                let a = phase + Double(k) * Phyllotaxis.goldenAngle
                let radial = normalize3(f.x * cos(a) + f.z * sin(a))
                let grow = normalize3(dir * 0.40 + radial * 0.9 + up * 0.15)
                var face = up - grow * dot3(up, grow)
                face = len3(face) > 1e-6 ? normalize3(face) : radial
                let len = size * (0.034 + rng.unit() * 0.010)
                let attach = b.start + (b.end - b.start) * t + radial * (b.rStart + (b.rEnd - b.rStart) * t)
                var leaf = Mesh3()
                BladeDescription.ceanothusLeaf.emit(into: &leaf, position: attach + grow * (len * 0.46),
                                                    xAxis: normalize3(cross3(grow, face)), yAxis: grow,
                                                    bentNormal: face, length: len, winding: .doubleSidedShell)
                foliage.append(leaf.smoothed())
            }
        }

        for c in californiaLilacClusters(spec: spec, branches: branches) {
            let h = c.length, r = h * ceanothusClusterAspect
            // A lumpy cone: revolve, push the wall in and out in an irregular pattern, THEN smooth —
            // the texture of hundreds of packed florets, at a cost a whole shrub of them can carry.
            let profile: [(r: Double, y: Double)] = [(0, 0), (r * 0.85, h * 0.18), (r, h * 0.42),
                                                     (r * 0.62, h * 0.74), (0, h)]
            var body = Mesh3()
            guard body.revolve(profile: profile.map { Mesh3.ProfilePoint(r: $0.r, y: $0.y) }, segments: 8) else { continue }
            let phase = rng.unit() * 2 * Double.pi
            body.positions = body.positions.map { p in
                let bump = 1 + 0.16 * sin(atan2(p.z, p.x) * 3 + p.y / h * 9 + phase)
                             * cos(atan2(p.z, p.x) * 2 - p.y / h * 5)
                return Vec3(p.x * bump, p.y, p.z * bump)
            }
            let f = frame(yAxis: c.axis)
            groups.append(body.smoothed().placed(xAxis: f.x, yAxis: f.y, zAxis: f.z, origin: c.base - c.axis * (h * 0.08)),
                          color: c.spent ? ceanothusSpent : ceanothusBlue)
        }

        var blooms = SpeciesColorGroups()
        blooms.append(young, color: ceanothusYoungStem)
        for g in groups.groups { blooms.append(g.mesh, color: g.color) }
        return Parts(stem: wood, foliage: foliage, blooms: blooms.groups)
    }
}

extension LeafSilhouette {
    /// A Ceanothus leaf: a small oval. Three controls — a shrub carries a few hundred of them.
    public static let ceanothusLeaf = LeafSilhouette(
        name: "ceanothusLeaf",
        hero: [(0.00, 0.00), (0.48, 1.00), (1.00, 0.00)])
}

extension BladeDescription {
    /// Small, glossy and CUPPED — a high fold, almost no droop.
    public static let ceanothusLeaf = BladeDescription(
        silhouette: .ceanothusLeaf, aspect: 0.58, fold: 0.55, curl: 0.04, subdivisions: 1)
}
