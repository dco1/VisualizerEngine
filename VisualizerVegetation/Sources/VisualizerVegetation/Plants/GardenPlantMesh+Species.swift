import Foundation
import simd
import VisualizerMaterials

// The shared kit the real-species garden plants (DH-0648) are built from. Every blade still goes
// through `LeafConstructor` — the ONE strip-stitch — so a species here is outlines, colours and a
// growth habit, never a second leaf or petal constructor.

// MARK: - Species outlines (data, per `LeafSilhouette`'s "a new species is a new constant")

extension LeafSilhouette {

    /// ONE thread-fine segment of a California poppy's finely dissected leaf. The leaf is built from
    /// many of these forking off a rachis — a feathery, fern-like read, never a broad blade. Three
    /// controls: the segment is a few millimetres wide and there are ~9 per leaf, so every extra
    /// control is multiplied across the whole mat.
    public static let poppyThread = LeafSilhouette(
        name: "poppyThread",
        hero: [(0.00, 0.00), (0.40, 1.00), (1.00, 0.00)])

    /// A California poppy petal: a broad FAN — narrow claw at the base, widening to a wide, softly
    /// squared rounded top. Four of them overlap into the cup.
    public static let poppyPetal = LeafSilhouette(
        name: "poppyPetal",
        hero: [(0.00, 0.00), (0.18, 0.46), (0.55, 0.92), (0.86, 0.96), (1.00, 0.00)])

    /// A Matilija petal: broad, rounded, a crepe-paper sheet nearly as wide as long.
    public static let matilijaPetal = LeafSilhouette(
        name: "matilijaPetal",
        hero: [(0.00, 0.00), (0.14, 0.50), (0.45, 0.94), (0.80, 0.90), (1.00, 0.00)])

    /// A Matilija leaf: deeply lobed, the lobes POINTED — the almost holly-jagged grey-green leaf.
    /// The corners here are wanted (unlike the monstera's), so it is left unrounded.
    public static let matilijaLeaf = LeafSilhouette(
        name: "matilijaLeaf",
        hero: [(0.00, 0.00), (0.22, 0.80), (0.36, 0.34), (0.54, 0.86),
               (0.68, 0.30), (0.84, 0.54), (1.00, 0.00)])
}

extension BladeDescription {
    /// The poppy's thread segment — narrow, rolled (high fold), barely drooping.
    public static let poppyThread = BladeDescription(
        silhouette: .poppyThread, aspect: 0.12, fold: 0.45, curl: 0.10, subdivisions: 1)

    /// The Matilija leaf — sessile on the stem, a soft droop.
    public static let matilijaLeaf = BladeDescription(
        silhouette: .matilijaLeaf, aspect: 0.72, fold: 0.26, curl: 0.18, subdivisions: 1)
}

extension GardenPlantMesh {

    // MARK: - Colour groups

    /// Accumulates meshes by albedo and hands back one `ColoredGroup` per colour, in first-use
    /// order — deterministic, and each colour is one draw group at the bridge.
    public struct SpeciesColorGroups {
        private var colors: [Vec3] = []
        private var meshes: [Mesh3] = []

        public mutating func append(_ mesh: Mesh3, color: Vec3) {
            guard !mesh.isEmpty else { return }
            if let i = colors.firstIndex(of: color) {
                meshes[i].append(mesh)
            } else {
                colors.append(color)
                meshes.append(mesh)
            }
        }

        public var groups: [PottedPlantMesh.ColoredGroup] {
            zip(colors, meshes).map { PottedPlantMesh.ColoredGroup(color: $0.0, mesh: $0.1) }
        }
    }

    // MARK: - Frames

    /// An orthonormal, right-handed frame whose Y is `axis` — for `Mesh3.placed`.
    public static func frame(yAxis axis: Vec3) -> (x: Vec3, y: Vec3, z: Vec3) {
        let y = normalize3(axis)
        let ref = abs(y.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
        let x = normalize3(cross3(y, ref))
        return (x, y, cross3(x, y))
    }

    /// A small revolved solid (a bud, a collar, a stamen boss) authored about +Y from `y = 0`, then
    /// stood on `axis` at `origin`. Smoothed on its own before it joins anything.
    public static func revolvedPart(_ profile: [(r: Double, y: Double)], segments: Int,
                             origin: Vec3, axis: Vec3) -> Mesh3 {
        var m = Mesh3()
        guard m.revolve(profile: profile.map { Mesh3.ProfilePoint(r: $0.r, y: $0.y) },
                        segments: segments) else { return Mesh3() }
        let f = frame(yAxis: axis)
        return m.smoothed().placed(xAxis: f.x, yAxis: f.y, zAxis: f.z, origin: origin)
    }

    // MARK: - Banded petals

    /// One colour band of a petal: it runs from the previous band's end to `vEnd` (0 = the claw at
    /// the base, 1 = the tip).
    public struct PetalBand {
        public var vEnd: Double
        public var color: Vec3
    }

    /// Emit ONE petal whose colour changes along its length — the poppy's deep-orange base paling
    /// toward the rim, a Matilija petal browned at a creased edge.
    ///
    /// `Mesh3` has no colour channel, so the petal is cut into bands, each band its own colour group.
    /// All bands share ONE placement, so the fold, curl and margin ripple are evaluated identically
    /// on both sides of every cut and the seam is geometrically closed. The whole petal is then
    /// displaced (optional crinkle) and smoothed as ONE part before it is split, so the shading
    /// normal is continuous across the seam too — smoothing each band alone would crease it.
    public static func emitBandedPetal(into groups: inout SpeciesColorGroups,
                                placement: LeafConstructor.Placement<Double>,
                                silhouette: LeafSilhouette, subdivisions: Int,
                                fold: Double, curl: Double, bands: [PetalBand],
                                waveAmplitude: Double = 0, wavePhase: Double = 0,
                                displace: ((Vec3) -> Vec3)? = nil) {
        let margin = silhouette.margin(tier: .hero, subdivisions: subdivisions)
        var petal = Mesh3()
        var ranges: [(Range<Int>, Vec3)] = []
        var vStart = 0.0
        for band in bands {
            let sub = marginSlice(margin, from: vStart, to: band.vEnd)
            let before = petal.triangleCount
            if sub.count >= 2 {
                LeafConstructor.emitBlade(into: &petal, placement: placement, margin: sub,
                                          fold: fold, curl: curl, winding: .doubleSidedShell,
                                          waveAmplitude: waveAmplitude, wavePhase: wavePhase)
            }
            ranges.append((before ..< petal.triangleCount, band.color))
            vStart = band.vEnd
        }
        if let displace {
            petal.positions = petal.positions.map(displace)
        }
        let smooth = petal.smoothed()
        for (range, color) in ranges where !range.isEmpty {
            groups.append(triangles(of: smooth, in: range), color: color)
        }
    }

    /// The half-margin between `v0` and `v1`, with the width interpolated at both cuts.
    public static func marginSlice(_ margin: [LeafSilhouette.Control],
                            from v0: Double, to v1: Double) -> [LeafSilhouette.Control] {
        guard v1 - v0 > 1e-6, margin.count >= 2 else { return [] }
        func u(at v: Double) -> Double {
            for i in 0 ..< margin.count - 1 {
                let a = margin[i], b = margin[i + 1]
                if v >= a.v && v <= b.v {
                    let span = b.v - a.v
                    return span < 1e-9 ? b.u : a.u + (b.u - a.u) * (v - a.v) / span
                }
            }
            return margin[margin.count - 1].u
        }
        var out = [LeafSilhouette.Control(v0, u(at: v0))]
        for c in margin where c.v > v0 + 1e-6 && c.v < v1 - 1e-6 { out.append(c) }
        out.append(LeafSilhouette.Control(v1, u(at: v1)))
        return out
    }

    /// A copy of the triangles `range` of `mesh` (vertices re-indexed).
    public static func triangles(of mesh: Mesh3, in range: Range<Int>) -> Mesh3 {
        var positions: [Vec3] = [], normals: [Vec3] = [], uvs: [Vec2] = [], indices: [UInt32] = []
        let hasUV = mesh.uvs.count == mesh.positions.count
        for t in range {
            for k in 0 ..< 3 {
                let i = Int(mesh.indices[t * 3 + k])
                indices.append(UInt32(positions.count))
                positions.append(mesh.positions[i])
                normals.append(mesh.normals[i])
                uvs.append(hasUV ? mesh.uvs[i] : .zero)
            }
        }
        return Mesh3(positions: positions, normals: normals, uvs: uvs, indices: indices)
    }

    /// A petal placement for a bloom on `axis` at `center`: the petal leaves the centre along
    /// `radial`, pitched `pitch` radians off the bloom axis (0 = standing up, π/2 = lying flat),
    /// presenting its INNER face (toward the axis and up) — the face a fold and a negative curl cup.
    public static func petalPlacement(center: Vec3, axis: Vec3, radial: Vec3, pitch: Double,
                               length: Double, width: Double, twist: Double = 0) -> LeafConstructor.Placement<Double> {
        let n = normalize3(axis)
        let yAxis = normalize3(n * cos(pitch) + radial * sin(pitch))
        var inner = normalize3(n * sin(pitch) - radial * cos(pitch))
        var xAxis = normalize3(cross3(yAxis, inner))
        if twist != 0 {   // roll the petal about its own midrib (a furled, spiralled closed flower)
            let c = cos(twist), s = sin(twist)
            (xAxis, inner) = (normalize3(xAxis * c + inner * s), normalize3(inner * c - xAxis * s))
        }
        return .init(position: center + yAxis * (length * 0.46), xAxis: xAxis, yAxis: yAxis,
                     bentNormal: inner, width: width, height: length)
    }

    /// The seeded generator every species uses, on its own lane so two species with one seed differ.
    public static func speciesRNG(_ spec: GardenPlantSpec, lane: UInt64) -> PottedPlantMesh.SplitMix {
        PottedPlantMesh.SplitMix(spec.seed &* 0x2545F4914F6CDD1D &+ lane &* 0x9E3779B97F4A7C15)
    }
}
