//  GuitarCarve.swift — item L's model: a block of wood carved into a rock-and-roll guitar, one kit tool a step.
//
//  `GuitarCarve.steps()` returns the whole build in order; every step's `mesh` is the result AFTER that step and
//  `tool` names the kit tool that did it. Coordinates are metres: x runs along the guitar (body at 0…0.5, neck on to
//  the headstock at ~1.05), y across it, z through it (the top face is +z). A cube/block of 1.06 × 0.36 × 0.05 m is
//  the start; nothing is modelled from scratch except where a step says "adds" (parts are added, not carved).
//
//  Guarantees: deterministic (no randomness; GPU and CPU twins agree within a cell); every step flagged `isSolid`
//  is closed by position (each edge in two faces), strings and the loose parts of step 6+ included; `carves` steps
//  strictly lose volume. Not guaranteed: sharp edges in the SDF/sculpt steps (a cell, ~5 mm, rounds them), a
//  physically sensible guitar (the proportions are a bold double-cutaway electric: Les Paul / SG-ish, 6-in-line).
//  Files: this one (types, write), GuitarCarve+Body.swift (steps 1–5), GuitarCarve+Parts.swift (6–8).

import Foundation
import simd

public struct GuitarStep: Sendable {
    public var title: String
    /// The result after this step.
    public var mesh: Mesh3
    /// The kit tool(s) that did it.
    public var tool: String
    /// Every edge shared by two faces (by position): a solid, safe to boolean / print.
    public var isSolid: Bool
    /// This step only removes material (volume must drop from the previous step).
    public var carves: Bool
    public init(title: String, mesh: Mesh3, tool: String, isSolid: Bool = true, carves: Bool = false) {
        self.title = title; self.mesh = mesh; self.tool = tool; self.isSolid = isSolid; self.carves = carves
    }
}

public enum GuitarCarve {
    /// The starting block, m.
    public static let blockLo = Vec3(0, -0.18, -0.025)
    public static let blockHi = Vec3(1.06, 0.18, 0.025)
    /// Neck's top surface z after step 4 and the fretboard's top after step 6.
    static let neckTop = 0.012
    static let neckBottom = -0.010

    public static func steps() -> [GuitarStep] {
        var out: [GuitarStep] = []
        let block = makeBlock()
        out.append(GuitarStep(title: "A block of wood", mesh: block, tool: "HalfEdgeMesh.box → toMesh3"))
        let body = carveOutline()
        out.append(GuitarStep(title: "Carve the guitar outline", mesh: body, tool: "SDF.intersecting + SDF.mesh (surface nets)", carves: true))
        let edged = roundEdges(body)
        out.append(GuitarStep(title: "Chamfer the body edges", mesh: edged, tool: "HalfEdgeMesh.inset + pushPull (face region)", carves: true))
        let necked = carveNeck(edged)
        out.append(GuitarStep(title: "Carve the neck and headstock", mesh: necked, tool: "Mesh3.booleanDifference (robust)", carves: true))
        let contoured = contour(necked)
        out.append(GuitarStep(title: "Contour the belly and forearm", mesh: contoured, tool: "SculptVolume brushes (GPU) + Mesh3.decimated", carves: true))
        let fitted = fitHardware(contoured)
        out.append(GuitarStep(title: "Pickup slots, pickguard, bridge, knobs", mesh: fitted,
                              tool: "Mesh3.booleanDifference/Union (robust), Mesh3.loft, ModelGraph array instancing"))
        let strung = addStrings(fitted)
        out.append(GuitarStep(title: "Strings and tuning pegs", mesh: strung, tool: "Mesh3.tube (sweep), Mesh3.mirrored / arrayed"))
        let done = finish(body: fitted, parts: stringParts())
        out.append(GuitarStep(title: "Bend the neck, grain the wood, unwrap UVs", mesh: done,
                              tool: "ModifierStack (SimpleDeform bend + Displace noise) + UV box projection / pack"))
        return out
    }

    // MARK: export

    /// Write every step as `NN-<slug>.obj` plus `steps.json` (title, tool, file, triangles, volume) into `folder`.
    @discardableResult
    public static func write(to folder: URL, steps given: [GuitarStep]? = nil) throws -> [URL] {
        let all = given ?? steps()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var urls: [URL] = [], rows: [[String: Any]] = []
        for (i, s) in all.enumerated() {
            let slug = s.title.lowercased().map { $0.isLetter || $0.isNumber ? String($0) : "-" }.joined()
                .split(separator: "-").joined(separator: "-")
            let name = String(format: "%02d-%@.obj", i + 1, String(slug.prefix(40)))
            let url = folder.appendingPathComponent(name)
            var opts = MeshIO.WriteOptions()
            opts.names = ["guitar-step-\(i + 1)"]
            try MeshIO.write([s.mesh], format: .obj, options: opts).write(to: url)
            urls.append(url)
            rows.append(["step": i + 1, "title": s.title, "tool": s.tool, "file": name, "triangles": s.mesh.triangleCount,
                         "volume_m3": s.mesh.signedVolume, "solid": s.isSolid, "carves": s.carves])
        }
        let json = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
        let jurl = folder.appendingPathComponent("steps.json")
        try json.write(to: jurl)
        return urls + [jurl]
    }

    // MARK: shared helpers

    /// Edges by position that are not in exactly two faces (0 = closed).
    public static func openEdges(_ m: Mesh3) -> Int {
        func key(_ p: Vec3) -> SIMD3<Int64> { SIMD3(Int64((p.x * 1e6).rounded()), Int64((p.y * 1e6).rounded()), Int64((p.z * 1e6).rounded())) }
        struct E: Hashable { let a: SIMD3<Int64>, b: SIMD3<Int64> }
        var count: [E: Int] = [:]
        var t = 0
        while t + 2 < m.indices.count {
            let k = (0..<3).map { key(m.positions[Int(m.indices[t + $0])]) }
            for c in 0..<3 where k[c] != k[(c + 1) % 3] {
                let a = k[c], b = k[(c + 1) % 3]
                let lessEq = (a.x, a.y, a.z) < (b.x, b.y, b.z)
                count[lessEq ? E(a: a, b: b) : E(a: b, b: a), default: 0] += 1
            }
            t += 3
        }
        return count.values.filter { $0 != 2 }.count
    }

    /// A closed axis-aligned box as an indexed Mesh3 (outward wound), from the half-edge kit's box.
    static func boxMesh(lo: Vec3, hi: Vec3) -> Mesh3 {
        HalfEdgeMesh.box(size: hi - lo, center: (lo + hi) * 0.5).toMesh3(flatNormals: true)
    }
}
