//  GuitarCarve+Parts.swift — steps 6–8: hardware (graph), strings and pegs, finishing stack. See GuitarCarve.swift.

import Foundation
import simd

extension GuitarCarve {
    /// Where things sit along x: the nut, and a 0.65 m scale length back to the bridge.
    static let nutX = 0.90
    static let scale = 0.65
    static var bridgeX: Double { nutX - scale }
    static let pickupX = [0.32, 0.415]
    static let fretTop = 0.0175

    // MARK: 6. hardware through the procedural graph

    static func fitHardware(_ body: Mesh3) -> Mesh3 {
        var m = body
        // pickguard: a shallow recess, then two humbucker slots through it; the plate (with holes a millimetre wider
        // than the slots) is glued into the recess, proud of the top, and a cover sits in each slot
        m = m.booleanDifference(boxMesh(lo: Vec3(0.22, -0.105, 0.0215), hi: Vec3(0.47, 0.085, 0.1)))
        var plate = boxMesh(lo: Vec3(0.2185, -0.1065, 0.0210), hi: Vec3(0.4715, 0.0865, 0.0275))
        for x in pickupX {
            m = m.booleanDifference(boxMesh(lo: Vec3(x - 0.036, -0.021, 0.011), hi: Vec3(x + 0.036, 0.021, 0.1)))
            plate = plate.booleanDifference(boxMesh(lo: Vec3(x - 0.038, -0.023, 0.0), hi: Vec3(x + 0.038, 0.023, 0.1)))
        }
        m = m.booleanUnion(plate)
        for x in pickupX { m = m.booleanUnion(boxMesh(lo: Vec3(x - 0.033, -0.018, 0.0105), hi: Vec3(x + 0.033, 0.018, 0.0335))) }
        // bridge + tailpiece bars across the body
        m = m.booleanUnion(boxMesh(lo: Vec3(bridgeX - 0.006, -0.05, 0.0235), hi: Vec3(bridgeX + 0.006, 0.05, 0.0345)))
        m = m.booleanUnion(boxMesh(lo: Vec3(bridgeX - 0.070, -0.05, 0.0235), hi: Vec3(bridgeX - 0.052, 0.05, 0.0325)))
        // four knobs: one cylinder, instanced 2 × 2 by the procedural graph's array nodes
        m = m.booleanUnion(knobs())
        // the fretboard: a loft whose width tapers to the nut, glued on the neck
        return m.booleanUnion(fretboard())
    }

    /// One knob arrayed 2 × 2, evaluated through a `ModelGraph` (importMesh → array → array).
    static func knobs() -> Mesh3 {
        let g = GraphBuilder()
        let one = g.importMesh(Mesh3.cylinder(from: Vec3(0.13, -0.10, 0.0235), to: Vec3(0.13, -0.10, 0.043), radius: 0.016))
        let grid = one.arrayed(count: 2, offset: Vec3(0.055, 0, 0)).arrayed(count: 2, offset: Vec3(0, 0.052, 0))
        return (try? g.build(output: grid).evaluateMesh()) ?? Mesh3()
    }

    /// Loft of a tapering board 0.495 … nut, whose bottom sinks half a millimetre into the neck.
    static func fretboard() -> Mesh3 {
        let n = 8
        let path = (0...n).map { Vec3(0.495 + (nutX - 0.495) * Double($0) / Double(n), 0, 0) }
        let sections: [[Vec2]] = (0...n).map { i in
            let w = fretWidth(path[i].x) / 2
            return [Vec2(neckTop - 0.0005, w), Vec2(fretTop, w), Vec2(fretTop, -w), Vec2(neckTop - 0.0005, -w)]
        }
        // the loft stands a section (u, v) up as world (z, −y) for a path along +x (probe-checked in the tests).
        return Mesh3.loft(sections: sections, along: path)
    }

    static func fretWidth(_ x: Double) -> Double { 0.052 - 0.008 * (x - 0.495) / (nutX - 0.495) }

    // MARK: 7. strings, frets, pegs

    /// Fret wire (17 boxes at the equal-temperament positions), six strings, six tuning posts with buttons.
    static func stringParts() -> Mesh3 {
        var parts = Mesh3()
        for n in 1...17 {
            let x = nutX - scale * (1 - pow(2, -Double(n) / 12)), w = fretWidth(x) / 2
            parts.append(boxMesh(lo: Vec3(x - 0.0008, -w, fretTop - 0.0003), hi: Vec3(x + 0.0008, w, fretTop + 0.0011)))
        }
        let radii = [0.0011, 0.0009, 0.0007, 0.0005, 0.0004, 0.0004]
        for j in 0..<6 {
            let f = Double(j) / 5
            let yb = -0.0255 + 0.051 * f, yn = -0.0175 + 0.035 * f
            let side = j < 3 ? -1.0 : 1.0
            let peg = Vec3(0.96 + 0.04 * Double(j < 3 ? 2 - j : j - 3), side * 0.026, 0.0)
            let path = [Vec3(bridgeX, yb, 0.0335), Vec3(nutX, yn, fretTop + 0.0015), Vec3(peg.x, peg.y, 0.016)]
            parts.append(Mesh3.tube(along: path, radius: radii[j], segments: 8))
        }
        // posts through the headstock + a button on each: built for one side, mirrored across y
        var side = Mesh3()
        for i in 0..<3 {
            let x = 0.96 + 0.04 * Double(i)
            side.append(Mesh3.cylinder(from: Vec3(x, -0.026, -0.0105), to: Vec3(x, -0.026, 0.0175), radius: 0.0035, chordError: 0.0002))
            side.append(Mesh3.cylinder(from: Vec3(x, -0.040, -0.0045), to: Vec3(x, -0.064, -0.0045), radius: 0.0075, chordError: 0.0002))
        }
        parts.append(side)
        parts.append(side.mirrored(.y))
        return parts
    }

    static func addStrings(_ body: Mesh3) -> Mesh3 {
        var m = body
        m.append(stringParts())
        return m
    }

    // MARK: 8. finishing stack

    /// Bend the neck a few degrees (SimpleDeform) and grain the wood (Displace noise, body only: the strings are
    /// finer than the grain), then unwrap: a box projection packed into 0…1.
    static func finish(body: Mesh3, parts: Mesh3) -> Mesh3 {
        var wood = ModifierStack()
        _ = wood.add(SimpleDeformModifier(.bend(angle: 0.05), origin: Vec3(0.5, 0, 0), axis: Vec3(1, 0, 0),
                                          bendDirection: Vec3(0, 0, 1), lower: 0, upper: 0.55), name: "Neck bow")
        _ = wood.add(DisplaceModifier.noise(strength: 0.0008, frequency: 90, octaves: 3, seed: 7), name: "Grain")
        var bend = ModifierStack()
        _ = bend.add(SimpleDeformModifier(.bend(angle: 0.05), origin: Vec3(0.5, 0, 0), axis: Vec3(1, 0, 0),
                                          bendDirection: Vec3(0, 0, 1), lower: 0, upper: 0.55), name: "Neck bow")
        var all = wood.evaluate(body)
        all.append(bend.evaluate(parts))
        let projected = all.projectedUVs(UVProjection(.box, scale: Vec2(4, 4)))
        return projected.packedUVs().mesh
    }
}
