//  GraphModifierRegistry.swift — the `modifier` node: any `Modifier` from the stack, by name and parameters.
//
//  Built-ins: mirror, array, radialArray, bevel, subdivide, solidify, displaceNoise, simpleDeform (bend / twist /
//  taper / stretch). `register` adds your own (a closure from `GraphArgs` to a Modifier). Modifiers that need a
//  mesh operand or a closure (boolean, shrinkwrap, custom displace) are not nameable here: use the `boolean` node.
//
//  Guarantees: unknown names and bad modes throw `GraphError.badParameter`, not trap. Not guaranteed: a modifier
//  registered by a client is part of the cache key only through its name and args, not its code.

import Foundation
import simd

public enum GraphModifierRegistry {
    public typealias Builder = @Sendable (GraphArgs) throws -> any Modifier
    private static let lock = NSLock()
    nonisolated(unsafe) private static var custom: [String: Builder] = [:]

    public static func register(_ name: String, _ builder: @escaping Builder) { lock.withLock { custom[name] = builder } }

    struct BadMode: Error { var message: String }

    static let builtin: [String: Builder] = [
        "mirror": { a in
            let axis: Mesh3.Axis = a.string("axis", "x") == "y" ? .y : a.string("axis", "x") == "z" ? .z : .x
            return MirrorModifier(axis, at: a.double("plane", 0), keepOriginal: a.bool("keepOriginal", true))
        },
        "array": { a in ArrayModifier(count: a.int("count", 2), offset: a.vec3("offset", Vec3(1, 0, 0))) },
        "radialArray": { a in ArrayModifier(radialCount: a.int("count", 6), around: a.vec3("centre", .zero), startAngle: a.double("startAngle", 0)) },
        "bevel": { a in BevelModifier(radius: a.double("radius", 0.01), creaseDegrees: a.double("crease", 30), segments: a.int("segments", 3)) },
        "subdivide": { a in SubdivideModifier(levels: a.int("levels", 1), creaseDegrees: a.double("crease", 35)) },
        "solidify": { a in
            SolidifyModifier(thickness: a.double("thickness", 0.01), offset: a.double("offset", -1), rim: a.bool("rim", true), evenThickness: a.bool("even", true))
        },
        "displaceNoise": { a in
            DisplaceModifier.noise(strength: a.double("strength", 0.01), frequency: a.double("frequency", 4), octaves: a.int("octaves", 4),
                                   seed: UInt64(Swift.max(0, a.int("seed", 0))), midlevel: a.double("midlevel", 0.5))
        },
        "simpleDeform": { a in
            let v = a.double("amount", 0.5)
            let mode: SimpleDeformModifier.Mode
            switch a.string("mode", "bend") {
            case "bend": mode = .bend(angle: v)
            case "twist": mode = .twist(angle: v)
            case "taper": mode = .taper(factor: v)
            case "stretch": mode = .stretch(factor: v)
            case let m: throw BadMode(message: "simpleDeform mode '\(m)' (bend, twist, taper, stretch)")
            }
            return SimpleDeformModifier(mode, origin: a.vec3("origin", .zero), axis: a.vec3("axis", Vec3(0, 1, 0)),
                                        lower: a.double("lower", 0), upper: a.double("upper", 1))
        },
    ]

    public static var names: [String] { (Array(builtin.keys) + lock.withLock { Array(custom.keys) }).sorted() }

    static func make(_ name: String, _ args: GraphArgs, node: Int) throws -> any Modifier {
        guard let b = builtin[name] ?? lock.withLock({ custom[name] }) else {
            throw GraphError.badParameter(node: node, message: "no modifier named '\(name)' (have: \(names.joined(separator: ", ")))")
        }
        do { return try b(args) } catch let e as BadMode { throw GraphError.badParameter(node: node, message: e.message) }
    }
}
