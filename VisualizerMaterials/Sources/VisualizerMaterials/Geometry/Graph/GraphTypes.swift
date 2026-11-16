//  GraphTypes.swift — the vocabulary of the procedural model graph (ticket item J, "Modelling kit").
//
//  A `ModelGraph` (ModelGraph.swift) is a DAG of `GraphNode`s. Every node has ONE output, of one `SocketType`, and
//  named, typed inputs; what a node does is its `NodeKind`, its numbers are `Param`s. All of it is plain Codable
//  data (no closures), so a graph is a document: save it, diff it, load it, get the same mesh back.
//
//  Guarantees: types are checked when a graph is validated or evaluated (`GraphError`), never silently coerced.
//  A scalar / vec3 input whose name equals a parameter of the node OVERRIDES that parameter (drive a radius from a
//  scalar node). Not guaranteed: parameter names are not checked against the kind (an unknown one is ignored).

import Foundation
import simd

public enum SocketType: String, Codable, Sendable, CaseIterable { case mesh, sdf, points, scalar, vec3, transform }

/// A node parameter. Encoded as a one-key object (`{"number": 2}`, `{"vec3": [1,2,3]}`), so JSON stays readable.
public enum Param: Hashable, Sendable, Codable {
    case number(Double), bool(Bool), string(String), vec3(Vec3), data(Data), table([String: Param])

    private enum Key: String, CodingKey { case number, bool, string, vec3, data, table }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        if let v = try c.decodeIfPresent(Double.self, forKey: .number) { self = .number(v) }
        else if let v = try c.decodeIfPresent(Bool.self, forKey: .bool) { self = .bool(v) }
        else if let v = try c.decodeIfPresent(String.self, forKey: .string) { self = .string(v) }
        else if let v = try c.decodeIfPresent(Vec3.self, forKey: .vec3) { self = .vec3(v) }
        else if let v = try c.decodeIfPresent(Data.self, forKey: .data) { self = .data(v) }
        else if let v = try c.decodeIfPresent([String: Param].self, forKey: .table) { self = .table(v) }
        else { throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "empty Param")) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Key.self)
        switch self {
        case .number(let v): try c.encode(v, forKey: .number)
        case .bool(let v): try c.encode(v, forKey: .bool)
        case .string(let v): try c.encode(v, forKey: .string)
        case .vec3(let v): try c.encode(v, forKey: .vec3)
        case .data(let v): try c.encode(v, forKey: .data)
        case .table(let v): try c.encode(v, forKey: .table)
        }
    }
}

public extension Param {
    static func int(_ v: Int) -> Param { .number(Double(v)) }
}

/// Typed, defaulted reads over a parameter dictionary (after socket overrides were merged in).
public struct GraphArgs: Sendable {
    public var values: [String: Param]
    public init(_ v: [String: Param] = [:]) { values = v }
    public func double(_ k: String, _ d: Double) -> Double { if case .number(let v)? = values[k] { return v }; return d }
    public func int(_ k: String, _ d: Int) -> Int { Int(double(k, Double(d)).rounded()) }
    public func bool(_ k: String, _ d: Bool) -> Bool {
        if case .bool(let v)? = values[k] { return v }
        if case .number(let v)? = values[k] { return v != 0 }
        return d
    }
    public func string(_ k: String, _ d: String) -> String { if case .string(let v)? = values[k] { return v }; return d }
    public func vec3(_ k: String, _ d: Vec3) -> Vec3 {
        if case .vec3(let v)? = values[k] { return v }
        if case .number(let v)? = values[k] { return Vec3(repeating: v) }
        return d
    }
    public func data(_ k: String) -> Data? { if case .data(let v)? = values[k] { return v }; return nil }
    public func table(_ k: String) -> GraphArgs { if case .table(let v)? = values[k] { return GraphArgs(v) }; return GraphArgs() }
}

/// 64-bit FNV-1a: the structural key of a node (kind + parameters + the keys of what feeds it).
struct GraphHasher {
    private(set) var state: UInt64 = 0xcbf2_9ce4_8422_2325
    mutating func add(_ b: UInt8) { state = (state ^ UInt64(b)) &* 0x100_0000_01b3 }
    mutating func add(_ v: UInt64) { for i in 0 ..< 8 { add(UInt8(truncatingIfNeeded: v >> (8 * UInt64(i)))) } }
    mutating func add(_ d: Double) { add(d.bitPattern) }
    mutating func add(_ s: String) { add(UInt64(s.utf8.count)); for b in s.utf8 { add(b) } }
    mutating func add(_ p: Param) {
        switch p {
        case .number(let v): add(1 as UInt8); add(v)
        case .bool(let v): add(2 as UInt8); add(v ? 1 : 0 as UInt8)
        case .string(let v): add(3 as UInt8); add(v)
        case .vec3(let v): add(4 as UInt8); add(v.x); add(v.y); add(v.z)
        case .data(let v): add(5 as UInt8); add(UInt64(v.count)); for b in v { add(b) }
        case .table(let t): add(6 as UInt8); for k in t.keys.sorted() { add(k); add(t[k]!) }
        }
    }
}

public enum GraphError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case cycle([Int])
    case missingNode(Int)
    case missingInput(node: Int, kind: String, input: String)
    case typeMismatch(node: Int, input: String, expected: SocketType, got: SocketType)
    case unknownInput(node: Int, kind: String, input: String)
    case badParameter(node: Int, message: String)
    case noOutput
    public var description: String {
        switch self {
        case .cycle(let p): return "graph has a cycle: " + p.map { "#\($0)" }.joined(separator: " → ")
        case .missingNode(let n): return "node #\(n) does not exist"
        case .missingInput(let n, let k, let i): return "node #\(n) (\(k)) has no input connected to '\(i)'"
        case .typeMismatch(let n, let i, let e, let g): return "node #\(n) input '\(i)' wants \(e.rawValue) but is fed a \(g.rawValue)"
        case .unknownInput(let n, let k, let i): return "node #\(n) (\(k)) has no input named '\(i)'"
        case .badParameter(let n, let m): return "node #\(n): \(m)"
        case .noOutput: return "graph has no output node"
        }
    }
    public var errorDescription: String? { description }
}

/// One node: its kind, parameters and the source node feeding each named input. Equatable / Codable.
public struct GraphNode: Hashable, Codable, Sendable, Identifiable {
    public var id: Int
    public var kind: NodeKind
    public var name: String
    public var params: [String: Param]
    /// Input socket name → id of the node whose (single) output feeds it.
    public var inputs: [String: Int]
    public init(id: Int, kind: NodeKind, name: String = "", params: [String: Param] = [:], inputs: [String: Int] = [:]) {
        self.id = id; self.kind = kind; self.name = name; self.params = params; self.inputs = inputs
    }
}
