//  ModelGraph.swift — a procedural model as data: a deterministic, Codable DAG of typed nodes (ticket item J).
//
//  Build it with `GraphBuilder` (GraphBuilder.swift) or by hand (`add`, `connect`, `setParam`), check it with
//  `validate()`, run it with `evaluate` (ModelGraph+Evaluate.swift, cached per node by structural key).
//
//  Guarantees: JSON round trip (`jsonData()` / `init(json:)`) is exact — same graph, same mesh; nodes encode in id
//  order with sorted keys, so the file is stable and diffable. Evaluation is deterministic: every random choice
//  (scatter, jitter) is seeded by a parameter. Cycles and type errors are reported as `GraphError`, never traps.
//  Not guaranteed: edits that leave dangling input ids are caught only at validate / evaluate time.

import Foundation

public struct ModelGraph: Codable, Equatable, Sendable {
    public private(set) var nodes: [GraphNode] = []
    /// The node `evaluate()` returns by default.
    public var output: Int?
    private var nextID = 1

    public init() {}

    // MARK: editing

    /// Add a node and return its id.
    @discardableResult
    public mutating func add(_ kind: NodeKind, name: String = "", params: [String: Param] = [:], inputs: [String: Int] = [:]) -> Int {
        let id = nextID; nextID += 1
        nodes.append(GraphNode(id: id, kind: kind, name: name, params: params, inputs: inputs))
        return id
    }

    public func node(_ id: Int) -> GraphNode? { nodes.first { $0.id == id } }

    private mutating func edit(_ id: Int, _ f: (inout GraphNode) -> Void) {
        if let i = nodes.firstIndex(where: { $0.id == id }) { f(&nodes[i]) }
    }

    public mutating func setParam(_ id: Int, _ key: String, _ value: Param) { edit(id) { $0.params[key] = value } }
    public mutating func setParam(_ id: Int, _ key: String, _ value: Double) { setParam(id, key, .number(value)) }
    public mutating func connect(_ source: Int, to id: Int, input: String) { edit(id) { $0.inputs[input] = source } }
    public mutating func disconnect(_ id: Int, input: String) { edit(id) { $0.inputs[input] = nil } }

    /// Remove a node and every connection that read it.
    public mutating func remove(_ id: Int) {
        nodes.removeAll { $0.id == id }
        for i in nodes.indices { nodes[i].inputs = nodes[i].inputs.filter { $0.value != id } }
        if output == id { output = nil }
    }

    // MARK: validation

    /// Every node's inputs exist and have the right type, required inputs are connected, there is no cycle.
    /// Throws the first problem found, in node-id order.
    public func validate() throws {
        let index = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        for n in nodes.sorted(by: { $0.id < $1.id }) {
            for (name, src) in n.inputs.sorted(by: { $0.key < $1.key }) {
                guard let s = index[src] else { throw GraphError.missingNode(src) }
                guard let want = n.kind.inputType(name) else { throw GraphError.unknownInput(node: n.id, kind: n.kind.rawValue, input: name) }
                let got = s.kind.spec.output
                guard want == got else { throw GraphError.typeMismatch(node: n.id, input: name, expected: want, got: got) }
            }
            for i in n.kind.spec.inputs where i.required && n.inputs[i.name] == nil {
                throw GraphError.missingInput(node: n.id, kind: n.kind.rawValue, input: i.name)
            }
            if n.kind.spec.variadic != nil, n.inputs.isEmpty { throw GraphError.missingInput(node: n.id, kind: n.kind.rawValue, input: "in0") }
        }
        if let o = output, index[o] == nil { throw GraphError.missingNode(o) }
        _ = try topologicalOrder()
    }

    /// Node ids, every node after the nodes it reads. Throws `.cycle(path)` naming the loop (first id repeated at the end).
    public func topologicalOrder() throws -> [Int] {
        let index = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        var state: [Int: Int] = [:]           // 1 = on the stack, 2 = done
        var order: [Int] = []
        var stack: [Int] = []
        func visit(_ id: Int) throws {
            if state[id] == 2 { return }
            if state[id] == 1 {
                let loop = Array(stack[(stack.firstIndex(of: id) ?? 0)...]) + [id]
                throw GraphError.cycle(loop)
            }
            guard let n = index[id] else { throw GraphError.missingNode(id) }
            state[id] = 1; stack.append(id)
            for src in n.inputs.sorted(by: { $0.key < $1.key }).map(\.value) { try visit(src) }
            stack.removeLast(); state[id] = 2; order.append(id)
        }
        for n in nodes.sorted(by: { $0.id < $1.id }) { try visit(n.id) }
        return order
    }

    // MARK: JSON

    public func jsonData(pretty: Bool = false) throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted] : [.sortedKeys]
        return try e.encode(self)
    }
    public init(json: Data) throws { self = try JSONDecoder().decode(ModelGraph.self, from: json) }
}
