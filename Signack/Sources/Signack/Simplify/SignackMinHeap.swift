/// A binary min-heap of (cost, payload) — the collapse queue. Ties break on payload so the
/// simplifier is deterministic.
struct SignackMinHeap<Payload: Comparable> {
    private var items: [(cost: Double, payload: Payload)] = []

    var isEmpty: Bool { items.isEmpty }

    @inline(__always) private func less(_ i: Int, _ j: Int) -> Bool {
        items[i].cost != items[j].cost ? items[i].cost < items[j].cost : items[i].payload < items[j].payload
    }

    mutating func push(_ cost: Double, _ payload: Payload) {
        items.append((cost, payload))
        var i = items.count - 1
        while i > 0 {
            let p = (i - 1) / 2
            guard less(i, p) else { break }
            items.swapAt(i, p)
            i = p
        }
    }

    mutating func pop() -> (cost: Double, payload: Payload)? {
        guard !items.isEmpty else { return nil }
        let top = items[0]
        let last = items.removeLast()
        if !items.isEmpty {
            items[0] = last
            var i = 0
            while true {
                let l = 2 * i + 1, r = l + 1
                var m = i
                if l < items.count, less(l, m) { m = l }
                if r < items.count, less(r, m) { m = r }
                if m == i { break }
                items.swapAt(i, m)
                i = m
            }
        }
        return top
    }
}
