import Dispatch

/// Order-preserving parallel map across all CPU cores, for **pure, deterministic**
/// transforms — the procedural material generators qualify: each returns a freshly
/// allocated result from a seeded generator with no shared mutable state, so running
/// them concurrently is data-race-free.
///
/// This is the lever that turns the ~38 s serial first-touch of the whole 512² material
/// library into a ~cores×-faster prewarm (the "app hangs the first time you place a
/// Placeable" fix — see `HouseSceneBuilder.prewarmMaterials`). The 512² bake-size bump
/// (materials commit `9f90e41`) quadrupled a cost that was already synthesizing the entire
/// library on the render thread; parallelizing + prewarming moves it off that thread.
///
/// Each iteration initializes exactly ONE distinct slot, so the concurrent writes never
/// overlap — no locking needed. `concurrentPerform` runs the work on the global pool and
/// blocks the caller until every slot is filled, so the returned array is fully populated
/// and in input order. Falls back to a serial `map` for 0/1 items (no dispatch overhead).
///
/// The transform MUST be pure (no shared mutable state, no ordering dependence between
/// items). `DaydreamCoreTests.ParallelMapTests` gates that it equals the serial `map`.
public func parallelMap<T, R>(_ items: [T], _ transform: (T) -> R) -> [R] {
    let n = items.count
    if n <= 1 { return items.map(transform) }
    // A raw slot buffer we initialize once per index. `nonisolated(unsafe)` waives the
    // Sendable check on the shared pointer: the safety argument is the disjoint-index
    // writes above, which the type system can't express but we uphold by construction.
    let buffer = UnsafeMutableBufferPointer<R>.allocate(capacity: n)
    defer { buffer.deinitialize(); buffer.deallocate() }
    nonisolated(unsafe) let slots = buffer
    nonisolated(unsafe) let work = transform
    nonisolated(unsafe) let source = items
    DispatchQueue.concurrentPerform(iterations: n) { i in
        slots.baseAddress!.advanced(by: i).initialize(to: work(source[i]))
    }
    return Array(buffer)
}
