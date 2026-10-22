import Dispatch

/// `(0..<count).map(body)` across cores, results in index order — deterministic regardless of
/// scheduling. Each slot is written by exactly one iteration.
enum SignackParallel {
    static func map<T: Sendable>(_ count: Int, _ body: @Sendable (Int) -> T) -> [T] {
        guard count > 1 else { return (0..<count).map(body) }
        let box = Slots<T>(count)
        DispatchQueue.concurrentPerform(iterations: count) { i in box.buffer[i] = body(i) }
        return box.buffer.map { $0! }
    }

    /// Fixed storage written slot-per-thread (never resized, so no exclusivity hazard).
    private final class Slots<T>: @unchecked Sendable {
        let buffer: UnsafeMutableBufferPointer<T?>
        init(_ n: Int) {
            buffer = .allocate(capacity: n)
            buffer.initialize(repeating: nil)
        }
        deinit {
            buffer.deinitialize()
            buffer.deallocate()
        }
    }
}
