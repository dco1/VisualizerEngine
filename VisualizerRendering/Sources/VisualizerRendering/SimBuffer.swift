import Foundation
import Metal

// ── SIM BUFFER ───────────────────────────────────────────────────────────────
//
// SimBuffer<T> is the shared-memory foundation for every GPU simulation in this
// project. Allocate once, write particles/constraints from Swift, hand the
// underlying MTLBuffer to any compute encoder. Nothing clever here — the value
// is one canonical type that every solver and every rendering bridge knows how
// to consume.
//
// STORAGE MODE
//   .storageModeShared: CPU and GPU share the same physical RAM (Apple Silicon).
//   No blit needed. Safe to read from Swift after a command buffer completes.
//
// WHERE TO GO NEXT
// ─────────────────
// • Triple buffering: if you ever write parameters from the main thread *while*
//   a GPU command buffer is reading the same buffer, add a semaphore + three
//   underlying MTLBuffers and rotate through them. Required for particle
//   emitters that stream new particles every frame. See Apple's "Synchronizing
//   CPU and GPU Work" sample.
//
// • Private storage for intermediate passes: constraint correction buffers,
//   neighbour lists, and spatial hashes never need to be read by Swift. Switch
//   those to .storageModePrivate to reduce cache-coherency overhead. Provide a
//   small .storageModeShared staging buffer for the initial upload via blit.
//
// • Grow / compact: capacity is fixed at init. When particles are born/died
//   at runtime (emitter, torn cloth), add a realloc() that doubles capacity
//   and copies existing contents with a MTLBlitCommandEncoder.

@MainActor
public final class SimBuffer<T> {

    public let buffer: MTLBuffer
    public let capacity: Int
    public private(set) var count: Int = 0

    public init?(device: MTLDevice, capacity: Int, label: String? = nil) {
        let byteLen = MemoryLayout<T>.stride * max(capacity, 1)
        guard let buf = device.makeBuffer(length: byteLen, options: .storageModeShared) else {
            return nil
        }
        if let label { buf.label = label }
        self.buffer   = buf
        self.capacity = capacity
    }

    // Typed pointer into the shared buffer. Only safe to access after the most
    // recent command buffer that writes this buffer has completed. Do not hold
    // this pointer across a command buffer commit.
    public var contents: UnsafeMutablePointer<T> {
        buffer.contents().bindMemory(to: T.self, capacity: capacity)
    }

    // Upload a Swift array into the buffer. Call this during scene setup or
    // whenever chain topology changes. Not safe to call while a GPU pass that
    // touches this buffer is in flight.
    public func write(_ elements: [T]) {
        precondition(elements.count <= capacity,
                     "SimBuffer<\(T.self)> overflow: \(elements.count) > \(capacity)")
        elements.withUnsafeBufferPointer {
            buffer.contents().copyMemory(
                from: $0.baseAddress!,
                byteCount: MemoryLayout<T>.stride * elements.count
            )
        }
        count = elements.count
    }
}
