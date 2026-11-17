import Foundation
import Metal

// POOLED MESH BUFFERS (Daydream DH-1676)
// =====================================
//
// Every `IlluminatoramaMesh` built from CPU data used to pay two `device.makeBuffer(bytes:)` calls —
// a kernel-backed allocation each. A wall drag on 4000 Sunset re-registers ~30 meshes a frame, so
// that was ~0.9 ms of every drag frame (DH-1676 profile: 173 samples in `IlluminatoramaMesh.init`).
// Small and medium buffers are now sub-allocated from a few shared `MTLHeap`s (measured on this M1:
// ~2.5 µs a buffer against ~6.5 µs from the device, 24 KB each).
//
// What does NOT change: a pooled buffer is an ordinary `MTLBuffer` at offset 0 — every bind, BLAS
// descriptor and GPU address the engine takes from `vertexBuffer` / `indexBuffer` works as before.
// The heap is `.tracked`, so hazard tracking is the same as a device buffer's, and its memory goes
// back to the heap when the buffer is released — which Metal defers until no command buffer still
// holds it, exactly as for a device buffer. A buffer the heaps cannot fit (too big, or the heaps
// fragmented and at their cap) falls back to `device.makeBuffer`.
// Lever: `IlluminatoramaMeshBufferPool.enabled` (`VIZ_ILLUMI_NO_MESH_POOL=1` starts it off).
public final class IlluminatoramaMeshBufferPool: @unchecked Sendable {

    /// A/B lever. Off = every mesh buffer comes from `device.makeBuffer`, as before DH-1676.
    /// Always off on the iOS Simulator: its Metal requires `.private` storage for heaps and traps
    /// (`MTLStorageModePrivate is required for heaps`) on the shared heaps this pool makes.
    #if targetEnvironment(simulator)
    nonisolated(unsafe) public static var enabled = false
    #else
    nonisolated(unsafe) public static var enabled = ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_MESH_POOL"] != "1"
    #endif
    /// Buffers larger than this go to the device (a big mesh would fragment a heap for little gain).
    static let maxPooledLength = 1 << 20
    static let heapSize = 32 << 20
    static let maxHeaps = 8

    private static let registryLock = NSLock()
    nonisolated(unsafe) private static var pools: [UInt64: IlluminatoramaMeshBufferPool] = [:]

    /// The pool for `device` (one per device, shared by every renderer on it).
    static func shared(for device: MTLDevice) -> IlluminatoramaMeshBufferPool {
        registryLock.lock(); defer { registryLock.unlock() }
        if let p = pools[device.registryID] { return p }
        let p = IlluminatoramaMeshBufferPool(device: device)
        pools[device.registryID] = p
        return p
    }

    private let device: MTLDevice
    private let lock = NSLock()
    private var heaps: [MTLHeap] = []
    /// Diagnostics: buffers served from a heap / sent to the device.
    public private(set) var pooledCount = 0
    public private(set) var fallbackCount = 0

    private init(device: MTLDevice) { self.device = device }

    /// A shared buffer holding `length` bytes copied from `bytes`: from a heap when it fits, else
    /// from the device. nil only when the device itself refuses.
    func makeBuffer(bytes: UnsafeRawPointer, length: Int) -> MTLBuffer? {
        if Self.enabled, length > 0, length <= Self.maxPooledLength, let b = heapBuffer(length) {
            memcpy(b.contents(), bytes, length)
            return b
        }
        lock.lock(); fallbackCount += 1; lock.unlock()
        return device.makeBuffer(bytes: bytes, length: length, options: .storageModeShared)
    }

    private func heapBuffer(_ length: Int) -> MTLBuffer? {
        lock.lock(); defer { lock.unlock() }
        for h in heaps.reversed() {
            if let b = h.makeBuffer(length: length, options: .storageModeShared) { pooledCount += 1; return b }
        }
        // A heap every buffer has left (a closed document's meshes) gives its memory back first.
        if heaps.count >= Self.maxHeaps { heaps.removeAll { $0.usedSize == 0 } }
        guard heaps.count < Self.maxHeaps else { return nil }
        let d = MTLHeapDescriptor()
        d.size = Self.heapSize
        d.storageMode = .shared
        d.hazardTrackingMode = .tracked
        d.type = .automatic
        guard let h = device.makeHeap(descriptor: d) else { return nil }
        h.label = "Illuminatorama.meshPool.\(heaps.count)"
        heaps.append(h)
        guard let b = h.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        pooledCount += 1
        return b
    }
}
