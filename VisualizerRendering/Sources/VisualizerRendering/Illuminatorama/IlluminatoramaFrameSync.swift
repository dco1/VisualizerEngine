import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

/// Thread-safe hand-off of "which LDR pool buffer finished rendering" from a
/// background `MTLCommandBuffer` completion handler to the `@MainActor` render
/// loop. Command buffers on one queue complete in commit order, so the last
/// `markCompleted` before a `takeCompleted` is the freshest finished frame.
/// Env-gated GPU-time meter. The frame-delivery metric in PerformanceMeasurement
/// is SceneKit's present cadence (vsync-capped, decoupled from this renderer's
/// GPU tick), so it can't see the per-config cost of RT GI rays / surface cache.
/// This reads `buf.gpuEndTime - buf.gpuStartTime` on each completion (background
/// thread) and rewrites the rolling average to `VIZ_ILLUMI_GPUMS_PATH`. Inert
/// when the env var is unset.
final class IlluminatoramaGPUMeter: @unchecked Sendable {
    let lock = NSLock()
    let path: String?
    var sumMs: Double = 0
    var count: Int = 0
    // Per-frame samples for the percentile distribution. On a shared / contended
    // machine the MEAN is dominated by occasional contention spikes (WindowServer,
    // screen-share, a sibling render) and reads ~the same at every SSAA scale, so
    // it can't show whether a render-cost change actually moved the STEADY frame.
    // The p50 rejects that tail and isolates the clean steady-state GPU frame —
    // the number that decides whether vsync can hold 60 (is it < 16.6 ms?). p95/p99
    // describe the drop-frame tail. Ring-bounded so a long run can't grow unbounded.
    var samples: [Double] = []
    static let maxSamples = 8192
    // Last-computed percentiles, refreshed every 30 frames and ALWAYS emitted, so
    // the sidecar file holds the full distribution no matter which frame the
    // headless snapshot reads it on (the recompute is throttled, the write is not).
    var p50: Double = 0, p95: Double = 0, p99: Double = 0
    var minMs: Double = 0, maxMs: Double = 0
    init() { path = ProcessInfo.processInfo.environment["VIZ_ILLUMI_GPUMS_PATH"] }
    func record(_ buf: MTLCommandBuffer) {
        guard let path else { return }
        let ms = (buf.gpuEndTime - buf.gpuStartTime) * 1000.0
        guard ms > 0 else { return }
        lock.lock()
        sumMs += ms; count += 1
        let avg = sumMs / Double(count); let n = count
        if samples.count >= Self.maxSamples { samples.removeFirst(Self.maxSamples / 4) }
        samples.append(ms)
        // Recompute percentiles every 30 frames (sort is O(n log n); throttle so
        // the diagnostic doesn't dominate the very thing it measures).
        if n % 30 == 0 || p50 == 0 {
            let s = samples.sorted()
            func pct(_ q: Double) -> Double { s[min(s.count - 1, max(0, Int(Double(s.count) * q)))] }
            p50 = pct(0.50); p95 = pct(0.95); p99 = pct(0.99)
            minMs = s.first ?? 0; maxMs = s.last ?? 0
        }
        let line = String(format: "gpuMsAvg=%.3f frames=%d p50=%.3f p95=%.3f p99=%.3f min=%.3f max=%.3f\n",
                          avg, n, p50, p95, p99, minMs, maxMs)
        lock.unlock()
        if let d = line.data(using: .utf8) { try? d.write(to: URL(fileURLWithPath: path)) }
    }
}

/// A one-shot "the GPU finished this frame" latch that an `async` caller can await.
///
/// Exists because the two ends of the handshake live on different threads and neither can be
/// moved: `MTLCommandBuffer.addCompletedHandler` fires on a Metal-owned background thread, and
/// the awaiter is main-actor code inside `awaitCommittedFrame()`. A `DispatchSemaphore` would
/// bridge them only by blocking a cooperative-pool thread, which is the stall this whole path
/// exists to remove.
///
/// Latched rather than gated: `signal()` may land BEFORE anyone awaits (a fast frame finishing
/// while the caller is still unwinding the commit), so `wait()` on an already-signalled latch
/// must return immediately instead of parking forever. Both calls are idempotent — a second
/// `signal()` is dropped and the continuation is resumed exactly once.
final class FrameCompletionSignal: @unchecked Sendable {
    let lock = NSLock()
    var completed = false
    var continuation: CheckedContinuation<Void, Never>?

    /// Called from the GPU completion handler. Resumes the awaiter OUTSIDE the lock —
    /// a continuation resume can run arbitrary code, and doing that under the lock would
    /// re-enter it if that code touched this latch again.
    func signal() {
        lock.lock()
        if completed { lock.unlock(); return }
        completed = true
        let waiter = continuation
        continuation = nil
        lock.unlock()
        waiter?.resume()
    }

    func wait() async {
        await withCheckedContinuation { (k: CheckedContinuation<Void, Never>) in
            lock.lock()
            if completed {
                lock.unlock()
                k.resume()          // already finished — never park
            } else {
                continuation = k
                lock.unlock()
            }
        }
    }
}

final class IlluminatoramaPresentSync: @unchecked Sendable {
    let lock = NSLock()
    var completedIdx: Int = -1
    /// Pool slots whose render is queued/running on the GPU but not yet complete.
    /// Inserted on the main thread (after commit) and removed on the GPU
    /// completion thread, so `acquireWriteTarget` never picks a buffer a second
    /// pipelined frame is still writing (`maxFramesInFlight = 2`).
    var inFlight: Set<Int> = []

    // ── Adaptive in-flight depth (latency guard) ───────────────────────────────
    // Pipelining a 2nd frame (`maxFramesInFlight = 2`) recovers the sub-tick GPU
    // idle on light/medium scenes (big fps win). But when a frame is much heavier
    // than the 60 Hz tick (e.g. RT glass + caustics at ~200–600 ms), the 2nd frame
    // adds ~one whole frame of INPUT LATENCY with no throughput gain, so the scene
    // feels sluggish to drag. So we shrink the effective depth back to 1 when the
    // recent GPU frame time is heavy — done WITHOUT touching the render() gate, by
    // simply HOLDING a semaphore permit in the completion handler instead of
    // signalling it (and signalling it back when frames get light again).
    /// EMA of recent GPU frame time (ms). Drives the depth decision.
    var gpuMsEMA: Double = 0
    /// Permits currently held back (0 ⇒ depth 2, 1 ⇒ depth 1). Never exceeds
    /// `maxFramesInFlight - 1`, so at least one permit always circulates (no deadlock).
    var heldPermits: Int = 0
    /// Set by `retire(semaphore:)` when the owning renderer tears down — no
    /// further permits may be held after that (see the deinit-balance contract).
    var retired = false
    /// Current effective depth target, with hysteresis so it doesn't flap.
    var depth: Int = IlluminatoramaRenderer.maxFramesInFlight
    /// Host opted out of the latency guard (`pipelineLatencyTolerant`): always full depth.
    var latencyTolerant = false
    func setLatencyTolerant(_ on: Bool) { lock.lock(); latencyTolerant = on; lock.unlock() }
    /// Hysteresis band (~2 ticks): pipeline only while the GPU frame is light.
    static let pipelineEnterMs = 30.0   // drop to 1→2 when EMA below this
    static let pipelineExitMs  = 36.0   // rise to 2→1 when EMA above this

    /// The most recent completed frame's GPU time (ms), for hosts' own diagnostics. Written
    /// under `lock` by `frameCompletedAdaptive` (completion-handler thread).
    var lastGpuMs: Double { lock.lock(); defer { lock.unlock() }; return _lastGpuMs }
    var _lastGpuMs: Double = 0

    /// Called from the GPU completion handler in place of `semaphore.signal()`.
    /// Updates the GPU-time EMA, recomputes the target depth, and either releases
    /// this frame's permit normally or HOLDS it to converge the effective in-flight
    /// capacity to that depth. Deadlock-free: `heldPermits` is capped at
    /// `maxFramesInFlight - 1`, so a permit always remains in circulation.
    func frameCompletedAdaptive(gpuMs: Double, semaphore: DispatchSemaphore) {
        lock.lock()
        _lastGpuMs = gpuMs
        if gpuMs > 0 {
            gpuMsEMA = gpuMsEMA == 0 ? gpuMs : gpuMsEMA * 0.9 + gpuMs * 0.1
        }
        if latencyTolerant {
            depth = IlluminatoramaRenderer.maxFramesInFlight
        } else if depth >= IlluminatoramaRenderer.maxFramesInFlight {
            if gpuMsEMA > Self.pipelineExitMs { depth = 1 }
        } else {
            if gpuMsEMA < Self.pipelineEnterMs { depth = IlluminatoramaRenderer.maxFramesInFlight }
        }
        let targetHeld = max(0, IlluminatoramaRenderer.maxFramesInFlight - depth)
        if heldPermits < targetHeld, !retired {
            heldPermits += 1            // hold this frame's permit (shrink capacity)
            lock.unlock()
        } else if heldPermits > targetHeld {
            heldPermits -= 1
            lock.unlock()
            semaphore.signal()          // this frame's permit …
            semaphore.signal()          // … plus one previously-held, to grow capacity
        } else {
            lock.unlock()
            semaphore.signal()          // normal release
        }
    }

    // ── Per-frame buffer-race probe (env-gated; see maxFramesInFlight comment) ──
    // `slotInFlight[s]` = a frame that wrote ring slot `s` is still reading it on
    // the GPU. `raceProbeBeginFrame` is called (main thread) when a new frame
    // claims its slot: if that slot is STILL in flight from an earlier frame, the
    // CPU is about to overwrite a buffer the GPU is reading — the exact race —
    // so it's counted. `raceProbeCompleteFrame` clears the flag on GPU completion.
    // Sized to maxFramesInFlight+1 (the ring is maxFramesInFlight; +1 is slack).
    var slotInFlight = [Bool](repeating: false, count: IlluminatoramaRenderer.maxFramesInFlight + 1)
    var raceEvents = 0
    var probeFrames = 0
    let racePath = ProcessInfo.processInfo.environment["VIZ_ILLUMI_RACE_PATH"]

    func raceProbeBeginFrame(slot: Int) {
        lock.lock()
        probeFrames += 1
        if slot >= 0, slot < slotInFlight.count {
            if slotInFlight[slot] { raceEvents += 1 }   // overwriting a slot still read by the GPU
            slotInFlight[slot] = true
        }
        let r = raceEvents, f = probeFrames
        lock.unlock()
        if let p = racePath {
            let line = "raceEvents=\(r) frames=\(f)\n"
            if let d = line.data(using: .utf8) { try? d.write(to: URL(fileURLWithPath: p)) }
        }
    }

    func raceProbeCompleteFrame(slot: Int) {
        lock.lock()
        if slot >= 0, slot < slotInFlight.count { slotInFlight[slot] = false }
        lock.unlock()
    }

    /// The owning renderer is tearing down: release every HELD permit and refuse to
    /// hold any more, so `inFlightSemaphore` can never be deallocated under-valued.
    /// A `DispatchSemaphore` whose value at dealloc is below its creation value hits
    /// the libdispatch trap "Semaphore object deallocated while in use" — which is
    /// exactly what happened when a heavy-frame scene (RT glass) ended with the
    /// adaptive guard holding a permit: the renderer + semaphore died one frame
    /// later and the SIGTRAP detonated in whatever unrelated code ran next (it
    /// presented as heap-corruption-looking crashes in the NEXT test's mesh build).
    /// The lock serializes against a completion handler mid-decision: either it
    /// holds first (drained here) or it sees `retired` and signals normally.
    /// Over-signalling past the creation value is safe — only UNDER-value traps.
    func retire(semaphore: DispatchSemaphore) {
        lock.lock()
        retired = true
        let toRelease = heldPermits
        heldPermits = 0
        lock.unlock()
        for _ in 0..<toRelease { semaphore.signal() }
    }

    /// Record (from the main thread, at commit) that pool buffer `idx` is now in flight.
    func markInFlight(_ idx: Int) {
        lock.lock()
        inFlight.insert(idx)
        lock.unlock()
    }

    /// True if `idx`'s render is still in flight (used by `acquireWriteTarget`).
    func isInFlight(_ idx: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return inFlight.contains(idx)
    }

    /// Bumped by `reset()`. A frame captures it when it claims a pool slot; its completion is
    /// ignored if the pool was rebuilt in between (see `markCompleted`).
    var generation = 0
    var currentGeneration: Int { lock.lock(); defer { lock.unlock() }; return generation }

    /// Record (from the GPU completion thread) that pool buffer `idx` is done.
    ///
    /// **A completion from a pool that no longer exists is dropped.** A resize rebuilds the LDR
    /// pool at the new size and resets this object, but a frame committed just before it is still
    /// on the GPU, writing a slot of the OLD pool. Accepting its index would promote the NEW
    /// pool's texture at that index — one nothing has ever written — and present it. A fresh
    /// private texture reads back as solid 255,0,255 on macOS, which is the fuchsia flash a
    /// window resize showed. It must not clear `inFlight[idx]` either: after the reset that entry
    /// belongs to a new-pool frame that may still be writing it.
    func markCompleted(_ idx: Int, generation frameGeneration: Int) {
        lock.lock()
        if frameGeneration == generation {
            completedIdx = idx
            inFlight.remove(idx)
        }
        lock.unlock()
    }

    /// Return the most recently completed pool index and clear it, or nil if
    /// nothing new finished since the last call.
    func takeCompleted() -> Int? {
        lock.lock()
        defer { lock.unlock() }
        let v = completedIdx
        completedIdx = -1
        return v >= 0 ? v : nil
    }

    func reset() {
        lock.lock()
        generation &+= 1          // in-flight frames of the old pool must not promote (markCompleted)
        completedIdx = -1
        inFlight.removeAll()
        // Reset the adaptive-depth EMA/target; leave `heldPermits` alone so no
        // held permit is lost — the next completion self-heals it (targetHeld 0).
        gpuMsEMA = 0
        depth = IlluminatoramaRenderer.maxFramesInFlight
        lock.unlock()
    }
}

public enum IlluminatoramaError: Error, LocalizedError {
    case libraryMissing
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .libraryMissing:
            return "VisualizerRendering Metal library could not be loaded — was the package compiled with Shaders/?"
        case .pipelineCreationFailed(let name):
            return "Illuminatorama pipeline failed to create: \(name)"
        case .bufferAllocationFailed(let what):
            return "Illuminatorama failed to allocate Metal resource: \(what)"
        }
    }
}
