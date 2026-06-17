import Foundation
import Metal
import OSLog
import VisualizerCore

// ── ILLUMINATORAMA PER-PASS GPU TIMER (#60 task 4 / #6) ──────────────────────
//
// True per-compute-pass GPU time via `MTLCounterSampleBuffer` (timestamp counter
// set, stage-boundary sampling). This is the granularity the occupancy tuning
// (`maxTotalThreadsPerThreadgroup` per kernel) needs — "which kernel costs what",
// measured not guessed.
//
// Why this ISN'T blocked on #6. #6 is hard because SceneKit hides its command
// buffer, so you can't attach a counter sample buffer to ITS passes. But
// Illuminatorama is a custom Metal renderer that owns its own `MTLCommandQueue`
// and command buffers — so for OUR passes we just create the compute encoder with
// a `MTLComputePassDescriptor` carrying a sample-buffer attachment. No SceneKit
// obstruction. (Per-FRAME total GPU time already exists via the command buffer's
// `gpuStartTime`/`gpuEndTime`; this adds the per-pass breakdown.)
//
// SELF-VALIDATION (no reference data to trust by eye). GPU timestamps come back
// in opaque ticks; converting to ms needs a CPU/GPU correlation, and a wrong
// scale would silently report bogus numbers. So `resolve()` cross-checks the SUM
// of the per-pass ms against the command buffer's own `gpuEndTime − gpuStartTime`
// (which is in SECONDS and needs no conversion). If they disagree by more than a
// tolerance the report is flagged UNVERIFIED — the same "certify against ground
// truth, don't self-certify" discipline used for the LTC fit.
//
// Off by default; gated on `VIZ_ILLUMI_PASS_PROFILE` (also the sidecar path, since
// os_log is eaten under the sandbox). Hardware without stage-boundary counter
// sampling silently no-ops (the encoders fall back to untimed creation).
final class IlluminatoramaPassTimer: @unchecked Sendable {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "illuminatoramaPassTimer")

    /// Max stage-boundary samples per frame (2 per timed pass → 16 passes).
    private static let capacity = 32

    let enabled: Bool
    private let sidecarPath: String?
    private let sampleBuffer: MTLCounterSampleBuffer?
    /// GPU-tick → nanosecond scale from a CPU/GPU correlation anchor pair.
    private let nsPerTick: Double

    // Per-frame state (render() is @MainActor, single-threaded encode, so plain
    // vars are safe to mutate during a frame; the completion handler reads a
    // snapshot passed into resolve()).
    private(set) var labels: [String] = []
    private var nextIndex = 0

    // Frame counter (mutated in resolve(), on the completion thread; guarded by a
    // lock since multiple frames' handlers can overlap).
    private let lock = NSLock()
    private var frameCount = 0

    init(device: MTLDevice) {
        let path = ProcessInfo.processInfo.environment["VIZ_ILLUMI_PASS_PROFILE"]
        sidecarPath = path
        // Capability + counter set.
        let supported = device.supportsCounterSampling(.atStageBoundary)
        let tsSet = device.counterSets?.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }
        if let path, !path.isEmpty, supported, let tsSet {
            let desc = MTLCounterSampleBufferDescriptor()
            desc.counterSet = tsSet
            desc.storageMode = .shared
            desc.sampleCount = Self.capacity
            sampleBuffer = try? device.makeCounterSampleBuffer(descriptor: desc)
        } else {
            sampleBuffer = nil
        }
        enabled = (sampleBuffer != nil)

        // CPU/GPU correlation: two anchors give the linear tick→ns scale. The CPU
        // MTLTimestamp is nanoseconds on Apple Silicon; the cross-check in
        // resolve() validates this empirically regardless.
        if enabled {
            var cpu0: MTLTimestamp = 0, gpu0: MTLTimestamp = 0
            var cpu1: MTLTimestamp = 0, gpu1: MTLTimestamp = 0
            device.__sampleTimestamps(&cpu0, gpuTimestamp: &gpu0)
            // a little CPU work so the anchors differ
            var spin = 0.0; for i in 0..<200_000 { spin += Double(i) * 1.0000001 }
            device.__sampleTimestamps(&cpu1, gpuTimestamp: &gpu1)
            let dCpu = Double(cpu1 &- cpu0), dGpu = Double(gpu1 &- gpu0)
            nsPerTick = (dGpu > 0 && dCpu > 0 && abs(spin) >= 0) ? dCpu / dGpu : 1.0
        } else {
            nsPerTick = 1.0
        }
        if let path, !path.isEmpty, !enabled {
            try? "pass-profile UNAVAILABLE (counter sampling unsupported or buffer alloc failed)\n"
                .write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    /// Reset per-frame sample assignment. Call at the start of each frame's encode.
    func beginFrame() {
        guard enabled else { return }
        labels.removeAll(keepingCapacity: true)
        nextIndex = 0
    }

    /// A compute-pass descriptor that samples the GPU timestamp at this encoder's
    /// start + end boundaries, or nil when timing is off / out of sample slots
    /// (caller then makes a plain, untimed encoder).
    func descriptor(label: String) -> MTLComputePassDescriptor? {
        guard enabled, let sampleBuffer, nextIndex + 2 <= Self.capacity else { return nil }
        let pd = MTLComputePassDescriptor()
        guard let att = pd.sampleBufferAttachments[0] else { return nil }
        att.sampleBuffer = sampleBuffer
        att.startOfEncoderSampleIndex = nextIndex
        att.endOfEncoderSampleIndex = nextIndex + 1
        labels.append(label)
        nextIndex += 2
        return pd
    }

    /// Resolve the frame's per-pass timestamps (call from the command buffer's
    /// completion handler). `frameGpuMs` is the buffer's own gpu time, used as the
    /// ground-truth cross-check.
    func resolve(frameGpuMs: Double) {
        guard enabled, let sampleBuffer, let sidecarPath else { return }
        let usedLabels = labels                 // snapshot from the encode pass
        let used = usedLabels.count * 2
        guard used > 0 else { return }
        guard let data = try? sampleBuffer.resolveCounterRange(0..<used) else { return }
        var times: [String: Double] = [:]
        var sum = 0.0
        data.withUnsafeBytes { raw in
            let ts = raw.bindMemory(to: MTLCounterResultTimestamp.self)
            for (k, label) in usedLabels.enumerated() {
                let start = ts[2 * k].timestamp
                let end = ts[2 * k + 1].timestamp
                guard end > start else { continue }
                let ms = Double(end - start) * nsPerTick / 1_000_000.0
                times[label, default: 0] += ms
                sum += ms
            }
        }
        // Cross-check the tick→ms conversion against the command buffer's own
        // gpu seconds. They won't match exactly (gaps between passes, encoder
        // overhead) but should be the same order; a wild mismatch ⇒ bad scale.
        let ratio = frameGpuMs > 0 ? sum / frameGpuMs : 0
        let verified = ratio >= 0.4 && ratio <= 1.2

        // Report the CURRENT frame's breakdown (the sidecar is rewritten every
        // frame, so a late read shows a steady-state frame). A cumulative average
        // would be skewed by expensive cold-start frames — misleading enough that
        // an early-frame average can exceed a steady-state frame's total.
        lock.lock()
        frameCount += 1
        let n = frameCount
        lock.unlock()
        let perPass = times.sorted { $0.value > $1.value }

        var out = "frame #\(n)  frameGpuMs=\(String(format: "%.3f", frameGpuMs))"
            + "  sumTimedPassMs=\(String(format: "%.3f", sum))"
            + "  timedFraction=\(String(format: "%.2f", ratio)) "
            + (verified ? "VERIFIED (tick→ms scale cross-checks vs frame gpu time)"
                        : "UNVERIFIED — tick scale suspect or many passes untimed") + "\n"
        for (l, ms) in perPass { out += String(format: "  %-22@ %.3f ms\n", l as NSString, ms) }
        try? out.write(toFile: sidecarPath, atomically: true, encoding: .utf8)
    }
}
