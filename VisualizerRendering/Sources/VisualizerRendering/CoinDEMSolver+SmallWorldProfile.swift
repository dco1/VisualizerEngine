import Foundation
import Metal

// ── Small-world PHASE PROFILE (diagnostic, default off) ────────────────────────────────────
//
// `VIZ_COINDEM_SW_PROFILE=<file>`: every small-world frame is encoded as one dispatch PER PHASE
// GROUP PER SUBSTEP (the `profileHook` seam's split — the kernel's phases in the kernel's order,
// each dispatch running only its own `CD_SW_PH_*` bits over one substep; the state they share lives
// in device memory, so the frame computes what the single dispatch computes), each dispatch timed
// with GPU timestamps at its encoder boundaries. The file is rewritten every 60 frames with the
// mean GPU ms per FRAME spent in each group, the frame's total, and the frames' body / contact
// counts. Only for measuring where a small world's frame time goes: the split adds a dispatch
// bubble per group, so the total reads higher than the one-dispatch frame.
@MainActor
final class CoinSmallWorldProfiler {
    static let path: String? = ProcessInfo.processInfo.environment["VIZ_COINDEM_SW_PROFILE"]

    /// Phase groups in the kernel's order within a substep, then the once-per-frame tail.
    /// `VIZ_COINDEM_SW_PROFILE_SPLITSOLVE=1` times the contact solve, the joint solve and the feet's
    /// iteration pass as three dispatches (each running all its iterations — a changed order, so the
    /// simulation drifts from the real one: attribution only).
    static let splitSolve = ProcessInfo.processInfo.environment["VIZ_COINDEM_SW_PROFILE_SPLITSOLVE"] == "1"
    static let groups: [(name: String, mask: UInt32)] = {
        let solve: UInt32 = 1 << 6, jsolve: UInt32 = 1 << 11, feetIt: UInt32 = 1 << 10
        var g: [(name: String, mask: UInt32)] = [("intVel", 1 << 8), ("feet", 1 << 0), ("generate", 1 << 1),
                                                 ("poly", 1 << 2), ("color", 1 << 3)]
        g.append(("warm+prep", UInt32(1 << 4) | UInt32(1 << 14)))
        g.append(("jointPrep", 1 << 5))
        if splitSolve {
            g += [("solve", solve), ("jointSolve", jsolve), ("feetIt", feetIt)]
        } else {
            g.append(("solve+jointSolve+feetIt", solve | jsolve | feetIt))
        }
        g += [("snapshot", 1 << 12), ("intPos", 1 << 9)]
        return g
    }()
    static let tail: (name: String, mask: UInt32) = ("sleep+transform", (1 << 7) | (1 << 13))

    private let sampleBuffer: MTLCounterSampleBuffer
    private let capacity: Int
    private let nsPerTick: Double
    private var sums: [String: Double] = [:]
    private var frames = 0
    private var bodies = 0, contacts = 0

    init?(device: MTLDevice) {
        guard Self.path != nil, device.supportsCounterSampling(.atDispatchBoundary) || device.supportsCounterSampling(.atStageBoundary),
              let ts = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else { return nil }
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = ts
        d.storageMode = .shared
        capacity = 2 * (Self.groups.count * 16 + 1)
        d.sampleCount = capacity
        guard let b = try? device.makeCounterSampleBuffer(descriptor: d) else { return nil }
        sampleBuffer = b
        var c0: MTLTimestamp = 0, g0: MTLTimestamp = 0, c1: MTLTimestamp = 0, g1: MTLTimestamp = 0
        device.__sampleTimestamps(&c0, gpuTimestamp: &g0)
        var spin = 0.0; for i in 0..<200_000 { spin += Double(i) * 1.0000001 }
        device.__sampleTimestamps(&c1, gpuTimestamp: &g1)
        let dc = Double(c1 &- c0), dg = Double(g1 &- g0)
        nsPerTick = (dg > 0 && dc > 0 && spin.isFinite) ? dc / dg : 1
    }

    /// Encode one frame of `steps` substeps as timed per-group dispatches through `encode`.
    func encodeFrame(_ cb: MTLCommandBuffer, steps: Int, bodies nb: Int, contacts nc: Int,
                     encode: (_ steps: Int, _ mask: UInt32, _ pass: MTLComputePassDescriptor?) -> Void) {
        var labels: [String] = []
        func pass() -> MTLComputePassDescriptor? {
            let i = labels.count * 2
            guard i + 2 <= capacity else { return nil }
            let pd = MTLComputePassDescriptor()
            guard let a = pd.sampleBufferAttachments[0] else { return nil }
            a.sampleBuffer = sampleBuffer
            a.startOfEncoderSampleIndex = i
            a.endOfEncoderSampleIndex = i + 1
            return pd
        }
        for _ in 0..<steps {
            for g in Self.groups {
                let p = pass()
                if p != nil { labels.append(g.name) }
                encode(1, g.mask, p)
            }
        }
        let p = pass()
        if p != nil { labels.append(Self.tail.name) }
        encode(0, Self.tail.mask, p)
        let buffer = sampleBuffer, scale = nsPerTick
        let frameLabels = labels
        cb.addCompletedHandler { [weak self] _ in
            guard let data = try? buffer.resolveCounterRange(0..<(frameLabels.count * 2)) else { return }
            var ms: [String: Double] = [:]
            data.withUnsafeBytes { raw in
                let ts = raw.bindMemory(to: MTLCounterResultTimestamp.self)
                for (k, l) in frameLabels.enumerated() where ts[2 * k + 1].timestamp > ts[2 * k].timestamp {
                    ms[l, default: 0] += Double(ts[2 * k + 1].timestamp - ts[2 * k].timestamp) * scale / 1e6
                }
            }
            Task { @MainActor [weak self] in self?.record(ms, bodies: nb, contacts: nc) }
        }
    }

    private func record(_ ms: [String: Double], bodies nb: Int, contacts nc: Int) {
        for (k, v) in ms { sums[k, default: 0] += v }
        frames += 1
        bodies += nb
        contacts += nc
        guard frames % 60 == 0, let path = Self.path else { return }
        let n = Double(frames)
        let total = sums.values.reduce(0, +) / n
        var text = String(format: "frames %d  mean GPU ms per frame (split dispatches): %.3f  bodies %.1f  contacts %.1f\n",
                          frames, total, Double(bodies) / n, Double(contacts) / n)
        for g in Self.groups.map(\.name) + [Self.tail.name] {
            text += String(format: "  %-26@ %7.3f ms  %5.1f %%\n", g as NSString, (sums[g] ?? 0) / n, 100 * (sums[g] ?? 0) / n / max(total, 1e-9))
        }
        try? text.write(toFile: path, atomically: true, encoding: .utf8)
    }
}
