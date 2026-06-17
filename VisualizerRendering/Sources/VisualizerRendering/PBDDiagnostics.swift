import Foundation
import OSLog
import simd
import VisualizerCore

// ── PBD DIAGNOSTICS ─────────────────────────────────────────────────────────
//
// Writes structured diagnostic lines about the PBD simulation state to a file
// in the sandbox Logs directory. Designed so the user can flip the toggle,
// reproduce the bug, and we can read the file directly — no copy-paste loop.
//
// Log path (resolved at runtime — `FileManager.default.urls(for: .libraryDirectory)`
// returns the sandboxed Library inside the container):
//
//   ~/Library/Containers/&lt;host-app-bundle-id&gt;/Data/Library/Logs/pbd-diag.log
//
// File is truncated each app launch so successive sessions don't get tangled.
// Writes are flushed on every line.
//
// WHAT GETS LOGGED
//
//   tick=12345 alive=8 worst=A↔B sep=...  +overlap details
//
// `worst` is the closest body pair by minimum capsule-capsule SDF distance.
// `sep` is the deficit (expected_separation − actual_separation); positive
// means visual interpenetration of that depth. Up to the top 5 offending
// pairs are listed per snapshot.
//
// The CPU-side capsule-capsule check mirrors the visual SDF surface, so a
// positive deficit here = a hot-dog-into-hot-dog visual overlap regardless of
// whether any individual particle is inside another body.

@MainActor
public final class PBDDiagnostics {

    public static let shared = PBDDiagnostics()

    private static let log = Logger(subsystem: AppLog.subsystem, category: "PBDDiag")

    /// Absolute path users (and Claude) read from. Reported by `pathLine()`
    /// at the top of every log so it's discoverable when only the file is
    /// shared somewhere else.
    public let path: URL

    private var handle: FileHandle?

    private init() {
        let logsDir = FileManager.default.urls(for: .libraryDirectory,
                                               in: .userDomainMask)[0]
            .appendingPathComponent("Logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: logsDir,
                                                 withIntermediateDirectories: true)
        path = logsDir.appendingPathComponent("pbd-diag.log")

        // Single-step rotation: at session start, move the previous session's
        // log (if any) to `pbd-diag.prev.log` and start a fresh main file.
        // This is the right place to do it — NOT on quit. Quitting before the
        // log is read would lose the data; rotating at launch preserves the
        // session that just ended on disk for as long as the user wants to
        // inspect it, while still avoiding unbounded growth and leaving at
        // most two small files on disk.
        let fm = FileManager.default
        let prevPath = logsDir.appendingPathComponent("pbd-diag.prev.log")
        if fm.fileExists(atPath: path.path) {
            try? fm.removeItem(at: prevPath)
            try? fm.moveItem(at: path, to: prevPath)
        }
        fm.createFile(atPath: path.path, contents: Data(), attributes: nil)
        handle = try? FileHandle(forWritingTo: path)

        Self.log.info("PBDDiag log: \(self.path.path, privacy: .public)")
        writeRaw("=== PBD diagnostics — session started \(Self.timestamp()) ===\n")
        writeRaw("path: \(path.path)\n")
        writeRaw("previous-session: \(prevPath.path)\n")
        writeRaw("---\n")
    }

    // ── Public API ─────────────────────────────────────────────────────────

    public func write(_ message: String) {
        writeRaw(Self.timestamp() + " " + message + "\n")
    }

    public func close() {
        try? handle?.close()
        handle = nil
    }

    // ── Internals ──────────────────────────────────────────────────────────

    private func writeRaw(_ s: String) {
        guard let handle, let data = s.data(using: .utf8) else { return }
        try? handle.write(contentsOf: data)
    }

    private static func timestamp() -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: Date())
    }
}

// ── Snapshot model + overlap detector ────────────────────────────────────────
//
// One snapshot per body, captured on CPU from the shared particle buffer. The
// capsule-capsule distance check between each pair of bodies (skipping self)
// gives us the *worst* separation deficit across the scene — a positive
// deficit is a visual interpenetration deeper than 0.
//
// Capsule-capsule = segment-to-segment closest distance, minus the sum of the
// two tube radii. Standard algorithm: clamp the unconstrained closest-points
// parameters into [0,1]² with the usual case analysis.

@MainActor
public struct PBDDiagSnapshot {
    public let ownerID: UInt32
    public let particles: [SIMD3<Float>]
    public let radius: Float
    /// Primary distance-constraint rest length. Used by chain-shape diagnostics
    /// to compute per-segment stretch %. Caller usually reads this from
    /// `solver.constraintBuffer.contents[0].restLength`.
    public let restLength: Float

    public init(ownerID: UInt32, particles: [SIMD3<Float>],
                radius: Float, restLength: Float) {
        self.ownerID    = ownerID
        self.particles  = particles
        self.radius     = radius
        self.restLength = restLength
    }
}

// Per-tube spine shape — primarily for diagnosing end-on kinking. Captures the
// two quantities a kink consists of:
//   • Stretch — actual segment length vs the primary constraint's rest length.
//     Large positive stretch = chain getting pulled apart; large negative
//     stretch = compression (the "hourglass pinch" signature).
//   • Kink angle — deviation from 180° (straight) at each interior hinge. A
//     fresh hot dog landing flat sits near 180° at every hinge; a visible kink
//     spikes a hinge below ~160°.
public struct PBDChainShape {
    public let ownerID: UInt32
    public let particleCount: Int
    /// Worst stretch as a signed fraction of rest length. +0.05 = 5 % extended;
    /// −0.20 = 20 % compressed. Magnitude is what matters for the verdict.
    public let worstStretch: Float
    public let worstStretchSegment: Int
    /// Deviation from 180° at the sharpest interior hinge, in degrees. 0 =
    /// perfectly straight; 90° = right-angle kink; 180° = folded onto itself.
    public let worstKinkDeg: Float
    public let worstKinkParticle: Int
}

@MainActor
public enum PBDDiagAnalyser {

    /// Per-pair report when overlap is detected.
    public struct Overlap {
        public let ownerA: UInt32
        public let ownerB: UInt32
        /// Index of the capsule in A's spine (a-side endpoint = particle i).
        public let capsuleA: Int
        public let capsuleB: Int
        /// `expected − actual`. Positive means visual interpenetration.
        public let deficit: Float
        /// Sum of the two tube radii (= the desired separation).
        public let expectedSeparation: Float
    }

    /// Per-snapshot chain shape — worst segment stretch and sharpest hinge
    /// angle. Used to diagnose end-on kinking: if `worstStretch` magnitude is
    /// large the primary constraints aren't converging (or compliance is too
    /// high); if `worstKinkDeg` is large but stretch is small, the bending
    /// constraints are the leak.
    public static func chainShape(_ s: PBDDiagSnapshot) -> PBDChainShape {
        let pts = s.particles
        let n = pts.count
        var worstStretch: Float = 0
        var worstStretchSeg = 0
        let rest = max(s.restLength, 1e-6)
        for i in 0..<(n - 1) {
            let len = simd_length(pts[i + 1] - pts[i])
            let stretch = (len / rest) - 1
            if abs(stretch) > abs(worstStretch) {
                worstStretch = stretch
                worstStretchSeg = i
            }
        }
        var worstKinkDeg: Float = 0
        var worstKinkIdx = 0
        if n >= 3 {
            for i in 1..<(n - 1) {
                let a = pts[i] - pts[i - 1]
                let b = pts[i + 1] - pts[i]
                let la = simd_length(a)
                let lb = simd_length(b)
                guard la > 1e-6, lb > 1e-6 else { continue }
                // cos(angle between a and b) — straight = +1, folded = -1.
                let c = max(-1.0, min(1.0, simd_dot(a, b) / (la * lb)))
                let deviationRad = acos(c)        // 0 = straight, π = folded
                let deviationDeg = deviationRad * 180 / .pi
                if deviationDeg > worstKinkDeg {
                    worstKinkDeg = deviationDeg
                    worstKinkIdx = i
                }
            }
        }
        return PBDChainShape(
            ownerID: s.ownerID, particleCount: n,
            worstStretch: worstStretch, worstStretchSegment: worstStretchSeg,
            worstKinkDeg: worstKinkDeg, worstKinkParticle: worstKinkIdx
        )
    }

    /// Worst-kinked snapshot across the scene. "Worst" weights stretch and
    /// kink angle equally — kink degrees / 5 (so 30° kink ≈ 6 % stretch) — so
    /// a single number drives the verdict. Returns nil for an empty list.
    public static func worstKinked(in snapshots: [PBDDiagSnapshot]) -> PBDChainShape? {
        var best: (score: Float, shape: PBDChainShape)? = nil
        for s in snapshots {
            let shape = chainShape(s)
            let score = abs(shape.worstStretch) * 100 + shape.worstKinkDeg * 0.2
            if best == nil || score > best!.score {
                best = (score, shape)
            }
        }
        return best?.shape
    }

    /// Scan every distinct pair of bodies. Returns up to `limit` worst
    /// overlaps, sorted by descending deficit. An empty result means every
    /// pair is at or beyond their expected separation — i.e. no visual
    /// interpenetration.
    public static func overlaps(in snapshots: [PBDDiagSnapshot],
                                limit: Int = 5) -> [Overlap] {
        var hits: [Overlap] = []
        for i in 0..<snapshots.count {
            let a = snapshots[i]
            guard a.particles.count >= 2 else { continue }
            for j in (i + 1)..<snapshots.count {
                let b = snapshots[j]
                guard b.particles.count >= 2 else { continue }
                let expected = a.radius + b.radius
                // Find the closest capsule pair (worst deficit).
                var worstDeficit: Float = 0
                var worstA = 0, worstB = 0
                for ai in 0..<(a.particles.count - 1) {
                    for bi in 0..<(b.particles.count - 1) {
                        let d = segSegDistance(
                            a.particles[ai], a.particles[ai + 1],
                            b.particles[bi], b.particles[bi + 1]
                        )
                        let deficit = expected - d
                        if deficit > worstDeficit {
                            worstDeficit = deficit
                            worstA = ai; worstB = bi
                        }
                    }
                }
                if worstDeficit > 0 {
                    hits.append(Overlap(
                        ownerA: a.ownerID, ownerB: b.ownerID,
                        capsuleA: worstA,  capsuleB: worstB,
                        deficit: worstDeficit,
                        expectedSeparation: expected
                    ))
                }
            }
        }
        hits.sort { $0.deficit > $1.deficit }
        return Array(hits.prefix(limit))
    }

    // Closest distance between two line segments [p1,p2] and [q1,q2].
    // Standard parametric closest-points algorithm with [0,1]² clamping —
    // see e.g. Christer Ericson, "Real-Time Collision Detection", §5.1.9.
    private static func segSegDistance(_ p1: SIMD3<Float>, _ p2: SIMD3<Float>,
                                       _ q1: SIMD3<Float>, _ q2: SIMD3<Float>) -> Float {
        let u = p2 - p1
        let v = q2 - q1
        let w = p1 - q1
        let a = simd_dot(u, u)
        let b = simd_dot(u, v)
        let c = simd_dot(v, v)
        let d = simd_dot(u, w)
        let e = simd_dot(v, w)
        let det = a * c - b * b
        var sN: Float; var sD = det
        var tN: Float; var tD = det
        let eps: Float = 1e-7

        if det < eps {
            // Parallel.
            sN = 0; sD = 1
            tN = e; tD = c
        } else {
            sN = b * e - c * d
            tN = a * e - b * d
            if sN < 0 { sN = 0; tN = e; tD = c }
            else if sN > sD { sN = sD; tN = e + b; tD = c }
        }

        if tN < 0 {
            tN = 0
            if -d < 0 { sN = 0 }
            else if -d > a { sN = sD }
            else { sN = -d; sD = a }
        } else if tN > tD {
            tN = tD
            let bd = -d + b
            if bd < 0 { sN = 0 }
            else if bd > a { sN = sD }
            else { sN = bd; sD = a }
        }

        let sc: Float = abs(sD) < eps ? 0 : sN / sD
        let tc: Float = abs(tD) < eps ? 0 : tN / tD
        let dP = w + sc * u - tc * v
        return simd_length(dP)
    }
}
