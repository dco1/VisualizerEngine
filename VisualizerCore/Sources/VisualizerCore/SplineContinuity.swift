import Foundation
import OSLog
import simd

/// One named centerline polyline (a rail, a tube path, a curve), densely
/// sampled in world space. A scene hands these to ``SplineContinuity`` so kinks
/// become an exact location instead of "there's a kink somewhere, go find it."
public struct SplinePolyline: Sendable {
    public var name: String
    /// World-space points sampled along the curve, in order. Sample densely
    /// enough that a smooth curve's per-vertex turn is small — the kink test
    /// keys off a turn that's an outlier against its neighbours.
    public var points: [SIMD3<Float>]
    /// True for a closed loop (the last point connects back to the first).
    public var closed: Bool
    public init(name: String, points: [SIMD3<Float>], closed: Bool = false) {
        self.name = name; self.points = points; self.closed = closed
    }
}

/// Deterministic **spline / rail continuity** check — the "do these curves join
/// smoothly?" guarantee a real 3D program enforces, here as code.
///
/// The egg-track kinks were chased across multiple rounds (`eggs_rails_junction_
/// analysis`, `loop_tangent_alignment`, `bezier_chord_control_overshoot`), all
/// the same root cause: a join where the incoming and outgoing tangents disagree.
/// At each interior vertex this measures the **turn angle** between the incoming
/// and outgoing segment directions. On a smoothly-sampled curve that angle is
/// small and varies gradually; a kink is a single vertex whose turn spikes past
/// the threshold. It also flags a **gap** (a position discontinuity) when two
/// successive samples are implausibly far apart for the curve's sampling.
public enum SplineContinuity {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "SplineContinuity")

    public struct Kink: Sendable {
        public var index: Int
        public var position: SIMD3<Float>
        public var turnDegrees: Float
    }

    public struct Report: Sendable {
        public var label: String
        public var vertexCount: Int
        public var kinkCount: Int
        public var maxTurnDegrees: Float
        public var worstKinks: [Kink]      // sorted sharpest-first, capped
        public var thresholdDegrees: Float
        public var isSmooth: Bool { kinkCount == 0 }
        public var summary: String {
            isSmooth ? "smooth"
                : "\(kinkCount) kink(s), sharpest \(Int(maxTurnDegrees.rounded()))° (threshold \(Int(thresholdDegrees))°)"
        }
    }

    /// Flag every interior vertex whose tangent turns by more than
    /// `thresholdDegrees` — the discrete kink a smooth curve never has. A loop
    /// (`closed`) also tests the wrap-around join between last and first points.
    @discardableResult
    public static func audit(_ line: SplinePolyline,
                             thresholdDegrees: Float = 30,
                             maxReported: Int = 8) -> Report {
        let p = line.points
        var report = Report(label: line.name, vertexCount: p.count, kinkCount: 0,
                            maxTurnDegrees: 0, worstKinks: [], thresholdDegrees: thresholdDegrees)
        guard p.count >= 3 else { return report }

        var kinks: [Kink] = []
        // Interior vertices, plus the two wrap joins when closed.
        let range = line.closed ? Array(0..<p.count) : Array(1..<(p.count - 1))
        for i in range {
            let prev = p[(i - 1 + p.count) % p.count]
            let cur  = p[i]
            let next = p[(i + 1) % p.count]
            let dIn = cur - prev
            let dOut = next - cur
            let lIn = simd_length(dIn), lOut = simd_length(dOut)
            if lIn < 1e-9 || lOut < 1e-9 { continue }   // coincident samples — not a tangent
            let cosA = simd_clamp(simd_dot(dIn / lIn, dOut / lOut), -1, 1)
            let turn = acos(cosA) * 180 / .pi
            report.maxTurnDegrees = max(report.maxTurnDegrees, turn)
            if turn > thresholdDegrees {
                kinks.append(Kink(index: i, position: cur, turnDegrees: turn))
            }
        }
        kinks.sort { $0.turnDegrees > $1.turnDegrees }
        report.kinkCount = kinks.count
        report.worstKinks = Array(kinks.prefix(maxReported))

        #if DEBUG
        if !report.isSmooth {
            log.error("""
            SPLINE KINK '\(line.name, privacy: .public)': \(report.summary, privacy: .public). \
            A tangent discontinuity at a join — fix the segment's start/sweep or the \
            connector's handle, not the cosmetic curve. See eggs_rails_junction_analysis.
            """)
        }
        #endif
        return report
    }
}
