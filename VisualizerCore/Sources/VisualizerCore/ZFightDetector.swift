import Foundation
import OSLog
import simd

/// A flat surface patch in world space — a face, or a planar mesh reduced to its
/// plane + a footprint radius. The producer extracts these from geometry; the
/// detector only needs the plane and a lateral extent.
public struct SurfacePatch: Sendable {
    public var name: String
    public var center: SIMD3<Float>
    public var normal: SIMD3<Float>   // unit
    public var radius: Float          // lateral footprint (how far the patch spans in-plane)
    public init(name: String, center: SIMD3<Float>, normal: SIMD3<Float>, radius: Float) {
        self.name = name; self.center = center
        let l = simd_length(normal)
        self.normal = l > 1e-9 ? normal / l : SIMD3(0, 1, 0)
        self.radius = radius
    }
}

/// Deterministic **z-fighting / coplanar** detector — the "two surfaces are
/// stacked at the same depth and will flicker" warning a real engine surfaces.
///
/// Two patches z-fight when they are (a) nearly coplanar (their normals are
/// parallel or anti-parallel), (b) separated along that normal by less than a
/// tiny depth epsilon, and (c) overlapping laterally (their footprints touch).
/// All three together are the exact condition for the depth buffer to flip
/// between them frame to frame. The detector reports each offending pair with
/// its separation, so a fix (nudge one surface, or merge them) is targeted.
public enum ZFightDetector {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "ZFightDetector")

    public struct Pair: Sendable {
        public var a: String
        public var b: String
        public var separation: Float   // metres along the shared normal
    }

    public struct Report: Sendable {
        public var patchCount: Int
        public var fightingPairs: Int
        public var worst: [Pair]       // smallest separation first
        public var depthEpsilon: Float
        public var isClean: Bool { fightingPairs == 0 }
    }

    /// `depthEpsilon` is the along-normal gap below which two coplanar,
    /// laterally-overlapping patches are treated as z-fighting (default 1 mm).
    /// `parallelDot` is how parallel the normals must be to count as coplanar
    /// (default 0.999 ≈ within ~2.5°).
    @discardableResult
    public static func audit(_ patches: [SurfacePatch],
                             depthEpsilon: Float = 0.001,
                             parallelDot: Float = 0.999,
                             maxReported: Int = 8) -> Report {
        var pairs: [Pair] = []
        for i in 0..<patches.count {
            for j in (i + 1)..<patches.count {
                let a = patches[i], b = patches[j]
                // (a) coplanar: normals parallel or anti-parallel
                let nd = simd_dot(a.normal, b.normal)
                guard abs(nd) >= parallelDot else { continue }
                let delta = b.center - a.center
                // (b) tiny separation measured along the shared normal
                let sep = abs(simd_dot(delta, a.normal))
                guard sep < depthEpsilon else { continue }
                // (c) lateral overlap: in-plane distance < combined footprint
                let lateral = simd_length(delta - simd_dot(delta, a.normal) * a.normal)
                guard lateral < (a.radius + b.radius) else { continue }
                pairs.append(Pair(a: a.name, b: b.name, separation: sep))
            }
        }
        pairs.sort { $0.separation < $1.separation }
        let report = Report(patchCount: patches.count, fightingPairs: pairs.count,
                            worst: Array(pairs.prefix(maxReported)), depthEpsilon: depthEpsilon)

        #if DEBUG
        if !report.isClean {
            log.error("""
            Z-FIGHT: \(report.fightingPairs) coplanar overlapping pair(s) within \
            \(depthEpsilon)m — they will flicker as the depth buffer flips. Nudge one \
            surface apart or merge them.
            """)
        }
        #endif
        return report
    }
}
