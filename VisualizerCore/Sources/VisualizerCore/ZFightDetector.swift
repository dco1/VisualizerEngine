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
    /// Optional ORIENTED half-extents in the patch's own plane.
    ///
    /// `radius` alone models a face as a disc, which is fine for the small
    /// square decals the SceneKit path emits and badly wrong for anything long:
    /// a 12 × 136 m roadway gets a ~68 m radius, so it "laterally overlaps"
    /// every coplanar face in the scene and the report fills with pairs that are
    /// nowhere near each other. (Measured on City Street Ultra: 2618 reported
    /// pairs, essentially all of them the main roadway against faces at the far
    /// end of the street.) When both patches carry these, the overlap test is a
    /// real 2D separating-axis test between two rectangles instead.
    public var halfU: SIMD3<Float>?
    public var halfV: SIMD3<Float>?
    /// Which OBJECT this face belongs to. Two faces of the same object are
    /// never a defect, however coplanar they are.
    ///
    /// One drawn thing is often several instances sharing a transform — a car
    /// is a body, glass, wheels and lamps on one matrix — so their bounding
    /// boxes coincide by construction and every face pairs with its opposite
    /// number. That produced 44 findings on the parked cars alone, all of them
    /// a car fighting itself. `nil` means ungrouped (never excluded).
    public var owner: Int?

    public init(name: String, center: SIMD3<Float>, normal: SIMD3<Float>, radius: Float,
                halfU: SIMD3<Float>? = nil, halfV: SIMD3<Float>? = nil,
                owner: Int? = nil) {
        self.name = name; self.center = center
        let l = simd_length(normal)
        self.normal = l > 1e-9 ? normal / l : SIMD3(0, 1, 0)
        self.radius = radius
        self.halfU = halfU; self.halfV = halfV
        self.owner = owner
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

    /// 2D separating-axis test between two oriented rectangles sharing a plane.
    /// Both are given by their half-extent vectors; `delta` is B's centre minus
    /// A's. Any axis that separates them means they do not overlap.
    private static func rectanglesOverlap(delta: SIMD3<Float>, normal: SIMD3<Float>,
                                          au: SIMD3<Float>, av: SIMD3<Float>,
                                          bu: SIMD3<Float>, bv: SIMD3<Float>,
                                          margin: Float) -> Bool {
        // Work in-plane: strip any along-normal component from the offset.
        let d = delta - simd_dot(delta, normal) * normal
        for axis in [au, av, bu, bv] {
            let len = simd_length(axis)
            guard len > 1e-9 else { continue }
            let n = axis / len
            let ra = abs(simd_dot(au, n)) + abs(simd_dot(av, n))
            let rb = abs(simd_dot(bu, n)) + abs(simd_dot(bv, n))
            // `- margin`: surfaces that merely ABUT are not fighting. A roadway
            // and the cross-street arms beside it share the ground plane and
            // touch along a kerb line; a sidewalk slab sits against the road
            // edge. Exact-touching counted as overlap turned all of those into
            // findings (measured: 540 pairs, nearly all of them adjacency).
            // Real z-fighting needs real shared area.
            if abs(simd_dot(d, n)) >= ra + rb - margin { return false }
        }
        return true
    }

    private static let log = Logger(subsystem: AppLog.subsystem, category: "ZFightDetector")

    public struct Pair: Sendable {
        public var a: String
        public var b: String
        public var separation: Float   // metres along the shared normal
        /// Where the fight is, in world space. A finding you cannot point at on
        /// screen costs a round of guessing to locate.
        public var at: SIMD3<Float> = .zero
        /// The shared face normal, so "which way does it face" is answerable.
        public var normal: SIMD3<Float> = .zero
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
    /// `requireSameFacing` keeps only pairs whose normals point the SAME way.
    ///
    /// Needed once patches describe FACES of solid objects rather than flat
    /// decals. Two coplanar faces pointing opposite ways are ordinary contact —
    /// a kerb against a slab, a cornice on a parapet — and the far one is culled,
    /// so nothing fights. Two pointing the same way are both rasterised at the
    /// same depth, which is the actual defect. Default `false` keeps every
    /// existing caller byte-identical.
    @discardableResult
    public static func audit(_ patches: [SurfacePatch],
                             depthEpsilon: Float = 0.001,
                             parallelDot: Float = 0.999,
                             requireSameFacing: Bool = false,
                             overlapMargin: Float = 0.01,
                             maxReported: Int = 8) -> Report {
        var pairs: [Pair] = []
        for i in 0..<patches.count {
            for j in (i + 1)..<patches.count {
                let a = patches[i], b = patches[j]
                // (0) same object — not a defect, see `SurfacePatch.owner`.
                if let oa = a.owner, let ob = b.owner, oa == ob { continue }
                // (a) coplanar: normals parallel or anti-parallel
                let nd = simd_dot(a.normal, b.normal)
                guard requireSameFacing ? (nd >= parallelDot) : (abs(nd) >= parallelDot)
                else { continue }
                let delta = b.center - a.center
                // (b) tiny separation measured along the shared normal
                let sep = abs(simd_dot(delta, a.normal))
                guard sep < depthEpsilon else { continue }
                // (c) lateral overlap. Rectangles when both patches describe
                //     one, discs otherwise.
                if let au = a.halfU, let av = a.halfV, let bu = b.halfU, let bv = b.halfV {
                    guard rectanglesOverlap(delta: delta, normal: a.normal,
                                            au: au, av: av, bu: bu, bv: bv,
                                            margin: overlapMargin) else { continue }
                } else {
                    let lateral = simd_length(delta - simd_dot(delta, a.normal) * a.normal)
                    guard lateral < (a.radius + b.radius) else { continue }
                }
                pairs.append(Pair(a: a.name, b: b.name, separation: sep,
                                  at: (a.center + b.center) * 0.5, normal: a.normal))
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
