import Foundation
import simd

// ── CoinDiagnostics ───────────────────────────────────────────────────────────
//
// Host-side correctness probe for CoinDEMSolver — the gate for "is the sim
// flawless?" before any cabinet/pusher work (mirrors how PBD tuned against a
// sustained-interpenetration metric). Reads the shared coinBuffer after a frame's
// command buffer has completed and reports the failure modes that matter for a
// coin pile: explosion (energy), tunnelling (below floor), and interpenetration.
//
// Penetration is an O(n²) scan over active coins — fine for the isolation tests
// (≤ a few thousand), not for the live tick loop.

@MainActor
public struct CoinDiagnostics {

    public struct Report: CustomStringConvertible {
        public var activeCount: Int
        public var kineticEnergy: Float      // Σ ½|v|²  (unit mass)
        public var maxSpeed: Float
        public var belowFloorCount: Int      // tunnelling
        public var minY: Float
        public var maxY: Float               // pile height
        public var maxPenetration: Float     // worst coin–coin MTV overlap (metres)
        public var meanPenetration: Float    // over penetrating pairs

        public var description: String {
            String(format:
                "coins=%d  KE=%.4f  vmax=%.3f  belowFloor=%d  y=[%.4f,%.4f]  pen(max=%.4f,mean=%.4f)",
                activeCount, kineticEnergy, maxSpeed, belowFloorCount, minY, maxY,
                maxPenetration, meanPenetration)
        }
    }

    public static func measure(_ solver: CoinDEMSolver) -> Report {
        let n = solver.highWater
        let ptr = solver.coinBuffer.buffer.contents().bindMemory(to: CoinBody.self, capacity: max(1, n))
        let R = solver.coinRadius, h = solver.halfThickness
        let floor = solver.floorY

        // Per-body ORIENTED collision state. Penetration is measured with the SAME
        // oriented capped-cylinder SAT the solver de-penetrates with (face normals +
        // centre line), NOT an upright-cylinder approximation. The old approximation
        // treated a coin leaning on a neighbour's rim as deeply interpenetrating — a
        // 7° tilt read as ~1.3h of fake overlap — disagreeing with the GPU contact
        // model by ~100×. Reading orientation here makes the host metric agree with
        // physical reality (and with `solver.measurePenetration`).
        struct Body { var p: SIMD3<Float>; var n: SIMD3<Float>; var r: Float; var ht: Float }
        var active: [Body] = []
        active.reserveCapacity(n)
        var ke: Float = 0, vmax: Float = 0
        var below = 0
        var minY = Float.greatestFiniteMagnitude, maxY = -Float.greatestFiniteMagnitude

        for i in 0..<n {
            let b = ptr[i]
            if b.posInvMass.w == 0 { continue }       // inactive
            let p = SIMD3(b.posInvMass.x, b.posInvMass.y, b.posInvMass.z)
            let v = SIMD3(b.vel.x, b.vel.y, b.vel.z)
            let q = simd_quatf(ix: b.orient.x, iy: b.orient.y, iz: b.orient.z, r: b.orient.w)
            // Per-body collider dims (mixed pile): radius rides prevPos.w, half-thick
            // rides vel.w — same fallbacks as cdRadiusOf/cdHalfThickOf in the kernel.
            let br = b.prevPos.w > 1e-4 ? b.prevPos.w : R
            let bht = b.vel.w > 1e-5 ? b.vel.w : h
            active.append(Body(p: p, n: simd_normalize(q.act(SIMD3(0, 1, 0))), r: br, ht: bht))
            let sp = simd_length(v)
            ke += 0.5 * sp * sp
            vmax = max(vmax, sp)
            if p.y < floor - h - 1e-4 { below += 1 }
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }

        // Oriented disk-vs-disk SAT (mirrors coinMeasurePenetration in CoinDEM.metal):
        // separating-axis test over each disk's face normal plus the centre line; the
        // smallest positive overlap is the MTV depth.
        func orientedOverlap(_ a: Body, _ b: Body) -> Float {
            let D = a.p - b.p
            let reach = (a.r * a.r + a.ht * a.ht).squareRoot()
                      + (b.r * b.r + b.ht * b.ht).squareRoot()
            if simd_length_squared(D) > reach * reach { return 0 }
            let dlen = simd_length(D)
            let axes = [a.n, b.n, dlen > 1e-5 ? D / dlen : SIMD3<Float>(0, 1, 0)]
            var minDepth = Float.greatestFiniteMagnitude
            for ax in axes {
                let di = abs(simd_dot(a.n, ax)), dj = abs(simd_dot(b.n, ax))
                let ei = a.ht * di + a.r * max(0, 1 - di * di).squareRoot()
                let ej = b.ht * dj + b.r * max(0, 1 - dj * dj).squareRoot()
                let depth = (ei + ej) - abs(simd_dot(D, ax))
                if depth <= 0 { return 0 }            // a separating axis exists
                minDepth = min(minDepth, depth)
            }
            return minDepth
        }

        var maxPen: Float = 0, sumPen: Float = 0, penCount = 0
        for i in 0..<active.count {
            for j in (i + 1)..<active.count {
                let mtv = orientedOverlap(active[i], active[j])
                if mtv > 0 {
                    maxPen = max(maxPen, mtv)
                    sumPen += mtv; penCount += 1
                }
            }
        }

        return Report(
            activeCount: active.count,
            kineticEnergy: ke, maxSpeed: vmax,
            belowFloorCount: below,
            minY: active.isEmpty ? 0 : minY, maxY: active.isEmpty ? 0 : maxY,
            maxPenetration: maxPen,
            meanPenetration: penCount > 0 ? sumPen / Float(penCount) : 0)
    }
}
