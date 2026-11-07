import Metal
import XCTest
import simd
@testable import VisualizerRendering

/// `PBDSolver.encodeSingleEncoder` (a blit and one compute encoder a substep) lands a rope exactly where the encoder-a-
/// sub-pass path puts it. Without long-range links: those run uncoloured (racing threads), so two runs of either path
/// drift apart by themselves; every coloured group is race-free and the two paths must agree to the bit.
@MainActor
final class PBDSingleEncoderTests: XCTestCase {
    private func rope(_ device: MTLDevice, selfCollide: Bool = false) throws -> PBDSolver {
        let s = try XCTUnwrap(PBDSolver(device: device, maxParticles: 26, maxConstraints: 26 * 8))
        s.configureChain(count: 26, segLength: 0.014, anchorFirst: true, bendStiffness: 0.5, longRangeStiffness: 0,
                         primaryCompliance: 2e-5)
        s.gravity = 9.81; s.damping = 0.998; s.floorY = -0.2; s.constraintIterations = 16
        if selfCollide { s.selfCollideEnabled = true; s.selfCollideRadius = 0.006 }
        // Laid out sideways, so it swings down and hits the floor.
        s.particleBuffer.write((0..<26).map { PBDParticle(position: SIMD3(Float($0) * 0.014, 0, 0), invMass: $0 == 0 ? 0 : 1) })
        return s
    }

    private func compare(selfCollide: Bool, viaProperty: Bool) throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let a = try rope(device, selfCollide: selfCollide), b = try rope(device, selfCollide: selfCollide)
        b.singleEncoder = viaProperty
        let h: Float = 1.0 / 240
        var hooks = 0
        for _ in 0..<60 {
            let cb = try XCTUnwrap(queue.makeCommandBuffer())
            for _ in 0..<4 { a.encode(to: cb, dt: h) }
            for _ in 0..<4 {
                if viaProperty { b.encode(to: cb, dt: h) }
                else { b.encodeSingleEncoder(to: cb, dt: h, label: "test", before: { _ in hooks += 1 }, after: { _ in hooks += 1 }) }
            }
            cb.commit(); cb.waitUntilCompleted()
        }
        let pa = a.particleBuffer.contents, pb = b.particleBuffer.contents
        var lowest: Float = 0
        for i in 0..<26 {
            XCTAssertEqual(pa[i].position, pb[i].position, "particle \(i)")
            lowest = min(lowest, pa[i].position.y)
        }
        XCTAssertLessThan(lowest, -0.1, "the rope should have swung down")
        XCTAssertEqual(hooks, viaProperty ? 0 : 60 * 4 * 2)
    }

    func testSingleEncoderMatchesPerPass() throws { try compare(selfCollide: false, viaProperty: false) }
    func testSingleEncoderPropertyMatchesPerPass() throws { try compare(selfCollide: false, viaProperty: true) }
    func testSingleEncoderWithSelfCollisionMatchesPerPass() throws { try compare(selfCollide: true, viaProperty: false) }
}
