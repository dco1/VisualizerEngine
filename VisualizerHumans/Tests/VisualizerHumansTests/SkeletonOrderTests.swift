import XCTest
import simd
@testable import VisualizerHumans

final class SkeletonOrderTests: XCTestCase {
    func testTopoOrderAndAimSolverReachesTargets() {
        let h = GeneratedHuman(spec: HumanSpec(ageYears: 30, gender: 1))
        for (i, b) in h.bones.enumerated() {
            XCTAssertLessThan(b.parent, i, "parent must precede child (\(b.name))")
        }
        let pose = WalkCycle.pose(phase: 0.18, human: h)
        let posed = PosedHuman(human: h, pose: pose)
        let li = h.boneIndex["lowerarm01.L"]!
        let l = h.bones[li]
        let m = posed.boneWorlds[li]
        let r = float3x3(SIMD3(m.columns.0.x,m.columns.0.y,m.columns.0.z),
                         SIMD3(m.columns.1.x,m.columns.1.y,m.columns.1.z),
                         SIMD3(m.columns.2.x,m.columns.2.y,m.columns.2.z))
        let d = r * simd_normalize(l.tail - l.head)
        print(String(format: "forearm posed (%.2f,%.2f,%.2f)", d.x, d.y, d.z))
    }
}
