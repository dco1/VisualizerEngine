import XCTest
import simd
@testable import VisualizerHumans

final class GeneratedHumanTests: XCTestCase {

    func testDatasetLoads() throws {
        let d = HumanDataset.shared
        XCTAssertEqual(d.origPositions.count, 19158)
        XCTAssertGreaterThan(d.renderOrigIndex.count, 14000)
        XCTAssertEqual(d.triangles.count % 3, 0)
        XCTAssertGreaterThan(d.morphs.count, 300)
        XCTAssertEqual(d.bones.count, 163)
        XCTAssertGreaterThan(d.joints.count, 300)
        // Every triangle index must be a valid render vertex; every render vertex a valid orig.
        let rc = UInt32(d.renderOrigIndex.count)
        XCTAssertTrue(d.triangles.allSatisfy { $0 < rc })
        let oc = UInt32(d.origPositions.count)
        XCTAssertTrue(d.renderOrigIndex.allSatisfy { $0 < oc })
    }

    func testMacroWeightsPartitionSensibly() {
        let d = HumanDataset.shared
        // A pure young adult: age anchors should put everything on "young".
        let w = MacroBlend.weights(for: HumanSpec(ageYears: 25, gender: 0, muscle: 0.5, weight: 0.5), dataset: d)
        XCTAssertFalse(w.isEmpty)
        for (idx, weight) in w {
            XCTAssertGreaterThan(weight, 0)
            XCTAssertLessThanOrEqual(weight, 1.001)
            let name = d.morphs[idx].name
            XCTAssertFalse(name.contains("-old"), "unexpected old target \(name) for a 25-year-old")
            XCTAssertFalse(name.contains("male") && !name.contains("female"),
                           "unexpected male target \(name) for gender 0")
        }
        // gender 0 must include female universal target
        XCTAssertTrue(w.keys.contains { d.morphs[$0].name.contains("female") })
    }

    func testHeightSliderChangesHeight() {
        let short = GeneratedHuman(spec: HumanSpec(ageYears: 30, gender: 1, height: 0.0))
        let tall = GeneratedHuman(spec: HumanSpec(ageYears: 30, gender: 1, height: 1.0))
        XCTAssertGreaterThan(tall.heightMeters, short.heightMeters + 0.15,
                             "max height should be clearly taller than min height")
        // Slider extremes are MakeHuman's dwarf↔giant range, deliberately beyond normal.
        XCTAssertGreaterThan(short.heightMeters, 1.0)
        XCTAssertLessThan(tall.heightMeters, 2.6)
        let average = GeneratedHuman(spec: HumanSpec(ageYears: 30, gender: 1, height: 0.5))
        XCTAssertEqual(average.heightMeters, 1.75, accuracy: 0.15,
                       "mid-slider adult male should land near real-world average")
    }

    func testChildIsSmallerThanAdult() {
        let child = GeneratedHuman(spec: HumanSpec(ageYears: 6))
        let adult = GeneratedHuman(spec: HumanSpec(ageYears: 30))
        XCTAssertLessThan(child.heightMeters, adult.heightMeters - 0.3)
    }

    func testSkinWeightsNormalized() {
        let h = GeneratedHuman(spec: HumanSpec())
        for w in h.skinWeights {
            XCTAssertEqual(w.sum(), 1.0, accuracy: 0.001)
        }
        let boneCount = UInt16(h.bones.count)
        for j in h.skinJoints {
            XCTAssertLessThan(j.max(), boneCount)
        }
    }

    func testSkeletonIsInsideBody() {
        let h = GeneratedHuman(spec: HumanSpec())
        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in h.positions {
            minP = simd_min(minP, p)
            maxP = simd_max(maxP, p)
        }
        let pad = SIMD3<Float>(repeating: 0.05)
        for bone in h.bones {
            XCTAssertTrue(all(bone.head .>= minP - pad) && all(bone.head .<= maxP + pad),
                          "bone \(bone.name) head \(bone.head) outside body bounds")
        }
        // Rest frames must be orthonormal (determinant ≈ +1).
        for bone in h.bones {
            let m = bone.restWorld
            let r = float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                             SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                             SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            XCTAssertEqual(r.determinant, 1.0, accuracy: 0.01, "non-orthonormal frame on \(bone.name)")
        }
    }

    func testDeterminismPerSpec() {
        let a = GeneratedHuman(spec: HumanSpec(ageYears: 40, seed: 7))
        let b = GeneratedHuman(spec: HumanSpec(ageYears: 40, seed: 7))
        XCTAssertEqual(a.positions[500], b.positions[500])
        XCTAssertEqual(a.heightMeters, b.heightMeters)
    }
}
