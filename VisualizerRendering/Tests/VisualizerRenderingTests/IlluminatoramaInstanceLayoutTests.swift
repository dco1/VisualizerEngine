import Metal
import XCTest
import simd
@testable import VisualizerRendering

/// The upload's kept layout (`InstanceLayout`): a list that only moved reuses its grouping and slot keys, a list whose
/// make-up changed regroups, and nothing derived from the mesh table goes stale when a mesh is replaced under a list
/// that holds still.
@MainActor
final class IlluminatoramaInstanceLayoutTests: XCTestCase {
    typealias R = IlluminatoramaRenderer

    private static func at(_ x: Float, _ y: Float = 0.3) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4(x, y, 0, 1)
        return m
    }

    private func ref(_ kind: String, x: Float, id: UInt64 = 0) -> R.InstanceRef {
        R.InstanceRef(meshKind: .custom(kind), data: IlluminatoramaInstance(modelMatrix: Self.at(x)), persistentID: id)
    }

    private func renderer() throws -> IlluminatoramaRenderer {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1.2, 4), target: SIMD3(0, 0.2, 0), up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 4, aspect: 1, zNear: 0.05, zFar: 50)
        return try IlluminatoramaRenderer(engine: .shared, width: 64, height: 64, camera: cam)
    }

    /// A triangle `offset` metres along x from the mesh's origin.
    private func triangle(_ device: MTLDevice, offset: Float) -> IlluminatoramaMesh {
        let n = SIMD3<Float>(0, 0, 1)
        let v = [SIMD3<Float>(offset, 0, 0), SIMD3(offset + 0.5, 0, 0), SIMD3(offset, 0.5, 0)]
            .map { IlluminatoramaVertex(position: $0, normal: n, uv: .zero) }
        return IlluminatoramaMesh(device: device, vertices: v, indices: [UInt16(0), 1, 2])
    }

    /// Bug: with the list held still the upload is skipped, so the cull bounds' key never moved when `setMesh` swapped
    /// a kind's mesh, and the group stayed culled by the old mesh's bounds.
    func testReplacingAMeshUnderAStillListRecullsIt() throws {
        let r = try renderer()
        r.setMesh(.custom("m"), triangle(r.device, offset: 60))       // far off to the side: culled
        r.instances = [ref("m", x: 0)]
        for _ in 0..<4 { _ = r.render(blocking: true) }
        XCTAssertGreaterThanOrEqual(r.stabilityDebug.full, 2, "the list must be still, so the upload is skipped")
        var before = r.visibilityCullStats
        _ = r.render(blocking: true)
        XCTAssertEqual(r.visibilityCullStats.gbufferDrawn - before.gbufferDrawn, 0, "off screen: culled")

        r.setMesh(.custom("m"), triangle(r.device, offset: 0))        // now in the middle of the frame
        before = r.visibilityCullStats
        _ = r.render(blocking: true)
        XCTAssertEqual(r.visibilityCullStats.gbufferDrawn - before.gbufferDrawn, 1, "the new mesh is on screen and must draw")
        XCTAssertEqual(r.visibilityCullStats.gbufferCulled - before.gbufferCulled, 0)
        XCTAssertTrue(r.gbs.meshGroupMeshes.first! === r.registeredMesh(.custom("m")), "each group's mesh follows the table")
    }

    func testAMovingListKeepsItsLayoutAndAChangedOneRegroups() throws {
        let r = try renderer()
        for k in ["a", "b", "c"] { r.setMesh(.custom(k), IlluminatoramaMesh.unitBox(device: r.device)) }
        r.instances = [ref("a", x: 1), ref("b", x: 2, id: 7), ref("a", x: 3), ref("c", x: 4)]
        _ = r.render(blocking: true)
        let serial = r.gbs.instanceLayout.serial
        let keys = r.gbs.lastSlotKeys
        XCTAssertEqual(r.gbs.meshGroups.map(\.kind), [.custom("a"), .custom("b"), .custom("c")])
        XCTAssertEqual(r.gbs.meshGroups.map(\.count), [2, 1, 1])
        XCTAssertEqual(r.gbs.meshGroupRange[.custom("a")], 0..<2)
        XCTAssertEqual(r.gbs.meshGroupRange[.custom("c")], 3..<4)

        // Moved, same make-up (kind strings built afresh, as a host does): the layout and keys are kept.
        r.instances = [ref("a", x: 1.1), ref("b", x: 2.1, id: 7), ref("a", x: 3.1), ref("c", x: 4.1)]
        _ = r.render(blocking: true)
        XCTAssertEqual(r.gbs.instanceLayout.serial, serial, "a list that only moved keeps its layout")
        XCTAssertEqual(r.gbs.lastSlotKeys, keys)
        let cur = r.currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: 4)
        XCTAssertEqual((0..<4).map { cur[$0].modelMatrix.columns.3.x }, [1.1, 3.1, 2.1, 4.1], "records in grouped order")

        // A kind comes in ahead of the others: regrouped, and every draw keeps its own history.
        r.instances = [ref("c", x: 0), ref("a", x: 1.2), ref("b", x: 2.2, id: 7), ref("a", x: 3.2), ref("c", x: 4.2)]
        _ = r.render(blocking: true)
        XCTAssertNotEqual(r.gbs.instanceLayout.serial, serial)
        XCTAssertEqual(r.gbs.meshGroups.map(\.kind), [.custom("c"), .custom("a"), .custom("b")])
        XCTAssertEqual(r.gbs.meshGroups.map(\.count), [2, 2, 1])
        XCTAssertTrue(r.gbs.prevAlignedActive)
        let prev = r.previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: 5)
        XCTAssertEqual((0..<5).map { prev[$0].modelMatrix.columns.3.x }, [4.1, 4.2, 1.1, 3.1, 2.1],
                       "c's old draw, the new c (no motion), the a's, b: each its own last record")
        XCTAssertEqual(r.gbs.meshGroupMeshes.count, 3)
        XCTAssertTrue(r.gbs.meshGroupMeshes[2] === r.registeredMesh(.custom("b")))
    }

    func testLayoutMatchesTheDictionaryGrouping() {
        let list = [ref("x", x: 0), ref("y", x: 1), ref("x", x: 2), ref("x", x: 3), ref("z", x: 4), ref("y", x: 5, id: 3)]
        let l = R.layout(of: list, reusing: .init())
        XCTAssertEqual(l.order, [0, 2, 3, 1, 5, 4])
        XCTAssertEqual(l.kindHashes, list.map(\.meshKind.hashValue))
        XCTAssertEqual(l.slotKeys, R.slotKeys(list, order: l.order))
        XCTAssertTrue(l.holds(list))
        XCTAssertFalse(l.holds(Array(list.dropLast())))
        var other = list
        other[5].persistentID = 4
        XCTAssertFalse(l.holds(other))
        other = list
        other[1].meshKind = .custom("w")
        XCTAssertFalse(l.holds(other))
    }

    func testSameInstancesAgreesWithEquality() {
        let a = [ref("a", x: 1), ref("b", x: 2, id: 9)]
        XCTAssertTrue(R.sameInstances(a, a))
        XCTAssertTrue(R.sameInstances(a, [ref("a", x: 1), ref("b", x: 2, id: 9)]))
        XCTAssertFalse(R.sameInstances(a, [ref("a", x: 1), ref("b", x: 2.5, id: 9)]))
        XCTAssertFalse(R.sameInstances(a, [ref("a", x: 1), ref("c", x: 2, id: 9)]))
        XCTAssertFalse(R.sameInstances(a, [ref("a", x: 1), ref("b", x: 2, id: 8)]))
        XCTAssertFalse(R.sameInstances(a, [ref("a", x: 1)]))
        var e = a
        e[0].lightEmission = SIMD3(1, 0, 0)
        XCTAssertFalse(R.sameInstances(a, e))
        e = a
        e[1].data.roughness = 0.77
        XCTAssertFalse(R.sameInstances(a, e))
        // A NaN that held still is unchanged (`==` alone would call it changed every frame).
        var n = a
        n[0].data.sheen = .nan
        XCTAssertTrue(R.sameInstances(n, n.map { $0 }))
        XCTAssertTrue(R.sameInstances([], []))
    }
}
