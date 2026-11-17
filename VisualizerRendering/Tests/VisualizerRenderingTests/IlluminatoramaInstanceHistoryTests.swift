import XCTest
import simd
@testable import VisualizerRendering

/// Each draw's motion vector must reproject with its OWN last-frame record. The pairing
/// (`InstanceSlotKey`, `previousSlots`) is what `alignPreviousInstances` lays last frame's
/// records out by.
@MainActor
final class IlluminatoramaInstanceHistoryTests: XCTestCase {
    typealias R = IlluminatoramaRenderer

    private func ref(_ kind: String, x: Float, id: UInt64 = 0) -> R.InstanceRef {
        R.InstanceRef(meshKind: .custom(kind), data: IlluminatoramaInstance(modelMatrix: Self.at(x)), persistentID: id)
    }

    static func at(_ x: Float) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4(x, 0.3, 0, 1)
        return m
    }

    /// Source indices in grouped order, as `uploadInstances` lays them: kinds in first-seen order.
    private func grouped(_ list: [R.InstanceRef]) -> [Int] {
        var order: [R.MeshKind] = []
        var by: [R.MeshKind: [Int]] = [:]
        for (i, r) in list.enumerated() {
            if by[r.meshKind] == nil { order.append(r.meshKind) }
            by[r.meshKind, default: []].append(i)
        }
        return order.flatMap { by[$0]! }
    }

    /// Who each of `now`'s grouped slots gets as its previous record (an x position), new draws as nil.
    private func pairing(before: [R.InstanceRef], now: [R.InstanceRef]) -> [Float?] {
        let bOrder = grouped(before), nOrder = grouped(now)
        let from = R.previousSlots(now: R.slotKeys(now, order: nOrder), before: R.slotKeys(before, order: bOrder))
        return from.map { $0.map { before[bOrder[$0]].data.modelMatrix.columns.3.x } }
    }

    func testAKindComingOnScreenLeavesTheOthersTheirOwnHistory() {
        let before = [ref("a", x: 1), ref("c", x: 3)]
        let now = [ref("a", x: 1.1), ref("b", x: 2), ref("c", x: 3.1)]
        // Slots now: a, b, c. Before the fix c's slot (2) read last frame's slot 2: nothing / another's.
        XCTAssertEqual(pairing(before: before, now: now), [1, nil, 3])
    }

    func testAKindLeavingLeavesTheOthersTheirOwnHistory() {
        let before = [ref("a", x: 1), ref("b", x: 2), ref("c", x: 3)]
        let now = [ref("a", x: 1), ref("c", x: 3)]
        XCTAssertEqual(pairing(before: before, now: now), [1, 3])
    }

    func testUnnamedDrawsOfAKindPairInOrder() {
        let before = [ref("a", x: 1), ref("a", x: 2)]
        let now = [ref("a", x: 1.5), ref("a", x: 2.5), ref("a", x: 9)]
        XCTAssertEqual(pairing(before: before, now: now), [1, 2, nil])
    }

    func testNamedDrawsFindThemselvesWhenTheirKindIsShared() {
        // Two hats share a mesh; a third comes on screen BEFORE them in the list.
        let before = [ref("brim", x: 1, id: 10), ref("brim", x: 2, id: 20)]
        let now = [ref("brim", x: 0, id: 30), ref("brim", x: 1, id: 10), ref("brim", x: 2, id: 20)]
        XCTAssertEqual(pairing(before: before, now: now), [nil, 1, 2])
    }

    func testASteadyListPairsSlotForSlot() {
        let list = [ref("a", x: 1), ref("b", x: 2, id: 5), ref("a", x: 3)]
        let keys = R.slotKeys(list, order: grouped(list))
        XCTAssertEqual(R.previousSlots(now: keys, before: keys), Array(0..<list.count).map { Optional($0) })
    }

    /// Through a real renderer: a kind coming on screen ahead of a box; the box's slot must read the box's
    /// own last-frame matrix as its previous record, and the newcomer its own (no motion).
    func testRendererBindsEachDrawsOwnPreviousRecord() throws {
        let cam = IlluminatoramaCamera(position: SIMD3(0, 1.2, 4), target: SIMD3(0, 0.2, 0), up: SIMD3(0, 1, 0),
                                       fovYRadians: .pi / 4, aspect: 1, zNear: 0.05, zFar: 50)
        let r = try IlluminatoramaRenderer(engine: .shared, width: 96, height: 96, camera: cam)
        r.setMesh(.box, IlluminatoramaMesh.unitBox(device: r.device))
        r.setMesh(.custom("second"), IlluminatoramaMesh.unitBox(device: r.device))
        r.instances = [.init(meshKind: .box, data: IlluminatoramaInstance(modelMatrix: Self.at(1)))]
        _ = r.render(blocking: true)
        _ = r.render(blocking: true)
        r.instances = [.init(meshKind: .custom("second"), data: IlluminatoramaInstance(modelMatrix: Self.at(-1))),
                       .init(meshKind: .box, data: IlluminatoramaInstance(modelMatrix: Self.at(1.2)))]
        _ = r.render(blocking: true)
        let prev = r.previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: 2)
        let slot = { (kind: R.MeshKind) in r.gbs.meshGroupRange[kind]!.lowerBound }
        XCTAssertEqual(prev[slot(.box)].modelMatrix.columns.3.x, 1, accuracy: 1e-6, "the box reprojects with its own last matrix")
        XCTAssertEqual(prev[slot(.custom("second"))].modelMatrix.columns.3.x, -1, accuracy: 1e-6, "a newcomer has no motion")
        // The next frame, steady again: the plain ping-pong, already in this layout.
        _ = r.render(blocking: true)
        XCTAssertFalse(r.gbs.prevAlignedActive)
        let steady = r.previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: 2)
        XCTAssertEqual(steady[slot(.box)].modelMatrix.columns.3.x, 1.2, accuracy: 1e-6)
    }
}
