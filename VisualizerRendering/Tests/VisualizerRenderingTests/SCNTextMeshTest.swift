import XCTest
import SceneKit
import Metal
@testable import VisualizerRendering

@MainActor
final class SCNTextMeshTest: XCTestCase {
    func testSCNTextConversion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let text = SCNText(string: "ABC", extrusionDepth: 0.05)
        text.font = NSFont.boldSystemFont(ofSize: 1.0)
        text.flatness = 0.1
        _ = text.boundingBox   // force layout

        let direct = IlluminatoramaMesh.from(scnGeometry: text, device: device)
        print("SCNTEXT direct = \(direct == nil ? "nil" : "ok (\(direct!.indexCount) idx)")")

        // Flattened clone bakes the lazy text into a concrete SCNGeometry.
        let node = SCNNode(geometry: text)
        let flat = node.flattenedClone()
        if let g = flat.geometry {
            let m = IlluminatoramaMesh.from(scnGeometry: g, device: device)
            print("SCNTEXT flattened = \(m == nil ? "nil" : "ok (\(m!.indexCount) idx)")  elements=\(g.elements.count)")
        } else {
            print("SCNTEXT flattened: no geometry")
        }
    }
}
