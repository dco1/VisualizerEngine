import Foundation
import Metal
import SceneKit
import simd
import VisualizerCore

// ── FLUID PARTICLE RENDERER ──────────────────────────────────────────────────
//
// Bridges an MLSMPMSolver's particle SimBuffer into an SCNGeometry rendered as
// hardware points. Same zero-copy contract as DynamicMesh: SceneKit reads
// `positionMass.xyz` from the same MTLBuffer the G2P kernel writes into. No
// CPU snapshot, no per-frame upload.
//
// Limits & scope
// ──────────────
// • This is the MVP visualizer: a single colour, screen-space-sized points.
//   For a real fluid look we'll swap it for marching cubes against the
//   solver's density buffer — see the FluidTest scene roadmap.
// • SCNGeometryElement with `primitiveType = .point` requires an explicit
//   index buffer (one UInt32 per particle); SceneKit doesn't draw a
//   no-index point list. Cheap — built once at init, never touched again.
// • The drawn point count is locked to `maxParticles` at init. The point
//   element is rebuilt only when the solver's `particleBuffer.count` drops
//   below `maxParticles` to avoid drawing garbage at the buffer tail.
//
// LIFECYCLE
// ──────────
// • Init once after building the solver; assign `renderer.node` somewhere in
//   the scene graph. Call `update()` from the per-frame tick before any
//   particles are added / removed.

@MainActor
public final class FluidParticleRenderer {

    /// The SCNNode you should add to your scene. Hosts the geometry whose
    /// vertex buffer is shared with the solver's particle buffer.
    public let node: SCNNode

    /// The MLS-MPM solver this renderer is reading from.
    public let solver: MLSMPMSolver

    private let geometryElement: SCNGeometryElement
    private let indexBuffer: MTLBuffer
    private let maxParticles: Int
    private var lastDrawnCount: Int = -1

    // ── Init ─────────────────────────────────────────────────────────────────

    public init?(solver: MLSMPMSolver,
                 pointSize: CGFloat = 6.0,
                 colour: SIMD3<Float> = SIMD3(0.45, 0.7, 1.0)) {
        let device = solver.device
        let cap = solver.particleBuffer.capacity

        // One UInt32 per particle — simple identity index list.
        let idxStride = MemoryLayout<UInt32>.stride
        guard let idxBuf = device.makeBuffer(length: idxStride * cap,
                                             options: .storageModeShared)
        else { return nil }
        idxBuf.label = "Fluid.particleIndices"
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: cap)
        for i in 0..<cap { idxPtr[i] = UInt32(i) }

        // Position source: aliased onto the solver's particle buffer.
        // MLSParticle starts with `positionMass` (SIMD4<Float>) at offset 0,
        // so the .xyz bytes 0..<12 are the world position.
        let particleStride = MemoryLayout<MLSParticle>.stride
        let posSource = SCNGeometrySource(
            buffer: solver.particleBuffer.buffer,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: cap,
            dataOffset: 0,
            dataStride: particleStride
        )

        // Initial element drawn over the full capacity. Will be replaced in
        // update() if `solver.particleBuffer.count` is lower.
        let element = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .point,
            primitiveCount: cap,
            bytesPerIndex: idxStride
        )
        element.minimumPointScreenSpaceRadius = pointSize * 0.6
        element.maximumPointScreenSpaceRadius = pointSize * 1.6
        element.pointSize = pointSize

        let geom = SCNGeometry(sources: [posSource], elements: [element])

        // Constant-lit material so we don't depend on lights for visibility.
        // Emission is set so the points read like little water droplets even
        // when no environment IBL is set on the scene.
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.diffuse.contents = PlatformColor(deviceRed: CGFloat(colour.x),
                                       green:     CGFloat(colour.y),
                                       blue:      CGFloat(colour.z),
                                       alpha:     1)
        mat.emission.contents = PlatformColor(deviceRed: CGFloat(colour.x),
                                        green:     CGFloat(colour.y),
                                        blue:      CGFloat(colour.z),
                                        alpha:     1)
        mat.writesToDepthBuffer = true
        mat.readsFromDepthBuffer = true
        geom.materials = [mat]

        let n = SCNNode(geometry: geom)
        n.name = "FluidParticles"

        self.solver          = solver
        self.geometryElement = element
        self.indexBuffer     = idxBuf
        self.maxParticles    = cap
        self.node            = n
    }

    // ── Per-frame update ─────────────────────────────────────────────────────

    /// Sync the drawn point count with `solver.particleBuffer.count`. Cheap
    /// — only does anything the frame the count changes. Position data
    /// flows in automatically because the geometry source aliases the
    /// solver's buffer.
    public func update() {
        let count = solver.particleBuffer.count
        if count == lastDrawnCount { return }

        let element = SCNGeometryElement(
            buffer: indexBuffer,
            primitiveType: .point,
            primitiveCount: max(0, count),
            bytesPerIndex: MemoryLayout<UInt32>.stride
        )
        element.minimumPointScreenSpaceRadius = geometryElement.minimumPointScreenSpaceRadius
        element.maximumPointScreenSpaceRadius = geometryElement.maximumPointScreenSpaceRadius
        element.pointSize = geometryElement.pointSize

        if let geom = node.geometry {
            // SCNGeometry.elements is read-only; rebuild via a new geometry
            // pointing at the same sources + the updated element.
            let newGeom = SCNGeometry(sources: geom.sources, elements: [element])
            newGeom.materials = geom.materials
            node.geometry = newGeom
        }

        lastDrawnCount = count
    }
}
