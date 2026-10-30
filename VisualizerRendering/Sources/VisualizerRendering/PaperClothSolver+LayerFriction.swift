import Metal

// Coulomb friction between cloth layers (`paperLayerFriction` in PaperCloth.metal): a node resting on
// another layer holds while its slip along it is under μs × what presses it into it. This file is the
// pass's own state — its pipeline, the per-node contact record the self-collision leaves it, and the
// uniforms mirror — so the solver carries one optional reference instead of three more stored buffers.

/// The layer-friction pass's GPU state. Built the first time `PaperClothSolver.layerFrictionStatic` > 0.
@MainActor
final class PaperLayerFriction {
    let pipeline: MTLComputePipelineState
    /// 2 × float4 per node: (direction to this node from the other layer), (that layer's per-substep motion, 1 = in contact).
    let contact: MTLBuffer
    let uniforms: MTLBuffer

    init?(engine: SimEngine, maxParticles: Int) {
        guard let pipe = engine.pipelineCache.pipelineState(name: "paperLayerFriction", device: engine.device),
              let c = engine.device.makeBuffer(length: max(1, maxParticles) * 2 * MemoryLayout<SIMD4<Float>>.stride,
                                               options: .storageModePrivate),
              let u = engine.device.makeBuffer(length: MemoryLayout<PaperLayerFrictionUniforms>.stride,
                                               options: .storageModeShared)
        else { return nil }
        c.label = "PaperCloth.layerContact"
        u.label = "PaperCloth.layerFrictionUniforms"
        pipeline = pipe; contact = c; uniforms = u
    }

    /// Forget last frame's contacts; the self-collision passes that follow record this frame's.
    func encodeClear(_ cb: MTLCommandBuffer, particleCount: Int) {
        guard particleCount > 0, let blit = cb.makeBlitCommandEncoder() else { return }
        blit.label = "PaperCloth.layerContactClear"
        blit.fill(buffer: contact, range: 0..<(particleCount * 2 * MemoryLayout<SIMD4<Float>>.stride), value: 0)
        blit.endEncoding()
    }

    func write(particleCount: Int, muStatic: Float, muKinetic: Float,
               gridW: Int, gridH: Int, verticesPerSheet: Int) {
        uniforms.contents().bindMemory(to: PaperLayerFrictionUniforms.self, capacity: 1).pointee =
            PaperLayerFrictionUniforms(particleCount: UInt32(particleCount), muStatic: muStatic,
                                       muKinetic: muKinetic, gridW: UInt32(gridW),
                                       gridH: UInt32(gridH), verticesPerSheet: UInt32(verticesPerSheet))
    }
}

/// Per-sheet layer facts for `paperSelfCollide` (`sheetLayers`): which PIECE a sheet belongs to (friction
/// only acts between pieces) and its own half-thickness (two sheets are held apart by the sum of theirs).
@MainActor
final class PaperSheetLayers {
    let buffer: MTLBuffer
    private let count: Int
    init?(device: MTLDevice, maxSheets: Int, defaultHalf: Float) {
        guard let b = device.makeBuffer(length: max(1, maxSheets) * MemoryLayout<SIMD2<UInt32>>.stride,
                                        options: .storageModeShared) else { return nil }
        b.label = "PaperCloth.sheetLayers"
        buffer = b; count = max(1, maxSheets)
        let p = b.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)
        for s in 0 ..< count { p[s] = SIMD2(UInt32(s), defaultHalf.bitPattern) }   // own piece, the shared band
    }
    func set(_ sheet: Int, group: Int, half: Float) {
        guard sheet >= 0, sheet < count else { return }
        buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: count)[sheet] = SIMD2(UInt32(group), half.bitPattern)
    }
}

// Keep field order IDENTICAL to `struct PaperLayerFrictionUniforms` in PaperCloth.metal.
struct PaperLayerFrictionUniforms {
    var particleCount: UInt32
    var muStatic: Float
    var muKinetic: Float
    var gridW: UInt32
    var gridH: UInt32
    var verticesPerSheet: UInt32
}
