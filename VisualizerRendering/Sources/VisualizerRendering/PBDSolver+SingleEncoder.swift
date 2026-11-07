import Metal

// ── One substep, one compute encoder ─────────────────────────────────────────
//
// `encode(to:dt:)`'s default path opens an encoder for every sub-pass: integrate, a blit for λ, each colour group of
// each constraint iteration, each SDF pass, the floor — 4 + 6 × `constraintIterations` encoders a substep. Their GPU
// work is a few microseconds; the encoders are the cost, all of it on the thread that encodes (Dad Hat Factory's
// princess veil: ~230 a frame, ~0.9 ms of an iPhone 15 Pro's main thread). Here the same kernels run in the same
// order in ONE serial compute encoder, after one blit that zeroes λ. A serial encoder runs its dispatches in order and
// tracks the buffers' hazards itself, so the result is the same to the bit (`PBDSingleEncoderTests`). No explicit
// `memoryBarrier`s: between every dispatch they made Dad Hat Factory's veil ~6× the GPU time on an iPhone 15 Pro
// (135 ms of GPU a second, from 21).

extension PBDSolver {
    /// One substep as a λ-reset blit plus one compute encoder (plus a snapshot blit and a second compute encoder when
    /// `selfCollideEnabled`, whose Jacobi read needs a copy taken between two dispatches).
    ///
    /// `before` / `after` encode the caller's own passes into the substep's (serial) compute encoder, ahead of the
    /// integration and after the last floor clamp — a pin, a frame kick, a collision against the caller's own surface —
    /// so they too cost no encoder of their own.
    ///
    /// The uniforms are one shared buffer the GPU reads when it runs, written here at encode time, so every substep in a
    /// command buffer sees the last one written: use one `dt` (and one collider count) across a command buffer's
    /// substeps, as with the default path.
    public func encodeSingleEncoder(to commandBuffer: MTLCommandBuffer, dt: Float, label: String = "PBD.substep",
                                    before: ((MTLComputeCommandEncoder) -> Void)? = nil,
                                    after: ((MTLComputeCommandEncoder) -> Void)? = nil) {
        writeIntegrateUniforms(dt: dt)
        if constraintBuffer.count > 0, let blit = commandBuffer.makeBlitCommandEncoder() {
            #if DEBUG
            blit.label = label + ".lambda"
            #endif
            dispatchLambdaReset(into: blit)
            blit.endEncoding()
        }
        guard var enc = commandBuffer.makeComputeCommandEncoder(dispatchType: .serial) else { return }
        enc.label = label
        before?(enc)
        dispatchIntegrate(into: enc)

        let l = constraintLayout
        let groups = [(l.primaryEvenStart, l.primaryEvenCount), (l.primaryOddStart, l.primaryOddCount),
                      (l.bendAStart, l.bendACount), (l.bendBStart, l.bendBCount), (l.bendCStart, l.bendCCount),
                      (l.longRangeStart, l.longRangeCount)]
        let collides = (colliders?.count ?? 0) > 0
        for _ in 0..<constraintIterations {
            for (start, count) in groups where count > 0 {
                dispatchConstraintSubpass(into: enc, start: start, count: count)
            }
            if collides { dispatchSDFCollide(into: enc) }
        }

        // Self-collision, once a substep after the loop (as `encodeConstraintsAndContacts`): its kernel reads a frozen
        // snapshot of the particles, copied by a blit, so the compute encoder ends and a second one picks up after it.
        if selfCollideEnabled, selfCollideRadius > 0, particleBuffer.count > 0, let snapshot = selfCollideSnapshotBuffer() {
            enc.endEncoding()
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                #if DEBUG
                blit.label = label + ".selfCollideSnapshot"
                #endif
                blit.copy(from: particleBuffer.buffer, sourceOffset: 0, to: snapshot, destinationOffset: 0,
                          size: particleBuffer.count * MemoryLayout<PBDParticle>.stride)
                blit.endEncoding()
            }
            guard let next = commandBuffer.makeComputeCommandEncoder(dispatchType: .serial) else { return }
            enc = next
            enc.label = label
            dispatchSelfCollide(into: enc, snapshot: snapshot)
        }

        dispatchFloor(into: enc)
        if collides { dispatchSDFCollide(into: enc); dispatchFloor(into: enc) }
        after?(enc)
        enc.endEncoding()
    }
}
