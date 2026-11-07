import Metal

// ── One substep, one compute encoder ─────────────────────────────────────────
//
// `encode(to:dt:)`'s default path opens an encoder for every sub-pass: integrate, a blit for λ, each colour group of
// each constraint iteration, each SDF pass, the floor — 4 + 6 × `constraintIterations` encoders a substep. Their GPU
// work is a few microseconds; the encoders are the cost, all of it on the thread that encodes (Dad Hat Factory's
// princess veil: ~230 a frame, ~0.9 ms of an iPhone 15 Pro's main thread). Here the same kernels run in the same
// order in ONE compute encoder, a buffer barrier between each, after one blit that zeroes λ. Within an encoder a
// barrier orders the dispatches the way separate encoders did, so the result is the same to the bit (Dad Hat
// Factory's `PBDBatchedTests` steps a rope both ways and compares every particle).

extension PBDSolver {
    /// One substep as a λ-reset blit plus one compute encoder (plus a snapshot blit and a second compute encoder when
    /// `selfCollideEnabled`, whose Jacobi read needs a copy taken between two dispatches).
    ///
    /// `before` / `after` encode the caller's own passes into the substep's compute encoder, ahead of the integration
    /// and after the last floor clamp, each fenced by a buffer barrier — a pin, a frame kick, a collision against the
    /// caller's own surface — so they too cost no encoder of their own.
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
        guard var enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = label
        if let before { before(enc); enc.memoryBarrier(scope: .buffers) }
        dispatchIntegrate(into: enc)
        enc.memoryBarrier(scope: .buffers)

        let l = constraintLayout
        let groups = [(l.primaryEvenStart, l.primaryEvenCount), (l.primaryOddStart, l.primaryOddCount),
                      (l.bendAStart, l.bendACount), (l.bendBStart, l.bendBCount), (l.bendCStart, l.bendCCount),
                      (l.longRangeStart, l.longRangeCount)]
        let collides = (colliders?.count ?? 0) > 0
        for _ in 0..<constraintIterations {
            for (start, count) in groups where count > 0 {
                dispatchConstraintSubpass(into: enc, start: start, count: count)
                enc.memoryBarrier(scope: .buffers)
            }
            if collides { dispatchSDFCollide(into: enc); enc.memoryBarrier(scope: .buffers) }
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
            guard let next = commandBuffer.makeComputeCommandEncoder() else { return }
            enc = next
            enc.label = label
            dispatchSelfCollide(into: enc, snapshot: snapshot)
            enc.memoryBarrier(scope: .buffers)
        }

        dispatchFloor(into: enc)
        enc.memoryBarrier(scope: .buffers)
        if collides {
            dispatchSDFCollide(into: enc); enc.memoryBarrier(scope: .buffers)
            dispatchFloor(into: enc); enc.memoryBarrier(scope: .buffers)
        }
        after?(enc)
        enc.endEncoding()
    }
}
