import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Phase 5 / A1 — dynamic feedback-driven residency. Run once per frame BEFORE
    /// the cache update (so the update relights this frame's resident set) and AFTER
    /// the incremental re-frame (so a promoted card's dirty flag isn't clobbered by
    /// the rigid-mover pass). Reads the PREVIOUS frame's `cardRequested` working set
    /// (after waiting for the in-flight frame), promotes sampled-but-non-resident cards into
    /// slots freed by unsampled residents, and rewrites `texelCard` / `cardRect` /
    /// `dirty` in place. No kernel change — A0's zero-rect fallback already covers a
    /// card the instant it's demoted. Logs the structural success metric
    /// (working-set cards already resident / total) for verification.
    func runStreamingResidency() {
        guard surfaceCacheStreaming, surfCardBudget > 0,
              scs.surfResidentCardCount < scs.surfCardCount,
              !scs.surfSlotRect.isEmpty,
              scs.surfCardSlot.count == scs.surfCardCount,
              let reqBuf = scs.surfCardRequestedBuffer,
              let rectBuf = scs.surfCardRectBuffer,
              let texelBuf = scs.surfTexelCardBuffer,
              let dirtyBuf = scs.surfCardDirtyBuffer else { return }
        awaitInFlightFrameBeforeCPUCacheWrite()   // reads cardRequested, rewrites texelCard/cardRect/dirty
        let req = reqBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
        // A2 — refresh per-card heat from this frame's working set (sampled → full,
        // else cool by one). Done before demotability so a just-sampled card is hot.
        let heatOn = scs.surfCardHeat.count == scs.surfCardCount
        if heatOn {
            for c in 0..<scs.surfCardCount {
                if req[c] != 0 { scs.surfCardHeat[c] = scs.surfStreamHeatFull }
                else if scs.surfCardHeat[c] > 0 { scs.surfCardHeat[c] -= 1 }
            }
        }
        // Working set (requested last frame) + how much of it was already resident
        // (the structural metric: → 100% as residency converges to the working set).
        var reqTotal = 0, reqResident = 0
        var promote: [Int] = []
        for c in 0..<scs.surfCardCount where req[c] != 0 {
            reqTotal += 1
            if scs.surfCardSlot[c] >= 0 { reqResident += 1 } else { promote.append(c) }
        }
        defer { memset(reqBuf.contents(), 0, scs.surfCardCount * MemoryLayout<UInt32>.stride) }
        guard !promote.isEmpty else {
            if reqTotal > 0 {
                Self.recordSurfCacheStats("stream: resident=\(scs.surfResidentCardCount) workingSet=\(reqTotal) residentHit=\(reqResident) promoted=0")
            }
            return
        }
        // Slots free to reassign: empty, OR (A2) occupant gone cold (heat 0). With
        // hysteresis off, falls back to "unsampled this frame" — the A1 behaviour.
        var demotable: [Int] = []
        for s in 0..<scs.surfSlotCard.count {
            let occ = scs.surfSlotCard[s]
            let cold = heatOn ? (occ >= 0 && scs.surfCardHeat[occ] <= 0) : (occ >= 0 && req[occ] == 0)
            if occ < 0 || cold { demotable.append(s) }
        }
        // A4 — importance order: under contention (promote.count > demotable.count or
        // the swap cap), the most-important requested cards take the least-important
        // freed slots, so the resident set converges to the NEAR working set.
        if surfaceCacheLOD, scs.surfCardCentroid.count == scs.surfCardCount {
            let cam = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self).cameraWorldPos
            let d0sq: Float = 12 * 12
            func imp(_ c: Int) -> Float {
                let d = scs.surfCardCentroid[c] - cam
                return 1.0 / (1.0 + simd_length_squared(d) / d0sq)
            }
            promote.sort { imp($0) > imp($1) }                 // nearest requested first
            demotable.sort {                                   // farthest / empty slot first
                let a = scs.surfSlotCard[$0], b = scs.surfSlotCard[$1]
                return (a >= 0 ? imp(a) : -1) < (b >= 0 ? imp(b) : -1)
            }
        }
        let rects = rectBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: scs.surfCardCount)
        let texel = texelBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfAtlasW * scs.surfAtlasH)
        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
        let nSwap = min(promote.count, demotable.count, scs.surfMaxSwapsPerFrame)
        for i in 0..<nSwap {
            let slot = demotable[i]
            let oldC = scs.surfSlotCard[slot]
            let newC = promote[i]
            if oldC >= 0 { scs.surfCardSlot[oldC] = -1; rects[oldC] = .zero      // demote → fallback
                if heatOn { scs.surfCardHeat[oldC] = 0 } }
            scs.surfSlotCard[slot] = newC
            scs.surfCardSlot[newC] = slot
            if heatOn { scs.surfCardHeat[newC] = scs.surfStreamHeatFull }           // promoted ⇒ hot
            let r = scs.surfSlotRect[slot]
            rects[newC] = r
            // Repoint the slot's atlas texels at the new occupant.
            let x0 = Int(r.x), y0 = Int(r.y), w = Int(r.z), h = Int(r.w)
            for yy in y0..<(y0 + h) {
                let rowBase = yy * scs.surfAtlasW
                for xx in x0..<(x0 + w) { texel[rowBase + xx] = UInt32(newC) }
            }
            dirty[newC] = 1   // cold slot (held the demoted card's radiance) → relight fresh
        }
        // A4 verification metric: mean camera-distance of resident cards (lower with
        // LOD on under pressure ⇒ nearer cards held).
        var distNote = ""
        if scs.surfCardCentroid.count == scs.surfCardCount {
            let cam = frameUniformBuffer.contents().load(as: IlluminatoramaFrameUniforms.self).cameraWorldPos
            var sum: Float = 0; var n = 0
            for occ in scs.surfSlotCard where occ >= 0 { sum += simd_length(scs.surfCardCentroid[occ] - cam); n += 1 }
            if n > 0 { distNote = String(format: " meanResidentDist=%.2f lod=\(surfaceCacheLOD ? 1 : 0)", sum / Float(n)) }
        }
        Self.recordSurfCacheStats("stream: resident=\(scs.surfResidentCardCount) workingSet=\(reqTotal) residentHit=\(reqResident) promoted=\(nSwap) demoted=\(nSwap)\(distNote)")
    }

    /// Build the grouped-order world-space soup + per-triangle cards + the
    /// `soupTriBase` map for the surface-cache TLAS path. Mesh object triangles
    /// are read once per mesh and transformed by each instance's model matrix in
    /// `meshGroups`/`instance` order — identical to the TLAS instance ordering,
    /// so `soupTriBase[i]` (set just before instance i's triangles are appended)
    /// + the hit's local `primitive_id` resolves to the global card index.
    func buildGroupedSurfaceSoup(total: Int) {
        scs.surfHasDeformingCards = false   // recomputed below; cleared so an early return can't leave it stale
        surfaceCacheCardlessInstances = 0
        // Cost gate: the instance-expanded triangle count, not the per-mesh count — of the meshes
        // that take cards. A mesh that opted out (`surfaceCacheCards = false`, DH-1314) adds no
        // soup triangles and no cards, so it does not count against the cap.
        let (expandedTris, excludedTris) = surfaceCacheTriangleCensus
        guard expandedTris > 0, expandedTris <= surfaceCacheMaxTriangles else {
            if expandedTris > surfaceCacheMaxTriangles {
                Self.log.notice("Surface cache (TLAS) skipped: \(expandedTris) tris > cap \(self.surfaceCacheMaxTriangles) (\(excludedTris) cardless tris not counted)")
            }
            // Leave the cache off for this scene; the kernel gates on the uniform.
            scs.surfCardCount = 0; rts.rtSoupTriBaseBuffer = nil; rts.rtSoupTriBaseCount = 0
            return
        }

        var positions: [SIMD3<Float>] = []; positions.reserveCapacity(expandedTris * 3)
        var indices: [UInt32] = [];        indices.reserveCapacity(expandedTris * 3)
        var triAlbedo: [SIMD3<Float>] = []; triAlbedo.reserveCapacity(expandedTris)
        var triNormal: [SIMD3<Float>] = []; triNormal.reserveCapacity(expandedTris)
        var triEmission: [SIMD3<Float>] = []; triEmission.reserveCapacity(expandedTris)
        var soupBase = [UInt32](repeating: 0, count: total)
        // #60 item 1 + Phase D — GPU-diff maps, accumulated in lockstep with the
        // appended soup triangles (so global triangle order matches exactly,
        // including dropped out-of-range triangles): owner instance, OBJECT-space
        // verts (rigid re-frame), vertex-index triples (deform re-frame reads the
        // LIVE position buffer through these), and the per-tri deforming flag.
        var gpuTriOwner: [UInt32] = [];     gpuTriOwner.reserveCapacity(expandedTris)
        var gpuObjVerts: [SIMD4<Float>] = []; gpuObjVerts.reserveCapacity(expandedTris * 3)
        var gpuTriIdx: [SIMD4<UInt32>] = []; gpuTriIdx.reserveCapacity(expandedTris)
        var gpuTriDeform: [Bool] = [];       gpuTriDeform.reserveCapacity(expandedTris)

        let instPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        // Phase C — deforming (DynamicMesh / GPU-fed) kinds have a `.private`
        // interleaved vertex buffer, so `objectTriangleSoup()` returns empty and
        // they'd be excluded from the cache. Source their build-pose soup from the
        // live shared buffers instead (the per-frame re-frame then keeps the cards
        // on the current surface). `surfHasDeformingCards` flips the update kernel
        // into honouring per-card dirty flags (α=1) so they re-light each frame.
        let deformingKinds = Set(fxs.gpuRepackTasks.map(\.kind))
        var soupByMesh: [ObjectIdentifier: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
        // A group that contributes NO triangles (nil mesh / empty soup) must still
        // leave its instances an empty, correctly-positioned soupBase range — else
        // they keep the init value 0, which makes the chart incremental path read
        // `[0, soupBase[i+1])` and mis-assign instance 0's cards to the empty
        // instance (last-writer-wins). Skipped instances also get mask=0 in the
        // TLAS (never hit), so this only makes soupBase monotonic — no consumer
        // reads the empty range.
        func markEmptyRange(_ group: MeshDrawGroup) {
            let base = UInt32(indices.count / 3)
            for k in 0..<group.count where group.start + k < total { soupBase[group.start + k] = base }
        }
        // DH-1314 — instances of a cardless mesh: an empty soup range here (so `soupBase` stays
        // monotonic for every CPU consumer), and the no-card sentinel in the GPU copy below.
        var cardlessInstances: [Int] = []
        for group in gbs.meshGroups {
            guard let mesh = meshes[group.kind] else { markEmptyRange(group); continue }
            if !mesh.surfaceCacheCards {
                markEmptyRange(group)
                for k in 0..<group.count where group.start + k < total { cardlessInstances.append(group.start + k) }
                continue
            }
            let isDeforming = deformingKinds.contains(group.kind)
            let soup: (positions: [SIMD3<Float>], indices: [UInt32])
            if isDeforming {
                guard let live = liveDeformingObjectSoup(kind: group.kind) else { markEmptyRange(group); continue }
                soup = live
            } else {
                let mid = ObjectIdentifier(mesh)
                soup = soupByMesh[mid] ?? {
                    let s = mesh.objectTriangleSoup(); soupByMesh[mid] = s; return s
                }()
            }
            // Both must be non-empty: a soup with indices but no positions
            // (degenerate read) would make every `positions[base + index]`
            // below reach past the block.
            guard !soup.indices.isEmpty, !soup.positions.isEmpty else { markEmptyRange(group); continue }
            if isDeforming { scs.surfHasDeformingCards = true }
            for k in 0..<group.count {
                let i = group.start + k
                guard i < total else { break }
                soupBase[i] = UInt32(indices.count / 3)
                let model = instPtr[i].modelMatrix
                let albedo = instPtr[i].albedo
                // DH-0537 (2) — a card carries its emitter's glow into the bounce. Both card
                // bakers accepted `triangleEmission` and neither call site passed it, so a lit
                // lamp shade contributed nothing to cached bounce light and a room lit by its own
                // fixtures bounced as if they were dark. Solid emission already folds intensity
                // in; a textured emitter has no mean here, exactly as `albedo` stands in for a
                // textured albedo.
                let emission = instPtr[i].emission
                let base = UInt32(positions.count)
                for p in soup.positions {
                    let w = model * SIMD4<Float>(p, 1)
                    positions.append(SIMD3(w.x, w.y, w.z))
                }
                // `soup.indices` must all be < soup.positions.count, or
                // `positions[Int(i0)]` below reads out of bounds and SIGTRAPs
                // the whole render loop. `from(scnGeometry:)` sanitises
                // registered meshes, but the live deforming path
                // (`liveDeformingObjectSoup`) reads positions and indices from
                // independent buffers, so guard here too rather than trust it.
                let vcount = UInt32(soup.positions.count)
                // One-time alarm if a mesh's indices reach past its own vertex
                // block — names the offending mesh in the unified log so a
                // recurrence (e.g. a new scene's hand-built soup) is diagnosable
                // without a debugger. The per-triangle guard below keeps it safe
                // regardless.
                if k == 0, let mx = soup.indices.max(), mx >= vcount {
                    Self.log.error("Surface soup: mesh \(String(describing: group.kind)) (deforming=\(isDeforming)) has index \(mx) >= vertexCount \(vcount); out-of-range triangles dropped.")
                }
                var t = 0
                while t + 2 < soup.indices.count {
                    let l0 = soup.indices[t], l1 = soup.indices[t + 1], l2 = soup.indices[t + 2]
                    t += 3
                    guard l0 < vcount, l1 < vcount, l2 < vcount else { continue }
                    let i0 = base + l0, i1 = base + l1, i2 = base + l2
                    indices.append(i0); indices.append(i1); indices.append(i2)
                    let w0 = positions[Int(i0)], w1 = positions[Int(i1)], w2 = positions[Int(i2)]
                    var n = simd_cross(w1 - w0, w2 - w0)
                    let len = simd_length(n)
                    n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
                    triNormal.append(n); triAlbedo.append(albedo); triEmission.append(emission)
                    // Phase D parity: with the freeze override the CPU path never
                    // re-frames a deforming kind (its object soup is empty there),
                    // so mark its triangles sentinel — never re-framed, dirty=0.
                    if isDeforming && scs.noDeformCardsOverride {
                        gpuTriOwner.append(UInt32.max)
                    } else {
                        gpuTriOwner.append(UInt32(i))
                    }
                    gpuTriDeform.append(isDeforming && !scs.noDeformCardsOverride)
                    gpuTriIdx.append(SIMD4<UInt32>(l0, l1, l2, 0))
                    gpuObjVerts.append(SIMD4(soup.positions[Int(l0)], 0))
                    gpuObjVerts.append(SIMD4(soup.positions[Int(l1)], 0))
                    gpuObjVerts.append(SIMD4(soup.positions[Int(l2)], 0))
                }
            }
        }
        guard !indices.isEmpty else { scs.surfCardCount = 0; return }

        // Build the primitive AS (cache-update trace target) + surface-cache cards.
        setRTGeometry(positions: positions, indices: indices,
                      triangleAlbedo: triAlbedo, triangleNormal: triNormal)
        // P2 — coplanar-chart merging is OPT-IN (`VIZ_SURFCACHE_CHARTS=1`).
        // Default is the P3 per-triangle path: on low/medium-tessellation geometry
        // (e.g. cityStreet — quad walls) P3's 2-triangles-per-tile packing already
        // hits ~2:1 tri:tile, which BEATS chart merging's ~1.7:1 there (measured:
        // charts 12.4k cards / 10.5 MB atlas vs per-tri 10.6k / 3.1 MB). Chart
        // merging wins only on HEAVILY-tessellated coplanar surfaces (many
        // triangles per flat face → N:1), and it gives larger/stabler cards for
        // incremental invalidation. See docs/illuminatorama README phase 4.38c.
        // Per-instance mesh kind (used by both paths to relate instances ↔ soup).
        var instanceKinds = [MeshKind](repeating: gbs.meshGroups.first?.kind ?? .box, count: total)
        for group in gbs.meshGroups {
            for k in 0..<group.count where group.start + k < total { instanceKinds[group.start + k] = group.kind }
        }
        let useCharts = Self.surfCacheCharts
        if useCharts {
            let g = Self.makeCoplanarChartCards(
                positions: positions, indices: indices,
                triangleAlbedo: triAlbedo, triangleNormal: triNormal,
                triangleEmission: triEmission)
            // Adaptive per-chart tile size: a chart's tile scales with its world
            // extent, shelf-packed, so big walls get more texels than small facets
            // and the atlas stays efficient across the chart-size spread.
            setSurfaceCacheCards(cards: g.cards, triCard: g.triCard,
                                 triUVa: g.triUVa, triUVc: g.triUVc,
                                 adaptiveTiles: true, texelsPerMeter: 5.0)
            // Chart-path incremental invalidation (#60 task 1). A chart is built in
            // WORLD space at the bake pose; record each card's frame back in its
            // owning instance's OBJECT space (M_bake⁻¹ · worldFrame) so the per-frame
            // re-frame is one affine transform (`M_now · objectFrame`) — exact for
            // rigid/affine motion, UVs untouched. Charts never cross an instance
            // (separate vertex blocks ⇒ no cross-instance edge adjacency), so each
            // instance owns a CONTIGUOUS card range. Deforming kinds are excluded
            // (their per-vertex shape change invalidates a planar chart frame; the
            // chart path is rigid-mover only — documented in the design note).
            if scs.surfCardCount == g.cards.count, !g.triCard.isEmpty {
                var cardInstance = [Int32](repeating: -1, count: g.cards.count)
                for i in 0..<total {
                    if deformingKinds.contains(instanceKinds[i]) { continue }
                    let firstTri = Int(soupBase[i])
                    let nextTri = (i + 1 < total) ? Int(soupBase[i + 1]) : g.triCard.count
                    for gt in firstTri..<nextTri where gt >= 0 && gt < g.triCard.count {
                        cardInstance[Int(g.triCard[gt])] = Int32(i)
                    }
                }
                // Per-instance bake-pose inverse — ONLY for invertible matrices. A
                // degenerate model matrix (e.g. a flattened/zero-scale node) would
                // make simd_inverse return NaN/inf and poison every per-frame
                // re-frame; skip it so that instance's cards stay at their baked
                // world frame and are simply never re-framed (they won't track if it
                // moves, but they won't corrupt — the acceptable degenerate fallback).
                var invByInstance = [Int32: simd_float4x4]()
                for i in 0..<total {
                    let m = instPtr[i].modelMatrix
                    if abs(simd_determinant(m)) > 1e-12 { invByInstance[Int32(i)] = simd_inverse(m) }
                }
                var objFrames = [SurfCardFrame](repeating: SurfCardFrame(), count: g.cards.count)
                var instCardLo = [Int32](repeating: -1, count: total)
                var instCardHi = [Int32](repeating: -1, count: total)
                // #60 item 1 — GPU-diff owner per card (−1 = static / degenerate
                // bake pose, never re-framed; mirrors the CPU skip below).
                var cardOwnerGPU = [Int32](repeating: -1, count: g.cards.count)
                for c in 0..<g.cards.count {
                    let card = g.cards[c]
                    let wO = SIMD3<Float>(card.origin.x, card.origin.y, card.origin.z)
                    let wU = SIMD3<Float>(card.uAxis.x, card.uAxis.y, card.uAxis.z)
                    let wV = SIMD3<Float>(card.vAxis.x, card.vAxis.y, card.vAxis.z)
                    let wN = SIMD3<Float>(card.normal.x, card.normal.y, card.normal.z)
                    let inst = cardInstance[c]
                    if inst >= 0, let invM = invByInstance[inst] {
                        let ii = Int(inst)
                        let o = invM * SIMD4<Float>(wO, 1)
                        let u = invM * SIMD4<Float>(wU, 0)
                        let v = invM * SIMD4<Float>(wV, 0)
                        let n = invM * SIMD4<Float>(wN, 0)
                        objFrames[c] = SurfCardFrame(origin: SIMD3(o.x, o.y, o.z), uAxis: SIMD3(u.x, u.y, u.z),
                                                     vAxis: SIMD3(v.x, v.y, v.z), normal: SIMD3(n.x, n.y, n.z))
                        if instCardLo[ii] < 0 || Int32(c) < instCardLo[ii] { instCardLo[ii] = Int32(c) }
                        if Int32(c + 1) > instCardHi[ii] { instCardHi[ii] = Int32(c + 1) }
                        cardOwnerGPU[c] = inst
                    } else {
                        objFrames[c] = SurfCardFrame(origin: wO, uAxis: wU, vAxis: wV, normal: wN)
                    }
                }
                scs.surfChartIncremental = ChartIncremental(objectFrame: objFrames,
                                                        instCardLo: instCardLo, instCardHi: instCardHi)
                scs.surfChartMovedScratch = [Bool](repeating: false, count: instCardLo.count)
                scs.surfTriCardCPU  = g.triCard      // guard parity with the per-triangle path
                scs.surfSoupBaseCPU = soupBase
                scs.surfIncrementalReady = true
                buildGPUDiffChartBuffers(objFrames: objFrames, cardOwner: cardOwnerGPU)
            }
        } else {
            let g = Self.makePerTriangleSurfaceCards(
                positions: positions, indices: indices,
                triangleAlbedo: triAlbedo, triangleNormal: triNormal,
                triangleEmission: triEmission,
                blockStarts: soupBase.map { Int($0) })
            setSurfaceCacheCards(cards: g.cards, triCard: g.triCard,
                                 triUVa: g.triUVa, triUVc: g.triUVc,
                                 tileSize: surfaceCachePerTriangleTileSize)
            // #60 item 6 — invariance guard (DEBUG). The fold bakes ONE per-mesh UV
            // table into the shared BLAS; that's only correct if every instance block
            // of a mesh got the same frame-A/B layout above. Verify the first block of
            // each kind matches the canonical `perMeshCardUVs` the BLAS bakes.
            Self.assertPerMeshUVInvariance(triUVa: g.triUVa, triUVc: g.triUVc,
                                           soupBase: soupBase, instanceKinds: instanceKinds,
                                           deformingKinds: deformingKinds, total: total)
            // Capture the CPU maps the incremental-invalidation prototype needs to
            // re-frame a moved instance's cards in place (per-triangle path only).
            // Frame-B membership is read back from the packed reference UVs:
            // frame A's uvA = (0,0,…); frame B's uvA = (1,1,…).
            scs.surfTriCardCPU   = g.triCard
            scs.surfTriIsFrameB  = g.triUVa.map { $0.x > 0.5 }
            scs.surfSoupBaseCPU  = soupBase
            scs.surfObjectSoupByKind.removeAll(keepingCapacity: true)
            for group in gbs.meshGroups where scs.surfObjectSoupByKind[group.kind] == nil {
                if let m = meshes[group.kind] { scs.surfObjectSoupByKind[group.kind] = m.objectTriangleSoup() }
            }
            scs.surfInstanceKind = instanceKinds
            scs.surfChartIncremental = nil   // per-triangle path owns the re-frame
            scs.surfIncrementalReady = true
            buildGPUDiffPerTriBuffers(triOwner: gpuTriOwner, objVerts: gpuObjVerts,
                                      isFrameB: scs.surfTriIsFrameB,
                                      triIdx: gpuTriIdx, deforming: gpuTriDeform)
            // Phase D — one deform dispatch per deforming instance. soupBase is
            // monotonic and each instance's appended triangles exactly fill
            // [soupBase[i], soupBase[i+1]), so base/count come straight from it.
            if scs.surfGPUDiffTriCount > 0, !scs.noDeformCardsOverride {
                for group in gbs.meshGroups where deformingKinds.contains(group.kind) {
                    for k in 0..<group.count {
                        let i = group.start + k
                        guard i < total else { break }
                        let base = Int(soupBase[i])
                        let next = (i + 1 < total) ? Int(soupBase[i + 1]) : scs.surfGPUDiffTriCount
                        guard next > base else { continue }
                        scs.surfGPUDiffDeformDispatches.append(SurfDeformDispatch(
                            kind: group.kind, instanceIndex: i,
                            triBase: base, triCount: next - base))
                    }
                }
            }
        }
        // The GPU copy: a cardless instance's slot holds the sentinel, so a TLAS hit on it resolves
        // to "no card" in every consumer instead of to its neighbour's triangles.
        var gpuSoupBase = soupBase
        for i in cardlessInstances { gpuSoupBase[i] = Self.surfaceCacheNoCardSoupBase }
        surfaceCacheCardlessInstances = cardlessInstances.count
        rts.rtSoupTriBaseBuffer = device.makeBuffer(
            bytes: gpuSoupBase, length: MemoryLayout<UInt32>.stride * gpuSoupBase.count,
            options: .storageModeShared)
        rts.rtSoupTriBaseBuffer?.label = "Illuminatorama.rt.soupTriBase"
        rts.rtSoupTriBaseCount = soupBase.count
    }
}
