import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    // ── Surface-cache cards ───────────────────────────────────────────
    //
    // Supply the planar cards (one per floor / wall / ceiling / window jamb)
    // and the per-triangle → card-UV mapping the surface cache needs. `triCard`
    // / `triUVa` / `triUVc` are indexed by the SAME `primitive_id` the RT AS
    // uses (so a ray hit maps to a card texel). Allocates the ping-pong
    // radiance atlas sized to fit `cards.count` tiles. Call once after
    // `setRTGeometry`; no-op without RT support.
    //
    // `triUVa[i] = (uvA.xy, uvB.xy)`, `triUVc[i] = (uvC.xy, 0, 0)` for triangle
    // i's three vertices in its card's [0,1]² parameterisation.
    /// Phase 4.38 — surface-cache generalisation (#17 item 3, P1). Synthesise a
    /// per-triangle "micro-card" parameterisation for an ARBITRARY triangle soup,
    /// so any extracted mesh — not just hand-authored planar quads like the room
    /// — can drive the surface cache. The insight: a single triangle IS planar,
    /// so it fits `SurfCard` exactly — `origin = v0`, `uAxis = v1−v0`,
    /// `vAxis = v2−v0`, and the reference UVs `A(0,0) B(1,0) C(0,1)` make the
    /// update kernel's `origin + u·uAxis + v·vAxis` reconstruct points in the
    /// triangle's plane (the triangle occupies the `u+v ≤ 1` half of its square
    /// tile). One card per triangle; `triCard[t] = t`. No new GPU code — the
    /// existing atlas / ping-pong / EMA / `sampleSurfCache` machinery is reused
    /// verbatim. Atlas size scales with triangle count (bounded by the live-RT
    /// triangle cap). **P3 (done):** two triangles share each square tile (frame
    /// A in the `u+v ≤ 1` lower-left half, frame B in the `u+v > 1` upper-right
    /// half) — since a single triangle already only used ~half its tile's texels,
    /// packing fills the wasted half with a second triangle for ~2× atlas/relight
    /// efficiency at the SAME per-triangle resolution. Coplanar-chart merge (P2)
    /// for better texel coherence remains the follow-up.
    ///
    /// Pair with `setRTGeometry(positions:indices:triangleAlbedo:triangleNormal:)`
    /// using the SAME `positions`/`indices`, then feed the result straight to
    /// `setSurfaceCacheCards`.
    ///
    /// `blockStarts` — triangle indices where each instance's block begins in the
    /// global soup. The 2-tri pairing NEVER crosses one, so every instance block
    /// of a given mesh produces an IDENTICAL frame-A/B layout. That mesh-invariance
    /// is what lets the per-triangle card UVs be baked into the shared BLAS
    /// `primitive_data` (#60 item 6) — without it, an odd-triangle-count mesh would
    /// flip A/B parity on each subsequent instance (or a pair would straddle two
    /// instances), and a single BLAS-baked UV would sample the wrong tile half for
    /// some instances. Empty `blockStarts` keeps the legacy global pairing.
    public static func makePerTriangleSurfaceCards(
        positions: [SIMD3<Float>],
        indices: [UInt32],
        triangleAlbedo: [SIMD3<Float>],
        triangleNormal: [SIMD3<Float>],
        triangleEmission: [SIMD3<Float>]? = nil,
        blockStarts: [Int] = []
    ) -> (cards: [SurfCard], triCard: [UInt32],
          triUVa: [SIMD4<Float>], triUVc: [SIMD4<Float>]) {
        let triCount = indices.count / 3
        // Instance-block boundaries the pairing must not cross (interior only).
        let boundary = Set(blockStarts.filter { $0 > 0 && $0 < triCount })
        var cards: [SurfCard] = [];   cards.reserveCapacity((triCount + 1) / 2)
        // triCard / triUVa / triUVc are indexed by PRIMITIVE (triangle), so they
        // stay length `triCount`. Pre-size with placeholders so a `continue`d
        // (degenerate-index) triangle keeps the array aligned with primitive ids.
        var triCard = [UInt32](repeating: 0, count: triCount)
        var triUVa  = [SIMD4<Float>](repeating: .zero, count: triCount)
        var triUVc  = [SIMD4<Float>](repeating: .zero, count: triCount)

        // P3 reference UVs. Frame A (lower-left half, u+v ≤ 1): A(0,0) B(1,0) C(0,1).
        // Frame B (upper-right half, u+v ≥ 1): A(1,1) B(0,1) C(1,0). The update
        // kernel mirrors B's texels back into a lower-left frame, so a hit's
        // barycentric→UV lands in the correct half of the SHARED tile.
        let uvAa = SIMD4<Float>(0, 0, 1, 0), uvAc = SIMD4<Float>(0, 1, 0, 0)
        let uvBa = SIMD4<Float>(1, 1, 0, 1), uvBc = SIMD4<Float>(1, 0, 0, 0)

        func vertsOK(_ t: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)? {
            let i0 = Int(indices[3 * t]), i1 = Int(indices[3 * t + 1]), i2 = Int(indices[3 * t + 2])
            guard i0 < positions.count, i1 < positions.count, i2 < positions.count,
                  t < triangleAlbedo.count, t < triangleNormal.count else { return nil }
            return (positions[i0], positions[i1], positions[i2])
        }

        // Pair consecutive triangles two-per-tile. A lone trailing triangle (odd
        // count, or when its partner has degenerate indices) becomes an unpacked
        // single-frame card occupying the tile's lower-left half.
        var t = 0
        while t < triCount {
            guard let (a0, a1, a2) = vertsOK(t) else { t += 1; continue }
            let aE = triangleEmission?[t] ?? .zero
            // Look for a valid partner triangle to share this tile — but never
            // pair across an instance-block boundary (keeps the layout mesh-invariant).
            if t + 1 < triCount, !boundary.contains(t + 1), let (b0, b1, b2) = vertsOK(t + 1) {
                let bE = triangleEmission?[t + 1] ?? .zero
                let card = UInt32(cards.count)
                cards.append(SurfCard(
                    aOrigin: a0, aUAxis: a1 - a0, aVAxis: a2 - a0,
                    aNormal: triangleNormal[t], aAlbedo: triangleAlbedo[t], aEmission: aE,
                    bOrigin: b0, bUAxis: b1 - b0, bVAxis: b2 - b0,
                    bNormal: triangleNormal[t + 1], bAlbedo: triangleAlbedo[t + 1], bEmission: bE))
                triCard[t] = card;     triUVa[t] = uvAa;     triUVc[t] = uvAc
                triCard[t + 1] = card; triUVa[t + 1] = uvBa; triUVc[t + 1] = uvBc
                t += 2
            } else {
                let card = UInt32(cards.count)
                cards.append(SurfCard(
                    origin: a0, uAxis: a1 - a0, vAxis: a2 - a0,
                    normal: triangleNormal[t], albedo: triangleAlbedo[t], emission: aE))
                triCard[t] = card; triUVa[t] = uvAa; triUVc[t] = uvAc
                t += 1
            }
        }
        return (cards, triCard, triUVa, triUVc)
    }

    /// Per-triangle card-frame UVs baked into a mesh's BLAS `primitive_data`
    /// (#60 item 6). One element per BLAS primitive (== mesh-local triangle),
    /// 24 B, `SIMD2<Float>` stride 8 ↔ Metal `float2` (the safe layout — NOT the
    /// `SIMD3`/`packed_float3` trap). A TLAS hit reads its card UVs straight from
    /// the AS payload (`res.primitive_data`) instead of the per-instance global
    /// `triUVa`/`triUVc` side buffers, dropping 32 B/hit of dependent loads on the
    /// surface-cache GI/reflection path.
    struct IlluminatoramaPrimUV { var uvA: SIMD2<Float>; var uvB: SIMD2<Float>; var uvC: SIMD2<Float> }

    /// The canonical, mesh-invariant frame-A/B UV table for a mesh's local
    /// triangles — the exact pattern `makePerTriangleSurfaceCards` produces for a
    /// single instance block (pairing reset at the block boundary). Triangle 2k
    /// gets frame A, 2k+1 gets frame B; a lone trailing triangle gets frame A.
    /// Depends ONLY on the triangle count, so it's identical for every instance of
    /// the mesh — that's the property the BLAS fold relies on. (In the global pass
    /// out-of-range triangles are pre-dropped before pairing, so the BLAS primitive
    /// count == the per-instance soup triangle count == this length — the same
    /// invariant the existing `soupTriBase[iid]+primitive_id` math already assumes.)
    static func perMeshCardUVs(triangleCount: Int) -> [IlluminatoramaPrimUV] {
        // Frame A (u+v ≤ 1): A(0,0) B(1,0) C(0,1). Frame B (u+v ≥ 1): A(1,1) B(0,1) C(1,0).
        let frameA = IlluminatoramaPrimUV(uvA: SIMD2(0, 0), uvB: SIMD2(1, 0), uvC: SIMD2(0, 1))
        let frameB = IlluminatoramaPrimUV(uvA: SIMD2(1, 1), uvB: SIMD2(0, 1), uvC: SIMD2(1, 0))
        var out = [IlluminatoramaPrimUV](repeating: frameA, count: max(0, triangleCount))
        var t = 0
        while t < triangleCount {
            out[t] = frameA
            if t + 1 < triangleCount { out[t + 1] = frameB; t += 2 } else { t += 1 }
        }
        return out
    }

    /// DEBUG guard for the #60-item-6 fold: the BLAS bakes `perMeshCardUVs`, so the
    /// global pass's per-instance UV layout MUST match it for every instance (not
    /// just the first). Spot-checks the first instance block of each mesh kind and
    /// logs the first mismatch. A failure means the packing isn't mesh-invariant —
    /// the fold would sample the wrong tile half — so it must never ship silent.
    static func assertPerMeshUVInvariance(triUVa: [SIMD4<Float>], triUVc: [SIMD4<Float>],
                                          soupBase: [UInt32], instanceKinds: [MeshKind],
                                          deformingKinds: Set<MeshKind>, total: Int) {
        #if DEBUG
        var checked = Set<MeshKind>()
        for i in 0..<total where i < instanceKinds.count {
            let kind = instanceKinds[i]
            if checked.contains(kind) { continue }
            // Deforming kinds aren't folded (their BLAS bakes no primitive data), so
            // a parity shift from their un-sanitised soup is harmless — skip them.
            if deformingKinds.contains(kind) { continue }
            let lo = Int(soupBase[i])
            let hi = (i + 1 < total) ? Int(soupBase[i + 1]) : triUVa.count
            let count = hi - lo
            guard count > 0, hi <= triUVa.count, hi <= triUVc.count else { continue }
            checked.insert(kind)
            let canon = perMeshCardUVs(triangleCount: count)
            for t in 0..<count {
                let a = triUVa[lo + t], c = triUVc[lo + t], p = canon[t]
                let okA = a.x == p.uvA.x && a.y == p.uvA.y && a.z == p.uvB.x && a.w == p.uvB.y
                let okC = c.x == p.uvC.x && c.y == p.uvC.y
                if !(okA && okC) {
                    Self.log.error("perMeshCardUVs invariance FAIL: kind=\(String(describing: kind)) localTri=\(t) global=\(lo + t) — fold would corrupt this mesh's cache UVs. (#60 item 6)")
                    break
                }
            }
        }
        #endif
    }

    /// Phase 4.38c — surface-cache generalisation (#17 item 3, P2). **Coplanar-
    /// chart merging.** Instead of one micro-card per triangle, merge adjacent
    /// COPLANAR same-material triangles into a single larger `SurfCard` "chart"
    /// with a shared 2D parameterisation. Wins over the per-triangle path:
    ///   • Far fewer cards — a triangulated wall / road / building face re-emerges
    ///     as one chart (the room's hand-authored quads come back automatically).
    ///   • Better texel coherence — a flat surface is one contiguous tile, not N
    ///     disjoint micro-tiles with seams.
    ///   • Larger, more STABLE cards — the unit incremental invalidation tracks.
    ///
    /// A chart is an UNPACKED single-frame `SurfCard` (frame A fills the whole
    /// tile) — exactly the room's hand-authored-quad mode — so the update kernel
    /// + the consumer samplers are reused VERBATIM (zero GPU change vs P1/P3).
    ///
    /// Grouping (flood fill over edge adjacency):
    ///   • coplanar — two edge-adjacent triangles with parallel normals
    ///     (`dot > normalCosThresh`) are coplanar by construction (a shared edge
    ///     lies in both planes; parallel normals ⇒ same plane).
    ///   • same material — albedo + emission within tolerance (keeps lit windows
    ///     out of wall charts; the card carries ONE albedo/emission).
    ///   • bounded extent — the chart's 2D bbox side is capped (`maxExtent`) so a
    ///     uniform atlas tile keeps a consistent texel density (a huge wall splits
    ///     into a few medium charts, not one giant blurry one — still far fewer
    ///     than per-triangle).
    ///
    /// Parameterisation: fix a 2D basis (tangent/bitangent) from the seed normal,
    /// project all chart verts, take the 2D bbox → frame `origin = P0 + umin·T +
    /// vmin·B`, `uAxis = (umax−umin)·T`, `vAxis = (vmax−vmin)·B` (so the kernel's
    /// `origin + s·uAxis + t·vAxis` reconstructs the plane). Each triangle's verts
    /// map to `(coord2D − bboxMin)/bboxSize ∈ [0,1]²`.
    public static func makeCoplanarChartCards(
        positions: [SIMD3<Float>],
        indices: [UInt32],
        triangleAlbedo: [SIMD3<Float>],
        triangleNormal: [SIMD3<Float>],
        triangleEmission: [SIMD3<Float>]? = nil,
        normalCosThresh: Float = 0.996,   // ≈5°
        albedoTol: Float = 0.05,
        emissionTol: Float = 0.15,
        maxExtent: Float = 4.0            // metres, chart 2D bbox side cap
    ) -> (cards: [SurfCard], triCard: [UInt32],
          triUVa: [SIMD4<Float>], triUVc: [SIMD4<Float>]) {
        let triCount = indices.count / 3
        var triCard = [UInt32](repeating: 0, count: triCount)
        var triUVa  = [SIMD4<Float>](repeating: .zero, count: triCount)
        var triUVc  = [SIMD4<Float>](repeating: .zero, count: triCount)
        var cards: [SurfCard] = []; cards.reserveCapacity(triCount / 4 + 1)

        func tri(_ t: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)? {
            let i0 = Int(indices[3*t]), i1 = Int(indices[3*t+1]), i2 = Int(indices[3*t+2])
            guard i0 < positions.count, i1 < positions.count, i2 < positions.count,
                  t < triangleAlbedo.count, t < triangleNormal.count else { return nil }
            return (positions[i0], positions[i1], positions[i2])
        }
        // Orthonormal 2D basis for a plane normal (Duff et al. branchless frame).
        func basis(_ n: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
            let s: Float = n.z >= 0 ? 1 : -1
            let a = -1 / (s + n.z)
            let b = n.x * n.y * a
            let tangent = SIMD3<Float>(1 + s * n.x * n.x * a, s * b, -s * n.x)
            let bitan = SIMD3<Float>(b, s + n.y * n.y * a, -n.y)
            return (tangent, bitan)
        }
        func matClose(_ t0: Int, _ t1: Int) -> Bool {
            let da = simd_abs(triangleAlbedo[t0] - triangleAlbedo[t1])
            if max(da.x, max(da.y, da.z)) > albedoTol { return false }
            let e0 = triangleEmission?[t0] ?? .zero, e1 = triangleEmission?[t1] ?? .zero
            let de = simd_abs(e0 - e1)
            return max(de.x, max(de.y, de.z)) <= emissionTol
        }

        // Edge → adjacent triangles. Key = sorted vertex-index pair packed in u64.
        var edgeMap: [UInt64: [Int]] = [:]; edgeMap.reserveCapacity(triCount * 3)
        func edgeKey(_ a: UInt32, _ b: UInt32) -> UInt64 {
            let lo = min(a, b), hi = max(a, b)
            return (UInt64(lo) << 32) | UInt64(hi)
        }
        for t in 0..<triCount {
            guard tri(t) != nil else { continue }
            let i0 = indices[3*t], i1 = indices[3*t+1], i2 = indices[3*t+2]
            edgeMap[edgeKey(i0, i1), default: []].append(t)
            edgeMap[edgeKey(i1, i2), default: []].append(t)
            edgeMap[edgeKey(i2, i0), default: []].append(t)
        }
        func neighbours(_ t: Int) -> [Int] {
            let i0 = indices[3*t], i1 = indices[3*t+1], i2 = indices[3*t+2]
            var out: [Int] = []
            for k in [edgeKey(i0, i1), edgeKey(i1, i2), edgeKey(i2, i0)] {
                if let ts = edgeMap[k] { for o in ts where o != t { out.append(o) } }
            }
            return out
        }

        var visited = [Bool](repeating: false, count: triCount)
        for seed in 0..<triCount {
            if visited[seed] { continue }
            guard let (s0, s1, s2) = tri(seed) else {
                // Degenerate triangle — lone zero card to keep primitive alignment.
                visited[seed] = true
                let card = UInt32(cards.count)
                cards.append(SurfCard(origin: .zero, uAxis: .zero, vAxis: .zero,
                                      normal: SIMD3<Float>(0, 1, 0), albedo: .zero, emission: .zero))
                triCard[seed] = card
                continue
            }
            let N = triangleNormal[seed]
            let (T, B) = basis(N)
            let P0 = s0
            func proj(_ v: SIMD3<Float>) -> SIMD2<Float> {
                let d = v - P0; return SIMD2(simd_dot(d, T), simd_dot(d, B))
            }
            // BFS, growing the chart while the 2D bbox stays within maxExtent.
            var members: [Int] = [seed]
            visited[seed] = true
            var uMin = Float.greatestFiniteMagnitude, uMax = -Float.greatestFiniteMagnitude
            var vMin = Float.greatestFiniteMagnitude, vMax = -Float.greatestFiniteMagnitude
            func extend(_ verts: [SIMD3<Float>]) {
                for v in verts { let p = proj(v)
                    uMin = min(uMin, p.x); uMax = max(uMax, p.x)
                    vMin = min(vMin, p.y); vMax = max(vMax, p.y) }
            }
            extend([s0, s1, s2])
            var queue = [seed]; var qi = 0
            while qi < queue.count {
                let cur = queue[qi]; qi += 1
                for nb in neighbours(cur) where !visited[nb] {
                    guard let (n0, n1, n2) = tri(nb) else { continue }
                    if simd_dot(triangleNormal[nb], N) < normalCosThresh { continue }
                    if !matClose(seed, nb) { continue }
                    // Tentative bbox if we add nb — reject if it busts the extent cap.
                    var tuMin = uMin, tuMax = uMax, tvMin = vMin, tvMax = vMax
                    for v in [n0, n1, n2] { let p = proj(v)
                        tuMin = min(tuMin, p.x); tuMax = max(tuMax, p.x)
                        tvMin = min(tvMin, p.y); tvMax = max(tvMax, p.y) }
                    if (tuMax - tuMin) > maxExtent || (tvMax - tvMin) > maxExtent { continue }
                    visited[nb] = true
                    members.append(nb)
                    uMin = tuMin; uMax = tuMax; vMin = tvMin; vMax = tvMax
                    queue.append(nb)
                }
            }
            // Build the chart card from the final bbox.
            let du = max(uMax - uMin, 1e-4), dv = max(vMax - vMin, 1e-4)
            let origin = P0 + uMin * T + vMin * B
            let card = UInt32(cards.count)
            cards.append(SurfCard(
                origin: origin, uAxis: du * T, vAxis: dv * B,
                normal: N, albedo: triangleAlbedo[seed],
                emission: triangleEmission?[seed] ?? .zero))
            // Map each member triangle's verts into the chart's [0,1]² space.
            for m in members {
                guard let (m0, m1, m2) = tri(m) else { continue }
                func uv(_ v: SIMD3<Float>) -> SIMD2<Float> {
                    let p = proj(v)
                    return SIMD2((p.x - uMin) / du, (p.y - vMin) / dv)
                }
                let a = uv(m0), b = uv(m1), c = uv(m2)
                triCard[m] = card
                triUVa[m] = SIMD4(a.x, a.y, b.x, b.y)
                triUVc[m] = SIMD4(c.x, c.y, 0, 0)
            }
        }
        return (cards, triCard, triUVa, triUVc)
    }

    /// Register surface-cache cards + their per-triangle UV map and allocate the
    /// ping-pong radiance atlas. Addressing is per-card RECT (`cardRect` +
    /// `texelCard`), shelf-packed — the uniform tile grid is the special case
    /// where every card requests the same tile size.
    ///
    /// `adaptiveTiles == false` (room quads, P3 per-triangle): every card gets a
    /// `tileSize²` tile. `adaptiveTiles == true` (P2 charts): each card's tile is
    /// sized to its world extent (`|uAxis| × |vAxis| · texelsPerMeter`, clamped),
    /// so a big wall chart gets more texels than a small facet and the atlas stays
    /// efficient regardless of chart-size spread.
    public func setSurfaceCacheCards(cards: [SurfCard],
                                     triCard: [UInt32],
                                     triUVa: [SIMD4<Float>],
                                     triUVc: [SIMD4<Float>],
                                     tileSize overrideTileSize: Int? = nil,
                                     adaptiveTiles: Bool = false,
                                     texelsPerMeter: Float = 6.0) {
        guard rtSupported, scs.surfCachePipeline != nil else { return }
        guard !cards.isEmpty, triCard.count == rts.rtTriangleCount,
              triUVa.count == rts.rtTriangleCount, triUVc.count == rts.rtTriangleCount else {
            Self.log.error("setSurfaceCacheCards: counts must match triangle/card sizes")
            return
        }
        let baseTile = max(2, overrideTileSize ?? scs.surfTileSize)
        scs.surfActiveTileSize = baseTile

        // (1) Per-card desired tile size (w,h) in texels.
        let minTile = 4, maxTile = 48
        var sizes = [SIMD2<Int>](repeating: SIMD2(baseTile, baseTile), count: cards.count)
        if adaptiveTiles {
            for c in 0..<cards.count {
                let wm = simd_length(SIMD3(cards[c].uAxis.x, cards[c].uAxis.y, cards[c].uAxis.z))
                let hm = simd_length(SIMD3(cards[c].vAxis.x, cards[c].vAxis.y, cards[c].vAxis.z))
                let tw = min(maxTile, max(minTile, Int((wm * texelsPerMeter).rounded())))
                let th = min(maxTile, max(minTile, Int((hm * texelsPerMeter).rounded())))
                sizes[c] = SIMD2(tw, th)
            }
        }

        // Phase 5 / A0 — residency budget. RESIDENT cards (the first `budget` in
        // bake order) get an atlas slot + are packed; the rest get a zero rect (no
        // slot), so the atlas is sized to the BUDGET, not the scene. A GI/reflection
        // hit on a non-resident card detects the zero rect and falls back. Disabled
        // (budget 0 / ≥ count) ⇒ every card resident, identical to before.
        let residentCount = (surfCardBudget > 0) ? min(surfCardBudget, cards.count) : cards.count
        let isResident: (Int) -> Bool = { $0 < residentCount }

        // (2) Shelf-pack the rects. Sort by height desc for tight shelves; place
        // left→right, wrap to a new shelf when the row exceeds the target width.
        // Only resident cards contribute area / get packed.
        let totalArea = (0..<cards.count).reduce(0) { isResident($1) ? $0 + sizes[$1].x * sizes[$1].y : $0 }
        let maxTileW = (0..<cards.count).reduce(baseTile) { isResident($1) ? max($0, sizes[$1].x) : $0 }
        let atlasW = max(maxTileW, Int((Double(totalArea).squareRoot() * 1.08).rounded(.up)))
        let order = (0..<cards.count).filter(isResident).sorted { sizes[$0].y > sizes[$1].y }
        var rects = [SIMD4<Float>](repeating: .zero, count: cards.count)  // x,y,w,h (.zero = non-resident)
        var penX = 0, penY = 0, shelfH = 0
        for c in order {
            let (w, h) = (sizes[c].x, sizes[c].y)
            if penX + w > atlasW { penX = 0; penY += shelfH; shelfH = 0 }
            rects[c] = SIMD4(Float(penX), Float(penY), Float(w), Float(h))
            penX += w
            shelfH = max(shelfH, h)
        }
        let atlasH = penY + shelfH
        scs.surfResidentCardCount = residentCount
        // Phase 5 / A1 — slot maps (only when a budget is active). A "slot" is a
        // bake-time resident card's atlas rect; A1 reassigns occupants per frame.
        if residentCount < cards.count {
            scs.surfSlotRect = (0..<residentCount).map { rects[$0] }
            scs.surfSlotCard = Array(0..<residentCount)
            scs.surfCardSlot = (0..<cards.count).map { $0 < residentCount ? $0 : -1 }
            scs.surfCardHeat = [Int](repeating: 0, count: cards.count)   // A2 — all cold at bake
            // A4 — bake-time world centroid per card (origin + ½(uAxis + vAxis)).
            scs.surfCardCentroid = cards.map {
                let o = SIMD3($0.origin.x, $0.origin.y, $0.origin.z)
                let u = SIMD3($0.uAxis.x, $0.uAxis.y, $0.uAxis.z)
                let v = SIMD3($0.vAxis.x, $0.vAxis.y, $0.vAxis.z)
                return o + 0.5 * (u + v)
            }
        } else {
            scs.surfSlotRect = []; scs.surfSlotCard = []; scs.surfCardSlot = []; scs.surfCardHeat = []
            scs.surfCardCentroid = []
        }
        guard atlasW > 0, atlasH > 0 else { Self.log.error("setSurfaceCacheCards: empty atlas"); return }

        // (3) texelCard map (uint per atlas texel; 0xFFFFFFFF = unused gap).
        var texelCard = [UInt32](repeating: 0xFFFF_FFFF, count: atlasW * atlasH)
        for c in 0..<cards.count {
            let r = rects[c]
            let x0 = Int(r.x), y0 = Int(r.y), w = Int(r.z), h = Int(r.w)
            for yy in y0..<(y0 + h) {
                let rowBase = yy * atlasW
                for xx in x0..<(x0 + w) { texelCard[rowBase + xx] = UInt32(c) }
            }
        }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: atlasW, height: atlasH, mipmapped: false)
        texDesc.usage = [.shaderRead, .shaderWrite]
        texDesc.storageMode = .private
        guard
            let cb = device.makeBuffer(bytes: cards,
                length: MemoryLayout<SurfCard>.stride * cards.count, options: .storageModeShared),
            let tcb = device.makeBuffer(bytes: triCard,
                length: MemoryLayout<UInt32>.stride * triCard.count, options: .storageModeShared),
            let tab = device.makeBuffer(bytes: triUVa,
                length: MemoryLayout<SIMD4<Float>>.stride * triUVa.count, options: .storageModeShared),
            let tcc = device.makeBuffer(bytes: triUVc,
                length: MemoryLayout<SIMD4<Float>>.stride * triUVc.count, options: .storageModeShared),
            let crb = device.makeBuffer(bytes: rects,
                length: MemoryLayout<SIMD4<Float>>.stride * rects.count, options: .storageModeShared),
            let txb = device.makeBuffer(bytes: texelCard,
                length: MemoryLayout<UInt32>.stride * texelCard.count, options: .storageModeShared),
            let atlasA = device.makeTexture(descriptor: texDesc),
            let atlasB = device.makeTexture(descriptor: texDesc)
        else {
            Self.log.error("setSurfaceCacheCards: allocation failed")
            return
        }
        cb.label = "Illuminatorama.surfcache.cards"
        crb.label = "Illuminatorama.surfcache.cardRect"
        txb.label = "Illuminatorama.surfcache.texelCard"
        atlasA.label = "Illuminatorama.surfcache.atlasA"
        atlasB.label = "Illuminatorama.surfcache.atlasB"
        scs.surfCardBuffer = cb
        scs.surfTriCardBuffer = tcb
        scs.surfTriUVaBuffer = tab
        scs.surfTriUVcBuffer = tcc
        scs.surfCardRectBuffer = crb
        scs.surfTexelCardBuffer = txb
        // Incremental-invalidation per-card dirty buffer (all clean initially).
        scs.surfCardDirtyBuffer = device.makeBuffer(
            length: max(1, MemoryLayout<UInt32>.stride * cards.count), options: .storageModeShared)
        scs.surfCardDirtyBuffer?.label = "Illuminatorama.surfcache.cardDirty"
        if let d = scs.surfCardDirtyBuffer { memset(d.contents(), 0, d.length) }
        scs.surfIncrementalReady = false   // bake site repopulates the CPU maps
        scs.surfChartIncremental = nil     // ditto (chart path repopulates if it bakes)
        scs.surfChartMovedScratch = []     // re-sized at the chart bake site
        // GPU-diff maps are topology-keyed too; the bake site rebuilds them.
        scs.surfGPUDiffTriOwnerBuffer = nil; scs.surfGPUDiffTriFlagsBuffer = nil
        scs.surfGPUDiffObjVertsBuffer = nil; scs.surfGPUDiffTriCount = 0
        scs.surfGPUDiffChartOwnerBuffer = nil; scs.surfGPUDiffChartFrameBuffer = nil
        scs.surfGPUDiffDeformIdxBuffer = nil; scs.surfGPUDiffDeformDispatches = []
        scs.surfAtlasA = atlasA
        scs.surfAtlasB = atlasB
        // DH-0622 — nil on alloc failure leaves the update (and every consumer) off.
        scs.surfIndirectAtlasA = device.makeTexture(descriptor: texDesc)
        scs.surfIndirectAtlasB = device.makeTexture(descriptor: texDesc)
        scs.surfIndirectAtlasA?.label = "Illuminatorama.surfcache.indirectA"
        scs.surfIndirectAtlasB?.label = "Illuminatorama.surfcache.indirectB"
        scs.surfCacheWarmPending = max(0, surfaceCacheWarmIterations)   // fresh textures are empty
        // Phase 5 / B1 — denoiser output (same descriptor; not ping-ponged). nil on
        // alloc failure just leaves consumers on the raw atlas (the filter no-ops).
        scs.surfAtlasDenoised = device.makeTexture(descriptor: texDesc)
        scs.surfAtlasDenoised?.label = "Illuminatorama.surfcache.atlasDenoised"
        // Phase 5 / A — residency feedback buffer (uint per card; 1 = sampled).
        // Shared so the host can drain + zero it per frame. nil ⇒ feedback no-ops.
        scs.surfCardRequestedBuffer = device.makeBuffer(
            length: max(1, MemoryLayout<UInt32>.stride * cards.count), options: .storageModeShared)
        scs.surfCardRequestedBuffer?.label = "Illuminatorama.surfcache.cardRequested"
        if let r = scs.surfCardRequestedBuffer { memset(r.contents(), 0, r.length) }
        scs.surfCardCount = cards.count
        scs.surfTilesPerRow = 1
        scs.surfAtlasW = atlasW
        scs.surfAtlasH = atlasH
        scs.surfUseAtlasA = true
        let fillPct = Int((Double(totalArea) / Double(atlasW * atlasH) * 100).rounded())
        let budgetNote = residentCount < cards.count ? " [budget: \(residentCount)/\(cards.count) resident]" : ""
        Self.log.notice("Surface cache: \(cards.count) cards, \(atlasW)×\(atlasH) atlas (\(fillPct)% packed)\(budgetNote)")
        Self.recordSurfCacheStats("cards=\(cards.count) resident=\(residentCount) tris=\(rts.rtTriangleCount) adaptive=\(adaptiveTiles) atlas=\(atlasW)x\(atlasH) texels=\(atlasW * atlasH) used=\(totalArea) fill=\(fillPct)% bytesRGBA16F=\(atlasW * atlasH * 8)")
    }
}
