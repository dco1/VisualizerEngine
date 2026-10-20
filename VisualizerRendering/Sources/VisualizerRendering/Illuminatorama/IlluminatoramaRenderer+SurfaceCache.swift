import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {

    /// Diagnostic: append surface-cache atlas sizing to the sidecar file named by
    /// `VIZ_ILLUMI_SURFCACHE_STATS_PATH` (sandbox eats os_log). Used to quantify
    /// the P2/P3 atlas-efficiency wins before/after. No-op without the env var.
    nonisolated static func recordSurfCacheStats(_ line: String) {
        guard let p = frameEnv("VIZ_ILLUMI_SURFCACHE_STATS_PATH") else { return }
        guard let data = (line + "\n").data(using: .utf8) else { return }
        if let h = FileHandle(forWritingAtPath: p) { h.seekToEndOfFile(); h.write(data); try? h.close() }
        else { try? data.write(to: URL(fileURLWithPath: p)) }
    }

    /// Allocate (zeroed) on first use; nil only on allocation failure (counters then no-op).
    func surfCounterBuffer(_ existing: MTLBuffer?, slots: Int, label: String) -> MTLBuffer? {
        if let existing { return existing }
        let b = device.makeBuffer(length: MemoryLayout<UInt32>.stride * slots, options: .storageModeShared)
        b?.label = label
        if let b { memset(b.contents(), 0, b.length) }
        return b
    }

    /// Read + zero a counter buffer. Opt-in diagnostic lane only, so it may wait for the
    /// in-flight frame whose kernel is still adding (the feedback drain does the same).
    func drainSurfCounters(_ b: MTLBuffer, slots: Int) -> [UInt32] {
        awaitInFlightFrameBeforeCPUCacheWrite()
        let p = b.contents().bindMemory(to: UInt32.self, capacity: slots)
        let counts = (0..<slots).map { p[$0] }
        memset(b.contents(), 0, MemoryLayout<UInt32>.stride * slots)
        return counts
    }

    /// `hits:` sidecar line from the 8 hit slots (GI 0–3, reflection 4–7, each
    /// cached / fallback / noCard / reshade). nil when the frame made no surface hit.
    nonisolated static func surfCacheHitLine(_ c: [UInt32]) -> String? {
        guard c.count >= 8 else { return nil }
        let total = c.prefix(8).reduce(0) { $0 + Int($1) }
        guard total > 0 else { return nil }
        let cached = Int(c[0]) + Int(c[4])
        let rate = String(format: "%.1f", Double(cached) / Double(total) * 100)
        func group(_ o: Int) -> String {
            "cached=\(c[o]) fallback=\(c[o + 1]) noCard=\(c[o + 2]) reshade=\(c[o + 3])"
        }
        return "hits: cacheHitRate=\(rate)% cached=\(cached) of \(total) surface hits"
            + " gi[\(group(0))] refl[\(group(4))]"
    }

    /// `atlas:` sidecar line — texels temporally reused vs recomputed from scratch.
    nonisolated static func surfCacheAtlasLine(reused: UInt32, refreshed: UInt32) -> String? {
        let total = Int(reused) + Int(refreshed)
        guard total > 0 else { return nil }
        let rate = String(format: "%.1f", Double(reused) / Double(total) * 100)
        return "atlas: reuseRate=\(rate)% reused=\(reused) refreshed=\(refreshed) of \(total) texels"
    }

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

    /// Whether the surface-cache update should run this frame.
    var surfCacheActive: Bool {
        surfaceCacheEnabled && rtEnabled && rtSupported
            && scs.surfCachePipeline != nil && scs.surfCardCount > 0 && rts.rtAccel != nil
    }

    /// Read a deforming (DynamicMesh / GPU-fed) mesh's CURRENT object-space
    /// triangle soup directly from the shared buffers the GPU traces: live
    /// positions from the repack task's `packed_float3` source buffer (the host
    /// or a kernel rewrites it every frame), indices from the mesh's combined
    /// index buffer. `IlluminatoramaMesh.objectTriangleSoup()` can't do this — the
    /// interleaved vertex buffer is `.private`, so it returns empty. Returns nil
    /// for a non-deforming kind, a private source buffer, or empty topology.
    func liveDeformingObjectSoup(kind: MeshKind)
        -> (positions: [SIMD3<Float>], indices: [UInt32])?
    {
        guard let mesh = meshes[kind],
              let task = fxs.gpuRepackTasks.first(where: { $0.kind == kind }),
              task.positionBuffer.storageMode == .shared,
              mesh.indexBuffer.storageMode == .shared else { return nil }
        let n = task.vertexCount
        guard n > 0, mesh.indexCount >= 3 else { return nil }
        // Source positions are packed_float3 (stride 12) — the layout the repack
        // kernel + BLAS read. Reading them as SIMD3<Float> (stride 16) would shear
        // every vertex past the first (the SIMD3 stride trap).
        let pbase = task.positionBuffer.contents().bindMemory(to: Float.self, capacity: n * 3)
        var positions = [SIMD3<Float>](); positions.reserveCapacity(n)
        for v in 0..<n { positions.append(SIMD3<Float>(pbase[3*v], pbase[3*v+1], pbase[3*v+2])) }
        let ibase = mesh.indexBuffer.contents()
        var indices = [UInt32](); indices.reserveCapacity(mesh.indexCount)
        if mesh.indexType == .uint16 {
            for i in 0..<mesh.indexCount { indices.append(UInt32(ibase.load(fromByteOffset: i * 2, as: UInt16.self))) }
        } else {
            for i in 0..<mesh.indexCount { indices.append(ibase.load(fromByteOffset: i * 4, as: UInt32.self)) }
        }
        return (positions, indices)
    }

    /// Re-frame dynamic surface cards in place each frame, BEFORE
    /// `encodeSurfaceCacheUpdate`, so the update kernel re-lights moved/deformed
    /// surfaces at their CURRENT pose. Two dynamic sources, one shared mechanism
    /// (write the card's plane + flag it dirty → kernel uses α=1 to discard stale
    /// history while stationary cards keep their accumulated multi-bounce):
    ///
    ///  • **Rigid incremental invalidation** (`surfaceCacheIncremental`, default
    ///    on): an instance whose model matrix changed re-frames from its STATIC
    ///    object soup × the new transform. See
    ///    docs/illuminatorama/surface-cache-incremental-invalidation.md.
    ///  • **Phase C — deforming meshes** (DynamicMesh / GPU-fed): a mesh whose
    ///    *shape* changes in place (the transform may never move, so the rigid
    ///    diff misses it). Re-frames from this frame's LIVE object soup ×
    ///    transform, every frame. This is what makes a rippling sheet / sloshing
    ///    fluid / waving flag both CATCH the room's bounced colour (its cards sit
    ///    on the real surface, sampling neighbours' caches) and CAST it back
    ///    (its now-correctly-placed cards re-light and are sampled by others'
    ///    GI/reflection rays — and, on the TLAS update path, with current-pose
    ///    occlusion). On by default whenever a deforming mesh drives the cache;
    ///    `VIZ_ILLUMI_NO_DEFORM_CARDS=1` freezes it at build pose for A/B.
    ///
    /// PERF (Phase C → D): the deforming re-frame was a per-frame CPU readback +
    /// transform of the live soup; Phase D moved it to
    /// `illumi_surfcache_reframe_deform` (a compute kernel writing
    /// `surfCardBuffer` straight from the solver's live position buffer). The
    /// CPU loop below remains only as the `VIZ_SURFCACHE_GPU_DIFF=0` fallback.
    ///
    /// `surfCardBuffer` is written in place (not ping-ponged). On the default GPU lane that
    /// is a kernel on this frame's command buffer, ordered by hazard tracking; on the CPU
    /// fallback it is a host write that first waits for the in-flight frame
    /// (`awaitInFlightFrameBeforeCPUCacheWrite`, DH-0537 #1) — the pool is 2.
    ///
    /// **#60 item 1 + Phase D — GPU-side transform diff.** Both dynamic sources
    /// now run as compute kernels on the frame's command buffer: RIGID movers
    /// via `illumi_surfcache_reframe_tri` / `_chart` (each thread diffs its
    /// owner instance's current-vs-previous model matrix inline and re-frames
    /// its card (half) in place), DEFORMING meshes via
    /// `illumi_surfcache_reframe_deform` (one dispatch per deforming instance,
    /// reading the solver's live position buffer — no CPU readback). All write
    /// the same `surfCardBuffer` / dirty flags the update kernel consumes later
    /// in the SAME command buffer (hazard tracking orders them). The CPU loops
    /// below remain solely as the `VIZ_SURFCACHE_GPU_DIFF=0` kill-switch / A/B
    /// reference (and the safety fallback when a map or live buffer is
    /// unresolvable mid-topology).
    func applyIncrementalSurfaceCacheUpdate(_ cb: MTLCommandBuffer) {
        guard surfCacheActive, scs.surfIncrementalReady,
              let dirtyBuf = scs.surfCardDirtyBuffer, let cardBuf = scs.surfCardBuffer else { return }
        // Chart path (`VIZ_SURFCACHE_CHARTS=1`): re-frame each moved rigid instance's
        // cards by a single affine transform of their baked object-space frame.
        if let chart = scs.surfChartIncremental {
            if surfaceCacheIncremental, !Self.surfCacheGPUDiffDisabled,
               let pipe = scs.surfReframeChartPipeline,
               let ownerBuf = scs.surfGPUDiffChartOwnerBuffer,
               let frameBuf = scs.surfGPUDiffChartFrameBuffer,
               let statsBuf = surfGPUDiffStatsBuffer,
               chart.instCardLo.count == instances.count {
                encodeGPUDiffReframe(cb, pipeline: pipe, threadCount: scs.surfCardCount,
                                     dirtyBuf: dirtyBuf, cardBuf: cardBuf,
                                     ownerBuf: ownerBuf, statsBuf: statsBuf,
                                     chartFrameBuf: frameBuf, tag: "[chart gpu]")
                return
            }
            applyIncrementalChartSurfaceCacheUpdate(chart, dirtyBuf: dirtyBuf, cardBuf: cardBuf)
            return
        }
        // Per-triangle path: GPU kernels (#60 item 1 + Phase D). The rigid
        // kernel owns every non-deforming card's diff + re-frame + dirty flag;
        // one deform dispatch per deforming instance re-frames its cards from
        // the LIVE position buffer (no readback). Falls back to the CPU loop
        // below when the kill-switch is set or any map / pipeline / live
        // buffer is missing.
        if !Self.surfCacheGPUDiffDisabled,
           encodeGPUDiffPerTriPath(cb, dirtyBuf: dirtyBuf, cardBuf: cardBuf) {
            return
        }
        guard scs.surfSoupBaseCPU.count == instances.count,        // topology matches the baked maps
              !scs.surfTriCardCPU.isEmpty else { return }

        // Which kinds deform in place this frame? (Phase C re-frames them always.)
        let deformingKinds: Set<MeshKind> =
            (scs.surfHasDeformingCards && !scs.noDeformCardsOverride) ? Set(fxs.gpuRepackTasks.map(\.kind)) : []
        let rigidIncremental = surfaceCacheIncremental
        // Nothing dynamic → leave every card fully accumulating (cheapest path).
        guard rigidIncremental || !deformingKinds.isEmpty else { return }

        // Snapshot each deforming kind's live object soup ONCE (shared by all its
        // instances). Indices are constant topology; positions are this frame's.
        var liveSoupByKind: [MeshKind: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
        for kind in deformingKinds {
            if let soup = liveDeformingObjectSoup(kind: kind) { liveSoupByKind[kind] = soup }
        }

        let total = instances.count
        let curPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let prevPtr = previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)

        // CPU fallback: it rewrites the card + dirty buffers the in-flight frame may be reading.
        awaitInFlightFrameBeforeCPUCacheWrite()
        // Clear last frame's dirty flags; cards that stop moving revert to EMA.
        memset(dirtyBuf.contents(), 0, dirtyBuf.length)
        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
        let cards = cardBuf.contents().bindMemory(to: SurfCard.self, capacity: scs.surfCardCount)
        let soupTriTotal = scs.surfTriCardCPU.count

        func moved(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
            let e: Float = 1e-5
            for c in 0..<4 {
                let d = a[c] - b[c]
                if abs(d.x) > e || abs(d.y) > e || abs(d.z) > e || abs(d.w) > e { return true }
            }
            return false
        }

        var movedInstances = 0
        var deformedInstances = 0
        var dirtiedCards = 0
        for i in 0..<total {
            let kind = scs.surfInstanceKind[i]
            let isDeforming = deformingKinds.contains(kind)
            let model = curPtr[i].modelMatrix
            // Re-frame when: a deforming instance (its verts changed — always), OR
            // a rigid instance whose transform moved (incremental invalidation).
            let soup: (positions: [SIMD3<Float>], indices: [UInt32])?
            if isDeforming {
                soup = liveSoupByKind[kind]
                deformedInstances += 1
            } else {
                guard rigidIncremental, moved(model, prevPtr[i].modelMatrix) else { continue }
                soup = scs.surfObjectSoupByKind[kind]
                movedInstances += 1
            }
            guard let soup, !soup.indices.isEmpty else { continue }
            let firstTri = Int(scs.surfSoupBaseCPU[i])
            let nextTri = (i + 1 < total) ? Int(scs.surfSoupBaseCPU[i + 1]) : soupTriTotal
            let localCount = min(nextTri - firstTri, soup.indices.count / 3)
            for j in 0..<max(0, localCount) {
                let gt = firstTri + j
                guard gt < soupTriTotal else { break }
                let i0 = Int(soup.indices[3*j]), i1 = Int(soup.indices[3*j+1]), i2 = Int(soup.indices[3*j+2])
                guard i0 < soup.positions.count, i1 < soup.positions.count, i2 < soup.positions.count else { continue }
                func world(_ p: SIMD3<Float>) -> SIMD3<Float> { let w = model * SIMD4<Float>(p, 1); return SIMD3(w.x, w.y, w.z) }
                let v0 = world(soup.positions[i0]), v1 = world(soup.positions[i1]), v2 = world(soup.positions[i2])
                var n = simd_cross(v1 - v0, v2 - v0)
                let len = simd_length(n); n = len > 1e-8 ? n / len : SIMD3<Float>(0, 1, 0)
                let card = Int(scs.surfTriCardCPU[gt])
                guard card < scs.surfCardCount else { continue }
                // Re-frame only this triangle's half of its (possibly packed) card;
                // preserve the packed flag (normal.w) + albedo/emission.
                if scs.surfTriIsFrameB[gt] {
                    cards[card].originB = SIMD4(v0, 0); cards[card].uAxisB = SIMD4(v1 - v0, 0)
                    cards[card].vAxisB = SIMD4(v2 - v0, 0); cards[card].normalB = SIMD4(n, cards[card].normalB.w)
                } else {
                    cards[card].origin = SIMD4(v0, 0); cards[card].uAxis = SIMD4(v1 - v0, 0)
                    cards[card].vAxis = SIMD4(v2 - v0, 0); cards[card].normal = SIMD4(n, cards[card].normal.w)
                }
                if dirty[card] == 0 { dirtiedCards += 1 }
                dirty[card] = 1
            }
        }
        // Diagnostic (env-gated, sandbox-safe sidecar): proves the dynamic path
        // fired this frame and how much it touched. No-op without
        // VIZ_ILLUMI_SURFCACHE_STATS_PATH.
        if movedInstances > 0 || deformedInstances > 0 {
            Self.recordSurfCacheStats("dynamic: moved=\(movedInstances) deformed=\(deformedInstances) dirtiedCards=\(dirtiedCards) of \(scs.surfCardCount)")
        }
    }

    /// Chart-path counterpart of `applyIncrementalSurfaceCacheUpdate` (#60 task 1).
    ///
    /// A coplanar chart card spans many triangles under ONE bbox-derived frame, so
    /// the per-triangle re-frame would clobber its parameterisation. Instead we use
    /// the card frame's **affine covariance**: for a rigid/affine model matrix `M`,
    /// the world frame is exactly `M · objectFrame` — `origin` as a point, `uAxis`
    /// / `vAxis` as vectors — and the fixed per-triangle UVs are preserved (verts
    /// and frame transform together). So each moved instance's cards re-frame with
    /// one matrix multiply per axis, no per-triangle work, no UV touch. Cards owned
    /// by a moved instance get α=1 (dirty) so they re-light from the new pose;
    /// every stationary card keeps its converged radiance.
    private func applyIncrementalChartSurfaceCacheUpdate(_ chart: ChartIncremental,
                                                         dirtyBuf: MTLBuffer, cardBuf: MTLBuffer) {
        // Charts are rigid-mover only — the deforming per-card path doesn't apply.
        guard surfaceCacheIncremental else { return }
        awaitInFlightFrameBeforeCPUCacheWrite()   // in-place card/dirty rewrite below
        let total = instances.count
        guard chart.instCardLo.count == total, chart.instCardHi.count == total,
              chart.objectFrame.count == scs.surfCardCount else { return }
        let curPtr = currentInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)
        let prevPtr = previousInstanceBuffer.contents().bindMemory(to: IlluminatoramaInstance.self, capacity: total)

        func moved(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
            let e: Float = 1e-5
            for c in 0..<4 {
                let d = a[c] - b[c]
                if abs(d.x) > e || abs(d.y) > e || abs(d.z) > e || abs(d.w) > e { return true }
            }
            return false
        }

        // Which instances moved this frame? (Decide before clearing dirty flags.)
        // Reuse the bake-time scratch buffer (no per-frame allocation); reset in
        // place. Fall back to a fresh buffer only if it's somehow mis-sized.
        if scs.surfChartMovedScratch.count != total {
            scs.surfChartMovedScratch = [Bool](repeating: false, count: total)
        } else {
            for i in 0..<total { scs.surfChartMovedScratch[i] = false }
        }
        var movedInstances = 0
        for i in 0..<total where moved(curPtr[i].modelMatrix, prevPtr[i].modelMatrix) {
            scs.surfChartMovedScratch[i] = true; movedInstances += 1
        }
        // Clear last frame's dirty flags so cards that stopped moving revert to EMA
        // (same reset semantics as the per-triangle path); skip the rest if static.
        memset(dirtyBuf.contents(), 0, dirtyBuf.length)
        guard movedInstances > 0 else { return }

        let dirty = dirtyBuf.contents().bindMemory(to: UInt32.self, capacity: scs.surfCardCount)
        let cards = cardBuf.contents().bindMemory(to: SurfCard.self, capacity: scs.surfCardCount)
        var dirtiedCards = 0
        for i in 0..<total where scs.surfChartMovedScratch[i] {
            let lo = Int(chart.instCardLo[i]), hi = Int(chart.instCardHi[i])
            guard lo >= 0, hi > lo, hi <= scs.surfCardCount else { continue }
            let m = curPtr[i].modelMatrix
            for c in lo..<hi {
                let f = chart.objectFrame[c]
                let o = m * SIMD4<Float>(f.origin, 1)
                let u = m * SIMD4<Float>(f.uAxis, 0)
                let v = m * SIMD4<Float>(f.vAxis, 0)
                let nn = m * SIMD4<Float>(f.normal, 0)
                var n = SIMD3<Float>(nn.x, nn.y, nn.z)
                let nl = simd_length(n); n = nl > 1e-8 ? n / nl : SIMD3<Float>(0, 1, 0)
                cards[c].origin = SIMD4(o.x, o.y, o.z, 0)
                cards[c].uAxis  = SIMD4(u.x, u.y, u.z, 0)
                cards[c].vAxis  = SIMD4(v.x, v.y, v.z, 0)
                cards[c].normal = SIMD4(n, cards[c].normal.w)   // preserve packed flag (0 for charts)
                if dirty[c] == 0 { dirtiedCards += 1 }
                dirty[c] = 1
            }
        }
        Self.recordSurfCacheStats("dynamic: moved=\(movedInstances) deformed=0 dirtiedCards=\(dirtiedCards) of \(scs.surfCardCount) [chart]")
    }

    /// Bake the per-triangle GPU-diff maps (#60 item 1 + Phase D): owner
    /// instance, flag bits (frame-B half / deforming), OBJECT-space verts (rigid
    /// re-frame), and vertex-index triples (deform re-frame) per grouped-soup
    /// triangle, accumulated in lockstep with the soup build so global triangle
    /// order matches exactly. Any mismatch leaves the buffers nil → the CPU
    /// re-frame path stays in charge.
    func buildGPUDiffPerTriBuffers(triOwner: [UInt32],
                                           objVerts: [SIMD4<Float>],
                                           isFrameB: [Bool],
                                           triIdx: [SIMD4<UInt32>],
                                           deforming: [Bool]) {
        scs.surfGPUDiffChartOwnerBuffer = nil; scs.surfGPUDiffChartFrameBuffer = nil
        scs.surfGPUDiffTriOwnerBuffer = nil; scs.surfGPUDiffTriFlagsBuffer = nil
        scs.surfGPUDiffObjVertsBuffer = nil; scs.surfGPUDiffTriCount = 0
        scs.surfGPUDiffDeformIdxBuffer = nil; scs.surfGPUDiffDeformDispatches = []
        let triCount = triOwner.count
        guard triCount > 0, triCount == rts.rtTriangleCount,
              objVerts.count == triCount * 3, isFrameB.count == triCount,
              triIdx.count == triCount, deforming.count == triCount,
              scs.surfReframeTriPipeline != nil else { return }
        var flags = [UInt32](repeating: 0, count: triCount)
        for t in 0..<triCount {
            flags[t] = (isFrameB[t] ? Self.scFlagFrameB : 0)
                     | (deforming[t] ? Self.scFlagDeforming : 0)
        }
        guard let ob = device.makeBuffer(bytes: triOwner,
                  length: MemoryLayout<UInt32>.stride * triCount, options: .storageModeShared),
              let fb = device.makeBuffer(bytes: flags,
                  length: MemoryLayout<UInt32>.stride * triCount, options: .storageModeShared),
              let vb = device.makeBuffer(bytes: objVerts,
                  length: MemoryLayout<SIMD4<Float>>.stride * objVerts.count, options: .storageModeShared),
              let ib = device.makeBuffer(bytes: triIdx,
                  length: MemoryLayout<SIMD4<UInt32>>.stride * triCount, options: .storageModeShared)
        else { return }
        ob.label = "Illuminatorama.surfcache.gpuDiff.triOwner"
        fb.label = "Illuminatorama.surfcache.gpuDiff.triFlags"
        vb.label = "Illuminatorama.surfcache.gpuDiff.objVerts"
        ib.label = "Illuminatorama.surfcache.gpuDiff.triIdx"
        scs.surfGPUDiffTriOwnerBuffer = ob
        scs.surfGPUDiffTriFlagsBuffer = fb
        scs.surfGPUDiffObjVertsBuffer = vb
        scs.surfGPUDiffDeformIdxBuffer = ib
        scs.surfGPUDiffTriCount = triCount
        for sb in scs.surfGPUDiffStatsRing { memset(sb.contents(), 0, sb.length) }
    }

    /// Chart-path GPU-diff maps (#60 item 1): per-card owner instance (−1 =
    /// static / degenerate, never re-framed) + bake-pose object-space frame,
    /// flattened to 4×float4 per card for the kernel.
    func buildGPUDiffChartBuffers(objFrames: [SurfCardFrame], cardOwner: [Int32]) {
        scs.surfGPUDiffTriOwnerBuffer = nil; scs.surfGPUDiffTriFlagsBuffer = nil
        scs.surfGPUDiffObjVertsBuffer = nil; scs.surfGPUDiffTriCount = 0
        scs.surfGPUDiffChartOwnerBuffer = nil; scs.surfGPUDiffChartFrameBuffer = nil
        guard scs.surfCardCount > 0, objFrames.count == scs.surfCardCount,
              cardOwner.count == scs.surfCardCount,
              scs.surfReframeChartPipeline != nil else { return }
        var frames = [SIMD4<Float>](); frames.reserveCapacity(scs.surfCardCount * 4)
        for f in objFrames {
            frames.append(SIMD4(f.origin, 0)); frames.append(SIMD4(f.uAxis, 0))
            frames.append(SIMD4(f.vAxis, 0));  frames.append(SIMD4(f.normal, 0))
        }
        guard let ob = device.makeBuffer(bytes: cardOwner,
                  length: MemoryLayout<Int32>.stride * cardOwner.count, options: .storageModeShared),
              let fb = device.makeBuffer(bytes: frames,
                  length: MemoryLayout<SIMD4<Float>>.stride * frames.count, options: .storageModeShared)
        else { return }
        ob.label = "Illuminatorama.surfcache.gpuDiff.cardOwner"
        fb.label = "Illuminatorama.surfcache.gpuDiff.cardFrame"
        scs.surfGPUDiffChartOwnerBuffer = ob
        scs.surfGPUDiffChartFrameBuffer = fb
        for sb in scs.surfGPUDiffStatsRing { memset(sb.contents(), 0, sb.length) }
    }

    /// Phase D + #60 item 1 — the full GPU per-triangle dynamic-card pass.
    /// Returns false (CPU fallback) unless every map, pipeline, and deforming
    /// instance's live position buffer is resolvable; on success encodes the
    /// rigid kernel (diff + re-frame + dirty ownership for every non-deforming
    /// card) followed by one deform dispatch per deforming instance.
    private func encodeGPUDiffPerTriPath(_ cb: MTLCommandBuffer,
                                         dirtyBuf: MTLBuffer, cardBuf: MTLBuffer) -> Bool {
        let deformingActive = scs.surfHasDeformingCards && !scs.noDeformCardsOverride
        guard surfaceCacheIncremental || deformingActive,
              let pipe = scs.surfReframeTriPipeline,
              let ownerBuf = scs.surfGPUDiffTriOwnerBuffer,
              let flagsBuf = scs.surfGPUDiffTriFlagsBuffer,
              let vertsBuf = scs.surfGPUDiffObjVertsBuffer,
              let statsBuf = surfGPUDiffStatsBuffer,
              let triCardBuf = scs.surfTriCardBuffer,
              scs.surfGPUDiffTriCount == rts.rtTriangleCount,
              scs.surfSoupBaseCPU.count == instances.count else { return false }
        // Resolve every deform dispatch's live position buffer up front — if any
        // is missing (solver unregistered ahead of the re-bake), take the CPU
        // path this frame rather than half-updating the dirty flags.
        var deformBindings: [(d: SurfDeformDispatch, positions: MTLBuffer, vertexCount: Int)] = []
        if deformingActive {
            guard scs.surfReframeDeformPipeline != nil, scs.surfGPUDiffDeformIdxBuffer != nil
            else { return false }
            for d in scs.surfGPUDiffDeformDispatches {
                guard d.instanceIndex < instances.count,
                      let task = fxs.gpuRepackTasks.first(where: { $0.kind == d.kind })
                else { return false }
                deformBindings.append((d, task.positionBuffer, task.vertexCount))
            }
        }
        encodeGPUDiffReframe(cb, pipeline: pipe, threadCount: scs.surfGPUDiffTriCount,
                             dirtyBuf: dirtyBuf, cardBuf: cardBuf,
                             ownerBuf: ownerBuf, statsBuf: statsBuf,
                             rigidEnabled: surfaceCacheIncremental,
                             triCardBuf: triCardBuf, triFlagsBuf: flagsBuf,
                             triVertsBuf: vertsBuf, tag: "[gpu]")
        if !deformBindings.isEmpty {
            encodeGPUDiffDeform(cb, bindings: deformBindings,
                                dirtyBuf: dirtyBuf, cardBuf: cardBuf, statsBuf: statsBuf)
        }
        return true
    }

    /// Phase D — encode the deforming-instance re-frames: one dispatch per
    /// deforming instance, threads = its local soup triangles, reading the
    /// solver's LIVE packed_float3 position buffer through the baked
    /// vertex-index triples. Runs after the rigid kernel (which skips
    /// deform-flagged triangles, so this pass solely owns these cards).
    private func encodeGPUDiffDeform(_ cb: MTLCommandBuffer,
                                     bindings: [(d: SurfDeformDispatch, positions: MTLBuffer, vertexCount: Int)],
                                     dirtyBuf: MTLBuffer, cardBuf: MTLBuffer,
                                     statsBuf: MTLBuffer) {
        guard let pipeline = scs.surfReframeDeformPipeline,
              let idxBuf = scs.surfGPUDiffDeformIdxBuffer,
              let triCardBuf = scs.surfTriCardBuffer,
              let flagsBuf = scs.surfGPUDiffTriFlagsBuffer else { return }
        guard let enc = timedComputeEncoder(cb, "surfcacheReframeDeform") else { return }
        enc.label = "Illuminatorama.surfcache.gpuDiff.deform"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(cardBuf, offset: 0, index: 0)
        enc.setBuffer(dirtyBuf, offset: 0, index: 1)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setBuffer(triCardBuf, offset: 0, index: 3)
        enc.setBuffer(flagsBuf, offset: 0, index: 4)
        enc.setBuffer(idxBuf, offset: 0, index: 5)
        enc.setBuffer(statsBuf, offset: 0, index: 8)
        for b in bindings {
            var u = SCDeformUniforms(triBase: UInt32(b.d.triBase),
                                     triCount: UInt32(b.d.triCount),
                                     instanceIndex: UInt32(b.d.instanceIndex),
                                     vertexCount: UInt32(b.vertexCount))
            enc.setBuffer(b.positions, offset: 0, index: 6)
            enc.setBytes(&u, length: MemoryLayout<SCDeformUniforms>.stride, index: 7)
            dispatch1D(enc, pipeline: pipeline, count: b.d.triCount)
        }
        enc.endEncoding()
    }

    /// Encode the GPU-side moved-set detection + re-frame (#60 item 1). One
    /// dispatch; the surface-cache update kernel later in the same command
    /// buffer reads the re-framed cards + dirty flags (hazard tracking orders
    /// the two). Also drains this ring slot's atomic stats into the diagnostic
    /// sidecar — the slot was last written `maxFramesInFlight` frames ago, and the
    /// in-flight semaphore guarantees that frame has completed (DH-0537 #1).
    /// DH-0537 #1 — before a CPU write into a buffer the PREVIOUS frame's GPU passes may still
    /// be reading. Under `maxFramesInFlight == 2` that frame can be executing while this frame's
    /// CPU encode runs, so an in-place rewrite of `surfCardBuffer` / `surfCardDirtyBuffer` /
    /// `texelCard` / `cardRect` / `cardRequested` was a read-during-write hazard (the comments
    /// said "SAFE only while maxFramesInFlight == 1", and the pool had been 2 for a long time).
    /// Waiting drains the pipeline for that one frame, so only the paths that touch GPU-owned
    /// memory from the CPU pay it: the `VIZ_SURFCACHE_GPU_DIFF=0` fallback loops, the chart
    /// CPU re-frame, streaming residency and the feedback drain. The default GPU-diff lane
    /// writes those buffers from kernels on THIS frame's command buffer, which hazard
    /// tracking orders, and never calls this.
    func awaitInFlightFrameBeforeCPUCacheWrite() {
        gbs.lastCommittedFrameCB?.waitUntilCompleted()   // gpu-ok: opt-in CPU cache paths only, never the default per-frame lane
    }

    private func encodeGPUDiffReframe(_ cb: MTLCommandBuffer,
                                      pipeline: MTLComputePipelineState,
                                      threadCount: Int,
                                      dirtyBuf: MTLBuffer, cardBuf: MTLBuffer,
                                      ownerBuf: MTLBuffer, statsBuf: MTLBuffer,
                                      rigidEnabled: Bool = true,
                                      chartFrameBuf: MTLBuffer? = nil,
                                      triCardBuf: MTLBuffer? = nil,
                                      triFlagsBuf: MTLBuffer? = nil,
                                      triVertsBuf: MTLBuffer? = nil,
                                      tag: String) {
        guard threadCount > 0 else { return }
        let stats = statsBuf.contents().bindMemory(to: UInt32.self, capacity: 4)
        if stats[0] > 0 || stats[1] > 0 || stats[2] > 0 {
            Self.recordSurfCacheStats("dynamic: moved=\(stats[0]) deformed=\(stats[2]) dirtiedCards=\(stats[1]) of \(scs.surfCardCount) \(tag)")
        }
        stats[0] = 0; stats[1] = 0; stats[2] = 0; stats[3] = 0
        var u = SCReframeUniforms(count: UInt32(threadCount),
                                  cardCount: UInt32(scs.surfCardCount),
                                  instanceCount: UInt32(instances.count),
                                  epsilon: 1e-5,   // matrix-diff threshold, mirrors the CPU loop
                                  rigidEnabled: rigidEnabled ? 1 : 0)
        guard let enc = timedComputeEncoder(cb, "surfcacheReframe") else { return }
        enc.label = "Illuminatorama.surfcache.gpuDiff"
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(cardBuf, offset: 0, index: 0)
        enc.setBuffer(dirtyBuf, offset: 0, index: 1)
        enc.setBuffer(currentInstanceBuffer, offset: 0, index: 2)
        enc.setBuffer(previousInstanceBuffer, offset: 0, index: 3)
        enc.setBuffer(ownerBuf, offset: 0, index: 4)
        if let triCardBuf, let triFlagsBuf, let triVertsBuf {
            enc.setBuffer(triCardBuf, offset: 0, index: 5)
            enc.setBuffer(triFlagsBuf, offset: 0, index: 6)
            enc.setBuffer(triVertsBuf, offset: 0, index: 7)
            enc.setBytes(&u, length: MemoryLayout<SCReframeUniforms>.stride, index: 8)
            enc.setBuffer(statsBuf, offset: 0, index: 9)
        } else if let chartFrameBuf {
            enc.setBuffer(chartFrameBuf, offset: 0, index: 5)
            enc.setBytes(&u, length: MemoryLayout<SCReframeUniforms>.stride, index: 6)
            enc.setBuffer(statsBuf, offset: 0, index: 7)
        }
        dispatch1D(enc, pipeline: pipeline, count: threadCount)
        enc.endEncoding()
    }

    /// Re-light the on-surface radiance atlas (ping-pong) before the RT pass.
    /// Reads PREVIOUS atlas for the indirect bounce (→ multi-bounce over frames);
    /// writes CURRENT. The RT pass then samples CURRENT at its hits.
    func encodeSurfaceCacheUpdate(_ cb: MTLCommandBuffer) {
        guard surfCacheActive,
              let cards = scs.surfCardBuffer, let tc = scs.surfTriCardBuffer,
              let ta = scs.surfTriUVaBuffer, let tcc = scs.surfTriUVcBuffer,
              let crect = scs.surfCardRectBuffer, let txcard = scs.surfTexelCardBuffer,
              let cdirty = scs.surfCardDirtyBuffer else { return }
        // §3 endpoint — trace the per-frame-refit TLAS when one is live (a moved
        // object's shadow/bounce-occlusion then tracks its motion); otherwise fall
        // back to the static-soup primitive AS (the soup-path scenes, no TLAS).
        let useTLAS = rts.rtTLASActive && scs.surfCacheTLASPipeline != nil
            && rts.rtTLAS != nil && rts.rtSoupTriBaseBuffer != nil
            && rts.rtSoupTriBaseCount == instances.count
        let pipeline: MTLComputePipelineState
        let accel: MTLAccelerationStructure
        if useTLAS {
            // Curve primitives (#60 item 7): a TLAS holding curve instances
            // must be traced with the curve-contract variant (curve hits then
            // occlude sun rays and terminate indirect rays — see the kernel).
            if rts.rtCurveInstanceCount > 0 {
                guard let cp = scs.surfCacheTLASCurvePipeline else { return }
                pipeline = cp
            } else {
                pipeline = scs.surfCacheTLASPipeline!
            }
            accel = rts.rtTLAS!
        } else {
            guard let soupAccel = rts.rtAccel else { return }
            // Curve primitives (#60 item 7 incr. 2): a soup AS holding a curve
            // geometry descriptor must be traced with the curve-contract variant
            // (curve hits occlude the sun ray + terminate indirect rays).
            if rts.rtSoupCurvesActive, let cp = scs.surfCacheSoupCurvePipeline {
                pipeline = cp
            } else if let soupPipe = scs.surfCachePipeline {
                pipeline = soupPipe
            } else { return }
            accel = soupAccel
        }
        // Honour per-card dirty flags (α=1 → re-light fresh at the re-framed pose)
        // for either dynamic source: rigid incremental invalidation OR Phase-C
        // deforming meshes (whose cards `applyIncrementalSurfaceCacheUpdate`
        // re-frames every frame). Both require the per-triangle CPU maps.
        let incremental = (surfaceCacheIncremental || scs.surfHasDeformingCards) && scs.surfIncrementalReady

        // DH-0622 — WARM-UP. A freshly baked (or freshly enabled) atlas holds no light, and every
        // bounce builds on the previous update's atlas, so at the normal blend a still's first frames
        // carry almost none of the multi-bounce light — measured cold on 4000 Sunset: +2.7 % after one
        // 32-frame export against a ~+22 % level-off. So the frame after a bake or an enable runs
        // `surfaceCacheWarmIterations` extra updates at α = 1 first: each carries the light one bounce
        // further, and the normal update then resumes smoothing from a filled atlas.
        let warm = scs.surfCacheWarmPending
        scs.surfCacheWarmPending = 0

        // DH-0653 — atlas reuse counter: report the previous frame's, arm this frame's final update.
        var atlasStats: MTLBuffer? = nil
        if Self.surfCacheStatsRequested {
            scs.surfAtlasStatsBuffer = surfCounterBuffer(scs.surfAtlasStatsBuffer, slots: 2,
                                                     label: "Illuminatorama.surfcache.atlasStats")
            if let sb = scs.surfAtlasStatsBuffer {
                let c = drainSurfCounters(sb, slots: 2)
                if let line = Self.surfCacheAtlasLine(reused: c[0], refreshed: c[1]) {
                    Self.recordSurfCacheStats(line)
                }
                atlasStats = sb
            }
        }

        for pass in 0...warm {
            let warming = pass < warm
            // Ping-pong swap at the START (mirrors DDGI): Current is what we write
            // AND what the RT pass reads after we return; Previous is the bounce
            // source. First frame after (re)alloc both are empty → indirect = 0.
            scs.surfUseAtlasA.toggle()
            guard let curAtlas = surfCacheCurrentAtlas,
                  let prevAtlas = surfCachePreviousAtlas,
                  let curIndirect = surfCacheCurrentIndirectAtlas,
                  let prevIndirect = surfCachePreviousIndirectAtlas else { return }

            scs.surfFrameSeed &+= 1
            var u = SurfCacheUniforms(
                sunDir: SIMD4(simd_normalize(rtSunDirection), 0),
                sunColor: SIMD4(rtSunColor, 0),
                skyAmbient: SIMD4(ambientColor, 0),
                atlasW: UInt32(scs.surfAtlasW), atlasH: UInt32(scs.surfAtlasH),
                tileSize: UInt32(scs.surfActiveTileSize), tilesPerRow: UInt32(scs.surfTilesPerRow),
                cardCount: UInt32(scs.surfCardCount), triangleCount: UInt32(rts.rtTriangleCount),
                indirectRays: UInt32(max(1, min(16, surfCacheIndirectRays))),
                frameSeed: scs.surfFrameSeed,
                alpha: warming ? 1.0 : max(0.02, min(1.0, surfCacheAlpha)), rayTMin: 0.004, maxDist: 60.0,
                incrementalEnabled: incremental ? 1 : 0)
            if !warming, atlasStats != nil { u.statsEnabled = 1 }
            u.backFaceGuard = surfaceCacheBackFaceGuard ? 1 : 0
            // DH-0872 — same coefficient the tonemap pass uses, applied to this pass's own
            // raw sky-miss sample (see the kernel's miss branch).
            u.scotopicDesaturation = max(0, scotopicDesaturation)
            u.cardlessReshade = surfaceCacheCardlessReshade ? 1 : 0
            memcpy(scs.surfCacheUniformBuffer.contents(), &u, MemoryLayout<SurfCacheUniforms>.stride)

            // Warm passes go untimed: the per-pass GPU timer holds 48 passes a frame.
            guard let enc = warming ? cb.makeComputeCommandEncoder() : timedComputeEncoder(cb, "surfcacheUpdate")
            else { return }
            enc.label = warming ? "Illuminatorama.surfcache.warm" : "Illuminatorama.surfcache"
            enc.setComputePipelineState(pipeline)
            enc.setTexture(curAtlas, index: 0)
            enc.setTexture(prevAtlas, index: 1)   // read access
            enc.setTexture(prevAtlas, index: 2)   // sample access
            enc.setTexture(equirectSky ?? ibls.dummySkyTexture, index: 3)
            enc.setTexture(curIndirect, index: 4)    // DH-0622 — write: the indirect-only EMA
            enc.setTexture(prevIndirect, index: 5)   // read
            enc.setAccelerationStructure(accel, bufferIndex: 0)
            enc.setBuffer(cards, offset: 0, index: 1)
            enc.setBuffer(tc, offset: 0, index: 2)
            enc.setBuffer(ta, offset: 0, index: 3)
            enc.setBuffer(tcc, offset: 0, index: 4)
            // Bytes per pass, not the shared uniform buffer: a buffer is read when the command buffer
            // EXECUTES, so every pass encoded into this frame would see the last pass's α and seed.
            enc.setBytes(&u, length: MemoryLayout<SurfCacheUniforms>.stride, index: 5)
            enc.setBuffer(crect, offset: 0, index: 6)
            enc.setBuffer(txcard, offset: 0, index: 7)
            enc.setBuffer(cdirty, offset: 0, index: 8)
            // DH-0653 counters (buffer 10, both variants). Dummy = the dirty buffer, which
            // the kernel never adds to while `statsEnabled` is 0.
            enc.setBuffer(atlasStats ?? cdirty, offset: 0, index: 10)
            if useTLAS {
                // TLAS hit → global soup tri via soupTriBase[instance_id]+primitive_id.
                // The TLAS references the BLASes, which reference the mesh vertex/index
                // buffers (and the curve BLASes the pooled curve buffers) — all must
                // be resident for the intersector.
                enc.setBuffer(rts.rtSoupTriBaseBuffer, offset: 0, index: 9)
                // DH-1314 — a cardless hit is re-shaded from the RT instance rows + object normals.
                enc.setBuffer(rts.rtInstanceDataBuffer ?? cdirty, offset: 0, index: 11)
                enc.setBuffer(rts.rtObjNormalBuffer ?? cdirty, offset: 0, index: 12)
                for blas in rts.rtBLASList { enc.useResource(blas, usage: .read) }
                for blas in rts.rtCurveBLASList { enc.useResource(blas, usage: .read) }
                for buf in rts.rtResidentBuffers { enc.useResource(buf, usage: .read) }
            } else if let vb = rts.rtVertexBuffer, let ib = rts.rtIndexBuffer {
                // Curve-only primitive AS at buffer(9) — the SOUP kernel's slot (the
                // TLAS variant uses 9 for soupTriBase; different pipeline). Dummy =
                // the triangle `accel` for the base (curve-free) variant.
                enc.setAccelerationStructure((rts.rtSoupCurvesActive ? rts.rtSoupCurveAccel : nil) ?? accel, bufferIndex: 9)
                enc.useResource(vb, usage: .read)
                enc.useResource(ib, usage: .read)
                // Curve AS + pool buffers the soup curve trace references (#60 item 7 incr. 2).
                if rts.rtSoupCurvesActive {
                    if let ca = rts.rtSoupCurveAccel { enc.useResource(ca, usage: .read) }
                    if let p = rts.rtSoupCurvePoolPoints { enc.useResource(p, usage: .read) }
                    if let r = rts.rtSoupCurveRadii { enc.useResource(r, usage: .read) }
                    if let s = rts.rtSoupCurveSegments { enc.useResource(s, usage: .read) }
                }
            }
            dispatch(enc, pipeline: pipeline, width: scs.surfAtlasW, height: scs.surfAtlasH)
            enc.endEncoding()
        }
    }

    // Phase 5 / B1 — cache-domain à-trous denoiser. Mirror of the Metal
    // `SurfAtrousUniforms` (stride 32).
    private struct SurfAtrousUniforms {
        var atlasW: UInt32; var atlasH: UInt32; var stepSize: UInt32; var radius: UInt32
        var phi: Float; var _ap0: Float = 0; var _ap1: Float = 0; var _ap2: Float = 0
    }

    /// Phase 5 / B1 — spatially denoise the just-updated radiance atlas into
    /// `surfAtlasDenoised`, which the GI/reflection consumers then read. Encoded
    /// AFTER `encodeSurfaceCacheUpdate` (current atlas written) and BEFORE the RT
    /// lighting pass (same command buffer ⇒ compute passes serialize). No-op when
    /// the denoiser is off, the pipeline is unavailable, or the textures aren't
    /// allocated — consumers fall back to the raw atlas via `surfConsumerAtlas`.
    func encodeSurfaceCacheDenoise(_ cb: MTLCommandBuffer) {
        guard surfaceCacheDenoise,
              let pipeline = scs.surfAtrousPipeline,
              let src = surfCacheCurrentIndirectAtlas,
              let dst = scs.surfAtlasDenoised,
              let txcard = scs.surfTexelCardBuffer,
              scs.surfCardCount > 0 else { return }
        guard let enc = timedComputeEncoder(cb, "surfcacheDenoise") else { return }
        enc.label = "Illuminatorama.surfcache.atrous"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(dst, index: 0)   // write
        enc.setTexture(src, index: 1)   // read (current atlas)
        enc.setBuffer(txcard, offset: 0, index: 0)
        var u = SurfAtrousUniforms(
            atlasW: UInt32(scs.surfAtlasW), atlasH: UInt32(scs.surfAtlasH),
            stepSize: 1, radius: 2, phi: 4.0)
        enc.setBytes(&u, length: MemoryLayout<SurfAtrousUniforms>.stride, index: 1)
        dispatch(enc, pipeline: pipeline, width: scs.surfAtlasW, height: scs.surfAtlasH)
        enc.endEncoding()
    }
}

// MARK: - SurfaceCacheState

/// Surface cache (Lumen-style card atlas): cards, atlases, charts, GPU diff buffers.
/// Owned by the renderer as `scs`; every member keeps the name it had on the renderer.
@MainActor
final class SurfaceCacheState {
    var surfCachePipeline: MTLComputePipelineState?
    /// §3 endpoint — TLAS-traced cache-update (traces the per-frame-refit instance
    /// AS so a moved object's shadow/bounce-occlusion tracks). Used in place of
    /// `surfCachePipeline` when a TLAS is live (`rtTLASActive`); the soup path keeps
    /// the primitive-AS `surfCachePipeline`.
    var surfCacheTLASPipeline: MTLComputePipelineState?
    var surfCardBuffer: MTLBuffer?

    var surfTriCardBuffer: MTLBuffer?  // uint per triangle → card index
    var surfTriUVaBuffer: MTLBuffer?  // float4 (uvA.xy, uvB.xy)
    var surfTriUVcBuffer: MTLBuffer?  // float4 (uvC.xy, 0, 0)
    // P2 — per-card atlas RECT addressing (replaces the uniform tilesPerRow×tileSize
    // grid). `cardRect[card]` = (atlasX, atlasY, tileW, tileH) in atlas pixels;
    // `texelCard[y·atlasW + x]` = the card owning that texel (0xFFFFFFFF = gap).
    // Lets charts of different world extent get different-sized tiles, shelf-packed
    // into one atlas — the uniform grid is just the special case (all tiles equal).
    var surfCardRectBuffer: MTLBuffer?  // float4 per card (x,y,w,h px)
    var surfTexelCardBuffer: MTLBuffer?  // uint per atlas texel → card index

    var surfAtrousPipeline: MTLComputePipelineState?

    var surfCardRequestedBuffer: MTLBuffer?  // uint per card (1 = sampled this frame)

    var surfResidentCardCount: Int = 0   // ≤ surfCardCount; == it when budget disabled

    var surfSlotRect: [SIMD4<Float>] = []   // per slot: atlas rect (bake-time resident-card rects)
    var surfSlotCard: [Int] = []            // slot → occupant card index
    var surfCardSlot: [Int] = []            // card → slot (−1 = non-resident)
    let surfMaxSwapsPerFrame = 256          // bound the per-frame texelCard rewrite + anti-thrash
    // Phase 5 / A2 — eviction hysteresis. Per-card "heat": set to `surfStreamHeatFull`
    // when sampled, decremented each frame otherwise. A resident card is demotable only
    // once its heat hits 0 (unsampled for HeatFull consecutive frames), so a card
    // sampled intermittently (grazing rays, the moving box's trail) stays resident
    // instead of flipping in/out every frame — cuts the steady-state swap churn.
    var surfCardHeat: [Int] = []
    let surfStreamHeatFull =
        Int(ProcessInfo.processInfo.environment["VIZ_ILLUMI_SURFCACHE_HEAT"] ?? "") ?? 4

    var surfCardCentroid: [SIMD3<Float>] = []   // bake-time world centroid per card
    var surfCardDirtyBuffer: MTLBuffer?  // uint per card (1 = re-light fresh)
    var surfTriCardCPU: [UInt32] = []       // triCard mirror (per soup triangle)
    var surfTriIsFrameB: [Bool] = []        // per triangle: B-half of its packed card?
    var surfSoupBaseCPU: [UInt32] = []       // per instance: first soup-triangle index
    var surfInstanceKind: [IlluminatoramaRenderer.MeshKind] = []    // per instance: mesh kind (for object soup)
    var surfObjectSoupByKind: [IlluminatoramaRenderer.MeshKind: (positions: [SIMD3<Float>], indices: [UInt32])] = [:]
    var surfIncrementalReady: Bool = false   // CPU maps valid for this topology

    var surfChartIncremental: IlluminatoramaRenderer.ChartIncremental?
    // Reused per-frame scratch for the chart incremental path's "which instances
    // moved" pass — sized once at bake time (when instance count is known) and
    // reset in place each frame, so the steady-state loop allocates nothing.
    var surfChartMovedScratch: [Bool] = []

    var surfReframeTriPipeline: MTLComputePipelineState?
    var surfReframeChartPipeline: MTLComputePipelineState?
    var surfReframeDeformPipeline: MTLComputePipelineState?
    var surfGPUDiffTriOwnerBuffer: MTLBuffer?  // uint per soup tri → instance
    var surfGPUDiffTriFlagsBuffer: MTLBuffer?  // uint per soup tri: SC_FLAG_* bits
    var surfGPUDiffObjVertsBuffer: MTLBuffer?  // float4 ×3 per soup tri (object space)
    var surfGPUDiffTriCount: Int = 0
    var surfGPUDiffChartOwnerBuffer: MTLBuffer?  // int32 per card (−1 = static)
    var surfGPUDiffChartFrameBuffer: MTLBuffer?  // float4 ×4 per card (object frame)
    // ── Phase D — deforming meshes, GPU-resident ──
    /// Vertex-index triple per GLOBAL soup triangle (topology constant per bake);
    /// the deform kernel reads these + the LIVE packed_float3 position buffer.
    var surfGPUDiffDeformIdxBuffer: MTLBuffer?

    var surfGPUDiffDeformDispatches: [IlluminatoramaRenderer.SurfDeformDispatch] = []
    /// 4×uint written by the kernels via relaxed atomics ([0] moved instances,
    /// [1] dirtied cards, [2] deformed instances, [3] spare), read + zeroed by the CPU
    /// for the `recordSurfCacheStats` sidecar. RING-buffered (DH-0537 #1): with
    /// `maxFramesInFlight == 2` the previous frame's kernel can still be incrementing while
    /// this frame's CPU reads and zeroes, so one buffer was a read-during-write race. A slot
    /// is reused every `maxFramesInFlight` frames, and the in-flight semaphore guarantees the
    /// frame that last wrote it has COMPLETED before this frame's encode begins.
    var surfGPUDiffStatsRing: [MTLBuffer] = []

    // Phase C — a deforming (DynamicMesh / GPU-fed) mesh contributed surface cards
    // this topology. Its cards are re-framed from LIVE vertices every frame
    // (`objectTriangleSoup()` is empty for its private vertex buffer, so it's
    // otherwise excluded from the cache), and the update kernel must honour the
    // per-card dirty flags (α=1) just as for rigid incremental invalidation.
    var surfHasDeformingCards: Bool = false
    /// Dev A/B override (`VIZ_ILLUMI_NO_DEFORM_CARDS=1`) — freeze deforming cards
    /// at their build pose (pre-Phase-C behaviour) so the catch/cast win is
    /// directly comparable. Off by default; the re-frame is on whenever a
    /// deforming mesh drives the cache.
    let noDeformCardsOverride =
        ProcessInfo.processInfo.environment["VIZ_ILLUMI_NO_DEFORM_CARDS"] == "1"
    var surfAtlasA: MTLTexture?
    var surfAtlasB: MTLTexture?
    /// DH-0622 — the INDIRECT term alone, ping-ponged in lockstep with the full atlas. The
    /// full atlas (direct + indirect) is only the update kernel's multi-bounce feedback; every
    /// consumer (GI, reflection, glass) shades its hit itself and reads only this.
    var surfIndirectAtlasA: MTLTexture?
    var surfIndirectAtlasB: MTLTexture?
    // Phase 5 / B1 — cache-domain denoiser output. NOT ping-ponged: recomputed
    // each frame from the just-updated current INDIRECT atlas (display-only; the EMA
    // feedback keeps reading the raw atlases). Bound to the GI/reflection consumers
    // in place of the current indirect atlas when `surfaceCacheDenoise` is on.
    var surfAtlasDenoised: MTLTexture?
    var surfUseAtlasA: Bool = true
    var surfCacheUniformBuffer: MTLBuffer!
    var surfCardCount: Int = 0
    /// Default per-card atlas tile size — sized for the room's big planar cards.
    var surfTileSize: Int = 64
    /// Tile size of the CURRENTLY-registered card set. Per-triangle micro-cards
    /// (Phase 4.38) pass a much smaller tile (a triangle doesn't need 64²), so
    /// the atlas stays bounded; the room keeps 64. Set by `setSurfaceCacheCards`.
    var surfActiveTileSize: Int = 64
    var surfTilesPerRow: Int = 1
    var surfAtlasW: Int = 1
    var surfAtlasH: Int = 1
    var surfFrameSeed: UInt32 = 0
    /// The (TLAS topology, triangle budget) the cards were last baked — or skipped — for. Lets a
    /// host turn the cache on over an already-built TLAS and get cards on the next frame, without
    /// an over-budget scene re-attempting the bake every frame (DH-0622).
    var surfCacheBakeKey: Int?
    /// Warm-up updates still owed by the latest bake or enable (DH-0622) — see `surfaceCacheWarmIterations`.
    var surfCacheWarmPending = 0

    var surfCacheTLASCurvePipeline: MTLComputePipelineState?

    var surfCacheSoupCurvePipeline: MTLComputePipelineState?

    // ── DH-0653: surface-cache hit/miss + atlas reuse counters ────────────────
    //
    // Two atomic counter buffers the kernels add to only while the stats sidecar is
    // set: `surfHitStatsBuffer` (8 slots, `illumi_rt_lighting_tlas` — what each GI /
    // reflection triangle hit was served by) and `surfAtlasStatsBuffer` (2 slots, the
    // update kernel — texels EMA-reused vs refreshed at α = 1). Each is drained at the
    // NEXT frame's encode, after the in-flight frame completes, and written as one
    // `hits:` / `atlas:` line. The DH-0622 trade ("cheaper than re-shade" → "richer
    // than re-shade") is the `cached` share of the `hits:` line.
    var surfHitStatsBuffer: MTLBuffer?
    var surfAtlasStatsBuffer: MTLBuffer?
}

// MARK: - Code moved out of the main renderer file

extension IlluminatoramaRenderer {
    // ── Surface cache (Lumen-style on-surface lit-radiance atlas) ────
    /// Mirror of `SurfCard` in IlluminatoramaSurfaceCache.metal — one atlas tile.
    /// A tile carries up to TWO planar triangle frames (P3 — 2-triangles-per-tile
    /// packing): frame A occupies the lower-left half (texel `u+v ≤ 1`), frame B
    /// the upper-right half (`u+v > 1`), split on the diagonal. `normal.w` is the
    /// PACKED flag: 1 ⇒ the tile is split between frame A and frame B; 0 ⇒ frame A
    /// fills the WHOLE tile (the room's hand-authored quad cards, where both of a
    /// quad's triangles map across the full tile, and a lone odd per-triangle
    /// card). All `SIMD4` (no SIMD3/packed_float3 stride trap).
    public struct SurfCard {
        // Frame A — lower-left half when packed, whole tile when unpacked.
        public var origin: SIMD4<Float>, uAxis: SIMD4<Float>, vAxis: SIMD4<Float>
        public var normal: SIMD4<Float>, albedo: SIMD4<Float>, emission: SIMD4<Float>
        // Frame B — upper-right half; meaningful only when `normal.w == 1`.
        public var originB: SIMD4<Float>, uAxisB: SIMD4<Float>, vAxisB: SIMD4<Float>
        public var normalB: SIMD4<Float>, albedoB: SIMD4<Float>, emissionB: SIMD4<Float>
        /// Unpacked single-frame card — frame A fills the whole tile (room quads,
        /// lone odd per-triangle card). Frame B left zero, packed flag (normal.w) 0.
        public init(origin: SIMD3<Float>, uAxis: SIMD3<Float>, vAxis: SIMD3<Float>,
                    normal: SIMD3<Float>, albedo: SIMD3<Float>, emission: SIMD3<Float>) {
            self.origin = SIMD4(origin, 0); self.uAxis = SIMD4(uAxis, 0)
            self.vAxis = SIMD4(vAxis, 0); self.normal = SIMD4(simd_normalize(normal), 0)
            self.albedo = SIMD4(albedo, 0); self.emission = SIMD4(emission, 0)
            self.originB = .zero; self.uAxisB = .zero; self.vAxisB = .zero
            self.normalB = .zero; self.albedoB = .zero; self.emissionB = .zero
        }
        /// Packed two-triangle card (P3): frame A in the lower-left half, frame B
        /// in the upper-right half. `normal.w = 1` flags the diagonal split.
        public init(aOrigin: SIMD3<Float>, aUAxis: SIMD3<Float>, aVAxis: SIMD3<Float>,
                    aNormal: SIMD3<Float>, aAlbedo: SIMD3<Float>, aEmission: SIMD3<Float>,
                    bOrigin: SIMD3<Float>, bUAxis: SIMD3<Float>, bVAxis: SIMD3<Float>,
                    bNormal: SIMD3<Float>, bAlbedo: SIMD3<Float>, bEmission: SIMD3<Float>) {
            self.origin = SIMD4(aOrigin, 0); self.uAxis = SIMD4(aUAxis, 0)
            self.vAxis = SIMD4(aVAxis, 0); self.normal = SIMD4(simd_normalize(aNormal), 1)
            self.albedo = SIMD4(aAlbedo, 0); self.emission = SIMD4(aEmission, 0)
            self.originB = SIMD4(bOrigin, 0); self.uAxisB = SIMD4(bUAxis, 0)
            self.vAxisB = SIMD4(bVAxis, 0); self.normalB = SIMD4(simd_normalize(bNormal), 0)
            self.albedoB = SIMD4(bAlbedo, 0); self.emissionB = SIMD4(bEmission, 0)
        }
    }

    /// Mirror of `SurfCacheUniforms` in IlluminatoramaSurfaceCache.metal.
    struct SurfCacheUniforms {
        var sunDir: SIMD4<Float>; var sunColor: SIMD4<Float>; var skyAmbient: SIMD4<Float>
        var atlasW: UInt32; var atlasH: UInt32; var tileSize: UInt32; var tilesPerRow: UInt32
        var cardCount: UInt32; var triangleCount: UInt32; var indirectRays: UInt32; var frameSeed: UInt32
        var alpha: Float; var rayTMin: Float; var maxDist: Float; var incrementalEnabled: UInt32 = 0
        // DH-0653 — atlas reuse counter gate; DH-0849 — back-face guard (+ pad to the Metal stride).
        var statsEnabled: UInt32 = 0; var backFaceGuard: UInt32 = 0
        /// DH-0872 — scotopic desaturation for this pass's own GI sky-MISS sample (the
        /// cache-update kernel gathers indirect independently of the RT lighting kernels'
        /// GI, and has the SAME raw-sample leak). Mirrors `scotopicDesaturation`. Repurposes
        /// `_scPad1` — same 4 bytes. 0 (day) ⇒ byte-identical.
        var scotopicDesaturation: Float = 0
        /// DH-1314 — mirrors `surfaceCacheCardlessReshade` (repurposes `_scPad2`, same 4 bytes).
        var cardlessReshade: UInt32 = 1
    }

    // ── Chart-path incremental invalidation (#60 task 1) ──
    // The per-triangle path (above) re-frames each moved triangle from its object
    // soup × model. A coplanar CHART card (`VIZ_SURFCACHE_CHARTS=1`) instead spans
    // many triangles with ONE bbox-derived frame + fixed per-triangle UVs, so the
    // per-triangle re-frame would clobber the chart's parameterisation. But a card
    // frame is AFFINE-COVARIANT: under a rigid/affine model matrix `M`, the world
    // frame is exactly `origin' = M·origin` (point), `uAxis'/vAxis' = M·axis`
    // (vector), and the per-triangle UVs (`uv = (proj(v)−bboxMin)/bboxSize`) are
    // invariant because verts and frame transform together. So a moved chart card
    // re-frames with a single affine transform of its BAKE-pose object-space frame
    // (`M_bake⁻¹ · worldFrame`), with no per-triangle work and no UV change. Built
    // only on the chart bake path; nil ⇒ per-triangle path (or no incremental).
    struct SurfCardFrame {
        var origin = SIMD3<Float>(0, 0, 0), uAxis = SIMD3<Float>(0, 0, 0)
        var vAxis = SIMD3<Float>(0, 0, 0), normal = SIMD3<Float>(0, 1, 0)
    }

    struct ChartIncremental {
        var objectFrame: [SurfCardFrame]   // per card: bake-pose frame in owner's object space
        var instCardLo: [Int32]            // per instance: first owned card (−1 if none)
        var instCardHi: [Int32]            // per instance: one-past-last owned card
    }

    // ── GPU-side transform diff (#60 item 1) ──
    // Moves the per-frame moved-set detection + card re-frame off the CPU: a
    // compute kernel on the frame's command buffer diffs each card's owner
    // instance (current vs previous model matrix) and re-frames moved cards in
    // place, replacing the CPU loops in `applyIncrementalSurfaceCacheUpdate` /
    // `applyIncrementalChartSurfaceCacheUpdate` for RIGID movers. The CPU path
    // stays as the fallback for Phase-C deforming scenes (their re-frame reads
    // the live soup — GPU-ifying that is the separate Phase D item) and as the
    // `VIZ_SURFCACHE_GPU_DIFF=0` kill-switch / A/B reference. The maps below are
    // baked alongside the CPU ones and cleared with them on topology change.
    static let surfCacheGPUDiffDisabled =
        ProcessInfo.processInfo.environment["VIZ_SURFCACHE_GPU_DIFF"] == "0"

    /// One entry per deforming INSTANCE: the per-dispatch parameters. The
    /// position buffer is looked up per encode from `gpuRepackTasks` by kind
    /// (the solver may re-register; the kind survives the topology bake).
    struct SurfDeformDispatch {
        var kind: MeshKind; var instanceIndex: Int; var triBase: Int; var triCount: Int
    }

    /// Swift mirror of the Metal `SCReframeUniforms` (32 B, passed via setBytes).
    struct SCReframeUniforms {
        var count: UInt32; var cardCount: UInt32; var instanceCount: UInt32; var epsilon: Float
        var rigidEnabled: UInt32; var _pad0: UInt32 = 0; var _pad1: UInt32 = 0; var _pad2: UInt32 = 0
    }

    /// Swift mirror of the Metal `SCDeformUniforms` (16 B, per deform dispatch).
    struct SCDeformUniforms {
        var triBase: UInt32; var triCount: UInt32; var instanceIndex: UInt32; var vertexCount: UInt32
    }

    /// Coplanar charts (`VIZ_SURFCACHE_CHARTS=1`) use a different, non-mesh-invariant
    /// UV parameterisation, so the BLAS UV fold is per-triangle-path only.
    static let surfCacheCharts =
        ProcessInfo.processInfo.environment["VIZ_SURFCACHE_CHARTS"] == "1"

    /// `soupTriBase` value of an instance whose mesh takes no cards (DH-1314). Every consumer
    /// resolves a TLAS hit as `soupTriBase[instance] + primitive`, bounds-checked against the
    /// soup's triangle count, so this reads "no card" everywhere; the cache-update kernel keys its
    /// cardless re-shade off it. Mirror of the Metal `SC_NO_CARD_SOUP_BASE`.
    static let surfaceCacheNoCardSoupBase: UInt32 = 0x8000_0000

    /// Six world-space face patches per instance, ready for
    /// `ZFightDetector.audit(_:requireSameFacing: true)`.
    ///
    /// This is the piece that was missing for native Illuminatorama scenes: the
    /// geometry auditors take `[SurfacePatch]`, and until now the only producer
    /// walked `SCNNode`s — which a native scene does not have. See
    /// `IlluminatoramaSurfaceAudit` for why it emits faces rather than objects.
    public func auditSurfacePatches(_ instances: [InstanceRef],
                                    name: ((Int, InstanceRef) -> String)? = nil) -> [SurfacePatch] {
        IlluminatoramaSurfaceAudit.facePatches(
            instances: instances.map { (kind: $0.meshKind,
                                        model: $0.data.modelMatrix,
                                        normalMatrix: $0.data.normalMatrix) },
            bounds: { [weak self] kind in self?.meshLocalBounds(kind) },
            name: { i in name?(i, instances[i]) ?? "\(instances[i].meshKind)#\(i)" })
    }

    /// The counters run exactly when the sidecar they report to exists. Read per frame
    /// (not cached) so a test can scope it with `setenv`/`unsetenv`.
    static var surfCacheStatsRequested: Bool {
        frameEnv("VIZ_ILLUMI_SURFCACHE_STATS_PATH") != nil
    }

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

// MARK: - Pipeline / resource construction (called from `init`, in the order it lists)

extension IlluminatoramaRenderer {
    /// Surface-cache update pipelines (RT-gated), uniforms, GPU-diff re-frame + denoise kernels and the stats ring.
    @MainActor static func buildSurfaceCachePipelines(into s: SurfaceCacheState, cache: SimPipelineCache, device: MTLDevice, library: MTLLibrary) throws {
        let curvesOff = Self.curvesOffConstants()
        // ── Surface-cache update pipeline (RT-gated) ─────────────────
        // Both kernels declare the kSCCurvesEnabled function constant (#60 item 7),
        // so both MUST be specialized via constantValues even for the base (false)
        // variant — Metal refuses a plain makeFunction(name:) pipeline for any
        // constant-declaring function. `curvesOff` was built above.
        var builtSurfCachePipeline: MTLComputePipelineState? = nil
        var builtSurfCacheTLASPipeline: MTLComputePipelineState? = nil
        if device.supportsRaytracing,
           let scFn = try? library.makeFunction(name: "illumi_surfcache_update",
                                                constantValues: curvesOff) {
            builtSurfCachePipeline = try? device.makeComputePipelineState(function: scFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        if device.supportsRaytracing,
           let scTLASFn = try? library.makeFunction(name: "illumi_surfcache_update_tlas",
                                                    constantValues: curvesOff) {
            builtSurfCacheTLASPipeline = try? device.makeComputePipelineState(function: scTLASFn) // gpu-ok: one-time init, RT-gated optional pipeline
        }
        s.surfCachePipeline = builtSurfCachePipeline
        s.surfCacheTLASPipeline = builtSurfCacheTLASPipeline
        guard let scUB = device.makeBuffer(length: MemoryLayout<SurfCacheUniforms>.stride,
                                            options: .storageModeShared) else {
            throw IlluminatoramaError.bufferAllocationFailed("surfCacheUniforms")
        }
        scUB.label = "Illuminatorama.surfcache.uniforms"
        s.surfCacheUniformBuffer = scUB
        // #60 item 1 + Phase D — GPU-side incremental-invalidation kernels (plain
        // compute, no RT dependency; nil leaves the CPU re-frame path in charge).
        s.surfReframeTriPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_tri", device: device)
        s.surfReframeChartPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_chart", device: device)
        s.surfReframeDeformPipeline =
            cache.pipelineState(name: "illumi_surfcache_reframe_deform", device: device)
        // Phase 5 / B1 — cache-domain denoiser (plain compute, no RT dependency;
        // nil leaves consumers reading the raw atlas).
        s.surfAtrousPipeline =
            cache.pipelineState(name: "illumi_surfcache_atrous", device: device)
        var statsRing: [MTLBuffer] = []
        for slot in 0..<IlluminatoramaRenderer.maxFramesInFlight {
            guard let b = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 4,
                                            options: .storageModeShared) else { continue }
            b.label = "Illuminatorama.surfcache.gpuDiffStats.\(slot)"
            memset(b.contents(), 0, b.length)
            statsRing.append(b)
        }
        s.surfGPUDiffStatsRing = statsRing
    }
}
