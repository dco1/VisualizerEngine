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
