import Foundation
import simd

// Trunk/branch revolution meshes, moss, ring stitching.
extension ForestTreeGeometry {
    // ──────────────────────────────────────────────────────────────────────
    // MARK: - Tree mesh primitives (ported, soup-emitting)
    // ──────────────────────────────────────────────────────────────────────

    /// Global multiplier on leaf node counts. The earlier 0.55 throttle left the
    /// crowns ~90% empty (sky visible straight through), which read as bare/dead
    /// saplings and starved the leaf-transmission feature of a backlit mass.
    /// `setRTGeometry` imposes no triangle cap and BVH traversal is ~log in tri
    /// count, so a dense static canopy is well within budget — a real broadleaf
    /// crown should be light-occluding. Raised to give that. Tunable in review.
    // HANG FIX (#58): the 462-second main-thread stall (spindump confirmed)
    // traced to `emitLeafCard` → `Soup.tri` inside `growBranchSoup`. Root
    // causes: leafDensityScale=8.5 × leafDensityMult=7-9 × leafNodeCap=90
    // × ~350 emitLeaves calls/tree pushed mid-tier trees to ~40k leaves
    // each (40k × 16 tri = 650k tris × 45 mid-trees = 29 M tris).
    // Fix: slash the three multipliers so the budget lands ~5 M tris total
    // (still dense enough for a convincing canopy). leafDensityMult per
    // species is also cut above (7→2.5, 9→3.0, 6.5→2.2). Build is now
    // dispatched off-main via buildRawSoup()/buildMesh(from:device:), so
    // even if the count grows these constants will never block the UI.
    // Density restored after hang fix: the 4M triCap + off-main async build
    // mean higher density is now safe. 5.0 × leafNodeCap=60 gives ~3-4M
    // leaf triangles at these DensityMult values (well under the 4M hard cap).
    // The computation before tri() is fast (SIMD math, no array writes past
    // the cap) so this is <1 s build time on the detached task.
    //
    // ── D3: expressed at FULL density (`foliageScale` 1.0), like everything else the
    // knob touches. 16.667 × 0.30 = 5.0, the historical value at the generator's
    // historical default, so a bake at 0.30 is unchanged to the node.
    public static let leafDensityScaleAtFullDensity: Float = 16.6667
    // BUDGET FIX: reduced from 60 to 20 — root-cause analysis showed that
    // 60 nodes × 24-tri hero cards × ~356 emitLeaves calls/tree = ~510k tris
    // per hero tree; 8 hero trees exhaust the 4M (now 7M) triCap before any
    // backdrop/cap/wall trees get geometry → invisible treeline. At 20 nodes,
    // hero cost is ~170k/tree, leaving 5.6M for the rest of the forest.
    // 20 leaves per branch is sufficient for a convincing overlapping canopy;
    // the node spacing on a 2m secondary branch is ~18cm vs 10cm leaf width.
    //
    // ── PHOTOREALISM #9 / D3: this cap is now DENSITY-SCALED, and that is the
    // whole reason the twig knob was dead. `rawNodes` runs to the hundreds on a
    // real twig, so `min(cap, rawNodes)` was pinned to the cap at EVERY foliage
    // scale — the `site.leafDensity *= scale` the baker applied was discarded by
    // the clamp on 8 of 9 measured trees. A cap that ignores the density knob IS
    // the density knob, and it was set to one value.
    //
    // Expressed at FULL density (scale 1.0) and multiplied by `site.foliageScale`
    // — the same field the crown-volume fill reads, so there is ONE dial with one
    // meaning across both leaf populations rather than a second per-species knob.
    // 66 is chosen so the generator's own 0.30 default reproduces the historical
    // cap exactly (66 × 0.30 = 19.8 → 20); the app's 0.55 gives 36.
    public static let leafNodeCapAtFullDensity = 66

    /// LOD helper: scale a tessellation count by a tree's detail factor.

    // ===== lod..captureWoodStrand (lines 3093-3438) =====
    public static func lod(_ n: Int, _ detail: Float) -> Int {
        max(5, Int((Float(n) * (0.55 + 0.45 * detail)).rounded()))
    }

    /// Trunk revolution mesh (port of `makeTrunkMesh`) baked into the soup.
    public static func emitTrunkMesh(_ s: inout Soup, world: simd_float4x4,
                                      height: Float, trunkR: Float, topMult: Float,
                                      age: Float, look: TreeLook, detail: Float,
                                      seed: UInt64,
                                      leanX: Float = 0, leanZ: Float = 0) {
        // Bark plates need finer tessellation to resolve as discrete scales with
        // real seam depth (front 2). Hero/near trunks get a denser radial+ring
        // grid; far LOD trunks stay cheap (sub-pixel plates there anyway). Trunks
        // are a tiny slice of the tri budget (~hundreds of trees × this grid) vs
        // the ~7M leaf cards, so this is affordable.
        let ringCount = detail >= 0.45 ? lod(30, detail) : lod(14, detail)
        let radialCount = detail >= 0.45 ? lod(34, detail) : lod(18, detail)
        var rng = ForestRNG(seed: seed)

        // Root flare: a real bole swells gently at the base, it does not trumpet.
        // The earlier 1.4 + age·1.2 (up to ~2.5×) read as a cartoon golf-tee.
        // Keep a believable buttress (≈1.25–1.6×) that relaxes to full radius
        // within the first ~12% of height.
        // Pronounced, VISIBLE root flare so the bole reads as a tree spreading into
        // the ground, not a celery stalk poked into the grass. Widens to 1.45–1.9×
        // at the base and relaxes over the first ~16% of height — the buttress sits
        // ABOVE the grass line (the buried skirt is only lightly sunk now).
        let baseFlare = 1.45 + age * 0.45
        let cps: [(t: Float, r: Float)] = [
            (0.00, baseFlare), (0.06, 1.30 + age * 0.18), (0.16, 1.06),
            (0.50, 1.00), (0.85, 0.93), (1.00, max(topMult, 0.78))
        ]
        func radiusMult(_ t: Float) -> Float { hermite(cps, t) }

        // GROUND BEDDING (#58): the trunk base is bedded BELOW the ground surface
        // so it never floats or leaves a gap over the uneven heightfield — the
        // flare emerges from the soil as a root buttress instead of a pole stuck
        // on a flat disc. The buried skirt widens (×1.35) and sinks `groundSink`
        // metres; the visible trunk top is unchanged (still at y = height).
        let groundSink: Float = 0.18   // light bed; the flare buttress stays ABOVE grass
        let buriedFlare = baseFlare * 1.25

        let perturbScale = 0.6 + age * 0.8
        var radialNoise = [Float](repeating: 0, count: radialCount)
        for i in 0..<radialCount { radialNoise[i] = (rng.unit() - 0.5) * 0.26 * perturbScale }
        // Trunk lean is now supplied by the caller (single source shared with the
        // crown/leader attachment, so the crown rides the leaned top). Consume the
        // two RNG draws the old in-line lean used so the per-ring bark noise below
        // is byte-identical to before.
        _ = rng.unit(); _ = rng.unit()

        var rows: [[(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)]] = []
        // Two buried root-skirt rings below ground (widest at the bottom) so the
        // base beds into the uneven floor with no gap, then the normal trunk rings.
        let buriedRings: [(y: Float, r: Float)] = [
            (-groundSink, trunkR * buriedFlare),
            (-groundSink * 0.45, trunkR * (baseFlare + buriedFlare) * 0.5)
        ]
        for br in buriedRings {
            var row: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
            for radialIdx in 0..<radialCount {
                let angle = Float(radialIdx) / Float(radialCount) * 2 * .pi
                let rJit = max(0.001, br.r * (1 + radialNoise[radialIdx] * 0.45))
                let lp = SIMD3<Float>(rJit * cos(angle), br.y, rJit * sin(angle))
                let ln = simd_normalize(SIMD3<Float>(cos(angle), 0.18, sin(angle)))
                let col = barkColorAt(look, u: Float(radialIdx) / Float(radialCount),
                                      t: 0, seed: seed)
                row.append((xf(world, lp), xfDir(world, ln), col))
            }
            rows.append(row)
        }
        for ringIdx in 0..<ringCount {
            let tLin = Float(ringIdx) / Float(ringCount - 1)
            let tShaped = pow(tLin, 0.75)
            let y = height * tShaped
            let r = trunkR * radiusMult(tShaped)
            let noiseScale = max(0.45, 1.0 - tShaped * 0.55)
            let ringLeanX = leanX * height * tShaped, ringLeanZ = leanZ * height * tShaped
            var row: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
            for radialIdx in 0..<radialCount {
                let angle = Float(radialIdx) / Float(radialCount) * 2 * .pi
                let uFrac = Float(radialIdx) / Float(radialCount)
                let perRing = (rng.unit() - 0.5) * 0.05 * perturbScale
                // ── Bark PLATES (front 2): real radial depth. Mid-plate crowns
                // bulge OUT, seams recess IN — the rake light casts shadow in the
                // furrows. Displacement fades over the canopy-reaching top where
                // bark smooths and the plates are sub-pixel anyway.
                let plate = barkPlateField(look: look, u: uFrac, t: tShaped, seed: seed)
                let plateFade = 1.0 - smoothstepf(0.55, 0.95, tShaped)   // smooth near top
                let plateDisp = plate.disp * plateFade
                let rJit = max(0.001, r * (1 + radialNoise[radialIdx] * noiseScale + perRing + plateDisp))
                let lp = SIMD3<Float>(rJit * cos(angle) + ringLeanX, y, rJit * sin(angle) + ringLeanZ)
                let upBias = max(0, 0.20 - tShaped) * 0.6
                // Bark-relief micro-normal: deep VERTICAL fissures (real bark
                // furrows run UP the trunk) — high angular frequency, low vertical
                // frequency — tilting the tangential normal so the rake light reads
                // fissured bark, not a plastic tube. Oak/maple are deeply furrowed;
                // birch is smooth (faint, with its horizontal-ish lenticel marks).
                // Coarse macro fissures only — the fine furrows are now per-pixel
                // (procedural bark normal in the G-buffer), so keep this gentle.
                let barkAmt: Float = look.isBirch ? 0.16 : 0.42
                let angFreq: Float = look.isBirch ? 5.0 : 9.0
                let fissure  = valueNoise(angle * angFreq, tShaped * 4.0 + Float(seed % 53)) - 0.5
                let microBk  = valueNoise(angle * 21.0, tShaped * 26.0 + Float(seed % 71)) - 0.5
                let tangH = SIMD3<Float>(-sin(angle), 0, cos(angle))
                var ln = SIMD3<Float>(cos(angle), upBias, sin(angle))
                ln += tangH * (fissure * barkAmt + microBk * 0.30)
                     + SIMD3<Float>(0, (microBk * 0.20), 0)
                // Plate shading normal: lifting/shed edges tilt the normal UP+OUT
                // so a shaggy maple scale catches a bright top rim and a dark
                // undercut; seam floors tuck the normal slightly inward.
                ln += SIMD3<Float>(0, plate.liftN * plateFade, 0)
                    + SIMD3<Float>(cos(angle), 0, sin(angle)) * (plate.disp * plateFade * 2.0)
                ln = simd_normalize(ln)
                var col = barkColorPlated(look, u: uFrac, t: tShaped, seed: seed, plate: plate)
                // ── Moss ALBEDO wash (front 3): even where the 3D moss cushions
                // sub-pixel at trunk distance, the bark in the moss zone reads as
                // damp green velvet — moss both coats (albedo) and lifts (cushions).
                // Tie the tint to the SHARED coverage field so wash and cushions
                // agree. Only for near trees (cushions only emit there too).
                if detail >= 0.45 {
                    let cov = mossCoverageAt(u: uFrac, t: tShaped, look: look,
                                             detail: detail, seed: seed)
                    if cov > 0.2 {
                        let mossWash = SIMD3<Float>(0.066, 0.140, 0.046)   // damp velvet, linear
                        // Furrow seams hold more moss (damper) → push tint there.
                        let amt = smoothstepf(0.2, 0.85, cov) * (0.55 + 0.4 * plate.seam)
                        col = mixv(col, mossWash, min(0.9, amt))
                    }
                }
                row.append((xf(world, lp), xfDir(world, ln), col))
            }
            rows.append(row)
        }
        stitchRings(&s, rows, radialCount: radialCount)

        // ── RT curve centerline (#60 item 7 2c) ─────────────────────────────
        // The trunk bole as one wood curve: 4 spine samples base→top along the
        // lean axis with the trunk's Hermite radius profile. The trunk is the
        // largest single wood mass per tree, so a static-soup RT shadow of it
        // would visibly detach from the swaying raster trunk — the curve sways.
        if s.captureCurves && s.windActive {
            let ts: [Float] = [0.0, 0.33, 0.66, 1.0]
            var pts: [SIMD3<Float>] = []; var rds: [Float] = []
            pts.reserveCapacity(4); rds.reserveCapacity(4)
            for t in ts {
                let y = height * t
                let lp = SIMD3<Float>(leanX * y, y, leanZ * y)
                pts.append(xf(world, lp))
                rds.append(max(0.001, trunkR * radiusMult(t)))
            }
            captureWoodStrand(&s, points: pts, radii: rds)
        }
    }

    // Shared moss coverage field (front 3): one source of truth so the bark
    // ALBEDO tint (in emitTrunkMesh) and the 3D moss CUSHIONS (in emitTrunkMoss)
    // agree on where moss sits. `t` = height fraction up the trunk; `u` = angle
    // fraction. Returns 0 (bare bark) → 1 (full moss). Climbs higher on the
    // shaded (anti-sun) side; irregular patches via noise; densest at the base.
    public static func mossCoverageAt(u: Float, t: Float, look: TreeLook,
                                       detail: Float, seed: UInt64) -> Float {
        let maxBandFrac: Float = detail >= 0.85 ? 0.55 : 0.42
        if t > maxBandFrac { return 0 }
        let worldShadedAz = atan2(-worldSunDir.z, -worldSunDir.x)
        // Per-tree damp-side scatter (deterministic from seed).
        let scatter = (Float(seed % 211) / 211.0 - 0.5) * 0.8
        let shadedAz = worldShadedAz + scatter
        let angle = u * 2 * .pi
        let tFrac = t / maxBandFrac                                   // 0 base → 1 top of band
        let up = 1.0 - smoothstepf(0.0, 1.0, tFrac)
        let shade = 0.5 + 0.5 * cos(angle - shadedAz)
        let sideHeight = (up * up) + (up - up * up) * shade
        let patch = valueNoise(angle * 2.6 + Float(seed % 41), t * 9.0 * (1.0/max(0.01,maxBandFrac)))
        let cov = sideHeight * (0.55 + 0.7 * shade) * (0.45 + 0.95 * patch)
        return max(0, min(1, cov))
    }

    // ── Trunk-base MOSS (front 3) ─────────────────────────────────────────────
    //
    // Real 3D moss cushions surface-parented to the ACTUAL bark surface (same
    // radius/flare equation as `emitTrunkMesh`, INCLUDING the plate displacement),
    // so the moss sits ON the bark — not a floor tint, not a decal floating above
    // the trunk. Each cushion is a small irregular outward-bulging velvet dome
    // built from a centre vertex + a fanned rim, clumped where the coverage field
    // is high. Coverage:
    //   • dense at the root flare, thinning UP the trunk (`band` falloff),
    //   • climbs HIGHER on the shaded (anti-sun) side — real moss prefers the
    //     damp north/shaded face,
    //   • a soft ground-contact SKIRT at the very base ties trunk→ground.
    // Damp dark-green; the cushion normal points outward so it shades as volume.
    public static func emitTrunkMoss(_ s: inout Soup, world: simd_float4x4,
                                      height: Float, trunkR: Float, topMult: Float,
                                      age: Float, look: TreeLook, detail: Float,
                                      seed: UInt64) {
        // Reproduce the trunk radius profile (must match emitTrunkMesh).
        let baseFlare = 1.45 + age * 0.45
        let cps: [(t: Float, r: Float)] = [
            (0.00, baseFlare), (0.06, 1.30 + age * 0.18), (0.16, 1.06),
            (0.50, 1.00), (0.85, 0.93), (1.00, max(topMult, 0.78))
        ]
        func radiusMult(_ t: Float) -> Float { hermite(cps, t) }

        var rng = ForestRNG(seed: seed &* 2654435761 &+ 99)

        // Shaded face = opposite the world sun azimuth, transformed into trunk
        // local angle. We work in trunk-local angle directly: the sun azimuth in
        // world maps to a local reference via the world matrix's yaw, but for the
        // coverage bias an approximate local sun angle is enough — derive it from
        // the world sun direction projected and rotated back. Cheap proxy: use the
        // tree seed to pick a damp-side azimuth biased toward the world shaded
        // hemisphere so the whole stand leans its moss the same way.
        let worldShadedAz = atan2(-worldSunDir.z, -worldSunDir.x)   // anti-sun, world
        let shadedAz = worldShadedAz + (rng.unit() - 0.5) * 0.8     // per-tree scatter

        // Moss only reads on resolvable near trees; far LOD trunks are sub-pixel.
        // Climb HIGHER on the trunk than the prior band (the lowest ~0.5 m is
        // occluded by the knee-high grass carpet at the hero camera, so the moss
        // has to reach up the VISIBLE mid-trunk to read at all and to "break the
        // clean-trunk CG read" the brief asks for). Half the trunk on the shaded
        // side for hero trees.
        let maxBandFrac: Float = detail >= 0.85 ? 0.55 : 0.42   // up to ~half trunk
        let cushionRings = detail >= 0.85 ? 11 : 7
        let cushionsPerRing = detail >= 0.85 ? 30 : 22

        // Walk a low-res lattice of candidate cushion sites on the lower trunk.
        for ri in 0..<cushionRings {
            // Spread cushions up the band (linear-ish so the visible mid-trunk is
            // covered, not just the grass-occluded base).
            let rt = Float(ri) / Float(cushionRings - 1)
            let tFrac = maxBandFrac * (0.15 + 0.85 * rt)
            let yLocalBase: Float = -0.16                       // start just below grass
            let y = yLocalBase + height * tFrac                 // trunkLen-relative band
            let tForR = max(0, min(1, (y) / max(0.01, height)))
            let r = trunkR * radiusMult(tForR)
            for ci in 0..<cushionsPerRing {
                let angle = (Float(ci) + rng.unit() * 0.7) / Float(cushionsPerRing) * 2 * .pi
                let uFracCov = angle / (2 * .pi)
                // Shared coverage field (same one the bark albedo wash reads) so
                // cushions land exactly where the trunk reads damp+green.
                let coverage = mossCoverageAt(u: uFracCov, t: tForR, look: look,
                                              detail: detail, seed: seed)
                let up = 1.0 - smoothstepf(0.0, maxBandFrac, tFrac)
                if coverage < 0.42 { continue }
                // Plate displacement at this site so the cushion hugs the bark
                // relief (sits in/over the furrows, not on a smooth phantom tube).
                let uFrac = angle / (2 * .pi)
                let plate = barkPlateField(look: look, u: uFrac, t: tForR, seed: seed)
                let plateFade = 1.0 - smoothstepf(0.55, 0.95, tForR)
                let rSurf = max(0.01, r * (1 + plate.disp * plateFade))
                let ca = cos(angle), sa = sin(angle)
                let outward = SIMD3<Float>(ca, 0, sa)
                let centreLocal = SIMD3<Float>(rSurf * ca, y, rSurf * sa)
                // Cushion size scales with coverage + a little jitter; bigger at base.
                let sz = (0.045 + 0.075 * coverage) * (0.7 + 0.6 * rng.unit())
                                                   * (0.85 + 0.4 * up)
                // Velvet dome: bulge OUT along the surface normal; rim sits flat on
                // the bark, centre lifts ~sz. Slight up-axis tilt so the cushion
                // mounds toward the light a touch (reads as a cushion, not a disc).
                let bulge = outward * (sz * 1.15) + SIMD3<Float>(0, sz * 0.25, 0)
                let centre = centreLocal + bulge
                // Damp dark-green moss colour with per-cushion + within-cushion var.
                let dampen = 0.78 + 0.34 * rng.unit()
                let mossBase = SIMD3<Float>(0.072, 0.150, 0.045)       // damp velvet, linear
                let mossEdge = SIMD3<Float>(0.052, 0.110, 0.040)       // darker contact
                let cCentre = mossBase * dampen
                // Tangent frame around the outward normal for the rim fan.
                let upRef: SIMD3<Float> = abs(outward.y) > 0.9 ? SIMD3(1,0,0) : SIMD3(0,1,0)
                let tx = simd_normalize(simd_cross(upRef, outward))
                let ty = simd_normalize(simd_cross(outward, tx))
                let rim = 7
                var prev = SIMD3<Float>(0,0,0)
                var prevN = outward
                var prevC = mossEdge
                for k in 0...rim {
                    let a = Float(k) / Float(rim) * 2 * .pi
                    let rr = sz * (0.75 + 0.45 * valueNoise(a * 1.7 + Float(ci), tFrac * 4))
                    let rimLocal = centreLocal
                        + (tx * cos(a) + ty * sin(a)) * rr
                        + outward * (sz * 0.20)        // rim lifts slightly off bark
                    let n = simd_normalize(outward * 0.7
                            + (tx * cos(a) + ty * sin(a)) * 0.3)
                    let cc = mixv(mossEdge, cCentre, 0.35) * (0.85 + 0.3 * valueNoise(a*3, Float(ci)))
                    if k > 0 {
                        // Fan triangle: centre(dome top) → prev rim → this rim.
                        s.tri(centre, prev, rimLocal,
                              n0: outward, n1: prevN, n2: n,
                              c0: cCentre, c1: prevC, c2: cc)
                    }
                    prev = rimLocal; prevN = n; prevC = cc
                }
            }
        }

        // Ground-contact SKIRT: a low ring of darker moss flaring out where the
        // root flare meets the soil, so the trunk→ground transition is mossy, not
        // a clean bark-on-grass seam. Surface-parented to the flare radius.
        let skirtN = detail >= 0.85 ? 30 : 20
        let skirtR = trunkR * baseFlare
        let contact = SIMD3<Float>(0.058, 0.115, 0.046)   // dark damp contact moss
        for i in 0..<skirtN {
            let a0 = Float(i) / Float(skirtN) * 2 * .pi
            let a1 = Float(i + 1) / Float(skirtN) * 2 * .pi
            func ringPt(_ a: Float, _ yy: Float, _ rad: Float) -> SIMD3<Float> {
                SIMD3<Float>(rad * cos(a), yy, rad * sin(a))
            }
            // Coverage gate so the skirt is patchy, denser on the shaded side.
            let shade0 = 0.5 + 0.5 * cos(a0 - shadedAz)
            let g = valueNoise(a0 * 3.1 + Float(seed % 23), 0.7) * (0.5 + 0.7 * shade0)
            if g < 0.42 { continue }
            let flare = 1.0 + 0.5 * g
            let yTop: Float = 0.02 + 0.12 * g                 // climbs a touch up the flare
            let inner0 = ringPt(a0, yTop, skirtR * 1.02)
            let inner1 = ringPt(a1, yTop, skirtR * 1.02)
            let outer0 = ringPt(a0, -0.06, skirtR * flare)    // flares onto the ground
            let outer1 = ringPt(a1, -0.06, skirtR * flare)
            let nUp = SIMD3<Float>(0, 1, 0)
            s.tri(inner0, outer0, outer1, n0: nUp, n1: nUp, n2: nUp,
                  c0: contact, c1: contact * 0.9, c2: contact * 0.9)
            s.tri(inner0, outer1, inner1, n0: nUp, n1: nUp, n2: nUp,
                  c0: contact, c1: contact * 0.9, c2: contact)
        }
    }

    /// Tapered branch revolution mesh with curved spine + Hermite collar
    /// (port of `makeBranchMesh`) baked into the soup.
    /// Append one wood centerline strand for the RT curve primitives (#60 item 7
    /// 2c) from 4 base→tip world-space spine samples + radii. Per-point windAttr
    /// uses the SAME formula as the G-buffer vertex wind tangent (`windTangent` in
    /// `tri()`): a height² sway weight off the current tree's `windBaseY`/
    /// `windCrownH`, the tree's `windPhase`, and no flutter (wood). The renderer's
    /// `illumi_curve_wind_displace` kernel re-applies `applyTreeWind` to these each
    /// frame, so the RT wood shadow sways in lockstep with the rastered tube.
    /// Gated by callers on `captureCurves && windActive`.
    public static func captureWoodStrand(_ s: inout Soup,
                                          points: [SIMD3<Float>], radii: [Float]) {
        guard points.count == 4, radii.count == 4 else { return }
        var wind: [SIMD4<Float>] = []
        wind.reserveCapacity(4)
        for p in points {
            let hw = max(0, min(1.0, (p.y - s.windBaseY) / s.windCrownH))
            wind.append(SIMD4(min(1.0, hw * hw), s.windPhase, 0, s.speciesMark))
        }
        s.curveStrands.append(Soup.CurveStrand(points: points, radii: radii, windAttr: wind))
    }


    // ===== emitBranchMesh+stitchRings (lines 3439-3716) =====
    public static func emitBranchMesh(_ s: inout Soup, world: simd_float4x4,
                                       length: Float, bottomR: Float, topR: Float,
                                       collarMult: Float, radialSegments: Int,
                                       curveAmount: Float, curveAxis: SIMD3<Float>,
                                       look: TreeLook, baseY: Float, seed: UInt64) {
        guard length > 1e-4 else { return }
        let cps: [(t: Float, r: Float)]
        if collarMult > 1.001 {
            cps = [(0.00, bottomR * collarMult), (0.18, bottomR), (1.00, max(0.001, topR))]
        } else {
            cps = [(0.00, bottomR), (1.00, max(0.001, topR))]
        }
        func radiusAt(_ t: Float) -> Float { hermite(cps, t) }

        let ringCount = max(6, curveAmount > 0.05 ? 9 : 6)
        let cAxRaw = SIMD3<Float>(curveAxis.x, 0, curveAxis.z)
        let cAxLen = simd_length(cAxRaw)
        let curveAxLocal = cAxLen > 1e-4 ? cAxRaw / cAxLen : SIMD3<Float>(1, 0, 0)
        let totalSideOffset = curveAmount * length

        var rows: [[(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)]] = []
        for ringIdx in 0..<ringCount {
            let tLin = Float(ringIdx) / Float(ringCount - 1)
            let t = pow(tLin, 0.65)
            let side = totalSideOffset * t * t
            let yOnSpine = length * t
            let tanSide = 2 * totalSideOffset * t
            let tanY = length
            let tanLen = max(1e-4, sqrt(tanSide * tanSide + tanY * tanY))
            let tangent = SIMD3<Float>(curveAxLocal.x * tanSide / tanLen,
                                       tanY / tanLen, curveAxLocal.z * tanSide / tanLen)
            let refUp: SIMD3<Float> = abs(tangent.y) > 0.97 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
            let ringX = simd_normalize(simd_cross(refUp, tangent))
            let ringZ = simd_normalize(simd_cross(tangent, ringX))
            let centre = SIMD3<Float>(curveAxLocal.x * side, yOnSpine, curveAxLocal.z * side)
            let r = max(0.001, radiusAt(t))
            var row: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
            for radIdx in 0..<radialSegments {
                let a = Float(radIdx) / Float(radialSegments) * 2 * .pi
                let ca = cos(a), sa = sin(a)
                let lp = centre + ringX * (r * ca) + ringZ * (r * sa)
                let ln = simd_normalize(ringX * ca + ringZ * sa)
                let col = barkColorAt(look, u: Float(radIdx) / Float(radialSegments),
                                      t: baseY + yOnSpine, seed: seed)
                row.append((xf(world, lp), xfDir(world, ln), col))
            }
            rows.append(row)
        }
        stitchRings(&s, rows, radialCount: radialSegments)

        // ── RT curve centerline (#60 item 7 2c) ─────────────────────────────
        // Structural wood only (leader=2 / primary=3 / branch=4); twigs (9) and
        // understory (0) are skipped — sub-pixel shadow casters that would swamp
        // the curve count. Sample the SAME swept spine the tube uses (4 points,
        // mirroring TreeGeometry.appendBranchToSkeleton: side = totalSideOffset·t²,
        // y = length·t), with the cubic-Hermite tube radius at each, transformed
        // into world space so the curve coincides with the rastered tube.
        if s.captureCurves && s.windActive && s.debugClass >= 2 && s.debugClass <= 4 {
            let ringTs: [Float] = [0.0, 0.33, 0.66, 1.0]
            var pts: [SIMD3<Float>] = []; var rds: [Float] = []
            pts.reserveCapacity(4); rds.reserveCapacity(4)
            for t in ringTs {
                let side = totalSideOffset * t * t
                let lp = SIMD3<Float>(curveAxLocal.x * side, length * t, curveAxLocal.z * side)
                pts.append(xf(world, lp))
                rds.append(max(0.001, radiusAt(t)))
            }
            captureWoodStrand(&s, points: pts, radii: rds)
        }
    }

    /// Connect consecutive rings (each `radialCount` verts) into a tube.
    public static func stitchRings(_ s: inout Soup,
                                    _ rows: [[(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)]],
                                    radialCount: Int) {
        for ri in 0..<(rows.count - 1) {
            let lo = rows[ri], hi = rows[ri + 1]
            for i in 0..<radialCount {
                let j = (i + 1) % radialCount
                s.tri(lo[i].0, lo[j].0, hi[j].0, n0: lo[i].1, n1: lo[j].1, n2: hi[j].1,
                      c0: lo[i].2, c1: lo[j].2, c2: hi[j].2)
                s.tri(lo[i].0, hi[j].0, hi[i].0, n0: lo[i].1, n1: hi[j].1, n2: hi[i].1,
                      c0: lo[i].2, c1: hi[j].2, c2: hi[i].2)
            }
        }
    }

}
