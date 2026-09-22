import Foundation
import simd

// Recursive branch/twig growth and tip rosettes.
extension ForestTreeGeometry {
    // ===== growBranchSoup (lines 2494-2868) =====
    public static func growBranchSoup(_ s: inout Soup, parentWorld: simd_float4x4,
                                       length: Float, bottomR: Float, topR: Float,
                                       pitch: Float, yaw: Float, roll: Float,
                                       collarMult: Float, depth: Int, maxDepth: Int,
                                       site: TreeSite, look: TreeLook,
                                       sunDir: SIMD3<Float>,
                                       crownCenter: SIMD3<Float>, crownRX: Float,
                                       crownRY: Float,
                                       rng: inout ForestRNG) {
        // LOBED ELLIPSOIDAL CROWN GATE (#58) — applied BEFORE emitBranchMesh.
        // Uses the branch TIP direction azimuth (not the base→crown vector) so that
        // each primary branch fanning out at a different yaw angle sees its own lobe
        // environment — the old approach used atan2(0,0) for all primaries (degenerate).
        // c < 0: base inside lobed ellipsoid → find exit, clamp tip there.
        // c ≥ 0: base outside → gate deeper branches; trim primaries proportionally.
        let segWorld = parentWorld * rotY(yaw) * rotX(pitch) * rotY(roll)
        let basePos  = SIMD3<Float>(segWorld.columns.3.x, segWorld.columns.3.y, segWorld.columns.3.z)
        let tipDir   = SIMD3<Float>(segWorld.columns.1.x, segWorld.columns.1.y, segWorld.columns.1.z)
        let crownRXS = max(crownRX, 0.1)
        let crownRYS = max(crownRY, 0.1)

        // ROUND 2: ONE canonical lobe field, shared with fillCrownVolume. The
        // previous pass computed a SEPARATE `lWarp` here that was "±0.4 apart" from
        // the fill's `warp` in any given direction — so deepening the fill's bays
        // exposed branch tips this clamp didn't track (the bare-twig-on-skyline
        // regression). Now the branch clamp evaluates the EXACT SAME CrownLobe in
        // the branch's growth direction, so a tip in a bay is clamped to that bay's
        // (shorter) radius and stays buried under the leaf mass.
        let lobe = crownLobeField(crownX: crownCenter.x, crownZ: crownCenter.z,
                                  crownRadius: crownRX)
        // Lobe evaluated in the branch's GROWTH direction (XZ azimuth + elevation).
        // cosT matches the fill convention (cos of polar angle, +1 = crown top) so
        // a horizontal branch sees cosT≈0 and a near-vertical leader sees cosT≈1.
        let branchAz = atan2(tipDir.z, tipDir.x)
        let branchEl = atan2(tipDir.y, max(0.001, sqrt(tipDir.x * tipDir.x + tipDir.z * tipDir.z)))
        let branchCosT = sin(branchEl)
        let (lWarp, _) = lobe.warp(az: branchAz, cosT: branchCosT)
        // SHADOW-SIDE STRUCTURAL FIX (#20 items 1+2, round 2): clamp branch TIPS
        // strictly INSIDE the leaf-fill shell so no bare twig can ever exceed the
        // foliage. The fill now reaches 0.95·fillRY = 0.95·1.30 = 1.235·crownRY
        // (up, ×warp); pulling the branch gate 1.12→1.04 means every primary/twig
        // tip ends well under that — leaves cluster at and just beyond the clamped
        // tip (physically correct: twigs end, foliage extends past them). This is
        // the closed-form fix for the residual bare birch fork + backdrop twig
        // fringe the side/top angles still showed: the silhouette is now owned
        // entirely by the fill envelope, never by exposed branchwork.
        // BIRCH FORK-TIP FIX (#20 structural item 2): birch runs no leader, so its
        // near-vertical PRIMARIES (depth 1) are the fork — and at the shared 1.04
        // limit their tips landed at ~1.04·crownRY, ABOVE the dense leaf top (~1.0·
        // crownRY on the vertical axis), poking bare wood on the skyline. Pull the
        // birch primary clamp in to 0.86·crownRY so the fork terminus sits inside
        // the dense mass; the emitTipRosette tuft then caps it from the silhouette.
        //
        // LATERAL-PRIMARY FIX (#20 re-review, horizontal axis of items 1+2): the
        // vertical leader/fork tips were closed, but the side/low/three-quarter
        // angles STILL showed bare near-horizontal/diagonal PRIMARY spokes exiting
        // the crown SIDE against the sky. Closed-form root cause: the fill's
        // HORIZONTAL radius is plain `crownRX` (only `ry` got the 1.30 fillRY
        // boost), so on the horizontal axis the leaf mass reaches ~0.95·crownRX·warp
        // — but the oak/maple primary clamp let tips run to 1.04·crownRX·lWarp.
        // warp and lWarp are different functions (±0.4 apart in a given direction),
        // so a horizontal primary in a peak-lWarp / gap-warp direction exited up to
        // ~0.2·crownRX PAST the foliage edge → bare lateral wood on the silhouette.
        // Re-widening the fill `rx` was explicitly ruled out (it thins opacity over
        // a larger volume + widens the silhouette). Instead pull the oak/maple
        // primary clamp IN to 0.86·crownRX — matching the birch reference — so EVERY
        // primary terminus (horizontal, diagonal, vertical) sits inside the dense
        // leaf mass with margin for the warp/lWarp mismatch. The fill envelope owns
        // the silhouette; branch wood is interior filigree, foliage extends past it.
        // Secondary/twig tips (depth>1) keep 1.04 — they live deep inside the fill
        // and never reach the silhouette. (HOLD: the commit-0d83980 canopy fixes.)
        // LATERAL-SPOKE round 2: 0.86 still left a bare SHAFT on near-horizontal
        // oak primaries (esp. the lobePitchBoost limbs) — the tip rosette capped the
        // terminus but the wood between where the limb LEFT the dense fill core and
        // its clamped tip stayed exposed in gap-warp azimuths (where the fill,
        // warp<1, pulls inward well short of 0.86). Pull oak/maple primaries to
        // 0.70·crownRX·lWarp so the whole limb shaft terminates inside the DENSE
        // core (the same interior depth the central leader uses, 0.55·crownRY), not
        // at the sparse nominal shell. Foliage extends past the wood (physically
        // correct). Secondary/twig tips keep 1.04 — they live deep inside, never on
        // the silhouette. (HOLD: the commit-0d83980 canopy fixes.)
        // ROUND 2 (re-render) — FILL-COUPLED CLAMP (the actual decoupling bug).
        // The earlier `k · lWarp` form looked unified but was NOT: the leaf FILL
        // shell cancels the warp in lobe PEAKS (rNorm = fillLimit / max(1,warp) →
        // shell radius = fillLimit, constant) and only pulls IN in the gaps — so the
        // fill's true normalized reach in any direction is
        //     R_fill = fillLimit · min(1, warp)        (fillLimit = 0.95)
        // Meanwhile `0.70·lWarp` GREW past fillLimit wherever warp>1.36 — exactly
        // where the deepened amp now drives the peaks — so a primary tip in a lobe
        // peak poked ~0.1–0.3·crownRX OUTSIDE the leaf envelope: the recurring bare
        // twig on the skyline, re-exposed the instant the silhouette was deepened.
        // Fix: clamp every branch tip to a FRACTION of the fill's actual reach, so
        // wood can never exit the foliage no matter how deep the warp swings. The
        // SAME `lWarp` (= the fill's `warp`) feeds both, now coupled multiplicatively
        // through R_fill rather than as an independent `k·lWarp` that outran it.
        let fillReach = Self.fillLimitConst * min(1.0, lWarp)   // = fill silhouette
        let lobeLimit: Float
        if look.isBirch && depth == 1 {
            // Birch fork: terminus well inside the dense top; rosette caps it.
            lobeLimit = max(0.48, 0.78 * fillReach)
        } else if depth == 1 {
            // Oak/maple primaries: whole limb shaft buried in the dense core.
            lobeLimit = max(0.52, 0.78 * fillReach)
        } else {
            // Secondary/twig tips live deep inside — they never reach the silhouette,
            // so they may run nearly to the fill edge (foliage extends past them).
            lobeLimit = max(0.58, 0.98 * fillReach)
        }

        // Quadratic: |baseRel + t·tipNorm|² = lobeLimit² in ellipsoid-scaled space.
        let rel     = basePos - crownCenter
        let baseRel = rel / SIMD3(crownRXS, crownRYS, crownRXS)
        let tipNorm = tipDir / SIMD3(crownRXS, crownRYS, crownRXS)
        let qa = simd_dot(tipNorm, tipNorm)
        let qb = 2 * simd_dot(baseRel, tipNorm)
        let qc = simd_dot(baseRel, baseRel) - lobeLimit * lobeLimit  // <0 = inside

        var clampedLength = length
        // True when the branch tip was clamped to the crown shell boundary — i.e.
        // its tip sits ON the silhouette. Birch fork tips that land here get a leaf
        // rosette so no bare wood reads on the skyline (#20 structural item 2).
        var clampedAtBoundary = false
        if qa > 0.001 {
            if qc >= 0 {
                // Base is OUTSIDE the lobed ellipsoid.
                if depth > 1 { return }
                // Primary (depth == 1): trim at the far-side exit so gap branches
                // are shorter while peak branches extend freely.
                let disc2 = qb * qb - 4 * qa * qc
                let tNear = disc2 >= 0 ? (-qb - sqrt(disc2)) / (2 * qa) : -1
                let tFar  = disc2 >= 0 ? (-qb + sqrt(disc2)) / (2 * qa) : -1
                // BARE-LIMB ROOT-CAUSE FIX (#20 lateral re-review): the neon-bark
                // pass proved the residual spokes are PRIMARIES whose base sits just
                // OUTSIDE the (now 0.70) lobed shell — they spring from the crown
                // bottom (crownWorld at trunkLen, normalised radius ~0.91 vertically)
                // and arc UP-AND-OUT. Three sub-cases all previously emitted the
                // FULL bare length into open sky:
                //   (a) disc2 < 0 — chord never enters the ellipsoid at all,
                //   (b) tFar <= 0 — both intersections behind the base (growing away),
                //   (c) tNear >= length — the limb stops before it re-enters the mass.
                // In every one the limb runs entirely OUTSIDE the leaf fill. Clamp it
                // to a short stub that stays tucked at the crown edge and flag it so
                // the primary-tip rosette curtains the stub end. Where the chord DOES
                // dip back through the mass (tNear in (0,length)), clamp the tip to
                // the far exit so the in-mass span is kept and the bare overhang cut.
                if disc2 >= 0 && tFar > 0 && tFar < length && tNear < length {
                    clampedLength = max(0.3, tFar)
                    clampedAtBoundary = true
                } else {
                    // Limb runs outside the mass for its whole length — stub it so no
                    // bare primary streaks across the sky; the rosette caps the stub.
                    // Stub length ~0.30 of the smaller crown radius keeps it a short
                    // collar nub tucked at the crown edge, never a sky-crossing arc.
                    let stub = 0.30 * min(crownRXS, crownRYS)
                    clampedLength = max(0.25, min(length, stub))
                    clampedAtBoundary = true
                }
            } else {
                // Base is INSIDE — find exit, clamp tip there.
                // disc always > 0 when qc < 0 (base inside).
                let disc2 = qb * qb - 4 * qa * qc
                let tExit = (-qb + sqrt(max(0, disc2))) / (2 * qa)
                if tExit > 0 && tExit < length {
                    clampedLength = max(0.15, tExit)
                    clampedAtBoundary = true
                    if depth > 1 && tExit < length * 0.25 { return }
                }
            }
        }

        let curveBase: Float
        switch depth {
        case 1:  curveBase = 0.18
        case 2:  curveBase = 0.22
        default: curveBase = 0.12
        }
        let curveAmount = curveBase * (0.6 + rng.unit() * 1.1)
        let curveYaw = rng.unit() * 2 * .pi
        let curveAxis = SIMD3<Float>(cos(curveYaw), 0, sin(curveYaw))

        let radSegs = lod(depth == 1 ? 12 : (depth == 2 ? 9 : 7), site.detail)
        s.debugClass = depth == 1 ? 3 : 4
        emitBranchMesh(&s, world: segWorld, length: clampedLength, bottomR: bottomR,
                       topR: topR, collarMult: collarMult, radialSegments: radSegs,
                       curveAmount: curveAmount, curveAxis: curveAxis,
                       look: look, baseY: 0, seed: site.seed)
        s.debugClass = 0

        // FORK / PRIMARY-TIP ROSETTE (#20 structural item 2 + lateral re-review):
        // cap any PRIMARY (depth==1) whose tip lands ON or NEAR the crown shell —
        // that tip is on the silhouette. Birch primaries fork near-vertical (the
        // original bare two-pronged armature); oak/maple primaries that exit the
        // crown SIDE showed the same bare wood on the lateral re-review.
        //
        // GATE: not just `clampedAtBoundary`. The residual lateral spoke the
        // re-review showed was an UN-clamped short primary — its natural length
        // never reached the 0.86 clamp shell, so clampedAtBoundary stayed false, yet
        // its tip ended in a GAP-WARP region where the leaf fill (warp<1) pulled the
        // mass inward, leaving the wood exposed. Compute the tip's normalised
        // ellipsoid radius and rosette whenever it exits the dense leaf core
        // (≥0.70 of the shell on its growth axis) — covers clamped AND near-edge
        // primaries. The rosette is a small tuft right at the tip; it curtains the
        // wood without re-widening the fill shell (the lever the reviewer warned off).
        let primTipLocal = curveTipOffset(length: clampedLength, curveAmount: curveAmount,
                                          curveAxis: curveAxis)
        let primTipTan = curveTipTangent(length: clampedLength, curveAmount: curveAmount,
                                         curveAxis: curveAxis)
        if depth == 1 {
            let tipWS = SIMD3<Float>((segWorld * translate(primTipLocal)).columns.3.x,
                                     (segWorld * translate(primTipLocal)).columns.3.y,
                                     (segWorld * translate(primTipLocal)).columns.3.z)
            let tRel = (tipWS - crownCenter) / SIMD3(crownRXS, crownRYS, crownRXS)
            let tipRadNorm = simd_length(tRel)
            if clampedAtBoundary || tipRadNorm >= 0.70 {
                let tipWorld = segWorld * translate(primTipLocal) * orientYTo(primTipTan)
                s.debugClass = 7
                emitTipRosette(&s, tipWorld: tipWorld, leafSize: look.leafCardSize,
                               look: look, site: site, rng: &rng)
                s.debugClass = 0
            }

            // ── TRUNK→BRANCH JUNCTION CLOTHING ────────────────────────────────
            // Root cause of the "trunk not connecting to its branches" read: a
            // depth-1 primary forks at the trunk top (crown-normalised y ≈ −0.9)
            // and arcs UP into the crown, but it receives NO foliage along its
            // length — emitLeaves only fires on the twig tiers (depth ≥ maxDepth−1)
            // and the only cap is the single tip rosette above, which sits deep in
            // the dense core. Meanwhile the volumetric fill's lower hemisphere is
            // pinched (teardrop ×0.66) and thinned (pow(hz,0.42)), so it never
            // reaches down to clothe the lower primary shaft. The result is a bare
            // neck of trunk-top + primary-base wood between the trunk and the
            // floating canopy. Fix (bottom-hemisphere ONLY — the skyline clamp at
            // the crown TOP is untouched, so no skyline-twig regression): hang a
            // few small leaf rosettes along the lower-to-mid span of each primary
            // so the wood crossing the junction reads as leaf-bearing, exactly as a
            // real broadleaf primary's lateral shoots do. Near trees only (far
            // primaries are sub-pixel and would only burn budget).
            // CONIFER-SPIRE ROOT CAUSE (found by keeping only the sparse tier): this
            // JUNCTION-CLOTHING loop hangs `stations` leaf rosettes at FIXED span
            // fractions (0.12..0.74) on EVERY primary. The primaries fan in a whorl
            // from one crown-base origin, so the rosettes land at the SAME few heights
            // around the axis → concentric horizontal RINGS of foliage. On a NEAR hero
            // tree the dense 12k-cluster fill buries the rings; but on the det 0.45–
            // 0.58 MID/DISTANT tier the fill is too sparse to merge them, so the rings
            // read as a tiered CONIFER spike in the clearing-gap sightline — the
            // "faceted conifer" the brief flags. The junction gap this clothing closes
            // is only actually VISIBLE at close range, so gate it to the NEAR tier
            // (det ≥ 0.85); the distant tier relies on the volumetric fill (whose
            // fill-coupled branch clamp now keeps every tip buried) for an unbroken,
            // un-tiered crown. Also JITTER the station fractions per primary so even
            // the near tier's rosettes don't align into rings. (HOLD: the close-up
            // bark/leaf/moss work and the live leaf transmission are untouched.)
            if site.detail >= 0.85 && clampedLength > 0.4 {
                let stations = 4
                // Per-primary phase so stations don't align across the whorl.
                let phase = (rng.unit() - 0.5) * 0.14
                for st in 0..<stations {
                    let jf = (rng.unit() - 0.5) * 0.10
                    let f = max(0.10, 0.12 + phase + jf
                                + (Float(st) + 0.5) / Float(stations) * 0.62)
                    let along = curveTipOffset(length: clampedLength * f,
                                               curveAmount: curveAmount,
                                               curveAxis: curveAxis)
                    let tan = curveTipTangent(length: clampedLength * f,
                                              curveAmount: curveAmount,
                                              curveAxis: curveAxis)
                    let stWorld = segWorld * translate(along) * orientYTo(tan)
                    s.debugClass = 8
                    emitTipRosette(&s, tipWorld: stWorld,
                                   leafSize: look.leafCardSize * 1.05,
                                   look: look, site: site, rng: &rng)
                    s.debugClass = 0
                }
            }
        }

        // Foliage policy (mirrors the SCN version's depth tiers). Use clampedLength
        // so leaf sprays anchor at the actual (clamped) branch tip, not beyond the crown.
        if depth >= maxDepth {
            emitLeaves(&s, world: segWorld, twigLength: clampedLength,
                       leafSize: look.leafCardSize,
                       density: site.leafDensity * look.leafDensityMult,
                       look: look, site: site, rng: &rng)
            let tipLocal = curveTipOffset(length: clampedLength, curveAmount: curveAmount,
                                          curveAxis: curveAxis)
            let tipTan = curveTipTangent(length: clampedLength, curveAmount: curveAmount,
                                         curveAxis: curveAxis)
            emitTwigCluster(&s, at: tipLocal, tangent: tipTan, parentWorld: segWorld,
                            parentTopR: topR, site: site, look: look,
                            crownCenter: crownCenter, crownRX: crownRX,
                            crownRY: crownRY, rng: &rng)
            // Also cap the bare birch terminus itself (covers fork tips that reached
            // maxDepth without a boundary clamp — e.g. an upward fork that ends just
            // shy of the shell with its last segment exposed against the sky).
            if look.isBirch && !clampedAtBoundary {
                let tipWorld = segWorld * translate(tipLocal) * orientYTo(tipTan)
                s.debugClass = 7
                emitTipRosette(&s, tipWorld: tipWorld, leafSize: look.leafCardSize,
                               look: look, site: site, rng: &rng)
                s.debugClass = 0
            }
            return
        }
        if depth == maxDepth - 1 {
            emitLeaves(&s, world: segWorld, twigLength: clampedLength,
                       leafSize: look.leafCardSize * 0.94,
                       density: site.leafDensity * look.leafDensityMult * 0.95,
                       look: look, site: site, rng: &rng)
        }
        if depth >= 2 && depth < maxDepth - 1 {
            emitLeaves(&s, world: segWorld, twigLength: clampedLength,
                       leafSize: look.leafCardSize * 0.85,
                       density: site.leafDensity * look.leafDensityMult * 0.65,
                       look: look, site: site, rng: &rng)
        }

        // Tip pivot: child branches grow off the (curved) tip, aligned to tangent.
        let tipLocal = curveTipOffset(length: clampedLength, curveAmount: curveAmount,
                                      curveAxis: curveAxis)
        var tipPivotWorld = segWorld * translate(tipLocal)
        if curveAmount > 0.01 {
            let tipTan = curveTipTangent(length: clampedLength, curveAmount: curveAmount,
                                         curveAxis: curveAxis)
            tipPivotWorld = tipPivotWorld * orientYTo(tipTan)
        }

        let childCount: Int
        if look.isBirch && depth >= 2 {
            childCount = 3 + Int(rng.unit() * 2.0)
        } else {
            switch depth {
            case 1:  childCount = 2 + Int(rng.unit() * 2.0)
            case 2:  childCount = 3 + Int(rng.unit() * 2.0)
            default: childCount = 3 + Int(rng.unit() * 2.0)
            }
        }
        // Child branches use the natural length scale; the lobed gate at the TOP
        // of growBranchSoup will clamp or skip any child whose base lands outside
        // the lobed ellipsoid — no need for a separate nextLength pre-clamp here.
        let nextLength = length * (0.60 + rng.unit() * 0.15)
        let nextBottomR = topR
        let nextTopR = topR * (0.60 + rng.unit() * 0.12)

        for c in 0..<childCount {
            let childYaw = Float(c) / Float(childCount) * 2 * .pi + (rng.unit() - 0.5) * 0.55
            let isSprout = rng.unit() < 0.40 && depth >= 2
            let childPitch: Float
            var lengthMult: Float = 1.0
            var radiusMult: Float = 1.0
            if isSprout {
                childPitch = 0.04 + rng.unit() * 0.30
                lengthMult = 0.55 + rng.unit() * 0.30
                radiusMult = 0.70 + rng.unit() * 0.18
            } else {
                childPitch = look.secondaryPitchMin + rng.unit() * look.secondaryPitchRange
            }
            let childAxisLocal = SIMD3<Float>(sin(childYaw + yaw), 0, cos(childYaw + yaw))
            let sunBias = simd_dot(childAxisLocal, sunDir)
            let finalLengthMult = lengthMult * (1.0 + sunBias * 0.25)
            let finalPitch = max(0.02, childPitch - sunBias * 0.18)
            let childRoll = (rng.unit() - 0.5) * 0.35
            let childDepth = depth + 1
            let childCollar: Float = childDepth == 2 ? 1.25 : 1.0
            growBranchSoup(&s, parentWorld: tipPivotWorld,
                           length: nextLength * finalLengthMult * (0.85 + rng.unit() * 0.30),
                           bottomR: nextBottomR * radiusMult, topR: nextTopR * radiusMult,
                           pitch: finalPitch, yaw: childYaw, roll: childRoll,
                           collarMult: childCollar, depth: childDepth, maxDepth: maxDepth,
                           site: site, look: look, sunDir: sunDir,
                           crownCenter: crownCenter, crownRX: crownRX, crownRY: crownRY,
                           rng: &rng)
        }
    }

    /// Terminal fractal twig spray (port of attachTwigCluster).

    // ===== emitTwigCluster (lines 2869-3002) =====
    public static func emitTwigCluster(_ s: inout Soup, at localTip: SIMD3<Float>,
                                        tangent: SIMD3<Float>,
                                        parentWorld: simd_float4x4, parentTopR: Float,
                                        site: TreeSite, look: TreeLook,
                                        crownCenter: SIMD3<Float>, crownRX: Float,
                                        crownRY: Float,
                                        rng: inout ForestRNG) {
        let prevDbg = s.debugClass; s.debugClass = 9; defer { s.debugClass = prevDbg }
        // Structural (#58): the terminal twig spray now BUILDS the crown volume,
        // because leaves cling tight to the twig instead of orbiting it in a fat
        // sphere. So fan MORE twigs, WIDER, and LONGER — a real broadleaf shoot
        // tip is a dense fork of leaf-bearing branchlets, and it's that fork (not
        // a radial leaf cloud) that gives the crown its broken, light-permeable
        // silhouette with visible dark twig filigree threading through it.
        // Restored to 4–7 twigs per cluster (was 3–5 after hang fix). With the
        // 4M triCap + async build, the extra twigs cost only SIMD math on the
        // detached thread — no main-actor risk. 4–7 gives a fuller, more
        // hemispherical terminal spray without the "sparse finger" look.
        let twigCount = 4 + Int(rng.unit() * 3.0)
        let pivotWorld = parentWorld * translate(localTip) * orientYTo(tangent)
        // Twigs are many leaf-internodes long so each carries a real spray of
        // leaves along its length (life-size leaves are tiny, and they now hang
        // tight to the twig, so the twig must be long enough to spread the crown).
        let baseLen = look.isBirch ? look.leafCardSize * 16.0 : look.leafCardSize * 13.0
        for i in 0..<twigCount {
            // Spread twigs through a hemisphere off the shoot tip (not a flat
            // ring): combine an even yaw fan with a wide pitch range so the
            // branchlets reach out AND up/down, filling a crown lobe volume.
            let yaw = Float(i) / Float(twigCount) * 2 * .pi + (rng.unit() - 0.5) * 0.9
            let pitch = 0.30 + rng.unit() * 1.05
            var twigLen = baseLen * (0.75 + rng.unit() * 0.55)
            // RACHIS-OCCLUSION (top-down fern fix): thin the finest twig tubes.
            // From directly above, the leaf-bearing twig tube is the visible
            // radiating "rachis" line of the fern read. A real leaf cluster hides
            // its own twig — the twig is far thinner than the leaf-mass it carries
            // and gets buried under the overlapping blades. Pull the leaf-bearing
            // branchlet radius WAY down (≈40% of the prior gauge) so it no longer
            // reads as a bare spine threading the leaf row from above. The tube is
            // still emitted (the backlit eye-level lace needs the dark filigree),
            // just thin enough that the clustered rosettes occlude it from the top.
            let twigR0 = max(0.0008, parentTopR * (0.22 + rng.unit() * 0.12))
            let twigR1 = twigR0 * 0.30
            let curveAmt = 0.10 + rng.unit() * 0.20
            let curveYaw = rng.unit() * 2 * .pi
            let cAxis = SIMD3<Float>(cos(curveYaw), 0, sin(curveYaw))
            let twigWorld = pivotWorld * rotY(yaw) * rotX(pitch)
            // SHELL CLAMP (#20, residual skyline-twig fix): clamp this terminal twig
            // so its tip stays inside the crown leaf-mass and never pokes bare on the
            // silhouette. Same ellipsoid-exit math as the branch gate; clamp to 0.92
            // of the shell so the twig (and its leaf spray) sit just inside the dense
            // canopy. Twigs whose BASE is already outside the shell are skipped.
            let twBase = SIMD3<Float>(twigWorld.columns.3.x, twigWorld.columns.3.y,
                                      twigWorld.columns.3.z)
            let twDir  = SIMD3<Float>(twigWorld.columns.1.x, twigWorld.columns.1.y,
                                      twigWorld.columns.1.z)
            let crX = max(crownRX, 0.1), crY = max(crownRY, 0.1)
            let twRel = (twBase - crownCenter) / SIMD3(crX, crY, crX)
            let twNrm = twDir / SIMD3(crX, crY, crX)
            let tqa = simd_dot(twNrm, twNrm)
            let tqb = 2 * simd_dot(twRel, twNrm)
            let twigShell: Float = 0.92
            let tqc = simd_dot(twRel, twRel) - twigShell * twigShell
            if tqa > 0.001 {
                if tqc >= 0 {
                    continue   // twig base already outside the leaf shell — drop it
                }
                let disc = tqb * tqb - 4 * tqa * tqc
                let tExit = (-tqb + sqrt(max(0, disc))) / (2 * tqa)
                if tExit > 0 && tExit < twigLen { twigLen = max(0.02, tExit) }
            }
            emitBranchMesh(&s, world: twigWorld, length: twigLen, bottomR: twigR0,
                           topR: twigR1, collarMult: 1.0,
                           radialSegments: lod(6, site.detail),
                           curveAmount: curveAmt, curveAxis: cAxis,
                           look: look, baseY: 0, seed: site.seed)
            emitLeaves(&s, world: twigWorld, twigLength: twigLen,
                       leafSize: look.leafCardSize * (0.85 + rng.unit() * 0.25),
                       density: site.leafDensity * look.leafDensityMult * 1.30,
                       look: look, site: site, rng: &rng)

            // BUDGET-DISABLED: twiglet tier (4th-gen leaf-bearing shoots) was
            // costing 2-4× the main twig budget alone. Root-cause analysis:
            // 8 hero trees × ~356 emitLeaves calls × 3 twiglets × 20 nodes ×
            // 24 tris = ~4M tris just for twiglets — consuming the entire
            // triCap before any backdrop/cap trees got geometry → invisible
            // treeline. Gated to detail ≥ 1.5 (never fires) instead of ≥ 0.9.
            // Re-enable only after verifying the triCap budget survives.
            if site.detail >= 1.5 {
                let twigletCount = 2 + Int(rng.unit() * 2.0)
                let twigTan = curveTipTangent(length: twigLen, curveAmount: curveAmt,
                                              curveAxis: cAxis)
                for j in 0..<twigletCount {
                    // Sprout from a point partway along the twig (arc-length
                    // distributed), angled off the twig axis.
                    let alongFrac = 0.45 + rng.unit() * 0.5
                    let sproutLocal = curveTipOffset(length: twigLen * alongFrac,
                                                     curveAmount: curveAmt,
                                                     curveAxis: cAxis)
                    let sproutWorld = twigWorld * translate(sproutLocal)
                                    * orientYTo(twigTan)
                    let tj = Float(j) / Float(twigletCount) * 2 * .pi
                           + (rng.unit() - 0.5) * 0.8
                    let tp = 0.45 + rng.unit() * 0.7
                    let tlen = twigLen * (0.35 + rng.unit() * 0.30)
                    // RACHIS-OCCLUSION: 4th-gen twiglets are the finest radiating
                    // rachis lines from above — keep them as a whisper so the leaf
                    // rosettes bury them. Half the prior gauge.
                    let tr0 = max(0.0005, twigR1 * (0.35 + rng.unit() * 0.18))
                    let twigletWorld = sproutWorld * rotY(tj) * rotX(tp)
                    emitBranchMesh(&s, world: twigletWorld, length: tlen,
                                   bottomR: tr0, topR: tr0 * 0.4, collarMult: 1.0,
                                   radialSegments: lod(5, site.detail),
                                   curveAmount: 0.10 + rng.unit() * 0.18,
                                   curveAxis: SIMD3(cos(tj), 0, sin(tj)),
                                   look: look, baseY: 0, seed: site.seed)
                    emitLeaves(&s, world: twigletWorld, twigLength: tlen,
                               leafSize: look.leafCardSize * (0.8 + rng.unit() * 0.25),
                               density: site.leafDensity * look.leafDensityMult * 1.15,
                               look: look, site: site, rng: &rng)
                }
            }
        }
    }

    /// BIRCH FORK-TIP ROSETTE (#20 structural item 2).
    /// Birch runs no central leader — its nearly-vertical primary branches ARE the
    /// fork, and once clamped to the crown boundary their last segment terminated
    /// at/above the leaf-fill shell, leaving a bare two-pronged armature on the
    /// skyline (the pale top-left tree in the side / three-quarter angles). This
    /// caps every birch fork terminus with a tight tuft of 3–5 leaf clusters so no
    /// fork tip is ever bare on the silhouette. The rosette is small and clings to
    /// the tip (it is NOT a fat radial cloud) — it just curtains the exposed wood.
    /// `tipWorld` already sits at the (clamped) branch tip with +Y along the tip
    /// tangent. Cards are sprayed into the forward hemisphere about that tangent.

    // ===== emitTipRosette (lines 3003-3092) =====
    public static func emitTipRosette(_ s: inout Soup, tipWorld: simd_float4x4,
                                       leafSize: Float, look: TreeLook,
                                       site: TreeSite, rng: inout ForestRNG) {
        let prevFoliage = s.foliageMark
        s.foliageMark = 1
        defer { s.foliageMark = prevFoliage }

        // 5–8 clusters × ~3 cards each gives a tuft dense enough to curtain the
        // fork wood on the silhouette (3–5 left bare gaps in the neon debug pass).
        let clusters = 5 + Int(rng.unit() * 4.0)        // 5–8 tip clusters
        let cardsPerCluster = 3
        let planeW = leafSize * look.leafAspectW
        let planeH = leafSize * look.leafAspectH
        let startAngle = rng.unit() * 2 * .pi
        for cc in 0..<(clusters * cardsPerCluster) {
            let c = cc / cardsPerCluster
            // Distribute clusters over the FULL sphere about the tip (incl. a little
            // below +Y) so the tuft caps the fork terminus from every angle and
            // drapes down over the exposed last segment, not just straight up.
            let yaw = startAngle + Float(c) / Float(clusters) * 2 * .pi
                    + (rng.unit() - 0.5) * 0.9
            let pitch = -0.35 + rng.unit() * 1.35       // down-and-around the tip
            let cosP = cos(pitch), sinP = sin(pitch)
            let petDir = simd_normalize(SIMD3<Float>(cos(yaw) * sinP, cosP, sin(yaw) * sinP))
            let reach = (0.20 + rng.unit() * 0.55) * leafSize
            let sizeJit = 0.70 + rng.unit() * 0.45
            let wLeaf = planeW * sizeJit, hLeaf = planeH * sizeJit
            let centerOffset = hLeaf * 0.5 + reach
            let leafPos = petDir * centerOffset
            let yAxis = petDir
            let upHint = SIMD3<Float>(0, 1, 0)
            let dotYU = simd_dot(yAxis, upHint)
            let zRaw = abs(dotYU) > 0.97
                ? simd_normalize(simd_cross(yAxis, SIMD3<Float>(1, 0, 0)))
                : simd_normalize(upHint - dotYU * yAxis)
            let xRaw = simd_normalize(simd_cross(yAxis, zRaw))
            let rollR = (rng.unit() - 0.5) * 2.0 * .pi
            let cr = cos(rollR), sr = sin(rollR)
            let zAxis = zRaw * cr - xRaw * sr
            let xAxis = zRaw * sr + xRaw * cr
            let wPos = xf(tipWorld, leafPos)
            let wX = xfDir(tipWorld, xAxis)
            let wY = xfDir(tipWorld, yAxis)
            let wZ = xfDir(tipWorld, zAxis)
            let outwardLocal = safeNormalize(petDir + SIMD3<Float>(0, 0.45, 0), fallback: petDir)
            let wOut = xfDir(tipWorld, outwardLocal)
            let bentN = safeNormalize(wOut * 0.72 + wZ * 0.28, fallback: wOut)
            let col = leafColorVary(look, frac: 1.0, rng: &rng)
            emitLeafCard(&s, pos: wPos, xAxis: wX, yAxis: wY, normal: bentN,
                         w: wLeaf, h: hLeaf, color: col,
                         species: look.species, detail: site.detail,
                         curl: look.leafCurl, rng: &rng)
        }
    }

}
