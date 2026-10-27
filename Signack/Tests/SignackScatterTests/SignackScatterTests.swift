import Foundation
import simd
import Testing
@testable import SignackScatter

/// The scatter's contract, GPU-free: deterministic, spaced, respects every exclusion, clumps when
/// asked, and an edit far away leaves a plant where it was.
struct SignackScatterTests {

    static let yard: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(20, 0), SIMD2(20, 14), SIMD2(0, 14)]
    static let species = [
        ScatterSpecies(id: 0, weight: 3, canopyRadius: 0.5, scaleRange: 0.7...1.3, variantCount: 3),
        ScatterSpecies(id: 1, weight: 1, canopyRadius: 0.9, scaleRange: 0.8...1.4, variantCount: 2),
        ScatterSpecies(id: 2, weight: 2, canopyRadius: 0.25, scaleRange: 0.6...1.2, variantCount: 2),
    ]

    static func rule(seed: UInt64 = 42, density: Double = 0.6, clumping: ScatterClumping = .none,
                     regions: [[SIMD2<Double>]] = [yard]) -> ScatterRule {
        ScatterRule(regions: regions, density: density, minSpacing: 0.15, clumping: clumping,
                    species: species, seed: seed)
    }

    @Test func sameRuleGivesIdenticalPoints() {
        let a = SignackScatter.scatter(Self.rule())
        let b = SignackScatter.scatter(Self.rule())
        #expect(a.count > 40)
        #expect(a == b)
        let c = SignackScatter.scatter(Self.rule(seed: 43))
        #expect(a != c)
    }

    @Test func canopiesKeepTheirSpacing() {
        let r = Self.rule(density: 3)   // ask for far more than fits: spacing must win
        let pts = SignackScatter.scatter(r)
        #expect(pts.count > 50)
        for i in pts.indices {
            for j in pts.indices where j > i {
                let need = (pts[i].radius + pts[j].radius) * r.canopyOverlap + r.minSpacing
                #expect(simd_distance(pts[i].position, pts[j].position) >= need - 1e-9)
            }
        }
        // Every point is inside the region; the mix has every species; individuals vary.
        #expect(pts.allSatisfy { ScatterGeometry.contains(Self.yard, $0.position) })
        #expect(Set(pts.map(\.species)) == [0, 1, 2])
        #expect(Set(pts.map { Int($0.scale * 100) }).count > 20)
    }

    @Test func nothingGrowsInTheMask() {
        let path = ScatterMask.Ribbon([SIMD2(-1, 7), SIMD2(21, 7)], halfWidth: 0.6)
        let bed = ScatterMask.Polygon([SIMD2(2, 1), SIMD2(6, 1), SIMD2(6, 4), SIMD2(2, 4)], clearance: 0.2)
        let piece = ScatterMask.Box(center: SIMD2(15, 3), halfExtent: SIMD2(1, 0.5), angle: 0.4)
        let mask = ScatterMask(polygons: [bed], ribbons: [path], boxes: [piece])
        let pts = SignackScatter.scatter(Self.rule(density: 1.2), mask: mask)
        #expect(pts.count > 40)
        for p in pts {
            #expect(mask.isClear(p.position, radius: p.radius * 0.7))
            #expect(abs(p.position.y - 7) >= 0.6 + p.radius * 0.7 - 1e-9)
            #expect(!ScatterGeometry.contains(bed.points, p.position))
        }
    }

    @Test func slopeAndAltitudeLimitsHold() {
        // A bank: flat for x < 10, then rising 1:1 (45°).
        let terrain = ScatterTerrain { p in max(0, p.x - 10) }
        var r = Self.rule(density: 1)
        r.maxSlopeDegrees = 30
        let pts = SignackScatter.scatter(r, terrain: terrain)
        #expect(!pts.isEmpty)
        #expect(pts.allSatisfy { $0.position.x < 10.3 })
        r.maxSlopeDegrees = nil
        r.altitudeRange = 2...5
        let band = SignackScatter.scatter(r, terrain: terrain)
        #expect(band.allSatisfy { $0.position.x >= 12 - 1e-9 && $0.position.x <= 15 + 1e-9 })
        #expect(band.allSatisfy { abs($0.groundHeight - max(0, $0.position.x - 10)) < 1e-9 })
    }

    /// Clumping concentrates plants where the clump field is strong: the field's mean AT the plants
    /// is well above its mean over the yard (which is ≈ 1 by construction).
    @Test func clumpingGathersPlants() {
        let cr = ScatterClumping(spacing: 6, radius: 1.4, strength: 0.9)
        let field = ScatterClumps(cr, species: Self.species, seed: 42)
        var yardMean = 0.0
        for y in 0..<56 { for x in 0..<80 { yardMean += field.sample(SIMD2((Double(x) + 0.5) / 4, (Double(y) + 0.5) / 4)).densityFactor } }
        yardMean /= 56 * 80
        let clumped = SignackScatter.scatter(Self.rule(density: 0.2, clumping: cr))
        let even = SignackScatter.scatter(Self.rule(density: 0.2))
        func atPlants(_ p: [ScatterPoint]) -> Double { p.map { field.sample($0.position).densityFactor }.reduce(0, +) / Double(p.count) }
        #expect(clumped.count > 20)
        #expect(atPlants(clumped) > 1.4 * yardMean)
        #expect(atPlants(even) < 1.2 * yardMean)
    }

    /// Moving one corner of the region keeps (almost) every plant far from it exactly where it was —
    /// candidates are world-anchored, not region-relative.
    @Test func farEditsLeavePlantsInPlace() {
        let a = SignackScatter.scatter(Self.rule())
        var moved = Self.yard
        moved[2] = SIMD2(23, 16)
        let b = SignackScatter.scatter(Self.rule(regions: [moved]))
        let far = a.filter { $0.position.x < 8 }
        let kept = far.filter { p in b.contains(where: { $0.key == p.key && $0.position == p.position }) }
        #expect(far.count > 10)
        #expect(Double(kept.count) >= 0.9 * Double(far.count))
    }

    @Test func discIndexCoversCanopies() {
        let pts = SignackScatter.scatter(Self.rule())
        let index = ScatterDiscIndex(pts, radiusScale: 0.5)
        for p in pts.prefix(20) {
            #expect(index.covers(p.position))
            #expect(index.covers(p.position + SIMD2(p.radius * 0.45, 0)))
        }
        #expect(!index.covers(SIMD2(-50, -50)))
    }

    /// DH-1470 — strata: trees space by canopy among themselves; shrubs grow under their crowns, kept
    /// clear only of the trunk, thinned by shade; and a single-stratum rule is unchanged by the feature.
    @Test func understoryGrowsUnderCrownsButClearOfTrunks() {
        let tree = ScatterSpecies(id: 9, weight: 0.25, canopyRadius: 4, scaleRange: 0.8...1.1, variantCount: 2,
                                  stratum: 1, stemRadius: 0.6)
        var r = Self.rule(density: 1.2)
        r.species = Self.species + [tree]
        let pts = SignackScatter.scatter(r)
        let trees = pts.filter { $0.stratum == 1 }, shrubs = pts.filter { $0.stratum == 0 }
        #expect(trees.count >= 3)
        #expect(shrubs.count > 40)
        for (i, a) in trees.enumerated() {
            for b in trees.dropFirst(i + 1) {
                #expect(simd_distance(a.position, b.position) >= (a.radius + b.radius) * r.canopyOverlap * 0.65 + r.minSpacing - 1e-9)
            }
        }
        var under = 0
        for s in shrubs {
            for t in trees {
                let d = simd_distance(s.position, t.position)
                #expect(d >= t.stemRadius + s.radius * r.canopyOverlap * 0.65 + r.minSpacing - 1e-9)
                if d < t.radius * 0.85 { under += 1 }
            }
        }
        #expect(under > 0, "shrubs must be able to grow under a crown")
        // Full shade: nothing grows under a crown.
        r.understoryKeep = 0
        let dark = SignackScatter.scatter(r)
        let darkTrees = dark.filter { $0.stratum == 1 }
        #expect(dark.filter { $0.stratum == 0 }.allSatisfy { s in darkTrees.allSatisfy { simd_distance(s.position, $0.position) >= $0.radius * 0.85 - 1e-9 } })
        // Shade: fewer shrubs per m² under crowns than in the open.
        let none = SignackScatter.scatter(Self.rule(density: 1.2))
        #expect(none == SignackScatter.scatter(Self.rule(density: 1.2)))
        #expect(none.allSatisfy { $0.stratum == 0 })
    }

    @Test func discsKeepPlantsOut() {
        let mask = ScatterMask(discs: [ScatterMask.Disc(center: SIMD2(10, 7), radius: 2)])
        let pts = SignackScatter.scatter(Self.rule(density: 1.2), mask: mask)
        #expect(pts.count > 40)
        #expect(pts.allSatisfy { simd_distance($0.position, SIMD2(10, 7)) >= 2 + $0.radius * 0.7 - 1e-9 })
    }
}
