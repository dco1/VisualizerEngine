import XCTest
import Foundation
import simd
@testable import VisualizerRendering

/// Gates for the two vegetation render-lifecycle helpers — the bake cache that prunes, and the
/// foliage colour/backlight stamp. Both replaced hand-written copies that had already drifted in a
/// shipping app, so what is asserted here is mostly "the copies' behaviour was preserved exactly".
/// GPU-free.
final class VegetationGlueTests: XCTestCase {

    // MARK: - KeyedMeshCache

    func testHitDoesNotPruneAndDoesNotBuild() {
        var cache = KeyedMeshCache<Int>()
        var builds = 0
        _ = cache.value(forKey: "a", live: { ["a"] }, build: { builds += 1; return 1 })

        // A second fetch of a live key must be a dictionary lookup and nothing else — no rebuild,
        // and crucially no live-set walk, which is why `live` is a closure rather than a value.
        var liveWalks = 0
        _ = cache.value(forKey: "a", live: { liveWalks += 1; return ["a"] }, build: { builds += 1; return 9 })
        XCTAssertEqual(builds, 1, "a cache hit rebuilt")
        XCTAssertEqual(liveWalks, 0, "a cache hit walked the live set — that is the hot path")
    }

    func testMissEvictsEntriesThatAreNoLongerLive() {
        var cache = KeyedMeshCache<Int>()
        _ = cache.value(forKey: "old", live: { ["old"] }, build: { 1 })
        XCTAssertNotNil(cache["old"])

        // "old" is gone from the document; fetching a new key must drop it.
        _ = cache.value(forKey: "new", live: { ["new"] }, build: { 2 })
        XCTAssertNil(cache["old"], "a stale bake survived a miss — this is the leak")
        XCTAssertNotNil(cache["new"])
        XCTAssertEqual(cache.count, 1)
    }

    func testTheKeyBeingFetchedIsRetainedEvenWhenTheLiveSetOmitsIt() {
        // A bake in progress is by definition about to be used — an element mid-drag is often in
        // neither the committed document nor the edit snapshot the live set was built from.
        var cache = KeyedMeshCache<Int>()
        let got = cache.value(forKey: "inflight", live: { [] }, build: { 7 })
        XCTAssertEqual(got, 7)
        XCTAssertEqual(cache["inflight"], 7, "the key being built was pruned by its own miss")
    }

    func testAFailedBuildIsNotCached() {
        // A degenerate element (no geometry) must not store a nil that shadows a later fix to its
        // parameters.
        var cache = KeyedMeshCache<Int>()
        XCTAssertNil(cache.value(forKey: "k", live: { [] }, build: { nil }))
        XCTAssertEqual(cache.count, 0)

        var builds = 0
        _ = cache.value(forKey: "k", live: { [] }, build: { builds += 1; return 3 })
        XCTAssertEqual(builds, 1, "a previously-failed key was not retried")
        XCTAssertEqual(cache["k"], 3)
    }

    /// **The DH-0659 shape.** A key built from raw unquantized slider values mints a fresh key every
    /// tick. Under the old hand-written caches that meant one registered mesh set pinned per
    /// intermediate value for the life of the bridge; the potted-plant cache was the one of six
    /// whose author never copied the eviction block. Through this type it cannot happen: only one
    /// value is live at a time, so the cache must stay at 1.
    func testASliderDragDoesNotGrowTheCache() {
        var cache = KeyedMeshCache<Int>()
        for tick in 0 ..< 240 {
            let key = "plant-size-\(Double(tick) * 0.00317)"   // a fresh key every tick
            _ = cache.value(forKey: key, live: { [key] }, build: { tick })
        }
        XCTAssertEqual(cache.count, 1,
                       "the cache grew with the drag — \(cache.count) entries pinned")
    }

    func testRemoveAllDropsEverything() {
        var cache = KeyedMeshCache<Int>()
        _ = cache.value(forKey: "a", live: { ["a"] }, build: { 1 })
        cache.removeAll()
        XCTAssertEqual(cache.count, 0)
    }

    // MARK: - FoliageVertexStamp

    /// The alpha is the thin-sheet subsurface flag, not decoration: an interior plant stamped
    /// translucent out-glows the room it stands in under a low sun. Pin both constants.
    func testBacklightAlphaConstants() {
        XCTAssertEqual(FoliageVertexStamp.Backlight.translucent.alpha, 0)
        XCTAssertEqual(FoliageVertexStamp.Backlight.opaque.alpha, 1)
    }

    /// **`shadeJitter` must reproduce the app-side generator bit-exactly**, or every plant in every
    /// document re-shades the moment the stamp moves into the engine. This re-implements the app's
    /// `HouseRenderBridge.SplitMix64` verbatim — including the `seed &* 0xD1B5… &+ 1` pre-mix that
    /// was written at the call sites, the zero-guard, and the golden-ratio advance BEFORE the mix —
    /// and asserts equality across a seed sweep. Three of those four details change the stream if
    /// dropped, and dropping one is exactly what a careless port does.
    func testShadeJitterMatchesTheAppGeneratorBitExactly() {
        struct LegacySplitMix64 {
            var state: UInt64
            init(_ seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
            mutating func next() -> UInt64 {
                state = state &+ 0x9E3779B97F4A7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
                z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
                return z ^ (z >> 31)
            }
            mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9007199254740992.0) }
        }

        // A WIDE sweep, not a handful of seeds. The first version of this test sampled nine and
        // passed while the implementation was one Float ULP off for ~5% of inputs — `spread` was
        // typed `Float`, so `0.07` widened to 0.07000000029802322. Nine samples is a 63% chance of
        // missing a 5% divergence; 50k makes it a certainty to catch.
        var mismatches = 0
        var worst: (seed: UInt64, got: Float, want: Float) = (0, 0, 0)
        for seed in stride(from: UInt64(0), to: 50_000, by: 1) {
            var rng = LegacySplitMix64(seed &* 0xD1B54A32D192ED03 &+ 1)
            let expected = Float(rng.unit() * 0.14 - 0.07)   // the literal the call sites used
            let got = FoliageVertexStamp.shadeJitter(seed: seed)
            if got != expected {
                mismatches += 1
                if abs(got - expected) > abs(worst.got - worst.want) { worst = (seed, got, expected) }
            }
        }
        XCTAssertEqual(mismatches, 0,
            "shade jitter differs from the app generator for \(mismatches)/50000 seeds — worst at "
            + "seed \(worst.seed): got \(worst.got), want \(worst.want). Every plant in every "
            + "document re-shades if this drifts.")

        // The extremes and a few hand-picked values, so a failure names something legible.
        for seed in [UInt64(0), 1, 2, 7, 42, 1234, 0xDEAD_BEEF, .max / 3, .max] {
            var rng = LegacySplitMix64(seed &* 0xD1B54A32D192ED03 &+ 1)
            XCTAssertEqual(FoliageVertexStamp.shadeJitter(seed: seed),
                           Float(rng.unit() * 0.14 - 0.07),
                           "shade jitter drifted for seed \(seed)")
        }
    }

    func testShadeJitterIsDeterministicAndInRange() {
        for seed in stride(from: UInt64(0), to: 500, by: 7) {
            let j = FoliageVertexStamp.shadeJitter(seed: seed)
            XCTAssertEqual(j, FoliageVertexStamp.shadeJitter(seed: seed), "not deterministic")
            XCTAssertGreaterThanOrEqual(j, -0.07)
            XCTAssertLessThan(j, 0.07)
        }
        // Different seeds must actually separate, or every plant reads the same green.
        let sample = Set((0 ..< 64).map { FoliageVertexStamp.shadeJitter(seed: UInt64($0)) })
        XCTAssertGreaterThan(sample.count, 50, "shade jitter is barely varying across seeds")
    }

    func testStampWritesColourAndAlphaToEveryVertexAndClampsAtZero() {
        var verts = (0 ..< 5).map { i in
            IlluminatoramaVertex(position: SIMD3(Float(i), 0, 0),
                                 normal: SIMD3(0, 1, 0),
                                 uv: .zero)
        }
        // A jitter more negative than the darkest channel must clamp, not wrap to a huge value.
        FoliageVertexStamp.stamp(&verts,
                                 color: SIMD3(0.20, 0.42, 0.06),
                                 backlight: .translucent,
                                 jitter: -0.5)
        for v in verts {
            XCTAssertEqual(v.color.x, 0, accuracy: 1e-6)
            XCTAssertEqual(v.color.y, 0, accuracy: 1e-6)   // 0.42 - 0.5 clamps
            XCTAssertEqual(v.color.z, 0, accuracy: 1e-6)
            XCTAssertEqual(v.color.w, 0, "translucent must write alpha 0")
        }

        FoliageVertexStamp.stamp(&verts, color: SIMD3(0.2, 0.4, 0.1), backlight: .opaque)
        for v in verts {
            XCTAssertEqual(v.color.x, 0.2, accuracy: 1e-6)
            XCTAssertEqual(v.color.y, 0.4, accuracy: 1e-6)
            XCTAssertEqual(v.color.z, 0.1, accuracy: 1e-6)
            XCTAssertEqual(v.color.w, 1, "opaque must write alpha 1")
        }
    }

    func testStampOnAnEmptyGroupIsANoOp() {
        var verts: [IlluminatoramaVertex] = []
        FoliageVertexStamp.stamp(&verts, color: SIMD3(1, 0, 0), backlight: .opaque)
        XCTAssertTrue(verts.isEmpty)
    }
}
