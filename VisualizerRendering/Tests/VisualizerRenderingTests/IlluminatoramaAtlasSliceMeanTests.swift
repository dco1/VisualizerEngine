import XCTest
import Metal
import CoreGraphics
@testable import VisualizerRendering

/// S2.5 — the per-slice mean table that feeds the variance-preserving hex blend.
///
/// The blend restores `σ²` by rescaling each tap's deviation from **μ**, so μ has to be the
/// mean of the distribution the sampler actually returns. For the albedo atlas that is a
/// `_srgb` texture view, i.e. the sampler returns *linear* values — and `mean(linear) ≠
/// linear(mean_srgb)`. The gap is not academic: for a half-black/half-white tile it is
/// 0.500 vs 0.214, so taking μ from the coarsest mip (which is what Mikkelsen's paper does,
/// on a linear texture) would rescale every de-repeated surface around a μ that is less than
/// half the true one — a large, systematic brightening. This pins the linear-space claim.
@MainActor
final class IlluminatoramaAtlasSliceMeanTests: XCTestCase {

    private func device() throws -> MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device on this host")
        }
        return d
    }

    /// A square image whose left half is black and right half is white — sRGB mean byte
    /// 127.5 (linear 0.214), true linear mean 0.500. The two answers are far apart.
    private func halfBlackHalfWhite(_ n: Int = 64) -> CGImage {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: nil, width: n, height: n, bitsPerComponent: 8,
                            bytesPerRow: n * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: n / 2, height: n))
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: n / 2, y: 0, width: n / 2, height: n))
        return ctx.makeImage()!
    }

    private func mean(_ atlas: IlluminatoramaTextureAtlas, _ slice: Int) -> SIMD4<Float> {
        atlas.sliceMeanBuffer.contents()
            .bindMemory(to: SIMD4<Float>.self, capacity: atlas.capacity)[slice]
    }

    func testAlbedoSliceMeanIsTheLINEARMeanNotTheSRGBMean() throws {
        let atlas = try IlluminatoramaTextureAtlas(device: device(),
                                                   pixelFormat: .bgra8Unorm_srgb,
                                                   sliceSize: 128, capacity: 4)
        guard let slice = atlas.register(contents: halfBlackHalfWhite() as AnyObject) else {
            return XCTFail("registration failed")
        }
        let mu = mean(atlas, Int(slice))
        XCTAssertEqual(mu.w, 1, "a host-uploaded slice must record its mean as VALID")
        // Linear: (0 + 1)/2. A few interpolated texels at the black/white seam pull it a
        // hair low, hence the 0.03 band — which is still 10× clear of the sRGB answer.
        XCTAssertEqual(mu.x, 0.5, accuracy: 0.03,
                       "slice mean must be mean(linear); got \(mu.x) (srgbToLinear(mean_srgb) would be ≈0.214)")
        XCTAssertEqual(mu.y, 0.5, accuracy: 0.03)
        XCTAssertEqual(mu.z, 0.5, accuracy: 0.03)
    }

    func testNonColorAtlasMeanIsRawNotLinearised() throws {
        let atlas = try IlluminatoramaTextureAtlas(device: device(),
                                                   pixelFormat: .bgra8Unorm,
                                                   sliceSize: 128, capacity: 4)
        guard let slice = atlas.register(contents: halfBlackHalfWhite() as AnyObject) else {
            return XCTFail("registration failed")
        }
        // A non-sRGB atlas hands the shader raw [0,1] bytes, so μ is the raw mean — which
        // for this image is also 0.5, but it must arrive WITHOUT an sRGB decode applied.
        let mu = mean(atlas, Int(slice))
        XCTAssertEqual(mu.w, 1)
        XCTAssertEqual(mu.x, 0.5, accuracy: 0.03,
                       "the non-colour atlas must not linearise; got \(mu.x)")
    }

    func testUnwrittenSlotsReadAsUnknownSoTheBlendStaysLinear() throws {
        let atlas = try IlluminatoramaTextureAtlas(device: device(),
                                                   pixelFormat: .bgra8Unorm_srgb,
                                                   sliceSize: 64, capacity: 4)
        for i in 0..<4 {
            XCTAssertEqual(mean(atlas, i).w, 0,
                           "slot \(i) must default to 'no mean recorded' — the shader then blends "
                           + "linearly instead of rescaling around a bogus μ")
        }
        guard let live = atlas.reserveLiveSlice() else { return XCTFail("reserve failed") }
        XCTAssertEqual(mean(atlas, Int(live)).w, 0,
                       "a live-blitted slice's contents never pass through the host — its μ must stay unknown")
    }

    func testGrowCarriesTheMeanTableForward() throws {
        // capacity 1 forces a grow on the second registration.
        let atlas = try IlluminatoramaTextureAtlas(device: device(),
                                                   pixelFormat: .bgra8Unorm_srgb,
                                                   sliceSize: 64, capacity: 1)
        guard let s0 = atlas.register(contents: halfBlackHalfWhite() as AnyObject) else {
            return XCTFail("first registration failed")
        }
        let before = mean(atlas, Int(s0))
        guard let s1 = atlas.register(contents: halfBlackHalfWhite(32) as AnyObject) else {
            return XCTFail("second registration failed (grow)")
        }
        XCTAssertGreaterThan(atlas.capacity, 1, "the atlas should have grown")
        XCTAssertNotEqual(s0, s1)
        let after = mean(atlas, Int(s0))
        XCTAssertEqual(after.w, 1, "grow() dropped the mean table — slice \(s0) reads as unknown after growth")
        XCTAssertEqual(after.x, before.x, accuracy: 1e-6)
        XCTAssertEqual(mean(atlas, Int(s1)).w, 1)
    }

    /// Companion to the grow test above, for the OTHER thing `grow()` used to drop.
    /// It copied `sourceLevel: 0` only, so every already-registered slice came out of a
    /// growth with an undefined mip chain — silently discarding S1.1's whole point on any
    /// scene big enough to exceed the initial capacity.
    func testGrowCarriesTheFullMipChain() throws {
        let dev = try device()
        let atlas = try IlluminatoramaTextureAtlas(device: dev,
                                                   pixelFormat: .bgra8Unorm,
                                                   sliceSize: 64, capacity: 1)
        guard let s0 = atlas.register(contents: halfBlackHalfWhite() as AnyObject) else {
            return XCTFail("first registration failed")
        }
        // Read the coarsest (1×1) level of slice 0 before and after the grow.
        func topMipByte() -> [UInt8] {
            var px = [UInt8](repeating: 0, count: 4)
            atlas.texture.getBytes(&px, bytesPerRow: 4, bytesPerImage: 4,
                                   from: MTLRegionMake2D(0, 0, 1, 1),
                                   mipmapLevel: atlas.mipLevels - 1, slice: Int(s0))
            return px
        }
        let before = topMipByte()
        _ = atlas.register(contents: halfBlackHalfWhite(32) as AnyObject)
        XCTAssertGreaterThan(atlas.capacity, 1, "the atlas should have grown")
        XCTAssertEqual(topMipByte(), before,
                       "grow() left slice \(s0)'s coarsest mip undefined — it copied level 0 only")
    }
}
