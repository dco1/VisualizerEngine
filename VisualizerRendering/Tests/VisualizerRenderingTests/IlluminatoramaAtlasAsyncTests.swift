import XCTest
import Metal
import CoreGraphics
@testable import VisualizerRendering

/// `registerAsync` draws a slice off the main thread; what lands in the atlas must be what
/// `register` puts there, byte for byte, at every mip level — with its mean and UV scale.
@MainActor
final class IlluminatoramaAtlasAsyncTests: XCTestCase {

    private func device() throws -> MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal device on this host") }
        return d
    }

    /// A gradient with a diagonal stripe, `w × h`: every level differs and a non-square one letterboxes.
    private func image(_ w: Int, _ h: Int) -> CGImage {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        for y in 0..<h {
            for x in 0..<w {
                let stripe = (x + y) % 17 < 3
                ctx.setFillColor(CGColor(red: CGFloat(x) / CGFloat(w), green: stripe ? 1 : 0.2,
                                         blue: CGFloat(y) / CGFloat(h), alpha: 1))
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        return ctx.makeImage()!
    }

    private func bytes(_ atlas: IlluminatoramaTextureAtlas, slice: Int, level: Int) -> [UInt8] {
        let n = max(1, atlas.sliceSize >> level)
        var out = [UInt8](repeating: 0, count: n * n * 4)
        out.withUnsafeMutableBytes { raw in
            atlas.texture.getBytes(raw.baseAddress!, bytesPerRow: n * 4, bytesPerImage: n * n * 4,
                                   from: MTLRegionMake3D(0, 0, 0, n, n, 1), mipmapLevel: level, slice: slice)
        }
        return out
    }

    private func assertSame(_ a: IlluminatoramaTextureAtlas, _ sa: Int32, _ b: IlluminatoramaTextureAtlas, _ sb: Int32,
                            file: StaticString = #filePath, line: UInt = #line) {
        for level in 0..<a.mipLevels {
            XCTAssertEqual(bytes(a, slice: Int(sa), level: level), bytes(b, slice: Int(sb), level: level),
                           "mip \(level) differs", file: file, line: line)
        }
        let ma = a.sliceMeanBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: a.capacity)[Int(sa)]
        let mb = b.sliceMeanBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: b.capacity)[Int(sb)]
        XCTAssertEqual(ma, mb, "mean differs", file: file, line: line)
        XCTAssertEqual(ma.w, 1, file: file, line: line)
        let ua = a.uvScaleBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: a.capacity)[Int(sa)]
        let ub = b.uvScaleBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: b.capacity)[Int(sb)]
        XCTAssertEqual(ua, ub, "uv scale differs", file: file, line: line)
    }

    func testAsyncSliceIsTheSyncSliceByteForByte() async throws {
        let d = try device()
        for format in [MTLPixelFormat.bgra8Unorm_srgb, .bgra8Unorm] {
            for (w, h) in [(96, 96), (128, 32), (24, 80)] {
                let img = image(w, h)
                let sync = try IlluminatoramaTextureAtlas(device: d, pixelFormat: format, sliceSize: 64, capacity: 2)
                let async = try IlluminatoramaTextureAtlas(device: d, pixelFormat: format, sliceSize: 64, capacity: 2)
                let s = try XCTUnwrap(sync.register(contents: img as AnyObject))
                let a = try XCTUnwrapAsync(await async.registerAsync(contents: img as AnyObject))
                assertSame(sync, s, async, a)
            }
        }
    }

    func testAsyncIsCachedAndSharesASlot() async throws {
        let atlas = try IlluminatoramaTextureAtlas(device: device(), sliceSize: 32, capacity: 2)
        let img = image(32, 32)
        async let first = atlas.registerAsync(contents: img as AnyObject)
        async let second = atlas.registerAsync(contents: img as AnyObject)
        let (a, b) = await (first, second)
        XCTAssertNotNil(a)
        XCTAssertEqual(a, b, "two registrations of one image share its slot")
        XCTAssertEqual(atlas.register(contents: img as AnyObject), a, "the sync path finds the async one's slot")
        XCTAssertEqual(atlas.nextSlice, 1, "only one slot taken")
    }

    func testAsyncGrowsAndReusesFreedSlots() async throws {
        let atlas = try IlluminatoramaTextureAtlas(device: device(), sliceSize: 16, capacity: 1)
        let a = await atlas.registerAsync(contents: image(16, 16) as AnyObject)
        let b = await atlas.registerAsync(contents: image(17, 17) as AnyObject)
        XCTAssertEqual(a, 0)
        XCTAssertEqual(b, 1, "a full atlas grows")
        XCTAssertGreaterThanOrEqual(atlas.capacity, 2)
        atlas.freeSlice(0)
        let c = await atlas.registerAsync(contents: image(18, 18) as AnyObject)
        XCTAssertEqual(c, 0, "a freed slot is taken first")
    }

    func testNothingUsableRegistersNothing() async throws {
        let atlas = try IlluminatoramaTextureAtlas(device: device(), sliceSize: 16, capacity: 1)
        let none = await atlas.registerAsync(contents: nil)
        let text = await atlas.registerAsync(contents: "not an image")
        XCTAssertNil(none)
        XCTAssertNil(text)
        XCTAssertEqual(atlas.nextSlice, 0)
    }
}

/// `XCTUnwrap` for an awaited value (XCTUnwrap's autoclosure can't hold an `await`).
private func XCTUnwrapAsync<T>(_ value: T?, file: StaticString = #filePath, line: UInt = #line) throws -> T {
    try XCTUnwrap(value, file: file, line: line)
}
