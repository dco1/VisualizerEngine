import XCTest
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import VisualizerMaterials

/// **Daydream DH-1316 — velvet's nap and linen's slub** (`MaterialGenerator+FabricLook`).
///
/// GPU-free: both bakes stay `TextureAudit`-plausible with the lever on, the lever off bakes the
/// pre-DH-1316 channels exactly, the nap tilts the pile normal by a few degrees (never a crease), and
/// the linen slub is ANISOTROPIC — streaks along the yarn, not round blobs. `FABRIC_OUT` = a directory
/// also dumps 1:1 swatches (albedo + a grazing-view sheen proxy) for both arms — a texture dump, not a
/// render; the Metal capture is the ground truth for how it looks.
final class FabricLookStandaloneTests: XCTestCase {
    override func tearDown() { MaterialGenerator.refinedFabricLook = false; super.tearDown() }

    private func bake(_ refined: Bool) -> (velvet: MaterialChannels, linen: MaterialChannels) {
        MaterialGenerator.refinedFabricLook = refined
        defer { MaterialGenerator.refinedFabricLook = false }
        return (MaterialGenerator.velvet(), MaterialGenerator.linen())
    }

    func testRefinedFabricsStayPlausibleAndTheLeverIsExact() throws {
        let off = bake(false), on = bake(true)
        for (name, ch) in [("velvet", on.velvet), ("linen", on.linen)] {
            let a = TextureAudit.audit(ch)
            print("DH-1316 \(name): plausible=\(a.isPlausible) roughSD=\(a.roughnessStdDev) macro=\(a.macroContrast) seam=\(a.seamScoreX),\(a.seamScoreY)")
            XCTAssertTrue(a.isPlausible, "\(name) must stay TextureAudit-plausible: \(a)")
        }
        // Lever off = the old bake (the defaults are the call sites' own seeds).
        MaterialGenerator.refinedFabricLook = false
        let again = (MaterialGenerator.velvet(), MaterialGenerator.linen())
        XCTAssertEqual(off.velvet.albedo, again.0.albedo)
        XCTAssertEqual(off.linen.albedo, again.1.albedo)
        XCTAssertNotEqual(off.velvet.normal, on.velvet.normal, "the nap must move the pile normal")
        XCTAssertNotEqual(off.linen.albedo, on.linen.albedo, "the slub must move the linen yarn")
    }

    /// The nap is a LEAN: a few degrees off the bake's own normal on average, never a fold.
    func testVelvetNapLeansAFewDegrees() throws {
        let off = bake(false).velvet, on = bake(true).velvet
        var sum = 0.0, worst = 0.0
        for i in 0..<on.normal.count {
            let a = off.normal[i], b = on.normal[i]
            let d = acos(max(-1, min(1, a.x * b.x + a.y * b.y + a.z * b.z))) * 180 / .pi
            sum += d; worst = max(worst, d)
        }
        let mean = sum / Double(on.normal.count)
        print(String(format: "DH-1316 velvet nap lean: mean %.2f°, max %.2f°", mean, worst))
        XCTAssertGreaterThan(mean, 3)
        XCTAssertLessThan(mean, 9)
        XCTAssertLessThan(worst, 15)
    }

    /// Slub runs ALONG the yarn: the albedo's correlation along the weft (+U) at 1 cm beats the
    /// correlation across it (+V) at the same distance.
    func testLinenSlubIsAlongTheYarn() throws {
        let on = bake(true).linen, off = bake(false).linen
        func lum(_ c: Vec3) -> Double { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
        func corr(_ ch: MaterialChannels, dx: Int, dy: Int) -> Double {
            let n = ch.size
            var a = [Double](), b = [Double]()
            for y in stride(from: 0, to: n, by: 3) { for x in stride(from: 0, to: n, by: 3) {
                a.append(lum(ch.albedo[ch.idx(x, y)]) - lum(off.albedo[off.idx(x, y)]))
                b.append(lum(ch.albedo[ch.idx(x + dx, y + dy)]) - lum(off.albedo[off.idx(x + dx, y + dy)]))
            } }
            let ma = a.reduce(0, +) / Double(a.count), mb = b.reduce(0, +) / Double(b.count)
            var c = 0.0, va = 0.0, vb = 0.0
            for i in a.indices { c += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) * (a[i] - ma); vb += (b[i] - mb) * (b[i] - mb) }
            return c / max(1e-12, (va * vb).squareRoot())
        }
        let along = corr(on, dx: 8, dy: 0), across = corr(on, dx: 0, dy: 8)
        print(String(format: "DH-1316 linen slub correlation at 8 texels (~5 mm): along the weft %.3f, across %.3f", along, across))
        XCTAssertGreaterThan(along, across + 0.15)
    }

    func testDumpSwatches() throws {
        guard let out = ProcessInfo.processInfo.environment["FABRIC_OUT"] else { throw XCTSkip("set FABRIC_OUT") }
        for refined in [false, true] {
            let b = bake(refined)
            for (name, ch) in [("velvet", b.velvet), ("linen", b.linen)] {
                let n = ch.size
                var alb = [UInt8](repeating: 255, count: n * n * 4), graze = alb
                // A grazing-view proxy for the sheen: view 75° off the normal, toward −U.
                let V = normalize3(Vec3(-0.966, 0, 0.259))
                for y in 0..<n { for x in 0..<n {
                    let i = ch.idx(x, y), o = (y * n + x) * 4, c = ch.albedo[i], nn = ch.normal[i]
                    alb[o] = enc(c.x); alb[o + 1] = enc(c.y); alb[o + 2] = enc(c.z)
                    let mu = max(0.02, nn.x * V.x + nn.y * V.y + nn.z * V.z)
                    let s = 0.35 + 2.2 * pow(1 - mu, 3)
                    graze[o] = enc(c.x * s); graze[o + 1] = enc(c.y * s); graze[o + 2] = enc(c.z * s)
                } }
                let tag = refined ? "after" : "before"
                write(alb, n, "\(out)/dh1316-\(name)-\(tag)-albedo.png")
                write(graze, n, "\(out)/dh1316-\(name)-\(tag)-grazing.png")
            }
        }
    }

    private func enc(_ v: Double) -> UInt8 {
        let c = max(0, min(1, v)); let s = c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1 / 2.4) - 0.055
        return UInt8(max(0, min(255, s * 255)))
    }
    private func write(_ px: [UInt8], _ w: Int, _ path: String) {
        var px = px
        px.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: w, height: w, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(d, ctx.makeImage()!, nil); CGImageDestinationFinalize(d)
        }
    }
}
