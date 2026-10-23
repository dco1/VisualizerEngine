import XCTest
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import VisualizerMaterials

/// Dumps tile swatches (albedo, roughness, raking-lit) so the realism pass can be looked at without
/// the GPU. A texture dump, NOT a render — the Metal capture is the ground truth for how it looks.
/// `TILE_OUT` = directory, `TILE_LABEL` = file prefix. Skips unless `TILE_OUT` is set.
final class TileRealismPreviewTests: XCTestCase {
    private func write(_ px: [UInt8], _ w: Int, _ h: Int, _ path: String) {
        var px = px
        px.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                    UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(d, ctx.makeImage()!, nil); CGImageDestinationFinalize(d)
        }
    }
    private func enc(_ v: Double) -> UInt8 {
        let c = max(0, min(1, v)); let s = c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1 / 2.4) - 0.055
        return UInt8(max(0, min(255, s * 255)))
    }

    static let cases: [(String, TileParams)] = [
        ("floor-satin", TileParams(pattern: UnitPatternParams(shape: .square, unitSizeMeters: 0.3048, bond: .stack,
                bodyColor: Vec3(0.80, 0.78, 0.72), jointColor: Vec3(0.95, 0.94, 0.91), jointFraction: 0.05,
                toneVariation: 0.10, seed: 11), finish: .satin, edge: .square, face: .procedural)),
        ("subway-gloss", TileParams(pattern: UnitPatternParams(shape: .rectangle, unitSizeMeters: 0.1524, bond: .running,
                bodyColor: Vec3(0.55, 0.64, 0.62), jointColor: Vec3(0.78, 0.77, 0.74), jointFraction: 0.04,
                toneVariation: 0.12, seed: 13), finish: .gloss, edge: .square, face: .procedural)),
        ("herringbone-matte", TileParams(pattern: UnitPatternParams(shape: .rectangle, unitSizeMeters: 0.1016, bond: .herringbone,
                bodyColor: Vec3(0.62, 0.60, 0.56), jointColor: Vec3(0.55, 0.54, 0.52), jointFraction: 0.05,
                toneVariation: 0.14, seed: 5), finish: .matte, edge: .square, face: .procedural)),
    ]

    func testDumpSwatches() throws {
        guard let out = ProcessInfo.processInfo.environment["TILE_OUT"] else { throw XCTSkip("set TILE_OUT") }
        let label = ProcessInfo.processInfo.environment["TILE_LABEL"] ?? "after"
        for (name, params) in Self.cases {
            let t0 = Date()
            let ch = MaterialGenerator.tile(size: 512, params: params, tiling: .librarySwatch)
            let ms = Date().timeIntervalSince(t0) * 1000
            let a = TextureAudit.audit(ch)
            print("TILE \(label) \(name) bake=\(Int(ms))ms plausible=\(a.isPlausible) seam=\(a.seamScoreX),\(a.seamScoreY) roughSD=\(a.roughnessStdDev) macro=\(a.macroContrast)")
            let n = ch.size, W = n   // one bake, 1:1 — grain is judged at texel scale
            var alb = [UInt8](repeating: 255, count: W * W * 4), rgh = alb, lit = alb
            let L = normalize3(Vec3(-0.92, 0.25, 0.30))
            for y in 0..<W { for x in 0..<W {
                let i = ch.idx(x, y), o = (y * W + x) * 4
                let c = ch.albedo[i]
                alb[o] = enc(c.x); alb[o + 1] = enc(c.y); alb[o + 2] = enc(c.z)
                let r = enc(ch.roughness[i]); rgh[o] = r; rgh[o + 1] = r; rgh[o + 2] = r
                let nn = ch.normal[i]
                let s = max(0, nn.x * L.x + nn.y * L.y + nn.z * L.z) * 1.6 + 0.10
                lit[o] = enc(c.x * s); lit[o + 1] = enc(c.y * s); lit[o + 2] = enc(c.z * s)
            } }
            write(alb, W, W, "\(out)/tile-\(label)-\(name)-albedo.png")
            write(rgh, W, W, "\(out)/tile-\(label)-\(name)-roughness.png")
            write(lit, W, W, "\(out)/tile-\(label)-\(name)-raking.png")
        }
    }
}
