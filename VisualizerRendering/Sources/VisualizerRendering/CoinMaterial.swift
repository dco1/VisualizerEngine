import Foundation
import SceneKit
import AppKit
import CoreGraphics

// ── CoinMaterial ──────────────────────────────────────────────────────────────
//
// The hero gold-token material + the procedural face stamp embossed INTO the
// metal (configurable text, default "TOKEN", as the arcade token's struck design). PBR
// values are from the Coin Pusher research: metalness 1, base colour the linear
// gold F0 (1.0, 0.78, 0.34) — set here as its sRGB equivalent #FFE29B for the
// SceneKit material — and a worn roughness ~0.4 broken up by the stamp.
//
// The relief is a generated normal map (height field → Sobel), not displaced
// geometry: it reads identically across thousands of instances at zero geometry
// cost, and the stamp lands only on the coin faces because CoinMesh maps the rim
// to a flat texel (see `faceUVRadius` there).

public enum CoinMaterial {

    /// Build the gold token material with the embossed `text` (default "TOKEN").
    public static func make(text: String = "TOKEN", textureSize: Int = 512) -> SCNMaterial {
        let (normal, roughness) = faceMaps(text: text, size: textureSize)
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.name = "CoinGold"
        // sRGB equivalent of linear gold F0 (1.0, 0.78, 0.34) ≈ #FFE29B.
        m.diffuse.contents = NSColor(srgbRed: 1.0, green: 0.886, blue: 0.608, alpha: 1.0)
        m.metalness.contents = 1.0
        m.roughness.contents = roughness
        m.normal.contents = normal
        m.normal.intensity = 0.85
        // Vertex-colour stream carries per-coin gold variation (baked at spawn).
        return m
    }

    // ── Procedural face stamp ─────────────────────────────────────────────────

    /// Returns (normalMap, roughnessMap) for the coin face. The textured content
    /// lives in a central disk (radius ≈ 0.45 in UV, matching CoinMesh); the
    /// border is flat so the rim stays smooth.
    static func faceMaps(text: String, size S: Int) -> (NSImage, NSImage) {
        let height = heightField(text: text, size: S)
        return (normalMap(from: height, size: S, strength: 2.4),
                roughnessMap(from: height, size: S))
    }

    /// Render the struck relief as a luminance height field in [0,1]: a beaded
    /// border ring + two ring lines + "36" over the lower word, then blurred so
    /// the normal slopes are smooth rather than cliff-edged. Internal so CoinMesh
    /// can sample the SAME field to displace the coin-face geometry (the emboss is
    /// geometric, since native Illuminatorama doesn't sample normal maps).
    static func heightField(text: String, size S: Int) -> [Float] {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8,
                                  bytesPerRow: S * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return [Float](repeating: 0, count: S * S) }

        let f = CGFloat(S)
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))   // base height 0
        ctx.fill(CGRect(x: 0, y: 0, width: f, height: f))

        let cx = f * 0.5, cy = f * 0.5
        let R = f * 0.45

        // Two ring lines just inside the face disk.
        ctx.setStrokeColor(CGColor(red: 0.7, green: 0.7, blue: 0.7, alpha: 1))
        ctx.setLineWidth(f * 0.012)
        ctx.addArc(center: CGPoint(x: cx, y: cy), radius: R * 0.96, startAngle: 0, endAngle: .pi*2, clockwise: false)
        ctx.strokePath()
        ctx.setLineWidth(f * 0.008)
        ctx.addArc(center: CGPoint(x: cx, y: cy), radius: R * 0.88, startAngle: 0, endAngle: .pi*2, clockwise: false)
        ctx.strokePath()

        // Beaded border between the two ring lines.
        ctx.setFillColor(CGColor(red: 0.85, green: 0.85, blue: 0.85, alpha: 1))
        let beadR = R * 0.92, beadCount = 48
        for i in 0..<beadCount {
            let a = CGFloat(i) / CGFloat(beadCount) * .pi * 2
            let bx = cx + cos(a) * beadR, by = cy + sin(a) * beadR
            let r = f * 0.012
            ctx.fillEllipse(in: CGRect(x: bx - r, y: by - r, width: r*2, height: r*2))
        }

        // Text: split into a top token "36" and the lower word(s), struck raised.
        let parts = text.split(separator: " ", maxSplits: 1).map(String.init)
        let top = parts.first ?? text
        let bottom = parts.count > 1 ? parts[1] : ""

        let ns = NSGraphicsContext(cgContext: ctx, flipped: false)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = ns
        func draw(_ s: String, fontSize: CGFloat, yCenterFrac: CGFloat) {
            guard !s.isEmpty else { return }
            let font = NSFont.systemFont(ofSize: fontSize, weight: .heavy)
            let attrs: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: NSColor(white: 0.95, alpha: 1.0),
                .kern: fontSize * 0.04,
            ]
            let str = NSAttributedString(string: s, attributes: attrs)
            let sz = str.size()
            str.draw(at: CGPoint(x: cx - sz.width/2, y: f * yCenterFrac - sz.height/2))
        }
        draw(top, fontSize: f * 0.30, yCenterFrac: 0.60)
        draw(bottom, fontSize: f * 0.20, yCenterFrac: 0.34)
        NSGraphicsContext.restoreGraphicsState()

        guard let data = ctx.data else { return [Float](repeating: 0, count: S * S) }
        let ptr = data.bindMemory(to: UInt8.self, capacity: S * S * 4)
        var h = [Float](repeating: 0, count: S * S)
        for i in 0..<(S * S) {
            // luminance of the premultiplied-RGBA pixel (alpha is 1 everywhere)
            let r = Float(ptr[i*4]), g = Float(ptr[i*4+1]), b = Float(ptr[i*4+2])
            h[i] = (0.299*r + 0.587*g + 0.114*b) / 255.0
        }
        return boxBlur(h, size: S, radius: 2)
    }

    private static func boxBlur(_ src: [Float], size S: Int, radius: Int) -> [Float] {
        guard radius > 0 else { return src }
        var tmp = [Float](repeating: 0, count: S * S)
        var out = [Float](repeating: 0, count: S * S)
        let norm = Float(2 * radius + 1)
        // horizontal
        for y in 0..<S {
            for x in 0..<S {
                var acc: Float = 0
                for k in -radius...radius {
                    let xx = min(S-1, max(0, x + k))
                    acc += src[y*S + xx]
                }
                tmp[y*S + x] = acc / norm
            }
        }
        // vertical
        for y in 0..<S {
            for x in 0..<S {
                var acc: Float = 0
                for k in -radius...radius {
                    let yy = min(S-1, max(0, y + k))
                    acc += tmp[yy*S + x]
                }
                out[y*S + x] = acc / norm
            }
        }
        return out
    }

    private static func normalMap(from h: [Float], size S: Int, strength: Float) -> NSImage {
        var px = [UInt8](repeating: 0, count: S * S * 4)
        for y in 0..<S {
            for x in 0..<S {
                let xl = h[y*S + max(0, x-1)], xr = h[y*S + min(S-1, x+1)]
                let yd = h[max(0, y-1)*S + x], yu = h[min(S-1, y+1)*S + x]
                let dx = (xr - xl) * strength
                let dy = (yu - yd) * strength
                var n = SIMD3<Float>(-dx, -dy, 1)
                n = n / max(1e-5, (n.x*n.x + n.y*n.y + n.z*n.z).squareRoot())
                let i = (y*S + x) * 4
                px[i]   = UInt8(max(0, min(255, (n.x * 0.5 + 0.5) * 255)))
                px[i+1] = UInt8(max(0, min(255, (n.y * 0.5 + 0.5) * 255)))
                px[i+2] = UInt8(max(0, min(255, (n.z * 0.5 + 0.5) * 255)))
                px[i+3] = 255
            }
        }
        return image(from: px, size: S)
    }

    private static func roughnessMap(from h: [Float], size S: Int) -> NSImage {
        // Struck/raised areas are slightly more polished (rubbed); field is duller.
        var px = [UInt8](repeating: 0, count: S * S * 4)
        for i in 0..<(S * S) {
            let relief = h[i]                          // 0 field … ~1 raised
            let rough = 0.46 - relief * 0.16           // 0.46 field → ~0.30 raised
            let v = UInt8(max(0, min(255, rough * 255)))
            px[i*4] = v; px[i*4+1] = v; px[i*4+2] = v; px[i*4+3] = 255
        }
        return image(from: px, size: S)
    }

    private static func image(from px: [UInt8], size S: Int) -> NSImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        var data = px
        let img: CGImage? = data.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: S, height: S,
                                      bitsPerComponent: 8, bytesPerRow: S * 4, space: cs,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            return ctx.makeImage()
        }
        guard let cg = img else { return NSImage(size: NSSize(width: S, height: S)) }
        return NSImage(cgImage: cg, size: NSSize(width: S, height: S))
    }
}
