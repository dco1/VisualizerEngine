import CoreGraphics
import Foundation

// ── PLATFORM TYPES (macOS + Mac Catalyst) ────────────────────────────────────
//
// The engine's image/colour plumbing is Core Graphics at heart, but SceneKit
// material slots and the texture-atlas ingest speak the platform's native
// image/colour classes: NSImage/NSColor under AppKit, UIImage/UIColor under
// UIKit (Mac Catalyst / iOS). These aliases + tiny shims let every call site
// compile unchanged on macOS (the aliases resolve to the exact AppKit types,
// so the public API is source-identical) while giving Catalyst the UIKit
// equivalents.
//
// The UIKit-side extensions deliberately mirror the AppKit initialiser
// spellings used in this repo (`deviceRed:`, `calibratedRed:`,
// `calibratedWhite:`, `srgbRed:`, `init(cgImage:size:)`) so ported files
// don't need per-call-site #if blocks. UIColor's sRGB-backed
// `init(red:green:blue:alpha:)` is the right stand-in for all of them —
// device/calibrated distinctions don't survive the trip into SceneKit
// anyway.

#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit

public typealias PlatformImage = NSImage
public typealias PlatformColor = NSColor
public typealias PlatformFont  = NSFont

public extension NSImage {
    /// CGImage extraction, platform-neutral spelling. (UIImage side is just
    /// its `cgImage` property.)
    var platformCGImage: CGImage? {
        var rect = CGRect(origin: .zero, size: size)
        return cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}

public extension NSColor {
    /// sRGB (r, g, b, a) or nil when the colour can't be converted
    /// (pattern/catalog colours). Callers pick their own fallback.
    var srgbComponents: SIMD4<Double>? {
        guard let c = usingColorSpace(.sRGB) else { return nil }
        return SIMD4(Double(c.redComponent), Double(c.greenComponent),
                     Double(c.blueComponent), Double(c.alphaComponent))
    }

    /// Failable-everywhere CGColor bridge (NSColor's init is failable,
    /// UIColor's isn't — this evens the spelling out).
    static func fromCG(_ cgColor: CGColor) -> PlatformColor? {
        NSColor(cgColor: cgColor)
    }
}

#else
import UIKit

public typealias PlatformImage = UIImage
public typealias PlatformColor = UIColor
public typealias PlatformFont  = UIFont

public extension UIImage {
    /// AppKit-spelling convenience: the `size` is implied by the CGImage
    /// (scale 1), matching how the engine builds its procedural textures.
    convenience init(cgImage: CGImage, size: CGSize) {
        self.init(cgImage: cgImage)
    }

    var platformCGImage: CGImage? { cgImage }
}

public extension UIColor {
    convenience init(deviceRed red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat) {
        self.init(red: red, green: green, blue: blue, alpha: alpha)
    }

    convenience init(calibratedRed red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat) {
        self.init(red: red, green: green, blue: blue, alpha: alpha)
    }

    convenience init(calibratedWhite white: CGFloat, alpha: CGFloat) {
        self.init(white: white, alpha: alpha)
    }

    convenience init(srgbRed red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat) {
        self.init(red: red, green: green, blue: blue, alpha: alpha)
    }

    /// sRGB (r, g, b, a) or nil when the colour can't be read as RGB or
    /// grayscale. Callers pick their own fallback.
    var srgbComponents: SIMD4<Double>? {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        if getRed(&r, green: &g, blue: &b, alpha: &a) {
            return SIMD4(Double(r), Double(g), Double(b), Double(a))
        }
        var w: CGFloat = 0
        if getWhite(&w, alpha: &a) {
            return SIMD4(Double(w), Double(w), Double(w), Double(a))
        }
        return nil
    }

    static func fromCG(_ cgColor: CGColor) -> PlatformColor? {
        UIColor(cgColor: cgColor)
    }
}
#endif
