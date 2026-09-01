import CoreGraphics
import Foundation

#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#else
import UIKit
#endif

// ── PROCEDURAL IMAGE ─────────────────────────────────────────────────────────
//
// The scenes rasterise most of their procedural textures (labels, stripes,
// gradients, noise fills) with AppKit's mutable-image idiom:
//
//     let img = NSImage(size: s); img.lockFocus(); defer { img.unlockFocus() }
//     guard let ctx = NSGraphicsContext.current?.cgContext else { return img }
//     … CoreGraphics / NSColor.setFill / NSBezierPath.fill / text.draw(at:) …
//     return img
//
// `lockFocus` has no UIKit equivalent (UIImage is immutable), so the Apple TV
// port swaps the one token that matters — `NSImage(size:)` → `ProceduralImage(
// size:)` — and leaves the ~70 drawing bodies untouched.
//
//   • macOS: `ProceduralImage` is an `NSImage` subclass that overrides
//     nothing. `lockFocus` / `unlockFocus` / `NSGraphicsContext.current` are
//     the inherited AppKit implementations, so the macOS pixels are produced by
//     exactly the code path they always were (same rasteriser, same backing
//     scale, same colour handling). Zero behaviour change by construction.
//
//   • UIKit: `ProceduralImage` is a `UIImage` subclass that owns a
//     CGBitmapContext while focus is locked and publishes the result through
//     the overridden `cgImage` / `size` / `scale` — the accessors SceneKit and
//     `UIImage.draw` read. The context is set up to match `lockFocus`:
//     bottom-left origin, points CTM at a 2× backing scale (what a Retina Mac's
//     lockFocus allocates), sRGB, premultiplied alpha, and pushed as UIKit's
//     current context so `UIColor.setFill()` / `UIRectFill` / `UIBezierPath
//     .fill()` inside the body work like their AppKit spellings did.
//
// The one thing that differs between the kits inside such a body is text and
// image drawing: UIKit's `NSAttributedString.draw(at:)` / `UIImage.draw(in:)`
// assume a flipped (top-left) context and would paint upside-down here. Those
// call sites use the `drawUnflipped(at:/in:)` spellings from
// AppKitCompat.swift — plain AppKit draws on macOS, orientation-corrected on
// UIKit.

#if canImport(AppKit) && !targetEnvironment(macCatalyst)

/// macOS: a plain `NSImage`. Exists only so the scene code has one spelling
/// on both platforms — see the header comment.
public final class ProceduralImage: NSImage, @unchecked Sendable {}

#else

/// UIKit: a canvas-backed `UIImage` that supports AppKit's
/// `lockFocus()` / `unlockFocus()` drawing idiom. See the header comment.
public final class ProceduralImage: UIImage, @unchecked Sendable {
    /// Pixels per point the canvas rasterises at. Matches `NSImage.lockFocus`
    /// on a Retina display, which is what every scene texture was authored against.
    public static let defaultScale: CGFloat = 2

    // Stored state has defaults and the only initialiser is a convenience one,
    // so the subclass inherits every UIImage designated/required init (UIImage
    // declares `init(imageLiteralResourceName:)` as required in an extension,
    // which a subclass with its own designated init can't satisfy).
    private var pointSize: CGSize = .zero
    private var canvasScale: CGFloat = ProceduralImage.defaultScale
    private var backing: CGImage?
    private var context: CGContext?

    public convenience init(size: CGSize, scale: CGFloat = ProceduralImage.defaultScale) {
        self.init()
        pointSize = size
        canvasScale = scale
    }

    // The accessors the consumers read. `size` is in points (== the requested
    // size, like NSImage.size); `scale` makes UIKit agree on the pixel count.
    public override var cgImage: CGImage? { backing }
    public override var size: CGSize { pointSize }
    public override var scale: CGFloat { canvasScale }
    public override var imageOrientation: UIImage.Orientation { .up }

    /// Begin drawing. Creates the bitmap context and makes it UIKit's current
    /// context (per-thread, like AppKit's focus stack). Balanced by
    /// `unlockFocus()`; nested or unbalanced calls are a programmer error.
    public func lockFocus() {
        precondition(context == nil, "ProceduralImage.lockFocus(): already focused")
        let pw = max(1, Int((pointSize.width * canvasScale).rounded(.up)))
        let ph = max(1, Int((pointSize.height * canvasScale).rounded(.up)))
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: pw, height: ph,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return
        }
        // Points CTM, bottom-left origin (CGBitmapContext's native orientation
        // — the same as an NSImage lockFocus context). No flip.
        if pointSize.width > 0, pointSize.height > 0 {
            ctx.scaleBy(x: CGFloat(pw) / pointSize.width, y: CGFloat(ph) / pointSize.height)
        }
        ctx.interpolationQuality = .high
        context = ctx
        UIGraphicsPushContext(ctx)
    }

    /// End drawing: pop the context and publish the bitmap through `cgImage`.
    public func unlockFocus() {
        guard let ctx = context else { return }
        UIGraphicsPopContext()
        backing = ctx.makeImage()
        context = nil
    }
}

#endif

/// Closure form of the same canvas, for new code that doesn't need the
/// lock/unlock idiom: `ProceduralCanvas.image(size: s) { ctx in … }`.
public enum ProceduralCanvas {
    public static func image(size: CGSize, _ draw: (CGContext) -> Void) -> PlatformImage {
        let img = ProceduralImage(size: size)
        img.lockFocus()
        defer { img.unlockFocus() }
        #if canImport(AppKit) && !targetEnvironment(macCatalyst)
        guard let ctx = NSGraphicsContext.current?.cgContext else { return img }
        #else
        guard let ctx = UIGraphicsGetCurrentContext() else { return img }
        #endif
        draw(ctx)
        return img
    }
}
