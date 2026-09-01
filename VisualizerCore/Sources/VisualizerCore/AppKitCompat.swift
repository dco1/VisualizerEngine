import CoreGraphics
import Foundation
import SwiftUI

// ── APPKIT COMPAT (UIKit-only platforms: tvOS / iOS) ─────────────────────────
//
// The Visualizer scene packages were authored against AppKit — `NSColor`,
// `NSImage`, `NSFont`, `NSBezierPath`, `NSGradient`, `NSRect`/`NSSize`/
// `NSPoint`, and the `NSGraphicsContext.current?.cgContext` drawing idiom. The
// Apple TV target compiles those same packages for tvOS, where AppKit doesn't
// exist. Rather than fork ~140 scene files onto `PlatformColor`/`PlatformImage`
// spellings (and keep two spellings alive forever), this file gives UIKit
// platforms the AppKit *names* as typealiases onto the UIKit equivalents plus
// the handful of AppKit-only members the scenes actually call. The macOS build
// is untouched: nothing here exists when real AppKit is importable, so the
// scene code is still compiled against genuine AppKit there.
//
// Scope is deliberately "what the scenes use" (measured by grep across
// Visualizer/Scenes), not "all of AppKit". When a scene needs another AppKit
// member on tvOS, add it here once — don't `#if os(tvOS)` around the call site.
//
// Things this file deliberately does NOT fake:
//   • `NSImage.lockFocus()` — UIImage is immutable; the scene-side pattern is
//     `ProceduralCanvas.image(size:) { ctx in … }` (PlatformDrawing.swift),
//     which is the same CGContext on both platforms.
//   • `NSEvent` / `NSGestureRecognizer` — input has no tvOS analogue; the few
//     scenes that read the mouse wrap that code in `#if !os(tvOS)`.
//
// Mac Catalyst is excluded on purpose: it can import real AppKit, and the
// engine's PlatformTypes.swift already routes Catalyst through UIKit with its
// own (PlatformColor-spelled) shims — adding AppKit-named aliases there would
// collide with the real declarations.

#if canImport(UIKit) && !canImport(AppKit)
import UIKit

// MARK: - Type names

public typealias NSColor = UIColor
public typealias NSImage = UIImage
public typealias NSFont = UIFont
public typealias NSBezierPath = UIBezierPath
public typealias NSSize = CGSize
public typealias NSPoint = CGPoint
public typealias NSRect = CGRect
public typealias NSEdgeInsets = UIEdgeInsets

// MARK: - NSColor

/// AppKit's `NSColorSpace` is only ever used by the scenes as the argument to
/// `usingColorSpace(_:)` (`.sRGB`, `.deviceRGB`, `.genericRGB`, …). UIColor is
/// always component-readable, so the conversion is the identity.
public struct NSColorSpace: Sendable, Hashable {
    public static let sRGB = NSColorSpace()
    public static let deviceRGB = NSColorSpace()
    public static let genericRGB = NSColorSpace()
    public static let extendedSRGB = NSColorSpace()
    public static let displayP3 = NSColorSpace()
    public static let genericGray = NSColorSpace()
    public static let deviceGray = NSColorSpace()
}

public extension UIColor {
    /// AppKit's HSB calibrated initialiser spelling.
    convenience init(calibratedHue hue: CGFloat, saturation: CGFloat, brightness: CGFloat, alpha: CGFloat) {
        self.init(hue: hue, saturation: saturation, brightness: brightness, alpha: alpha)
    }

    convenience init(deviceWhite white: CGFloat, alpha: CGFloat) {
        self.init(white: white, alpha: alpha)
    }

    convenience init(genericGamma22White white: CGFloat, alpha: CGFloat) {
        self.init(white: white, alpha: alpha)
    }

    convenience init(deviceHue hue: CGFloat, saturation: CGFloat, brightness: CGFloat, alpha: CGFloat) {
        self.init(hue: hue, saturation: saturation, brightness: brightness, alpha: alpha)
    }

    /// Identity on UIKit — every UIColor the scenes build is RGB(A)-readable.
    /// Kept failable to match AppKit's signature so `guard let` sites compile.
    func usingColorSpace(_ space: NSColorSpace) -> UIColor? { self }

    /// Named like AppKit's `usingColorSpaceName`-era API for the odd caller.
    func usingColorSpaceName(_ name: String) -> UIColor? { self }

    private var rgba: (r: CGFloat, g: CGFloat, b: CGFloat, a: CGFloat) {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        if getRed(&r, green: &g, blue: &b, alpha: &a) { return (r, g, b, a) }
        var w: CGFloat = 0
        if getWhite(&w, alpha: &a) { return (w, w, w, a) }
        return (0, 0, 0, 1)
    }

    private var hsba: (h: CGFloat, s: CGFloat, b: CGFloat, a: CGFloat) {
        var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        if getHue(&h, saturation: &s, brightness: &b, alpha: &a) { return (h, s, b, a) }
        let c = rgba
        return (0, 0, max(c.r, c.g, c.b), c.a)
    }

    var redComponent: CGFloat { rgba.r }
    var greenComponent: CGFloat { rgba.g }
    var blueComponent: CGFloat { rgba.b }
    var alphaComponent: CGFloat { rgba.a }
    var hueComponent: CGFloat { hsba.h }
    var saturationComponent: CGFloat { hsba.s }
    var brightnessComponent: CGFloat { hsba.b }
    /// Grey level (AppKit only defines this for grey colours; here it's the
    /// luma-free average so the call compiles and behaves sanely for RGB).
    var whiteComponent: CGFloat { let c = rgba; return (c.r + c.g + c.b) / 3 }

    /// AppKit's `blended(withFraction:of:)` — linear RGBA interpolation.
    /// `fraction` 0 = self, 1 = `other`.
    func blended(withFraction fraction: CGFloat, of other: UIColor) -> UIColor? {
        let a = rgba, b = other.rgba
        let t = min(max(fraction, 0), 1)
        return UIColor(red: a.r + (b.r - a.r) * t,
                       green: a.g + (b.g - a.g) * t,
                       blue: a.b + (b.b - a.b) * t,
                       alpha: a.a + (b.a - a.a) * t)
    }

    /// AppKit's `shadow(withLevel:)` — blend toward black.
    func shadow(withLevel level: CGFloat) -> UIColor? { blended(withFraction: level, of: .black) }
    /// AppKit's `highlight(withLevel:)` — blend toward white.
    func highlight(withLevel level: CGFloat) -> UIColor? { blended(withFraction: level, of: .white) }

    // AppKit semantic colours the scenes reference. UIKit's dynamic equivalents.
    static var labelColor: UIColor { .label }
    static var secondaryLabelColor: UIColor { .secondaryLabel }
    static var textColor: UIColor { .label }
    // tvOS has no `systemBackground`; a fixed dark/light pair stands in.
    static var windowBackgroundColor: UIColor {
        UIColor { $0.userInterfaceStyle == .dark ? UIColor(white: 0.11, alpha: 1) : .white }
    }
    static var controlBackgroundColor: UIColor { windowBackgroundColor }
    static var controlAccentColor: UIColor { .systemBlue }
    static var textBackgroundColor: UIColor { windowBackgroundColor }
}

// MARK: - SwiftUI Color ↔ NSColor

public extension Color {
    /// AppKit-spelled bridge (`Color(nsColor:)`); UIKit's is `Color(uiColor:)`.
    init(nsColor: UIColor) { self.init(uiColor: nsColor) }
}

// MARK: - NSImage

public extension UIImage {
    /// AppKit's CGImage accessor spelling. The proposed rect / context / hints
    /// are ignored — UIImage is a single bitmap.
    func cgImage(forProposedRect rect: UnsafeMutablePointer<CGRect>?,
                 context: Any?, hints: [AnyHashable: Any]?) -> CGImage? {
        cgImage
    }

    /// Draw the image into the *current* graphics context honouring AppKit's
    /// unflipped (bottom-left-origin) convention. `UIImage.draw(in:)` assumes
    /// UIKit's flipped context and would paint upside-down inside a
    /// `ProceduralCanvas` (whose context is bottom-left, like `lockFocus`);
    /// this flips around the target rect so the result matches macOS.
    func drawUnflipped(in rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext(), let cg = cgImage else { return }
        ctx.saveGState()
        // CGContext.draw(_:in:) is orientation-neutral (it draws the bitmap
        // with its first row at the TOP of `rect` in a bottom-left context),
        // which is exactly what AppKit's NSImage.draw(in:) produces.
        ctx.draw(cg, in: rect)
        ctx.restoreGState()
    }
}

// MARK: - NSGraphicsContext

/// The `NSGraphicsContext.current?.cgContext` idiom, resolved to UIKit's
/// current context. `NSGraphicsContext.current` is non-nil inside a
/// `ProceduralCanvas.image(size:)` closure (the canvas pushes its CGContext).
public struct NSGraphicsContext {
    public let cgContext: CGContext

    public static var current: NSGraphicsContext? {
        UIGraphicsGetCurrentContext().map(NSGraphicsContext.init(cgContext:))
    }

    public init(cgContext: CGContext) { self.cgContext = cgContext }
    public init(cgContext: CGContext, flipped: Bool) { self.cgContext = cgContext }

    public static func saveGraphicsState() { UIGraphicsGetCurrentContext()?.saveGState() }
    public static func restoreGraphicsState() { UIGraphicsGetCurrentContext()?.restoreGState() }

    /// AppKit's `shouldAntialias` on the current context.
    public var shouldAntialias: Bool {
        get { true }
        nonmutating set { cgContext.setShouldAntialias(newValue) }
    }
    public var imageInterpolation: CGInterpolationQuality {
        get { cgContext.interpolationQuality }
        nonmutating set { cgContext.interpolationQuality = newValue }
    }
}

// MARK: - NSRect / NSBezierPath drawing

public extension CGRect {
    /// AppKit's `NSRect.fill()` — fill with the current fill colour.
    func fill() { UIRectFill(self) }
    /// AppKit's `NSRect.frame()` — stroke a 1-pt frame with the current colour.
    func frame() { UIRectFrame(self) }
    func fill(using op: CGBlendMode) { UIRectFillUsingBlendMode(self, op) }
}

public extension UIBezierPath {
    /// AppKit's `line(to:)` spelling (UIKit: `addLine(to:)`).
    func line(to point: CGPoint) { addLine(to: point) }

    /// AppKit's cubic spelling (UIKit: `addCurve(to:controlPoint1:controlPoint2:)`).
    func curve(to endPoint: CGPoint, controlPoint1: CGPoint, controlPoint2: CGPoint) {
        addCurve(to: endPoint, controlPoint1: controlPoint1, controlPoint2: controlPoint2)
    }

    func appendRect(_ rect: CGRect) { append(UIBezierPath(rect: rect)) }
    func appendOval(in rect: CGRect) { append(UIBezierPath(ovalIn: rect)) }
    func appendRoundedRect(_ rect: CGRect, xRadius: CGFloat, yRadius: CGFloat) {
        append(UIBezierPath(roundedRect: rect, cornerRadius: min(xRadius, yRadius)))
    }
    func appendBezierPath(_ path: UIBezierPath) { append(path) }

    /// AppKit's arc — angles in DEGREES, counter-clockwise (θ increasing) by
    /// default. Goes through the underlying CGPath rather than UIKit's
    /// `addArc(withCenter:…)` so the direction flag has Quartz's y-up meaning
    /// — verified identical to `NSBezierPath.appendArc` (CG `clockwise:false`
    /// ≡ AppKit default). Like AppKit/CG, a line joins the current point to
    /// the arc start when the path already has one.
    func appendArc(withCenter center: CGPoint, radius: CGFloat,
                   startAngle: CGFloat, endAngle: CGFloat, clockwise: Bool = false) {
        let m = cgPath.mutableCopy() ?? CGMutablePath()
        m.addArc(center: center, radius: radius,
                 startAngle: startAngle * .pi / 180, endAngle: endAngle * .pi / 180,
                 clockwise: clockwise)
        cgPath = m
    }

    /// AppKit's `appendArc(from:to:radius:)` (tangent arc; UIKit has no
    /// direct API — route through the CGPath).
    func appendArc(from fromPoint: CGPoint, to toPoint: CGPoint, radius: CGFloat) {
        let m = cgPath.mutableCopy() ?? CGMutablePath()
        m.addArc(tangent1End: fromPoint, tangent2End: toPoint, radius: radius)
        cgPath = m
    }

    /// AppKit's `NSBezierPath(roundedRect:xRadius:yRadius:)`.
    convenience init(roundedRect rect: CGRect, xRadius: CGFloat, yRadius: CGFloat) {
        self.init(roundedRect: rect, cornerRadius: min(xRadius, yRadius))
    }

    /// AppKit's `windingRule` — UIKit exposes the same switch as a Bool.
    enum WindingRule { case nonZero, evenOdd }
    var windingRule: WindingRule {
        get { usesEvenOddFillRule ? .evenOdd : .nonZero }
        set { usesEvenOddFillRule = (newValue == .evenOdd) }
    }

    /// AppKit's `transform(using:)` (there it takes Foundation's
    /// `AffineTransform`, which doesn't exist on UIKit platforms — callers pass
    /// a `CGAffineTransform`; UIKit applies it via `apply(_:)`).
    func transform(using transform: CGAffineTransform) { apply(transform) }
}

// MARK: - NSGradient

/// CGGradient-backed stand-in for AppKit's `NSGradient`, covering the draw
/// calls the scenes use. Draws into the *current* graphics context (i.e. inside
/// a `ProceduralCanvas` closure), clipping to the target rect/path like AppKit.
public final class NSGradient {
    public struct DrawingOptions: OptionSet, Sendable {
        public let rawValue: UInt
        public init(rawValue: UInt) { self.rawValue = rawValue }
        public static let drawsBeforeStartingLocation = DrawingOptions(rawValue: 1 << 0)
        public static let drawsAfterEndingLocation = DrawingOptions(rawValue: 1 << 1)
    }

    private let cg: CGGradient

    public convenience init?(starting: UIColor, ending: UIColor) {
        self.init(colors: [starting, ending])
    }

    public convenience init?(colors: [UIColor]) {
        let n = colors.count
        guard n >= 2 else { return nil }
        let locs = (0..<n).map { CGFloat($0) / CGFloat(n - 1) }
        self.init(colors: colors, atLocations: locs, colorSpace: nil)
    }

    public init?(colors: [UIColor], atLocations locations: [CGFloat], colorSpace: NSColorSpace?) {
        guard colors.count >= 2, colors.count == locations.count else { return nil }
        let space = CGColorSpaceCreateDeviceRGB()
        let cgColors = colors.map { $0.cgColor.converted(to: space, intent: .defaultIntent, options: nil) ?? $0.cgColor }
        guard let g = CGGradient(colorsSpace: space, colors: cgColors as CFArray, locations: locations) else { return nil }
        cg = g
    }

    public convenience init?(colorsAndLocations: (UIColor, CGFloat)...) {
        self.init(colors: colorsAndLocations.map(\.0),
                  atLocations: colorsAndLocations.map(\.1), colorSpace: nil)
    }

    /// Linear gradient across `rect` at `angle` degrees (AppKit convention:
    /// 0° = left→right, 90° = bottom→top in the unflipped context).
    public func draw(in rect: CGRect, angle: CGFloat) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.saveGState()
        ctx.clip(to: rect)
        let (start, end) = Self.axis(in: rect, angleDegrees: angle)
        ctx.drawLinearGradient(cg, start: start, end: end,
                               options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        ctx.restoreGState()
    }

    /// Linear gradient clipped to `path` at `angle` degrees.
    public func draw(in path: UIBezierPath, angle: CGFloat) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.saveGState()
        path.addClip()
        let (start, end) = Self.axis(in: path.bounds, angleDegrees: angle)
        ctx.drawLinearGradient(cg, start: start, end: end,
                               options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        ctx.restoreGState()
    }

    /// Linear gradient between two points (no clip — AppKit semantics).
    public func draw(from startingPoint: CGPoint, to endingPoint: CGPoint, options: DrawingOptions) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        var o: CGGradientDrawingOptions = []
        if options.contains(.drawsBeforeStartingLocation) { o.insert(.drawsBeforeStartLocation) }
        if options.contains(.drawsAfterEndingLocation) { o.insert(.drawsAfterEndLocation) }
        ctx.drawLinearGradient(cg, start: startingPoint, end: endingPoint, options: o)
    }

    /// Radial gradient (AppKit `draw(fromCenter:radius:toCenter:radius:options:)`).
    public func draw(fromCenter startCenter: CGPoint, radius startRadius: CGFloat,
                     toCenter endCenter: CGPoint, radius endRadius: CGFloat,
                     options: DrawingOptions) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        var o: CGGradientDrawingOptions = []
        if options.contains(.drawsBeforeStartingLocation) { o.insert(.drawsBeforeStartLocation) }
        if options.contains(.drawsAfterEndingLocation) { o.insert(.drawsAfterEndLocation) }
        ctx.drawRadialGradient(cg, startCenter: startCenter, startRadius: startRadius,
                               endCenter: endCenter, endRadius: endRadius, options: o)
    }

    /// Radial gradient filling `rect` (AppKit `draw(in:relativeCenterPosition:)`).
    public func draw(in rect: CGRect, relativeCenterPosition: CGPoint) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.saveGState()
        ctx.clip(to: rect)
        let c = CGPoint(x: rect.midX + relativeCenterPosition.x * rect.width / 2,
                        y: rect.midY + relativeCenterPosition.y * rect.height / 2)
        let r = hypot(rect.width, rect.height) / 2
        ctx.drawRadialGradient(cg, startCenter: c, startRadius: 0, endCenter: c, endRadius: r,
                               options: [.drawsAfterEndLocation])
        ctx.restoreGState()
    }

    /// AppKit's rule: the gradient axis passes through the rect's centre, at
    /// `angle`, long enough that the rect's corners project onto [0, 1].
    private static func axis(in rect: CGRect, angleDegrees: CGFloat) -> (CGPoint, CGPoint) {
        let theta = angleDegrees * .pi / 180
        let dir = CGPoint(x: cos(theta), y: sin(theta))
        let half = (abs(dir.x) * rect.width + abs(dir.y) * rect.height) / 2
        let c = CGPoint(x: rect.midX, y: rect.midY)
        return (CGPoint(x: c.x - dir.x * half, y: c.y - dir.y * half),
                CGPoint(x: c.x + dir.x * half, y: c.y + dir.y * half))
    }
}

// MARK: - Text drawing in an unflipped context

public extension NSAttributedString {
    /// Draw honouring AppKit's unflipped (bottom-left) convention. UIKit's
    /// `draw(at:)` assumes a flipped context and paints text upside-down inside
    /// a `ProceduralCanvas`; this flips around the text's own box so the glyphs
    /// land exactly where macOS puts them (baseline-up from `point`).
    func drawUnflipped(at point: CGPoint) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        let s = size()
        ctx.saveGState()
        ctx.translateBy(x: point.x, y: point.y + s.height)
        ctx.scaleBy(x: 1, y: -1)
        draw(at: .zero)
        ctx.restoreGState()
    }

    /// Unflipped-context variant of `draw(in:)`.
    func drawUnflipped(in rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.saveGState()
        ctx.translateBy(x: rect.minX, y: rect.maxY)
        ctx.scaleBy(x: 1, y: -1)
        draw(in: CGRect(origin: .zero, size: rect.size))
        ctx.restoreGState()
    }
}

public extension NSString {
    func drawUnflipped(at point: CGPoint, withAttributes attrs: [NSAttributedString.Key: Any]?) {
        NSAttributedString(string: self as String, attributes: attrs).drawUnflipped(at: point)
    }
    func drawUnflipped(in rect: CGRect, withAttributes attrs: [NSAttributedString.Key: Any]?) {
        NSAttributedString(string: self as String, attributes: attrs).drawUnflipped(in: rect)
    }
}

#else
#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit

// On macOS the "unflipped" spellings are plain AppKit draws — AppKit string and
// image drawing already respect the current context's flippedness, so the
// scene code can use one spelling on both platforms.
public extension NSAttributedString {
    func drawUnflipped(at point: CGPoint) { draw(at: point) }
    func drawUnflipped(in rect: CGRect) { draw(in: rect) }
}
public extension NSString {
    func drawUnflipped(at point: CGPoint, withAttributes attrs: [NSAttributedString.Key: Any]?) {
        draw(at: point, withAttributes: attrs)
    }
    func drawUnflipped(in rect: CGRect, withAttributes attrs: [NSAttributedString.Key: Any]?) {
        draw(in: rect, withAttributes: attrs)
    }
}
public extension NSImage {
    func drawUnflipped(in rect: CGRect) { draw(in: rect) }
}
#endif
#endif
