import Foundation
import Metal
import CoreGraphics
import ImageIO

// Headless output for `IlluminatoramaRenderer`: render without a window or MTKView, read the
// frame back, write a PNG. This is the one implementation — hosts (Daydream Home's GPU tests,
// Visualizer scenes, scripts) call it instead of each carrying a private blit + PNG copy.

/// A frame read back from `outputTexture`.
public struct IlluminatoramaFrame: Sendable {
    public let width: Int
    public let height: Int
    public let pixelFormat: MTLPixelFormat
    /// Tightly packed 8-bit, 4 bytes per pixel, in `pixelFormat`'s channel order.
    public let bytes: [UInt8]

    /// True when `bytes` are B,G,R,A (so luma weights land on the right channels).
    public var isBGRA: Bool { pixelFormat == .bgra8Unorm || pixelFormat == .bgra8Unorm_srgb }

    public var cgImage: CGImage? {
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
            .union(isBGRA ? .byteOrder32Little : .byteOrder32Big)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: space, bitmapInfo: info, provider: provider,
                       decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Write as PNG; returns false if the image or destination could not be made.
    @discardableResult
    public func writePNG(to path: String) -> Bool {
        guard let cg = cgImage,
              let dest = CGImageDestinationCreateWithURL(
                URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest)
    }
}

public extension IlluminatoramaRenderer {

    /// Blit `texture` (possibly private storage) into a shared copy and return its bytes.
    /// nil for formats other than 8-bit RGBA/BGRA.
    static func readFrame(_ texture: MTLTexture, queue: MTLCommandQueue) -> IlluminatoramaFrame? {
        switch texture.pixelFormat {
        case .bgra8Unorm, .bgra8Unorm_srgb, .rgba8Unorm, .rgba8Unorm_srgb: break
        default: return nil
        }
        let w = texture.width, h = texture.height
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: texture.pixelFormat, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        desc.usage = [.shaderRead]
        guard let dst = queue.device.makeTexture(descriptor: desc),
              let cmd = queue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: dst, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()   // gpu-ok: headless readback, never the per-frame lane
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        bytes.withUnsafeMutableBytes {
            dst.getBytes($0.baseAddress!, bytesPerRow: w * 4,
                         from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                         size: MTLSize(width: w, height: h, depth: 1)),
                         mipmapLevel: 0)
        }
        return IlluminatoramaFrame(width: w, height: h, pixelFormat: texture.pixelFormat, bytes: bytes)
    }

    /// Read back the renderer's current `outputTexture`.
    func readFrame() -> IlluminatoramaFrame? {
        Self.readFrame(outputTexture, queue: commandQueue)
    }

    /// Render `frames` blocking frames (the last one settles temporal accumulation) with no
    /// window attached, then read the result back. `frames` ≥ 1.
    func renderHeadless(frames: Int = 1) -> IlluminatoramaFrame? {
        for _ in 0..<max(1, frames) { _ = render(blocking: true) }
        return readFrame()
    }

    /// `renderHeadless`, then write the PNG. Returns the frame so a caller can also measure it.
    @discardableResult
    func renderHeadlessPNG(to path: String, frames: Int = 1) -> IlluminatoramaFrame? {
        guard let frame = renderHeadless(frames: frames) else { return nil }
        return frame.writePNG(to: path) ? frame : nil
    }
}
