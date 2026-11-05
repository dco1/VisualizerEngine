#if canImport(AppKit) && !targetEnvironment(macCatalyst)
import AppKit
#else
import UIKit
#endif
import CoreGraphics
import Foundation
import Metal
import VisualizerCore

// ── Drawing a slice off the main thread ───────────────────────────────────────
//
// A slice's pixels are one Core Graphics draw per mip level from the original image plus a mean over
// level 0 (`prepare`): on a phone, tens of milliseconds a 512² map, all of it on the main thread when
// it ran inside `register(contents:)`. `prepare` touches no atlas state, so `registerAsync` runs it on
// a background task and comes back to the main actor only to take a slot and copy the finished levels
// in (`place` → `commit`). Same bytes either way: `register(contents:)` is `prepare` + `place` inline.

/// A CGImage is immutable; this carries one across to the background draw.
private struct SendableImage: @unchecked Sendable { let image: CGImage }

extension IlluminatoramaTextureAtlas {
    /// One slice's finished pixels, ready to copy in.
    struct PreparedSlice: Sendable {
        /// Level 0 first; level `l` is `n × n` BGRA with `n = max(1, sliceSize >> l)`.
        var levels: [[UInt8]]
        /// The fraction of the square the aspect-preserved image fills (`uvScaleBuffer`).
        var uvScale: SIMD2<Float>
        /// Level 0's mean over the image, in the space the shader samples (`sliceMeanBuffer`).
        var mean: SIMD4<Float>
    }

    /// What `register` was handed: a slot already decided (or refused), an image to draw, or nothing usable.
    enum Resolved {
        case none
        case cached(Int32?)
        case image(ObjectIdentifier, AnyObject, CGImage)
    }

    /// Like `register(contents:)`, but the drawing runs off the main thread: the main actor only takes a
    /// slot and copies the finished levels in. The slice it returns is the same as `register`'s would
    /// be, byte for byte. Two calls for the same image while one is drawing both draw; the second to
    /// finish takes the first's slot.
    public func registerAsync(contents: Any?) async -> Int32? {
        switch resolve(contents) {
        case .none: return nil
        case .cached(let slice): return slice
        case .image(let key, let keyObject, let cg):
            let source = SendableImage(image: cg)
            let size = sliceSize, levels = mipLevels, format = pixelFormat
            let prepared = await Task.detached(priority: .userInitiated) {
                Self.prepare(source.image, sliceSize: size, mipLevels: levels, pixelFormat: format)
            }.value
            if let done = sliceForObject[key] { return done.slice }     // registered while this one drew
            guard let prepared, prepared.levels.count == mipLevels, levels == mipLevels else {
                sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: nil)
                return nil
            }
            return place(prepared, key: key, keyObject: keyObject)
        }
    }

    /// Pull a CGImage out of type-erased material contents, or the cached answer for them.
    func resolve(_ contents: Any?) -> Resolved {
        guard let contents else { return .none }
        // The common cases that arrive from SCNMaterial: NSImage / UIImage (asset or file loads) and
        // CGImage (programmatic, Core Graphics output). Paths, MTLTextures and colours aren't this path.
        if let img = contents as? PlatformImage {
            let key = ObjectIdentifier(img)
            if let cached = sliceForObject[key] { return .cached(cached.slice) }
            guard let cg = Self.cgImage(from: img) else {
                sliceForObject[key] = SliceCacheEntry(retained: img, slice: nil)
                return .cached(nil)
            }
            return .image(key, img, cg)
        }
        if CFGetTypeID(contents as CFTypeRef) == CGImage.typeID {
            // `contents as? CGImage` is rejected by Swift 6 (CF conditional downcasts always succeed at the
            // type-system level); the type ID confirms it, then force-cast once.
            let img = contents as! CGImage
            let key = ObjectIdentifier(img as AnyObject)
            if let cached = sliceForObject[key] { return .cached(cached.slice) }
            return .image(key, img as AnyObject, img)
        }
        return .none
    }

    /// Take a slot for `prepared` (a freed one first, else the next, growing the atlas when it's full),
    /// copy it in and remember it under `key`. Nil when the atlas can't grow.
    func place(_ prepared: PreparedSlice, key: ObjectIdentifier, keyObject: AnyObject) -> Int32? {
        let slice: Int32
        if let reused = freeSlices.popLast() {
            slice = reused
        } else {
            // Phase 4.4 — out of slots: double the capacity by blitting the existing slices into a larger
            // array (the renderer rebinds `texture` each draw). Hard-capped to keep VRAM sane.
            if nextSlice >= capacity {
                let maxCap = 1024
                guard capacity < maxCap, grow(to: min(capacity * 2, maxCap)) else {
                    Self.log.warning("Atlas exhausted at capacity=\(self.capacity, privacy: .public); falling back to average colour")
                    sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: nil)
                    return nil
                }
            }
            slice = Int32(nextSlice)
            nextSlice += 1
        }
        commit(prepared, to: Int(slice))
        sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: slice)
        return slice
    }

    /// Copy finished levels into `slice`, with its mean and UV scale. `.shared` storage: `replace` writes
    /// each level directly, no blit encoder.
    func commit(_ prepared: PreparedSlice, to slice: Int) {
        for (level, bytes) in prepared.levels.enumerated() {
            let n = max(1, sliceSize >> level)
            bytes.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                texture.replace(region: MTLRegionMake3D(0, 0, 0, n, n, 1), mipmapLevel: level, slice: slice,
                                withBytes: base, bytesPerRow: n * 4, bytesPerImage: n * n * 4)
            }
        }
        sliceMeanBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: capacity)[slice] = prepared.mean
        uvScaleBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: capacity)[slice] = prepared.uvScale
    }

    /// Draw `cgImage` into a slice's levels. Touches no atlas state, so it runs on any thread. Nil when a
    /// bitmap context can't be made.
    nonisolated static func prepare(_ cgImage: CGImage, sliceSize: Int, mipLevels: Int,
                                    pixelFormat: MTLPixelFormat) -> PreparedSlice? {
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        // bgra8Unorm_srgb expects BGRA channel order. CG's bitmap info uses
        // .premultipliedFirst + byteOrder32Little to lay out BGRA on a
        // little-endian host.
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
                       | CGBitmapInfo.byteOrder32Little.rawValue
        // Phase 4.25 — aspect-preserving letterbox. Instead of squishing a
        // non-square source into the full square (which distorted Forest bark,
        // Eggs warehouse walls, VintageDiner tile), fit the image to the slice
        // preserving its aspect and record the filled fraction as `uvScale`.
        //
        // Anchor at CG-top (y = n − drawH): a CGBitmapContext stores row 0 at the
        // image top, and Metal texture v=0 is buffer row 0, so drawing at high
        // CG-y lands the image in texture v ∈ [0, uvScale.y] — exactly the
        // sub-rect the shader samples. (origin x = 0 → u ∈ [0, uvScale.x].)
        let aspect = Double(cgImage.width) / Double(max(cgImage.height, 1))  // w / h
        var sx: CGFloat = 1, sy: CGFloat = 1
        if aspect >= 1 { sy = CGFloat(1.0 / aspect) } else { sx = CGFloat(aspect) }

        var levels: [[UInt8]] = []
        levels.reserveCapacity(mipLevels)
        var mean = SIMD4<Float>(0, 0, 0, 0)
        // S1.1 — build the FULL mip chain, one Core Graphics draw per level from the
        // ORIGINAL source rather than a box-reduction of level 0 (better filtering, and
        // no re-quantising an 8-bit image repeatedly).
        for level in 0..<mipLevels {
            let n = max(1, sliceSize >> level)
            let bytesPerRow = n * 4
            var buffer = [UInt8](repeating: 0, count: bytesPerRow * n)
            let drew: Bool = buffer.withUnsafeMutableBytes { raw -> Bool in
                guard let base = raw.baseAddress,
                      let ctx = CGContext(data: base, width: n, height: n,
                                          bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                          space: cs, bitmapInfo: bitmapInfo)
                else { return false }
                let drawW = CGFloat(n) * sx
                let drawH = CGFloat(n) * sy
                ctx.interpolationQuality = .high
                // WRAP-FILL rather than letterbox-and-leave-black. For a SQUARE source
                // (sx == sy == 1, which is every slice Daydream Home registers) this is
                // exactly ONE draw at (0, 0, n, n) — byte-identical to the pre-S1.1
                // level-0 upload. For a letterboxed source it additionally TILES the
                // image across the empty band. Level 0 is unchanged either way because
                // the shader clamps its reads to [halfTexel, sc − halfTexel] and never
                // touches the padding; what it buys is that the COARSE levels average
                // image-against-image instead of image-against-BLACK. Without it a 4:1
                // wall texture would read ~4× too dark in the distance, since the mip
                // pyramid would be averaging in the zeroed band.
                let stepX = max(drawW, 1), stepY = max(drawH, 1)
                let nx = min(64, Int((CGFloat(n) / stepX).rounded(.up)))
                let ny = min(64, Int((CGFloat(n) / stepY).rounded(.up)))
                for j in 0..<ny {
                    let y = CGFloat(n) - drawH - CGFloat(j) * stepY
                    for i in 0..<nx {
                        ctx.draw(cgImage, in: CGRect(x: CGFloat(i) * stepX, y: y,
                                                     width: drawW, height: drawH))
                    }
                }
                return true
            }
            guard drew else { return nil }
            // S2.5 — μ for the variance-preserving hex blend, taken from LEVEL 0 over the
            // VALID (pre-wrap-fill) sub-rect only, in the space the shader samples. See
            // `sliceMeanBuffer` for why neither the coarsest mip nor the full square works.
            if level == 0 {
                let lut = (pixelFormat == .bgra8Unorm_srgb || pixelFormat == .rgba8Unorm_srgb)
                        ? srgbToLinear256 : identity256
                // Buffer row 0 is CG-top, which is texture v = 0 — the same anchor the
                // draw above uses — so the valid image occupies rows 0..<n·sy, cols 0..<n·sx.
                let vw = max(1, min(n, Int((CGFloat(n) * sx).rounded())))
                let vh = max(1, min(n, Int((CGFloat(n) * sy).rounded())))
                var sumB = 0.0, sumG = 0.0, sumR = 0.0
                buffer.withUnsafeBytes { raw in
                    guard let p = raw.bindMemory(to: UInt8.self).baseAddress else { return }
                    for y in 0..<vh {
                        var o = y * bytesPerRow
                        for _ in 0..<vw {
                            sumB += Double(lut[Int(p[o])])          // BGRA byte order
                            sumG += Double(lut[Int(p[o + 1])])
                            sumR += Double(lut[Int(p[o + 2])])
                            o += 4
                        }
                    }
                }
                let inv = 1.0 / Double(vw * vh)
                mean = SIMD4<Float>(Float(sumR * inv), Float(sumG * inv), Float(sumB * inv), 1)
            }
            levels.append(buffer)
        }
        return PreparedSlice(levels: levels, uvScale: SIMD2<Float>(Float(sx), Float(sy)), mean: mean)
    }
}
