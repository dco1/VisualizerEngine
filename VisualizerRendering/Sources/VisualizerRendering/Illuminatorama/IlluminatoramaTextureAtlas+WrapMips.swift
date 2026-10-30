import CoreGraphics
import Foundation

// DH-1305 — every atlas slice is a TILE: the shader samples it with `address::repeat` (or the
// manual `fract()` of a letterboxed axis), so texel 0 sits next to texel n−1. Core Graphics
// resamples an image as if it ended at its border — the filter footprint of an edge texel is
// CLAMPED, replicating the edge column outward instead of reading the opposite edge. On a level
// that is downsampled from the source, any feature straddling the tile edge (a grout line, a plank
// joint, a weave row) is therefore counted up to twice: measured on a 4 px grout line split 2 + 2
// across the edge of a 512² tile, the 32² level's edge texel read 108 against 189 for the same
// line in the tile interior (8²: 120 vs 208). Every UV-tile boundary then carries a dark line
// that gets WIDER with distance — the coarse moiré lattice the DH-1305 filtering audit measured on
// the tile floors (detail 1.5–2.0× a 64-spp reference, pixel error > 1, i.e. structure the
// reference does not have).
//
// The fix is to filter the tile as what it is: resample a 3 × 3 TILING of the source in one draw
// and keep the centre, so the edge texels' filter footprint reads the real neighbours. (Drawing nine
// separate copies does not do it — CG resamples each draw in isolation, clamped at its own border;
// measured identical, 108 vs 189.) Square sources only; a letterboxed slice keeps the clamped path.
extension IlluminatoramaTextureAtlas {

    /// Lever (DH-1305), read at every slice registration. Ships `false` = the clamp-at-the-border
    /// draw, byte-identical to before: turning it on moved Daydream's
    /// `testCascadeSplitDoesNotKinkAShadowEdgeCrossingIt` (an edge tracker run over a TEXTURED
    /// floor: 2.28× → 3.63× against a 2.6 bar) and that has to be attributed before it ships —
    /// see Daydream DH-1597 / DH-1599. Measured ON in Daydream's filtering audit: carpet far-band
    /// shimmer 2.65 → 1.06 (reference 1.19), oak far mean 0.986 → 0.999 of the reference, the
    /// wall-tile UV-seam lattice gone (far detail 2.05 → 1.63× with the resolve lever).
    nonisolated(unsafe) public static var wrapFilteredMips = false

    /// `image` tiled 3 × 3 at its own size — nine exact 1:1 copies, no resampling.
    static func tiledSource(_ image: CGImage, space: CGColorSpace, bitmapInfo: UInt32) -> CGImage? {
        let w = image.width, h = image.height
        guard w > 0, h > 0, w <= 4096, h <= 4096,
              let ctx = CGContext(data: nil, width: 3 * w, height: 3 * h, bitsPerComponent: 8,
                                  bytesPerRow: 3 * w * 4, space: space, bitmapInfo: bitmapInfo)
        else { return nil }
        ctx.interpolationQuality = .none
        for j in 0..<3 { for i in 0..<3 {
            ctx.draw(image, in: CGRect(x: i * w, y: j * h, width: w, height: h))
        } }
        return ctx.makeImage()
    }

    /// One `n × n` level, resampled from the 3 × 3 `tiled` source in a single draw and cropped to
    /// the centre period. nil (⇒ the caller's clamped draw) for a level that is a 1:1 copy of the
    /// source, where no filter footprint reaches the border and the result would be identical.
    static func drawWrapped(tiled: CGImage, source: CGImage, n: Int,
                            space: CGColorSpace, bitmapInfo: UInt32) -> [UInt8]? {
        if source.width == n, source.height == n { return nil }
        let big = 3 * n
        guard let ctx = CGContext(data: nil, width: big, height: big, bitsPerComponent: 8,
                                  bytesPerRow: big * 4, space: space, bitmapInfo: bitmapInfo),
              let data = ctx.data
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(tiled, in: CGRect(x: 0, y: 0, width: big, height: big))
        let src = data.bindMemory(to: UInt8.self, capacity: big * big * 4)
        var out = [UInt8](repeating: 0, count: n * n * 4)
        out.withUnsafeMutableBytes { raw in
            guard let dst = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            // The tiling is symmetric, so the centre period is rows n..<2n, cols n..<2n.
            for r in 0..<n { memcpy(dst + r * n * 4, src + ((n + r) * big + n) * 4, n * 4) }
        }
        return out
    }
}
