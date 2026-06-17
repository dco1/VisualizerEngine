import AppKit

/// Procedural texture helpers shared by all scene packages.
///
/// Originally lived inside `HotdogPlusTextures` in the HotdogDropPlus scene.
/// Moved here so Eggs, Ufo, VintageDiner, PizzaTurntablePlus, GiantGummyBearsPlus,
/// FloatingFlowersPlus, HungryPlanet, BeerStein, and others can call them without
/// taking a cross-scene dependency.
public enum ProceduralTextures {

    /// Multi-octave value noise sampled to a height field in [0,1]. Each
    /// octave is a grid of `cells` × `cells` random values, bilinearly
    /// (smoothstep-) interpolated to the output size. Cell-1 wrap so the
    /// result tiles seamlessly.
    public static func valueNoise(
        width: Int,
        height: Int,
        octaves: [(cells: Int, amp: Double)]
    ) -> [[Double]] {
        var grid = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        var totalAmp = 0.0
        for (cells, amp) in octaves {
            let octave = noiseOctave(cells: cells, width: width, height: height)
            for y in 0..<height {
                for x in 0..<width {
                    grid[y][x] += octave[y][x] * amp
                }
            }
            totalAmp += amp
        }
        if totalAmp > 0 {
            for y in 0..<height {
                for x in 0..<width {
                    grid[y][x] /= totalAmp
                }
            }
        }
        return grid
    }

    private static func noiseOctave(cells: Int, width: Int, height: Int) -> [[Double]] {
        var corners = Array(repeating: Array(repeating: 0.0, count: cells + 1), count: cells + 1)
        for j in 0..<cells {
            for i in 0..<cells {
                corners[j][i] = Double.random(in: 0...1)
            }
        }
        for j in 0..<cells {
            corners[j][cells] = corners[j][0]
        }
        for i in 0...cells {
            corners[cells][i] = corners[0][i]
        }
        var out = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        for y in 0..<height {
            let fy = Double(y) / Double(height) * Double(cells)
            let j0 = Int(fy)
            let j1 = min(j0 + 1, cells)
            let dy = fy - Double(j0)
            let dyS = dy * dy * (3 - 2 * dy)
            for x in 0..<width {
                let fx = Double(x) / Double(width) * Double(cells)
                let i0 = Int(fx)
                let i1 = min(i0 + 1, cells)
                let dx = fx - Double(i0)
                let dxS = dx * dx * (3 - 2 * dx)
                let a = corners[j0][i0] * (1 - dxS) + corners[j0][i1] * dxS
                let b = corners[j1][i0] * (1 - dxS) + corners[j1][i1] * dxS
                out[y][x] = a * (1 - dyS) + b * dyS
            }
        }
        return out
    }

    /// Convert a height field to a tangent-space normal map.
    public static func heightToNormalMap(_ height: [[Double]], strength: Double) -> NSImage {
        let h = height.count
        let w = height[0].count
        return makeImage(width: w, height: h) { x, y in
            let xR = (x + 1) % w
            let xL = (x - 1 + w) % w
            let yD = (y + 1) % h
            let yU = (y - 1 + h) % h
            let dx = (height[y][xR] - height[y][xL]) * strength
            let dy = (height[yD][x] - height[yU][x]) * strength
            let nx = -dx
            let ny = -dy
            let nz = 1.0
            let len = sqrt(nx * nx + ny * ny + nz * nz)
            let r = UInt8((((nx / len) + 1) / 2 * 255).rounded())
            let g = UInt8((((ny / len) + 1) / 2 * 255).rounded())
            let b = UInt8(((nz / len) * 255).rounded())
            return (r, g, b, 255)
        }
    }

    /// Build an `NSImage` from a per-pixel render closure. The closure returns
    /// 8-bit RGBA; the backing buffer is wrapped in a CGImage and handed to AppKit.
    public static func makeImage(
        width: Int,
        height: Int,
        render: (Int, Int) -> (r: UInt8, g: UInt8, b: UInt8, a: UInt8)
    ) -> NSImage {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        for y in 0..<height {
            for x in 0..<width {
                let (r, g, b, a) = render(x, y)
                let i = (y * width + x) * bytesPerPixel
                pixels[i] = r
                pixels[i + 1] = g
                pixels[i + 2] = b
                pixels[i + 3] = a
            }
        }
        let data = Data(pixels)
        let provider = CGDataProvider(data: data as CFData)!
        let cs = CGColorSpaceCreateDeviceRGB()
        let cg = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: cs,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: true,
            intent: .defaultIntent
        )!
        return NSImage(cgImage: cg, size: NSSize(width: width, height: height))
    }
}
