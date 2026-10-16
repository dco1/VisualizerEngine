import Foundation

/// **The supersample → output reduction, as a declared reconstruction filter (Daydream DH-1365).**
///
/// Every frame is rendered at `output × internalRenderScale` and reduced to the output size in ONE
/// place: the HDR read at the top of `illumi_tonemap_fs` (IlluminatoramaTonemap.metal), on the
/// scene-referred, linear-light radiance the TAA accumulation hands downstream — before white
/// balance, exposure, the lens's cos⁴ falloff and the display transform. That is where a camera's
/// sensor does it (photosites integrate light, then a tone curve is applied), and it is the only
/// place a filter means the same thing at every exposure and every grade: filtering after the
/// display transform would average CODE VALUES, so an edge's anti-aliasing would move with the
/// exposure dial and a dark/bright boundary would come out brighter than its light says
/// (the classic gamma-space AA error). Exposure, white balance and the cos⁴ field are per-pixel
/// scalings that commute with a linear filter, so nothing between the filter and the display
/// transform cares where the reduction happened.
///
/// ## The legacy reduction (`Kernel.legacy`, kernel 0 — every host's default)
///
/// Four `filter::linear` taps at ±½ INTERNAL texel around the output pixel's centre. Its shape
/// depends on the scale, which the shader comment above it does not say:
///
/// * **2.0×**: the taps land on texel centres — a 2 × 2 BOX (one output pixel wide). Combined with
///   the still's jitter (a box over each internal pixel) the still's whole reconstruction is a
///   one-output-pixel box: a sinc response, −13 dB first side lobe, the most aliasing-prone of the
///   textbook filters.
/// * **1.0×**: the taps land on texel CORNERS, so each averages a 2 × 2 block and the four together
///   are a 3 × 3 [1 2 1]⊗[1 2 1]/16 TENT — a blur, not the "single bilinear read" the shader comment
///   claims. Any frame rendered at scale 1 (the draft tier, an export whose supersample floored at 1)
///   is softened by it.
/// * **Fractional** (1.5× live, ~1.6–1.78× on a 4K export under `maxInternalMegapixels`): the tap
///   positions walk across the texel grid from one output pixel to the next, so the kernel's shape
///   changes per pixel with period `1/frac(scale)` — a fixed-pattern modulation on fine detail.
///
/// `ReconstructionFilterTests` measures all three on the CPU through `gather1D`, which mirrors the
/// shader's arithmetic exactly.
///
/// ## The declared kernels (opt-in)
///
/// A separable gather over every internal texel within `radius` OUTPUT pixels of the output pixel's
/// centre, each weighted by the kernel at its centre's distance and normalised by the weights'
/// sum — so a flat field is reproduced exactly at any scale and any phase, and the kernel's width is
/// fixed in output pixels (the picture's grid) whatever the supersample is.
///
/// * `blackmanHarris` — the 4-term Blackman–Harris window over ±radius. NON-NEGATIVE everywhere,
///   which is the property that matters in linear HDR: a filter's negative lobe multiplies the
///   brightest neighbour, so a window 50× brighter than the wall beside it turns a −1 % lobe into a
///   −50 % dark halo along the frame. The default the Maximum output contract asks for.
/// * `mitchellNetravali` — B = C = ⅓ over ±radius (its natural support is ±2). Sharper, with the
///   small negative lobe Mitchell & Netravali chose for perceived sharpness; offered for A/B. Its
///   result is clamped at 0 per channel, which removes a NEGATIVE pixel but not a dark halo.
/// * `tent`, `box` — the two textbook baselines, for measurement.
///
/// The shader evaluates exactly `weight(_:)` below; the two are kept in step by
/// `ReconstructionFilterTests.testShaderSourceMirrorsTheSwiftKernelCoefficients`.
public struct IlluminatoramaReconstructionFilter: Sendable, Equatable {

    /// Which kernel. The raw value is what the shader switches on (`FrameUniforms.taaJitterDelta.z`);
    /// 0 is the legacy 4-tap path and is never a declared kernel.
    public enum Kernel: UInt32, Sendable, CaseIterable {
        case legacy = 0
        case box = 1
        case tent = 2
        case mitchellNetravali = 3
        case blackmanHarris = 4
    }

    public var kernel: Kernel
    /// Half-width of the kernel's support, in OUTPUT pixels. Ignored by `.legacy`.
    public var radius: Float

    public init(kernel: Kernel, radius: Float) {
        self.kernel = kernel
        self.radius = radius
    }

    /// The engine's default: the 4-tap bilinear reduction, byte-identical to every frame before it.
    public static let legacy = IlluminatoramaReconstructionFilter(kernel: .legacy, radius: 0)

    /// Blackman–Harris at its measured default radius (see `ReconstructionFilterTests`).
    public static func blackmanHarris(radius: Float = 1.5) -> Self { Self(kernel: .blackmanHarris, radius: radius) }
    /// Mitchell–Netravali (B = C = ⅓) at its natural ±2 output-pixel support.
    public static func mitchellNetravali(radius: Float = 2.0) -> Self { Self(kernel: .mitchellNetravali, radius: radius) }

    /// The shader's tap loop bound: internal texels each side of the centre. A kernel whose
    /// `radius × scale` would reach further is narrowed to fit (`effectiveRadius`), never truncated —
    /// truncating a kernel's tails changes its shape; narrowing it changes only its width.
    public static let maxTapHalfWidth = 8

    /// The radius the shader actually uses at `scale`, in output pixels.
    public func effectiveRadius(scale: Float) -> Float {
        guard kernel != .legacy else { return 0 }
        let s = max(scale, 1e-3)
        return max(0, min(radius, Float(Self.maxTapHalfWidth) / s))
    }

    /// What the engine writes into `FrameUniforms.taaJitterDelta.zw`: (kernel, radius in output px).
    /// (0, 0) for `.legacy`, which the shader reads as "take the 4-tap path".
    public var uniform: SIMD2<Float> {
        kernel == .legacy ? .zero : SIMD2(Float(kernel.rawValue), max(0, radius))
    }

    // MARK: kernel coefficients (mirrored by the shader — keep in step)

    /// 4-term Blackman–Harris (Harris 1978, "minimum 4-term", −92 dB side lobes).
    public static let blackmanHarrisA: (Double, Double, Double, Double) = (0.35875, 0.48829, 0.14128, 0.01168)
    /// Mitchell–Netravali B and C.
    public static let mitchellB = 1.0 / 3.0
    public static let mitchellC = 1.0 / 3.0

    /// The kernel's 1-D weight at `x` OUTPUT pixels from the centre, unnormalised (peak need not be 1;
    /// every use divides by the sum of the weights it gathered). 0 outside ±radius, and 0 for `.legacy`
    /// (which is not a kernel — see `gather1D`).
    public func weight(_ x: Double) -> Double {
        let r = Double(radius)
        guard r > 0 else { return 0 }
        let ax = abs(x)
        switch kernel {
        case .legacy:
            return 0
        case .box:
            return ax < r ? 1 : (ax == r ? 0.5 : 0)
        case .tent:
            return max(0, 1 - ax / r)
        case .mitchellNetravali:
            // Natural support ±2, stretched to ±radius.
            let t = ax * 2 / r
            let B = Self.mitchellB, C = Self.mitchellC
            if t < 1 {
                return ((12 - 9 * B - 6 * C) * t * t * t + (-18 + 12 * B + 6 * C) * t * t + (6 - 2 * B)) / 6
            } else if t < 2 {
                return ((-B - 6 * C) * t * t * t + (6 * B + 30 * C) * t * t + (-12 * B - 48 * C) * t + (8 * B + 24 * C)) / 6
            }
            return 0
        case .blackmanHarris:
            guard ax <= r else { return 0 }
            let a = Self.blackmanHarrisA
            let p = Double.pi * x / r          // −π … π across the support, peak at 0
            return a.0 + a.1 * cos(p) + a.2 * cos(2 * p) + a.3 * cos(3 * p)
        }
    }

    // MARK: the gather, exactly as the shader runs it

    /// The internal texels one output pixel reads along ONE axis, and their normalised weights —
    /// the CPU mirror of the shader, for both branches. The 2-D gather is the outer product (both
    /// branches are separable).
    ///
    /// - Parameters:
    ///   - outputPixel: the output pixel's index along the axis (its centre is at `outputPixel + 0.5`).
    ///   - scale: internal texels per output pixel along the axis (`internal / output`).
    ///   - internalCount: the internal texel count along the axis (edge clamping, like the sampler).
    /// - Returns: (texel index, weight) pairs; weights sum to 1. A texel may appear more than once
    ///   at a clamped edge.
    public func gather1D(outputPixel: Int, scale: Double, internalCount: Int) -> [(texel: Int, weight: Double)] {
        let n = max(internalCount, 1)
        // Continuous texel coordinate of the output pixel's centre: texel j's centre sits at j.
        let centre = (Double(outputPixel) + 0.5) * scale - 0.5
        func clampIndex(_ j: Int) -> Int { min(max(j, 0), n - 1) }
        if kernel == .legacy {
            // Four `filter::linear` taps at ±½ internal texel (two per axis), each a 2-texel
            // bilinear blend; the shader averages them.
            var out: [(Int, Double)] = []
            for off in [-0.5, 0.5] {
                let t = centre + off
                let j0 = Int(floor(t))
                let f = t - Double(j0)
                out.append((clampIndex(j0), 0.5 * (1 - f)))
                out.append((clampIndex(j0 + 1), 0.5 * f))
            }
            return out
        }
        let narrowed = Self(kernel: kernel, radius: effectiveRadius(scale: Float(scale)))
        let r = Double(narrowed.radius)
        let reach = r * scale
        let lo = Int(ceil(centre - reach)), hi = Int(floor(centre + reach))
        var taps: [(Int, Double)] = []
        var sum = 0.0
        if lo <= hi {
            for j in lo...hi {
                let w = narrowed.weight((Double(j) - centre) / scale)
                if w != 0 { taps.append((clampIndex(j), w)); sum += w }
            }
        }
        // A kernel narrower than the texel spacing can miss every centre: fall back to the
        // nearest texel (a point sample), which is what the shader does too.
        guard sum != 0 else { return [(clampIndex(Int((centre).rounded())), 1)] }
        return taps.map { ($0.0, $0.1 / sum) }
    }
}
