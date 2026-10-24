/// **Export Quality ▸ Maximum — the OUTPUT half of the quality contract** (DH-1365 step 1, DH-1368).
///
/// How big a Maximum still is and how densely it is sampled, decided once, GPU-free, from the
/// picture's shape and the device's memory. The other half — how the supersampled frame is REDUCED
/// to those pixels — is the engine's `IlluminatoramaReconstructionFilter` (a Blackman–Harris gather in
/// linear HDR), which the app pairs with this plan.
///
/// ## Why the plan pins the supersample at an integer 2×
///
/// `PhotoQuality` (Standard / High) caps a still at `maxInternalMegapixels` and lets the supersample
/// FLOAT to fit: a 3840 × 2400 export runs at 1.61×, 3840 × 2160 at 1.70×, Danny's 2560 × 1316
/// window at 1.78×. At a fractional scale the engine's legacy 4-tap reduction changes shape from one
/// output pixel to the next — a 5–7 % fixed pattern on fine detail, measured on the CPU mirror
/// (`ReconstructionFilterTests`) — and the still never reaches the 2× its own table measured as the
/// knee. Maximum keeps `supersample` (2.0) and spends MEMORY instead: the cap is the device's own
/// working set, not a constant sized for the smallest Mac.
///
/// ## What happens when it does not fit
///
/// Sampling is kept and the OUTPUT shrinks — never below `PhotoQuality.targetLongEdge`, so Maximum
/// is never a smaller picture than High. Only if even that cannot be held at 2× does the scale give
/// way (to the largest that fits, never below 1×), exactly as `PhotoQuality.scale` would. Either way
/// `limitedBy` says so, for the export's Render Details. Rendering a larger picture than one frame can
/// hold needs TILING, which is not built: every frame-referred effect (bloom/glare's far tail, the
/// cos⁴ vignette, lens CA, DOF's circle of confusion, the exposure meter, grain seeds) would have to
/// learn it is drawing a tile of a larger frame (filed with this ticket's report).
///
/// Pure — `PhotoOutputContractTests` checks the whole policy with no GPU.
public enum PhotoOutputContract {

    /// What a Maximum still asks for.
    public struct Request: Equatable, Sendable {
        /// Output long edge, in pixels. Never below the canvas's own long edge (a photo is never
        /// smaller than the screen it was taken from — `PhotoQuality.outputSize`'s floor).
        public var longEdge: Int
        /// Internal texels per output pixel, per axis. 2.0 = 4 samples per output pixel per frame.
        public var supersample: Float
        public init(longEdge: Int = PhotoOutputContract.maximumLongEdge, supersample: Float = 2.0) {
            self.longEdge = longEdge
            self.supersample = supersample
        }
    }

    public enum Limit: String, Equatable, Sendable {
        /// The internal target set would not fit the memory budget.
        case memory
        /// An internal dimension would pass the GPU's texture limit.
        case textureSize
    }

    /// The decided still.
    public struct Plan: Equatable, Sendable {
        public let width: Int
        public let height: Int
        /// Internal texels per output pixel, per axis.
        public let scale: Float
        /// Internal (rendered) megapixels.
        public var internalMegapixels: Double { Double(width) * Double(height) * Double(scale * scale) / 1_000_000 }
        /// The internal target set's estimated size (`bytesPerInternalPixel`).
        public var estimatedTargetBytes: Double { internalMegapixels * 1_000_000 * PhotoOutputContract.bytesPerInternalPixel }
        /// nil ⇒ the request was met; otherwise what cut it down.
        public let limitedBy: Limit?
    }

    /// Metal memory per internal pixel on the still lane, MEASURED (DH-1365, 2026-10-15, 4000 Sunset
    /// "Living Room", Standard path lane): `currentAllocatedSize` after the settle read 10.24 GB at
    /// 3840 × 2400 @ 1.614× (24.0 MPix internal) and 20.24 GB at 5120 × 3200 @ 2× (65.5 MPix) — a
    /// marginal 241 B per internal pixel (the raster target set's 194 B, `IlluminatoramaRenderer.resize`,
    /// plus the path lane's per-pixel buffers) on top of ~4.4 GB that does not scale with the frame
    /// (meshes, atlases, TLAS, radiance cache).
    public static let bytesPerInternalPixel: Double = 241

    /// Maximum's picture: a 5K long edge (Danny, 2026-10-15), at `Request.supersample`.
    public static let maximumLongEdge = 5120

    /// Share of the device's recommended working set the per-pixel memory may take; the rest is the
    /// frame-independent ~4.4 GB and headroom. At 0.6 the measured 5K @ 2× (15.8 GB per-pixel, 20.2 GB
    /// total) fits this M1 Max's 26.8 GB working set.
    public static let targetBudgetFraction: Double = 0.6

    /// Apple-silicon Metal's largest 2-D texture dimension.
    public static let maxTextureDimension = 16_384

    /// The plan for a picture of `canvasWidth × canvasHeight` (its aspect is kept exactly), on a
    /// device whose recommended working set is `workingSetBytes`.
    public static func plan(canvasWidth: Int, canvasHeight: Int,
                            request: Request = Request(),
                            workingSetBytes: UInt64) -> Plan {
        guard canvasWidth > 0, canvasHeight > 0 else {
            return Plan(width: max(canvasWidth, 1), height: max(canvasHeight, 1), scale: 1, limitedBy: nil)
        }
        let s = max(1.0, Double(request.supersample))
        let budgetPixels = Double(workingSetBytes) * targetBudgetFraction / bytesPerInternalPixel
        let canvasLong = max(canvasWidth, canvasHeight)
        let aspectShort = Double(min(canvasWidth, canvasHeight)) / Double(canvasLong)

        func size(longEdge: Int) -> (w: Int, h: Int) {
            let f = Double(longEdge) / Double(canvasLong)
            return (max(1, Int((Double(canvasWidth) * f).rounded())), max(1, Int((Double(canvasHeight) * f).rounded())))
        }
        /// Largest long edge whose internal frame at `scale` fits both limits, and which one binds.
        func fittingLongEdge(scale: Double) -> (edge: Int, limit: Limit) {
            let byMemory = (budgetPixels / (scale * scale * aspectShort)).squareRoot()
            let byTexture = Double(maxTextureDimension) / scale
            return byMemory < byTexture ? (Int(byMemory.rounded(.down)), .memory)
                                        : (Int(byTexture.rounded(.down)), .textureSize)
        }

        let wanted = max(request.longEdge, canvasLong)
        let fit = fittingLongEdge(scale: s)
        if fit.edge >= wanted {
            let o = size(longEdge: wanted)
            return Plan(width: o.w, height: o.h, scale: Float(s), limitedBy: nil)
        }
        // Keep the sampling; shrink the picture — but never below what High already exports.
        let floorEdge = max(PhotoQuality.targetLongEdge, canvasLong)
        if fit.edge >= floorEdge {
            let o = size(longEdge: fit.edge)
            return Plan(width: o.w, height: o.h, scale: Float(s), limitedBy: fit.limit)
        }
        // Even High's picture will not hold 2×: High's size, the largest scale that fits (≥ 1).
        let o = size(longEdge: min(wanted, floorEdge))
        let px = Double(o.w) * Double(o.h)
        let byMemory = (budgetPixels / px).squareRoot()
        let byTexture = Double(maxTextureDimension) / Double(max(o.w, o.h))
        let scale = max(1.0, min(s, byMemory, byTexture))
        return Plan(width: o.w, height: o.h, scale: Float(scale),
                    limitedBy: byMemory <= byTexture ? .memory : .textureSize)
    }
}
