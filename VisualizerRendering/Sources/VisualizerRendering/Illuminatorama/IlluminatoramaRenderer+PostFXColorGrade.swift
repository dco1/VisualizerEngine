import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// Set the film stock's colour cube.
    ///
    /// The size is read OFF THE TEXTURE, not passed in and not assumed: the cube used to
    /// be a hardcoded 16 on both sides, so shipping the 33-cube the stocks are actually
    /// authored at would have silently sampled a 33³ texture on 16-cell coordinates —
    /// wrong everywhere, and wrong in a way that reads as a grade rather than as a bug.
    /// - Parameters:
    ///   - texture: An N×N×N 3D texture (any sampleable colour format — the app ships
    ///     `rgba16Float`). Non-cubic or non-3D textures are rejected. `nil` disables.
    ///   - strength: Blend weight toward the stock's rendering [0, 1]. Defaults 1.
    public func setFilmLUT(_ texture: MTLTexture?, strength: Float = 1.0) {
        guard let texture else {
            filmLUTTexture = nil
            filmLUTStrength = strength
            return
        }
        guard texture.textureType == .type3D,
              texture.width == texture.height, texture.height == texture.depth,
              texture.width >= 2 else {
            assertionFailure("film LUT must be a cubic 3D texture, got \(texture.textureType) \(texture.width)×\(texture.height)×\(texture.depth)")
            filmLUTTexture = nil
            return
        }
        filmLUTTexture = texture
        filmLUTStrength = strength
        filmLUTSize = Float(texture.width)
    }

    /// Diagnostic A/B override: `VIZ_ILLUMI_TAA=1`/`0` forces the TAA toggle for
    /// every Illuminatorama renderer in the process, bypassing both the panel and
    /// any scene `sharedTAAOverride`. Lets a headless render exercise the TAA path
    /// without UI. `nil` when the env var is unset. Read once.
    static let taaEnvOverride: Bool? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_TAA"] else { return nil }
        return v == "1" || v.lowercased() == "true"
    }()

    // Issue #65 — headless motion-blur override. Motion blur is otherwise driven
    // only by the panel, so a headless render can't exercise it; `VIZ_ILLUMI_MOTIONBLUR`
    // (a float, e.g. `1.0`) forces `motionBlurStrength` for every renderer in the
    // process so the velocity-buffer streak can be render-verified. `nil` when unset.
    nonisolated static let motionBlurEnvOverride: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_MOTIONBLUR"] else { return nil }
        return Float(v)
    }()

    // Issue #65 — headless contact-shadow override. Contact shadows are otherwise
    // driven only by the panel, so a headless render can't exercise them;
    // `VIZ_ILLUMI_CONTACTSHADOW` (a float strength, e.g. `0.85`) forces
    // `contactShadowStrength` for every renderer in the process so the contact
    // darkening can be render-verified A/B. `nil` when unset.
    nonisolated static let contactShadowEnvOverride: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_CONTACTSHADOW"] else { return nil }
        return Float(v)
    }()

    // ── Issue #65: colour-grading LUT ─────────────────────────────────────────

    /// IEEE-754 binary32 → binary16 (half) bit pattern for `rgba16Float` uploads.
    /// Software conversion: Swift's `Float16` is unavailable under Mac Catalyst.
    /// LUT values are finite and in [0, 1], so the round-to-nearest-even path is
    /// all that matters; sub-normals/inf/NaN are handled for completeness.
    @inline(__always)
    static func float32to16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let expo = Int32((bits >> 23) & 0xFF) - 127 + 15
        let mant = bits & 0x7F_FFFF
        if expo >= 0x1F {                                   // overflow / inf / NaN
            let isNaN = (bits & 0x7F80_0000) == 0x7F80_0000 && mant != 0
            return sign | (isNaN ? 0x7E00 : 0x7C00)
        }
        if expo <= 0 {                                      // sub-normal / zero
            guard expo > -11 else { return sign }
            let m = (mant | 0x80_0000) >> UInt32(1 - expo + 13)
            return sign | UInt16(m)
        }
        // Normal: round mantissa to 10 bits, nearest-even.
        var half = UInt32(expo) << 10 | (mant >> 13)
        let roundBits = mant & 0x1FFF
        if roundBits > 0x1000 || (roundBits == 0x1000 && (half & 1) == 1) { half += 1 }
        return sign | UInt16(truncatingIfNeeded: half)
    }

    /// Build an identity 3D LUT (`rgba16Float`, `size³`) where the texel at
    /// (r,g,b) holds the colour (r,g,b)/(size-1). Sampling it with a colour in
    /// [0,1]³ (half-texel-corrected) returns that colour unchanged — an exact
    /// no-op grade. Uploaded once; `.shared` storage so the CPU bake is visible.
    static func makeIdentityColorLUT(device: MTLDevice, size: Int) -> MTLTexture? {
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.label = "Illuminatorama.colorLUT"
        var data = [UInt16](repeating: 0, count: size * size * size * 4)
        let inv = 1.0 / Float(size - 1)
        var i = 0
        for b in 0..<size {
            for g in 0..<size {
                for r in 0..<size {
                    data[i+0] = float32to16(Float(r) * inv)
                    data[i+1] = float32to16(Float(g) * inv)
                    data[i+2] = float32to16(Float(b) * inv)
                    data[i+3] = float32to16(1.0)
                    i += 4
                }
            }
        }
        let bpr = size * 4 * MemoryLayout<UInt16>.size
        let bpi = bpr * size
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size),
                        mipmapLevel: 0, slice: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: bpr, bytesPerImage: bpi)
        }
        return tex
    }

    /// Replace the colour-grade LUT from packed RGBA float data laid out as the
    /// standard .cube ordering (red fastest, then green, then blue). `rgba` must
    /// hold `size³` SIMD4 entries. Pass `nil` to reset to identity. The grade only
    /// shows when `colorLUTAmount > 0`. Display-space (the LUT is sampled after
    /// tonemapping, on the sRGB-encoded colour).
    @discardableResult
    public func setColorLUT(size: Int, rgba: [SIMD4<Float>]?) -> Bool {
        guard let rgba else {
            if let id = Self.makeIdentityColorLUT(device: device, size: Self.colorLUTDefaultSize) {
                pfx.colorLUT = id; pfx.colorLUTSize = Self.colorLUTDefaultSize; return true
            }
            return false
        }
        guard size >= 2, rgba.count == size * size * size else { return false }
        let d = MTLTextureDescriptor()
        d.textureType = .type3D
        d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return false }
        tex.label = "Illuminatorama.colorLUT(custom)"
        var data = [UInt16](repeating: 0, count: rgba.count * 4)
        for (j, c) in rgba.enumerated() {
            data[j*4+0] = Self.float32to16(c.x)
            data[j*4+1] = Self.float32to16(c.y)
            data[j*4+2] = Self.float32to16(c.z)
            data[j*4+3] = Self.float32to16(1.0)
        }
        let bpr = size * 4 * MemoryLayout<UInt16>.size
        let bpi = bpr * size
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size),
                        mipmapLevel: 0, slice: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: bpr, bytesPerImage: bpi)
        }
        pfx.colorLUT = tex; pfx.colorLUTSize = size
        return true
    }

    /// Parse an Adobe/Resolve `.cube` 3D LUT and install it. Returns false on a
    /// malformed file. Supports `LUT_3D_SIZE` + `r g b` rows (the common subset).
    @discardableResult
    public func loadCubeLUT(contentsOf url: URL) -> Bool {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return false }
        var size = 0
        var entries: [SIMD4<Float>] = []
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            if line.uppercased().hasPrefix("LUT_3D_SIZE") {
                size = Int(line.split(separator: " ").last.map(String.init) ?? "") ?? 0
                if size > 1 { entries.reserveCapacity(size*size*size) }
                continue
            }
            let comps = line.split(separator: " ").compactMap { Float($0) }
            if comps.count == 3 { entries.append(SIMD4(comps[0], comps[1], comps[2], 1)) }
        }
        guard size > 1, entries.count == size*size*size else { return false }
        return setColorLUT(size: size, rgba: entries)
    }

    /// Generate one of the built-in procedural grades as packed RGBA LUT data
    /// (.cube order, red fastest). Returns nil for `.none` (caller installs the
    /// identity LUT). The grade is computed in DISPLAY (~sRGB) space, matching how
    /// the tonemap samples the LUT.
    static func generateColorGradeLUT(look: IlluminatoramaColorGradeLook,
                                      size: Int) -> [SIMD4<Float>]? {
        guard look != .none, size >= 2 else { return nil }
        func luma(_ c: SIMD3<Float>) -> Float { c.x*0.2126 + c.y*0.7152 + c.z*0.0722 }
        func contrast(_ c: SIMD3<Float>, _ k: Float) -> SIMD3<Float> { (c - 0.5) * k + 0.5 }
        func smooth(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
            let t = max(0, min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t)
        }
        var out = [SIMD4<Float>](); out.reserveCapacity(size*size*size)
        let inv = 1.0 / Float(size - 1)
        for b in 0..<size { for g in 0..<size { for r in 0..<size {
            var c = SIMD3(Float(r), Float(g), Float(b)) * inv
            switch look {
            case .none: break
            case .tealOrange:
                let t = smooth(0.2, 0.85, luma(c))
                let shadow = SIMD3<Float>(-0.02, 0.03, 0.08)   // push shadows teal
                let high   = SIMD3<Float>( 0.09, 0.02, -0.07)  // push highlights orange
                c += shadow * (1 - t) + high * t
                c = contrast(c, 1.10)
            case .warm:
                c *= SIMD3<Float>(1.07, 1.01, 0.90); c = contrast(c, 1.05)
            case .cool:
                c *= SIMD3<Float>(0.92, 1.00, 1.09); c = contrast(c, 1.05)
            case .noir:
                let l = luma(c); c = contrast(SIMD3(l, l, l), 1.22)
            case .vibrant:
                let l = luma(c); c = l + (c - l) * 1.45; c = contrast(c, 1.06)
            }
            c = simd_clamp(c, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
            out.append(SIMD4(c, 1))
        }}}
        return out
    }

    /// Apply the shared panel's colour grade (issue #65). Rebakes the LUT only when
    /// the chosen look changes (the CPU bake is a per-frame no-op in steady state),
    /// and sets `colorLUTAmount` each frame. Called from `render()` alongside the
    /// other shared-FX appliers, so every Illuminatorama scene inherits it.
    func applySharedColorGrade(_ s: IlluminatoramaSharedSettings = .shared) {
        if s.colorGradeLook != pfx.appliedColorGradeLook {
            pfx.appliedColorGradeLook = s.colorGradeLook
            let data = Self.generateColorGradeLUT(look: s.colorGradeLook,
                                                  size: Self.colorLUTDefaultSize)
            setColorLUT(size: Self.colorLUTDefaultSize, rgba: data)
        }
        colorLUTAmount = (s.colorGradeLook == .none) ? 0 : Float(s.colorLUTAmount)
    }

    /// Advance the eased Post-FX shadows toward their slider targets. `Instant`
    /// (tau = 0) snaps; otherwise exponential smoothing `k = 1 − e^(−dt/tau)`.
    /// Self-timed off `CACurrentMediaTime` so it needs no dt from the caller, and
    /// the first call (lastEaseTime == 0 → dt 0 → k 1) seeds the shadows at the
    /// current targets (no startup glide). Mirrors AspectTest's `easeCamera`.
    func advancePostFXEasing() {
        let now = CACurrentMediaTime()
        let dt = pfx.lastPostFXEaseTime == 0 ? 0 : max(0, min(0.1, fixedFrameDelta ?? (now - pfx.lastPostFXEaseTime)))
        pfx.lastPostFXEaseTime = now
        let k = (postFXEasingTau <= 0 || dt == 0) ? 1.0 : 1 - exp(-dt / postFXEasingTau)
        let kf = Float(k)
        pfx.easedExposure          += (exposure              - pfx.easedExposure)          * kf
        pfx.easedBloomThreshold    += (bloomThreshold        - pfx.easedBloomThreshold)    * kf
        pfx.easedBloomIntensity    += (bloomIntensity        - pfx.easedBloomIntensity)    * kf
        pfx.easedChromaticAberration += (max(0, chromaticAberration) - pfx.easedChromaticAberration) * kf
        pfx.easedFringe            += (max(0, fringe)        - pfx.easedFringe)            * kf
        pfx.easedFringeTint        += (fringeTint            - pfx.easedFringeTint)        * kf
        pfx.easedSphericalAberration += (max(0, sphericalAberration) - pfx.easedSphericalAberration) * kf
        pfx.easedHalationIntensity += (max(0, halationIntensity) - pfx.easedHalationIntensity) * kf
        pfx.easedHalationThreshold += (halationThreshold        - pfx.easedHalationThreshold) * kf
        pfx.easedHalationRadius    += (max(0, halationRadius)   - pfx.easedHalationRadius)    * kf
        pfx.easedHalationTint      += (halationTint             - pfx.easedHalationTint)      * kf
        // Exponential easing approaches 0 asymptotically, so a scene that turns halation
        // OFF would keep encoding three dispatches forever on a residue of ~1e-9. Snap
        // the last sliver: `encodeHalationPasses` and the shader branch both key off
        // this value, so the effect switches off exactly when the passes stop.
        if halationIntensity <= 0 && pfx.easedHalationIntensity < 1e-3 { pfx.easedHalationIntensity = 0 }
    }

    // Issue #65 — colour-grade LUT default per-axis resolution. 33 is the .cube
    // de-facto standard (matches DaVinci / most authoring tools).
    static let colorLUTDefaultSize: Int = 33
}
