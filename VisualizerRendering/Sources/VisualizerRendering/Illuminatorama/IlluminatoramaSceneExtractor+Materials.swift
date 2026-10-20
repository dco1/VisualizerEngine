import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

extension IlluminatoramaSceneExtractor {

    // ── Material extraction ───────────────────────────────────────────────────

    struct Material {
        var albedo: SIMD3<Float>
        var metallic: Float
        var roughness: Float
        var emission: SIMD3<Float>
        /// Effective emissive radiance for the Phase 4.27 light synthesis —
        /// solid emission colour OR an emission texture's average colour,
        /// each × `emission.intensity`, clamped. Kept separate from
        /// `emission` (the G-buffer term) so texture-driven glow can light
        /// the scene without washing the rendered surface toward white.
        var lightEmission: SIMD3<Float> = .zero
        /// Multiplier on the emission TEXTURE sample (Phase 4.27b) so a
        /// texture-driven glow renders at its tuned `emission.intensity`.
        var emissionIntensity: Float = 1.0
        var albedoTextureSlice: Int32 = -1
        var metallicTextureSlice: Int32 = -1
        var roughnessTextureSlice: Int32 = -1
        var normalTextureSlice: Int32 = -1
        var emissionTextureSlice: Int32 = -1
    }


    /// Average colour of an emission texture (NSImage / CGImage), used as a
    /// coarse radiance estimate so texture-driven glow (Pizza's rainbow oven
    /// coils) can synthesise a light. Rasterises the image to 1×1 and reads
    /// the pixel; cached by contents identity. Returns nil for non-image
    /// contents (the caller falls back to the solid-colour path).
    private func emissionTextureAverage(_ contents: Any?) -> SIMD3<Float>? {
        guard let contents = contents else { return nil }
        let key = ObjectIdentifier(contents as AnyObject)
        if let cached = emissionAvgCache[key] { return cached }
        let cg: CGImage?
        if let img = contents as? PlatformImage {
            cg = Self.cgImage(from: img)
        } else if CFGetTypeID(contents as CFTypeRef) == CGImage.typeID {
            cg = (contents as! CGImage)
        } else {
            cg = nil
        }
        guard let cgImage = cg else {
            emissionAvgCache[key] = .some(nil)
            return nil
        }
        var px: [UInt8] = [0, 0, 0, 0]
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: &px, width: 1, height: 1,
                                   bitsPerComponent: 8, bytesPerRow: 4,
                                   space: cs, bitmapInfo: info) else {
            emissionAvgCache[key] = .some(nil)
            return nil
        }
        ctx.interpolationQuality = .medium
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        // Device-RGB bytes are ~sRGB; rough-decode to linear (gamma 2.2) so
        // the radiance estimate is in the same space as the lighting maths.
        let avg = SIMD3<Float>(pow(Float(px[0]) / 255.0, 2.2),
                               pow(Float(px[1]) / 255.0, 2.2),
                               pow(Float(px[2]) / 255.0, 2.2))
        emissionAvgCache[key] = .some(avg)
        return avg
    }

    func extractMaterial(_ material: SCNMaterial?,
                                  renderer: IlluminatoramaRenderer) -> Material {
        guard let material = material else {
            return Material(albedo: SIMD3(0.8, 0.8, 0.8),
                            metallic: 0.0,
                            roughness: 0.5,
                            emission: .zero)
        }

        // Phase 4.0/4.1 — register image-based material properties as
        // texture-atlas slices. A diffuse image goes into the sRGB albedo
        // atlas; metalness / roughness images go into the linear
        // non-colour atlas. Each registration returns nil when the input
        // isn't an image or the atlas is full — the existing scalar
        // fallback then drives that property.
        //
        // Diagnostic env var `VIZ_ILLUMINATORAMA_NO_TEXTURES=1` bypasses
        // registration so you can A/B the textured vs scalar paths
        // without rebuilding. (Also keyed off the older
        // `VIZ_DISABLE_TEXTURE_ATLAS` name for back-compat with shell
        // scripts.)
        // Per MATERIAL, every frame (`IlluminatoramaOverlay.tick` → `extractFrame`), so `getenv`,
        // not two whole-environment Dictionary builds (see `IlluminatoramaRenderer.frameEnv`).
        let disableTex =
            IlluminatoramaRenderer.frameEnv("VIZ_ILLUMINATORAMA_NO_TEXTURES") == "1"
            || IlluminatoramaRenderer.frameEnv("VIZ_DISABLE_TEXTURE_ATLAS") == "1"
        let albedoSlice: Int32
        let metallicSlice: Int32
        let roughnessSlice: Int32
        let normalSlice: Int32
        let emissionSlice: Int32
        if disableTex {
            albedoSlice = -1
            metallicSlice = -1
            roughnessSlice = -1
            normalSlice = -1
            emissionSlice = -1
        } else {
            albedoSlice = renderer.albedoAtlas.register(contents: material.diffuse.contents) ?? -1
            metallicSlice = renderer.nonColorAtlas.register(contents: material.metalness.contents) ?? -1
            roughnessSlice = renderer.nonColorAtlas.register(contents: material.roughness.contents) ?? -1
            normalSlice = renderer.nonColorAtlas.register(contents: material.normal.contents) ?? -1
            emissionSlice = renderer.albedoAtlas.register(contents: material.emission.contents) ?? -1
        }
        let albedo = scnContentsToRGB(material.diffuse.contents)
                    ?? SIMD3(0.8, 0.8, 0.8)

        // Honour `emission.intensity` (the SCN HDR-emissive knob). The
        // canonical neon-tube/glow recipe is a moderate emission COLOUR ×
        // a high `emission.intensity` (often 2–8); without folding the
        // intensity in, every emissive surface rendered at intensity 1, so
        // glowing things read flat AND the Phase 4.27 emissive-light
        // synthesis (which keys off this scalar) never fired on them.
        //
        // Texture-driven emission (`emission.contents` is an image, e.g.
        // Pizza's rainbow coils) is left to the shader's emission-texture
        // sample — we do NOT inject a white scalar here, which would wash
        // the texture toward white in the G-buffer. Lighting from
        // texture-emissive surfaces needs a per-texture average-radiance
        // estimate; that's a follow-on.
        let emissionIntensity = Float(material.emission.intensity)
        var emission = (scnContentsToRGB(material.emission.contents) ?? .zero)
                     * emissionIntensity
        // Clamp peak emission so an extreme `emission.intensity` (neon signs
        // run 4–8×) can't blow the whole frame to white through bloom +
        // tonemap. Moderate emissive (CityStreet windows ≈ 1–3×) stays under
        // the cap unaffected; only runaway neon is compressed.
        let ePeak = max(emission.x, max(emission.y, emission.z))
        if ePeak > emissiveRenderClamp {
            emission *= (emissiveRenderClamp / ePeak)
        }

        // Phase 4.27a — effective emissive radiance for light synthesis. Use
        // the solid emission colour when there is one; otherwise (texture-
        // driven glow, e.g. Pizza's rainbow coils) fall back to the emission
        // texture's average colour. × intensity, clamped like the rendered
        // term. Kept separate from `emission` so the texture average never
        // washes the rendered G-buffer surface.
        let lightBase = scnContentsToRGB(material.emission.contents)
                     ?? emissionTextureAverage(material.emission.contents)
                     ?? .zero
        var lightEmission = lightBase * emissionIntensity
        let lPeak = max(lightEmission.x, max(lightEmission.y, lightEmission.z))
        if lPeak > emissiveRenderClamp {
            lightEmission *= (emissiveRenderClamp / lPeak)
        }

        // Metallic / roughness: PBR materials carry these as
        // `SCNMaterialProperty.contents` (a number 0–1) or as a texture.
        // Lambert / Blinn carry only shininess; derive a roughness.
        let metallic: Float
        let roughness: Float
        if material.lightingModel == .physicallyBased {
            metallic = scnContentsToScalar(material.metalness.contents) ?? 0.0
            roughness = scnContentsToScalar(material.roughness.contents) ?? 0.5
        } else {
            metallic = 0.0
            // Phong shininess (0 = matte ... ~128 = mirror-ish). Map
            // log-spaced shininess to perceptual roughness, clamping.
            let shininess = Float(material.shininess)
            // shininess=0 → roughness 1; shininess≥128 → roughness 0.05
            let r = 1.0 - min(1.0, shininess / 128.0)
            roughness = max(0.05, r)
        }
        return Material(albedo: albedo,
                        metallic: metallic,
                        roughness: roughness,
                        emission: emission,
                        lightEmission: lightEmission,
                        emissionIntensity: emissionIntensity,
                        albedoTextureSlice: albedoSlice,
                        metallicTextureSlice: metallicSlice,
                        roughnessTextureSlice: roughnessSlice,
                        normalTextureSlice: normalSlice,
                        emissionTextureSlice: emissionSlice)
    }

    // ── Equirect-sky extraction ─────────────────────────────────────────────

    /// Surface the host SCNScene's `lightingEnvironment.contents` as an
    /// `MTLTexture` for `IlluminatoramaRenderer.equirectSky`. Cached per
    /// content identity so we upload each unique image once. Returns nil
    /// when no environment is set or the contents are a type we can't
    /// convert (MTLTexture passed directly is uncommon in app-level
    /// scenes; can be added later).
    ///
    /// Falls back to `scene.background.contents` when `lightingEnvironment`
    /// is unset — many scenes (FloatingFlowers, NapaValley, etc.) only
    /// set the background image and rely on SCN to auto-derive the IBL
    /// from it. Picking up the background as the equirectSky also makes
    /// the rendered sky pixels (depth=1.0 in `illumi_lighting`) match
    /// what the SCN renderer shows, which is the main visible difference
    /// for those scenes.
    // ── Phase 4.14 — per-node material overrides ───────────────────

    func applyNodeOverrides(node: SCNNode, to material: inout Material) {
        guard let ov = node.illuminatoramaOverride else { return }
        if let v = ov.albedo    { material.albedo    = v }
        if let v = ov.metallic  { material.metallic  = v }
        if let v = ov.roughness { material.roughness = v }
        if let v = ov.emission  {
            material.emission = v
            // Keep the light-synthesis radiance in step with an overridden
            // emission so an override-driven glow also lights the scene.
            material.lightEmission = v
        }
    }

    func extractEquirectSky(scene: SCNScene) -> MTLTexture? {
        let contents = scene.lightingEnvironment.contents
                    ?? scene.background.contents
        guard let contents = contents else { return nil }
        // Fast-path: already an MTLTexture (some scenes do this for
        // VolumetricCloudRenderer's HDR output). Hand it straight through.
        if let direct = contents as? MTLTexture { return direct }
        // NSImage / CGImage path — same identity-cache trick as the
        // material atlas.
        let key: ObjectIdentifier
        let cg: CGImage
        if let img = contents as? PlatformImage {
            key = ObjectIdentifier(img)
            if let cached = equirectSkyCache[key] { return cached }
            guard let extracted = Self.cgImage(from: img) else {
                equirectSkyCache[key] = .some(nil)
                return nil
            }
            cg = extracted
        } else if CFGetTypeID(contents as CFTypeRef) == CGImage.typeID {
            let img = contents as! CGImage
            key = ObjectIdentifier(img as AnyObject)
            if let cached = equirectSkyCache[key] { return cached }
            cg = img
        } else {
            return nil
        }
        let texture = Self.uploadEquirect(cgImage: cg, device: device)
        equirectSkyCache[key] = .some(texture)
        return texture
    }

    /// Upload a CGImage as an `rgba16Float` equirect texture, keeping its
    /// source resolution (capped at 4096×2048 to avoid pathological VRAM
    /// hits from arbitrary asset sizes). LDR source values go to half-
    /// floats unchanged — proper HDR (.hdr / .exr loaded via ImageIO with
    /// the float-data flag) would preserve >1 luminance; LDR sources
    /// stay clamped at 1.0. Sufficient for the Plus-scene environments
    /// shipped as 8-bit sRGB equirect PNGs today.
    private static func uploadEquirect(cgImage: CGImage,
                                        device: MTLDevice) -> MTLTexture? {
        let w = min(4096, cgImage.width)
        let h = min(2048, cgImage.height)
        guard w > 0, h > 0 else { return nil }
        let bytesPerRow8 = w * 4
        var buf8 = [UInt8](repeating: 0, count: bytesPerRow8 * h)
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
                       | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = buf8.withUnsafeMutableBytes({ raw -> CGContext? in
            guard let base = raw.baseAddress else { return nil }
            return CGContext(data: base, width: w, height: h,
                             bitsPerComponent: 8, bytesPerRow: bytesPerRow8,
                             space: cs, bitmapInfo: bitmapInfo)
        }) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Convert BGRA8 → RGBA32Float. Float16 is unavailable on x86_64 so
        // the buffer uses Float; rgba32Float textures are read identically
        // to rgba16Float in Metal shaders. sRGB → linear via gamma 2.2.
        var linear = [Float](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            let b = Float(buf8[i * 4 + 0]) / 255.0
            let g = Float(buf8[i * 4 + 1]) / 255.0
            let r = Float(buf8[i * 4 + 2]) / 255.0
            linear[i * 4 + 0] = pow(r, 2.2)
            linear[i * 4 + 1] = pow(g, 2.2)
            linear[i * 4 + 2] = pow(b, 2.2)
            linear[i * 4 + 3] = 1.0
        }
        let d = MTLTextureDescriptor()
        d.textureType = .type2D
        d.pixelFormat = .rgba32Float
        d.width = w
        d.height = h
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        tex.label = "Illuminatorama.sceneEquirect"
        linear.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            tex.replace(region: MTLRegionMake2D(0, 0, w, h),
                        mipmapLevel: 0,
                        withBytes: base,
                        bytesPerRow: w * 4 * MemoryLayout<Float>.stride)
        }
        return tex
    }

    private static func cgImage(from image: PlatformImage) -> CGImage? {
        image.platformCGImage
    }
}

// MARK: - Color/contents helpers (file-private, free functions so they
//        can be reused by IlluminatoramaMesh.from(scnGeometry:) too).

@MainActor
func scnColorToRGB(_ color: PlatformColor) -> SIMD3<Float> {
    // Convert to sRGB components; srgbComponents returns nil for some
    // non-RGB input colours (named, pattern, etc.); fall back to mid-grey
    // rather than letting a force-unwrap crash on a content edge case.
    let rgb = color.srgbComponents ?? SIMD4(0.5, 0.5, 0.5, 1)
    // sRGB-encoded components → LINEAR. Illuminatorama shades in linear and
    // tonemaps to sRGB on store; the texture-atlas, emission-average, and
    // environment paths all decode with `pow(x, 2.2)` (see
    // emissionTextureAverage / uploadEnvironment). Solid-colour albedo/
    // emission/light-colour used to skip this decode, so a SceneKit material's
    // sRGB diffuse was fed in as if linear — translated scenes washed out
    // (green felt → khaki). Decode here so colour matches the textured paths.
    return SIMD3(pow(Float(rgb.x), 2.2),
                 pow(Float(rgb.y), 2.2),
                 pow(Float(rgb.z), 2.2))
}

@MainActor
func scnContentsToRGB(_ contents: Any?) -> SIMD3<Float>? {
    if let color = contents as? PlatformColor {
        return scnColorToRGB(color)
    }
    if let cf = contents, CFGetTypeID(cf as CFTypeRef) == CGColor.typeID {
        let cgColor = cf as! CGColor
        if let platform = PlatformColor.fromCG(cgColor) {
            return scnColorToRGB(platform)
        }
    }
    if let num = contents as? NSNumber {
        let v = Float(truncating: num)
        return SIMD3(repeating: v)
    }
    // Image / texture contents — Phase 2.6 leaves these unhandled
    // (sampling them on the GPU needs the renderer to take a texture per
    // instance, which is Phase 4-level work). Returning nil signals
    // "fall back to default" at the call site.
    return nil
}

func scnContentsToScalar(_ contents: Any?) -> Float? {
    if let num = contents as? NSNumber {
        return Float(truncating: num)
    }
    if let color = contents as? PlatformColor {
        // Some scenes pack a single-channel value into a colour. Take the
        // red channel as the value.
        let rgb = color.srgbComponents ?? SIMD4(0.5, 0.5, 0.5, 1)
        return Float(rgb.x)
    }
    return nil
}
