import Metal
import simd

/// Per-instance dielectric material for the ray-traced glass pass. The author-
/// facing knobs: index of refraction, a Beer–Lambert absorption tint + strength,
/// surface roughness (frosted glass), an extra reflection scale, and optional
/// chromatic dispersion. A scene flags an object as glass with one of these and
/// the renderer routes it through `illumi_glass_rt_fs` (true entry+exit
/// refraction traced against the scene TLAS).
public struct IlluminatoramaGlassMaterial: Equatable, Sendable {
    /// Beer–Lambert absorption colour — the tint the glass imparts as light
    /// travels through it. `(1,1,1)` is perfectly clear; a colour absorbs the
    /// complementary channels with path length. Also tints the Fresnel reflection.
    public var tint: SIMD3<Float>
    /// Index of refraction. Air ≈ 1.0, water 1.33, window glass 1.5,
    /// sapphire 1.77, diamond 2.42. Drives both the refraction bend and the
    /// Fresnel reflectance (F0 = ((ior−1)/(ior+1))²).
    public var ior: Float
    /// Surface roughness, 0 = optically polished, 1 = heavily frosted. Jitters
    /// the refracted + reflected rays in a cone; denoised by the renderer's TAA.
    public var roughness: Float
    /// Beer–Lambert absorption strength, per metre of path INSIDE the glass.
    /// 0 = no volumetric absorption (only the `tint` on transmission). Higher
    /// values make thick/coloured glass read denser and darker at its core.
    public var density: Float
    /// Scale on the Fresnel reflection term (1 = physical). Lower it to calm a
    /// busy reflection; raise it for a more mirror-like rim.
    public var reflectivity: Float
    /// Chromatic dispersion: 0 = off, larger spreads the per-wavelength IOR so
    /// the glass throws a prism rainbow. Costs 3× the refraction trace when > 0.
    public var dispersion: Float

    public init(tint: SIMD3<Float> = SIMD3(repeating: 1),
                ior: Float = 1.5,
                roughness: Float = 0,
                density: Float = 0,
                reflectivity: Float = 1,
                dispersion: Float = 0) {
        self.tint = tint
        self.ior = ior
        self.roughness = roughness
        self.density = density
        self.reflectivity = reflectivity
        self.dispersion = dispersion
    }

    /// Optically-clear window glass.
    public static let clearGlass = IlluminatoramaGlassMaterial(ior: 1.5)
    /// Water (IOR 1.33), faint blue-green tint.
    public static let water = IlluminatoramaGlassMaterial(
        tint: SIMD3(0.86, 0.95, 0.97), ior: 1.333, density: 0.6)
    /// Sapphire (IOR 1.77), cool blue absorption.
    public static let sapphire = IlluminatoramaGlassMaterial(
        tint: SIMD3(0.45, 0.62, 0.95), ior: 1.77, density: 2.0)
    /// Diamond (IOR 2.42) with strong dispersion (fire).
    public static let diamond = IlluminatoramaGlassMaterial(
        ior: 2.42, dispersion: 1.0)
}

/// One glass instance: a transform plus an `IlluminatoramaGlassMaterial`, packed
/// for the GPU. This is the single glass currency — the raster pass reads it for
/// the rasterised front surface's material, and the TLAS carries it so a
/// refraction/reflection ray that lands on glass knows its IOR/tint. Field
/// layout mirrors `GlassInstance` in IlluminatoramaGlassRT.metal (stride 176).
public struct IlluminatoramaGlassInstance {
    public var modelMatrix: simd_float4x4
    public var normalMatrix: simd_float4x4
    public var tintIor: SIMD4<Float>        // xyz = tint, w = ior
    public var rdrf: SIMD4<Float>           // x = roughness, y = density, z = reflectivity, w = reserved
    public var dispersionPad: SIMD4<Float>  // x = dispersion, yzw reserved

    public init(modelMatrix: simd_float4x4, material: IlluminatoramaGlassMaterial) {
        self.modelMatrix = modelMatrix
        self.normalMatrix = IlluminatoramaInstance.normalMatrix(from: modelMatrix)
        self.tintIor = SIMD4(material.tint, max(1, material.ior))
        self.rdrf = SIMD4(max(0, material.roughness), max(0, material.density),
                          max(0, material.reflectivity), 0)
        self.dispersionPad = SIMD4(max(0, material.dispersion), 0, 0, 0)
    }

    public mutating func setTransform(_ m: simd_float4x4) {
        self.modelMatrix = m
        self.normalMatrix = IlluminatoramaInstance.normalMatrix(from: m)
    }

    /// IOR readback (used by the renderer when packing the TLAS glass-data buffer).
    public var ior: Float { tintIor.w }

    /// Convenience initialiser from loose material knobs — mirrors the old
    /// repurposed-`IlluminatoramaInstance` glass encoding (tint = albedo,
    /// reflectivity = metallic, roughness, …) plus an explicit IOR, so a scene
    /// migrating from the legacy screen-space glass needs a one-line change.
    public init(modelMatrix: simd_float4x4, tint: SIMD3<Float>, ior: Float,
                roughness: Float = 0, reflectivity: Float = 1,
                density: Float = 0, dispersion: Float = 0) {
        self.init(modelMatrix: modelMatrix,
                  material: IlluminatoramaGlassMaterial(
                    tint: tint, ior: ior, roughness: roughness,
                    density: density, reflectivity: reflectivity,
                    dispersion: dispersion))
    }
}

/// GPU material for a glass TLAS hit, indexed by `instance_id - glassInstanceBase`
/// in the bounce loop. Mirror of `RTGlassData` in IlluminatoramaGlassRT.metal
/// (stride 48).
struct IlluminatoramaRTGlassData {
    var tintIor: SIMD4<Float>
    var rdrf: SIMD4<Float>
    var dispersionPad: SIMD4<Float>
}

/// Per-frame uniforms for the RT glass fragment shader. Mirror of
/// `GlassRTUniforms` in IlluminatoramaGlassRT.metal.
struct IlluminatoramaGlassRTUniforms {
    var cameraWorldPos: SIMD3<Float> = .zero; var rayTMin: Float = 0.004
    var sunDir: SIMD3<Float> = SIMD3(0, 1, 0); var sunSoftnessRad: Float = 0.01
    var sunColor: SIMD3<Float> = SIMD3(repeating: 1); var reflStrength: Float = 1
    var skyAmbient: SIMD3<Float> = .zero; var skyIntensity: Float = 1
    var glassInstanceBase: UInt32 = 0
    var maxBounces: UInt32 = 6
    var shadowRays: UInt32 = 1
    var frameSeed: UInt32 = 0
    var surfCacheEnabled: UInt32 = 0
    var surfTriCount: UInt32 = 0
    var surfAtlasW: UInt32 = 0; var surfAtlasH: UInt32 = 0
    var dispersionEnabled: UInt32 = 0
    /// Cheap glass mode for the fallback shader: 0 = plain Fresnel+sky (unchanged
    /// for every other scene), 1 = synthetic tinted+glint, 2 = screen-space refraction.
    var cheapGlassMode: UInt32 = 0
    /// Viewport size in pixels (mode 2 + SS parity map `clipPos.xy` → backdrop UV).
    var viewW: Float = 0; var viewH: Float = 0
    /// Screen-space transmission parity on the RT path. 0 = OFF ⇒ exactly the previous
    /// behaviour. When on, the transmitted lobe of a pane is taken from the pre-glass
    /// composite at near-normal incidence (where a thin pane's ray displacement is
    /// sub-pixel) instead of being re-derived by `shadeSecondarySurface`, so the world
    /// through a window is the same pixels as the world beside it. Fades back to the
    /// traced path over cosθ 0.95 → 0.80. Field order must match `GlassRTUniforms` in
    /// IlluminatoramaGlassRT.metal — it sits between `viewH` and `time`.
    var ssTransmissionEnabled: UInt32 = 0
    // ── Thin-film iridescence (soap bubbles — Bubble Lab) ────────────
    /// Animation clock (seconds) for the swirling film-thickness field. Driven
    /// from `renderer.time`.
    var time: Float = 0
    /// Master strength of the thin-film interference coat added in cheap mode 2.
    /// 0 = OFF → an EXACT no-op (every other cheap-glass scene is unaffected).
    var thinFilmStrength: Float = 0
    /// Base soap-film thickness in nanometres at the bubble's equator (the
    /// interference order). ~100–600 nm is the visible first/second-order range.
    var filmThicknessNm: Float = 320
    /// Refractive index of the soap film itself (water+surfactant ≈ 1.33).
    var filmIOR: Float = 1.33
    // ── Oscillation-mode surface undulation (soap bubbles) ───────────
    /// Radial vertex-displacement amplitude (object-space units, sphere r=1).
    /// 0 = OFF → the glass VS skips displacement (every other scene unaffected).
    var wobbleAmp: Float = 0
    /// Global rate multiplier for the beating oscillation modes.
    var wobbleFreq: Float = 1
    // ── Through-glass parity with the deferred pass ──────────────────
    // Everything below defaults to "off / neutral", so a host that sets none of
    // it renders exactly as before.
    /// Number of local point / spot lights bound at fragment buffers 14 / 15. A
    /// re-shaded opaque hit accumulates them with the same falloff + cone math the
    /// deferred kernel uses (no shadow rays — cheap), so a lamp-lit room seen
    /// through a window isn't black.
    var pointLightCount: UInt32 = 0
    var spotLightCount: UInt32 = 0
    /// Interior day-light separation, mirroring `FrameUniforms.interiorMask` /
    /// `interiorIBL*` / `interiorAmbient`. A hit whose instance layer intersects
    /// `interiorMask` gets its sky-IBL scaled by mix(side, up, saturate(N.y)) and
    /// its ambient supplement scaled by `interiorAmbient` — without this the glass
    /// pass fills interiors at raw exterior strength and lands several times short.
    var interiorMask: UInt32 = 0
    var interiorIBLUp: Float = 1
    var interiorIBLSide: Float = 1
    var interiorAmbient: Float = 1
    /// Whether `objUV` (fragment buffer 12) + the albedo atlas (fragment texture 4)
    /// are bound and safe to sample. 0 ⇒ fall back to the per-instance mean albedo.
    var albedoAtlasEnabled: UInt32 = 0
    /// Bound of `objUV` in float2 entries — a hit past it falls back (guards a mesh
    /// whose UVs weren't CPU-readable).
    var objUVCount: UInt32 = 0
    /// Analytic night sky, mirroring `FrameUniforms.nightSkyParams` / `nightMoonDir` /
    /// `nightSunDir`. The sky dome is baked WITHOUT celestials (they blur at dome
    /// resolution), so a refracted or reflected ray that samples only the dome renders
    /// the night sky as flat black — a window at night showed an empty pane while the
    /// sky beside it carried the full star field. All-zero ⇒ exact no-op.
    var nightSkyParams: SIMD4<Float> = .zero
    var nightMoonDir: SIMD4<Float> = SIMD4(0, 1, 0, 0)
    var nightSunDir: SIMD4<Float> = SIMD4(0, -1, 0, 0)
    /// Angular size of one output pixel (radians) — sets the star point-spread and the
    /// moon's limb anti-aliasing, exactly as the primary sky branch computes it.
    var nightPixAngle: Float = 0
    var _nightPad0: Float = 0; var _nightPad1: Float = 0; var _nightPad2: Float = 0
    // Interior irradiance bands — mirror of the Metal GlassRTUniforms tail.
    var interiorIrrUp: SIMD4<Float> = .zero
    var interiorIrrSide: SIMD4<Float> = .zero
    var interiorIrrDown: SIMD4<Float> = .zero
    // Per-room band level (S3.5 Stage E) — so a room seen through a pane is scaled
    // like the room beside it. Same packing as the frame twin.
    var interiorRoomGain0: SIMD4<Float> = .one
    var interiorRoomGain1: SIMD4<Float> = .one
    var interiorRoomGain2: SIMD4<Float> = .one
    var interiorRoomGain3: SIMD4<Float> = .one
    var interiorRoomGain4: SIMD4<Float> = .one
    var interiorRoomGain5: SIMD4<Float> = .one
    var interiorRoomGain6: SIMD4<Float> = .one
    var interiorRoomGain7: SIMD4<Float> = .one
    var interiorRoomGainMeta: SIMD4<Float> = .zero
    /// Physical night sky through the glass — mirror of the Metal GlassRTUniforms tail
    /// (and of the frame's `nightSkyExtra` / `nightSkyExtra2` / `nightCelestial`).
    var nightSkyExtra: SIMD4<Float> = .zero
    var nightSkyExtra2: SIMD4<Float> = .zero
    var nightCelestial: SIMD4<Float> = .zero
    /// Defocus trace (`rtGlassDefocusTraceFactor`) — x = mode (0 OFF ⇒ exactly the previous
    /// behaviour, 1 = the reduced-resolution trace pass, 2 = the full-resolution pass that takes
    /// its result where the lens blurs the pane), y = factor k, z = the full-res pass's |CoC|
    /// gate (px), w = the low-res pass's (lower) gate. Mirror of the Metal `defocus`.
    var defocus: SIMD4<Float> = .zero
    /// The DOF's lens for the gate: x = focus distance (m), y = CoC diameter coefficient (px),
    /// z = max CoC radius (px), w = diffraction floor (px).
    var defocusLens: SIMD4<Float> = .zero
    /// xyz = the camera's unit view axis (depth is measured along it, as the DOF measures it).
    var cameraForward: SIMD4<Float> = SIMD4(0, 0, -1, 0)

    mutating func setInteriorRoomGains(_ gains: [Float], enabled: Bool) {
        let p = InteriorRoomGains.pack(gains, enabled: enabled)
        interiorRoomGain0 = p.v[0]; interiorRoomGain1 = p.v[1]
        interiorRoomGain2 = p.v[2]; interiorRoomGain3 = p.v[3]
        interiorRoomGain4 = p.v[4]; interiorRoomGain5 = p.v[5]
        interiorRoomGain6 = p.v[6]; interiorRoomGain7 = p.v[7]
        interiorRoomGainMeta = p.meta
    }
}

/// Per-frame uniforms for the glass caustics kernels. Mirror of `CausticUniforms`
/// in IlluminatoramaCaustics.metal (stride 272 — each `SIMD3 + Float` pair is 32 B,
/// since both Swift `SIMD3<Float>` and Metal `float3` are 16-aligned/16-size).
struct IlluminatoramaCausticUniforms {
    var invViewProjection: simd_float4x4 = matrix_identity_float4x4
    var cameraWorldPos: SIMD3<Float> = .zero; var rayTMin: Float = 0.004
    var sunDir: SIMD3<Float> = SIMD3(0, 1, 0); var photonEnergy: Float = 1
    var sunColor: SIMD3<Float> = SIMD3(repeating: 1); var decay: Float = 0.9
    var aabbMin: SIMD3<Float> = .zero; var strength: Float = 1
    var aabbMax: SIMD3<Float> = .zero; var _pad0: Float = 0
    var width: UInt32 = 0; var height: UInt32 = 0
    var glassInstanceBase: UInt32 = 0
    var photonCount: UInt32 = 0
    var surfTriCount: UInt32 = 0; var surfAtlasW: UInt32 = 0; var surfAtlasH: UInt32 = 0; var frameSeed: UInt32 = 0
    var maxGlassBounces: UInt32 = 6; var glassDiscCount: UInt32 = 0; var _pad2: UInt32 = 0; var _pad3: UInt32 = 0
}


extension IlluminatoramaGlassInstance: Equatable {}
