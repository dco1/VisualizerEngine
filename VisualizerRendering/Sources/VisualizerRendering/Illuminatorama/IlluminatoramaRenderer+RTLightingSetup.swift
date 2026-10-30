import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    /// **Daydream DH-1320 — which of a host's room-fill STAND-INS the one-bounce traced GI owns.**
    /// A host that lights interiors without transport (interior irradiance bands, an ambient
    /// supplement, window apertures as area lights) is describing the light a traced bounce
    /// brings back: the bounce's misses ARE the sky through the window and its hits are lit
    /// surfaces. With the traced GI running, each named stand-in is handed to it the way
    /// `rtGIReplacesDiffuseSky` hands over the outdoor-cube sky (the lighting pass writes the share
    /// to the hand-off target, the traced pass subtracts it where its rays ran) — the one-bounce
    /// sibling of `rtPathTracedGI`'s whole-indirect hand-off, one bit per stand-in so each can be
    /// measured alone. Needs `rtGIReplacesDiffuseSky`; ignored on the path lane (which already
    /// owns all of it). Empty (default) ⇒ byte-identical.
    public struct GIRoomFillHandoff: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }
        /// The whole diffuse IBL — interior bands, room gains, interior scalars (frame bit 16).
        public static let bands = GIRoomFillHandoff(rawValue: 1)
        /// The ambient supplement (frame bit 32).
        public static let ambient = GIRoomFillHandoff(rawValue: 2)
        /// Area lights flagged `isAperture` — window portals (frame bit 64).
        public static let apertures = GIRoomFillHandoff(rawValue: 4)
        public static let all: GIRoomFillHandoff = [.bands, .ambient, .apertures]
    }

    // ── Who computes the sun's DIRECT term (C1) ──────────────────────
    //
    // THE BUG THIS EXISTS TO MAKE IMPOSSIBLE: the RT lighting kernels end with
    // `outHDR += direct + indirect + reflection`, and `direct` was computed
    // unconditionally whenever `shadowRays > 0` (default 4). The frame graph never
    // suppressed the DEFERRED lighting pass — it only chose *which* RT pass ran —
    // so a host that flipped `rtEnabled` on a normally-lit scene got the sun
    // TWICE, on top of everything else. Every Visualizer host avoided that by
    // CONVENTION (each one zeroes `directionalLightColor` + `shadowsEnabled`
    // itself, and `IlluminatoramaSceneExtractor` does it for the descriptor path);
    // a convention is exactly what a new host forgets. Daydream Home forgot it, and
    // "RT GI washes interiors out" was mostly this.
    //
    // Ownership is now a stated property of the renderer, and the RT kernels are
    // told the answer in a uniform (`directSunEnabled`) instead of assuming it.
    public enum RTSunOwnership: Sendable {
        /// The DEFERRED lighting pass owns the sun; the RT pass contributes only
        /// GI + reflections. In this mode `rtGIStrength = 0` + `rtReflStrength = 0`
        /// makes an RT-on frame BIT-IDENTICAL to an RT-off frame — the property
        /// `IlluminatoramaRTSunOwnershipTests` gates, and the cheapest possible
        /// alarm on this whole class of bug.
        ///
        /// This is the right mode when the deferred BRDF is richer than the RT
        /// one (it is: cascades, clearcoat, sheen, anisotropy, transmission) and
        /// RT is wanted for transport, not for the sun.
        case deferred
        /// The RT pass owns the sun — soft shadow rays sampled across the sun's
        /// angular disc, i.e. a true penumbra the cascades can't produce. The
        /// renderer FORCES the deferred directional light + cascaded shadows off
        /// for the frame, so choosing this can't leave the sun double-counted.
        case rayTraced
        /// Resolve from observable state: `rayTraced` iff the host has already
        /// zeroed `directionalLightColor` (the hand-over every RT host in
        /// Visualizer performs), else `deferred`. This is the default because it
        /// reproduces EXACTLY what every existing host does today — the scenes
        /// that hand the sun over keep RT's sun, and a host that never did (i.e.
        /// was double-counting) stops.
        case automatic
    }

    // ── TLAS instance masks (C3) ─────────────────────────────────────
    //
    // Three states, not two. The first two are old:
    //   0x01  traced — an ordinary opaque/curve instance.
    //   0x02  AAA glass — a pane, so it never casts a solid shadow.
    //   0x00  excluded — present in the TLAS layout, never intersected. The RT
    //         representation lives elsewhere (a superquadric's proxy, a curve twin).
    // and the third is new:
    //   0x04  INVISIBLE OCCLUDER — real to LIGHT, never drawn. Daydream Home's
    //         lighting-only `ceilshadow.*` ceilings are the case: a roofless
    //         dollhouse needs a ceiling that stops the sun (it is already a
    //         shadow-map occluder) but must not appear in the picture. With only
    //         the binary states available, the bridge had to pick "excluded", and
    //         then every GI ray leaving an interior surface through the open top
    //         missed everything and returned FULL SKY — the patio flood
    //         `+InteriorLight` exists to prevent.
    //
    // The split is by ray PURPOSE, not by pass: transport rays (shadow, GI) trace
    // `0x01|0x04`; camera-visible rays (glass primary/refraction, glossy
    // reflections) trace without 0x04, so an undrawn slab is never shown.
    static let rtMaskTraced: UInt32 = 0x01

    static let rtMaskGlass: UInt32 = 0x02

    static let rtMaskInvisibleOccluder: UInt32 = 0x04

    /// Shadow + GI ray mask: opaque/curve + invisible occluders.
    static let rtTransportRayMask: UInt32 = rtMaskTraced | rtMaskInvisibleOccluder

    /// Restart the RT GI accumulation on the next frame (its history reprojects against nothing),
    /// WITHOUT `resetTemporalHistory`'s exposure re-seed. For a host whose still discards warm-up
    /// frames: in `rtGIStaticAccumulation` mode nothing forgets them otherwise (DH-0887).
    /// `keepingRadianceCache` — restart the GI running mean but keep the path lane's radiance cache (a
    /// still that has just BUILT it in held-out frames, DH-1344).
    public func resetRTGIAccumulation(keepingRadianceCache: Bool = false) {
        let rcClear = rts.rcNeedsClear
        rtGINeedsFirstFrame = true
        if keepingRadianceCache { rts.rcNeedsClear = rcClear }
    }

    /// Diagnostic A/B override: `VIZ_ILLUMI_SVGF=1`/`0` forces the SVGF cascade
    /// on/off for every Illuminatorama renderer in the process, so a headless
    /// render can exercise the à-trous path (issue #65) without touching the
    /// default. `nil` when unset. Read once. `effectiveSVGFEnabled` consults it.
    static let svgfEnvOverride: Bool? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF"] else { return nil }
        return v == "1" || v.lowercased() == "true"
    }()

    /// Diagnostic A/B override for `svgfLumFloor` (`VIZ_ILLUMI_SVGF_LUMFLOOR`).
    static let svgfLumFloorEnv: Float? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF_LUMFLOOR"],
              let f = Float(v) else { return nil }
        return f
    }()

    /// Diagnostic A/B override for cascade level count (`VIZ_ILLUMI_SVGF_LEVELS`).
    static let svgfLevelsEnv: Int? = {
        guard let v = ProcessInfo.processInfo.environment["VIZ_ILLUMI_SVGF_LEVELS"],
              let i = Int(v) else { return nil }
        return i
    }()

    // ── Hardware ray tracing state ───────────────────────────────────
    /// Mirror of `RTUniforms` in IlluminatoramaRT.metal.
    struct RTUniforms {
        var invViewProjection: simd_float4x4
        var cameraWorldPos: SIMD3<Float>; var _pad0: Float = 0
        var sunDir: SIMD3<Float>;         var sunSoftnessRad: Float
        var sunColor: SIMD3<Float>;       var giStrength: Float
        var skyAmbient: SIMD3<Float>;     var specStrength: Float
        var width: UInt32; var height: UInt32
        var shadowRays: UInt32; var giRays: UInt32
        var frameSeed: UInt32; var rayTMin: Float; var maxGIDist: Float; var triangleCount: UInt32
        var reflStrength: Float; var reflMaxDist: Float; var reflRoughnessCutoff: Float
        var reflRays: UInt32; var reflEnabled: UInt32
        // Surface cache: read cached radiance at GI/reflection hits when enabled.
        var surfCacheEnabled: UInt32; var surfTileSize: UInt32; var surfTilesPerRow: UInt32
        var surfAtlasW: UInt32; var surfAtlasH: UInt32
        var _padEmitters: UInt32 = 0
        var leafTransmission: Float = 0      // mirrors _padRT0 in the Metal struct
        // Curve primitives (#60 item 7, incr. 2): 1 ⇒ soup AS holds a curve
        // geometry descriptor; mirrors `curvesEnabled` (was `_padRT1`) in Metal.
        var curvesEnabled: UInt32 = 0
        /// C1 — 0 ⇒ the deferred pass already shaded the sun here (mirrors
        /// `directSunEnabled`, was `_padRT2`). See `RTSunOwnership`.
        var directSunEnabled: UInt32 = 0
        /// DH-0872 — scotopic desaturation for the GI sky-miss sample only. Mirrors
        /// `scotopicDesaturation` (the renderer's own property, also fed to the
        /// tonemap pass). 0 (day) ⇒ byte-identical.
        var scotopicDesaturation: Float = 0
    }

    /// Mirror of `RTDenoiseUniforms` in Illuminatorama.metal (stride 32, 16-aligned).
    struct RTDenoiseUniforms {
        var width: UInt32; var height: UInt32; var enabled: UInt32; var radius: UInt32
        var kDepth: Float; var kNorm: Float; var _pad0: Float = 0; var _pad1: Float = 0
    }

    /// Mirror of `RTGITemporalUniforms` in Illuminatorama.metal (stride 32, 16-aligned).
    struct RTGITemporalUniforms {
        var width: UInt32; var height: UInt32; var enabled: UInt32; var isFirstFrame: UInt32
        var blend: Float; var gammaClamp: Float; var staticAccumulation: UInt32 = 0; var _pad1: Float = 0
    }

    // ── TLAS — instanced RT for animated extracted scenes ────────────
    // Per-mesh BLAS built once (object space); a TLAS of per-instance
    // transforms refit each frame. Replaces the room-only / soup primitive-AS
    // path for extracted overlay scenes so RT generalises to ANIMATION without
    // a per-frame vertex transform + geometry rebuild.
    /// Mirror of `RTInstanceData` in IlluminatoramaRTInstanced.metal.
    struct RTInstanceData {
        var nrm0: SIMD4<Float>; var nrm1: SIMD4<Float>; var nrm2: SIMD4<Float>
        var albedoTriBase: SIMD4<Float>   // xyz = albedo, w = objNormal base (as Float)
        /// xyz = the instance's SELF-LIT radiance (the emission scalar, intensity
        /// already folded in). A lamp shade, a TV panel and the night window glow are
        /// all lit by this term and by nothing else, so a secondary ray that omits it
        /// renders every one of them dim or absent — seen through a pane, the lamp
        /// lighting the room went missing. Read by `secondaryEmission` in
        /// IlluminatoramaSecondary.h. w: GLASS rows only — first corner-normal entry + 1 in
        /// `rtGlassCornerNormalBuffer` (0 = none); opaque rows keep it 0.
        var emissionPad: SIMD4<Float> = .zero
    }

    /// Mirror of `RTInstUniforms`.
    struct RTInstUniforms {
        var invViewProjection: simd_float4x4
        /// 1 ⇒ the traced GI replaces the deferred diffuse sky (`rtGIReplacesDiffuseSky`). Was `_pad0`.
        var cameraWorldPos: SIMD3<Float>; var giReplacesDiffuseSky: UInt32 = 0
        var sunDir: SIMD3<Float>; var sunSoftnessRad: Float
        var sunColor: SIMD3<Float>; var giStrength: Float
        var skyAmbient: SIMD3<Float>; var specStrength: Float
        var width: UInt32; var height: UInt32
        var shadowRays: UInt32; var giRays: UInt32
        var frameSeed: UInt32; var rayTMin: Float; var maxGIDist: Float
        /// DH-0896 — 1 ⇒ `specIBLTexture` holds this frame's deferred specular-IBL share and
        /// a reflection HIT replaces it. Was `_pad1` (same 4 bytes, stride unchanged).
        var reflReplacesIBL: UInt32 = 0
        var reflStrength: Float; var reflMaxDist: Float; var reflRoughnessCutoff: Float
        var reflRays: UInt32; var reflEnabled: UInt32
        // Surface cache (P1c) — mirror of the Metal RTInstUniforms tail.
        var surfCacheEnabled: UInt32 = 0
        var surfTileSize: UInt32 = 1; var surfTilesPerRow: UInt32 = 1
        var surfAtlasW: UInt32 = 1; var surfAtlasH: UInt32 = 1
        var surfTriCount: UInt32 = 0
        var debugSurfCacheGI: UInt32 = 0
        // Curve primitives (#60 item 7): TLAS instance ids ≥ curveInstanceBase
        // are curve sets (id − base = set index). 0 sets ⇒ fields unread.
        var curveInstanceBase: UInt32 = 0
        var curveSetCount: UInt32 = 0
        // Phase 5 / B0 — DebugTerm.surfaceCacheVariance isolation (was _padc0).
        var debugSurfCacheVar: UInt32 = 0
        // Phase 5 / A — residency feedback gate (was _padc1).
        var surfFeedbackEnabled: UInt32 = 0
        // ── Secondary-hit shading parity (shared with the AAA glass pass) ────
        // Feed `SecondaryShadeParams` in IlluminatoramaSecondary.h. Defaults are
        // off/neutral, so a host that wires none of them renders as before.
        var skyIntensity: Float = 1
        var interiorMask: UInt32 = 0
        var interiorIBLUp: Float = 1
        var interiorIBLSide: Float = 1
        var interiorAmbient: Float = 1
        var albedoAtlasEnabled: UInt32 = 0
        var objUVCount: UInt32 = 0
        var pointLightCount: UInt32 = 0
        var spotLightCount: UInt32 = 0
        // C1 — 0 ⇒ the deferred pass already shaded the sun here; this kernel must
        // not add it again (see `RTSunOwnership`). C3 — the shadow/GI ray mask,
        // 0x01 opaque | 0x04 invisible occluder.
        var directSunEnabled: UInt32 = 0
        var transportRayMask: UInt32 = 0x05
        // DH-0653 — hit/miss counter gate + chart overlay (were _padIrr0/_padIrr1).
        var surfStatsEnabled: UInt32 = 0; var debugSurfCacheCharts: UInt32 = 0
        /// DH-0872 — scotopic desaturation for the GI sky-miss sample only. Mirrors
        /// `scotopicDesaturation` (also fed to the tonemap pass). Repurposes
        /// `_padIrr2` — same 4 bytes. 0 (day) ⇒ byte-identical.
        var scotopicDesaturation: Float = 0
        // Interior irradiance bands — mirror of the Metal RTInstUniforms tail.
        var interiorIrrUp: SIMD4<Float> = .zero
        var interiorIrrSide: SIMD4<Float> = .zero
        var interiorIrrDown: SIMD4<Float> = .zero
        // Per-room band level (S3.5 Stage E) — so a room in a reflection is scaled
        // like the room seen directly. Same packing as the frame twin.
        var interiorRoomGain0: SIMD4<Float> = .one
        var interiorRoomGain1: SIMD4<Float> = .one
        var interiorRoomGain2: SIMD4<Float> = .one
        var interiorRoomGain3: SIMD4<Float> = .one
        var interiorRoomGain4: SIMD4<Float> = .one
        var interiorRoomGain5: SIMD4<Float> = .one
        var interiorRoomGain6: SIMD4<Float> = .one
        var interiorRoomGain7: SIMD4<Float> = .one
        var interiorRoomGainMeta: SIMD4<Float> = .zero
        // DH-0887 — the GI directions' contiguous progressive index (Metal twin appended last).
        var giProgressiveIndex: UInt32 = 0
        /// DH-0715 — the traced-origin snap's reach (was `padProg0`, same 4 bytes).
        var surfaceSnapULPs: Float = 0
        /// DH-0989 — path bounces (0 ⇒ one-bounce GI) and the per-path luminance cap.
        var pathBounces: UInt32 = 0
        var pathClamp: Float = 0
        /// DH-1011 — `kPathFlag*` (Metal twin appended last; three pads keep the 16-byte tail).
        var pathFlags: UInt32 = 0
        /// DH-1293 — the lamps' near field at path vertices (were `padPath0/padPath1`, same bytes).
        var pathLampMinDist: Float = 0, pathLampSourceRadius: Float = 0
        /// DH-1344 — the path lane's radiance-cache cell (m); 0 ⇒ off (was `padPath2`).
        var radCacheCell: Float = 0
        /// DH-1308 — the glossy vertices' roughness floor (Metal twin appended last; three pads keep the
        /// 16-byte tail).
        var pathGlossyMinRoughness: Float = 0
        var pathGlossyLampSize: Float = 1, pathGlossyLampMinRoughness: Float = 0
        /// DH-1300 — the raster instances bound as `layerInstances` (was `padGlossy2`, same 4 bytes).
        var layerInstanceCount: UInt32 = 0
        /// DH-0539 — 1 ⇒ the reflection goes to the specular denoise chain, not into the HDR (Metal twin appended last).
        var specDenoise: UInt32 = 0
        /// DH-0952 — the snap probe's gate (Metal twin appended last).
        var surfaceSnapMinUpY: Float = -2
        var surfaceSnapClearance: Float = 0
        /// DH-0952 — the reflection's sampling (`rtReflSampling`; Metal twin appended last). 0 ⇒ byte-identical.
        var reflSampling: UInt32 = 0

        mutating func setInteriorRoomGains(_ gains: [Float], enabled: Bool) {
            let p = InteriorRoomGains.pack(gains, enabled: enabled)
            interiorRoomGain0 = p.v[0]; interiorRoomGain1 = p.v[1]
            interiorRoomGain2 = p.v[2]; interiorRoomGain3 = p.v[3]
            interiorRoomGain4 = p.v[4]; interiorRoomGain5 = p.v[5]
            interiorRoomGain6 = p.v[6]; interiorRoomGain7 = p.v[7]
            interiorRoomGainMeta = p.meta
        }
    }

    /// **The ONE place an instance becomes `RTInstanceData`.** The normal matrix's
    /// column `.w` lanes are dead weight for RT (every consumer builds a float3x3
    /// from `.xyz`), so they carry the two facts a re-shaded secondary hit needs to
    /// shade like the deferred pass does:
    ///   nrm0.w = albedo atlas slice PLUS ONE — so the memset-0 default (curve
    ///            instances, unslotted instances) decodes to −1 = "untextured, use
    ///            the mean albedo" rather than slice 0
    ///   nrm1.w = the instance's light-layer bits (bit-cast; drives light masking +
    ///            the interior day-light separation)
    /// plus `emissionPad`, the self-lit radiance.
    ///
    /// This exists because the per-frame REFIT used to write `dataPtr[i].nrm0 =
    /// nm.columns.0` directly — which wipes both packed lanes, so from the second
    /// frame after a rebuild every secondary hit silently lost its texture and its
    /// light masking, and (once emission joined the struct) would have kept a stale
    /// glow while the real one ramped with nightfall. Rebuild and refit must produce
    /// byte-identical rows; the only way to guarantee that is to have one function.
    ///
    /// `normalBase` is the instance's offset into the concatenated `objNormal` /
    /// `objUV`, which is topology (fixed at rebuild), so the refit passes back the
    /// value already in the row.
    static func rtInstanceRow(_ inst: IlluminatoramaInstance, normalBase: Float) -> RTInstanceData {
        var c0 = inst.normalMatrix.columns.0, c1 = inst.normalMatrix.columns.1
        var c2 = inst.normalMatrix.columns.2
        c0.w = Float(inst.albedoTextureSlice + 1)
        c1.w = Float(bitPattern: inst.layer)
        // DH-1308 — the layered stack a path vertex samples (`secondaryMaterial` / `secondaryCoat`):
        // the base's metalness + roughness in `emissionPad.w`, the clearcoat's strength + roughness in
        // `nrm2.w` (0 strength ⇒ no coat). Read only under `kPathFlagGlossyVertices`.
        c2.w = inst.clearcoat > 0 ? rtPackMaterialLane(inst.clearcoat, inst.clearcoatRoughness) : 0
        // Signack S5 — a coat-free row carries its thin-sheet transmission ratio as 1 + t (≥ 1, so the
        // coat decode still reads "none"); `thinSheetRatioFromLane` (IlluminatoramaLeafOptics.h).
        if inst.clearcoat <= 0, inst.thinTransmission > 0 { c2.w = 1 + min(inst.thinTransmission, 1) }
        return RTInstanceData(nrm0: c0, nrm1: c1, nrm2: c2,
                              albedoTriBase: SIMD4(inst.albedo, normalBase),
                              emissionPad: SIMD4(inst.emission, rtPackBaseLane(inst)))
    }

    /// DH-1301 — the base layer's lane: metalness + roughness (`rtPackMaterialLane`), or, for a PURE
    /// dielectric with a diffuse roughness, an integer part 1001…2001 = 1001 + round(r·1000) — decoded
    /// as metalness 0 by `secondaryMaterial` and as the roughness by `secondaryDiffuseRoughness`.
    /// Same rule as the G-buffer's (`illumiPackMetallicDiffuseRoughness`): only `metallic == 0` carries it.
    nonisolated static func rtPackBaseLane(_ inst: IlluminatoramaInstance) -> Float {
        let dr = inst.diffuseRoughness
        guard dr > 0, inst.metallic <= 0 else { return rtPackMaterialLane(inst.metallic, inst.roughness) }
        let aq = 1001 + (min(dr, 1) * 1000).rounded()
        return -(1 + 2 * aq + min(max(inst.roughness, 0), 0.999))
    }

    /// DH-1308 — two [0, 1] material scalars in ONE float lane, `−(1 + 2·round(a·1000) + min(b, 0.999))`:
    /// always ≤ −1, so a glass row's corner-normal base (≥ 0) and an unpacked row's 0 both decode to
    /// "absent" (`secondaryUnpackMaterialLane`, IlluminatoramaSecondary.h). `a` keeps 1/1000, `b` ~1e-4.
    nonisolated static func rtPackMaterialLane(_ a: Float, _ b: Float) -> Float {
        let aq = (min(max(a, 0), 1) * 1000).rounded()
        return -(1 + 2 * aq + min(max(b, 0), 0.999))
    }

    /// Glass sibling of `rtInstanceRow`. A dielectric boundary is never a shaded hit
    /// surface — the bounce loop reads its material from `rtGlassDataBuffer` — so the
    /// packed lanes are pinned to explicit "untextured / every layer / no emission"
    /// values instead of being left to whatever the normal matrix's padding holds.
    static func rtGlassInstanceRow(_ inst: IlluminatoramaGlassInstance, normalBase: Float,
                                           cornerBase: Float = 0) -> RTInstanceData {
        var c0 = inst.normalMatrix.columns.0, c1 = inst.normalMatrix.columns.1
        c0.w = 0                                    // untextured
        c1.w = Float(bitPattern: UInt32.max)        // unmasked by any light layer
        return RTInstanceData(nrm0: c0, nrm1: c1, nrm2: inst.normalMatrix.columns.2,
                              albedoTriBase: SIMD4(inst.tintIor.x, inst.tintIor.y, inst.tintIor.z, normalBase),
                              // w = first corner-normal entry + 1 (0 = none); a glass boundary
                              // has no emission, so the xyz stay zero.
                              emissionPad: SIMD4(0, 0, 0, cornerBase))
    }

    /// Everything the RT path caches per registered mesh: its BLAS, its object-space
    /// per-triangle normals, and — for a deforming (GPU-fed) mesh — the descriptor and
    /// scratch its per-frame in-place refit needs. ONE entry, not four parallel maps, so a
    /// mesh's RT state can only ever be added, found and dropped as a unit.
    ///
    /// **The entry OWNS its mesh, and that is load-bearing.** The cache is keyed by
    /// `ObjectIdentifier(mesh)` — a raw ADDRESS. A document editor rebuilds its whole scene
    /// on every edit (`HouseRenderBridge.load()` drops the previous scene's mesh handles and
    /// registers new ones), and `IlluminatoramaMeshHandle.deinit` evicts the renderer entry
    /// ASYNCHRONOUSLY (`Task { @MainActor … removeMesh }`). So without a strong reference the
    /// key dangles the moment that task runs, and the very next `IlluminatoramaMesh` the
    /// allocator hands out can land on the recycled address — at which point the cache
    /// "hits" and gives brand-new geometry an acceleration structure built from a mesh it
    /// has never seen, still referencing the dead mesh's released vertex/index buffers.
    /// Holding the mesh pins the address for exactly as long as the entry exists;
    /// `pruneRTMeshCache()` is the only thing that lets go.
    struct RTMeshEntry {
        /// STRONG on purpose — see above. Never make this `weak`/`unowned`.
        let mesh: IlluminatoramaMesh
        let blas: MTLAccelerationStructure
        let normals: [SIMD4<Float>]
        /// Per-triangle mesh UVs (3 per triangle), in the SAME triangle order as
        /// `normals` — so one `normalBase + primitive_id` resolves both a hit's
        /// world normal and its texture coordinates. Feeds `rtObjUVBuffer`.
        let uvs: [SIMD2<Float>]
        // Phase B — deforming-mesh BLAS refit. A DynamicMesh / GPU-fed mesh keeps a
        // STABLE topology (vertex+index count fixed; only positions move each frame),
        // so its BLAS is built once with `.refit` usage and re-fit per frame from the
        // live (repacked) vertex buffer — instead of being frozen at build-time verts,
        // which made the RT trace see stale geometry (shadows/GI lit for the old pose).
        // The refit is encoded on the FRAME command buffer AFTER `encodeGPURepacks`, so
        // Metal sequences repack-write → refit-read; the BLAS references `mesh.vertexBuffer`,
        // the same buffer the repack writes in place. `nil` for static meshes.
        var refit: (desc: MTLPrimitiveAccelerationStructureDescriptor, scratch: MTLBuffer)?
        /// Per-triangle-CORNER shading normals (3 per triangle, `normals` order) — filled only
        /// for a mesh some glass instance uses, and read only by the glass bounce loop, where a
        /// dielectric boundary's normal is its refraction (`objectCornerNormals`). Empty for
        /// every opaque-only mesh, so the scene pays nothing for it. Feeds
        /// `rtGlassCornerNormalBuffer`.
        var cornerNormals: [SIMD4<Float>] = []
    }

    /// Drop every cached entry whose mesh is no longer registered, releasing its BLAS, its
    /// normals and — the point of the whole exercise — the mesh itself, so the address it
    /// occupied becomes reusable only once nothing can look it up any more.
    ///
    /// Called from `rebuildRTAccel()`, i.e. exactly when the TLAS topology changed, which is
    /// the only moment the set of live meshes can have moved.
    func pruneRTMeshCache() {
        // Filtered unconditionally: equal counts do NOT imply equal sets (a scene can shed
        // one mesh and gain another in the same rebuild), and this runs on topology changes
        // only, never per frame.
        let live = Set(meshes.values.map(ObjectIdentifier.init))
        rts.rtMeshCache = rts.rtMeshCache.filter { live.contains($0.key) }
        rts.rtHasDeformingBLAS = rts.rtMeshCache.values.contains { $0.refit != nil }
    }

    /// Mirror of `RTCurveSetData` in IlluminatoramaRTInstanced.metal (112 B,
    /// SIMD4-aligned — no SIMD3 trap). m0..m3 = the set's object→world matrix
    /// columns; meta.x = the set's first segment index in the pooled buffers.
    struct RTCurveSetData {
        var m0: SIMD4<Float>; var m1: SIMD4<Float>; var m2: SIMD4<Float>; var m3: SIMD4<Float>
        var albedoRoughness: SIMD4<Float>
        var emissionPad: SIMD4<Float>
        var meta: SIMD4<UInt32>
    }

    /// Mirror of `CurveWindUniforms` in IlluminatoramaRT.metal (16 B).
    struct CurveWindUniforms {
        var time: Float; var windStrength: Float; var windHeading: Float; var pointCount: UInt32
    }

    /// Rebuild this many frames running ⇒ topology is thrashing ⇒ disable RT.
    /// Allows a few legitimate settle-frames at scene load before giving up.
    /// (Backup gate — the size cap below is the primary defence, because the
    /// thrash counter can't catch a SINGLE catastrophic rebuild that never
    /// returns.)
    static let rtRebuildThrashLimit: Int = 6

    /// Scenes past ANY of these caps are too heavy for per-frame RT in the live
    /// tick loop (a single rebuild — thousands of BLAS + CPU normal readbacks +
    /// `waitUntilCompleted` — would stall for seconds-to-minutes). Checked UP
    /// FRONT, before the first build, so a heavy scene auto-disables instead of
    /// hanging. The triangle cap is the real cost predictor (catches both
    /// many-small-mesh and few-huge-mesh scenes); the instance / mesh-group
    /// caps catch BLAS/TLAS build fan-out. Tuned for "light enough that a
    /// per-frame rebuild stays sub-frame" — the room / a box scene pass; Forest
    /// (thousands of leaf cards) does not.
    static let rtMaxInstancesForLiveRT: Int = 2048

    static let rtMaxMeshGroupsForLiveRT: Int = 128

    static let rtMaxTrianglesForLiveRT: Int = 150_000

    /// Triangle cap for the GLASS-ONLY TLAS (AAA glass opted in, but full-scene RT
    /// lighting is OFF). The strict 150k cap above is sized so a per-pixel RT
    /// lighting/GI pass stays sub-frame — but a glass-only TLAS is traced by ONLY
    /// the handful of glass fragments in frame, so the same opaque geometry is far
    /// cheaper to carry. An architectural interior (walls/floor/ceiling/furniture +
    /// a lawn's worth of ground triangles) routinely lands in the 150k–600k range;
    /// this higher ceiling lets its windows refract the real room without tripping
    /// the hang-guard. Still bounded so a pathological scene auto-disables rather
    /// than stalling the first BLAS build. Only consulted on the glass-only path.
    ///
    /// **2.5 M is measured; the old 1.2 M was not** (Daydream DH-1367, `DocumentRTCensusTests
    /// .testRTBuildCostVersusTriangleCount`: the real 4000 Sunset (north side) document — which
    /// sat at 1.247 M and lost every RT pass, its mirrors showing the sky — padded with distinct
    /// ~250 k-triangle meshes, live lane, 1600×1000, Debug):
    ///
    ///     unique tris │ 1.25 M  1.50 M  2.00 M  2.49 M  3.24 M
    ///     rebuild     │   75     114     193     170     240  ms
    ///     steady      │   92.1   90.3    85.9    94.6    90.5 ms
    ///     GPU alloc   │ 4879    4931    5034    5137    5291  MB
    ///
    /// The steady frame does not move (BVH traversal is logarithmic); the rebuild frame grows
    /// ~8 ms per 100 k triangles and memory ~0.2 KB per triangle. A rebuild runs only on a
    /// TOPOLOGY change, which a host's settled lane sees once per committed edit. Nowhere on
    /// the ladder is the seconds-to-minutes stall this guard exists to prevent; 2.5 M is ~2× the
    /// heaviest real document and keeps the worst rebuild near 200 ms.
    static let rtMaxTrianglesForGlassOnlyRT: Int = 2_500_000

    /// Mesh-GROUP cap for the GLASS-ONLY TLAS — the exact asymmetry the triangle caps
    /// above already encode (150 k live vs 1.2 M glass-only), applied to the other axis.
    ///
    /// **Why it had to move.** 128 is a *live-RT* number: there, a topology change can
    /// mean a full rebuild every frame in the tick loop, so the cap has to keep one
    /// rebuild well inside a frame. The glass-only path rebuilds only on a real topology
    /// change (the thrash guard below still catches per-frame churn) and hosts run it in a
    /// settled/photo lane, so it can carry far more. Meanwhile 128 is *small* for real
    /// content: an architectural document's group count is "how many distinct geometries",
    /// which scales with rooms and cabinetry — one group per wall edge, per room floor,
    /// per millwork sub-part. A 2-storey furnished house measures 209 and every pane in it
    /// silently dropped to the flat Fresnel fallback, scene-wide, on exactly the class of
    /// scene AAA glass exists for.
    ///
    /// **512 is measured, not guessed** (Daydream `DocumentRTCensusTests
    /// .testRTGlassBuildCostVersusMeshGroupCount`: a real document padded with N extra
    /// distinct 12-triangle groups, so group count is the only variable moving):
    ///
    ///     groups │  209    309    509    909   1509
    ///     rebuild│ 21.2   29.1   34.5   46.2   60.2  ms
    ///     steady │ 22.4   21.6   23.7   29.1   30.5  ms
    ///
    /// **There is no cliff on this axis** — the rebuild frame is linear at ≈0.030 ms per
    /// group out to 3× this cap, so the failure mode a cap must prevent (a first build that
    /// never returns) is a triangle/instance phenomenon, not a group one. At 512 the
    /// worst-case rebuild frame is ~34 ms: a hitch in a lane that has no vsync budget, and
    /// nowhere near the seconds-to-minutes stall the guard exists for. The cap stays finite
    /// so a pathological scene still auto-disables rather than degrading unboundedly.
    ///
    /// The live-RT path keeps 128, untouched — Visualizer's live-RT scenes are unaffected.
    static let rtMaxMeshGroupsForGlassOnlyRT: Int = 512

    /// Reset the RT auto-disable guard. Called on every scene attach (via
    /// `IlluminatoramaOverlay.setExtractedSceneRT`) so a freshly-shown scene
    /// gets a clean chance at RT regardless of what the previous scene did.
    public func resetRTGuard() {
        rts.rtGuard.reset()
        rts.rtConsecutiveRebuilds = 0
    }

    // ── Hardware RT geometry ──────────────────────────────────────────
    //
    // Supply the static world-space triangle soup the RT pass traces against
    // (the room's walls / floor / ceiling, with the window left as a hole so
    // sun rays pass through). Per-triangle albedo + geometric normal travel in
    // parallel arrays so the GI bounce can shade hit points. Builds the
    // primitive acceleration structure once; call again only if the geometry
    // changes. No-op on hardware without RT support.
    /// Diagnostic: append an AS-build command buffer's error to the sidecar file
    /// named by `VIZ_ILLUMI_CBERROR_PATH` (sandbox eats os_log). No-op without it.
    nonisolated static func recordASBuildError(_ cmd: MTLCommandBuffer, _ label: String) {
        guard cmd.status == .error || cmd.error != nil,
              let p = ProcessInfo.processInfo.environment["VIZ_ILLUMI_CBERROR_PATH"] else { return }
        let nsErr = cmd.error as NSError?
        let line = "\(label) status=\(cmd.status.rawValue) domain=\(nsErr?.domain ?? "?") code=\(nsErr?.code ?? 0) \(nsErr?.localizedDescription ?? "")\n"
        guard let data = line.data(using: .utf8) else { return }
        if let h = FileHandle(forWritingAtPath: p) { h.seekToEndOfFile(); h.write(data); try? h.close() }
        else { try? data.write(to: URL(fileURLWithPath: p)) }
    }

    public func setRTGeometry(positions: [SIMD3<Float>],
                              indices: [UInt32],
                              triangleAlbedo: [SIMD3<Float>],
                              triangleNormal: [SIMD3<Float>]) {
        guard rtSupported, rts.rtPipeline != nil else { return }
        guard !positions.isEmpty, indices.count % 3 == 0, !indices.isEmpty else { return }
        let triCount = indices.count / 3
        guard triangleAlbedo.count == triCount, triangleNormal.count == triCount else {
            Self.log.error("setRTGeometry: per-triangle arrays must match triangle count")
            return
        }
        // Enforce the index↔vertex invariant at this registration boundary too
        // (symmetric with IlluminatoramaMesh.sanitiseIndices on the from() path):
        // an index >= positions.count would make the GPU acceleration-structure
        // BUILD read vertices past the buffer — a GPU fault that's far harder to
        // diagnose than this early return (the app sandbox eats os_log, so a fault
        // shows up only as a dropped frame / device loss). Callers are controlled
        // today, so this is defense-in-depth, not a known bug.
        // See docs/known-issues/illuminatorama-index-exceeds-vertexcount.md.
        if let mx = indices.max(), Int(mx) >= positions.count {
            Self.log.error("setRTGeometry: index \(mx) >= vertexCount \(positions.count); rejecting (would fault the AS build).")
            return
        }
        // Vertex positions as packed float3 (12-byte) for the AS triangle
        // descriptor; index buffer as uint32; per-tri albedo/normal as float4.
        var packedPos = [SIMD3<Float>]()  // 16-byte stride; AS reads .float3 with stride 16
        packedPos.reserveCapacity(positions.count)
        packedPos.append(contentsOf: positions)
        let albedo4 = triangleAlbedo.map { SIMD4<Float>($0, 1) }
        let normal4 = triangleNormal.map { SIMD4<Float>(simd_normalize($0), 0) }

        guard
            let vb = device.makeBuffer(bytes: packedPos,
                length: MemoryLayout<SIMD3<Float>>.stride * packedPos.count,
                options: .storageModeShared),
            let ib = device.makeBuffer(bytes: indices,
                length: MemoryLayout<UInt32>.stride * indices.count,
                options: .storageModeShared),
            let ab = device.makeBuffer(bytes: albedo4,
                length: MemoryLayout<SIMD4<Float>>.stride * albedo4.count,
                options: .storageModeShared),
            let nb = device.makeBuffer(bytes: normal4,
                length: MemoryLayout<SIMD4<Float>>.stride * normal4.count,
                options: .storageModeShared)
        else {
            Self.log.error("setRTGeometry: buffer allocation failed")
            return
        }
        vb.label = "Illuminatorama.rt.positions"
        ib.label = "Illuminatorama.rt.indices"
        ab.label = "Illuminatorama.rt.triAlbedo"
        nb.label = "Illuminatorama.rt.triNormal"
        rts.rtVertexBuffer = vb
        rts.rtIndexBuffer = ib
        rts.rtTriAlbedoBuffer = ab
        rts.rtTriNormalBuffer = nb
        rts.rtTriangleCount = triCount

        buildSoupAccel()   // builds the primitive AS (triangle soup + curve geom when active)
        Self.log.notice("RT geometry built: \(triCount) triangles")
    }
}
