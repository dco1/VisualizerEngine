import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

extension IlluminatoramaRenderer {
    // ── Surface cache (Lumen-style on-surface lit-radiance atlas) ────
    /// Mirror of `SurfCard` in IlluminatoramaSurfaceCache.metal — one atlas tile.
    /// A tile carries up to TWO planar triangle frames (P3 — 2-triangles-per-tile
    /// packing): frame A occupies the lower-left half (texel `u+v ≤ 1`), frame B
    /// the upper-right half (`u+v > 1`), split on the diagonal. `normal.w` is the
    /// PACKED flag: 1 ⇒ the tile is split between frame A and frame B; 0 ⇒ frame A
    /// fills the WHOLE tile (the room's hand-authored quad cards, where both of a
    /// quad's triangles map across the full tile, and a lone odd per-triangle
    /// card). All `SIMD4` (no SIMD3/packed_float3 stride trap).
    public struct SurfCard {
        // Frame A — lower-left half when packed, whole tile when unpacked.
        public var origin: SIMD4<Float>, uAxis: SIMD4<Float>, vAxis: SIMD4<Float>
        public var normal: SIMD4<Float>, albedo: SIMD4<Float>, emission: SIMD4<Float>
        // Frame B — upper-right half; meaningful only when `normal.w == 1`.
        public var originB: SIMD4<Float>, uAxisB: SIMD4<Float>, vAxisB: SIMD4<Float>
        public var normalB: SIMD4<Float>, albedoB: SIMD4<Float>, emissionB: SIMD4<Float>
        /// Unpacked single-frame card — frame A fills the whole tile (room quads,
        /// lone odd per-triangle card). Frame B left zero, packed flag (normal.w) 0.
        public init(origin: SIMD3<Float>, uAxis: SIMD3<Float>, vAxis: SIMD3<Float>,
                    normal: SIMD3<Float>, albedo: SIMD3<Float>, emission: SIMD3<Float>) {
            self.origin = SIMD4(origin, 0); self.uAxis = SIMD4(uAxis, 0)
            self.vAxis = SIMD4(vAxis, 0); self.normal = SIMD4(simd_normalize(normal), 0)
            self.albedo = SIMD4(albedo, 0); self.emission = SIMD4(emission, 0)
            self.originB = .zero; self.uAxisB = .zero; self.vAxisB = .zero
            self.normalB = .zero; self.albedoB = .zero; self.emissionB = .zero
        }
        /// Packed two-triangle card (P3): frame A in the lower-left half, frame B
        /// in the upper-right half. `normal.w = 1` flags the diagonal split.
        public init(aOrigin: SIMD3<Float>, aUAxis: SIMD3<Float>, aVAxis: SIMD3<Float>,
                    aNormal: SIMD3<Float>, aAlbedo: SIMD3<Float>, aEmission: SIMD3<Float>,
                    bOrigin: SIMD3<Float>, bUAxis: SIMD3<Float>, bVAxis: SIMD3<Float>,
                    bNormal: SIMD3<Float>, bAlbedo: SIMD3<Float>, bEmission: SIMD3<Float>) {
            self.origin = SIMD4(aOrigin, 0); self.uAxis = SIMD4(aUAxis, 0)
            self.vAxis = SIMD4(aVAxis, 0); self.normal = SIMD4(simd_normalize(aNormal), 1)
            self.albedo = SIMD4(aAlbedo, 0); self.emission = SIMD4(aEmission, 0)
            self.originB = SIMD4(bOrigin, 0); self.uAxisB = SIMD4(bUAxis, 0)
            self.vAxisB = SIMD4(bVAxis, 0); self.normalB = SIMD4(simd_normalize(bNormal), 0)
            self.albedoB = SIMD4(bAlbedo, 0); self.emissionB = SIMD4(bEmission, 0)
        }
    }

    /// Mirror of `SurfCacheUniforms` in IlluminatoramaSurfaceCache.metal.
    struct SurfCacheUniforms {
        var sunDir: SIMD4<Float>; var sunColor: SIMD4<Float>; var skyAmbient: SIMD4<Float>
        var atlasW: UInt32; var atlasH: UInt32; var tileSize: UInt32; var tilesPerRow: UInt32
        var cardCount: UInt32; var triangleCount: UInt32; var indirectRays: UInt32; var frameSeed: UInt32
        var alpha: Float; var rayTMin: Float; var maxDist: Float; var incrementalEnabled: UInt32 = 0
        // DH-0653 — atlas reuse counter gate; DH-0849 — back-face guard (+ pad to the Metal stride).
        var statsEnabled: UInt32 = 0; var backFaceGuard: UInt32 = 0
        /// DH-0872 — scotopic desaturation for this pass's own GI sky-MISS sample (the
        /// cache-update kernel gathers indirect independently of the RT lighting kernels'
        /// GI, and has the SAME raw-sample leak). Mirrors `scotopicDesaturation`. Repurposes
        /// `_scPad1` — same 4 bytes. 0 (day) ⇒ byte-identical.
        var scotopicDesaturation: Float = 0
        /// DH-1314 — mirrors `surfaceCacheCardlessReshade` (repurposes `_scPad2`, same 4 bytes).
        var cardlessReshade: UInt32 = 1
    }

    // ── Chart-path incremental invalidation (#60 task 1) ──
    // The per-triangle path (above) re-frames each moved triangle from its object
    // soup × model. A coplanar CHART card (`VIZ_SURFCACHE_CHARTS=1`) instead spans
    // many triangles with ONE bbox-derived frame + fixed per-triangle UVs, so the
    // per-triangle re-frame would clobber the chart's parameterisation. But a card
    // frame is AFFINE-COVARIANT: under a rigid/affine model matrix `M`, the world
    // frame is exactly `origin' = M·origin` (point), `uAxis'/vAxis' = M·axis`
    // (vector), and the per-triangle UVs (`uv = (proj(v)−bboxMin)/bboxSize`) are
    // invariant because verts and frame transform together. So a moved chart card
    // re-frames with a single affine transform of its BAKE-pose object-space frame
    // (`M_bake⁻¹ · worldFrame`), with no per-triangle work and no UV change. Built
    // only on the chart bake path; nil ⇒ per-triangle path (or no incremental).
    struct SurfCardFrame {
        var origin = SIMD3<Float>(0, 0, 0), uAxis = SIMD3<Float>(0, 0, 0)
        var vAxis = SIMD3<Float>(0, 0, 0), normal = SIMD3<Float>(0, 1, 0)
    }

    struct ChartIncremental {
        var objectFrame: [SurfCardFrame]   // per card: bake-pose frame in owner's object space
        var instCardLo: [Int32]            // per instance: first owned card (−1 if none)
        var instCardHi: [Int32]            // per instance: one-past-last owned card
    }

    // ── GPU-side transform diff (#60 item 1) ──
    // Moves the per-frame moved-set detection + card re-frame off the CPU: a
    // compute kernel on the frame's command buffer diffs each card's owner
    // instance (current vs previous model matrix) and re-frames moved cards in
    // place, replacing the CPU loops in `applyIncrementalSurfaceCacheUpdate` /
    // `applyIncrementalChartSurfaceCacheUpdate` for RIGID movers. The CPU path
    // stays as the fallback for Phase-C deforming scenes (their re-frame reads
    // the live soup — GPU-ifying that is the separate Phase D item) and as the
    // `VIZ_SURFCACHE_GPU_DIFF=0` kill-switch / A/B reference. The maps below are
    // baked alongside the CPU ones and cleared with them on topology change.
    static let surfCacheGPUDiffDisabled =
        ProcessInfo.processInfo.environment["VIZ_SURFCACHE_GPU_DIFF"] == "0"

    /// One entry per deforming INSTANCE: the per-dispatch parameters. The
    /// position buffer is looked up per encode from `gpuRepackTasks` by kind
    /// (the solver may re-register; the kind survives the topology bake).
    struct SurfDeformDispatch {
        var kind: MeshKind; var instanceIndex: Int; var triBase: Int; var triCount: Int
    }

    /// Swift mirror of the Metal `SCReframeUniforms` (32 B, passed via setBytes).
    struct SCReframeUniforms {
        var count: UInt32; var cardCount: UInt32; var instanceCount: UInt32; var epsilon: Float
        var rigidEnabled: UInt32; var _pad0: UInt32 = 0; var _pad1: UInt32 = 0; var _pad2: UInt32 = 0
    }

    /// Swift mirror of the Metal `SCDeformUniforms` (16 B, per deform dispatch).
    struct SCDeformUniforms {
        var triBase: UInt32; var triCount: UInt32; var instanceIndex: UInt32; var vertexCount: UInt32
    }

    /// Coplanar charts (`VIZ_SURFCACHE_CHARTS=1`) use a different, non-mesh-invariant
    /// UV parameterisation, so the BLAS UV fold is per-triangle-path only.
    static let surfCacheCharts =
        ProcessInfo.processInfo.environment["VIZ_SURFCACHE_CHARTS"] == "1"

    /// `soupTriBase` value of an instance whose mesh takes no cards (DH-1314). Every consumer
    /// resolves a TLAS hit as `soupTriBase[instance] + primitive`, bounds-checked against the
    /// soup's triangle count, so this reads "no card" everywhere; the cache-update kernel keys its
    /// cardless re-shade off it. Mirror of the Metal `SC_NO_CARD_SOUP_BASE`.
    static let surfaceCacheNoCardSoupBase: UInt32 = 0x8000_0000

    /// Six world-space face patches per instance, ready for
    /// `ZFightDetector.audit(_:requireSameFacing: true)`.
    ///
    /// This is the piece that was missing for native Illuminatorama scenes: the
    /// geometry auditors take `[SurfacePatch]`, and until now the only producer
    /// walked `SCNNode`s — which a native scene does not have. See
    /// `IlluminatoramaSurfaceAudit` for why it emits faces rather than objects.
    public func auditSurfacePatches(_ instances: [InstanceRef],
                                    name: ((Int, InstanceRef) -> String)? = nil) -> [SurfacePatch] {
        IlluminatoramaSurfaceAudit.facePatches(
            instances: instances.map { (kind: $0.meshKind,
                                        model: $0.data.modelMatrix,
                                        normalMatrix: $0.data.normalMatrix) },
            bounds: { [weak self] kind in self?.meshLocalBounds(kind) },
            name: { i in name?(i, instances[i]) ?? "\(instances[i].meshKind)#\(i)" })
    }

    /// The counters run exactly when the sidecar they report to exists. Read per frame
    /// (not cached) so a test can scope it with `setenv`/`unsetenv`.
    static var surfCacheStatsRequested: Bool {
        frameEnv("VIZ_ILLUMI_SURFCACHE_STATS_PATH") != nil
    }
}
