import AppKit
import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── VOLUMETRIC CLOUD 3D PASS ────────────────────────────────────────────────
//
// True depth-correct raymarched cloud. Reads the FrameCompositor's depth
// texture to terminate each pixel's cloud march at scene depth — so
// elements rendered with depth-write (shells, stars) properly occlude the
// cloud at their world-space depth.
//
// The lighting model is the same proven recipe `VolumetricSky.metal` ships
// for the kilometre-scale sky-dome use case — light-march toward a key
// direction for self-shadow, dual-lobe Henyey-Greenstein phase function,
// powder term so cumulus silhouettes pop against the sky. Without that
// recipe a thin slab viewed close-up renders as featureless cotton; with
// it, the cloud reads as cumulus puffs at any viewing distance.
//
// BURST INTEGRATION
//
// Burst illumination is handled OUT-OF-LOOP — accumulated as a screen-space
// additive contribution weighted by cloud opacity, rather than summed
// per-march-step. Per-step summing made each burst paint a vertical
// "search-light" stripe down through the cloud at the camera angles
// Fireworks+ uses; out-of-loop with screen-space attenuation keeps burst
// glow soft and confined to where the cloud actually exists.

@MainActor
public final class VolumetricCloud3DPass {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "VolumetricCloud3DPass")

    public let device: MTLDevice
    public let engine: SimEngine
    public let burstField: BurstLightField

    private let pipeline: MTLComputePipelineState
    private let noiseVolume: NoiseVolume
    private let uniformBuffer: MTLBuffer

    // ── Live tunables ───────────────────────────────────────────────────────

    /// Cloud band altitude (metres above origin). 4..18 is THICK enough
    /// (14 m vertical) that the puffs read as distinct three-dimensional
    /// cumulus from a 30 m camera, rather than a thin paper deck. Real
    /// fair-weather cumulus is typically 1-2 km tall against a viewer
    /// kilometres away; scaling the scene down to ~10 m camera distance,
    /// 14 m of vertical billow keeps the proportions correct.
    public var slabBaseY: Float = 5
    public var slabTopY: Float = 13
    /// World-units → noise-tile scaling. With a 128-tile noise volume,
    /// `hScl = 0.7` gives ~180 m horizontal feature spacing — the
    /// lowest FBM octave dominates, producing a small number of LARGE
    /// distinct cumulus heads instead of a tangle of small ones.
    /// Smaller features (higher hScl) make the cloud read as "tangle
    /// of grey hair" at this 50 m camera distance.
    public var horizontalScale: Float = 0.7
    /// 0 = clear sky, 1 = overcast. The shape/erosion separation in
    /// the kernel needs a slightly higher coverage to land enough puffs
    /// in the FOV (edge-only erosion eats some of the original puff
    /// area). ~0.45 gives 3-5 distinct cumulus heads with big gaps of
    /// clear sky between them.
    public var coverage: Float = 0.45
    /// Mass multiplier on raw cloud density. 1.0 leaves headroom so
    /// the high `absorption` (2.2) doesn't drive every "in cloud"
    /// pixel to fully opaque white — we want puff interiors that go
    /// dense quickly but still admit some interior darkness from the
    /// light-march; pushing density too high overrides that.
    public var density: Float = 1.0
    /// 0..1 — Worley erosion intensity, applied ONLY at the puff
    /// silhouette (the kernel masks it to a thin shell around the
    /// boundary; the solid interior is unaffected). Carves cellular
    /// wispy detail along the edge — what real cumulus shows where
    /// condensation meets dry air. Was 0 previously because the
    /// erosion used to apply everywhere and made the cloud look like
    /// a tangle of grey hair; now that it's edge-masked, a moderate
    /// value is the right look.
    public var erosion: Float = 0.6
    /// Anvil-shoulder weight in the height shaping. 0 = pure
    /// stratocumulus, 0.3+ = visible congestus shoulder near the top.
    public var anvil: Float = 0.25

    // ── Lighting ────────────────────────────────────────────────────────────
    //
    // Real cumulus is bright on the lit side and pretty dark on the
    // shadowed underside. The contrast is what makes the volume read
    // as three-dimensional. We push this hard:
    //   • lightDirection is OFF-AXIS, not straight up — so puffs have
    //     a visible lit side + shadow side, not just a halo round the
    //     silhouette.
    //   • lightIntensity is generous so the lit caps clearly pop after
    //     the ACES tonemap.
    //   • ambientTint is dim — fills the shadow side without flattening
    //     the contrast.
    //   • powder is high so the lit edge of dense puffs has the classic
    //     "silver lining" pop (Schneider's powder approximation of
    //     multiple-scattering).
    //   • absorption is high so the inner cloud quickly goes opaque
    //     and the shadow side stays dark.

    /// Direction TO the key light. For a night-fireworks scene this
    /// is "moonlight from low in the sky" — pushed MORE lateral than
    /// a straight-up moon so the cloud bodies show a clearly lit side
    /// + a shadowed side instead of an evenly-lit dome from the
    /// camera's top-down POV. ChatGPT's "directionality" lever:
    /// without a strong off-axis key, the volume reads as a flat
    /// brightness with no depth cues.
    public var lightDirection: SIMD3<Float> = simd_normalize(SIMD3(0.85, 0.40, 0.35))
    /// Linear HDR colour of the key light. Cool moonlit white.
    public var lightColor: SIMD3<Float> = SIMD3(0.92, 0.95, 1.08)
    /// Multiplier on the key light contribution after HG phase + powder.
    /// Backed off from 2.6 — with absorption at 2.2 and powder at 0.90,
    /// 2.6 drove the lit side of every puff well past 1.0 HDR and the
    /// ACES tonemap saturated them all to white, killing cloud body
    /// detail. 1.6 keeps the lit caps bright but articulated.
    public var lightIntensity: Float = 1.6
    /// Sky ambient — dim blue fill on the shadow side. Going brighter
    /// here flattens the cumulus volume into a flat grey wash.
    public var ambientTint: SIMD3<Float> = SIMD3(0.12, 0.16, 0.24)
    /// Henyey-Greenstein forward-scatter eccentricity (0..0.95). Higher
    /// = sharper "silver lining" around the key direction. 0.82 puts
    /// us solidly in the "real cloud particle scattering" range
    /// (cloud particles' Mie scatter peaks ~0.8 forward).
    public var phaseG: Float = 0.82
    /// Powder term strength (0..1). Schneider's powder approximation
    /// darkens THIN cloud regions (low integrated density along the
    /// view ray) to fake the missing multiple-scattering term — it's
    /// what gives cumulus its classic dark-edge / bright-core pop.
    /// BUT: when a burst lights the cloud from inside, the powder
    /// makes the cloud's perimeter visibly DARK against the burst-lit
    /// sky behind it, reading as a hard black outline rather than a
    /// soft feather. 0.30 keeps a hint of edge darkening for shape
    /// definition without the burst-time outlining artefact.
    public var powder: Float = 0.30
    /// Beer-Lambert absorption coefficient. 1.6 keeps "aggressive
    /// interior extinction" (puff interiors darken quickly along the
    /// ray) without driving the lit-side HDR past where ACES can
    /// preserve any colour gradation. Was 2.2; that was too far.
    public var absorption: Float = 1.6
    /// Per-step inner light-march toward the key.
    public var lightMarchSteps: Int = 5

    // ── Burst integration ──────────────────────────────────────────────────

    /// Multiplier on the additive burst contribution to the cloud.
    /// Backed off from the 0.55 attempt — at that level the bursts
    /// painted the entire cloud body bright, overwhelming the moonlit
    /// ambient and the cumulus shape. 0.22 keeps the burst tint
    /// visible as colour bleeding into the cloud nearest the
    /// explosion without flooding the whole deck.
    public var burstLightingGain: Float = 0.22

    // ── Compositing ─────────────────────────────────────────────────────────

    /// Cap on accumulated alpha so the densest puffs don't fully wipe
    /// out shells / streaks rendered behind them.
    public var maxAlpha: Float = 0.88
    /// Number of march steps along the camera ray. Higher = smoother
    /// integration at higher GPU cost. Bumped 72 → 96 — at 72 a
    /// faint diagonal-grid pattern was still visible in cloud bodies
    /// (the step cadence aliasing against the noise volume's mid-
    /// frequency taps even with blue-noise jitter). 96 + the lowered
    /// macro / Worley frequencies in the kernel kills the pattern
    /// outright at a modest GPU cost.
    public var stepCount: Int = 96

    /// Throttle counter — kernel currently runs every frame, but kept for
    /// future quality presets.
    private var encodeFrameCount: UInt = 0
    private let encodeEveryNFrames: UInt = 1

    public init?(
        engine: SimEngine,
        burstField: BurstLightField
    ) {
        guard let pipeline = engine.pipeline("volumetricCloud3D") else {
            Self.log.error("volumetricCloud3D pipeline lookup failed")
            return nil
        }
        let device = engine.device
        guard let uBuf = device.makeBuffer(
            length: MemoryLayout<CloudVolume3DUniforms>.stride,
            options: .storageModeShared
        ) else {
            Self.log.error("CloudVolume3D uniform buffer alloc failed")
            return nil
        }
        uBuf.label = "VolumetricCloud3D.uniforms"

        self.device = device
        self.engine = engine
        self.burstField = burstField
        self.pipeline = pipeline
        self.noiseVolume = NoiseVolume(engine: engine)
        self.uniformBuffer = uBuf
    }

    /// Encode the cloud pass against the compositor's HDR + depth targets.
    /// Must be called AFTER other scene passes have written colour and
    /// depth — the cloud reads depth to terminate, and read-writes colour
    /// to alpha-blend over.
    public func encode(
        into cb: MTLCommandBuffer,
        target: FrameCompositor,
        cameraPosition: SIMD3<Float>,
        inverseViewProjection: float4x4,
        zNear: Float,
        zFar: Float,
        time: Float
    ) {
        encodeFrameCount &+= 1
        if encodeFrameCount % encodeEveryNFrames != 0 { return }

        let burstAsFloat = Float(bitPattern: UInt32(burstField.activeCount))

        // Pack inverse-VP matrix rows. Metal compute will dot them against
        // the (ndc, 1) vector to un-project.
        let row0 = SIMD4<Float>(inverseViewProjection.columns.0.x,
                                inverseViewProjection.columns.1.x,
                                inverseViewProjection.columns.2.x,
                                inverseViewProjection.columns.3.x)
        let row1 = SIMD4<Float>(inverseViewProjection.columns.0.y,
                                inverseViewProjection.columns.1.y,
                                inverseViewProjection.columns.2.y,
                                inverseViewProjection.columns.3.y)
        let row2 = SIMD4<Float>(inverseViewProjection.columns.0.z,
                                inverseViewProjection.columns.1.z,
                                inverseViewProjection.columns.2.z,
                                inverseViewProjection.columns.3.z)
        let row3 = SIMD4<Float>(inverseViewProjection.columns.0.w,
                                inverseViewProjection.columns.1.w,
                                inverseViewProjection.columns.2.w,
                                inverseViewProjection.columns.3.w)

        let lightDir = simd_normalize(lightDirection)
        var u = CloudVolume3DUniforms(
            invViewProjRow0: row0,
            invViewProjRow1: row1,
            invViewProjRow2: row2,
            invViewProjRow3: row3,
            cameraPos:       SIMD4(cameraPosition.x, cameraPosition.y, cameraPosition.z, 0),
            slabAndCounts:   SIMD4(slabBaseY, slabTopY, horizontalScale, burstAsFloat),
            shapeAndTime:    SIMD4(coverage, density, erosion, time),
            lightDirAndAnvil: SIMD4(lightDir.x, lightDir.y, lightDir.z, anvil),
            lightColorAndIntens: SIMD4(lightColor.x, lightColor.y, lightColor.z, lightIntensity),
            ambientAndPhase: SIMD4(ambientTint.x, ambientTint.y, ambientTint.z, phaseG),
            marchParams:     SIMD4(Float(stepCount), maxAlpha,
                                   Float(lightMarchSteps), powder),
            absorptionAndBurst: SIMD4(absorption, burstLightingGain, zNear, zFar)
        )
        uniformBuffer.contents().copyMemory(
            from: &u, byteCount: MemoryLayout<CloudVolume3DUniforms>.stride
        )

        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "VolumetricCloud3D"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(target.hdrColorTexture, index: 0)
        enc.setTexture(target.depthTexture,    index: 1)
        enc.setTexture(noiseVolume.texture,    index: 2)
        enc.setBuffer(burstField.buffer.buffer, offset: 0, index: 0)
        enc.setBuffer(uniformBuffer,            offset: 0, index: 1)

        let w = target.hdrColorTexture.width
        let h = target.hdrColorTexture.height
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: w, height: h, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

// Mirror of `CloudVolume3DUniforms` in VolumetricCloud3D.metal.
struct CloudVolume3DUniforms {
    var invViewProjRow0: SIMD4<Float>
    var invViewProjRow1: SIMD4<Float>
    var invViewProjRow2: SIMD4<Float>
    var invViewProjRow3: SIMD4<Float>
    var cameraPos:           SIMD4<Float>
    var slabAndCounts:       SIMD4<Float>
    var shapeAndTime:        SIMD4<Float>
    var lightDirAndAnvil:    SIMD4<Float>
    var lightColorAndIntens: SIMD4<Float>
    var ambientAndPhase:     SIMD4<Float>
    var marchParams:         SIMD4<Float>
    var absorptionAndBurst:  SIMD4<Float>
}
