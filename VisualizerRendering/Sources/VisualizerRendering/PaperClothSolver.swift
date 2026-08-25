import Foundation
import Metal
import OSLog
import simd
import VisualizerCore

// ── PAPER CLOTH SOLVER ───────────────────────────────────────────────────────
//
// 2-D XPBD cloth for sheets of paper (GitHub #23). Sheets of subdivided grid
// particles that bend, fold, twist, self-shadow, and collide with each other,
// blown by wind. A 2-D sibling of PBDFieldSolver — same shared-buffer, graph-
// coloured, fixed-substep machinery, extended from 1-D chains to a 2-D grid.
//
// All N sheets live in ONE flat particle buffer; sheet s occupies the
// contiguous slice [s·M, (s+1)·M) where M = gridW·gridH. Constraints are
// concatenated as eight disjoint colour groups so a single global dispatch per
// colour is race-free:
//
//   [stretch H-even | H-odd | V-even | V-odd | shear A-even | A-odd | B-even | B-odd]
//
//   • stretch H : horizontal grid edges (x,y)-(x+1,y), 2-coloured by x parity.
//   • stretch V : vertical   grid edges (x,y)-(x,y+1), 2-coloured by y parity.
//   • shear A   : "\" diagonals (x,y)-(x+1,y+1),       2-coloured by x parity.
//   • shear B   : "/" diagonals (x+1,y)-(x,y+1),       2-coloured by x parity.
//
// The existing pbdConstraint (XPBD distance) + pbdIntegrate (Verlet) kernels
// from PBD.metal handle integrate + every stretch/shear colour with no change —
// each subpass binds the constraint + λ buffers offset to its colour group,
// exactly like PBDFieldSolver.encodeConstraintSubpass.
//
// Per-frame the deformed grid is packed into per-sheet packed_float3
// position + normal buffers (paperWritePositions / paperRecomputeNormals), which
// the host hands to IlluminatoramaRenderer.registerGPUMesh — GPU-resident end to
// end, no CPU readback.
//
// PHASING (see #23): this file starts with stretch + shear + render packing
// (a hanging / free sheet). Wind/aero, dihedral bend, SDF colliders, and
// spatial-hash self-collision are layered on in later passes; the buffer layout
// and substep skeleton are built to receive them without a rewrite.

@MainActor
public final class PaperClothSolver {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "PaperClothSolver")

    public let engine: SimEngine
    public var device: MTLDevice { engine.device }

    // ── Pipelines ──────────────────────────────────────────────────────
    private let integratePipeline:  MTLComputePipelineState
    private let constraintPipeline: MTLComputePipelineState
    private let windPipeline:       MTLComputePipelineState
    private let floorPipeline:      MTLComputePipelineState
    private let collidePipeline:    MTLComputePipelineState
    private let hashClearPipeline:  MTLComputePipelineState
    private let hashCountPipeline:  MTLComputePipelineState
    private let hashScanPipeline:   MTLComputePipelineState
    private let hashScatterPipeline:MTLComputePipelineState
    private let selfCollidePipeline:MTLComputePipelineState
    private let writePosPipeline:   MTLComputePipelineState
    private let normalsPipeline:    MTLComputePipelineState
    private let bendPipeline:       MTLComputePipelineState
    private let stickPipeline:      MTLComputePipelineState

    // ── Particle / constraint storage (all sheets, flat) ────────────────
    public let particleBuffer:   SimBuffer<PBDParticle>
    public let constraintBuffer: SimBuffer<PBDConstraint>
    private let colliderBuffer: SimBuffer<PBDCollider>
    private var colliderCount: Int = 0
    private let lambdaBuffer:  MTLBuffer
    // Dihedral (cloth) bend — its own constraint type, λ buffer and uniforms.
    // Empty unless `configureSheets(dihedralCompliance:)` is given a value, so
    // every existing caller (paper) pays nothing. See paperBendConstraint in
    // PaperCloth.metal for why cloth needs a hinge and paper doesn't.
    private let bendConstraintBuffer: SimBuffer<PaperBendConstraint>
    private let bendLambdaBuffer: MTLBuffer
    private let bendUniformBuffer: MTLBuffer   // PaperBendUniforms
    private let stickUniformBuffer: MTLBuffer  // PaperStickUniforms
    private var bendGroupStart: [Int] = []
    private var bendGroupCount: [Int] = []
    private let uniformBuffer: MTLBuffer       // PBDUniforms (shared kernels)
    private let windUniformBuffer: MTLBuffer   // PaperWindUniforms (wind/aero kernel)
    private let meshUniformBuffer: MTLBuffer   // PaperMeshUniforms (mesh-pack kernels)

    // Self-collision spatial hash (counting sort over all particles).
    private static let hashTableSize = 1 << 14   // 16384 buckets
    private let cellCountsBuffer: MTLBuffer       // atomic_uint[tableSize] (count → cursor → count)
    private let cellOffsetsBuffer: MTLBuffer      // uint[tableSize]
    private let sortedBuffer: MTLBuffer           // uint[maxParticles]
    private let hashUniformBuffer: MTLBuffer      // PaperHashUniforms
    public var selfCollisionEnabled: Bool = true
    /// DEBUG A/B (VIZ_PAPER_LEGACY=1): reproduce the old summed/unclamped self-
    /// collision pushout + the old oversized radius, to measure the blowup the
    /// fix removes. Off in normal use.
    public var legacyPushout: Bool =
        ProcessInfo.processInfo.environment["VIZ_PAPER_LEGACY"] == "1"
    /// DEBUG (VIZ_PAPER_STATS=1): every 30 frames, CPU-read the shared particle +
    /// constraint buffers and log per-vertex jaggedness metrics (max edge strain,
    /// max vertex speed). Verification instrumentation, not the render path.
    private let statsEnabled =
        ProcessInfo.processInfo.environment["VIZ_PAPER_STATS"] == "1"
    private var statsFrame = 0
    // DEBUG isolation kill-switches — disable one force stage at a time and watch
    // maxStrain to find which stage flings the isolated vertices.
    private let killWind = ProcessInfo.processInfo.environment["VIZ_PAPER_NO_WIND"] == "1"
    private let killSDF  = ProcessInfo.processInfo.environment["VIZ_PAPER_NO_SDF"]  == "1"
    private let killFloor = ProcessInfo.processInfo.environment["VIZ_PAPER_NO_FLOOR"] == "1"
    private let killSelf = ProcessInfo.processInfo.environment["VIZ_PAPER_NO_SELF"] == "1"
    /// Collision radius — particles separate to 2·radius. MUST stay ≈ 0.7× rest
    /// edge: if it's much larger than the mesh spacing, each vertex straddles a
    /// whole patch of the other sheet and the pushout (even averaged) over-thickens
    /// contacts. 0.006 → 1.2 cm separation, matching `clothThickness` (paper is thin).
    /// Swept self-collision radius. **Keep it well under HALF the grid spacing.**
    ///
    /// It is a radius for catching one fold landing on another, not a cloth thickness, and there
    /// is no guard: set from a quilt's 20 mm loft on a sheet whose cells were 37 mm, every
    /// neighbouring pair sat permanently in contact and the sheet inflated like a balloon and
    /// curled its edges upward (Daydream Home, 2026-08-21). See `recommendedSelfRadius`.
    public var selfRadius: Float = 0.006
    /// Spatial-hash cell size — set ≥ 2·radius so the 27-cell stencil covers 2·radius
    /// (≈ 2× rest edge keeps buckets small for perf).
    public var selfCellSize: Float = 0.024

    // ── Per-sheet render output (packed_float3, stride 12) ──────────────
    // Allocated as raw shared MTLBuffers of M·12 bytes — the same packed layout
    // DynamicMesh uses (see its STRIDE CONTRACT note). The Illuminatorama repack
    // kernel + the SCNGeometrySource path both read packed_float3 at stride 12.
    private(set) public var positionBuffers: [MTLBuffer] = []
    private(set) public var normalBuffers:   [MTLBuffer] = []
    /// Shared triangle-list index buffer (uint32). Identical for every same-
    /// resolution sheet, so one buffer is handed to all sheets' descriptors.
    private(set) public var indexBuffer: MTLBuffer?
    private(set) public var indexCount: Int = 0
    /// Shared per-vertex UV buffer (packed_float2, stride 8) — grid-local
    /// (col/(W-1), row/(H-1)). Same for every sheet, so paper textures map 1:1.
    private(set) public var uvBuffer: MTLBuffer?

    // ── Layout ──────────────────────────────────────────────────────────
    public private(set) var sheetCount: Int = 0
    public private(set) var gridW: Int = 0
    public private(set) var gridH: Int = 0
    public var verticesPerSheet: Int { gridW * gridH }

    // Disjoint colour-group (start, count) pairs into constraintBuffer / lambda.
    // Order: stretch H-even/odd, V-even/odd, shear A-even/odd, B-even/odd, then
    // skip-one bend H (3-colour by x%3) and V (3-colour by y%3) — 14 groups.
    // Dihedral hinge constraints (cloth only) colour separately into
    // `bendGroupStart`/`bendGroupCount` — 16 groups; see configureSheets.
    private var groupStart: [Int] = []
    private var groupCount: [Int] = []

    // ── Tunables ────────────────────────────────────────────────────────
    public var gravity: Float = 9.81
    public var damping:  Float = 0.99
    public var constraintIterations: Int = 10

    // Wind / aero (0 = inert). See PaperCloth.metal::paperWind.
    public var windAmp:    Float = 0
    public var windFreq:   Float = 0.5
    public var windScroll: Float = 0.8
    public var windDir:    SIMD2<Float> = SIMD2(0, 1)   // (x, z)
    public var aero:       Float = 7.0
    public var drag:       Float = 1.3
    public var turb:       Float = 1.6
    // Updraft jet over the screen mouth (fountain). See PaperCloth.metal::paperWind.
    public var mouth:      SIMD3<Float> = .zero
    public var jetHeight:  Float = 0.5    // thin nozzle
    public var jetRadius:  Float = 0.5    // nozzle footprint
    public var jetUpdraft: Float = 0
    public var wallRadius: Float = 1.3    // lateral soft-wall (wider than nozzle)

    // Collision. `floorEnabled` clamps particles above `floorY`; SDF colliders
    // (set via setColliders) push paper out of the monitor body etc. `selfRadius`
    // is the paper's swept thickness added to every collider's surface.
    public var floorEnabled: Bool = false
    public var floorY: Float = 0
    public var clothThickness: Float = 0.012
    public var collideStiffness: Float = 1.0
    /// Tangential-velocity RETENTION on contact, not friction — 0 stops a sliding contact dead,
    /// 1 is frictionless. The name reads the other way round, and taking it at face value is a
    /// real trap: there is no static threshold anywhere in the solve, so a sheet at rest is never
    /// actually held. At 0.55 a duvet baked onto a bed spent the settle creeping and had slid off
    /// onto the floor by 4.5 s (Daydream Home, 2026-08-21). Bedding-on-a-mattress wants ~0.05.
    public var collideFriction: Float = 0.45
    /// Max per-substep SDF push-out (metres). Caps the body/ceiling collider so a
    /// sheet vertex buried deep in a solid box can't be teleported the full half-
    /// thickness in one step (which tears it from its still-outside neighbour into
    /// a spike). ~2× the grid rest edge — buried patches climb out over a few frames.
    public var sdfMaxPushPerStep: Float = 0.01
    /// STATIC friction: below this tangential speed (m/s), a particle resting on a collider or
    /// the floor does not slide at all — the stick regime `collideFriction` (a retention RATE,
    /// no threshold) never had. 0 disables the pass entirely (default — the paper fountain's
    /// pages are meant to slide). Cloth that has to REST on something wants ~0.05–0.15.
    public var stickSpeed: Float = 0
    /// How far from a surface still counts as resting for the stick pass, over and above
    /// `clothThickness` (which inflates the colliders the same way the collide pass does).
    public var stickBand: Float = 0.006
    /// Dihedral CREASING — the plastic half of fabric bending; see PaperBendUniforms in
    /// PaperCloth.metal. A hinge folded further than `dihedralYieldAngle` (radians of
    /// deviation from rest) has its rest angle creep toward the folded pose by
    /// `dihedralCreepRate` per solver visit. Elastic-only bending gives every fold the
    /// same radius, which reads as rubber; creasing is what puts a crisp ridge on a
    /// waterfall fold. Defaults keep it OFF (elastic only) for every existing caller.
    public var dihedralYieldAngle: Float = .pi
    public var dihedralCreepRate: Float = 0
    private let fixedDt: Float = 1.0 / 120.0
    private var accumulator: Float = 0
    private var time: Float = 0

    // Capacity ceilings (set at init).
    private let maxSheets: Int
    private let maxParticles: Int
    private let maxConstraints: Int

    // ── Init ────────────────────────────────────────────────────────────

    public init?(engine: SimEngine, maxSheets: Int, gridW: Int, gridH: Int,
                 maxColliders: Int = 16) {
        guard gridW >= 2, gridH >= 2, maxSheets >= 1 else { return nil }
        let cache = engine.pipelineCache
        let dev = engine.device
        guard let pbd = cache.pbdPipelines(for: dev),
              let wind = cache.pipelineState(name: "paperWind", device: dev),
              let hClear = cache.pipelineState(name: "paperHashClear", device: dev),
              let hCount = cache.pipelineState(name: "paperHashCount", device: dev),
              let hScan = cache.pipelineState(name: "paperHashScan", device: dev),
              let hScatter = cache.pipelineState(name: "paperHashScatter", device: dev),
              let selfCol = cache.pipelineState(name: "paperSelfCollide", device: dev),
              let writePos = cache.pipelineState(name: "paperWritePositions", device: dev),
              let normals  = cache.pipelineState(name: "paperRecomputeNormals", device: dev),
              let bend     = cache.pipelineState(name: "paperBendConstraint", device: dev),
              let stick    = cache.pipelineState(name: "paperStaticFriction", device: dev)
        else {
            Self.log.error("PaperCloth pipeline cache failed — check PaperCloth.metal ships in VisualizerRendering/Shaders/")
            return nil
        }

        let device = engine.device
        let M = gridW * gridH
        let maxParticles = maxSheets * M
        // Upper bound on constraints per sheet: stretch (~2/vert) + shear
        // (~2/vert) + skip-one bend (~2/vert) ≈ 6 per vertex.
        let perSheet = gridW * gridH * 6
        let maxConstraints = maxSheets * perSheet
        let lambdaBytes = MemoryLayout<Float>.stride * max(maxConstraints, 1)
        // Dihedral hinges: one per quad (the shared "/" diagonal) + one per
        // interior grid edge (both axes) < 3 per vertex.
        let maxBendConstraints = maxSheets * gridW * gridH * 3
        let bendLambdaBytes = MemoryLayout<Float>.stride * max(maxBendConstraints, 1)

        guard
            let pBuf = SimBuffer<PBDParticle>(device: device, capacity: maxParticles,
                                              label: "PaperCloth.particles"),
            let cBuf = SimBuffer<PBDConstraint>(device: device, capacity: maxConstraints,
                                                label: "PaperCloth.constraints"),
            let bBuf = SimBuffer<PaperBendConstraint>(device: device, capacity: maxBendConstraints,
                                                      label: "PaperCloth.bendConstraints"),
            let blBuf = device.makeBuffer(length: bendLambdaBytes, options: .storageModePrivate),
            let buBuf = device.makeBuffer(length: MemoryLayout<PaperBendUniforms>.stride,
                                          options: .storageModeShared),
            let suBuf = device.makeBuffer(length: MemoryLayout<PaperStickUniforms>.stride,
                                          options: .storageModeShared),
            let colBuf = SimBuffer<PBDCollider>(device: device, capacity: max(1, maxColliders),
                                                label: "PaperCloth.colliders"),
            let uBuf = device.makeBuffer(length: MemoryLayout<PBDUniforms>.stride,
                                         options: .storageModeShared),
            let wBuf = device.makeBuffer(length: MemoryLayout<PaperWindUniforms>.stride,
                                         options: .storageModeShared),
            let mBuf = device.makeBuffer(length: MemoryLayout<PaperMeshUniforms>.stride,
                                         options: .storageModeShared),
            let lBuf = device.makeBuffer(length: lambdaBytes, options: .storageModePrivate),
            let ccBuf = device.makeBuffer(length: Self.hashTableSize * MemoryLayout<UInt32>.stride,
                                          options: .storageModePrivate),
            let coBuf = device.makeBuffer(length: Self.hashTableSize * MemoryLayout<UInt32>.stride,
                                          options: .storageModePrivate),
            let srBuf = device.makeBuffer(length: max(1, maxParticles) * MemoryLayout<UInt32>.stride,
                                          options: .storageModePrivate),
            let huBuf = device.makeBuffer(length: MemoryLayout<PaperHashUniforms>.stride,
                                          options: .storageModeShared)
        else {
            Self.log.error("PaperCloth buffer allocation failed")
            return nil
        }
        uBuf.label = "PaperCloth.uniforms"
        wBuf.label = "PaperCloth.windUniforms"
        mBuf.label = "PaperCloth.meshUniforms"
        lBuf.label = "PaperCloth.lambda"
        blBuf.label = "PaperCloth.bendLambda"
        buBuf.label = "PaperCloth.bendUniforms"
        suBuf.label = "PaperCloth.stickUniforms"

        self.engine = engine
        self.integratePipeline  = pbd.integrate
        self.constraintPipeline = pbd.constraint
        self.windPipeline       = wind
        self.floorPipeline      = pbd.floor
        self.collidePipeline    = pbd.collide
        self.hashClearPipeline  = hClear
        self.hashCountPipeline  = hCount
        self.hashScanPipeline   = hScan
        self.hashScatterPipeline = hScatter
        self.selfCollidePipeline = selfCol
        self.writePosPipeline   = writePos
        self.normalsPipeline    = normals
        self.bendPipeline       = bend
        self.stickPipeline      = stick
        self.particleBuffer     = pBuf
        self.constraintBuffer   = cBuf
        self.bendConstraintBuffer = bBuf
        self.bendLambdaBuffer   = blBuf
        self.bendUniformBuffer  = buBuf
        self.stickUniformBuffer = suBuf
        self.colliderBuffer     = colBuf
        self.uniformBuffer      = uBuf
        self.windUniformBuffer  = wBuf
        self.meshUniformBuffer  = mBuf
        self.lambdaBuffer       = lBuf
        self.cellCountsBuffer   = ccBuf
        self.cellOffsetsBuffer  = coBuf
        self.sortedBuffer       = srBuf
        self.hashUniformBuffer  = huBuf
        self.maxSheets       = maxSheets
        self.maxParticles    = maxParticles
        self.maxConstraints  = maxConstraints
        self.gridW = gridW
        self.gridH = gridH
    }

    // ── Sheet placement spec ────────────────────────────────────────────

    public struct SheetSpec {
        /// World position of grid vertex (col 0, row 0).
        public var origin: SIMD3<Float>
        /// Direction of increasing column (will be scaled by `sizeW`). Need not
        /// be unit — it's normalised internally.
        public var right: SIMD3<Float>
        /// Direction of increasing row (will be scaled by `sizeH`).
        public var down: SIMD3<Float>
        public var sizeW: Float
        public var sizeH: Float
        /// Pin row 0 (the col-0..W-1 edge at the origin) in place — a hanging
        /// cloth. `false` = fully free (blown by wind).
        public var pinFirstRow: Bool

        public init(origin: SIMD3<Float>, right: SIMD3<Float>, down: SIMD3<Float>,
                    sizeW: Float, sizeH: Float, pinFirstRow: Bool = false) {
            self.origin = origin; self.right = right; self.down = down
            self.sizeW = sizeW; self.sizeH = sizeH; self.pinFirstRow = pinFirstRow
        }
    }

    // ── Configuration ───────────────────────────────────────────────────

    /// Build `specs.count` sheets into the flat buffers, generate the eight
    /// constraint colour groups, allocate per-sheet render buffers, and build
    /// the shared triangle index buffer. Recallable to rebuild (e.g. sheet-count
    /// change), as long as counts stay within the init capacities.
    /// `dihedralCompliance` — nil (the default) keeps the paper model: skip-one
    /// distance bend only, zero cost, byte-identical for every existing caller.
    /// A value adds a real XPBD dihedral HINGE across every triangle pair (the
    /// quad diagonal + both interior grid-edge directions), which is what makes
    /// CLOTH roll smoothly where paper creases — see paperBendConstraint in
    /// PaperCloth.metal for the mechanism. Cloth callers pair it with a soft
    /// (large) `bendCompliance` so the chord term stops fighting the hinge.
    public func configureSheets(_ specs: [SheetSpec],
                                stretchCompliance: Float = 1.0e-6,
                                shearCompliance: Float = 5.0e-6,
                                bendCompliance: Float = 4.0e-5,
                                dihedralCompliance: Float? = nil) {
        precondition(specs.count <= maxSheets, "configureSheets: \(specs.count) > maxSheets \(maxSheets)")
        let W = gridW, H = gridH, M = W * H
        sheetCount = specs.count

        // ── Particles ──
        var particles = [PBDParticle](); particles.reserveCapacity(specs.count * M)
        for spec in specs {
            let r = simd_normalize(spec.right)
            let d = simd_normalize(spec.down)
            for y in 0..<H {
                for x in 0..<W {
                    let fx = W > 1 ? Float(x) / Float(W - 1) : 0
                    let fy = H > 1 ? Float(y) / Float(H - 1) : 0
                    let p = spec.origin + r * (spec.sizeW * fx) + d * (spec.sizeH * fy)
                    let pinned = spec.pinFirstRow && y == 0
                    var particle = PBDParticle(position: p, invMass: pinned ? 0 : 1)
                    particle.prevPositionAndPad = SIMD4(p, 0)
                    particles.append(particle)
                }
            }
        }
        particleBuffer.write(particles)

        // ── Constraints (8 colour groups) ──
        // index of (x,y) in sheet s
        func gi(_ s: Int, _ x: Int, _ y: Int) -> UInt32 { UInt32(s * M + y * W + x) }
        func restLen(_ a: UInt32, _ b: UInt32) -> Float {
            simd_length(particles[Int(b)].position - particles[Int(a)].position)
        }
        // 14 colour groups: stretch H 0/1, V 2/3, shear A 4/5, B 6/7,
        // skip-one bend H 8/9/10 (x%3), V 11/12/13 (y%3).
        var groups: [[PBDConstraint]] = Array(repeating: [], count: 14)
        for s in 0..<specs.count {
            // Stretch — horizontal & vertical grid edges.
            for y in 0..<H {
                for x in 0..<(W - 1) {
                    let a = gi(s, x, y), b = gi(s, x + 1, y)
                    groups[x % 2 == 0 ? 0 : 1].append(
                        PBDConstraint(i: a, j: b, restLength: restLen(a, b), compliance: stretchCompliance))
                }
            }
            for y in 0..<(H - 1) {
                for x in 0..<W {
                    let a = gi(s, x, y), b = gi(s, x, y + 1)
                    groups[y % 2 == 0 ? 2 : 3].append(
                        PBDConstraint(i: a, j: b, restLength: restLen(a, b), compliance: stretchCompliance))
                }
            }
            // Shear — both cell diagonals.
            for y in 0..<(H - 1) {
                for x in 0..<(W - 1) {
                    let a = gi(s, x, y), b = gi(s, x + 1, y + 1)
                    groups[x % 2 == 0 ? 4 : 5].append(
                        PBDConstraint(i: a, j: b, restLength: restLen(a, b), compliance: shearCompliance))
                    let c = gi(s, x + 1, y), e = gi(s, x, y + 1)
                    groups[x % 2 == 0 ? 6 : 7].append(
                        PBDConstraint(i: c, j: e, restLength: restLen(c, e), compliance: shearCompliance))
                }
            }
            // Bend — skip-one distance (resists flexing → paper-like stiffness).
            // 3-colour along each axis (x%3 / y%3) so skip-2 neighbours that
            // share a vertex never land in the same dispatch.
            for y in 0..<H {
                for x in 0..<(W - 2) {
                    let a = gi(s, x, y), b = gi(s, x + 2, y)
                    groups[8 + (x % 3)].append(
                        PBDConstraint(i: a, j: b, restLength: restLen(a, b), compliance: bendCompliance))
                }
            }
            for y in 0..<(H - 2) {
                for x in 0..<W {
                    let a = gi(s, x, y), b = gi(s, x, y + 2)
                    groups[11 + (y % 3)].append(
                        PBDConstraint(i: a, j: b, restLength: restLen(a, b), compliance: bendCompliance))
                }
            }
        }
        var all = [PBDConstraint](); all.reserveCapacity(groups.reduce(0) { $0 + $1.count })
        groupStart.removeAll(keepingCapacity: true)
        groupCount.removeAll(keepingCapacity: true)
        var start = 0
        for g in groups {
            groupStart.append(start)
            groupCount.append(g.count)
            all += g
            start += g.count
        }
        constraintBuffer.write(all)

        // ── Dihedral hinge constraints (cloth bend) ──
        //
        // One hinge per triangle pair of the render triangulation (see
        // buildIndexBuffer: quad (x,y) splits along the "/" diagonal
        // (x,y+1)–(x+1,y)). Three families, each graph-coloured so no two
        // hinges in a dispatch share a particle:
        //   D — the quad's own diagonal.        Stencil 2×2   → 4 colours (x%2, y%2).
        //   V — interior vertical grid edges.   Stencil 3×2   → 6 colours (x%3, y%2).
        //   H — interior horizontal grid edges. Stencil 2×3   → 6 colours (x%2, y%3).
        // Rest angle π: the sheet is cut flat, so every hinge restores toward flat.
        bendGroupStart.removeAll(keepingCapacity: true)
        bendGroupCount.removeAll(keepingCapacity: true)
        if let alpha = dihedralCompliance {
            func hinge(_ e0: UInt32, _ e1: UInt32, _ w0: UInt32, _ w1: UInt32) -> PaperBendConstraint {
                PaperBendConstraint(v: SIMD4(e0, e1, w0, w1), restAngle: .pi, compliance: alpha)
            }
            var bendGroups: [[PaperBendConstraint]] = Array(repeating: [], count: 16)
            for s in 0..<specs.count {
                for y in 0..<(H - 1) {
                    for x in 0..<(W - 1) {
                        // D: shared "/" diagonal; wings are the quad's other corners.
                        bendGroups[(x % 2) + 2 * (y % 2)].append(
                            hinge(gi(s, x + 1, y), gi(s, x, y + 1), gi(s, x, y), gi(s, x + 1, y + 1)))
                        // V: edge (x,y)–(x,y+1) between quads (x-1,y) and (x,y).
                        if x >= 1 {
                            bendGroups[4 + (x % 3) + 3 * (y % 2)].append(
                                hinge(gi(s, x, y), gi(s, x, y + 1), gi(s, x + 1, y), gi(s, x - 1, y + 1)))
                        }
                        // H: edge (x,y)–(x+1,y) between quads (x,y-1) and (x,y).
                        if y >= 1 {
                            bendGroups[10 + (x % 2) + 2 * (y % 3)].append(
                                hinge(gi(s, x, y), gi(s, x + 1, y), gi(s, x, y + 1), gi(s, x + 1, y - 1)))
                        }
                    }
                }
            }
            var allBend = [PaperBendConstraint]()
            allBend.reserveCapacity(bendGroups.reduce(0) { $0 + $1.count })
            var bendStart = 0
            for g in bendGroups {
                bendGroupStart.append(bendStart)
                bendGroupCount.append(g.count)
                allBend += g
                bendStart += g.count
            }
            if !allBend.isEmpty { bendConstraintBuffer.write(allBend) }
        }

        // ── Per-sheet render buffers (packed_float3, stride 12) ──
        positionBuffers.removeAll(keepingCapacity: true)
        normalBuffers.removeAll(keepingCapacity: true)
        let packedBytes = M * 3 * MemoryLayout<Float>.stride   // M × 12
        for s in 0..<specs.count {
            let pb = device.makeBuffer(length: packedBytes, options: .storageModeShared)
            let nb = device.makeBuffer(length: packedBytes, options: .storageModeShared)
            pb?.label = "PaperCloth.pos[\(s)]"
            nb?.label = "PaperCloth.norm[\(s)]"
            positionBuffers.append(pb!)
            normalBuffers.append(nb!)
        }

        // ── Shared triangle index + UV buffers ──
        buildIndexBuffer()
        buildUVBuffer()

        // Seed render buffers from the rest pose so the mesh is valid before the
        // first substep runs (the host registers the mesh immediately).
        seedRenderBuffers(specs: specs)

        // Mesh-pack uniforms (same for every sheet — base handled via offset).
        let mPtr = meshUniformBuffer.contents().bindMemory(to: PaperMeshUniforms.self, capacity: 1)
        mPtr.pointee = PaperMeshUniforms(vertexCount: UInt32(M),
                                         gridW: UInt32(W), gridH: UInt32(H), _pad: 0)
    }

    private func buildIndexBuffer() {
        let W = gridW, H = gridH
        var idx = [UInt32](); idx.reserveCapacity((W - 1) * (H - 1) * 6)
        for y in 0..<(H - 1) {
            for x in 0..<(W - 1) {
                let i00 = UInt32(y * W + x)
                let i10 = UInt32(y * W + x + 1)
                let i01 = UInt32((y + 1) * W + x)
                let i11 = UInt32((y + 1) * W + x + 1)
                idx.append(contentsOf: [i00, i01, i10, i10, i01, i11])
            }
        }
        indexCount = idx.count
        indexBuffer = idx.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: idx.count * MemoryLayout<UInt32>.stride,
                              options: .storageModeShared)
        }
        indexBuffer?.label = "PaperCloth.indices"
    }

    private func buildUVBuffer() {
        let W = gridW, H = gridH, M = W * H
        var uv = [Float](repeating: 0, count: M * 2)   // packed float2
        for y in 0..<H {
            for x in 0..<W {
                let i = y * W + x
                uv[2*i]   = W > 1 ? Float(x) / Float(W - 1) : 0
                uv[2*i+1] = H > 1 ? Float(y) / Float(H - 1) : 0
            }
        }
        uvBuffer = uv.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: M * 2 * MemoryLayout<Float>.stride,
                              options: .storageModeShared)
        }
        uvBuffer?.label = "PaperCloth.uv"
    }

    /// CPU-seed each sheet's packed position + flat normal buffers from the rest
    /// pose, so the registered mesh renders correctly on frame 0 (before the
    /// first GPU substep + pack runs). Writes packed (3 contiguous floats).
    private func seedRenderBuffers(specs: [SheetSpec]) {
        let W = gridW, H = gridH, M = W * H
        for s in 0..<specs.count {
            let spec = specs[s]
            let r = simd_normalize(spec.right)
            let d = simd_normalize(spec.down)
            var nrm = simd_normalize(simd_cross(r, d))
            if !nrm.x.isFinite { nrm = SIMD3(0, 1, 0) }
            let pp = positionBuffers[s].contents().bindMemory(to: Float.self, capacity: M * 3)
            let np = normalBuffers[s].contents().bindMemory(to: Float.self, capacity: M * 3)
            for y in 0..<H {
                for x in 0..<W {
                    let fx = W > 1 ? Float(x) / Float(W - 1) : 0
                    let fy = H > 1 ? Float(y) / Float(H - 1) : 0
                    let p = spec.origin + r * (spec.sizeW * fx) + d * (spec.sizeH * fy)
                    let i = y * W + x
                    pp[3*i] = p.x; pp[3*i+1] = p.y; pp[3*i+2] = p.z
                    np[3*i] = nrm.x; np[3*i+1] = nrm.y; np[3*i+2] = nrm.z
                }
            }
        }
    }

    // ── Per-frame substep loop ──────────────────────────────────────────

    /// Accumulate wallDt and encode up to 4 fixed substeps + the per-sheet mesh
    /// pack into `cb`. The caller commits (no waitUntilCompleted). Wind time
    /// advances by wallDt regardless of substep count.
    public func encode(to cb: MTLCommandBuffer, wallDt: Float) {
        guard sheetCount > 0 else { return }
        if statsEnabled {
            statsFrame += 1
            if statsFrame % 30 == 0 { logStats() }   // reads LAST committed frame's state
        }
        time += wallDt
        accumulator += wallDt
        var steps = 0
        while accumulator >= fixedDt && steps < 4 {
            encodeSubstep(to: cb, dt: fixedDt)
            accumulator -= fixedDt
            steps += 1
        }
        // Inter-sheet (and self) collision — once per frame after the substeps.
        if selfCollisionEnabled, !killSelf, particleBuffer.count > 0 {
            encodeSelfCollide(cb)
        }
        // Pack the deformed grid into the render buffers once per frame.
        encodeMeshPack(cb)
    }

    private func encodeSelfCollide(_ cb: MTLCommandBuffer) {
        let n = particleBuffer.count
        writeHashUniforms()
        var T = UInt32(Self.hashTableSize)

        if let e = cb.makeComputeCommandEncoder() {
            e.label = "PaperCloth.hashClear"; e.setComputePipelineState(hashClearPipeline)
            e.setBuffer(cellCountsBuffer, offset: 0, index: 0)
            e.setBytes(&T, length: 4, index: 1)
            dispatch1D(e, pipeline: hashClearPipeline, count: Self.hashTableSize); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "PaperCloth.hashCount"; e.setComputePipelineState(hashCountPipeline)
            e.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
            e.setBuffer(cellCountsBuffer, offset: 0, index: 1)
            e.setBuffer(hashUniformBuffer, offset: 0, index: 2)
            dispatch1D(e, pipeline: hashCountPipeline, count: n); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "PaperCloth.hashScan"; e.setComputePipelineState(hashScanPipeline)
            e.setBuffer(cellCountsBuffer, offset: 0, index: 0)
            e.setBuffer(cellOffsetsBuffer, offset: 0, index: 1)
            e.setBytes(&T, length: 4, index: 2)
            e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.label = "PaperCloth.hashScatter"; e.setComputePipelineState(hashScatterPipeline)
            e.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
            e.setBuffer(cellCountsBuffer, offset: 0, index: 1)   // reused as write cursor
            e.setBuffer(cellOffsetsBuffer, offset: 0, index: 2)
            e.setBuffer(sortedBuffer, offset: 0, index: 3)
            e.setBuffer(hashUniformBuffer, offset: 0, index: 4)
            dispatch1D(e, pipeline: hashScatterPipeline, count: n); e.endEncoding()
        }
        // Two Jacobi relaxation passes against the same buckets.
        for _ in 0..<2 {
            if let e = cb.makeComputeCommandEncoder() {
                e.label = "PaperCloth.selfCollide"; e.setComputePipelineState(selfCollidePipeline)
                e.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
                e.setBuffer(cellOffsetsBuffer, offset: 0, index: 1)
                e.setBuffer(cellCountsBuffer, offset: 0, index: 2)   // per-bucket count post-scatter
                e.setBuffer(sortedBuffer, offset: 0, index: 3)
                e.setBuffer(hashUniformBuffer, offset: 0, index: 4)
                dispatch1D(e, pipeline: selfCollidePipeline, count: n); e.endEncoding()
            }
        }
    }

    /// CPU-read the shared particle + stretch-constraint buffers and log the two
    /// metrics that capture "jagged points": max per-vertex SPEED (the velocity
    /// the pushout injects via the prev-shift) and max grid-edge STRAIN
    /// (|edge−rest|/rest — a smooth sheet stays near 0, a yanked vertex spikes it).
    /// Debug-only (VIZ_PAPER_STATS=1); a full-buffer CPU read, not the render path.
    private func logStats() {
        let n = particleBuffer.count
        guard n > 0, constraintBuffer.count > 0 else { return }
        let pPtr = particleBuffer.buffer.contents().bindMemory(to: PBDParticle.self, capacity: n)
        let cPtr = constraintBuffer.buffer.contents().bindMemory(to: PBDConstraint.self,
                                                                 capacity: constraintBuffer.count)
        let dt = fixedDt
        let M = verticesPerSheet, W = gridW, H = gridH
        var maxSpeed: Float = 0, nonFinite = 0, fastVtx = -1
        for k in 0..<n {
            let p = pPtr[k]
            let pos = p.position
            if !pos.x.isFinite || !pos.y.isFinite || !pos.z.isFinite { nonFinite += 1; continue }
            let prev = SIMD3(p.prevPositionAndPad.x, p.prevPositionAndPad.y, p.prevPositionAndPad.z)
            let sp = simd_length(pos - prev) / dt
            if sp > maxSpeed { maxSpeed = sp; fastVtx = k }
        }
        // Stretch strain over colour groups 0..3 (the grid edges); record where the
        // worst edge sits (which sheet, which grid cell, on which border).
        var maxStrain: Float = 0, sumStrain: Float = 0, nStrain = 0, worstI = -1
        for g in 0..<min(4, groupCount.count) {
            let s = groupStart[g]
            for idx in s..<(s + groupCount[g]) {
                let con = cPtr[idx]
                guard con.restLength > 1e-9 else { continue }
                let d = simd_length(pPtr[Int(con.j)].position - pPtr[Int(con.i)].position)
                guard d.isFinite else { continue }
                let strain = abs(d - con.restLength) / con.restLength
                if strain > maxStrain { maxStrain = strain; worstI = Int(con.i) }
                sumStrain += strain; nStrain += 1
            }
        }
        let meanStrain = nStrain > 0 ? sumStrain / Float(nStrain) : 0
        // Decode worst-edge + fastest-vertex grid coords.
        func loc(_ gIdx: Int) -> String {
            guard gIdx >= 0, M > 0 else { return "n/a" }
            let sh = gIdx / M, lo = gIdx % M, x = lo % W, y = lo / W
            let border = (x == 0 || x == W-1 || y == 0 || y == H-1) ? "BORDER" : "inner "
            return "sheet\(sh) (\(x),\(y)) \(border)"
        }
        let fastPos = fastVtx >= 0 ? pPtr[fastVtx].position : SIMD3<Float>(0,0,0)
        let mode = legacyPushout ? "LEGACY" : "FIXED "
        let line = String(
            format: "[paperStats %@] t=%6.2f  maxSpeed=%8.2f@%@ pos(%.2f,%.2f,%.2f)  maxStrain=%9.2f@%@  meanStrain=%6.4f  nonFinite=%d/%d",
            mode, time, maxSpeed, loc(fastVtx), fastPos.x, fastPos.y, fastPos.z,
            maxStrain, loc(worstI), meanStrain, nonFinite, n)
        Self.log.info("\(line, privacy: .public)")
        let path = ProcessInfo.processInfo.environment["VIZ_PAPER_STATS_LOG"] ?? "/tmp/paper_stats.log"
        if let h = FileHandle(forWritingAtPath: path) {
            h.seekToEndOfFile(); h.write(Data((line + "\n").utf8)); h.closeFile()
        } else {
            try? (line + "\n").write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    private func writeHashUniforms() {
        // Legacy A/B reproduces BOTH old conditions: the oversized radius (0.018 →
        // minDist 0.036, far beyond the mesh spacing) AND the summed/unclamped
        // pushout. Either alone is milder; the blowup is the combination.
        let radius = legacyPushout ? 0.018 : selfRadius
        let cell   = legacyPushout ? 0.05  : selfCellSize
        let p = hashUniformBuffer.contents().bindMemory(to: PaperHashUniforms.self, capacity: 1)
        p.pointee = PaperHashUniforms(
            particleCount: UInt32(particleBuffer.count),
            tableSize: UInt32(Self.hashTableSize),
            cellSize: cell, radius: radius,
            gridW: UInt32(gridW), verticesPerSheet: UInt32(verticesPerSheet),
            skipRadius: 1, legacy: legacyPushout ? 1 : 0)
    }

    private func encodeSubstep(to cb: MTLCommandBuffer, dt: Float) {
        writeUniforms(dt: dt)

        // 1. Integrate (Verlet + gravity + damping).
        encodePass(cb, pipeline: integratePipeline,
                   buffers: [particleBuffer.buffer, constraintBuffer.buffer, uniformBuffer],
                   count: particleBuffer.count, label: "PaperCloth.integrate")

        // 1b. Wind + aero (injects velocity via prev-shift, like grass forces).
        if !killWind, windAmp != 0 || jetUpdraft != 0 || turb != 0 {
            writeWindUniforms(dt: dt)
            encodePass(cb, pipeline: windPipeline,
                       buffers: [particleBuffer.buffer, windUniformBuffer],
                       count: particleBuffer.count, label: "PaperCloth.wind")
        }

        // 2. Reset XPBD λ for this substep (distance AND dihedral).
        if constraintBuffer.count > 0, let blit = cb.makeBlitCommandEncoder() {
            blit.label = "PaperCloth.lambdaReset"
            blit.fill(buffer: lambdaBuffer,
                      range: 0..<(MemoryLayout<Float>.stride * constraintBuffer.count), value: 0)
            if bendConstraintBuffer.count > 0 {
                blit.fill(buffer: bendLambdaBuffer,
                          range: 0..<(MemoryLayout<Float>.stride * bendConstraintBuffer.count),
                          value: 0)
            }
            blit.endEncoding()
        }
        if bendConstraintBuffer.count > 0 {
            let bPtr = bendUniformBuffer.contents().bindMemory(to: PaperBendUniforms.self, capacity: 1)
            bPtr.pointee = PaperBendUniforms(constraintCount: UInt32(bendConstraintBuffer.count), dt: dt,
                                             yieldAngle: dihedralYieldAngle, creepRate: dihedralCreepRate)
        }

        // 3. Constraint iterations — each iteration cycles every distance colour
        //    group, then (cloth only) every dihedral hinge colour group.
        for _ in 0..<constraintIterations {
            for g in groupCount.indices where groupCount[g] > 0 {
                encodeConstraintSubpass(cb, start: groupStart[g], count: groupCount[g],
                                        label: "PaperCloth.c[\(g)]")
            }
            for g in bendGroupCount.indices where bendGroupCount[g] > 0 {
                encodeBendSubpass(cb, start: bendGroupStart[g], count: bendGroupCount[g],
                                  label: "PaperCloth.bend[\(g)]")
            }
        }

        // 4. Collision — AFTER constraints so contacts win the final position.
        if colliderCount > 0, !killSDF {
            encodePass(cb, pipeline: collidePipeline,
                       buffers: [particleBuffer.buffer, colliderBuffer.buffer, uniformBuffer],
                       count: particleBuffer.count, label: "PaperCloth.sdf")
        }
        if floorEnabled, !killFloor {
            encodePass(cb, pipeline: floorPipeline,
                       buffers: [particleBuffer.buffer, constraintBuffer.buffer, uniformBuffer],
                       count: particleBuffer.count, label: "PaperCloth.floor")
        }

        // 5. Static friction — LAST, so it judges the substep's final motion.
        //    See paperStaticFriction in PaperCloth.metal: resting contacts below
        //    `stickSpeed` hold instead of creeping.
        if stickSpeed > 0, colliderCount > 0 || floorEnabled {
            let sPtr = stickUniformBuffer.contents().bindMemory(to: PaperStickUniforms.self, capacity: 1)
            sPtr.pointee = PaperStickUniforms(
                particleCount: UInt32(particleBuffer.count),
                colliderCount: UInt32(colliderCount),
                dt: dt,
                floorY: floorEnabled ? floorY : -1e9,
                contactBand: clothThickness + stickBand,
                stickSpeed: stickSpeed,
                selfRadius: clothThickness)
            encodePass(cb, pipeline: stickPipeline,
                       buffers: [particleBuffer.buffer, colliderBuffer.buffer, stickUniformBuffer],
                       count: particleBuffer.count, label: "PaperCloth.stick")
        }
    }

    /// Set the static SDF colliders the paper collides with (monitor body,
    /// floor box, etc.). Tag each with an ownerID != .max so the kernel's
    /// skip-self filter never drops them (paper particles carry no ownerID).
    /// **Every collider must carry an `ownerID` other than `.max`.**
    ///
    /// The collide kernel skips any collider whose owner matches the solver's, and this solver
    /// runs with `ownerID: .max` because paper particles carry no owner — so a collider built
    /// with `PBDCollider.box(center:halfExtents:)`'s DEFAULT owner is silently dropped, every
    /// time, and the cloth falls straight through it. This is asserted in debug rather than left
    /// as prose, because prose did not stop it happening (Daydream Home, 2026-08-21).
    // ── Baking a settled drape ──────────────────────────────────────────────
    //
    // The solver's defaults are tuned for its origin scene: paper flying in a fountain. Draping
    // cloth ONTO something and letting it come to rest is a different job with different traps,
    // and these three helpers are what it took to get a duvet onto a bed (Daydream Home,
    // 2026-08-21). They are additive — nothing here changes an existing caller.

    /// Kill every air force: wind, aero, drag, turbulence, the updraft jet, and the lateral soft
    /// wall. A settling bake wants gravity and nothing else.
    ///
    /// Worth calling explicitly even though it looks like a no-op on a fresh solver: `windAmp` is
    /// 0 by default but `aero`, `drag` and `turb` are NOT, and `wallRadius` is 1.3 m — which a
    /// queen bed's corners sit exactly on. Left alone they kicked a settling sheet up and rolled
    /// it into a tube.
    public func stillAir() {
        windAmp = 0; aero = 0; drag = 0; turb = 0
        jetUpdraft = 0
        wallRadius = .greatestFiniteMagnitude
    }

    /// A self-collision radius that is safe for a given grid spacing — comfortably under half,
    /// so neighbouring particles are never permanently in contact.
    public static func recommendedSelfRadius(forSpacing spacing: Float) -> Float {
        max(1e-4, spacing * 0.18)
    }

    /// Fold the sheet's overhang down over an axis-aligned box BEFORE the first step, so the
    /// solver relaxes a drape instead of inventing one.
    ///
    /// **This is the difference between cloth that hangs and cloth that curls.** A flat sheet
    /// released above a mattress has to find its overhang by billowing: the free edge swings past
    /// vertical, the surplus buckles, and the fold then LOCKS — gravity does not undo a stable
    /// fold, and no amount of settling did. The measured profile fell 100 mm and came straight
    /// back up 110 mm with three nodes stacked at one x, cloth folded in half against itself.
    /// Softening the bend did not fix it; disabling self-collision did not fix it; it was present
    /// in the raw solver output before any post-processing. Starting the cloth where it is going
    /// did fix it, and the same profile then descended monotonically for its full 263 mm.
    ///
    /// Call after `configureSheets` and before stepping. `top` is the height the cloth rests at;
    /// anything whose plan position lies outside `halfExtents` is taken down the nearer face by
    /// however far past the edge it was.
    ///
    /// **Corners start as a GATHER, not a tent.** Cloth past the edge in BOTH plan axes used to
    /// be split by `dx > dz` onto one of the two faces, which folds the corner surplus into a
    /// symmetric diagonal ridge — and that tent is a stable equilibrium: the settle cannot break
    /// its symmetry, so the corner freezes as a stiff pyramid instead of falling (the classic
    /// synthetic-bedspread corner). Real cloth never passes through that state. Corner cloth is
    /// instead hung RADIALLY around the corner's vertical edge on a shallow cone
    /// (`cornerSlope`): the quarter-arc a radius maps onto is several times shorter than the
    /// cloth that has to occupy it, so the surplus has no flat minimum at all — it must buckle
    /// into folds, which self-collision then stacks into the hanging cascade a tablecloth or
    /// duvet corner actually forms. The cone blends smoothly into both face mappings
    /// (`sin 2φ` → 0 at either boundary; `y = top − r` matches `top − over` there).
    public func preDrape(overCenter center: SIMD3<Float>, halfExtents: SIMD3<Float>,
                         clearance: Float = 0.004, cornerSlope: Float = 0.25) {
        guard sheetCount > 0 else { return }
        let p = particleBuffer.contents
        let top = center.y + halfExtents.y
        for n in 0 ..< (verticesPerSheet * sheetCount) {
            let q = p[n].position
            let dx = abs(q.x - center.x) - halfExtents.x
            let dz = abs(q.z - center.z) - halfExtents.z
            let over = max(dx, dz)
            guard over > 0 else { continue }
            let sx = q.x < center.x ? -Float(1) : 1
            let sz = q.z < center.z ? -Float(1) : 1
            var x: Float, y: Float, z: Float
            if dx > 0, dz > 0 {
                // Corner: hang radially around the corner edge on a shallow cone.
                let r = sqrt(dx * dx + dz * dz)
                let phi = atan2(dz, dx)                       // 0 at the x face, π/2 at z
                let rad = clearance + cornerSlope * sin(2 * phi) * r
                x = center.x + sx * (halfExtents.x + rad * cos(phi))
                z = center.z + sz * (halfExtents.z + rad * sin(phi))
                y = top - r
            } else {
                x = dx > dz ? center.x + sx * (halfExtents.x + clearance) : q.x
                z = dz >= dx ? center.z + sz * (halfExtents.z + clearance) : q.z
                y = top - over
            }
            p[n].positionAndInvMass = SIMD4(x, y, z, p[n].invMass)
            p[n].prevPositionAndPad = SIMD4(x, y, z, 0)
        }
    }

    /// Pin a square patch of one sheet at wherever it currently is, by zeroing its inverse mass.
    ///
    /// For a BAKE, the settle is a race between two things: the hems need time to finish falling,
    /// and `collideFriction` has no static threshold, so given that time the whole sheet creeps
    /// and eventually walks off whatever it is lying on. Pinning the middle — the part of a
    /// draped cloth that genuinely does not move — buys the edges as long as they need.
    ///
    /// Call it AFTER the cloth has landed, never before: pinned at its release pose the sheet
    /// hangs from its own middle like a tablecloth on a nail.
    public func pinPatch(sheet: Int = 0, halfCells: Int) {
        guard sheet >= 0, sheet < sheetCount, gridW > 0, gridH > 0 else { return }
        let base = sheet * verticesPerSheet
        let p = particleBuffer.contents
        for j in max(0, gridH / 2 - halfCells) ... min(gridH - 1, gridH / 2 + halfCells) {
            for i in max(0, gridW / 2 - halfCells) ... min(gridW - 1, gridW / 2 + halfCells) {
                p[base + j * gridW + i].positionAndInvMass.w = 0
            }
        }
    }

    public func setColliders(_ colliders: [PBDCollider]) {
        assert(!colliders.contains { $0.meta.x == .max },
               "PaperClothSolver: a collider with ownerID == .max is skipped by the collide "
               + "kernel and will do nothing. Tag colliders with any other id.")
        let clipped = colliders.count > colliderBuffer.capacity
            ? Array(colliders.prefix(colliderBuffer.capacity)) : colliders
        colliderBuffer.write(clipped)
        colliderCount = clipped.count
    }

    private func encodeMeshPack(_ cb: MTLCommandBuffer) {
        let M = verticesPerSheet
        let pStride = MemoryLayout<PBDParticle>.stride
        for s in 0..<sheetCount {
            let baseOffset = s * M * pStride
            // Positions
            if let enc = cb.makeComputeCommandEncoder() {
                enc.label = "PaperCloth.writePos[\(s)]"
                enc.setComputePipelineState(writePosPipeline)
                enc.setBuffer(particleBuffer.buffer, offset: baseOffset, index: 0)
                enc.setBuffer(positionBuffers[s], offset: 0, index: 1)
                enc.setBuffer(meshUniformBuffer, offset: 0, index: 2)
                dispatch1D(enc, pipeline: writePosPipeline, count: M)
                enc.endEncoding()
            }
            // Normals
            if let enc = cb.makeComputeCommandEncoder() {
                enc.label = "PaperCloth.normals[\(s)]"
                enc.setComputePipelineState(normalsPipeline)
                enc.setBuffer(particleBuffer.buffer, offset: baseOffset, index: 0)
                enc.setBuffer(normalBuffers[s], offset: 0, index: 1)
                enc.setBuffer(meshUniformBuffer, offset: 0, index: 2)
                dispatch1D(enc, pipeline: normalsPipeline, count: M)
                enc.endEncoding()
            }
        }
    }

    private func writeUniforms(dt: Float) {
        let uPtr = uniformBuffer.contents().bindMemory(to: PBDUniforms.self, capacity: 1)
        uPtr.pointee = PBDUniforms(
            dt: dt, gravity: gravity, damping: damping,
            floorY: floorEnabled ? floorY : -1e9,
            particleCount: UInt32(particleBuffer.count),
            constraintCount: UInt32(constraintBuffer.count),
            floorBand: 0.01, colliderCount: UInt32(colliderCount), ownerID: .max,
            collideStiffness: collideStiffness, collideFriction: collideFriction,
            selfRadius: clothThickness,
            velRestitution: 0.2, floorRestitution: 0.1,
            maxSDFPush: sdfMaxPushPerStep)
    }

    private func writeWindUniforms(dt: Float) {
        let wPtr = windUniformBuffer.contents().bindMemory(to: PaperWindUniforms.self, capacity: 1)
        wPtr.pointee = PaperWindUniforms(
            particleCount: UInt32(particleBuffer.count),
            verticesPerSheet: UInt32(verticesPerSheet),
            gridW: UInt32(gridW), gridH: UInt32(gridH),
            dt: dt, time: time,
            windAmp: windAmp, windFreq: windFreq, windScroll: windScroll,
            windDirX: windDir.x, windDirZ: windDir.y,
            turb: turb, aero: aero, drag: drag,
            mouthX: mouth.x, mouthY: mouth.y, mouthZ: mouth.z,
            jetHeight: jetHeight, jetRadius: jetRadius, jetUpdraft: jetUpdraft,
            wallRadius: wallRadius)
    }

    // ── Encoders ────────────────────────────────────────────────────────

    private func encodePass(_ cb: MTLCommandBuffer, pipeline: MTLComputePipelineState,
                            buffers: [MTLBuffer], count: Int, label: String) {
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        dispatch1D(enc, pipeline: pipeline, count: count)
        enc.endEncoding()
    }

    private func encodeConstraintSubpass(_ cb: MTLCommandBuffer,
                                         start: Int, count: Int, label: String) {
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(constraintPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(constraintBuffer.buffer,
                      offset: start * MemoryLayout<PBDConstraint>.stride, index: 1)
        enc.setBuffer(uniformBuffer, offset: 0, index: 2)
        enc.setBuffer(lambdaBuffer, offset: start * MemoryLayout<Float>.stride, index: 3)
        dispatch1D(enc, pipeline: constraintPipeline, count: count)
        enc.endEncoding()
    }

    private func encodeBendSubpass(_ cb: MTLCommandBuffer,
                                   start: Int, count: Int, label: String) {
        guard count > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(bendPipeline)
        enc.setBuffer(particleBuffer.buffer, offset: 0, index: 0)
        enc.setBuffer(bendConstraintBuffer.buffer,
                      offset: start * MemoryLayout<PaperBendConstraint>.stride, index: 1)
        enc.setBuffer(bendUniformBuffer, offset: 0, index: 2)
        enc.setBuffer(bendLambdaBuffer, offset: start * MemoryLayout<Float>.stride, index: 3)
        dispatch1D(enc, pipeline: bendPipeline, count: count)
        enc.endEncoding()
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState, count: Int) {
        let w = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
    }
}

// ── PaperBendConstraint mirror ───────────────────────────────────────────────
//
// Keep field order IDENTICAL to `struct PaperBendConstraint` in PaperCloth.metal.
// (uint4 ↔ SIMD4<UInt32>, float2 ↔ SIMD2<Float> — 32 bytes, no bare float3.)
struct PaperBendConstraint {
    /// x,y = the hinge's shared-edge vertices; z,w = the two wing vertices.
    var v: SIMD4<UInt32>
    /// Dihedral angle at rest — π for cloth cut flat.
    var restAngle: Float
    /// XPBD α: 0 = rigid (holds a crease), large = soft.
    var compliance: Float
    var _pad: SIMD2<Float> = .zero
}

// ── PaperBendUniforms mirror ─────────────────────────────────────────────────
//
// Keep field order IDENTICAL to `struct PaperBendUniforms` in PaperCloth.metal.
struct PaperBendUniforms {
    var constraintCount: UInt32
    var dt: Float
    var yieldAngle: Float
    var creepRate: Float
}

// ── PaperStickUniforms mirror ────────────────────────────────────────────────
//
// Keep field order IDENTICAL to `struct PaperStickUniforms` in PaperCloth.metal.
struct PaperStickUniforms {
    var particleCount: UInt32
    var colliderCount: UInt32
    var dt: Float
    var floorY: Float
    var contactBand: Float
    var stickSpeed: Float
    var selfRadius: Float
    var _pad: Float = 0
}

// ── PaperMeshUniforms mirror ─────────────────────────────────────────────────
//
// Keep field order IDENTICAL to `struct PaperMeshUniforms` in PaperCloth.metal.
struct PaperMeshUniforms {
    var vertexCount: UInt32
    var gridW: UInt32
    var gridH: UInt32
    var _pad: UInt32 = 0
}

// ── PaperWindUniforms mirror ─────────────────────────────────────────────────
//
// Keep field order IDENTICAL to `struct PaperWindUniforms` in PaperCloth.metal.
struct PaperWindUniforms {
    var particleCount: UInt32
    var verticesPerSheet: UInt32
    var gridW: UInt32
    var gridH: UInt32
    var dt: Float
    var time: Float
    var windAmp: Float
    var windFreq: Float
    var windScroll: Float
    var windDirX: Float
    var windDirZ: Float
    var turb: Float
    var aero: Float
    var drag: Float
    var mouthX: Float
    var mouthY: Float
    var mouthZ: Float
    var jetHeight: Float
    var jetRadius: Float
    var jetUpdraft: Float
    var wallRadius: Float
    var _pad1: Float = 0
}

// ── PaperHashUniforms mirror ─────────────────────────────────────────────────
//
// Keep field order IDENTICAL to `struct PaperHashUniforms` in PaperCloth.metal.
struct PaperHashUniforms {
    var particleCount: UInt32
    var tableSize: UInt32
    var cellSize: Float
    var radius: Float
    var gridW: UInt32
    var verticesPerSheet: UInt32
    var skipRadius: UInt32
    var legacy: UInt32 = 0
}
