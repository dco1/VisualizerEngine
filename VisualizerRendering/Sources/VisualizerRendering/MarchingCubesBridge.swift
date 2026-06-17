import Foundation
import Metal
import OSLog
import SceneKit
import simd
import VisualizerCore

// ── MARCHING CUBES BRIDGE ────────────────────────────────────────────────────
//
// Bridges a scalar density field (e.g. MLSMPMSolver.gridMassBuffer) to an
// SCNGeometry that renders as the corresponding isosurface. Same zero-copy
// contract as DynamicMesh: SceneKit's render thread reads the same MTLBuffers
// the MC kernel writes into.
//
// SHAPE — variable topology
// ──────────────────────────
// Each frame the surface produces a different triangle count. To avoid a
// per-frame SCNGeometryElement rebuild OR a CPU↔GPU sync on an atomic vertex
// counter, the kernel writes into PER-CELL slots:
//
//     vertex slot range for cell `(cx, cy, cz)` =
//         (cx + cellsX * (cy + cellsY * cz)) * 15  ..  + 15
//
// Each cell writes exactly 15 vertices (5 triangles × 3 verts) regardless of
// case. Unused slots are filled with degenerate triangles (3 verts at the
// same point) which the rasterizer culls. SCNGeometryElement's primitive
// count is therefore fixed at (cells × 5) and never has to be rebuilt.
//
// PARAMETERISATION
// ────────────────
// The bridge knows the density-field point resolution (= densityResolution),
// cell size, and grid origin in world space. The "cell" count for MC is
// `densityResolution - 1` along each axis: an N×N×N density field has
// (N-1)³ cells (each cell has 8 corners from the field).
//
// USAGE
// ─────
//     let bridge = MarchingCubesBridge(solver: fluidSolver)!
//     scene.rootNode.addChildNode(bridge.node)
//
//     // each frame, alongside the solver's own encode:
//     solver.encode(to: cb, wallDt: dt)
//     bridge.encode(to: cb, densityField: solver.gridMassBuffer, isovalue: 3.5)

@MainActor
public final class MarchingCubesBridge {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                    category: "MarchingCubesBridge")

    public let device: MTLDevice
    public let node: SCNNode

    // ── Compute resources ────────────────────────────────────────────────────

    private let mcPipeline: MTLComputePipelineState
    private let smoothPipeline: MTLComputePipelineState
    /// Wide uncapped Gaussian "fuse" pre-pass — distinct from `smoothPipeline`
    /// (which is mass-capped to one cell). Lazily looked up the first time a
    /// non-zero `densityFuseRadiusCells` is set so water scenes that never fuse
    /// don't pay the lookup. See `densityFuseRadiusCells`.
    private var fusePipeline: MTLComputePipelineState?

    /// Lorensen-Cline triangle table, 256 × 16 Int32. Uploaded once.
    private let triTableBuffer: MTLBuffer

    /// Per-cell vertex slot buffer — `cellCount * 15` packed_float3 entries.
    /// Aliased into the SCNGeometry as the position source.
    ///
    /// This is the READ buffer (A). SceneKit's SCNGeometrySource and
    /// Illuminatorama's repack task hold a permanent reference to this buffer.
    /// The MC kernel never writes here directly — it writes into
    /// `positionBufferWrite` (B) and the encode method blits B→A at the tail
    /// of every command buffer, so SceneKit/Illuminatorama always read a
    /// completed frame rather than a partially-written one.
    ///
    /// Exposed publicly so downstream GPU passes (e.g. CausticsRT) can read
    /// the live surface mesh without rebuilding their own copy. Read-only
    /// outside the bridge's encode path.
    public let positionBuffer: MTLBuffer

    /// Write target for the MC kernel — the B side of the double-buffer pair.
    /// Kernel writes here each frame; encode blits this into `positionBuffer`
    /// (A) before the command buffer commits. Private: callers never need it.
    private let positionBufferWrite: MTLBuffer

    /// Same shape — aliased as the normal source. Public for the same
    /// reason: CausticsRT samples interpolated normals at barycentric points.
    /// This is the READ buffer (A); `normalBufferWrite` (B) is the write target.
    public let normalBuffer: MTLBuffer

    /// Write target for MC normals — the B side. Blitted into `normalBuffer`
    /// (A) at the tail of every encode call.
    private let normalBufferWrite: MTLBuffer

    /// Total vertex slot count — `cellCount * 15`. Public so downstream
    /// passes that dispatch over the surface buffer (e.g. CausticsRT's
    /// compact pass, sized as `vertexCount / 3` triangle slots) can size
    /// their grid correctly.
    public var vertexCount: Int { maxVertexCount }

    /// Identity index buffer (0, 1, 2, …, maxVerts-1). Built once at init.
    /// Required because SCNGeometryElement has no non-indexed primitive
    /// constructor — but using identity indices means we still amount to
    /// a non-indexed draw on the GPU side.
    private let indexBuffer: MTLBuffer

    /// Ping-pong buffers for the density smoothing pre-pass. Sized to the
    /// density field (`densityResolution³` floats). Two buffers so we can
    /// run N iterations by alternating reads + writes. With smoothingIterations
    /// == 0 these are never touched and MC reads the source directly.
    private let smoothedBufferA: MTLBuffer
    private let smoothedBufferB: MTLBuffer

    /// Scratch target for the wide-Gaussian fuse pre-pass (same size as the
    /// density field). Only written when `densityFuseRadiusCells > 0`.
    private let fusedBuffer: MTLBuffer

    /// How many smoothing iterations to run before MC. Mutable so the scene
    /// controller can drive it from a slider.
    public var smoothingIterations: Int = 2

    /// Cylindrical density clip radius (world metres; 0 = disabled), axis
    /// vertical through the grid's XZ centre. For fluid held in a round
    /// vessel: the density field smears past any wall thinner than the
    /// P2G + fuse + smooth spread (~4 cells), so the isosurface pokes through
    /// solid vessel geometry. Set this to the vessel's interior radius and
    /// the MC kernel closes the surface exactly at the wall — matching where
    /// the colliders already stop the particles.
    public var clipCylinderRadius: Float = 0

    /// PER-INSTANCE wide-Gaussian density-fuse radius, in CELLS. Default 0 =
    /// OFF (water scenes — FluidTest, Hotdog Waterslide — are unchanged).
    ///
    /// When > 0, a single UNCAPPED Gaussian blur of this radius runs BEFORE the
    /// (optional) mass-capped smoothing pass + MC. Unlike `smoothingIterations`
    /// (which is mass-capped to one cell and so preserves per-clump density
    /// maxima), this pass lets mass genuinely migrate between neighbouring
    /// particle clumps so they fuse into ONE smooth field instead of a "string
    /// of pearls" of metaball lobes. Set it to ≈1.6–2.0× the particle spacing
    /// in cells for a low-particle-count PASTE (e.g. guac); then raise the
    /// isovalue to compensate for the spread so the surface doesn't balloon.
    ///
    /// Clamped to [0, 3] in the kernel (7³ tap budget). This is a per-bridge
    /// property, NOT a shared global — it lives on the instance so the guac can
    /// widen its kernel without regressing the shared water solver path.
    public var densityFuseRadiusCells: Int = 0

    /// Current water-tint RGB driving the shader's `u_tintR/G/B` uniforms.
    /// Defaults to the previous hardcoded teal. Mutated via `setWaterTint`,
    /// and re-applied automatically whenever `setSurfaceStyle` swaps the
    /// materials so the new pair inherits the same colour.
    private var currentTint: SIMD3<Float> = SIMD3<Float>(0.12, 0.32, 0.42)

    // ── Diagnostic material factories ────────────────────────────────────────
    //
    // Built into the bridge (not the scene controller) so any FluidTest-style
    // scene can flip into a diagnostic style without duplicating the recipe.
    // Replace the surface node's material with one of these to investigate
    // geometry issues:
    //
    //   defaultWaterMaterial  — production translucent water (the one used at
    //                           init time).
    //   opaqueSolidMaterial   — flat-shaded opaque blue. Hides nothing; any
    //                           hole in the surface shows the background
    //                           directly, any wrong-winding triangle reads
    //                           as the same colour (no front/back trick).
    //   wireframeMaterial     — opaque blue with `fillMode = .lines`. Surface
    //                           triangulation is directly visible — degenerate
    //                           slots collapsed to a point, holes are obvious
    //                           gaps in the wire grid.
    //
    // All three respect a `cullBack` toggle: when true, sets
    // `isDoubleSided = false` so only triangles SceneKit considers front-
    // facing draw. Anything that disappears under cullBack is wound backwards
    // relative to SceneKit's CW-front convention.

    /// Return the (front, back) material pair for a style. With two
    /// geometry elements (one cull-back, one cull-front), each material's
    /// shader sees genuinely front-facing or back-facing fragments — so
    /// the front/back colours actually differ in the rendered image.
    ///
    /// For .water and .fishTank the two materials are CLONES of the same
    /// shader (single shared recipe, uses `abs(dot(N,V))` so front and
    /// back fragments composite identically) — this eliminates the
    /// front/back element seam where the two shader paths produced
    /// visibly different colours on adjacent fragments.
    ///
    /// `cullBack: true` returns nil for the back material (the controller
    /// then assigns a fully-transparent placeholder) — useful for diagnosis
    /// when you want to see only the production front-side rendering.
    public static func materials(style: MaterialStyle,
                                 cullBack: Bool) -> (SCNMaterial, SCNMaterial) {
        let (front, back): (SCNMaterial, SCNMaterial)
        switch style {
        case .water:
            front = waterUnifiedMaterial()
            back  = waterUnifiedMaterial()
        case .fishTank:
            front = fishTankMaterial()
            back  = fishTankMaterial()
        case .opaque:
            front = diagnosticOpaqueMaterial(front: true)
            back  = diagnosticOpaqueMaterial(front: false)
        case .sauce:
            front = sauceMaterial()
            back  = sauceMaterial()
        case .wireframe:
            front = diagnosticWireframeMaterial(front: true)
            back  = diagnosticWireframeMaterial(front: false)
        case .faceDebug:
            front = diagnosticFaceMaterial(front: true)
            back  = diagnosticFaceMaterial(front: false)
        case .volumeDebug:
            front = diagnosticVolumeMaterial(front: true)
            back  = diagnosticVolumeMaterial(front: false)
        }
        if cullBack {
            // Hide the back element by making its material fully
            // transparent and depth-write-free so it contributes nothing.
            back.transparency = 0.0
            back.writesToDepthBuffer = false
        }
        return (front, back)
    }

    public enum MaterialStyle {
        case water
        case fishTank
        case opaque
        case sauce          // opaque PBR body tinted by `setWaterTint(r:g:b:)`.
                            // Replaces the diagnostic `.opaque` for production
                            // food-sauce surfaces (cheese, chocolate, mustard,
                            // avocado paste). Roughness mid-low so the IBL
                            // produces a soft sheen — reads as a real wet
                            // viscous body, not the diagnostic flat blue.
        case wireframe
        case faceDebug
        case volumeDebug    // red front + green back, BOTH semi-transparent
                            // with depth-write OFF, so the back compositing is
                            // genuinely visible behind the front. Reveals
                            // whether the body reads as volume or shell.
    }

    private let uniformBuffer: MTLBuffer

    // ── Field geometry ───────────────────────────────────────────────────────

    public let densityResolution: SIMD3<UInt32>
    public let cellSize: Float
    public let gridOrigin: SIMD3<Float>

    private let cellResolution: SIMD3<UInt32>
    private let cellCount: Int
    private let maxVertexCount: Int

    // ── Init ─────────────────────────────────────────────────────────────────

    /// Convenience initializer that wires the bridge to an MLSMPMSolver's grid.
    /// The `densityResolution`, `cellSize`, and `gridOrigin` are read from the
    /// solver so the surface stays aligned with the simulation.
    public convenience init?(solver: MLSMPMSolver,
                             material: SCNMaterial? = nil) {
        self.init(engine: SimEngine.shared,
                  densityResolution: solver.gridResolution,
                  cellSize: solver.cellSize,
                  gridOrigin: solver.boundsMin,
                  material: material)
    }

    /// Generic initializer. Use this when the density field comes from
    /// somewhere other than an MLS-MPM solver (a CPU-built SDF, a separate
    /// splat-and-blur pipeline, etc.).
    public init?(engine: SimEngine,
                 densityResolution: SIMD3<UInt32>,
                 cellSize: Float,
                 gridOrigin: SIMD3<Float>,
                 material: SCNMaterial? = nil) {
        let device = engine.device

        guard
            let pipeline = engine.pipeline("marchingCubesExtract"),
            let smoothP  = engine.pipeline("mcSmoothDensity")
        else {
            Self.log.error("MC pipeline lookup failed — check MarchingCubes.metal is in VisualizerRendering/Shaders/")
            return nil
        }

        // (N-1)³ cells of 15 vertex slots each.
        let cellsX = max(1, Int(densityResolution.x) - 1)
        let cellsY = max(1, Int(densityResolution.y) - 1)
        let cellsZ = max(1, Int(densityResolution.z) - 1)
        let cellCount = cellsX * cellsY * cellsZ
        let maxVerts = cellCount * 15

        // Output buffers. Zero-initialised — Metal's makeBuffer guarantees
        // that — so any cell that never gets touched (e.g. boundary cells
        // whose dispatch coords are out of range) draws as a single point
        // at the origin, which is a zero-area triangle, which is invisible.
        //
        // Double-buffered: the MC kernel writes into the *Write (B) buffers;
        // a blit at the end of every encode copies B→A. SceneKit's
        // SCNGeometrySource and Illuminatorama's repack task always hold
        // a reference to the A buffers, so they see a completed frame rather
        // than a partially-written one.
        let stride12 = MemoryLayout<Float>.stride * 3   // packed_float3
        guard
            let posBuf  = device.makeBuffer(length: stride12 * maxVerts,
                                            options: .storageModePrivate),
            let nrmBuf  = device.makeBuffer(length: stride12 * maxVerts,
                                            options: .storageModePrivate),
            let posBufW = device.makeBuffer(length: stride12 * maxVerts,
                                            options: .storageModePrivate),
            let nrmBufW = device.makeBuffer(length: stride12 * maxVerts,
                                            options: .storageModePrivate)
        else {
            Self.log.error("MC output buffer allocation failed (maxVerts = \(maxVerts))")
            return nil
        }
        posBuf.label  = "MC.positions"
        nrmBuf.label  = "MC.normals"
        posBufW.label = "MC.positions.write"
        nrmBufW.label = "MC.normals.write"

        // Identity index buffer.
        let idxStride = MemoryLayout<UInt32>.stride
        guard let idxBuf = device.makeBuffer(length: idxStride * maxVerts,
                                             options: .storageModeShared)
        else {
            Self.log.error("MC index buffer allocation failed")
            return nil
        }
        idxBuf.label = "MC.indices"
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self,
                                                  capacity: maxVerts)
        for i in 0..<maxVerts { idxPtr[i] = UInt32(i) }

        // Tri-table upload (256 × 16 Int32 = 16 KB).
        let triStride = MemoryLayout<Int32>.stride
        guard let triBuf = MarchingCubesTables.triTable
                .withUnsafeBufferPointer({ ptr -> MTLBuffer? in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: triStride * ptr.count,
                              options: .storageModeShared)
        }) else {
            Self.log.error("MC triTable buffer allocation failed")
            return nil
        }
        triBuf.label = "MC.triTable"

        // Uniforms (shared so we can rewrite isovalue per frame cheaply).
        guard let uBuf = device.makeBuffer(length: MemoryLayout<MCUniforms>.stride,
                                           options: .storageModeShared)
        else {
            Self.log.error("MC uniform buffer allocation failed")
            return nil
        }
        uBuf.label = "MC.uniforms"

        // Smoothing ping-pong buffers, same size as the source density field.
        let densityCount = Int(densityResolution.x) * Int(densityResolution.y) * Int(densityResolution.z)
        guard
            let sA = device.makeBuffer(length: MemoryLayout<Float>.stride * densityCount,
                                       options: .storageModePrivate),
            let sB = device.makeBuffer(length: MemoryLayout<Float>.stride * densityCount,
                                       options: .storageModePrivate),
            let sF = device.makeBuffer(length: MemoryLayout<Float>.stride * densityCount,
                                       options: .storageModePrivate)
        else {
            Self.log.error("MC smoothing buffer allocation failed")
            return nil
        }
        sA.label = "MC.smoothedA"
        sB.label = "MC.smoothedB"
        sF.label = "MC.fused"

        // ── SCNGeometry wiring ───────────────────────────────────────────
        let posSource = SCNGeometrySource(
            buffer: posBuf,
            vertexFormat: .float3,
            semantic: .vertex,
            vertexCount: maxVerts,
            dataOffset: 0,
            dataStride: stride12
        )
        let nrmSource = SCNGeometrySource(
            buffer: nrmBuf,
            vertexFormat: .float3,
            semantic: .normal,
            vertexCount: maxVerts,
            dataOffset: 0,
            dataStride: stride12
        )
        // TWO elements over the same index buffer. SceneKit draws each
        // element with its own material — element 0 gets the front
        // material (cull-back: only front-winding triangles render),
        // element 1 gets the back material (cull-front: only back-winding
        // triangles render). This is the way to genuinely shade front and
        // back differently in SceneKit: `isDoubleSided = true` auto-flips
        // the surface normal for back-facing rasterizations, so a single
        // shader modifier sees `dot(N, V) > 0` for EVERY fragment and
        // can't distinguish sides. With explicit cull modes the normal
        // isn't flipped — each material's shader sees the actual outward
        // normal and renders accordingly.
        //
        // Vertex shader work doubles (each triangle rasterised twice) but
        // each pass culls half the triangles at the rasteriser, so net
        // fragment work is roughly the same as a single isDoubleSided pass.
        let frontElement = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .triangles,
            primitiveCount: maxVerts / 3,
            bytesPerIndex: idxStride
        )
        let backElement = SCNGeometryElement(
            buffer: idxBuf,
            primitiveType: .triangles,
            primitiveCount: maxVerts / 3,
            bytesPerIndex: idxStride
        )

        // DRAW ORDER — back element FIRST, then front composites OVER it.
        // With `writesToDepthBuffer = false` on both translucent materials,
        // SceneKit draws elements in array order without per-fragment
        // back-to-front sorting. The back element (cull-front, renders
        // away-from-camera triangles) is physically BEHIND the front
        // element (cull-back, renders toward-camera triangles) from the
        // viewer's POV — so back must be drawn first to look right.
        //
        // The earlier order [front, back] put the dark back-fragment
        // colour on top of the brighter front in screen space, which
        // read as "no body" because the back's translucent fragments
        // mostly let the background grid through instead of being
        // composited onto an already-opaque front surface.
        let geom = SCNGeometry(sources: [posSource, nrmSource], // winding-ok: positions + normals are written by the MC compute kernel into MTLBuffers; no CPU-side arrays exist to audit, and the two-element back/front pair is intentionally double-sided for translucent-water compositing
                               elements: [backElement, frontElement])
        // Default to the production water style. `setSurfaceStyle()` can
        // swap to debug variants without rebuilding the geometry. If the
        // caller passed a custom `material:` we honour it as the FRONT
        // and pair with the default back; full custom front+back via the
        // public materials() API.
        let (defaultFront, defaultBack) = Self.materials(style: .water, cullBack: false)
        let frontMat: SCNMaterial
        if let m = material {
            frontMat = m
        } else {
            frontMat = defaultFront
        }
        let backMat = defaultBack
        frontMat.cullMode = .back
        backMat.cullMode  = .front
        // Materials are indexed by ELEMENT position in `geom.elements`.
        // Elements are [back, front], so materials must mirror: [back, front].
        geom.materials = [backMat, frontMat]

        let n = SCNNode(geometry: geom)
        n.name = "MarchingCubesSurface"

        self.device               = device
        self.mcPipeline           = pipeline
        self.smoothPipeline       = smoothP
        self.triTableBuffer       = triBuf
        self.positionBuffer       = posBuf
        self.normalBuffer         = nrmBuf
        self.positionBufferWrite  = posBufW
        self.normalBufferWrite    = nrmBufW
        self.indexBuffer          = idxBuf
        self.smoothedBufferA      = sA
        self.smoothedBufferB      = sB
        self.fusedBuffer          = sF
        self.uniformBuffer        = uBuf
        self.densityResolution = densityResolution
        self.cellSize          = cellSize
        self.gridOrigin        = gridOrigin
        self.cellResolution    = SIMD3<UInt32>(UInt32(cellsX), UInt32(cellsY), UInt32(cellsZ))
        self.cellCount         = cellCount
        self.maxVertexCount    = maxVerts
        self.node              = n

        // Phase 4.13b — publish a GPU mesh descriptor on the geometry so
        // Illuminatorama's scene extractor can build an interleaved
        // `IlluminatoramaVertex` buffer + repack task on first encounter
        // and render the marching-cubes isosurface natively.
        //
        // SHAPE — single element across the whole `maxVerts` slot range
        // (the bridge's two-element SCN trick is a SceneKit-only cull-
        // mode workaround that doesn't apply to Illuminatorama's
        // deferred G-buffer pass; one draw call is enough). Degenerate
        // slots stay zero-area triangles and cull out naturally.
        //
        // Index type — MC's identity index buffer is `UInt32`, so we
        // bridge as `.uint32`. The Phase 4.13a descriptor handles both
        // 16- and 32-bit indices through `bodyIndexType`; the renderer's
        // combined-index blit reads `descriptor.bodyIndexType` to size
        // the copy correctly.
        geom.illuminatoramaGPUMesh = IlluminatoramaGPUMeshDescriptor(
            positionBuffer: posBuf,
            normalBuffer: nrmBuf,
            positionStride: stride12,
            normalStride: stride12,
            vertexCount: maxVerts,
            bodyIndexBuffer: idxBuf,
            bodyIndexCount: maxVerts,
            bodyIndexType: .uint32,
            capIndexBuffer: nil,
            capIndexCount: 0,
            capIndexType: .uint32,
            // The MC fluid surface is open + moves: when it tilts/pours its back
            // side faces the camera. Render two-sided so it doesn't go hollow.
            doubleSided: true
        )
    }

    // ── Encode ───────────────────────────────────────────────────────────────

    /// Encode one MC extraction into the given command buffer. Must be
    /// dispatched AFTER whatever pass produces `densityField` for this
    /// frame (e.g. after MLSMPMSolver's P2G). The bridge does no
    /// synchronisation of its own — the inter-encoder ordering Metal
    /// already guarantees within a single command buffer is enough.
    public func encode(to cb: MTLCommandBuffer,
                       densityField: MTLBuffer,
                       isovalue: Float) {
        // Write uniforms.
        let uPtr = uniformBuffer.contents().bindMemory(to: MCUniforms.self,
                                                       capacity: 1)
        let fuseR = max(0, min(3, densityFuseRadiusCells))
        uPtr.pointee = MCUniforms(
            gridRes:  SIMD4(densityResolution, 0),
            dxOrigin: SIMD4(cellSize, gridOrigin.x, gridOrigin.y, gridOrigin.z),
            isoLevel: SIMD4(isovalue, Float(fuseR), max(0, clipCylinderRadius), 0)
        )

        // ── Wide-Gaussian density FUSE pre-pass (optional) ───────────────
        // When `densityFuseRadiusCells > 0`, spread + merge neighbouring
        // particle clumps into one smooth field BEFORE the (mass-capped)
        // smoothing pass and MC. This is the fix for the "string of pearls"
        // metaball lobing a low-particle-count paste shows: the capped
        // smoothing pass below CANNOT fuse separate clumps (it clamps each
        // cell to its source-neighbourhood max), so a wider uncapped pass has
        // to run first. Water scenes leave the radius at 0 and skip this.
        var sourceForChain = densityField
        if fuseR > 0, let fuseP = ensureFusePipeline() {
            if let enc = cb.makeComputeCommandEncoder() {
                enc.label = "MC.fuseDensity"
                enc.setComputePipelineState(fuseP)
                enc.setBuffer(densityField, offset: 0, index: 0)
                enc.setBuffer(fusedBuffer,  offset: 0, index: 2)
                enc.setBuffer(uniformBuffer, offset: 0, index: 3)
                let dispatch = MTLSize(
                    width:  Int(densityResolution.x),
                    height: Int(densityResolution.y),
                    depth:  Int(densityResolution.z))
                let tile = MTLSize(width: 4, height: 4, depth: 4)
                enc.dispatchThreads(dispatch, threadsPerThreadgroup: tile)
                enc.endEncoding()
                sourceForChain = fusedBuffer
            }
        }

        // ── Density smoothing pre-pass (optional) ────────────────────────
        // Ping-pong N iterations of the 27-tap box blur. The first iteration
        // reads from the caller's `densityField` buffer; subsequent ones
        // alternate between the bridge's two internal buffers. The MC kernel
        // then reads from whichever buffer holds the final result. When
        // smoothingIterations == 0 we skip this entire block and MC reads
        // the source field directly.
        //
        // EVERY iteration also reads from `densityField` as a separate
        // source binding — the kernel uses it to cap each output cell to
        // the max source value in its neighbourhood, preventing the smoothed
        // halo from escaping the genuine fluid volume by more than one cell.
        let iterations = max(0, smoothingIterations)
        let fieldForMC: MTLBuffer
        if iterations == 0 {
            fieldForMC = sourceForChain
        } else {
            var src = sourceForChain
            var dst = smoothedBufferA
            for _ in 0..<iterations {
                // `original` is the FUSED field (or raw density when fuse is
                // off) so the mass-containment cap measures against the field
                // the surface should hug, not the pre-fuse lobed field.
                encodeSmoothPass(into: cb, source: src, original: sourceForChain, target: dst)
                // Alternate. After the first iteration `src` is always one
                // of the two internal buffers; we never write back to the
                // caller's source.
                if src === sourceForChain {
                    src = smoothedBufferA
                    dst = smoothedBufferB
                } else if src === smoothedBufferA {
                    src = smoothedBufferB
                    dst = smoothedBufferA
                } else {
                    src = smoothedBufferA
                    dst = smoothedBufferB
                }
            }
            // Last write target became the next iteration's source — so the
            // FINAL written buffer is what `src` was set to after the last
            // swap. Two iters: A then B → final = B. One iter: A → final = A.
            fieldForMC = src
        }

        // ── MC extract ───────────────────────────────────────────────────
        // Write into the B (write) buffers. SceneKit's SCNGeometrySource and
        // Illuminatorama's repack task always bind to the A (read) buffers;
        // the blit below copies B→A within this CB so they see a complete
        // frame rather than partially-written data. One-frame lag is
        // intentional and invisible above ~10 fps.
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MC.extract"
        enc.setComputePipelineState(mcPipeline)
        enc.setBuffer(fieldForMC,           offset: 0, index: 0)
        enc.setBuffer(positionBufferWrite,  offset: 0, index: 1)
        enc.setBuffer(normalBufferWrite,    offset: 0, index: 2)
        enc.setBuffer(triTableBuffer,       offset: 0, index: 3)
        enc.setBuffer(uniformBuffer,        offset: 0, index: 4)

        // 3-D dispatch over the (N-1)³ cell grid. The kernel early-exits if
        // any coord is at the high edge, but laying the dispatch out as cell
        // resolution exactly means no wasted threads.
        let dispatch = MTLSize(
            width:  Int(cellResolution.x),
            height: Int(cellResolution.y),
            depth:  Int(cellResolution.z)
        )
        // Modest tile — the kernel touches 8 corners + 6 neighbours for
        // gradients, so locality is limited; 4³ keeps register pressure sane.
        let tile = MTLSize(width: 4, height: 4, depth: 4)
        enc.dispatchThreads(dispatch, threadsPerThreadgroup: tile)
        enc.endEncoding()

        // ── B→A blit ─────────────────────────────────────────────────────
        // Copy the freshly-written B buffers into the stable A buffers that
        // SceneKit/Illuminatorama hold. Within a single command buffer Metal
        // guarantees the blit runs after the compute pass above, so A always
        // receives a complete, not-in-flight frame of MC vertices.
        if let blit = cb.makeBlitCommandEncoder() {
            blit.label = "MC.copyToReadBuffers"
            blit.copy(from: positionBufferWrite, sourceOffset: 0,
                      to: positionBuffer, destinationOffset: 0,
                      size: positionBufferWrite.length)
            blit.copy(from: normalBufferWrite, sourceOffset: 0,
                      to: normalBuffer, destinationOffset: 0,
                      size: normalBufferWrite.length)
            blit.endEncoding()
        }
    }

    /// Lazily resolve the wide-Gaussian fuse pipeline the first time it's
    /// needed. Cached on the instance. Returns nil (and logs once) if the
    /// kernel can't be found — the encode path then just skips the fuse pass
    /// and falls back to the un-fused field, which is the pre-existing
    /// behaviour, so a missing kernel degrades gracefully rather than crashing.
    private func ensureFusePipeline() -> MTLComputePipelineState? {
        if let p = fusePipeline { return p }
        guard let p = SimEngine.shared.pipeline("mcFuseDensity") else {  // gpu-ok: setup-time pipeline lookup, cached on instance
            Self.log.error("MC fuse pipeline lookup failed — density fuse disabled")
            return nil
        }
        fusePipeline = p
        return p
    }

    /// One iteration of the density smoothing pre-pass.
    /// `source` is the input being smoothed this iteration (alternates each
    /// pass). `original` is the ORIGINAL un-smoothed density field, passed
    /// every iteration so the kernel can cap mass diffusion to within one
    /// cell of where the source genuinely had mass.
    private func encodeSmoothPass(into cb: MTLCommandBuffer,
                                  source: MTLBuffer,
                                  original: MTLBuffer,
                                  target: MTLBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MC.smoothDensity"
        enc.setComputePipelineState(smoothPipeline)
        enc.setBuffer(source,         offset: 0, index: 0)
        enc.setBuffer(original,       offset: 0, index: 1)
        enc.setBuffer(target,         offset: 0, index: 2)
        enc.setBuffer(uniformBuffer,  offset: 0, index: 3)
        let dispatch = MTLSize(
            width:  Int(densityResolution.x),
            height: Int(densityResolution.y),
            depth:  Int(densityResolution.z)
        )
        let tile = MTLSize(width: 4, height: 4, depth: 4)
        enc.dispatchThreads(dispatch, threadsPerThreadgroup: tile)
        enc.endEncoding()
    }

    // ── Default material ─────────────────────────────────────────────────────

    /// Unified water material — used on BOTH the cull-back and cull-front
    /// elements. Single shader recipe that uses `abs(dot(N,V))` so front
    /// and back fragments composite identically. Eliminates the visible
    /// front/back element seam the previous split-shader approach produced
    /// (`/tmp/fluid-night-B.png` showed a triangular pink area where only
    /// the back drew and a clear grey area where the front overdrew).
    ///
    /// Visual recipe: analytic refraction of the view ray through the
    /// water surface, Beer-Lambert absorption along the refracted ray to
    /// the pool floor, procedural floor grid sampled at the refracted hit
    /// point. View-angle-independent — same pixel reads the same colour
    /// regardless of camera position.
    private static func waterUnifiedMaterial() -> SCNMaterial {
        let m = waterBaseMaterial()
        // Real Beer-Lambert absorption based on per-fragment water depth.
        //
        // Each front-face fragment is a point on the upper surface of the
        // water; the thickness of water BELOW that point down to the floor
        // is `worldY - floorY`. Beer's law gives transmittance through that
        // column as `exp(-absorption * thickness)`. Opacity = `1 -
        // transmittance` — thin water is mostly transparent (the floor
        // grid shows through unattenuated), thick water saturates the
        // tint and fully occludes.
        //
        // UNIFORM BINDING — IMPORTANT: per the working pattern in
        // GummyBearPlusGeometry / StreetLampGeometry, `#pragma arguments`
        // uniforms must be (1) declared with bare names (no `u_` prefix),
        // (2) `setValue(forKey:)` called BEFORE assigning
        // `material.shaderModifiers`, not after. Earlier attempt used
        // `u_tintR/G/B` set AFTER shaderModifiers — the uniforms never
        // reached the shader and the water rendered as garbage white.
        // Beer-Lambert thickness — `_surface.position` is in WORLD
        // space in a .surface modifier (per Apple's SceneKit docs and
        // the shader_modifier_uniform_types memory). Thickness =
        // worldY - floorY = metres of water above the pool floor at
        // this fragment.
        //
        // Transmittance = exp(-absorption * thickness) → alpha
        // = 1 - transmittance. Thin water leaves alpha low so the
        // background grid shows through tinted; thick water saturates
        // the tint and occludes the floor.
        //
        // Why not u_inverseViewTransform — attempting `worldPos = u_iv *
        // float4(_surface.position, 1)` in a .surface modifier triggers
        // the magenta shader-compile fallback on this SDK. The world-
        // space position is already in _surface.position; the matrix
        // multiply is unnecessary AND broken here.
        // Uniform tint + uniform alpha across the whole surface.
        // The previous Beer-Lambert formula computed thickness as
        // `_surface.position.y - floorY` — the fragment's height above
        // the floor — which is physically meaningful for a flat pool's
        // top face but inverts the read for a 3D body (the bottom of a
        // falling cube went transparent, sides faded from solid-top to
        // transparent-bottom, the body read as a melting candle rather
        // than a coloured solid). True depth-of-medium absorption needs
        // either a back-face depth pre-pass or volumetric raymarching;
        // both are real engineering investments.
        //
        // For now: drive opacity by a slider-controlled `surfaceOpacity`
        // uniform that doesn't vary across the surface. The whole MC
        // shell reads as the same tinted shape, top to bottom. The
        // `absorption` and `floorY` uniforms are kept for forward
        // compatibility — when proper depth-absorption is wired up,
        // they'll come back into play.
        // ANALYTIC REFRACTION + BEER-LAMBERT through known floor plane.
        //
        // Each front-face fragment is on the upper surface of the water.
        // We compute the view ray entering the water (camera→fragment),
        // refract it through the surface normal via Snell's law (water
        // IOR 1.33 → eta 0.752), and intersect the refracted ray with
        // the pool floor plane (y = floorY). The hit point's XZ feeds
        // a procedural grid sample — this IS the floor as seen through
        // the water, distorted by refraction.
        //
        // True Beer-Lambert: thickness = LENGTH of the refracted ray
        // from surface to floor. Unlike `worldY - floorY`, this is
        // direction-aware — looking down through a 30 cm pool gives
        // thickness ≈ 30 cm, looking sideways through the same pool
        // can give thickness > 30 cm (longer path). Pixel reads the
        // same regardless of view angle. transmittance = exp(-k*t).
        //
        // Final colour = refracted_floor_through_water + sky_reflection_at_surface
        // where transmittance modulates the floor contribution and the
        // PBR pass adds sky spec on top via IBL.
        //
        // Uniforms (bare names, set BEFORE shaderModifiers — see other
        // shader memory):
        //   tintR/G/B       water tint
        //   floorY          world Y of the pool floor (default 0)
        //   absorption      per-metre extinction (default 2.5)
        //   gridCellSize    procedural floor grid cell (default 0.375)
        //   gridMinXZ_X     procedural floor XZ origin so we mod with it
        //   gridMinXZ_Z     (kept as two scalars since vector uniforms
        //                    don't bridge reliably)
        // Unified shader. Uses `abs(dot(N,V))` style logic: we flip the
        // normal so it always points TOWARD the camera, then run the
        // exact same Snell + Beer-Lambert math regardless of whether
        // this is a front or back element fragment. The two elements
        // get identical results at every overlapping pixel so the
        // previous seam — where the back triangle drew dark and the
        // front triangle drew light at the same screen position — is
        // gone.
        //
        // NaN safety: refract() can produce non-finite components at
        // grazing angles where MC normals are noisy. We guard with
        // length_squared and isfinite and fall back to straight-down
        // when the refraction is unstable, so the magenta pinpricks
        // at MC corner cells stop appearing.
        let modifier = """
        #pragma arguments
        float tintR;
        float tintG;
        float tintB;
        float floorY;
        float absorption;
        float gridCellSize;
        float gridMinXZ_X;
        float gridMinXZ_Z;
        #pragma body
        float3 tint = float3(tintR, tintG, tintB);
        float3 pos = _surface.position;
        float3 V = normalize(_surface.view);
        float3 N = normalize(_surface.normal);
        // Flip normal toward the camera so front and back fragments
        // refract the view ray the same way. abs(dot(N,V)) trick.
        if (dot(N, V) < 0.0) { N = -N; }
        float3 incident = -V;
        float3 refracted = refract(incident, N, 0.752);
        // NaN / TIR guard — fall back to straight-down so the pixel
        // still resolves to a sensible floor sample instead of a
        // magenta shader-error fragment.
        if (length_squared(refracted) < 1e-6 ||
            !isfinite(refracted.x) || !isfinite(refracted.y) || !isfinite(refracted.z)) {
            refracted = float3(0.0, -1.0, 0.0);
        }
        float denom = refracted.y;
        float t = (denom < -1e-4) ? (floorY - pos.y) / denom : 6.0;
        t = clamp(t, 0.0, 6.0);
        float3 hit = pos + refracted * t;
        float2 gridUV = float2(hit.x - gridMinXZ_X, hit.z - gridMinXZ_Z) / gridCellSize;
        float2 cellPos = fract(gridUV);
        float lineW = 0.06;
        bool onLine = (cellPos.x < lineW) || (cellPos.x > (1.0 - lineW))
                   || (cellPos.y < lineW) || (cellPos.y > (1.0 - lineW));
        float3 floorBase = float3(0.55, 0.63, 0.71);
        float3 floorLine = float3(0.24, 0.31, 0.39);
        float3 refractedFloor = onLine ? floorLine : floorBase;
        float transmittance = exp(-absorption * t);
        float3 throughWater = refractedFloor * transmittance + tint * (1.0 - transmittance);
        _surface.diffuse.rgb = throughWater;
        _surface.diffuse.a = 0.95;
        """
        m.setValue(NSNumber(value: Float(0.9)),  forKey: "tintR")
        m.setValue(NSNumber(value: Float(0.15)), forKey: "tintG")
        m.setValue(NSNumber(value: Float(0.15)), forKey: "tintB")
        m.setValue(NSNumber(value: Float(0.0)),  forKey: "floorY")
        m.setValue(NSNumber(value: Float(2.5)),  forKey: "absorption")
        m.setValue(NSNumber(value: Float(0.375)), forKey: "gridCellSize")
        m.setValue(NSNumber(value: Float(-3.75)), forKey: "gridMinXZ_X")
        m.setValue(NSNumber(value: Float(-3.75)), forKey: "gridMinXZ_Z")
        m.shaderModifiers = [.surface: modifier]
        return m
    }

    /// Fish Tank material — the "AAA" continuous-body look. Renders the
    /// entire MC mesh as one unified tinted volume: surface + interior
    /// composite to a single colour rather than a translucent shell with
    /// a separate particle cloud inside.
    ///
    /// Recipe (single shader on both elements, abs(dot(N,V)) so front
    /// and back fragments produce the same image):
    ///
    ///   1. Refract the view ray at the surface (Snell, IOR 1.33).
    ///   2. Cast the refracted ray to the floor plane and to the box
    ///      side walls; take the nearer hit as the body exit point.
    ///      Thickness = distance from surface to exit. View-direction-
    ///      independent.
    ///   3. Beer-Lambert through that thickness produces a tinted body
    ///      colour — deep parts saturate to the tint, thin edges let
    ///      the refracted scene show through.
    ///   4. Fresnel-weighted IBL specular on top (PBR `.physicallyBased`
    ///      base material) — the high-frequency surface highlight that
    ///      sells "this is real water," with the unified body underneath
    ///      providing the volumetric mass.
    ///
    /// The shader writes `_surface.diffuse.rgb` for the body colour and
    /// leaves PBR to add the IBL spec. `_surface.metalness=0` +
    /// `_surface.roughness=0.05` give a crisp sky reflection through
    /// SceneKit's standard PBR Fresnel — which IS the surface highlight
    /// that makes the "single continuous body" read as water rather than
    /// as a tinted blob.
    private static func fishTankMaterial() -> SCNMaterial {
        let m = SCNMaterial()
        // SOLID OPAQUE water — the user's diagnostic snapshot proved
        // that two-layer translucent compositing cannot fake actual
        // water volume (the MC mesh is a 2D shell with no rendering
        // BETWEEN the front and back faces, so you see straight through
        // to the ball/floor). Real water in a fish tank absorbs almost
        // all light within ~10 cm of saturated tint — the body IS a
        // solid colored mass to the eye.
        //
        // This material renders the MC surface OPAQUE with depth-write
        // ON, so the water becomes a solid coloured object that:
        //  • occludes anything it covers (ball, floor) — what a real
        //    body of water does
        //  • shows the body shape (top surface, sides, splash crowns)
        //    via the MC mesh's silhouette
        //  • tints by view-ray-to-floor distance so deeper parts read
        //    darker / more saturated (subtle volume-feel without true
        //    volumetric rendering)
        //  • has a Fresnel-weighted sky reflection on top for the
        //    glossy wet surface highlight
        //
        // The ball is hidden where the water covers it (correct — you
        // shouldn't see through dense red liquid), visible where it
        // pokes above the surface (the water mass has been displaced).
        m.lightingModel = .constant
        m.diffuse.contents = NSColor.black
        m.transparency = 1.0
        m.transparencyMode = .singleLayer
        // CRITICAL — isDoubleSided = true so the body looks SOLID from
        // below the surface too. The MC mesh doesn't always close the
        // body cleanly at the grid boundary (sides of the water column
        // touching the tank walls have no MC-generated surface). Without
        // double-sided, looking up at the surface from below the water
        // level shows through to the floor. With double-sided + opaque,
        // the underside of the top face also paints as the body, so the
        // water is a SOLID coloured mass from any angle.
        m.isDoubleSided = true
        m.cullMode = .back         // honoured by per-element override
        // OPAQUE — depth-write ON so the water occludes anything it
        // covers, READ ON so it doesn't draw through opaque objects
        // in front of it (none currently, but defensive).
        m.writesToDepthBuffer = true
        m.readsFromDepthBuffer = true
        let modifier = """
        #pragma arguments
        float tintR;
        float tintG;
        float tintB;
        float floorY;
        float absorption;
        float gridCellSize;
        float gridMinXZ_X;
        float gridMinXZ_Z;
        float boxMinX;
        float boxMaxX;
        float boxMinZ;
        float boxMaxZ;
        #pragma body
        float3 tint = float3(tintR, tintG, tintB);
        float absorptionLocal = absorption;
        float floorYLocal = floorY;
        float gridCellSizeLocal = gridCellSize;
        float gridMinXZ_X_local = gridMinXZ_X;
        float gridMinXZ_Z_local = gridMinXZ_Z;
        float boxMinX_local = boxMinX;
        float boxMaxX_local = boxMaxX;
        float boxMinZ_local = boxMinZ;
        float boxMaxZ_local = boxMaxZ;
        float3 pos = _surface.position;
        float3 V = normalize(_surface.view);
        float3 N = normalize(_surface.normal);
        // Camera-facing normal — front + back fragments shade identically.
        if (dot(N, V) < 0.0) { N = -N; }
        float NdotV = saturate(dot(N, V));
        float3 incident = -V;
        float3 refracted = refract(incident, N, 0.752);
        if (length_squared(refracted) < 1e-6 ||
            !isfinite(refracted.x) || !isfinite(refracted.y) || !isfinite(refracted.z)) {
            refracted = float3(0.0, -1.0, 0.0);
        }
        // Body-exit distance: take the NEAREST positive hit between the
        // refracted ray and (floor plane, side walls). This is what makes
        // the body read as a CONTAINED volume rather than a thin shell.
        float tFloor = 1e6;
        if (refracted.y < -1e-4) {
            tFloor = (floorYLocal - pos.y) / refracted.y;
        }
        float tXmin = (refracted.x < -1e-4) ? (boxMinX_local - pos.x) / refracted.x : 1e6;
        float tXmax = (refracted.x >  1e-4) ? (boxMaxX_local - pos.x) / refracted.x : 1e6;
        float tZmin = (refracted.z < -1e-4) ? (boxMinZ_local - pos.z) / refracted.z : 1e6;
        float tZmax = (refracted.z >  1e-4) ? (boxMaxZ_local - pos.z) / refracted.z : 1e6;
        float t = min(min(min(tFloor, tXmin), min(tXmax, tZmin)), tZmax);
        // Min clamp 0.4 m: for BACK-element fragments at the floor of
        // the body, t collapses to 0 (refracted ray hits floor instantly
        // since the fragment IS at the floor). Without this clamp the
        // back layer paints near-pure refractedFloor (light grey) and,
        // because SceneKit's translucent sort order for two same-bbox
        // elements is undefined, it sometimes draws OVER the front-face
        // tint, washing the body to white. Clamping t to 0.4 m means
        // every body fragment composites at least 30% tint (exp(-4.5*0.4)
        // ≈ 0.17), so order-independence is no longer a problem and the
        // body always reads as saturated tinted water.
        t = clamp(t, 0.40, 6.0);
        float3 hit = pos + refracted * t;
        // Procedural floor grid sample at the refracted hit point — only
        // contributes when the hit happens at floor Y (and is unimportant
        // when the exit is a side wall, since the body absorbs almost
        // everything across the longer thickness then).
        float2 gridUV = float2(hit.x - gridMinXZ_X_local, hit.z - gridMinXZ_Z_local) / gridCellSizeLocal;
        float2 cellPos = fract(gridUV);
        float lineW = 0.06;
        bool onLine = (cellPos.x < lineW) || (cellPos.x > (1.0 - lineW))
                   || (cellPos.y < lineW) || (cellPos.y > (1.0 - lineW));
        float3 floorBase = float3(0.55, 0.63, 0.71);
        float3 floorLine = float3(0.24, 0.31, 0.39);
        float3 refractedFloor = onLine ? floorLine : floorBase;
        // OPAQUE body composition. View-aware Beer-Lambert tints the
        // surface by how much water the camera ray traverses to reach
        // the back of the body. Deep parts read as saturated tint; thin
        // edges (splash crowns, ridge near the silhouette) bias toward
        // the refracted-floor sample so the body has a tonal range
        // instead of being flat-colour everywhere.
        //
        // We don't need a per-layer alpha here — the material is opaque
        // and the body is presented as a single solid mass. Tint
        // saturation comes from the Beer-Lambert mix, surface highlight
        // from the Fresnel sky blend below.
        float transmittance = exp(-absorptionLocal * t);
        float3 body = mix(tint, refractedFloor, transmittance);
        // Procedural sky reflection. We can't sample SceneKit's IBL
        // probe from a .constant shader, so synthesise a sky from the
        // reflected view direction. The gradient mirrors makeEnvironmentImage
        // (cool deep blue at the floor → bright cool sky at the zenith)
        // so the reflection reads coherent with the scene backdrop.
        float3 R = reflect(-V, N);
        float skyT = saturate(R.y * 0.5 + 0.5);
        float3 skyLow  = float3(0.10, 0.16, 0.28);
        float3 skyMid  = float3(0.55, 0.72, 0.85);
        float3 skyHigh = float3(0.92, 0.98, 1.15);
        float3 sky;
        if (skyT < 0.55) {
            sky = mix(skyLow, skyMid, skyT / 0.55);
        } else {
            sky = mix(skyMid, skyHigh, (skyT - 0.55) / 0.45);
        }
        // Fake sun disk for a directional anchor on the highlight —
        // exp falloff biased to the upper hemisphere. Without this, the
        // reflection is too diffuse and the surface reads as plastic.
        float3 sunDir = normalize(float3(-0.30, 0.80, -0.20));
        float sunDot = saturate(dot(R, sunDir));
        float sun = pow(sunDot, 64.0) * 2.0;
        sky += float3(1.0, 0.95, 0.85) * sun;
        // Schlick Fresnel for water (F0 ≈ 0.02). Edge fragments reflect
        // most; head-on fragments transmit most. This is THE thing that
        // sells a glossy fluid surface.
        float fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
        float3 final = mix(body, sky, fresnel);
        _surface.diffuse.rgb = final;
        _surface.diffuse.a = 1.0;
        """
        m.setValue(NSNumber(value: Float(0.05)), forKey: "tintR")
        m.setValue(NSNumber(value: Float(0.30)), forKey: "tintG")
        m.setValue(NSNumber(value: Float(0.55)), forKey: "tintB")
        m.setValue(NSNumber(value: Float(0.0)),  forKey: "floorY")
        m.setValue(NSNumber(value: Float(2.5)),  forKey: "absorption")
        m.setValue(NSNumber(value: Float(0.9375)), forKey: "gridCellSize")
        m.setValue(NSNumber(value: Float(-3.75)),  forKey: "gridMinXZ_X")
        m.setValue(NSNumber(value: Float(-3.75)),  forKey: "gridMinXZ_Z")
        m.setValue(NSNumber(value: Float(-1.5)),   forKey: "boxMinX")
        m.setValue(NSNumber(value: Float( 1.5)),   forKey: "boxMaxX")
        m.setValue(NSNumber(value: Float(-1.5)),   forKey: "boxMinZ")
        m.setValue(NSNumber(value: Float( 1.5)),   forKey: "boxMaxZ")
        m.shaderModifiers = [.surface: modifier]
        return m
    }

    /// Shared PBR base for both front and back water materials.
    ///
    /// `metalness = 0`, `roughness = 0.06` is the canonical "smooth dielectric"
    /// pairing — water's IOR (1.33) maps to F0 ≈ 0.02, which is what
    /// SceneKit's PBR uses by default for non-metals. The very low roughness
    /// gives a crisp IBL specular at grazing (the reflection-of-the-sky
    /// look); the front/back materials' shader modifiers handle the diffuse
    /// (Beer-Lambert-flavoured) colour. `fresnelExponent` ONLY affects the
    /// transparent slot's edge fade and doesn't override PBR's internal
    /// Fresnel — it's left at the default.
    private static func waterBaseMaterial() -> SCNMaterial {
        let m = SCNMaterial()
        // .constant — output the shader's `_surface.diffuse.rgb` directly
        // to screen with no light interaction. The water's body colour is
        // a fully-baked composite (refracted floor + Beer-Lambert + tint)
        // computed in the modifier; bringing in directional / ambient /
        // IBL would only ADD brightness on top, pushing saturated tints
        // toward white. Surface highlights can be re-added later via an
        // additive overlay plane (per shader_modifier_overlay_pattern).
        m.lightingModel = .constant
        m.diffuse.contents  = NSColor(deviceRed: 0.04, green: 0.18, blue: 0.28, alpha: 1)
        m.metalness.contents = 0.0
        // 0.55 — diffuses the IBL spec into a soft glow instead of a
        // sharp mirror. At low roughness the cool sky reflection
        // dominates whatever tinted diffuse the modifier writes, washing
        // saturated reds toward white-cyan. 0.55 keeps the surface
        // looking like a fluid (not chalk) while letting the tint read.
        m.roughness.contents = 0.55
        m.transparency = 1.0
        m.transparencyMode = .singleLayer
        // ── Critical: cullMode pinned by element index, NOT by isDoubleSided ──
        // isDoubleSided = false means SceneKit honours cullMode. The bridge
        // assigns one material to a cull-back element (renders fronts) and
        // the other to a cull-front element (renders backs). The cullMode
        // we set here is overridden by the explicit assignment in the
        // materials() factory — but defaults to .back here so a stray
        // single-element use still does something sensible.
        m.isDoubleSided = false
        m.cullMode = .back
        // Don't write depth — multiple translucent layers (front +
        // back element of the same mesh, foam particles, glass box)
        // need to composite by alpha, not z-clip each other.
        // `node.renderingOrder = 50` (set externally) forces the water
        // to draw AFTER the floor so the depth-read isn't fighting
        // the floor's depth.
        m.writesToDepthBuffer = false
        m.readsFromDepthBuffer = true
        return m
    }

    /// Legacy single-material entrypoint, kept so the bridge init can pass
    /// `material: nil` and get a sensible default. Now just returns the
    /// FRONT half of the production pair — the bridge wires both halves
    /// via `materials(style:cullBack:)` anyway.
    private static func defaultWaterMaterial(cullBack: Bool = false) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        // Deep interior colour — set very dark so the surface reads as
        // "looking down into water" by default, with the rim modifier below
        // lifting it toward the lighter rim colour only near the silhouette.
        // The previous values were too light, making the whole surface read
        // as whitish foam at any non-head-on angle.
        m.diffuse.contents  = NSColor(deviceRed: 0.04, green: 0.12, blue: 0.30, alpha: 1)
        m.metalness.contents = 0.0
        m.roughness.contents = 0.08
        // Set to 1 because the shader modifier writes the effective alpha
        // directly to `_surface.diffuse.a`. With transparency < 1 the two
        // values would multiply and stomp on each other — that's what was
        // making the body more transparent than the modifier intended.
        m.transparency = 1.0
        m.transparencyMode = .singleLayer
        m.isDoubleSided = !cullBack
        m.fresnelExponent = 3.0          // sharper specular pickup at edges
        // THIS is what was missing — without it, SceneKit's default depth
        // write rejects the BACK face wherever it's behind the FRONT face
        // in screen space, so the volume of the fluid below the visible
        // surface never gets composited in. With depth-write off, the
        // front and back faces both contribute to the same pixel and we
        // get the actual "you can see deeper water through the surface"
        // look. Depth-READ stays on so any opaque geometry that's in
        // front of the fluid still occludes correctly.
        m.writesToDepthBuffer = false
        m.readsFromDepthBuffer = true

        // ── Fragment shader modifier ──────────────────────────────────────
        // Three cheats that together approximate Beer-Lambert extinction
        // and give a sense of looking through a real fluid volume:
        //
        //  1. Front vs back differentiation via the sign of dot(N, V). Our
        //     normals always point outward (negative gradient of density),
        //     so dot > 0 means we're looking at the front of the surface
        //     and dot < 0 means we're looking at the back (only visible
        //     because the geometry between us and it is also translucent).
        //
        //  2. FRONT faces render semi-transparent so the camera sees
        //     through them to the back wall of the volume below. The deep
        //     blue tints what we see; the rim brightening lifts the edge.
        //
        //  3. BACK faces render almost opaque and EXTRA-DARK. Looking
        //     through the front of a real pool, the back of the water is
        //     where most light has been absorbed traveling through the
        //     fluid column — so it reads as a deep shadow region, which
        //     in turn lets the viewer perceive volume between the bright
        //     surface and the dark back. Without this differentiation,
        //     both faces just blend to "the same colour twice" and the
        //     pool reads as a thin sheet no matter what alpha you pick.
        //
        // Rim is `(1 - NdotV)^4` so the brightening only kicks in at the
        // actual silhouette (NdotV < ~0.3) instead of bleaching the whole
        // front face — without this, any non-perfectly-perpendicular view
        // of a flat pool surface looked uniformly white.
        // NOTE: `_surface.view` is ALREADY the direction from the shaded
        // point toward the camera (per SceneKit's convention, confirmed
        // by every other shader modifier in this codebase). Do NOT negate
        // it — that was the bug that flipped my front/back detection for
        // weeks and made the visible surface render with the dim back-face
        // colour while the actually-hidden back face got the bright front
        // colour. With outward normals and unflipped V:
        //   dot(N, V) > 0  ⇔  surface faces the camera (front)
        //   dot(N, V) < 0  ⇔  surface faces away (back, visible only when
        //                     drawn through a translucent front face)
        let modifier = """
        #pragma body
        float3 N = normalize(_surface.normal);
        float3 V = normalize(_surface.view);
        float NdotV_signed = dot(N, V);
        bool isFront = NdotV_signed > 0.0;
        float NdotV = saturate(abs(NdotV_signed));
        float rim = 1.0 - NdotV;
        rim = rim * rim;                                 // softer falloff
        float3 deep      = float3(0.08, 0.22, 0.50);
        float3 rimColor  = float3(0.60, 0.82, 0.96);
        float3 frontColor = mix(deep, rimColor, rim);
        // Back face = the "far side" of the fluid volume seen through the
        // translucent front face. Multiply front by 0.65, not 0.3 — the
        // previous heavier darkening made the pool's bottom face (visible
        // through the top from above) read as a black shadow tab. 0.65
        // suggests depth without going opaque-black.
        float3 backColor  = frontColor * 0.65;
        _surface.diffuse.rgb = isFront ? frontColor : backColor;
        // FRONT semi-transparent so we see through to the back; BACK nearly
        // opaque so it solidly fills the perceived "depth" of the volume.
        _surface.diffuse.a = isFront ? mix(0.65, 0.35, rim) : 0.92;
        """
        m.shaderModifiers = [.surface: modifier]
        return m
    }

    /// Production opaque-body material for cheese / chocolate / mustard /
    /// avocado-paste scenes. Tinted by `setWaterTint(r:g:b:)` via the same
    /// `tintR/G/B` uniforms the other styles consume; uses SceneKit's
    /// physically-based pipeline so IBL adds a soft sheen + Fresnel rim
    /// (the difference between "this looks like cheese" and "this looks
    /// like opaque plastic"). Mid-low roughness (0.35) for a wet glossy
    /// sauce read; metalness 0 (food is dielectric). Two-sided draw via
    /// `cullMode = .back` per element so the bridge's [back, front]
    /// element ordering produces a proper opaque body.
    private static func sauceMaterial() -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        // Default tint = warm cheese yellow so a fresh material renders
        // sensibly before the host calls setWaterTint(). Overwritten by
        // applyTintToCurrentMaterials() on first push.
        m.diffuse.contents = NSColor(deviceRed: 0.96, green: 0.74, blue: 0.26, alpha: 1)
        m.metalness.contents = 0.0
        m.roughness.contents = 0.35
        m.transparency = 1.0
        m.isDoubleSided = false
        m.cullMode = .back
        m.writesToDepthBuffer = true
        m.readsFromDepthBuffer = true
        // Shader modifier that overrides `_surface.diffuse.rgb` with the
        // host-supplied `tintR/G/B` uniforms. Bare names + setValue BEFORE
        // shaderModifiers (the working pattern documented elsewhere in
        // this file). All the other uniforms (floorY, absorption, …) are
        // declared so applyTintToCurrentMaterials's KVC writes don't log
        // an "unknown key" warning — the modifier ignores them.
        // Modifier writes RGB from the tint AND alpha from a separate
        // `sauceAlpha` uniform. SceneKit's PBR pipeline applies the
        // material's `transparency` property as a SCALING factor on top
        // of the modifier-supplied alpha — values < 1 there get blended
        // into a translucent silhouette. Setting BOTH the modifier
        // alpha and material.transparency gives us a robust dial that
        // doesn't get swallowed by the PBR opacity pipeline.
        // The translucent path (honey, anything with sauceAlpha < 1) gets
        // a "honey-lens" effect: perturb the surface normal AND modulate
        // the alpha with 3D sinusoidal noise based on world position.
        //
        //  • Wavy normal → IBL specular dances across the surface like
        //    sun on real honey
        //  • Wavy alpha → honey appears thick in some spots, thin in
        //    others; the chip behind shows MORE through thin spots and
        //    LESS through thick spots → reads as warped lens distortion
        //
        // This is NOT true Snell refraction (that would require sampling
        // the framebuffer with refracted UVs, which SCNMaterial shader
        // modifiers can't easily do). It IS a convincing approximation
        // — the chip behind doesn't shift position, but its visibility
        // varies wavily across the honey, which the eye reads as
        // optical distortion. Opaque sauces (sauceAlpha == 1) skip the
        // perturbation entirely so cheese stays smooth.
        let modifier = """
        #pragma arguments
        float tintR;
        float tintG;
        float tintB;
        float sauceAlpha;
        float floorY;
        float absorption;
        float gridCellSize;
        float gridMinXZ_X;
        float gridMinXZ_Z;
        float boxMinX;
        float boxMaxX;
        float boxMinZ;
        float boxMaxZ;
        #pragma body
        _surface.diffuse.rgb = float3(tintR, tintG, tintB);
        _surface.diffuse.a = sauceAlpha;

        // Honey-lens distortion. Active only for translucent sauces.
        // Anchored to the surface NORMAL (not world position) so the
        // pattern travels with the honey as it flows — no crawling
        // waves under the surface.
        //
        // Strengths tuned for "looks like real wavy honey" without the
        // alien-cloud effect that the previous (more aggressive) pass
        // produced:
        //   • Normal perturbation: 0.35 (mild specular wiggle)
        //   • Alpha thickness modulation: 0.25 (subtle thick/thin)
        //   • Single-octave noise (one sin per axis) — clean waves,
        //     not bubble-cell noise that read as "alien interior"
        float lensStrength = 1.0 - sauceAlpha;
        float3 N = normalize(_surface.normal);
        float nx = sin(N.x * 7.0 + N.z * 4.0);
        float ny = sin(N.y * 8.0 + N.x * 3.0);
        float nz = sin(N.z * 6.0 + N.y * 5.0);

        // Perturb the normal — spec highlights dance left/right + up.
        float3 nbump = float3(nx, ny * 0.4, nz);
        _surface.normal = normalize(_surface.normal + nbump * lensStrength * 0.35);

        // Vary the alpha by the average so honey thickness reads
        // wavy. ±25 %; subtle enough to read as "lens" not "swiss
        // cheese."
        float thicknessNoise = (nx + ny + nz) / 3.0;
        _surface.diffuse.a = clamp(
            sauceAlpha * (1.0 + 0.25 * thicknessNoise * lensStrength),
            0.05, 1.0);
        """
        m.setValue(NSNumber(value: Float(0.96)), forKey: "tintR")
        m.setValue(NSNumber(value: Float(0.74)), forKey: "tintG")
        m.setValue(NSNumber(value: Float(0.26)), forKey: "tintB")
        m.setValue(NSNumber(value: Float(1.0)),  forKey: "sauceAlpha")
        m.setValue(NSNumber(value: Float(0)),    forKey: "floorY")
        m.setValue(NSNumber(value: Float(0)),    forKey: "absorption")
        m.setValue(NSNumber(value: Float(0.375)),forKey: "gridCellSize")
        m.setValue(NSNumber(value: Float(0)),    forKey: "gridMinXZ_X")
        m.setValue(NSNumber(value: Float(0)),    forKey: "gridMinXZ_Z")
        m.setValue(NSNumber(value: Float(0)),    forKey: "boxMinX")
        m.setValue(NSNumber(value: Float(0)),    forKey: "boxMaxX")
        m.setValue(NSNumber(value: Float(0)),    forKey: "boxMinZ")
        m.setValue(NSNumber(value: Float(0)),    forKey: "boxMaxZ")
        m.shaderModifiers = [.surface: modifier]
        return m
    }

    /// Diagnostic: flat-shaded opaque blue. Two slightly-different shades
    /// for front and back so you can see (in stereo with the wireframe)
    /// which side of the mesh dominates each pixel.
    private static func diagnosticOpaqueMaterial(front: Bool) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .blinn
        m.diffuse.contents = front
            ? NSColor(deviceRed: 0.25, green: 0.55, blue: 0.85, alpha: 1)   // light blue
            : NSColor(deviceRed: 0.10, green: 0.25, blue: 0.50, alpha: 1)   // dark blue
        m.transparency = 1.0
        m.isDoubleSided = false
        m.cullMode = .back   // overridden per-element by the bridge
        return m
    }

    /// Diagnostic: wireframe. Both elements draw their triangle edges; back
    /// element in a slightly dimmer tone so you can tell them apart if
    /// they overlap in screen space.
    private static func diagnosticWireframeMaterial(front: Bool) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = front
            ? NSColor(deviceRed: 0.40, green: 0.80, blue: 1.0, alpha: 1)
            : NSColor(deviceRed: 0.20, green: 0.45, blue: 0.65, alpha: 1)
        m.transparency = 1.0
        m.isDoubleSided = false
        m.cullMode = .back
        return m
    }

    /// Diagnostic: red for the front element, green for the back element.
    /// With the two-element setup, each colour is hard-coded per-element
    /// (no per-fragment dot-product check needed). What you should see:
    ///
    ///   - Visible surface of the pool from outside: RED everywhere
    ///   - Through the translucent red, GREEN visible where the back face
    ///     of the volume sits behind the front
    ///   - At the silhouette: a thin GREEN rim peeking past the front
    ///
    /// If you ever see only red (no green compositing through), the back
    /// element isn't drawing — check depth-write configuration. If you
    /// see only green, the front element is occluding nothing — possibly
    /// transparency mode or render order issue.
    private static func diagnosticFaceMaterial(front: Bool) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = front
            ? NSColor.red
            : NSColor.green
        m.transparency = 1.0
        m.transparencyMode = .singleLayer
        m.isDoubleSided = false
        m.cullMode = .back   // overridden per-element by the bridge
        m.writesToDepthBuffer = false
        m.readsFromDepthBuffer = true
        let modifier = """
        #pragma body
        // Semi-transparent so back element shows THROUGH front; element
        // colour is set on `_surface.diffuse.rgb` via the material's
        // diffuse contents above, so the modifier only needs to set
        // alpha here.
        _surface.diffuse.a = 0.55;
        """
        m.shaderModifiers = [.surface: modifier]
        return m
    }

    /// User-asked diagnostic: red front, green back, BOTH semi-transparent
    /// and depth-write OFF so the back genuinely composites through the
    /// front instead of being depth-rejected. Reveals at a glance whether
    /// the body reads as a real volume (you see green TINTED RED across
    /// the silhouette = front+back composite) vs. a thin shell (only red,
    /// no green visible anywhere = back face is occluded).
    ///
    /// Differs from `diagnosticFaceMaterial` (used by `.faceDebug`): that
    /// one's front-element material has the SAME shader as the back, just
    /// different diffuse — depth-write is off but the shader doesn't
    /// vary front vs back. With volumeDebug each element is its own colour
    /// AND fully translucent (alpha 0.5) so the eye reads the front+back
    /// composite literally.
    private static func diagnosticVolumeMaterial(front: Bool) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = front
            ? NSColor(deviceRed: 1.00, green: 0.10, blue: 0.10, alpha: 1)   // saturated red
            : NSColor(deviceRed: 0.10, green: 1.00, blue: 0.10, alpha: 1)   // saturated green
        m.transparency = 1.0
        m.transparencyMode = .singleLayer
        m.isDoubleSided = false
        m.cullMode = .back   // overridden per-element by setSurfaceStyle()
        // BOTH off — neither face writes depth, so each face's translucent
        // fragments composite over whatever's already in the framebuffer
        // without occluding the other. With element order [back, front],
        // back paints first → green into framebuffer → front (red) blends
        // over → final pixel = ~half-red, quarter-green = orange tint
        // visible anywhere the body has both faces overlapping.
        m.writesToDepthBuffer = false
        m.readsFromDepthBuffer = true
        let modifier = """
        #pragma body
        _surface.diffuse.a = 0.50;
        """
        m.shaderModifiers = [.surface: modifier]
        return m
    }

    /// Flip the surface materials between `.fill` and `.lines`. Called by
    /// the scene controller alongside switching to the wireframe material.
    /// Safe to call with `wireframe: false` to revert. Both front and
    /// back materials get the same fillMode.
    public func setWireframe(_ wireframe: Bool) {
        let mode: SCNFillMode = wireframe ? .lines : .fill
        node.geometry?.materials.forEach { $0.fillMode = mode }
    }

    /// Replace BOTH surface materials with one of the styles, set per-
    /// element cull modes to pin front-to-element-0 and back-to-element-1,
    /// and apply the wireframe fillMode if needed.
    ///
    /// The cull mode override here is what makes the two-element trick
    /// work: SceneKit honours per-material cullMode when isDoubleSided is
    /// false (which the materials() factory enforces). Element ORDER is
    /// [back, front] (see geometry construction) so the back element
    /// draws first and the front composites OVER it — the materials
    /// array must mirror that order: [backMat, frontMat].
    public func setSurfaceStyle(_ style: MaterialStyle, cullBack: Bool) {
        let (front, back) = Self.materials(style: style, cullBack: cullBack)
        // Pin cull modes per element. cullMode pre-set on the material in
        // the factory is just a default; the bridge re-asserts it here so
        // it doesn't matter which order callers passed the materials in.
        front.cullMode = .back
        back.cullMode  = .front
        node.geometry?.materials = [back, front]
        setWireframe(style == .wireframe)
        // Re-apply the persisted tint to the fresh materials — otherwise
        // swapping style would reset the colour to the factory teal.
        applyTintToCurrentMaterials()
        // Same for alpha (honey wants translucent; cheese opaque). The
        // factory `sauceMaterial()` defaults to opaque, so without this
        // a Honey → Cheese → Honey toggle would forget the alpha.
        setSauceAlpha(currentSauceAlpha)
    }

    /// Set the water-tint RGB (linear 0..1). Plumbed through the front
    /// AND back shader modifiers as three scalar `tintR/G/B` uniforms;
    /// the back element re-derives a darker base from the same tint.
    /// Persisted across material swaps via `setSurfaceStyle`.
    public func setWaterTint(r: Float, g: Float, b: Float) {
        currentTint = SIMD3<Float>(r, g, b)
        applyTintToCurrentMaterials()
    }

    /// Set the surface alpha (transparency). 1 = fully opaque (default,
    /// what cheese / mustard / chocolate / avocado want). Values below
    /// 1 (honey ≈ 0.30) drive SCNMaterial.transparency so the MC mesh
    /// composites translucently against what's behind it.
    ///
    /// Three SceneKit material flags need to flip in tandem for the
    /// transparency to read as "translucent honey" instead of "slightly
    /// dimmed opaque cheese":
    ///   1. `transparency = alpha` (the actual alpha).
    ///   2. `writesToDepthBuffer = false` so the sauce DOESN'T occlude
    ///      what's behind it in the depth buffer — without this the
    ///      chip / floor visible through the honey gets depth-rejected
    ///      and you see only the honey's diffuse over the dark
    ///      background.
    ///   3. `transparencyMode = .dualLayer` so the back face of the
    ///      sauce composites against the front face — produces a
    ///      proper "look through" effect rather than a flat alpha
    ///      blend that pretends the sauce is infinitely thin.
    /// At alpha == 1 we flip the flags back so opaque sauces (cheese
    /// etc.) get proper depth writes for crisp silhouettes.
    public func setSauceAlpha(_ alpha: Float) {
        let clamped = max(0, min(1, alpha))
        let newIsOpaque = clamped >= 0.999
        let oldIsOpaque = currentSauceAlpha >= 0.999
        currentSauceAlpha = clamped
        let alphaNS = NSNumber(value: currentSauceAlpha)
        // Per-frame alpha update is fine. But the depth-write +
        // transparency-mode flags are STRUCTURAL — flipping them every
        // tick (even when the value didn't change) makes SceneKit
        // rebuild its render graph each frame, which spams
        // "Pass SSAOxxx is not linked to the rendering graph"
        // warnings and trashes the SSAO / floor sub-pipeline. Only
        // touch those flags when crossing the opaque ↔ translucent
        // boundary.
        let opacityCrossing = (newIsOpaque != oldIsOpaque)
        node.geometry?.materials.forEach { m in
            m.setValue(alphaNS, forKey: "sauceAlpha")
            m.transparency = CGFloat(currentSauceAlpha)
            if opacityCrossing {
                m.writesToDepthBuffer = newIsOpaque
                m.readsFromDepthBuffer = true
                m.transparencyMode = newIsOpaque ? .default : .dualLayer
            }
        }
    }

    private var currentSauceAlpha: Float = 1.0

    /// Set the Beer-Lambert absorption coefficient (per metre). Drives
    /// `transmittance = exp(-absorption * thickness)` in the water
    /// shader, where `thickness` is the refracted ray length from the
    /// fragment to the pool floor — direction-aware, so the body reads
    /// the same regardless of view angle.
    public func setAbsorption(_ absorption: Float) {
        currentAbsorption = max(0, absorption)
        applyTintToCurrentMaterials()
    }

    /// Set the procedural floor's geometry — used by the water shader's
    /// refraction sampling. `floorY` is the world Y of the pool floor;
    /// `gridCellSize` is the physical cell size; `gridOrigin` is the
    /// world XZ corner the grid is anchored to. Match the pool tile's
    /// configuration so the refracted floor lines up with the real one
    /// at the silhouette of the water.
    public func setFloorGrid(floorY: Float,
                             gridCellSize: Float,
                             gridOriginX: Float,
                             gridOriginZ: Float) {
        currentFloorY = floorY
        currentGridCellSize = gridCellSize
        currentGridOriginX = gridOriginX
        currentGridOriginZ = gridOriginZ
        applyTintToCurrentMaterials()
    }

    /// Set the world-space XZ bounds of the box that contains the fluid.
    /// Used only by the `.fishTank` material's body-exit raycast — the
    /// shader takes the nearest hit between the refracted ray and (floor,
    /// box side walls) as the body exit point, which is what produces the
    /// "one continuous body" look from any view angle. The water shader
    /// (.water style) only intersects against the floor and ignores
    /// these.
    public func setBoxBounds(minX: Float, maxX: Float, minZ: Float, maxZ: Float) {
        currentBoxMinX = minX
        currentBoxMaxX = maxX
        currentBoxMinZ = minZ
        currentBoxMaxZ = maxZ
        applyTintToCurrentMaterials()
    }

    private var currentFloorY: Float = 0
    private var currentAbsorption: Float = 2.5
    private var currentGridCellSize: Float = 0.375
    private var currentGridOriginX: Float = -3.75
    private var currentGridOriginZ: Float = -3.75
    private var currentBoxMinX: Float = -1.5
    private var currentBoxMaxX: Float =  1.5
    private var currentBoxMinZ: Float = -1.5
    private var currentBoxMaxZ: Float =  1.5

    private func applyTintToCurrentMaterials() {
        let r   = NSNumber(value: currentTint.x)
        let g   = NSNumber(value: currentTint.y)
        let b   = NSNumber(value: currentTint.z)
        let fy  = NSNumber(value: currentFloorY)
        let ab  = NSNumber(value: currentAbsorption)
        let gcs = NSNumber(value: currentGridCellSize)
        let gox = NSNumber(value: currentGridOriginX)
        let goz = NSNumber(value: currentGridOriginZ)
        let bxn = NSNumber(value: currentBoxMinX)
        let bxx = NSNumber(value: currentBoxMaxX)
        let bzn = NSNumber(value: currentBoxMinZ)
        let bzx = NSNumber(value: currentBoxMaxZ)
        node.geometry?.materials.forEach { m in
            m.setValue(r,   forKey: "tintR")
            m.setValue(g,   forKey: "tintG")
            m.setValue(b,   forKey: "tintB")
            m.setValue(fy,  forKey: "floorY")
            m.setValue(ab,  forKey: "absorption")
            m.setValue(gcs, forKey: "gridCellSize")
            m.setValue(gox, forKey: "gridMinXZ_X")
            m.setValue(goz, forKey: "gridMinXZ_Z")
            m.setValue(bxn, forKey: "boxMinX")
            m.setValue(bxx, forKey: "boxMaxX")
            m.setValue(bzn, forKey: "boxMinZ")
            m.setValue(bzx, forKey: "boxMaxZ")
        }
    }
}

// MUST match struct MCUniforms in MarchingCubes.metal.
private struct MCUniforms {
    var gridRes:  SIMD4<UInt32>   // xyz = density-field point res, w = unused
    var dxOrigin: SIMD4<Float>    // x = cell size, yzw = grid origin
    var isoLevel: SIMD4<Float>    // x = iso threshold, yzw = unused
}
