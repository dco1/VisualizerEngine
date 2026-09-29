import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Depth of field against the EXACT thin lens: PARTIAL OCCLUSION (`dofPartialOcclusion`, the layered
/// composite) and the pupil's orientation in front of the focus plane. The real kernels (tile → dilate
/// → prefilter + mips → min-CoC pyramid → [quarter → half →] gather), compiled from source, run on a
/// synthetic scene of fronto-parallel layers fed as view-space depth (identity inverse projection), and
/// each result is held against the thin-lens image of the same layers integrated on the CPU over the
/// pupil: from pupil point u the pixel p sees the nearest layer present at p + u·c(z) (c the signed CoC
/// radius, the gather's own `coc_radius`), the pupil being the gather's own aperture (blade polygon ∩
/// the cat's-eye circle at p's field position) — the geometry of the aperture-sampled reference
/// (`dofReferenceAccumulation`) the Digital Clock is measured against.
///
/// The cases the sharp-subject gather's old rules got wrong, and the invariant they must keep:
///  • a THIN BAR in front of the focus plane over a far plane (a crane's falls against the sky);
///  • a defocused LAYER EDGE against a farther layer (a blurred wall against the sky): the lens draws
///    a symmetric ramp; the old rule stepped at the silhouette;
///  • a defocused FOREGROUND SLAB's edge over a far plane (a crane's mast): the lens ramps on both
///    sides of the silhouette; the old rule kept the slab opaque right to its edge;
///  • a defocused foreground over a SHARP plane: half the pixel at the silhouette, not a third;
///  • a SHARP SUBJECT over a far wall (VZ-0188's invariant): opaque, the wall its own colour;
///  • the NEAR-FIELD DISC: in front of the focus plane a point's disc is the aperture turned half
///    round (every gather — this was a bug in all of them: the cat's eye and odd blades drawn the
///    wrong way up and the near field's blur shifted by c·|offAxis|/2).
@MainActor
final class IlluminatoramaDOFPartialOcclusionTests: XCTestCase {

    /// Mirrors the Metal `DOFParams` (float4x4, 6 floats, 5 uints, 8 floats; stride 144).
    private struct Params {
        var invProjection: simd_float4x4
        var focusDist: Float, cocCoefficient: Float, maxRadius: Float
        var blades: Float, bladeRotation: Float, catsEye: Float
        var width: UInt32, height: UInt32, tileW: UInt32, tileH: UInt32, tileSize: UInt32
        var prefilterScale: Float, cocFloor: Float, subjectAware: Float
        var fastGather: Float = 0, halfMinCoC: Float = 3, quarterMinCoC: Float = 0
        var partialOcclusion: Float = 0
        var nearPyramid: Float = 0
    }

    private let W = 160, H = 160
    /// ½·k with k = 36: c(z) = 18·(1 − z)/z px, the focus plane at 1 m.
    private let k: Float = 36

    /// The aperture: blade count (< 3 ⇒ round), rotation, cat's eye — the gather's `aperture_mask`.
    private struct Iris { var blades: Float = 0, rotation: Float = 0, catsEye: Float = 0 }

    /// One fronto-parallel layer: depth (view z, m), where it is in the PINHOLE image, its colour.
    private struct Layer {
        var z: Float
        var covers: (SIMD2<Float>) -> Bool
        var colour: (SIMD2<Float>) -> SIMD3<Float>
    }

    private func coc(_ z: Float) -> Float { 0.5 * k * (1 - z) / z }

    /// The pinhole frame: the nearest layer at each pixel centre.
    private func pinhole(_ layers: [Layer]) -> (colour: [SIMD3<Float>], depth: [Float]) {
        var col = [SIMD3<Float>](repeating: .zero, count: W * H), dep = [Float](repeating: 1e4, count: W * H)
        let order = layers.sorted { $0.z < $1.z }
        for y in 0..<H {
            for x in 0..<W {
                let pc = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                if let l = order.first(where: { $0.covers(pc) }) {
                    col[y * W + x] = l.colour(pc); dep[y * W + x] = l.z
                }
            }
        }
        return (col, dep)
    }

    /// Is pupil point `u` (units of the iris circumradius, image orientation) inside the aperture seen
    /// from field position `uv`? The gather's `aperture_mask` without its 1-px anti-aliasing.
    private func inAperture(_ u: SIMD2<Float>, uv: SIMD2<Float>, _ iris: Iris) -> Bool {
        var bound: Float = 1
        if iris.blades >= 3 {
            let seg = 2 * Float.pi / iris.blades
            let theta = atan2(u.y, u.x) + iris.rotation
            let a = fmod(fmod(theta, seg) + seg, seg) - 0.5 * seg
            bound = cos(Float.pi / iris.blades) / max(cos(a), 1e-3)
        }
        if simd_length(u) > bound { return false }
        if iris.catsEye > 0, simd_length(u + uv * iris.catsEye) > 1 { return false }
        return true
    }

    /// The thin lens itself, on the CPU: the mean over a fine grid of pupil points of what the ray from
    /// pupil point u through the pixel centre meets first — the layer present at p + u·c(z).
    private func lens(_ layers: [Layer], rows: ClosedRange<Int>, cols: ClosedRange<Int>? = nil,
                      iris: Iris = Iris(), samples n: Int = 48) -> [Int: SIMD3<Float>] {
        let order = layers.sorted { $0.z < $1.z }
        var grid: [SIMD2<Float>] = []
        for i in 0..<n {
            for j in 0..<n {
                let u = SIMD2((Float(i) + 0.5) / Float(n) * 2 - 1, (Float(j) + 0.5) / Float(n) * 2 - 1)
                if simd_length(u) <= 1 { grid.append(u) }
            }
        }
        var out: [Int: SIMD3<Float>] = [:]
        for y in rows {
            for x in cols ?? 0...(W - 1) {
                let pc = SIMD2(Float(x) + 0.5, Float(y) + 0.5)
                let uv = pc / SIMD2(Float(W), Float(H)) * 2 - 1
                var acc = SIMD3<Float>(repeating: 0)
                var count = 0
                for u in grid where inAperture(u, uv: uv, iris) {
                    count += 1
                    for l in order {
                        let hit = pc + u * coc(l.z)
                        if l.covers(hit) { acc += l.colour(hit); break }
                    }
                }
                out[y * W + x] = acc / Float(max(1, count))
            }
        }
        return out
    }

    /// `nearPyramid` (default: with `partial`, as the renderer runs it) = the layered composite reads
    /// the near field as coverage from its own pyramid; false = hit-or-miss near-field texels.
    /// `subjectAware` false = the engine's DEFAULT gather (no `dofSharpSubjects` — what Daydream Home
    /// runs); the fast gather and the layered composite exist only in sharp-subject mode.
    private func run(_ layers: [Layer], partial: Bool, fast: Bool = false, quarter: Float = 0,
                     iris: Iris = Iris(), nearPyramid: Bool? = nil,
                     subjectAware: Bool = true, prefilterScale: Float = 1) throws -> [SIMD3<Float>] {
        let nearOn = partial && (nearPyramid ?? true)
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaDOF.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let lib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: url)
        func pipe(_ n: String) throws -> MTLComputePipelineState {
            let fn = try XCTUnwrap(lib.makeFunction(name: n), "missing kernel \(n)")
            return try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        }
        let tileP = try pipe("illumi_dof_tile"), dilP = try pipe("illumi_dof_dilate")
        let preP = try pipe("illumi_dof_prefilter"), dofP = try pipe("illumi_dof")
        let minP = try pipe("illumi_dof_mincoc"), halfP = try pipe("illumi_dof_half")
        let quarterP = try pipe("illumi_dof_quarter")
        func tex(_ fmt: MTLPixelFormat, _ w: Int, _ h: Int, mips: Int = 1, shared: Bool = false) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.mipmapLevelCount = mips
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = shared ? .shared : .private
            return device.makeTexture(descriptor: d)!
        }
        let frame = pinhole(layers)
        let hdr = tex(.rgba16Float, W, H, shared: true)
        let depth = tex(.r32Float, W, H, shared: true)
        var px = [Float16](repeating: 0, count: W * H * 4)
        for i in 0..<(W * H) {
            let c = frame.colour[i]
            px[i * 4] = Float16(c.x); px[i * 4 + 1] = Float16(c.y); px[i * 4 + 2] = Float16(c.z); px[i * 4 + 3] = 1
        }
        px.withUnsafeBytes { hdr.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                         withBytes: $0.baseAddress!, bytesPerRow: W * 8) }
        frame.depth.withUnsafeBytes { depth.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                                    withBytes: $0.baseAddress!, bytesPerRow: W * 4) }
        let tile = 16, tw = (W + tile - 1) / tile, th = (H + tile - 1) / tile
        let tiles = tex(.rg32Float, tw, th), dilated = tex(.rg32Float, tw, th)
        let pw = (W + 1) / 2, ph = (H + 1) / 2
        let mips = Int(log2(Double(max(pw, ph))).rounded(.down)) + 1
        let pre = tex(.rgba16Float, pw, ph, mips: mips)
        // As the renderer allocates it: R = least |CoC|; + G/B the signed-CoC range for the layered composite.
        let minFormat: MTLPixelFormat = partial ? .rgba16Float : .r16Float
        let minCoC = tex(minFormat, pw, ph, mips: mips)
        let minLevels = (0..<mips).map { minCoC.makeTextureView(pixelFormat: minFormat, textureType: .type2D,
                                                                 levels: $0..<($0 + 1), slices: 0..<1)! }
        // The near-field coverage pyramid (as the renderer allocates it while the composite is on).
        let near = tex(.rgba16Float, pw, ph, mips: mips), nearCoC = tex(.r16Float, pw, ph, mips: mips)
        let out = tex(.rgba16Float, W, H)
        let half = tex(.rgba16Float, pw, ph)
        let qw = (W + 3) / 4, qh = (H + 3) / 4
        let quarterTex = tex(.rgba16Float, qw, qh)
        var p = Params(invProjection: matrix_identity_float4x4, focusDist: 1, cocCoefficient: k, maxRadius: 32,
                       blades: iris.blades, bladeRotation: iris.rotation, catsEye: iris.catsEye,
                       width: UInt32(W), height: UInt32(H), tileW: UInt32(tw), tileH: UInt32(th), tileSize: UInt32(tile),
                       prefilterScale: prefilterScale, cocFloor: 0, subjectAware: subjectAware ? 1 : 0,
                       fastGather: fast ? 1 : 0, halfMinCoC: 3,
                       quarterMinCoC: fast && quarter > 0 ? max(6, quarter) : 0,
                       partialOcclusion: partial ? 1 : 0, nearPyramid: nearOn ? 1 : 0)
        XCTAssertEqual(MemoryLayout<Params>.stride, 144)
        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        func dispatch(_ ps: MTLComputePipelineState, _ w: Int, _ h: Int, params: Bool = true,
                      _ bind: (MTLComputeCommandEncoder) -> Void) {
            let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(ps)
            bind(e)
            if params { e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0) }
            e.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            e.endEncoding()
        }
        dispatch(tileP, tw, th) { $0.setTexture(depth, index: 0); $0.setTexture(tiles, index: 1) }
        dispatch(dilP, tw, th) { $0.setTexture(tiles, index: 0); $0.setTexture(dilated, index: 1) }
        dispatch(preP, pw, ph) {
            $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(pre, index: 2)
            $0.setTexture(minCoC, index: 3); $0.setTexture(near, index: 4); $0.setTexture(nearCoC, index: 5)
        }
        let blit = cb.makeBlitCommandEncoder()!
        blit.generateMipmaps(for: pre)
        if nearOn { blit.generateMipmaps(for: near); blit.generateMipmaps(for: nearCoC) }
        blit.endEncoding()
        for l in 1..<mips {
            dispatch(minP, minLevels[l].width, minLevels[l].height, params: false) {
                $0.setTexture(minLevels[l - 1], index: 0); $0.setTexture(minLevels[l], index: 1)
            }
        }
        let quarterOn = fast && quarter > 0
        if quarterOn {
            dispatch(quarterP, qw, qh) {
                $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(quarterTex, index: 2)
                $0.setTexture(dilated, index: 3); $0.setTexture(pre, index: 4); $0.setTexture(minCoC, index: 5)
                $0.setTexture(near, index: 6); $0.setTexture(nearCoC, index: 7)
            }
        }
        if fast {
            dispatch(halfP, pw, ph) {
                $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(half, index: 2)
                $0.setTexture(dilated, index: 3); $0.setTexture(pre, index: 4); $0.setTexture(minCoC, index: 5)
                $0.setTexture(quarterOn ? quarterTex : pre, index: 6)
                $0.setTexture(near, index: 7); $0.setTexture(nearCoC, index: 8)
            }
        }
        dispatch(dofP, W, H) {
            $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(out, index: 2)
            $0.setTexture(dilated, index: 3); $0.setTexture(pre, index: 4); $0.setTexture(minCoC, index: 5)
            $0.setTexture(fast ? half : pre, index: 6)
            $0.setTexture(quarterOn ? quarterTex : pre, index: 7)
            $0.setTexture(near, index: 8); $0.setTexture(nearCoC, index: 9)
        }
        let buf = device.makeBuffer(length: W * H * 8, options: .storageModeShared)!
        let b2 = cb.makeBlitCommandEncoder()!
        b2.copy(from: out, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: W, height: H, depth: 1), to: buf, destinationOffset: 0,
                destinationBytesPerRow: W * 8, destinationBytesPerImage: W * H * 8)
        b2.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        XCTAssertNil(cb.error, "DOF command buffer failed: \(String(describing: cb.error))")
        let f = buf.contents().bindMemory(to: Float16.self, capacity: W * H * 4)
        return (0..<(W * H)).map { SIMD3(Float(f[$0 * 4]), Float(f[$0 * 4 + 1]), Float(f[$0 * 4 + 2])) }
    }

    private func lum(_ c: SIMD3<Float>) -> Float { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }

    /// Mean and worst |gather − lens| (luminance, in units of the scene's layer contrast) over `pixels`.
    private func error(_ img: [SIMD3<Float>], _ ref: [Int: SIMD3<Float>], _ pixels: [Int],
                       contrast: Float) -> (mean: Float, worst: Float) {
        var s: Float = 0, w: Float = 0
        for i in pixels {
            let e = abs(lum(img[i]) - lum(ref[i]!)) / contrast
            s += e; w = max(w, e)
        }
        return (s / Float(max(1, pixels.count)), w)
    }

    // ── Scenes ────────────────────────────────────────────────────────────────

    private let dark = SIMD3<Float>(0.02, 0.02, 0.02), bright = SIMD3<Float>(1.0, 0.9, 0.8)

    /// A 3-px bar at z 0.64 (c = +10.1 px, in FRONT of focus) over a plane at z 4 (c = −13.5 px). Its
    /// edges lie between pixel centres (x 79…82), so the pinhole frame holds the 3 px the lens sees.
    private var thinBar: [Layer] {
        [Layer(z: 0.64, covers: { abs($0.x - 80.5) < 1.5 }, colour: { _ in self.dark }),
         Layer(z: 4, covers: { _ in true }, colour: { _ in self.bright })]
    }

    /// A defocused wall (z 1.6, c = −6.75 px) filling x < 80 against a farther plane (z 8, c = −15.75).
    private var layerEdge: [Layer] {
        [Layer(z: 1.6, covers: { $0.x < 80 }, colour: { _ in SIMD3(0.2, 0.2, 0.2) }),
         Layer(z: 8, covers: { _ in true }, colour: { _ in self.bright })]
    }

    /// A defocused FOREGROUND slab (z 0.75, c = +6 px) filling x < 80 over a far plane (z 50, c = −17.6):
    /// a crane's mast against the sky, the far layer three times as defocused as the slab.
    private var foregroundSlab: [Layer] {
        [Layer(z: 0.75, covers: { $0.x < 80 }, colour: { _ in SIMD3(0.9, 0.55, 0.1) }),
         Layer(z: 50, covers: { _ in true }, colour: { _ in SIMD3(0.35, 0.5, 0.8) })]
    }

    /// The same slab over a SHARP plane on the focus plane, striped so its own texture is visible.
    private var foregroundOverSharp: [Layer] {
        [Layer(z: 0.6, covers: { $0.x < 80 }, colour: { _ in SIMD3(0.9, 0.55, 0.1) }),
         Layer(z: 1, covers: { _ in true },
               colour: { p in (Int(floor(p.y / 4)) & 1) == 0 ? SIMD3(0.05, 0.08, 0.15) : SIMD3(0.10, 0.14, 0.25) })]
    }

    /// An in-focus disc (z 1, r 24) over a wall at z 2 (c = −9 px) — VZ-0188's sharp subject.
    private var sharpSubject: [Layer] {
        [Layer(z: 1, covers: { simd_distance($0, SIMD2(80, 80)) < 24 }, colour: { _ in SIMD3(1.0, 0.03, 0.02) }),
         Layer(z: 2, covers: { _ in true }, colour: { _ in SIMD3(0.010, 0.012, 0.020) })]
    }

    /// Two point lights on a black field off the frame centre (the cat's eye bites there): one in FRONT
    /// of the focus plane (z 0.6, c = +12 px) at (40, 60), one BEHIND it (z 3, c = −12) at (120, 100).
    private var twoPoints: [Layer] {
        [Layer(z: 0.6, covers: { abs($0.x - 40) < 1 && abs($0.y - 60) < 1 }, colour: { _ in SIMD3(repeating: 8) }),
         Layer(z: 3, covers: { _ in true },
               colour: { abs($0.x - 120) < 1 && abs($0.y - 100) < 1 ? SIMD3(repeating: 8) : SIMD3(repeating: 0) })]
    }

    /// A blurred wall (z 1.6, c = −6.75 px) filling x < 80 against a far plane (z 50, c = −17.6), with a
    /// thin member in FRONT of the focus plane (z 0.75, c = +6 px) at x 98…101 — within the near
    /// spiral's reach of the wall's edge, not within its disc. The far plane's pixels beside the wall
    /// see the wall as a small front (its disc 0.38 of theirs) and the member as a near-field front.
    private var layerEdgeBesideNearMember: [Layer] {
        [Layer(z: 0.75, covers: { abs($0.x - 99.5) < 1.5 }, colour: { _ in SIMD3(0.9, 0.55, 0.1) }),
         Layer(z: 1.6, covers: { $0.x < 80 }, colour: { _ in SIMD3(0.2, 0.2, 0.2) }),
         Layer(z: 50, covers: { _ in true }, colour: { _ in self.bright })]
    }

    private let rows = 76...83

    private func rowPixels(_ xs: ClosedRange<Int>) -> [Int] { rows.flatMap { y in xs.map { y * W + $0 } } }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// The crane's falls: a thin member in front of the focus plane melts into what is behind it.
    func testThinForegroundBarBecomesAsTransparentAsTheLensMakesIt() throws {
        let ref = lens(thinBar, rows: rows)
        let contrast = lum(bright) - lum(dark)
        let bar = rowPixels(79...81), around = rowPixels(60...100)
        for fast in [false, true] {
            let old = try run(thinBar, partial: false, fast: fast)
            let new = try run(thinBar, partial: true, fast: fast)
            let eOld = error(old, ref, bar, contrast: contrast), eNew = error(new, ref, bar, contrast: contrast)
            let aOld = error(old, ref, around, contrast: contrast), aNew = error(new, ref, around, contrast: contrast)
            print("DOF thin bar (fast \(fast)): lens \(lum(ref[80 * W + 80]!)), old \(lum(old[80 * W + 80])), "
                  + "new \(lum(new[80 * W + 80])); bar err old \(eOld) new \(eNew); ±20 px old \(aOld) new \(aNew)")
            XCTAssertGreaterThan(eOld.worst, 0.05, "the probe must see the old rule keep the bar too opaque")
            XCTAssertLessThan(eNew.mean, eOld.mean * 0.5, "the bar must be nearer the lens than the old rule")
            XCTAssertLessThan(eNew.worst, 0.08, "the bar's pixels are not what the lens sees: \(eNew)")
            XCTAssertLessThan(aNew.mean, 0.03, "the bar's neighbourhood drifts from the lens: \(aNew)")
        }
    }

    /// A blurred wall's edge against the sky: a ramp through 50 % at the silhouette, not a step.
    func testDefocusedLayerEdgeRampsLikeTheLens() throws {
        let ref = lens(layerEdge, rows: rows)
        let contrast = lum(bright) - 0.2
        let edge = rowPixels(66...94), nearSide = rowPixels(74...79)
        for fast in [false, true] {
            let old = try run(layerEdge, partial: false, fast: fast)
            let new = try run(layerEdge, partial: true, fast: fast)
            let eOld = error(old, ref, edge, contrast: contrast), eNew = error(new, ref, edge, contrast: contrast)
            let nOld = error(old, ref, nearSide, contrast: contrast), nNew = error(new, ref, nearSide, contrast: contrast)
            print("DOF layer edge (fast \(fast)): ±14 px err old \(eOld) new \(eNew); near side old \(nOld) new \(nNew)")
            XCTAssertGreaterThan(nOld.worst, 0.15, "the probe must see the old rule's step")
            XCTAssertLessThan(nNew.worst, 0.08, "the near side of the edge is not the lens's ramp: \(nNew)")
            XCTAssertLessThan(eNew.mean, 0.035, "the edge drifts from the lens: \(eNew)")
        }
    }

    /// A wall's edge against the sky with a crane's falls within reach: the far plane's pixels take the
    /// wall's coverage ONCE. (The near spiral used to count every front it met, so a front behind the
    /// focus plane inside its disc was counted by both spirals: the wall's coverage doubled along the
    /// sky's edge wherever a near-field member was within reach — a dark line on the Digital Clock's
    /// wall top, 0.30 where the aperture-sampled reference has 0.41.) And the small-front spiral: the
    /// wall's 6.75-px disc is measured by taps fitted to it, not by the 18-px own disc's.
    func testFarLayerEdgeBesideANearMemberTakesTheWallOnce() throws {
        let ref = lens(layerEdgeBesideNearMember, rows: rows)
        let contrast = lum(bright) - 0.2
        let farSide = rowPixels(80...88), edge = rowPixels(70...92)
        for fast in [false, true] {
            let img = try run(layerEdgeBesideNearMember, partial: true, fast: fast)
            let eFar = error(img, ref, farSide, contrast: contrast), eEdge = error(img, ref, edge, contrast: contrast)
            var prof = ""
            for x in stride(from: 72, through: 90, by: 2) {
                prof += String(format: " %d:%.3f/%.3f", x, lum(ref[80 * W + x]!), lum(img[80 * W + x]))
            }
            print("DOF wall edge beside a near member (fast \(fast)): far side \(eFar), ±11 px \(eEdge); lens/gather\(prof)")
            // (Counted twice, the wall's coverage darkens the far side by ~0.15 of the contrast over
            // several pixels; the fast gather's sparser ring leaves single pixels up to ~0.1 off.)
            XCTAssertLessThan(eFar.worst, fast ? 0.12 : 0.08, "the far plane beside the wall is not what the lens sees: \(eFar)")
            XCTAssertLessThan(eFar.mean, 0.03, "the far plane beside the wall drifts from the lens: \(eFar)")
            XCTAssertLessThan(eEdge.mean, 0.035, "the wall's edge drifts from the lens: \(eEdge)")
        }
    }

    /// A crane's mast: a defocused FOREGROUND edge ramps on both sides of its silhouette — the rays that
    /// pass its edge see the far plane (hidden behind the slab in the pinhole frame; the holes carry it).
    func testForegroundSlabEdgeRampsOnBothSides() throws {
        let slab = SIMD3<Float>(0.9, 0.55, 0.1), far = SIMD3<Float>(0.35, 0.5, 0.8)
        let ref = lens(foregroundSlab, rows: rows)
        let contrast = abs(lum(slab) - lum(far))
        let inside = rowPixels(74...79), edge = rowPixels(64...96)
        for fast in [false, true] {
            let old = try run(foregroundSlab, partial: false, fast: fast)
            let new = try run(foregroundSlab, partial: true, fast: fast)
            let iOld = error(old, ref, inside, contrast: contrast), iNew = error(new, ref, inside, contrast: contrast)
            let eNew = error(new, ref, edge, contrast: contrast)
            print("DOF foreground edge (fast \(fast)): inside 6 px old \(iOld) new \(iNew); ±16 px new \(eNew); "
                  + "at the silhouette lens \(lum(ref[80 * W + 79]!)) old \(lum(old[80 * W + 79])) new \(lum(new[80 * W + 79]))")
            XCTAssertGreaterThan(iOld.worst, 0.2, "the probe must see the old rule's opaque inner edge")
            XCTAssertLessThan(iNew.worst, 0.08, "the slab's inner edge is not the lens's ramp: \(iNew)")
            XCTAssertLessThan(eNew.mean, 0.03, "the slab's edge drifts from the lens: \(eNew)")
        }
    }

    /// A defocused foreground over a SHARP plane covers half the pixel at its silhouette, as the lens
    /// does — composited over, not averaged in (which gave it a third).
    func testForegroundOverSharpPlaneCoversAsTheLensDoes() throws {
        let ref = lens(foregroundOverSharp, rows: rows)
        let contrast = lum(SIMD3(0.9, 0.55, 0.1)) - lum(SIMD3(0.05, 0.08, 0.15))
        let sharpSide = rowPixels(80...92)
        for fast in [false, true] {
            let old = try run(foregroundOverSharp, partial: false, fast: fast)
            let new = try run(foregroundOverSharp, partial: true, fast: fast)
            let eOld = error(old, ref, sharpSide, contrast: contrast), eNew = error(new, ref, sharpSide, contrast: contrast)
            print("DOF foreground over sharp (fast \(fast)): sharp side 12 px old \(eOld) new \(eNew); "
                  + "at the silhouette lens \(lum(ref[80 * W + 80]!)) old \(lum(old[80 * W + 80])) new \(lum(new[80 * W + 80]))")
            XCTAssertGreaterThan(eOld.worst, 0.1, "the probe must see the old averaging's thin coverage")
            XCTAssertLessThan(eNew.worst, 0.07, "the foreground's coverage over the sharp plane is not the lens's: \(eNew)")
        }
    }

    /// VZ-0188's invariant holds under the layered composite: an in-focus subject stays opaque, and
    /// the wall round it takes none of its colour.
    func testSharpSubjectStaysOpaqueUnderTheLayeredComposite() throws {
        let subject = SIMD3<Float>(1.0, 0.03, 0.02), wall = SIMD3<Float>(0.010, 0.012, 0.020)
        for (fast, quarter) in [(false, Float(0)), (true, Float(0)), (true, Float(6))] {
            let img = try run(sharpSubject, partial: true, fast: fast, quarter: quarter)
            var subjWorst: Float = 0, wallWorst: Float = 0
            for y in 0..<H {
                for x in 0..<W {
                    let d = simd_distance(SIMD2(Float(x) + 0.5, Float(y) + 0.5), SIMD2(80, 80)) - 24
                    let c = img[y * W + x]
                    if d <= -1.5 { subjWorst = max(subjWorst, simd_reduce_max(simd_abs(c - subject) / subject)) }
                    if d >= 2.5 { wallWorst = max(wallWorst, simd_reduce_max(simd_abs(c - wall) / wall)) }
                }
            }
            print("DOF sharp subject under the layered composite (fast \(fast), quarter \(quarter)): "
                  + "subject worst \(subjWorst), wall worst \(wallWorst)")
            XCTAssertLessThan(subjWorst, 0.05, "the in-focus subject went see-through: \(subjWorst)")
            XCTAssertLessThan(wallWorst, 0.05, "the wall took on the subject's colour: \(wallWorst)")
        }
    }

    /// In front of the focus plane a point's disc is the aperture turned half round: with a 5-bladed
    /// iris and a cat's eye, the gather's bokeh of a near and a far point must both sit where the lens
    /// puts them — the cat's eye moves the visible pupil off its centre, so a disc drawn the wrong way
    /// round lands its centroid c·|shift| on the wrong side (every gather: the pupil point is the tap's
    /// offset over its SIGNED CoC).
    func testNearFieldDiscIsTheApertureTurnedHalfRound() throws {
        let iris = Iris(blades: 5, rotation: 0.26, catsEye: 0.3)
        let ref = lens(twoPoints, rows: 44...116, cols: 24...136, iris: iris)
        func window(_ centre: SIMD2<Int>) -> [Int] {
            ((centre.y - 16)...(centre.y + 16)).flatMap { y in ((centre.x - 16)...(centre.x + 16)).map { y * W + $0 } }
        }
        func centroid(_ lumAt: (Int) -> Float, _ px: [Int]) -> SIMD2<Float> {
            var s = SIMD2<Float>(0, 0), m: Float = 0
            for i in px {
                let l = lumAt(i)
                s += l * SIMD2(Float(i % W) + 0.5, Float(i / W) + 0.5); m += l
            }
            return s / max(m, 1e-6)
        }
        let near = window(SIMD2(40, 60)), far = window(SIMD2(120, 100))
        let refNear = centroid({ self.lum(ref[$0]!) }, near), refFar = centroid({ self.lum(ref[$0]!) }, far)
        // (ss false: the engine's DEFAULT gather — no sharp-subject mode, what Daydream Home runs — whose
        // scatter-as-gather aperture test is the same line.)
        for (partial, fast, ss) in [(false, false, false), (false, false, true), (false, true, true),
                                    (true, false, true), (true, true, true)] {
            let img = try run(twoPoints, partial: partial, fast: fast, iris: iris, subjectAware: ss)
            let gNear = centroid({ self.lum(img[$0]) }, near), gFar = centroid({ self.lum(img[$0]) }, far)
            var eNear: Float = 0, eFar: Float = 0
            for i in near { eNear += abs(lum(img[i]) - lum(ref[i]!)) }
            for i in far { eFar += abs(lum(img[i]) - lum(ref[i]!)) }
            eNear /= Float(near.count); eFar /= Float(far.count)
            print("DOF near/far disc vs the lens (partial \(partial), fast \(fast), sharp subjects \(ss)): centroid offset near "
                  + "\(simd_distance(gNear, refNear)) px, far \(simd_distance(gFar, refFar)) px (lens near \(refNear - SIMD2(40, 60)), "
                  + "far \(refFar - SIMD2(120, 100))); mean |Δ| near \(eNear), far \(eFar)")
            if ProcessInfo.processInfo.environment["DOF_PROFILE"] != nil {
                var g = "DOF near grid (partial \(partial), fast \(fast)):\n"
                for y in stride(from: 46, through: 74, by: 2) {
                    for x in stride(from: 26, through: 54, by: 2) {
                        g += String(format: "%4.0f/%-4.0f ", lum(ref[y * W + x]!) * 1000, lum(img[y * W + x]) * 1000)
                    }
                    g += "\n"
                }
                print(g)
            }
            // The layered composite lands within 0.06 / 0.27 px (full / fast) and fails when its near disc
            // is drawn the wrong way round. The plain gathers' own sampling leaves 0.44 / 0.59 px here, and
            // drawn the wrong way round they move only to 0.56 / 0.62 px (the lens's shift at a 0.3 cat's
            // eye mid-frame is 0.15 px), so this bound does NOT detect their orientation — that is
            // `testNearFieldDiscTurnsHalfRoundUnderAStrongCatsEye`'s job (verifier mutation check).
            XCTAssertLessThan(simd_distance(gNear, refNear), partial ? 0.35 : 0.7,
                              "the near-field disc is not where the lens puts it")
            XCTAssertLessThan(simd_distance(gFar, refFar), 0.35, "the far-field disc is not where the lens puts it")
            XCTAssertLessThan(eFar, 0.02, "the far-field disc is not the lens's")
            // The layered composite reads a near-field front as COVERAGE from the near-field pyramid: an
            // isolated 2×2-px near-field highlight is integrated over each tap's footprint instead of
            // being hit by ~1 tap per pixel (hit-or-miss texels: 0.022 / 0.031 here, grainier than the
            // plain gather's 0.007 / 0.011; with the pyramid 0.005 / 0.008).
            XCTAssertLessThan(eNear, partial ? 0.012 : 0.02, "the near-field disc is not the lens's")
        }
    }

    /// The same orientation where it can be SEEN past the gathers' own sampling: a strong cat's eye
    /// (0.6) at the frame's corners bites a third off the pupil, so the lens moves each disc's centroid
    /// ~2.3 px off its point — toward the frame centre for the far point, AWAY from it for the near one
    /// (the near disc is the clipped pupil turned half round). Drawn the wrong way up, the near disc's
    /// centroid lands on the other side of its point. (The test above, at a 0.3 cat's eye near
    /// mid-frame, cannot tell: the bug moves the default gather's near centroid only 0.44 → 0.56 px,
    /// under its 0.7-px bound.) Every gather — the default one first (Daydream Home's).
    ///
    /// The plain gathers run with point taps (`prefilterScale` 0) here: with the prefilter pyramid their
    /// taps BESIDE a near-field point read it prefiltered into the far layer round it and scatter it
    /// with the far layer's (upright) aperture — a leak that pulls the near disc's centroid ~2.7 px
    /// toward the frame centre whichever way the near test is drawn, so it would hide the orientation.
    /// The layered composite reads the near field from its own pyramid and runs as the renderer runs it.
    func testNearFieldDiscTurnsHalfRoundUnderAStrongCatsEye() throws {
        let iris = Iris(blades: 5, rotation: 0.26, catsEye: 0.6)
        let nearAt = SIMD2<Int>(26, 26), farAt = SIMD2<Int>(134, 134)
        let points = [
            Layer(z: 0.6, covers: { abs($0.x - Float(nearAt.x)) < 1 && abs($0.y - Float(nearAt.y)) < 1 },
                  colour: { _ in SIMD3(repeating: 8) }),
            Layer(z: 3, covers: { _ in true },
                  colour: { abs($0.x - Float(farAt.x)) < 1 && abs($0.y - Float(farAt.y)) < 1
                            ? SIMD3(repeating: 8) : SIMD3(repeating: 0) })]
        func window(_ c: SIMD2<Int>) -> [Int] {
            ((c.y - 16)...(c.y + 16)).flatMap { y in ((c.x - 16)...(c.x + 16)).map { y * W + $0 } }
        }
        func centroid(_ lumAt: (Int) -> Float, _ px: [Int]) -> SIMD2<Float> {
            var s = SIMD2<Float>(0, 0), m: Float = 0
            for i in px {
                let l = lumAt(i)
                s += l * SIMD2(Float(i % W) + 0.5, Float(i / W) + 0.5); m += l
            }
            return s / max(m, 1e-6)
        }
        let near = window(nearAt), far = window(farAt)
        let refN = lens(points, rows: (nearAt.y - 16)...(nearAt.y + 16), cols: (nearAt.x - 16)...(nearAt.x + 16), iris: iris)
        let refF = lens(points, rows: (farAt.y - 16)...(farAt.y + 16), cols: (farAt.x - 16)...(farAt.x + 16), iris: iris)
        let pN = SIMD2<Float>(Float(nearAt.x), Float(nearAt.y)), pF = SIMD2<Float>(Float(farAt.x), Float(farAt.y))
        let lensN = centroid({ self.lum(refN[$0]!) }, near) - pN, lensF = centroid({ self.lum(refF[$0]!) }, far) - pF
        print("DOF strong cat's eye: lens near shift \(lensN), far \(lensF)")
        XCTAssertGreaterThan(simd_length(lensN), 1.5, "the probe needs a visible clip to tell the orientation")
        let centre = SIMD2<Float>(Float(W), Float(H)) * 0.5
        XCTAssertGreaterThan(simd_dot(lensN, pN - centre), 0, "the lens moves a near disc AWAY from the frame centre")
        XCTAssertLessThan(simd_dot(lensF, pF - centre), 0, "the lens moves a far disc TOWARD the frame centre")
        for (partial, fast, ss) in [(false, false, false), (false, false, true), (false, true, true),
                                    (true, false, true), (true, true, true)] {
            let img = try run(points, partial: partial, fast: fast, iris: iris, subjectAware: ss,
                              prefilterScale: partial ? 1 : 0)
            let gN = centroid({ self.lum(img[$0]) }, near) - pN, gF = centroid({ self.lum(img[$0]) }, far) - pF
            // How much of the lens's shift each disc reproduces, along it: 1 = the lens, ~−1 = upside down.
            let fN = simd_dot(gN, lensN) / simd_length_squared(lensN), fF = simd_dot(gF, lensF) / simd_length_squared(lensF)
            print("DOF strong cat's eye (partial \(partial), fast \(fast), sharp subjects \(ss)): near shift \(gN) "
                  + "(\(fN) of the lens's), far \(gF) (\(fF))")
            XCTAssertGreaterThan(fN, 0.3, "the near-field disc is not the clipped aperture turned half round "
                                 + "(partial \(partial), fast \(fast), sharp subjects \(ss))")
            XCTAssertGreaterThan(fF, 0.3, "the far-field disc is not the clipped aperture "
                                 + "(partial \(partial), fast \(fast), sharp subjects \(ss))")
            if partial {
                XCTAssertLessThan(simd_distance(gN, lensN), 0.35 * simd_length(lensN),
                                  "the layered composite's near disc is not where the lens puts it (fast \(fast))")
            }
        }
    }

    /// Off is deterministic (the partial-occlusion switch and the pyramid's G/B are then unread).
    func testPartialOcclusionOffLeavesTheGatherUnchanged() throws {
        for fast in [false, true] {
            let a = try run(layerEdge, partial: false, fast: fast)
            let b = try run(layerEdge, partial: false, fast: fast)
            XCTAssertEqual(a, b)
        }
    }
}
