import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Depth of field (`IlluminatoramaDOF.metal`) on a synthetic sharp subject in front of a far,
/// defocused plane: the REAL four kernels (tile → dilate → prefilter + mips → gather), compiled
/// from source, fed a view-space depth directly (identity inverse projection).
///
/// The invariant (VZ-0188, `dofSharpSubjects`): an IN-FOCUS subject stays opaque right up to its
/// silhouette, and the defocused plane behind it keeps its own colour beyond the subject's own
/// (≈ 0) confusion disc — even when the subject is ~100× brighter than the plane (a red-lit toy
/// against a night wall). The legacy gather is run too, so the probe is shown not to be vacuous.
@MainActor
final class IlluminatoramaDOFSubjectTests: XCTestCase {

    /// Mirrors the Metal `DOFParams` (float4x4, 6 floats, 5 uints, 6 floats; stride 144).
    private struct Params {
        var invProjection: simd_float4x4
        var focusDist: Float, cocCoefficient: Float, maxRadius: Float
        var blades: Float, bladeRotation: Float, catsEye: Float
        var width: UInt32, height: UInt32, tileW: UInt32, tileH: UInt32, tileSize: UInt32
        var prefilterScale: Float, cocFloor: Float, subjectAware: Float
        var fastGather: Float = 0, halfMinCoC: Float = 3, quarterMinCoC: Float = 0
    }

    /// Share of the quarter tier's texels the last `run(quarter:)` gathered (alpha > 0).
    private var lastQuarterCoverage: Float = 0

    private let W = 160, H = 160
    private let centre = SIMD2<Float>(80, 80), radius: Float = 24
    private let wall = SIMD3<Float>(0.010, 0.012, 0.020)
    private let subject = SIMD3<Float>(1.0, 0.03, 0.02)

    /// Distance (px) of pixel centre (x, y) from the subject's silhouette: < 0 inside.
    private func edgeDistance(_ x: Int, _ y: Int) -> Float {
        simd_distance(SIMD2(Float(x) + 0.5, Float(y) + 0.5), centre) - radius
    }

    /// The wall's colour at a pixel: flat, or (textured) a pattern the blur has real work on.
    /// Bright points on the plane (HDR highlights ~150× the wall): their defocused images are bokeh
    /// balls with sharp rims — the case the quarter tier's smoothness guard exists for.
    private var brightDots = false
    private func wallColour(_ x: Int, _ y: Int, textured: Bool) -> SIMD3<Float> {
        if brightDots && (x % 37 == 5) && (y % 29 == 7) && edgeDistance(x, y) > 14 { return wall * 150 }
        guard textured else { return wall }
        let k = 1 + 0.6 * sin(Float(x) / 2.3) * sin(Float(y) / 3.1) + 0.3 * sin(Float(x + 2 * y) / 1.7)
        return wall * k
    }

    private func run(subjectAware: Bool, subjectZ: Float = 1, fast: Bool = false,
                     texturedWall: Bool = false, halfMin: Float = 3, quarter: Float = 0) throws -> [SIMD3<Float>] {
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
        // Colour and view-space depth: the subject ON the focus plane (1 m), the plane at 2 m.
        let hdr = tex(.rgba16Float, W, H, shared: true)
        let depth = tex(.r32Float, W, H, shared: true)
        var px = [Float16](repeating: 0, count: W * H * 4)
        var dz = [Float](repeating: 0, count: W * H)
        for y in 0..<H {
            for x in 0..<W {
                let inside = edgeDistance(x, y) < 0
                let c = inside ? subject : wallColour(x, y, textured: texturedWall)
                let i = (y * W + x) * 4
                px[i] = Float16(c.x); px[i + 1] = Float16(c.y); px[i + 2] = Float16(c.z); px[i + 3] = 1
                dz[y * W + x] = inside ? subjectZ : 2.0
            }
        }
        px.withUnsafeBytes { hdr.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                         withBytes: $0.baseAddress!, bytesPerRow: W * 8) }
        dz.withUnsafeBytes { depth.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0,
                                           withBytes: $0.baseAddress!, bytesPerRow: W * 4) }
        let tile = 16, tw = (W + tile - 1) / tile, th = (H + tile - 1) / tile
        let tiles = tex(.rg32Float, tw, th), dilated = tex(.rg32Float, tw, th)
        let pw = (W + 1) / 2, ph = (H + 1) / 2
        let mips = Int(log2(Double(max(pw, ph))).rounded(.down)) + 1
        let pre = tex(.rgba16Float, pw, ph, mips: mips)
        let minCoC = tex(.r16Float, pw, ph, mips: mips)
        let minLevels = (0..<mips).map { minCoC.makeTextureView(pixelFormat: .r16Float, textureType: .type2D,
                                                                 levels: $0..<($0 + 1), slices: 0..<1)! }
        let out = tex(.rgba16Float, W, H)
        let half = tex(.rgba16Float, pw, ph)
        let qw = (W + 3) / 4, qh = (H + 3) / 4
        let quarterTex = tex(.rgba16Float, qw, qh, shared: true)
        // |CoC| = ½·k·|z_f − z|/z = 9 px at the plane (k = 36), 0 on the subject.
        var p = Params(invProjection: matrix_identity_float4x4, focusDist: 1, cocCoefficient: 36, maxRadius: 32,
                       blades: 0, bladeRotation: 0, catsEye: 0,
                       width: UInt32(W), height: UInt32(H), tileW: UInt32(tw), tileH: UInt32(th), tileSize: UInt32(tile),
                       prefilterScale: 1, cocFloor: 0, subjectAware: subjectAware ? 1 : 0,
                       fastGather: fast ? 1 : 0, halfMinCoC: halfMin,
                       quarterMinCoC: fast && quarter > 0 ? max(6, quarter) : 0)
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
            $0.setTexture(minCoC, index: 3)
        }
        let blit = cb.makeBlitCommandEncoder()!
        blit.generateMipmaps(for: pre)
        blit.endEncoding()
        if subjectAware {
            for l in 1..<mips {
                dispatch(minP, minLevels[l].width, minLevels[l].height, params: false) {
                    $0.setTexture(minLevels[l - 1], index: 0); $0.setTexture(minLevels[l], index: 1)
                }
            }
        }
        let quarterOn = fast && quarter > 0
        if quarterOn {
            dispatch(quarterP, qw, qh) {
                $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(quarterTex, index: 2)
                $0.setTexture(dilated, index: 3); $0.setTexture(pre, index: 4); $0.setTexture(minCoC, index: 5)
            }
        }
        if fast {
            dispatch(halfP, pw, ph) {
                $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(half, index: 2)
                $0.setTexture(dilated, index: 3); $0.setTexture(pre, index: 4); $0.setTexture(minCoC, index: 5)
                $0.setTexture(quarterOn ? quarterTex : pre, index: 6)
            }
        }
        dispatch(dofP, W, H) {
            $0.setTexture(hdr, index: 0); $0.setTexture(depth, index: 1); $0.setTexture(out, index: 2)
            $0.setTexture(dilated, index: 3); $0.setTexture(pre, index: 4); $0.setTexture(minCoC, index: 5)
            $0.setTexture(fast ? half : pre, index: 6)
            $0.setTexture(quarterOn ? quarterTex : pre, index: 7)
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
        if quarterOn {
            var q = [Float16](repeating: 0, count: qw * qh * 4)
            q.withUnsafeMutableBytes { quarterTex.getBytes($0.baseAddress!, bytesPerRow: qw * 8,
                                                             from: MTLRegionMake2D(0, 0, qw, qh), mipmapLevel: 0) }
            // alpha = the block's SIGNED CoC (0 = declined): a plane behind the focus plane is negative.
            let covered = (0..<(qw * qh)).filter { q[$0 * 4 + 3] != 0 }.count
            lastQuarterCoverage = Float(covered) / Float(qw * qh)
        } else {
            lastQuarterCoverage = 0
        }
        let f = buf.contents().bindMemory(to: Float16.self, capacity: W * H * 4)
        return (0..<(W * H)).map { SIMD3(Float(f[$0 * 4]), Float(f[$0 * 4 + 1]), Float(f[$0 * 4 + 2])) }
    }

    /// Worst relative error (per channel, vs `ref`) over the pixels `keep` selects.
    private func worst(_ img: [SIMD3<Float>], _ ref: SIMD3<Float>, _ keep: (Int, Int) -> Bool) -> Float {
        var e: Float = 0
        for y in 0..<H {
            for x in 0..<W where keep(x, y) {
                let d = simd_abs(img[y * W + x] - ref) / ref
                e = max(e, max(d.x, max(d.y, d.z)))
            }
        }
        return e
    }

    func testSharpSubjectStaysOpaqueAndLeavesTheDefocusedPlaneItsOwnColour() throws {
        try check(subjectZ: 1, subjectCoC: 0)
    }

    /// The same with the subject a hair off the focus plane (CoC ≈ 1.15 px) — a toy just in front
    /// of the focus distance, the Digital Clock case.
    func testNearlyFocusedSubjectStaysOpaque() throws {
        try check(subjectZ: 0.94, subjectCoC: 0.5 * 36 * 0.06 / 0.94)
    }

    /// A subject just BEHIND the focus plane, |CoC| ≈ 2 px: still carries pyramid weight, so the
    /// plane's prefiltered taps beside it must see it as another layer (the Digital Clock's W1).
    func testSlightlyDefocusedFarSubjectDoesNotSmearIntoThePlane() throws {
        try check(subjectZ: 1.125, subjectCoC: 0.5 * 36 * 0.125 / 1.125)
    }

    private func check(subjectZ: Float, subjectCoC c: Float, fast: Bool = false, quarter: Float = 0) throws {
        let fixed = try run(subjectAware: true, subjectZ: subjectZ, fast: fast, quarter: quarter)
        if quarter > 0 {
            XCTAssertGreaterThan(lastQuarterCoverage, 0.3, "the quarter tier must actually serve the plane")
        }
        let legacy = try run(subjectAware: false, subjectZ: subjectZ)
        // The plane beyond the subject's own disc (+ the 1 px anti-aliasing of the iris mask).
        let planeKeep: (Int, Int) -> Bool = { x, y in self.edgeDistance(x, y) >= c + 2.5 }
        // The subject inside its silhouette, beyond the reach of its own disc's edge.
        let subjKeep: (Int, Int) -> Bool = { x, y in self.edgeDistance(x, y) <= -(c + 1.5) }
        let fPlane = worst(fixed, wall, planeKeep), lPlane = worst(legacy, wall, planeKeep)
        let fSubj = worst(fixed, subject, subjKeep), lSubj = worst(legacy, subject, subjKeep)
        print("DOF sharp subject (CoC \(c) px): plane worst — legacy \(lPlane), fixed \(fPlane); subject worst — legacy \(lSubj), fixed \(fSubj)")
        XCTAssertLessThan(fPlane, 0.05, "the defocused plane took on the subject's colour: \(fPlane)")
        XCTAssertLessThan(fSubj, 0.05, "the in-focus subject went see-through: \(fSubj)")
        XCTAssertGreaterThan(max(lPlane, lSubj), 0.25, "the probe must see the legacy gather's fringe")
    }

    /// A plane with nothing in front of it blurs to itself either way.
    func testUniformDefocusedPlaneIsUnchanged() throws {
        let fixed = try run(subjectAware: true)
        let far: (Int, Int) -> Bool = { x, y in self.edgeDistance(x, y) >= 40 }
        XCTAssertLessThan(worst(fixed, wall, far), 0.01)
    }

    // ── Fast gather (VZ-0191): the same invariants, and the full gather's result ─────────────

    func testFastGatherKeepsTheSharpSubjectInvariants() throws {
        try check(subjectZ: 1, subjectCoC: 0, fast: true)
        try check(subjectZ: 0.94, subjectCoC: 0.5 * 36 * 0.06 / 0.94, fast: true)
        try check(subjectZ: 1.125, subjectCoC: 0.5 * 36 * 0.125 / 1.125, fast: true)
    }

    /// Against the full sharp-subject gather on a TEXTURED defocused wall: the exact reach, the
    /// sparser prefiltered ring and the half-res background together stay within a few percent.
    func testFastGatherMatchesTheFullGather() throws {
        let full = try run(subjectAware: true, texturedWall: true)
        let fast = try run(subjectAware: true, fast: true, texturedWall: true)
        var sumRel: Float = 0, maxRel: Float = 0, n = 0
        var subjMax: Float = 0
        for y in 0..<H {
            for x in 0..<W {
                let a = full[y * W + x], b = fast[y * W + x]
                let lum = { (c: SIMD3<Float>) in 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
                let rel = abs(lum(a) - lum(b)) / max(lum(a), 1e-6)
                if edgeDistance(x, y) >= 3 { sumRel += rel; maxRel = max(maxRel, rel); n += 1 }
                if edgeDistance(x, y) <= -2 { subjMax = max(subjMax, rel) }
            }
        }
        let mean = sumRel / Float(max(n, 1))
        print("DOF fast vs full (textured wall): plane mean \(mean), plane max \(maxRel), subject max \(subjMax)")
        XCTAssertLessThan(mean, 0.02, "fast gather's plane drifts from the full gather: mean \(mean)")
        XCTAssertLessThan(maxRel, 0.12, "fast gather's plane worst pixel: \(maxRel)")
        XCTAssertLessThan(subjMax, 0.02, "fast gather changed the in-focus subject: \(subjMax)")
    }

    // ── Quarter-res tier (VZ-0197, `dofQuarterResMinCoC`): the same invariants and result ─────

    /// The quarter tier engages on the defocused plane (9 px here, gate 6) and still keeps the
    /// sharp subject opaque and the plane its own colour — its one-layer tests and the min-CoC
    /// guard hand every block near the silhouette to the half and full tiers.
    func testQuarterTierKeepsTheSharpSubjectInvariants() throws {
        try check(subjectZ: 1, subjectCoC: 0, fast: true, quarter: 6)
        try check(subjectZ: 0.94, subjectCoC: 0.5 * 36 * 0.06 / 0.94, fast: true, quarter: 6)
        try check(subjectZ: 1.125, subjectCoC: 0.5 * 36 * 0.125 / 1.125, fast: true, quarter: 6)
    }

    /// Against the full sharp-subject gather on the TEXTURED wall — the same bounds the fast
    /// gather is held to — and against the fast gather without the tier (what it replaces).
    func testQuarterTierMatchesTheFullGather() throws {
        let full = try run(subjectAware: true, texturedWall: true)
        let fast = try run(subjectAware: true, fast: true, texturedWall: true)
        let quarter = try run(subjectAware: true, fast: true, texturedWall: true, quarter: 6)
        XCTAssertGreaterThan(lastQuarterCoverage, 0.3, "the quarter tier must actually serve the plane")
        let lum = { (c: SIMD3<Float>) in 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
        var sumRel: Float = 0, maxRel: Float = 0, n = 0, subjMax: Float = 0, sumVsFast: Float = 0
        var changed = 0
        for y in 0..<H {
            for x in 0..<W {
                let a = full[y * W + x], b = quarter[y * W + x], c = fast[y * W + x]
                let rel = abs(lum(a) - lum(b)) / max(lum(a), 1e-6)
                if edgeDistance(x, y) >= 3 {
                    sumRel += rel; maxRel = max(maxRel, rel); n += 1
                    sumVsFast += abs(lum(c) - lum(b)) / max(lum(c), 1e-6)
                }
                if edgeDistance(x, y) <= -2 { subjMax = max(subjMax, rel) }
                if b != c { changed += 1 }
            }
        }
        let mean = sumRel / Float(max(n, 1)), meanVsFast = sumVsFast / Float(max(n, 1))
        print("DOF quarter vs full (textured wall): plane mean \(mean), plane max \(maxRel), subject max \(subjMax); vs fast mean \(meanVsFast); coverage \(lastQuarterCoverage)")
        XCTAssertGreaterThan(changed, 0, "the probe must see the quarter tier at work")
        XCTAssertLessThan(mean, 0.02, "quarter tier's plane drifts from the full gather: mean \(mean)")
        XCTAssertLessThan(maxRel, 0.12, "quarter tier's plane worst pixel: \(maxRel)")
        XCTAssertLessThan(subjMax, 0.02, "quarter tier changed the in-focus subject: \(subjMax)")
    }

    /// Off (0) is the fast gather exactly: the new binding is unread and no texel changes.
    func testQuarterTierOffIsTheFastGatherExactly() throws {
        let a = try run(subjectAware: true, fast: true, texturedWall: true)
        let b = try run(subjectAware: true, fast: true, texturedWall: true, quarter: 0)
        XCTAssertEqual(a, b)
    }

    /// Bokeh rims: HDR-bright points on the defocused plane blur into sharp-rimmed discs. The quarter
    /// tier must hand the pixels beside a rim to the finer tiers, so its result stays as close to the
    /// full gather as the fast gather's own is (a 4-px reconstruction would soften and shift the rim).
    func testQuarterTierKeepsBokehRims() throws {
        brightDots = true
        defer { brightDots = false }
        let full = try run(subjectAware: true, texturedWall: true)
        let fast = try run(subjectAware: true, fast: true, texturedWall: true)
        let quarter = try run(subjectAware: true, fast: true, texturedWall: true, quarter: 6)
        let lum = { (c: SIMD3<Float>) in 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
        var fastMax: Float = 0, quarterMax: Float = 0, fastSum: Float = 0, quarterSum: Float = 0, n = 0
        for y in 0..<H {
            for x in 0..<W where edgeDistance(x, y) >= 3 {
                let a = lum(full[y * W + x])
                let ef = abs(a - lum(fast[y * W + x])) / max(a, 1e-6)
                let eq = abs(a - lum(quarter[y * W + x])) / max(a, 1e-6)
                fastMax = max(fastMax, ef); quarterMax = max(quarterMax, eq)
                fastSum += ef; quarterSum += eq; n += 1
            }
        }
        print("DOF bokeh rims vs full: fast mean \(fastSum / Float(n)) max \(fastMax); quarter mean \(quarterSum / Float(n)) max \(quarterMax)")
        XCTAssertLessThan(quarterSum / Float(n), fastSum / Float(n) + 0.005, "the quarter tier drifts from the full gather round bokeh")
        XCTAssertLessThan(quarterMax, max(0.12, fastMax * 1.25), "the quarter tier softened a bokeh rim: \(quarterMax)")
    }
}
