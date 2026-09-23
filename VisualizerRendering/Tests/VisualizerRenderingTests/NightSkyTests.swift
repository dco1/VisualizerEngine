import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// The physical night sky's math, pinned as numbers.
///
/// CPU half — `NightSkyEphemeris` (phase ↔ elongation, the celestial pole, the effective-sun
/// phase construction, air mass / extinction, the lunar phase law).
///
/// GPU half — the REAL shader functions in IlluminatoramaNightSky.h, compiled from source
/// with tiny test kernels (deterministic: pure hash functions of the lattice, no time, no
/// frame state):
///   • the star lattice is EQUAL-AREA (the old lat-long grid put ~1/cos(el) more stars per
///     steradian toward the pole — the zenith clump),
///   • the magnitude distribution has the naked-eye slope, log10 N(<m) ≈ 0.5·m,
///   • the moon's terminator is a lit sphere's (a half moon is lit on exactly one side of
///     its diameter, a gibbous moon's lit fraction is (1 + cos α)/2 — not the old pie-slice),
///   • the shader's extinction agrees with the Swift mirror the hosts use for moonlight.
final class NightSkyTests: XCTestCase {

    // MARK: - CPU: ephemeris

    func testFullMoonOpposesTheSunAndNewMoonSitsOnIt() {
        let full = NightSkyEphemeris(latitudeDeg: 34.7, dayOfYear: 85, localSolarHour: 23,
                                     moonAgeDays: NightSkyEphemeris.synodicMonth / 2)
        XCTAssertEqual(simd_dot(full.sunENU, full.moonENU), -1, accuracy: 1e-6)
        XCTAssertEqual(full.moonIlluminatedFraction, 1, accuracy: 1e-6)
        XCTAssertEqual(Double(full.moonFluxFraction), 1, accuracy: 1e-4)

        let new = NightSkyEphemeris(latitudeDeg: 34.7, dayOfYear: 85, localSolarHour: 12, moonAgeDays: 0)
        XCTAssertEqual(simd_dot(new.sunENU, new.moonENU), 1, accuracy: 1e-6)
        XCTAssertEqual(new.moonIlluminatedFraction, 0, accuracy: 1e-6)

        // First quarter: 90° east of the sun, half lit, ~8–11 % of a full moon's flux
        // (Allen's law — the opposition surge makes a half moon far less than half as bright).
        let q = NightSkyEphemeris(latitudeDeg: 34.7, dayOfYear: 85, localSolarHour: 18,
                                  moonAgeDays: NightSkyEphemeris.synodicMonth / 4)
        XCTAssertEqual(simd_dot(q.sunENU, q.moonENU), 0, accuracy: 1e-6)
        XCTAssertEqual(q.moonIlluminatedFraction, 0.5, accuracy: 1e-6)
        XCTAssertGreaterThan(q.moonFluxFraction, 0.07)
        XCTAssertLessThan(q.moonFluxFraction, 0.12)
    }

    func testSunPathMatchesTheHourAngleModel() {
        // Local noon at the equinox-ish date: the sun due south at 90° − φ + δ.
        let e = NightSkyEphemeris(latitudeDeg: 34.7, dayOfYear: 85, localSolarHour: 12, moonAgeDays: 0)
        let el = asin(e.sunENU.z) * 180 / .pi
        let dec = asin(sin(NightSkyEphemeris.obliquity) * sin(2 * .pi * 5 / 365.2422)) * 180 / .pi
        XCTAssertEqual(el, 90 - 34.7 + dec, accuracy: 1e-6)
        XCTAssertEqual(e.sunENU.x, 0, accuracy: 1e-9)        // due south: no east component
        XCTAssertLessThan(e.sunENU.y, 0)
        // Six hours later it has set in the west.
        let dusk = NightSkyEphemeris(latitudeDeg: 34.7, dayOfYear: 85, localSolarHour: 18, moonAgeDays: 0)
        XCTAssertLessThan(dusk.sunENU.x, -0.9)
        XCTAssertEqual(asin(dusk.sunENU.z) * 180 / .pi, 0, accuracy: 2.5)
    }

    func testCelestialPoleIsFixedAtTheLatitudeAndTheStarsTurnAboutIt() {
        let east = SIMD3<Float>(1, 0, 0), north = SIMD3<Float>(0, 0, -1), up = SIMD3<Float>(0, 1, 0)
        var previousRA0: SIMD3<Float>? = nil
        for hour in stride(from: 0.0, through: 24.0, by: 3.0) {
            let w = NightSkyEphemeris(latitudeDeg: 34.7, dayOfYear: 85, localSolarHour: hour, moonAgeDays: 5)
                .world(east: east, north: north, up: up)
            // The pole: due north at an altitude equal to the latitude, at every hour.
            XCTAssertEqual(asin(w.celestialPole.y) * 180 / .pi, 34.7, accuracy: 1e-3)
            XCTAssertEqual(w.celestialPole.x, 0, accuracy: 1e-5)
            // The quaternion is world → equatorial: the world pole maps to +z.
            let q = simd_quatf(ix: w.celestialOrientation.x, iy: w.celestialOrientation.y,
                               iz: w.celestialOrientation.z, r: w.celestialOrientation.w)
            let p = q.act(w.celestialPole)
            XCTAssertEqual(p.z, 1, accuracy: 1e-5)
            // RA 0h on the equator moves 45° per 3 h (plus the sun's ~1°/day, nil within a day).
            let ra0 = q.inverse.act(SIMD3<Float>(1, 0, 0))
            if let prev = previousRA0 {
                let deg = acos(max(-1, min(1, simd_dot(prev, ra0)))) * 180 / .pi
                XCTAssertEqual(deg, 45, accuracy: 0.2)
            }
            previousRA0 = ra0
            // The world sun direction agrees with the equatorial frame (sun on the ecliptic).
            XCTAssertEqual(simd_length(w.sunToward), 1, accuracy: 1e-5)
        }
    }

    func testEffectiveSunGivesTheRequestedPhaseOnTheSunwardSide() {
        let moon = simd_normalize(SIMD3<Float>(0.3, 0.5, -0.8))
        let trueSun = simd_normalize(SIMD3<Float>(-0.9, -0.3, 0.1))
        for k: Float in [0.05, 0.25, 0.5, 0.75, 0.85, 0.98] {
            let L = NightSkyEphemeris.effectiveSun(moonDir: moon, trueToSun: trueSun, illuminatedFraction: k)
            XCTAssertEqual(simd_length(L), 1, accuracy: 1e-5)
            XCTAssertEqual(0.5 * (1 + simd_dot(L, -moon)), k, accuracy: 1e-4, "phase \(k)")
            // Lit limb faces the true sun: L's in-disk component points the same way.
            let tSun = trueSun - simd_dot(trueSun, moon) * moon
            let tL = L - simd_dot(L, moon) * moon
            XCTAssertGreaterThan(simd_dot(tSun, tL), 0)
        }
    }

    func testAirmassAndExtinction() {
        XCTAssertEqual(NightSkyEphemeris.airmass(sinElevation: 1), 1, accuracy: 1e-3)
        XCTAssertEqual(NightSkyEphemeris.airmass(sinElevation: sin(30 * .pi / 180)), 2, accuracy: 0.01)
        XCTAssertEqual(NightSkyEphemeris.airmass(sinElevation: 0), 38.1, accuracy: 0.5)
        var last = SIMD3<Float>(repeating: 2)
        for deg in stride(from: 90.0, through: 0.0, by: -5.0) {
            let t = NightSkyEphemeris.extinction(sinElevation: Float(sin(deg * .pi / 180)))
            XCTAssertTrue(all(t .<= last), "extinction monotonic toward the horizon")
            XCTAssertLessThan(t.z, t.x, "blue is extinguished more than red — low stars redden")
            last = t
        }
        // ~0.19 mag in V at the zenith; gone (> 7 mag) at the horizon.
        let zen = NightSkyEphemeris.extinction(sinElevation: 1)
        XCTAssertEqual(-2.5 * log10(zen.y), 0.19, accuracy: 0.005)
        XCTAssertGreaterThan(-2.5 * log10(NightSkyEphemeris.extinction(sinElevation: 0).y), 7)
    }

    // MARK: - GPU: the shader functions themselves

    private struct GPU {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let library: MTLLibrary
    }

    private func gpu(_ kernels: String) throws -> GPU {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/IlluminatoramaNightSky.h")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let header = try MetalSourceLoader.source(contentsOf: url)
        let library = try device.makeLibrary(source: header + "\n" + kernels, options: nil)
        return GPU(device: device, queue: queue, library: library)
    }

    private func run(_ g: GPU, _ name: String, threads: Int, buffers: [MTLBuffer]) throws {
        guard let fn = g.library.makeFunction(name: name) else { return XCTFail("\(name) missing") }
        let pso = try g.device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        guard let cb = g.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw XCTSkip("no encoder")
        }
        enc.setComputePipelineState(pso)
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()  // gpu-ok: test harness
        if let e = cb.error { throw e }
    }

    /// Enumerate every star the lattice defines (one thread per lattice cell in the shell's
    /// bounding cube), with its direction and magnitude.
    private func allStars() throws -> [(dir: SIMD3<Float>, mag: Float)] {
        let g = try gpu("""
        kernel void enumerateStars(device float4* out [[buffer(0)]],
                                   device atomic_uint* count [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]]) {
            const int N = 99;                              // cells −49 … 49 per axis
            int3 c = int3(int(tid % N), int((tid / N) % N), int(tid / (N * N))) - 49;
            float3 centre = float3(c) + 0.5f;
            if (fabs(length(centre) - kStarShellR) > 1.5f) return;
            uint ch = nightCellHash(c);
            int n = nightStarCount(c, ch);
            for (int i = 0; i < n; ++i) {
                NightStar s = nightStarInCell(c, ch, i);
                if (!s.valid) continue;
                uint k = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
                if (k < 60000u) out[k] = float4(s.dir, s.mag);
            }
        }
        """)
        let cap = 60000
        guard let out = g.device.makeBuffer(length: cap * 16, options: .storageModeShared),
              let count = g.device.makeBuffer(length: 4, options: .storageModeShared) else { throw XCTSkip("no buffers") }
        memset(count.contents(), 0, 4)
        try run(g, "enumerateStars", threads: 99 * 99 * 99, buffers: [out, count])
        let n = min(Int(count.contents().load(as: UInt32.self)), cap)
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: cap)
        return (0..<n).map { (SIMD3(p[$0].x, p[$0].y, p[$0].z), p[$0].w) }
    }

    func testStarLatticeIsEqualAreaAndHasTheNakedEyeMagnitudeSlope() throws {
        let stars = try allStars()
        // ~23 160 expected (λ·R²·2h·4π); Poisson scatter is < 1 %.
        XCTAssertEqual(Double(stars.count), 23_160, accuracy: 23_160 * 0.03, "total star count")

        // Equal area: count per equal-solid-angle band of the EQUATORIAL z (sin dec) — ten bands
        // of 0.2 each, from the south pole cap to the north. Undo the deliberate galactic-plane
        // boost first (each star weighted by 1 / its expected boost) so only the placement is
        // tested. The old lat-long grid gives the polar bands many times the equatorial ones.
        let galN = SIMD3<Float>(-0.86768, -0.19804, 0.45598)
        var bands = [Double](repeating: 0, count: 10)
        for s in stars {
            let b = asin(max(-1, min(1, simd_dot(s.dir, galN))))
            let boost = (1 + 2.0 * exp(-(b * b) / (0.21 * 0.21))) / 1.368
            bands[min(9, Int((s.dir.z + 1) / 0.2))] += 1 / Double(boost)
        }
        let mean = bands.reduce(0, +) / 10
        for (i, n) in bands.enumerated() {
            XCTAssertEqual(n / mean, 1, accuracy: 0.12, "band \(i) (sin dec \(Double(i) * 0.2 - 1)…) density")
        }
        // The polar caps specifically (the old zenith clump lived there).
        XCTAssertEqual(bands[9] / bands[4], 1, accuracy: 0.15, "pole cap vs equator")

        // Magnitudes: log10 N(<m) rises 0.5 per magnitude (±0.06) between m = 2 and 6, and the
        // sky holds ~5000 stars brighter than 6 like the real one.
        func n(_ m: Float) -> Double { Double(stars.filter { $0.mag < m }.count) }
        let slope = (log10(n(6)) - log10(n(2))) / 4
        XCTAssertEqual(slope, 0.5, accuracy: 0.06, "magnitude-count slope")
        XCTAssertEqual(n(6), 5000, accuracy: 750, "stars brighter than m = 6")
        XCTAssertGreaterThanOrEqual(stars.map(\.mag).min() ?? 99, -1.47, "nothing brighter than Sirius")
        XCTAssertLessThanOrEqual(stars.map(\.mag).max() ?? 0, 7.31)
    }

    /// Render the moon disk onto a grid of rays and read back its radiance.
    private func moonGrid(g: GPU, toSun: SIMD3<Float>, size: Int) throws -> [Float] {
        guard let out = g.device.makeBuffer(length: size * size * 4, options: .storageModeShared),
              let sun = g.device.makeBuffer(bytes: [SIMD4<Float>(toSun, 0)], length: 16, options: .storageModeShared)
        else { throw XCTSkip("no buffers") }
        var sz = UInt32(size)
        guard let szb = g.device.makeBuffer(bytes: &sz, length: 4, options: .storageModeShared) else { throw XCTSkip("no buffers") }
        try run(g, "moonGrid", threads: size * size, buffers: [out, sun, szb])
        let p = out.contents().bindMemory(to: Float.self, capacity: size * size)
        return Array(UnsafeBufferPointer(start: p, count: size * size))
    }

    private let moonKernel = """
    // The moon straight overhead (no extinction), north = +z world, radius 0.2 rad so the grid
    // resolves it; earthshine, aureole off — only the sunlit hemisphere shows.
    kernel void moonGrid(device float* out [[buffer(0)]], constant float4& sun [[buffer(1)]],
                         constant uint& size [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
        uint x = tid % size, y = tid / size;
        float2 q = (float2(float(x), float(y)) + 0.5f) / float(size) * 2.0f - 1.0f;   // −1 … 1 of the radius
        NightSkyParams p = nightSkyOff();
        p.model = 1; p.moonIntensity = 1; p.moonAngRadius = 0.2f;
        p.moonDir = float3(0, 1, 0); p.toSun = sun.xyz;
        float3 f = p.moonDir;
        float3 N = float3(0, 0, 1);                 // celestial north (identity frame)
        float3 R = cross(f, N);
        float s = sin(0.2f);
        float3 ray = normalize(f * sqrt(max(0.0f, 1.0f - s * s * dot(q, q))) + (q.x * R + q.y * N) * s);
        out[tid] = nightMoonPhysical(ray, p, N, 1e-4f).g;
    }
    """

    func testMoonTerminatorIsALitSphereNotAPieSlice() throws {
        let g = try gpu(moonKernel)
        let size = 96
        let f = SIMD3<Float>(0, 1, 0)
        let R = simd_cross(f, SIMD3<Float>(0, 0, 1))     // +x on the disk grid = −x world
        for alphaDeg: Float in [30, 60, 90, 120, 150] {
            let a = alphaDeg * .pi / 180
            // Sun direction making phase angle α with the moon→Earth direction, lit side +x.
            let toSun = simd_normalize(-cos(a) * f + sin(a) * R)
            let img = try moonGrid(g: g, toSun: toSun, size: size)
            var inDisk = 0, lit = 0
            var halfViolations = 0
            for y in 0..<size {
                for x in 0..<size {
                    let q = (SIMD2<Float>(Float(x), Float(y)) + 0.5) / Float(size) * 2 - 1
                    guard simd_length(q) < 0.97 else { continue }
                    inDisk += 1
                    let v = img[y * size + x]
                    if v > 1e-3 { lit += 1 }
                    // Every point lit by a lit sphere lies on the sunward side of the disk
                    // centre line x = −cos α·… : for α ≤ 90° the whole +x half is lit, for
                    // α ≥ 90° the whole −x half is dark. The old moon lit a sector of angles.
                    if alphaDeg <= 90 && q.x > 0.08 && v <= 1e-3 { halfViolations += 1 }
                    if alphaDeg >= 90 && q.x < -0.08 && v > 1e-3 { halfViolations += 1 }
                }
            }
            let expected = 0.5 * (1 + cos(a))   // illuminated fraction of the disk AREA
            XCTAssertEqual(Float(lit) / Float(inDisk), expected, accuracy: 0.04, "lit fraction at α = \(alphaDeg)°")
            XCTAssertEqual(halfViolations, 0, "terminator side at α = \(alphaDeg)°")
        }
    }

    func testMoonDiskIsFiniteEverywhereIncludingItsCentre() throws {
        // The aureole's θ = √(2(1 − cos θ)) went NaN where cos θ rounds past 1 — a black pixel
        // at the disk centre. Sweep rays through and around the centre.
        let g = try gpu("""
        kernel void celestialsNearMoon(device float4* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
            NightSkyParams p = nightSkyOff();
            p.model = 1; p.moonIntensity = 1; p.moonHalo = 1; p.earthshine = 1; p.radiance = 2.4e-6f;
            p.starBrightness = 1; p.milkyWay = 1; p.twinkle = 1; p.clock = 3.0f;
            p.moonDir = normalize(float3(0.2f, 0.5f, -0.8f)); p.toSun = normalize(float3(-0.5f, -0.4f, 0.7f));
            float o = (float(tid) - 512.0f) * 2e-6f;   // ±1e-3 rad around the centre, sub-ulp near 0
            float3 ray = normalize(p.moonDir + float3(o, 0.0f, 0.3f * o));
            out[tid] = float4(nightCelestials(ray, p, 6e-5f, float3(0.003f)), 0);
        }
        """)
        guard let out = g.device.makeBuffer(length: 1024 * 16, options: .storageModeShared) else { throw XCTSkip("no buffers") }
        try run(g, "celestialsNearMoon", threads: 1024, buffers: [out])
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: 1024)
        for i in 0..<1024 {
            XCTAssertTrue(p[i].x.isFinite && p[i].y.isFinite && p[i].z.isFinite, "non-finite at \(i)")
            XCTAssertGreaterThan(p[i].y, 0.05, "the disk is lit at \(i)")
        }
    }

    func testShaderExtinctionMatchesTheSwiftMirror() throws {
        let g = try gpu("""
        kernel void ext(device float4* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
            float s = float(tid) / 63.0f;
            out[tid] = float4(nightExtinction(s), nightAirmass(s));
        }
        """)
        guard let out = g.device.makeBuffer(length: 64 * 16, options: .storageModeShared) else { throw XCTSkip("no buffers") }
        try run(g, "ext", threads: 64, buffers: [out])
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: 64)
        for i in 0..<64 {
            let s = Float(i) / 63
            let t = NightSkyEphemeris.extinction(sinElevation: s)
            XCTAssertEqual(p[i].w, NightSkyEphemeris.airmass(sinElevation: s), accuracy: 0.02 * p[i].w)
            XCTAssertEqual(p[i].x, t.x, accuracy: 2e-3)
            XCTAssertEqual(p[i].z, t.z, accuracy: 2e-3)
        }
    }

    func testZeroedParamsAreANoOpInBothModels() throws {
        let g = try gpu("""
        kernel void zeroed(device float4* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
            float a = float(tid) * 0.37f;
            float3 ray = normalize(float3(cos(a), 0.2f + 0.8f * fract(a), sin(a)));
            NightSkyParams p = nightSkyOff();
            float3 legacy = nightCelestials(ray, p, 1e-3f, float3(0.01f));
            p.model = 1;
            float3 physical = nightCelestials(ray, p, 1e-3f, float3(0.01f));
            out[tid] = float4(legacy, dot(physical, 1.0f));
        }
        """)
        guard let out = g.device.makeBuffer(length: 256 * 16, options: .storageModeShared) else { throw XCTSkip("no buffers") }
        try run(g, "zeroed", threads: 256, buffers: [out])
        let p = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: 256)
        for i in 0..<256 { XCTAssertEqual(p[i], .zero, "ray \(i)") }
    }

    // MARK: - The dome kernel (VolumetricSky.metal `volSkyRender`), end to end

    /// Render the real sky-dome kernel from source with the renderer's own `SkyUniforms`.
    private func renderDome(_ params: VolumetricCloudRenderer.Params, W: Int = 2048, H: Int = 1024) throws
        -> (W: Int, H: Int, px: [Float16]) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no queue") }  // gpu-ok: test harness
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/VolumetricSky.metal")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no shader") }
        let lib = try MetalSourceLoader.makeLibrary(device: device, contentsOf: url)
        guard let fn = lib.makeFunction(name: "volSkyRender") else { throw XCTSkip("volSkyRender missing") }
        let pipeline = try device.makeComputePipelineState(function: fn)  // gpu-ok: test harness
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: W, height: H, mipmapped: false)
        td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
        let nd = MTLTextureDescriptor()
        nd.textureType = .type3D; nd.pixelFormat = .rgba16Float
        nd.width = 4; nd.height = 4; nd.depth = 4; nd.usage = [.shaderRead]; nd.storageMode = .shared
        var uniforms = SkyUniforms(params: params, time: 0)
        guard let target = device.makeTexture(descriptor: td), let noise = device.makeTexture(descriptor: nd),
              let uBuf = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<SkyUniforms>.stride, options: .storageModeShared),
              let lightBuf = device.makeBuffer(length: 256, options: .storageModeShared),
              let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder()
        else { throw XCTSkip("no resources") }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(target, index: 0); enc.setTexture(noise, index: 1)
        enc.setBuffer(uBuf, offset: 0, index: 0); enc.setBuffer(lightBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()  // gpu-ok: test harness
        var px = [Float16](repeating: 0, count: W * H * 4)
        px.withUnsafeMutableBytes { raw in
            target.getBytes(raw.baseAddress!, bytesPerRow: W * 8, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
        }
        return (W, H, px)
    }

    /// Radiance of the dome texel a world direction lands on (the writer's convention:
    /// u = atan2(z, x) / 2π, v = 0.5 − asin(y) / π).
    private func domeSample(_ d: (W: Int, H: Int, px: [Float16]), _ dir: SIMD3<Float>) -> SIMD3<Float> {
        let n = simd_normalize(dir)
        var u = atan2(n.z, n.x) / (2 * .pi); if u < 0 { u += 1 }
        let v = 0.5 - asin(n.y) / .pi
        let x = min(d.W - 1, Int(u * Float(d.W))), y = min(d.H - 1, Int(v * Float(d.H)))
        let i = (y * d.W + x) * 4
        return SIMD3(Float(d.px[i]), Float(d.px[i + 1]), Float(d.px[i + 2]))
    }

    private func nightParams() -> VolumetricCloudRenderer.Params {
        var p = VolumetricCloudRenderer.Params()
        p.sunDir = simd_normalize(SIMD3<Float>(0.3, 0.8, 0.2))     // sun ~53° below the horizon: full night
        p.coverage = 0; p.density = 0; p.cirrusCoverage = 0
        p.starBrightness = 0
        p.moonIntensity = 1
        p.moonDir = simd_normalize(SIMD3<Float>(0.0, 0.5, -0.866))  // 30° up
        return p
    }

    /// The LEGACY dome moon (what NapaValley / Ufo / FireworksUltra / City Street Ultra bake) is
    /// now the shared sphere-lit disk: a half moon is lit on the sunward half of its diameter and
    /// dark on the other — the old kernel lit a pie-slice of ANGLES around the centre, with a
    /// flat dot at the centre. It keeps its historical ~3.2° size.
    func testLegacyDomeMoonIsASphereLitHalfMoonAtItsHistoricalSize() throws {
        var p = nightParams()
        p.moonPhase = 0.5
        let dome = try renderDome(p)
        let m = simd_normalize(p.moonDir)
        let toSun = -simd_normalize(p.sunDir)
        let t = simd_normalize(toSun - simd_dot(toSun, m) * m)          // the sunward side of the disk
        let b = simd_normalize(simd_cross(m, t))                        // along the terminator
        let sky = domeSample(dome, simd_normalize(m + 0.12 * t))        // ~7°: well outside the disk
        func at(_ deg: Float, _ axis: SIMD3<Float>) -> Float {
            simd_reduce_add(domeSample(dome, simd_normalize(m + tan(deg * .pi / 180) * axis)))
        }
        // Lit half: bright across it — not only near the old wedge.
        for deg: Float in [0.8, 1.6, 2.4] {
            XCTAssertGreaterThan(at(deg, t), simd_reduce_add(sky) + 0.3, "lit half at \(deg)°")
            XCTAssertLessThan(at(deg, -t), simd_reduce_add(sky) + 0.15, "dark half at \(deg)°")
        }
        // Along the terminator (a half moon's is a straight diameter) the lit side of every
        // row stays lit: sample 0.8° sunward of the terminator at points up and down it.
        for deg: Float in [-2.0, -1.0, 1.0, 2.0] {
            let d = simd_normalize(m + tan(deg * .pi / 180) * b + tan(0.8 * .pi / 180) * t)
            XCTAssertGreaterThan(simd_reduce_add(domeSample(dome, d)), simd_reduce_add(sky) + 0.2, "row \(deg)°")
        }
        // Historical size: lit at 2.8°, sky by 4°.
        XCTAssertGreaterThan(at(2.8, t), simd_reduce_add(sky) + 0.2)
        XCTAssertLessThan(at(4.2, t), simd_reduce_add(sky) + 0.05)

        // Gibbous (0.85 — Superbloom's old value, where the Pac-Man showed): cos α = 0.7, so the
        // terminator crosses the anti-sun axis at 0.7 of the radius. Everything nearer the centre
        // is lit — the old kernel cut a dark WEDGE from the far limb right into the centre.
        p.moonPhase = 0.85
        let gib = try renderDome(p)
        let r = NightSkyEphemeris.legacyDomeMoonAngularRadius * 180 / .pi
        func g(_ frac: Float, _ axis: SIMD3<Float>) -> Float {
            simd_reduce_add(domeSample(gib, simd_normalize(m + tan(frac * r * .pi / 180) * axis)))
        }
        for frac: Float in [0.0, 0.25, 0.5] {
            XCTAssertGreaterThan(g(frac, -t), simd_reduce_add(sky) + 0.2, "gibbous lit at \(frac) R anti-sunward")
            XCTAssertGreaterThan(g(frac, b), simd_reduce_add(sky) + 0.2, "gibbous lit at \(frac) R along the terminator axis")
        }
        XCTAssertLessThan(g(0.88, -t), simd_reduce_add(sky) + 0.1, "gibbous dark beyond the terminator")
    }

    /// The PHYSICAL model's sky glow: a moonlit sky is BLUE and far brighter than a moonless
    /// one (the nishita march with the moon as the light), and a moonless sky still carries
    /// airglow — it is dark, not zero.
    func testPhysicalDomeMoonlitSkyIsBlueAndBrighterThanMoonless() throws {
        var p = nightParams()
        p.nightSkyModel = .physical
        p.nightRadiance = 2.4e-6
        p.celestialsInDome = false
        p.moonDir = -simd_normalize(p.sunDir) * SIMD3(-1, -1, -1)  // opposite the sun: a full moon
        let lit = try renderDome(p, W: 512, H: 256)
        var dark = p; dark.moonIntensity = 0
        let moonless = try renderDome(dark, W: 512, H: 256)
        let zenith = SIMD3<Float>(0.05, 1, 0.02)
        let zl = domeSample(lit, zenith), zd = domeSample(moonless, zenith)
        XCTAssertGreaterThan(zd.y, 0, "moonless sky keeps its airglow")
        XCTAssertGreaterThan(zl.y, zd.y * 5, "a full moon lights the sky (\(zl) vs \(zd))")
        XCTAssertGreaterThan(zl.z, zl.x, "moonlit sky is Rayleigh blue (\(zl))")
        // The legacy model adds no glow at all (byte-identical sky for non-opting hosts).
        var legacy = nightParams(); legacy.celestialsInDome = false
        let l = try renderDome(legacy, W: 512, H: 256)
        var legacyNoMoon = legacy; legacyNoMoon.moonIntensity = 0
        let l0 = try renderDome(legacyNoMoon, W: 512, H: 256)
        XCTAssertEqual(domeSample(l, zenith), domeSample(l0, zenith), "legacy: the moon adds no sky glow")
    }
}
