import Foundation
import simd
import VisualizerMaterials
import VisualizerRendering
import VisualizerVegetation

/// A streamed field of California wildflowers on a host's ground.
///
/// What it draws, by distance from the camera:
///  - **near** — every plant is a GPU instance of one baked library variant (full detail within
///    `fullRadius`, a lite twin with most leaflets dropped past it), built per 8 m quadrant so a
///    quadrant's LOD only changes when the camera crosses ~8 m;
///  - **mid** — each 16 m cell is ONE merged mesh of cards derived from the same variants, thinned
///    with distance (survivors widen so the covered fraction is conserved);
///  - nothing past `midRadius` — the host's own ground and grass carry the colour there.
///
/// Cells outside the view (four inward planes through the camera, widened) are not shown, and
/// cells within `alwaysKeepRadius` always are (shadows, a quick turn). The scatter is a pure
/// function of (cell, density, seed), built on background threads, so scrolling the camera never
/// re-rolls a plant.
///
/// Usage: create with the renderer + the host's ground; each frame call `update()`, then add
/// `refs` to the frame's instances. All geometry is single-sided data the engine draws like any
/// other instance, so shadows, RT and TAA treat it as scene content.
@MainActor
public final class FloraField {

    public struct Config: Sendable {
        /// Sowing density multiplier (1 = the engine's natural ~7 plants/m²).
        public var density: Double = 1
        public var seed: UInt64 = 0x5EB1_0000_CA11
        /// World-space rectangle (x0, z0, x1, z1) the field may grow in.
        public var bounds: SIMD4<Double>
        /// Closest-point distance within which cells are real instanced plants.
        public var nearRadius: Double = 22
        /// Quadrants this close get the full variants; the rest the lite twins.
        public var fullRadius: Double = 4
        /// Cells past this are not drawn.
        public var midRadius: Double = 240
        /// Cells this close are kept even outside the view cone.
        public var alwaysKeepRadius: Double = 7
        /// Per-instance wind gate (0 = still).
        public var windScale: Float = 1
        /// Every plant this many times its natural size (a host's "height" dial).
        public var heightScale: Double = 1
        /// Keep the instanced plants out of the ray-traced scene (raster + shadow maps still draw
        /// them). Thousands of plants in the acceleration structure cost a rebuild per camera move.
        public var rtExclude = false
        public init(bounds: SIMD4<Double>) { self.bounds = bounds }
    }

    /// The host's surface height at world (x, z), and how much of the field it wants there
    /// (0 = none: a slab, a street; 1 = full). Called from background threads.
    public typealias Ground = @Sendable (Double, Double) -> Double
    public typealias Coverage = @Sendable (Double, Double) -> Float

    public private(set) var config: Config
    private let renderer: IlluminatoramaRenderer
    private let ground: Ground
    private let coverage: Coverage

    /// The instances to draw this frame. Valid after `update()`.
    public private(set) var refs: [IlluminatoramaRenderer.InstanceRef] = []
    /// Plants drawn as real instances, and mid cells drawn as cards (test/diagnostic).
    public private(set) var nearInstanceCount = 0
    public private(set) var midCellCount = 0

    /// Where each instanced plant stands (the mid cards come first in `refs` and are cell-local).
    public var nearPlantPositions: [SIMD3<Float>] {
        refs.dropFirst(midCellCount).map { SIMD3($0.data.modelMatrix.columns.3.x, $0.data.modelMatrix.columns.3.y, $0.data.modelMatrix.columns.3.z) }
    }

    // ── Streaming state ──
    private let jobs: FloraJobQueue
    private var kinds: [Int: (full: IlluminatoramaRenderer.MeshKind, lite: IlluminatoramaRenderer.MeshKind)] = [:]
    private var handles: [IlluminatoramaMeshHandle] = []   // keeps the registered meshes alive
    private var cells: [FloraCellKey: [FloraPlacement]] = [:]
    private var cellPending: Set<FloraCellKey> = []
    private struct MidKey: Hashable, Comparable {
        var cell: FloraCellKey, level: Int
        static func < (a: Self, b: Self) -> Bool { (a.cell, a.level) < (b.cell, b.level) }
    }
    private var midReady: [MidKey: IlluminatoramaMeshHandle?] = [:]
    private var midPending: Set<MidKey> = []
    private struct NearKey: Hashable, Comparable {
        var cell: FloraCellKey, quadrant: Int, full: Bool
        static func < (a: Self, b: Self) -> Bool { (a.cell, a.quadrant, a.full ? 1 : 0) < (b.cell, b.quadrant, b.full ? 1 : 0) }
    }
    private var nearCache: [NearKey: [IlluminatoramaRenderer.InstanceRef]] = [:]
    private struct Composition: Equatable { var mid: [MidKey] = []; var near: [NearKey] = [] }
    private var composition = Composition()
    private var lastScan: SIMD4<Double>?
    private var scanDirty = true
    private static let maxJobsInFlight = 16

    public init?(renderer: IlluminatoramaRenderer, config: Config, ground: @escaping Ground, coverage: @escaping Coverage) {
        self.renderer = renderer
        self.config = config
        self.ground = ground
        self.coverage = coverage
        self.jobs = FloraJobQueue(maxConcurrent: max(2, ProcessInfo.processInfo.activeProcessorCount - 3))
        // Register every library variant (full + lite) once. The bake is process-wide and cached.
        for (slot, plant) in FloraLibrary.plants.sorted(by: { $0.key < $1.key }) {
            guard !plant.full.vertices.isEmpty else { continue }
            let full = renderer.registerMesh(IlluminatoramaMesh(device: renderer.device, vertices: plant.full.vertices,
                                                                indices: plant.full.indices))
            let lite = renderer.registerMesh(IlluminatoramaMesh(device: renderer.device, vertices: plant.lite.vertices,
                                                                indices: plant.lite.indices))
            handles += [full, lite]
            kinds[slot] = (full.kind, lite.kind)
        }
        if kinds.isEmpty { return nil }
    }

    /// Re-sow at a new density (drops every cache; cells rebuild on the next `update()`).
    public func setDensity(_ d: Double) {
        guard d != config.density else { return }
        config.density = d
        cells.removeAll(); midReady.removeAll(); nearCache.removeAll()
        cellPending.removeAll(); midPending.removeAll()
        jobs.cancelAll()
        generation += 1
        scanDirty = true
    }
    private var generation = 0

    // MARK: - Per frame

    /// One streaming step. Returns true when `refs` changed. Cheap when the eye hasn't moved: the
    /// visibility scan only reruns after ~0.5 m of travel or ~1° of turn, or a finished job.
    @discardableResult
    public func update() -> Bool {
        let cam = renderer.camera
        let eye = cam.position
        let drained = drain()
        if drained { scanDirty = true }
        let f = simd_normalize(cam.target - cam.position)
        let now = SIMD4<Double>(Double(eye.x), Double(eye.y), Double(eye.z), Double(atan2(f.x, f.z)) + Double(f.y) * 7)
        if let last = lastScan, !scanDirty,
           simd_length(SIMD3(now.x - last.x, now.y - last.y, now.z - last.z)) < 0.5, abs(now.w - last.w) < 0.017,
           jobs.outstanding >= Self.maxJobsInFlight / 2 || (cellPending.isEmpty && midPending.isEmpty) {
            return false
        }
        lastScan = now
        scanDirty = false

        let planes = viewPlanes(cam)
        let C = Double(FloraScatter.cellSize)
        let b = config.bounds
        let ex = Double(eye.x), ez = Double(eye.z)
        // The cells the field may occupy, clipped to the radius around the eye.
        let reach = config.midRadius
        let ix0 = Int((max(b.x, ex - reach) / C).rounded(.down)), ix1 = Int((min(b.z, ex + reach) / C).rounded(.down))
        let iz0 = Int((max(b.y, ez - reach) / C).rounded(.down)), iz1 = Int((min(b.w, ez + reach) / C).rounded(.down))
        var want: [(priority: Double, submit: () -> Void)] = []
        var next = Composition()
        if ix0 <= ix1, iz0 <= iz1 {
            for iz in iz0...iz1 {
                for ix in ix0...ix1 {
                    let key = FloraCellKey(ix: ix, iz: iz)
                    let x0 = Double(ix) * C, z0 = Double(iz) * C
                    let d = Self.closest(ex, ez, x0, z0, C)
                    guard d <= config.midRadius else { continue }
                    let near = d <= config.nearRadius
                    if d > config.alwaysKeepRadius, !inView(planes, x: x0 + C / 2, z: z0 + C / 2, y: Double(eye.y), radius: C * 0.72 + 2) {
                        continue
                    }
                    guard let placements = cells[key] else {
                        if !cellPending.contains(key) {
                            want.append((d * 0.5, { [weak self] in self?.requestCell(key) }))
                        }
                        continue
                    }
                    if near {
                        for q in 0..<4 {
                            let qx0 = x0 + Double(q % 2) * C / 2, qz0 = z0 + Double(q / 2) * C / 2
                            let full = Self.closest(ex, ez, qx0, qz0, C / 2) < config.fullRadius
                            next.near.append(NearKey(cell: key, quadrant: q, full: full))
                        }
                        continue
                    }
                    let cx = x0 + C / 2 - ex, cz = z0 + C / 2 - ez
                    guard let level = FloraScatter.midLevel((cx * cx + cz * cz).squareRoot(), radius: config.midRadius) else { continue }
                    let mk = MidKey(cell: key, level: level)
                    if midReady[mk] != nil { next.mid.append(mk); continue }
                    if !midPending.contains(mk) {
                        want.append((d + 30, { [weak self] in self?.requestMid(mk, placements) }))
                    }
                    // Until this level is built, show whichever level of the cell is ready.
                    if let alt = (0..<FloraScatter.midKeep.count).map({ MidKey(cell: key, level: $0) })
                        .first(where: { midReady[$0] != nil }) { next.mid.append(alt) }
                }
            }
        }
        let budget = Self.maxJobsInFlight - jobs.outstanding
        if budget > 0 { for w in want.sorted(by: { $0.priority < $1.priority }).prefix(budget) { w.submit() } }
        next.mid.sort(); next.near.sort()
        var changed = false
        if next != composition {
            composition = next
            assemble()
            changed = true
        }
        evict()
        return changed || drained
    }

    /// Block until the visible field is streamed (or `timeout` s): a headless capture must not
    /// photograph bare ground.
    public func warm(timeout: Double = 8) {
        let start = Date()
        while Date().timeIntervalSince(start) < timeout {
            scanDirty = true
            update()
            if cellPending.isEmpty && midPending.isEmpty && jobs.outstanding == 0 { update(); break }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    // MARK: - Jobs

    private func requestCell(_ key: FloraCellKey) {
        cellPending.insert(key)
        let density = config.density, seed = config.seed, gen = generation, scale = config.heightScale
        let ground = ground, coverage = coverage
        jobs.submit {
            .cell(key, FloraScatter.scatter(key, density: density, seed: seed, heightScale: scale, ground: ground, coverage: coverage), gen)
        }
    }

    private func requestMid(_ mk: MidKey, _ placements: [FloraPlacement]) {
        midPending.insert(mk)
        let gen = generation
        jobs.submit {
            let data = FloraScatter.buildMid(cell: mk.cell, level: mk.level, placements: placements)
            return .mid(mk.cell, mk.level, data.indices.isEmpty ? nil : data, gen)
        }
    }

    /// Land finished jobs. True when anything arrived.
    private func drain() -> Bool {
        let results = jobs.drain()
        for r in results {
            switch r {
            case .cell(let k, let ps, let gen):
                guard gen == generation else { continue }
                cellPending.remove(k)
                cells[k] = ps
            case .mid(let cell, let level, let data, let gen):
                guard gen == generation else { continue }
                let mk = MidKey(cell: cell, level: level)
                midPending.remove(mk)
                midReady[mk] = data.map { d in
                    renderer.registerMesh(IlluminatoramaMesh(device: renderer.device, vertices: d.vertices, indices: d.indices))
                }
            }
        }
        return !results.isEmpty
    }

    // MARK: - Assembly

    private func assemble() {
        var out: [IlluminatoramaRenderer.InstanceRef] = []
        out.reserveCapacity(refs.count + 1024)
        let C = Double(FloraScatter.cellSize)
        var mids = 0
        for k in composition.mid {
            guard let h = midReady[k] ?? nil else { continue }
            var m = matrix_identity_float4x4
            m.columns.3 = SIMD4(Float(Double(k.cell.ix) * C), 0, Float(Double(k.cell.iz) * C), 1)
            var inst = IlluminatoramaInstance(modelMatrix: m, albedo: SIMD3(1, 1, 1), metallic: 0, roughness: 0.72)
            if config.rtExclude { inst.rtExclude = IlluminatoramaInstance.rtExcludeFully }
            out.append(.init(meshKind: h.kind, data: inst))
            mids += 1
        }
        var near = 0
        for k in composition.near {
            if nearCache[k] == nil, let ps = cells[k.cell] {
                var refs: [IlluminatoramaRenderer.InstanceRef] = []
                let qx0 = Double(k.cell.ix) * C + Double(k.quadrant % 2) * C / 2
                let qz0 = Double(k.cell.iz) * C + Double(k.quadrant / 2) * C / 2
                for p in ps where p.position.x >= qx0 && p.position.x < qx0 + C / 2
                    && p.position.z >= qz0 && p.position.z < qz0 + C / 2 {
                    guard let kinds = kinds[p.slot] else { continue }
                    let c = cos(p.heading) * p.scale, s = sin(p.heading) * p.scale
                    // The rotation `WildflowerPlantingMesh.layers` seats a plant with.
                    let m = simd_float4x4(columns: (SIMD4(c, 0, -s, 0), SIMD4(0, p.scale, 0, 0), SIMD4(s, 0, c, 0),
                                                    SIMD4(Float(p.position.x), Float(p.position.y), Float(p.position.z), 1)))
                    var inst = IlluminatoramaInstance(modelMatrix: m, albedo: SIMD3(repeating: p.tint),
                                                      metallic: 0, roughness: 0.62)
                    inst.windScale = config.windScale
                    if config.rtExclude { inst.rtExclude = IlluminatoramaInstance.rtExcludeFully }
                    refs.append(.init(meshKind: k.full ? kinds.full : kinds.lite, data: inst))
                }
                nearCache[k] = refs
            }
            if let r = nearCache[k] { out.append(contentsOf: r); near += r.count }
        }
        if nearCache.count > 4 * composition.near.count + 64 {
            let keep = Set(composition.near)
            nearCache = nearCache.filter { keep.contains($0.key) }
        }
        nearInstanceCount = near
        midCellCount = mids
        refs = out
    }

    private func evict() {
        guard cells.count > 900 || midReady.count > 500 else { return }
        let shown = Set(composition.mid)
        if midReady.count > 500 { for k in midReady.keys where !shown.contains(k) { midReady[k] = nil } }
        if cells.count > 900 {
            let keepCells = Set(composition.near.map(\.cell)).union(composition.mid.map(\.cell))
            for k in cells.keys where !keepCells.contains(k) { cells[k] = nil }
        }
    }

    // MARK: - Geometry

    private static func closest(_ ex: Double, _ ez: Double, _ x0: Double, _ z0: Double, _ size: Double) -> Double {
        let dx = max(x0 - ex, 0, ex - (x0 + size)), dz = max(z0 - ez, 0, ez - (z0 + size))
        return (dx * dx + dz * dz).squareRoot()
    }

    /// Four inward planes through the camera bounding its view cone, widened so a jittered frame
    /// or a camera that moved since never exposes a missing cell. An orthographic camera passes all.
    private func viewPlanes(_ cam: IlluminatoramaCamera) -> [SIMD4<Float>] {
        guard !cam.isOrthographic else { return [] }
        let f = simd_normalize(cam.target - cam.position)
        let r = simd_normalize(simd_cross(f, cam.up))
        let u = simd_cross(r, f)
        let tanY = tan(cam.fovYRadians * 0.5) * 1.25
        let tanX = tanY * max(cam.aspect, 1) * 1.15
        func plane(_ n: SIMD3<Float>) -> SIMD4<Float> {
            let nn = simd_normalize(n)
            return SIMD4(nn, -simd_dot(nn, cam.position))
        }
        return [plane(r + f * tanX), plane(-r + f * tanX), plane(u + f * tanY), plane(-u + f * tanY)]
    }

    /// Sphere-vs-planes: a cell is out only if it lies wholly behind one plane. `y` is the eye
    /// height; a cell's own height is within a few metres of the ground the camera looks at, so
    /// the sphere is centred at the cell's ground height.
    private func inView(_ planes: [SIMD4<Float>], x: Double, z: Double, y: Double, radius: Double) -> Bool {
        guard !planes.isEmpty else { return true }
        let c = SIMD3<Float>(Float(x), Float(ground(x, z) + 0.4), Float(z))
        for p in planes where simd_dot(SIMD3(p.x, p.y, p.z), c) + p.w < -Float(radius) { return false }
        return true
    }
}

enum FloraJobResult: @unchecked Sendable {
    case cell(FloraCellKey, [FloraPlacement], Int)
    case mid(FloraCellKey, Int, FloraLibrary.MeshData?, Int)
}

/// Background job runner: bounded concurrency, results handed back to the main actor through a
/// locked inbox drained once per update.
final class FloraJobQueue: @unchecked Sendable {
    private let queue: OperationQueue
    private let lock = NSLock()
    private var inbox: [FloraJobResult] = []
    private var running = 0

    init(maxConcurrent: Int) {
        queue = OperationQueue()
        queue.name = "flora.stream"
        queue.maxConcurrentOperationCount = maxConcurrent
        queue.qualityOfService = .userInitiated
    }

    var outstanding: Int { lock.lock(); defer { lock.unlock() }; return running }

    func submit(_ work: @escaping @Sendable () -> FloraJobResult?) {
        lock.lock(); running += 1; lock.unlock()
        queue.addOperation { [weak self] in
            let r = work()
            guard let self else { return }
            self.lock.lock()
            if let r { self.inbox.append(r) }
            self.running -= 1
            self.lock.unlock()
        }
    }

    func drain() -> [FloraJobResult] {
        lock.lock(); defer { lock.unlock() }
        let out = inbox
        inbox.removeAll(keepingCapacity: true)
        return out
    }

    func cancelAll() { queue.cancelAllOperations() }
}
