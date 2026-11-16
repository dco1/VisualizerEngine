//  SculptVolume.swift — a dense voxel grid of signed distances you sculpt with brush stamps (ticket item I).
//
//  The field is `Float`, negative inside, in metres, over the box [origin, origin + (samples − 1)·cell], held in ONE
//  page-aligned allocation that is also wrapped (no copy) as an `MTLBuffer` where there is a Metal device: the field stays
//  resident between stamps, the GPU brush kernels (`Sculpt+GPU.swift`) write it in place and the CPU reads the same bytes.
//  Brush maths lives twice: `SculptKernel.swift` (CPU twin and oracle) and the MSL in `Sculpt+GPU.swift`; tests hold them equal.
//
//  Guarantees: a stamp reads and writes only the voxels in its brush's bounding box, so it costs O(radius³) whatever the
//  grid size (`lastStampVoxels`); undo restores the field bit-for-bit, from snapshots of only the 8³ blocks a stroke touched;
//  `mesh()` is closed, edge-manifold and outward wound (the item-C surface-nets topology), with vertices projected onto the
//  trilinear field and gradient normals.
//  Not guaranteed / gaps: the grid is dense (blocks are the unit of undo, not of storage: no sparse allocation); the field
//  is not re-distanced, so heavy sculpting drifts from a true distance (the zero set stays right; smooth also eases it);
//  refinement is GLOBAL 2× (see `refine()`), not local; topology stage of meshing stays CPU (as in item C); an edit near the
//  box's edge is clipped by it, and the mesher forces the box boundary outside, so a solid touching the box is capped flat.

import Foundation
import simd

public final class SculptVolume {
    public static let blockSize = 8

    public private(set) var origin: Vec3
    public private(set) var cell: Double
    public private(set) var samples: SIMD3<Int>
    var store: SculptStore
    var maskStore: SculptStore?
    /// Run brushes on the GPU when there is a Metal device (tests switch it off for the CPU oracle).
    public var useGPU: Bool = true
    /// Voxels the last stamp visited (its brush bounding box, mirrors included).
    public internal(set) var lastStampVoxels = 0
    /// A stroke whose radius is under this many cells refines the grid first (0 = never).
    public var refineBelowCells: Double = 2.0
    /// Refinement stops when the grid would pass this many samples.
    public var maxSamples = 80_000_000

    var undoStack: [SculptUndoStep] = []
    var redoStack: [SculptUndoStep] = []
    var pending: AnyObject?                         // last GPU command buffer, waited on by `sync()`

    public var count: Int { samples.x * samples.y * samples.z }
    public var lo: Vec3 { origin }
    public var hi: Vec3 { origin + Vec3(Double(samples.x - 1), Double(samples.y - 1), Double(samples.z - 1)) * cell }

    // MARK: creation

    init(origin: Vec3, cell: Double, samples: SIMD3<Int>) {
        precondition(samples.x >= 4 && samples.y >= 4 && samples.z >= 4 && cell > 0, "SculptVolume: grid too small")
        (self.origin, self.cell, self.samples) = (origin, cell, samples)
        store = SculptStore(count: samples.x * samples.y * samples.z)
    }

    /// Sample `sdf` over `[lo, hi]` at `cell`.
    public convenience init(sdf: SDF, lo: Vec3, hi: Vec3, cell: Double) {
        let cells = SIMD3(Int(((hi.x - lo.x) / cell).rounded(.up)), Int(((hi.y - lo.y) / cell).rounded(.up)),
                          Int(((hi.z - lo.z) / cell).rounded(.up)))
        self.init(origin: lo, cell: cell, samples: cells &+ 1)
        var grid: [Float]? = nil
        if let program = sdf.gpuProgram() { grid = SDFGPU.shared?.grid(program, lo: lo, cell: cell, cells: cells) }
        let values = grid ?? sdf.cpuGrid(lo: lo, cell: cell, cells: cells)
        values.withUnsafeBufferPointer { store.ptr.update(from: $0.baseAddress!, count: values.count) }
    }

    /// A closed mesh as a field (`SDFMeshField`), the box padded by `padding` cells.
    public convenience init(mesh: Mesh3, cell: Double, padding: Int = 4) {
        var lo = Vec3(repeating: .infinity), hi = Vec3(repeating: -.infinity)
        for p in mesh.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let pad = Vec3(repeating: Double(padding) * cell)
        self.init(sdf: .mesh(mesh), lo: lo - pad, hi: hi + pad, cell: cell)
    }

    /// Nothing in it: every sample is "far outside". Build material with `combine(_:)`.
    public convenience init(emptyLo lo: Vec3, hi: Vec3, cell: Double) {
        let cells = SIMD3(Int(((hi.x - lo.x) / cell).rounded(.up)), Int(((hi.y - lo.y) / cell).rounded(.up)),
                          Int(((hi.z - lo.z) / cell).rounded(.up)))
        self.init(origin: lo, cell: cell, samples: cells &+ 1)
        let far = Float(len3(hi - lo))
        store.ptr.update(repeating: far, count: count)
    }

    deinit { sync() }

    // MARK: reading

    @inline(__always) func index(_ i: Int, _ j: Int, _ k: Int) -> Int { (k * samples.y + j) * samples.x + i }

    public func sample(_ i: Int, _ j: Int, _ k: Int) -> Float { sync(); return store.ptr[index(i, j, k)] }

    public func position(_ i: Int, _ j: Int, _ k: Int) -> Vec3 { origin + Vec3(Double(i), Double(j), Double(k)) * cell }

    /// Trilinear field value at `p` (clamped to the box).
    public func value(at p: Vec3) -> Double {
        sync()
        let g = SIMD3<Float>((p - origin) / cell)
        let ptr = store.ptr, n = samples
        return Double(SculptKernel.trilinear(g, n) { ptr[(($2 * n.y + $1) * n.x) + $0] })
    }

    /// Unit gradient (outward normal) at `p` by central differences.
    public func normal(at p: Vec3) -> Vec3 {
        let h = cell * 0.5
        let g = Vec3(value(at: p + Vec3(h, 0, 0)) - value(at: p - Vec3(h, 0, 0)),
                     value(at: p + Vec3(0, h, 0)) - value(at: p - Vec3(0, h, 0)),
                     value(at: p + Vec3(0, 0, h)) - value(at: p - Vec3(0, 0, h)))
        return len3(g) > 1e-12 ? normalize3(g) : Vec3(0, 1, 0)
    }

    /// The solid's volume in m³, sub-voxel (each sample's occupancy is 0.5 − v/cell clamped to 0…1).
    public func volume() -> Double {
        sync()
        var sum = 0.0
        let n = samples
        for k in 1..<(n.z - 1) { for j in 1..<(n.y - 1) { for i in 1..<(n.x - 1) {
            sum += Double(min(max(0.5 - store.ptr[index(i, j, k)] / Float(cell), 0), 1))
        } } }
        return sum * cell * cell * cell
    }

    /// A copy of the field (tests, snapshots).
    public func fieldCopy() -> [Float] { sync(); return Array(UnsafeBufferPointer(start: store.ptr, count: count)) }

    // MARK: mask

    /// Per-voxel protection 0…1 (1 = frozen, 0 = free to edit; a stamp's weight is multiplied by 1 − mask).
    public func setMask(_ f: (Vec3) -> Float) {
        sync()
        let m = maskStore ?? SculptStore(count: count)
        maskStore = m
        for k in 0..<samples.z { for j in 0..<samples.y { for i in 0..<samples.x {
            m.ptr[index(i, j, k)] = min(max(f(position(i, j, k)), 0), 1)
        } } }
    }
    public func clearMask() { sync(); maskStore = nil }
    public var hasMask: Bool { maskStore != nil }

    // MARK: sync

    /// Wait for the GPU stamps in flight.
    public func sync() { SculptGPU.wait(self) }
}

/// The field's memory: page-aligned so Metal can wrap it with no copy.
final class SculptStore: @unchecked Sendable {
    let count: Int, bytes: Int
    let ptr: UnsafeMutablePointer<Float>
    var gpu: AnyObject?                             // an MTLBuffer when there is a device
    init(count: Int) {
        self.count = count
        let page = Int(getpagesize())
        bytes = (count * 4 + page - 1) / page * page
        var raw: UnsafeMutableRawPointer?
        posix_memalign(&raw, page, bytes)
        memset(raw!, 0, bytes)
        ptr = raw!.assumingMemoryBound(to: Float.self)
        gpu = SculptGPU.wrap(self)
    }
    deinit { gpu = nil; free(ptr) }
}

// MARK: undo

/// The blocks one stroke touched: their values BEFORE the stroke (after an undo, the values to redo).
struct SculptUndoStep {
    var blocks: [Int: [Float]]
    var samples: SIMD3<Int>
}

extension SculptVolume {
    var blocksPerAxis: SIMD3<Int> {
        let b = SculptVolume.blockSize
        return SIMD3((samples.x + b - 1) / b, (samples.y + b - 1) / b, (samples.z + b - 1) / b)
    }

    /// Visit the samples of block `key` in a fixed order (x fastest).
    func forBlock(_ key: Int, _ body: (Int) -> Void) {
        let b = SculptVolume.blockSize, nb = blocksPerAxis
        let bx = key % nb.x, by = (key / nb.x) % nb.y, bz = key / (nb.x * nb.y)
        for k in (bz * b)..<min((bz + 1) * b, samples.z) { for j in (by * b)..<min((by + 1) * b, samples.y) {
            for i in (bx * b)..<min((bx + 1) * b, samples.x) { body(index(i, j, k)) }
        } }
    }

    func snapshot(_ key: Int) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(SculptVolume.blockSize * SculptVolume.blockSize * SculptVolume.blockSize)
        forBlock(key) { out.append(store.ptr[$0]) }
        return out
    }

    func swapBlocks(_ step: SculptUndoStep) -> SculptUndoStep {
        var other = SculptUndoStep(blocks: [:], samples: samples)
        for (key, old) in step.blocks {
            other.blocks[key] = snapshot(key)
            var n = 0
            forBlock(key) { store.ptr[$0] = old[n]; n += 1 }
        }
        return other
    }

    public var canUndo: Bool { !undoStack.isEmpty }
    public var canRedo: Bool { !redoStack.isEmpty }
    /// Number of 8³ blocks the next undo would restore.
    public var undoBlockCount: Int { undoStack.last?.blocks.count ?? 0 }

    public func undo() {
        sync()
        guard let s = undoStack.popLast() else { return }
        redoStack.append(swapBlocks(s))
    }
    public func redo() {
        sync()
        guard let s = redoStack.popLast() else { return }
        undoStack.append(swapBlocks(s))
    }

    /// Record the blocks of `lo…hi` (sample indices, inclusive) the stroke has not saved yet. Call BEFORE writing them.
    func save(lo: SIMD3<Int>, hi: SIMD3<Int>, into step: inout SculptUndoStep) {
        let b = SculptVolume.blockSize, nb = blocksPerAxis
        for bz in (lo.z / b)...(hi.z / b) { for by in (lo.y / b)...(hi.y / b) { for bx in (lo.x / b)...(hi.x / b) {
            let key = (bz * nb.y + by) * nb.x + bx
            if step.blocks[key] == nil { step.blocks[key] = snapshot(key) }
        } } }
    }

    // MARK: refinement (two-level, global)

    /// Double the resolution of the whole grid (trilinear upsampling; `cell` halves, the box stays). This is the
    /// "dyntopo-like" step: a stroke whose brush is much smaller than a cell calls it at `begin` (up to `maxSamples`), so
    /// detail finer than the old cell can exist. It is GLOBAL (8× the memory), not local, and it clears the undo history
    /// (block keys change). Returns false when it would pass `maxSamples`.
    @discardableResult
    public func refine() -> Bool {
        let n = samples, m = SIMD3(2 * (n.x - 1) + 1, 2 * (n.y - 1) + 1, 2 * (n.z - 1) + 1)
        guard m.x * m.y * m.z <= maxSamples else { return false }
        sync()
        let fresh = SculptStore(count: m.x * m.y * m.z)
        if !(useGPU && SculptGPU.upsample(from: store, n, to: fresh, m)) {
            SculptKernel.upsample(from: store.ptr, n, to: fresh.ptr, m)
        }
        if maskStore != nil {                         // masks carry over, nearest
            let mm = SculptStore(count: m.x * m.y * m.z)
            for k in 0..<m.z { for j in 0..<m.y { for i in 0..<m.x {
                mm.ptr[(k * m.y + j) * m.x + i] = maskStore!.ptr[index(min(i / 2, n.x - 1), min(j / 2, n.y - 1), min(k / 2, n.z - 1))]
            } } }
            maskStore = mm
        }
        store = fresh; samples = m; cell *= 0.5
        undoStack.removeAll(); redoStack.removeAll()
        return true
    }
}
