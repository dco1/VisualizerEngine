//  SculptStroke.swift — brushes and the stroke API of the sculpt kit (ticket item I).
//
//      let stroke = volume.beginStroke(SculptBrush(.draw, radius: 0.02, strength: 0.3))
//      stroke.stamp(position: p, normal: n, pressure: 0.8)      // as many as you like, along the pen
//      stroke.end()                                              // one undo step: only the blocks touched
//
//  A stamp is applied at once, on the GPU when `volume.useGPU` and a device exist (one command buffer a stamp, in order, not
//  waited on until something reads the field), else on the CPU twin. Symmetry expands one stamp into up to 8 mirrored
//  ones (about the planes through `symmetryOrigin`); a stamp on a plane is not applied twice. `grab` is incremental: each
//  stamp drags the material under the PREVIOUS stamp's position to this one's (the first stamp only anchors).
//  Guarantees: a stamp reads and writes only its brush's bounding box (+ the drag for grab). Not guaranteed: draw/inflate
//  etc. act on the field everywhere in the brush sphere, so a thin sheet inside the brush is moved from both sides.

import Foundation
import simd

public enum SculptBrushKind: Sendable, CaseIterable {
    case draw, clay, grab, smooth, inflate, crease, flatten, pinch
}

public struct SculptSymmetry: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
    public static let x = SculptSymmetry(rawValue: 1), y = SculptSymmetry(rawValue: 2), z = SculptSymmetry(rawValue: 4)
}

public struct SculptBrush: Sendable {
    public var kind: SculptBrushKind
    /// Metres.
    public var radius: Double
    /// Peak move at full pressure, as a fraction of the radius (draw / inflate / crease); the blend factor 0…1 for smooth,
    /// flatten, pinch, grab.
    public var strength: Double
    public var falloff: SculptFalloff = .smooth
    /// Draw / clay / inflate / crease: remove material instead of adding it.
    public var subtract = false
    public var symmetry: SculptSymmetry = []
    public var symmetryOrigin = Vec3.zero
    public init(_ kind: SculptBrushKind, radius: Double, strength: Double = 0.5, falloff: SculptFalloff = .smooth,
                subtract: Bool = false, symmetry: SculptSymmetry = [], symmetryOrigin: Vec3 = .zero) {
        self.kind = kind; self.radius = radius; self.strength = strength; self.falloff = falloff
        self.subtract = subtract; self.symmetry = symmetry; self.symmetryOrigin = symmetryOrigin
    }
}

public final class SculptStroke {
    let volume: SculptVolume
    public let brush: SculptBrush
    private var step: SculptUndoStep
    private var last: Vec3?
    private var open = true
    /// Samples visited by the most recent stamp (all mirrors).
    public private(set) var lastVoxels = 0

    init(volume: SculptVolume, brush: SculptBrush) {
        self.volume = volume; self.brush = brush
        step = SculptUndoStep(blocks: [:], samples: volume.samples)
    }

    /// Stamp at `position` (world metres) with surface `normal` (need not be unit) and pen `pressure` 0…1.
    public func stamp(position: Vec3, normal: Vec3 = Vec3(0, 1, 0), pressure: Double = 1) {
        precondition(open, "SculptStroke: stamp after end")
        let n = len3(normal) > 1e-12 ? normalize3(normal) : Vec3(0, 1, 0)
        var drag = Vec3.zero
        var center = position
        if brush.kind == .grab {
            guard let prev = last else { last = position; return }
            drag = position - prev; center = prev
        }
        last = position
        var total = 0
        var done: [Vec3] = []
        for m in 0..<8 where SculptSymmetry(rawValue: m).isSubset(of: brush.symmetry) {
            func mirror(_ v: Vec3, about o: Vec3) -> Vec3 {
                var r = v - o
                if m & 1 != 0 { r.x = -r.x }
                if m & 2 != 0 { r.y = -r.y }
                if m & 4 != 0 { r.z = -r.z }
                return r + o
            }
            let c = mirror(center, about: brush.symmetryOrigin)
            if done.contains(where: { len3($0 - c) < 1e-9 }) { continue }
            done.append(c)
            let nn = mirror(n, about: .zero), dd = mirror(drag, about: .zero)
            total += apply(center: c, normal: nn, drag: dd, pressure: pressure)
        }
        lastVoxels = total
        volume.lastStampVoxels = total
    }

    private func apply(center c: Vec3, normal n: Vec3, drag: Vec3, pressure: Double) -> Int {
        let v = volume, r = brush.radius, cell = v.cell
        let reach = r + len3(drag) + cell
        let lo = SIMD3(Int(((c.x - reach - v.origin.x) / cell).rounded(.down)), Int(((c.y - reach - v.origin.y) / cell).rounded(.down)),
                       Int(((c.z - reach - v.origin.z) / cell).rounded(.down)))
        let hi = SIMD3(Int(((c.x + reach - v.origin.x) / cell).rounded(.up)), Int(((c.y + reach - v.origin.y) / cell).rounded(.up)),
                       Int(((c.z + reach - v.origin.z) / cell).rounded(.up)))
        let l = simd_max(lo, SIMD3<Int>(repeating: 0)), h = simd_min(hi, v.samples &- 1)
        guard l.x <= h.x, l.y <= h.y, l.z <= h.z else { return 0 }
        v.save(lo: l, hi: h, into: &step)

        let kind: UInt32
        switch brush.kind {
        case .draw, .clay: kind = SculptKernel.draw
        case .grab: kind = SculptKernel.grab
        case .smooth: kind = SculptKernel.smooth
        case .inflate: kind = SculptKernel.inflate
        case .crease: kind = SculptKernel.crease
        case .flatten: kind = SculptKernel.flatten
        case .pinch: kind = SculptKernel.pinch
        }
        let p = Float(min(max(pressure, 0), 1))
        var amount = Float(brush.strength * r) * p
        // draw / inflate add by default, a crease cuts by default; `subtract` flips each.
        if brush.subtract { amount = -amount }
        let size = h &- l &+ 1
        var P = SculptParams()
        P.centerRadius = SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(r))
        P.normalAmount = SIMD4(Float(n.x), Float(n.y), Float(n.z), amount)
        P.drag = SIMD4(Float(drag.x), Float(drag.y), Float(drag.z), Float(brush.strength) * p)
        if brush.kind == .grab { P.drag.w = Float(brush.strength) }
        P.gridOrigin = SIMD4(Float(v.origin.x), Float(v.origin.y), Float(v.origin.z), Float(cell))
        P.lo = SIMD4(UInt32(l.x), UInt32(l.y), UInt32(l.z), kind)
        P.size = SIMD4(UInt32(size.x), UInt32(size.y), UInt32(size.z), brush.falloff.rawValue)
        P.dims = SIMD4(UInt32(v.samples.x), UInt32(v.samples.y), UInt32(v.samples.z), v.maskStore == nil ? 0 : 1)
        if !(v.useGPU && SculptGPU.apply(P, to: v)) {
            v.sync()
            SculptKernel.apply(P, field: v.store.ptr, mask: v.maskStore.map { UnsafePointer($0.ptr) })
        }
        return size.x * size.y * size.z
    }

    /// Finish: the stroke becomes one undo step (empty strokes leave no step).
    public func end() {
        guard open else { return }
        open = false
        volume.sync()
        guard !step.blocks.isEmpty else { return }
        volume.undoStack.append(step)
        volume.redoStack.removeAll()
    }
}

public extension SculptVolume {
    /// Open a stroke. A brush much smaller than a cell first refines the grid (`refineBelowCells`, `refine()`).
    func beginStroke(_ brush: SculptBrush) -> SculptStroke {
        while refineBelowCells > 0 && brush.radius < refineBelowCells * cell, refine() {}
        return SculptStroke(volume: self, brush: brush)
    }

    /// Union (or, with `subtract`, difference) of `sdf` into the field inside `[lo, hi]` — how an empty volume gets its
    /// first material. CPU, hard boolean (min / max of the fields), one undo step.
    func combine(_ sdf: SDF, subtract: Bool = false, lo: Vec3, hi: Vec3) {
        sync()
        let a = SIMD3<Double>((lo - origin) / cell).rounded(.down), b = SIMD3<Double>((hi - origin) / cell).rounded(.up)
        let l = simd_max(SIMD3<Int>(Int(a.x), Int(a.y), Int(a.z)), SIMD3<Int>(repeating: 0))
        let h = simd_min(SIMD3(Int(b.x), Int(b.y), Int(b.z)), samples &- 1)
        guard l.x <= h.x, l.y <= h.y, l.z <= h.z else { return }
        var step = SculptUndoStep(blocks: [:], samples: samples)
        save(lo: l, hi: h, into: &step)
        for k in l.z...h.z { for j in l.y...h.y { for i in l.x...h.x {
            let d = Float(sdf(position(i, j, k))), idx = index(i, j, k), v = store.ptr[idx]
            store.ptr[idx] = subtract ? max(v, -d) : min(v, d)
        } } }
        undoStack.append(step); redoStack.removeAll()
    }
}
