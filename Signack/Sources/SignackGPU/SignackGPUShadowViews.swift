import Metal
import simd
import Signack

/// DH-1465 — the shadow views a cut is culled against, as the GPU reads them: the byte-for-byte
/// mirror of `SignackGPUShadowViews` in `SignackCut.metal` (784 B: 8 × 6 planes, then count + pool).
///
/// Views past `SignackShadowViews.maxViews` are dropped here; a host maps only the views it passed
/// to records, so a dropped view draws the whole-cut shadow record (never wrong, only not culled).
public struct SignackGPUShadowViewUniforms: Sendable {
    public static let byteLength = 784
    var words: [SIMD4<Float>]

    public init(_ views: [SignackFrustum], poolCapacity: Int) {
        let pass = SIMD4<Float>(0, 0, 0, 1e30)   // a plane every sphere is inside
        words = Array(repeating: pass, count: 6 * SignackShadowViews.maxViews + 1)
        let n = min(views.count, SignackShadowViews.maxViews)
        for v in 0..<n {
            for (p, plane) in views[v].planes.prefix(6).enumerated() { words[6 * v + p] = plane }
        }
        words[6 * SignackShadowViews.maxViews] = SIMD4(Float(bitPattern: UInt32(n)),
                                                      Float(bitPattern: UInt32(poolCapacity)), 0, 0)
    }

    func bind(_ enc: MTLComputeCommandEncoder, index: Int) {
        words.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: Self.byteLength, index: index) }
    }
}

extension SignackGPUCut {
    /// Indices each shadow view's record drew in the last completed frame (async, may lag).
    public func lastShadowViewIndexCounts(_ count: Int) -> [Int] {
        (0..<min(count, SignackShadowViews.maxViews)).map {
            Int(drawArgs.contents().load(fromByteOffset: SignackGPUDrawArgs.shadowViewOffset($0), as: UInt32.self))
        }
    }

    /// View `v`'s record as (indexStart, indexCount) — tests and diagnostics, after completion.
    public func shadowViewRecord(_ v: Int) -> (start: Int, count: Int) {
        let o = SignackGPUDrawArgs.shadowViewOffset(v)
        let p = drawArgs.contents()
        return (Int(p.load(fromByteOffset: o + 8, as: UInt32.self)), Int(p.load(fromByteOffset: o, as: UInt32.self)))
    }

    /// The last completed selection's shadow-view membership: for each of the first `count` views,
    /// the cut clusters that can land in it (ascending). Read only after completion.
    public func readShadowViewSelection(_ count: Int) -> [[UInt32]] {
        let f = flags.contents().bindMemory(to: UInt8.self, capacity: hierarchy.clusterCount)
        let m = viewMask.contents().bindMemory(to: UInt8.self, capacity: hierarchy.clusterCount)
        return (0..<min(count, SignackShadowViews.maxViews)).map { v in
            (0..<hierarchy.clusterCount).compactMap { i in f[i] != 0 && m[i] & UInt8(1 << v) != 0 ? UInt32(i) : nil }
        }
    }
}
