import Foundation
import Metal
import simd

/// The cloud sun-transmittance map (`cloudShadowEnabled`, Daydream DH-1462): see
/// IlluminatoramaCloudShadow.h for the buffer layout and the lanes that read it.
final class CloudShadowState {
    /// Two buffers, alternated on every rebuild, so a frame still in flight never reads a half-written map.
    var buffers: [MTLBuffer] = []
    var current = 0
    var key: UInt64?
    /// Eight floats, enabled = 0: what every lane binds while the feature is off.
    var dummy: MTLBuffer?
    /// Rebuilds so far (a host or gate reads it to prove a still sky bakes once).
    var builds = 0
    /// A CPU copy of the last bake the GPU FINISHED (DH-1492), taken in the command buffer's completion
    /// handler: a CPU reading never looks at a GPU buffer, because with wind the double buffer can be
    /// re-baked while an older frame is still in flight. nil until the first bake lands.
    let completed = CloudShadowSnapshot()
}

/// The lock-guarded CPU copy `CloudShadowState.completed` keeps (written on Metal's completion thread).
final class CloudShadowSnapshot: @unchecked Sendable {
    private var snapshot: [Float]?
    private var meanT: Float = 1
    private let lock = NSLock()
    func markCompleted(_ buffer: MTLBuffer) {
        let n = buffer.length / MemoryLayout<Float>.stride
        let copy = Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float.self, capacity: n), count: n))
        // The map's mean, strength applied — the deck's average shade over the whole square.
        var mean: Float = 1
        if n > 8 {
            let texels = copy[8...]
            let raw = texels.reduce(0, +) / Float(texels.count)
            mean = 1 + (raw - 1) * max(0, min(1, copy[6]))
        }
        lock.lock(); snapshot = copy; meanT = mean; lock.unlock()
    }
    var mean: Float { lock.lock(); defer { lock.unlock() }; return snapshot == nil ? 1 : meanT }
    func withSnapshot<R>(_ body: ([Float]) -> R) -> R? {
        lock.lock(); defer { lock.unlock() }
        return snapshot.map(body)
    }
    func invalidateCompleted() { lock.lock(); snapshot = nil; lock.unlock() }
}

extension IlluminatoramaRenderer {

    /// Rebuilds of the cloud shadow map since launch.
    public var cloudShadowBuilds: Int { cloudShadow.builds }

    /// The current map's mean transmittance and its lit fraction (T > 0.5) — nil while off. For gates
    /// and captures (reads the shared buffer; call after the frame completed).
    public var cloudShadowMapStats: (mean: Float, lit: Float, centre: SIMD2<Float>)? {
        let st = cloudShadow
        guard cloudShadowEnabled, st.key != nil, st.buffers.count == 2 else { return nil }
        let b = st.buffers[st.current]
        let p = b.contents().bindMemory(to: Float.self, capacity: b.length / 4)
        let res = Int(p[4]), n = res * res
        guard n > 0 else { return nil }
        var sum: Float = 0, lit = 0
        for i in 0..<n { let t = p[8 + i]; sum += t; if t > 0.5 { lit += 1 } }
        return (sum / Float(n), Float(lit) / Float(n), SIMD2(p[0], p[1]))
    }

    /// The cloud deck's sun transmittance at world point `p` (`towardSun` = unit direction to the sun),
    /// read on the CPU from the last COMPLETED bake — the exact mirror of `illumiCloudSunTransmittance`
    /// (IlluminatoramaCloudShadow.h), so a host's CPU reading of the sun (an exposure meter, a room's
    /// daylight intake, a ground-fill estimate) sees the same cloud the shaders light with (DH-1492).
    /// Exactly 1 while the feature is off or before the first bake has landed.
    public func cloudSunTransmittance(at p: SIMD3<Float>, towardSun ld: SIMD3<Float>) -> Float {
        let st = cloudShadow
        guard cloudShadowEnabled, st.key != nil else { return 1 }
        return st.completed.withSnapshot { map in
            map.withUnsafeBufferPointer { Self.cloudSunTransmittance(header: $0.baseAddress!, count: $0.count, at: p, towardSun: ld) }
        } ?? 1
    }

    /// The deck's MEAN sun transmittance over the whole baked square (last completed bake; 1 while off) —
    /// for a far-field reading such as the ground the sky dome's lower half reflects (DH-1492).
    public var cloudSunMeanTransmittance: Float {
        guard cloudShadowEnabled, cloudShadow.key != nil else { return 1 }
        return cloudShadow.completed.mean
    }

    /// The lookup itself over a map's floats (header + texels) — public for a parity test against the shader.
    public static func cloudSunTransmittance(header cs: UnsafePointer<Float>, count: Int,
                                             at p: SIMD3<Float>, towardSun ld: SIMD3<Float>) -> Float {
        guard count >= 8, cs[3] > 0, ld.y > 0.02 else { return 1 }
        func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
            let t = max(0, min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t)
        }
        let h = p.y - cs[5]
        let limit = cs[7]
        let hw: Float = limit > 0 ? 1 - smoothstep(limit, limit + 0.5, h) : 1
        guard hw > 0 else { return 1 }
        let ext = max(cs[2], 1)
        let res = max(Int(cs[4]), 2)
        guard count >= 8 + res * res else { return 1 }
        let g = SIMD2(p.x, p.z) - SIMD2(ld.x, ld.z) * (h / ld.y)
        let uv = (g - SIMD2(cs[0], cs[1])) / ext + 0.5
        let edge = min(min(uv.x, 1 - uv.x), min(uv.y, 1 - uv.y))
        guard edge > 0 else { return 1 }
        func texel(_ x: Int, _ y: Int) -> Float {
            cs[8 + min(max(y, 0), res - 1) * res + min(max(x, 0), res - 1)]
        }
        let q = uv * Float(res) - 0.5
        let x0 = Int(q.x.rounded(.down)), y0 = Int(q.y.rounded(.down))
        let fx = q.x - Float(x0), fy = q.y - Float(y0)
        let top = texel(x0, y0) + (texel(x0 + 1, y0) - texel(x0, y0)) * fx
        let bot = texel(x0, y0 + 1) + (texel(x0 + 1, y0 + 1) - texel(x0, y0 + 1)) * fx
        var t = top + (bot - top) * fy
        t = 1 + (t - 1) * smoothstep(0, 0.08, edge)
        return 1 + (t - 1) * (max(0, min(1, cs[6])) * hw)
    }

    /// The buffer every sun-lit lane binds: the live map, or the disabled dummy.
    var cloudShadowBinding: MTLBuffer? {
        let st = cloudShadow
        if st.dummy == nil {
            var zeros = [Float](repeating: 0, count: 8)
            st.dummy = device.makeBuffer(bytes: &zeros, length: 32, options: .storageModeShared)
            st.dummy?.label = "Illuminatorama.cloudShadow.dummy"
        }
        guard cloudShadowEnabled, st.key != nil, st.buffers.count == 2 else { return st.dummy }
        return st.buffers[st.current]
    }

    /// Bake the map when the sky (deck, wind clock, sun) or its placement changed. The centre follows the
    /// camera in steps of an eighth of the extent, so an orbit re-bakes only when it crosses a step.
    func encodeCloudShadowPass(_ cb: MTLCommandBuffer) {
        let st = cloudShadow
        guard cloudShadowEnabled,
              let noise = cloudNoiseTexture,
              let skyU = cloudSkyUniforms,
              let pipe = engine.pipelineCache.pipelineState(name: "volSkyCloudShadowMap", device: device)
        else { st.key = nil; st.completed.invalidateCompleted(); return }
        let res = max(16, min(1024, cloudShadowResolution))
        let ext = max(100, cloudShadowExtent)
        let steps = max(8, cloudShadowSteps)
        let length = (8 + res * res) * MemoryLayout<Float>.stride
        if st.buffers.count != 2 || st.buffers[0].length != length {
            st.buffers = (0..<2).compactMap { i in
                let b = device.makeBuffer(length: length, options: .storageModeShared)
                b?.label = "Illuminatorama.cloudShadow.\(i)"
                return b
            }
            st.key = nil
            st.completed.invalidateCompleted()
            guard st.buffers.count == 2 else { return }
        }
        let cam = camera.position
        let stepM = ext / 8
        let centre = SIMD2<Float>((cam.x / stepM).rounded() * stepM, (cam.z / stepM).rounded() * stepM)
        var h: UInt64 = 0xcbf29ce484222325
        func mix(_ v: UInt32) { h = (h ^ UInt64(v)) &* 0x100000001b3 }
        let words = SkyUniforms.hostPrefixLength / 4
        let p = skyU.contents().bindMemory(to: UInt32.self, capacity: words)
        for i in 4..<words { mix(p[i]) }    // the sky minus its camera position
        for f in [centre.x, centre.y, ext, cloudShadowGroundY, cloudShadowStrength, cloudShadowHeightLimit] {
            mix(f.bitPattern)
        }
        mix(UInt32(res)); mix(UInt32(steps))
        guard st.key != h else { return }

        let next = 1 - st.current
        let buf = st.buffers[next]
        let hdr = buf.contents().bindMemory(to: Float.self, capacity: 8)
        let header: [Float] = [centre.x, centre.y, ext, 1, Float(res), cloudShadowGroundY,
                               max(0, min(1, cloudShadowStrength)), max(0, cloudShadowHeightLimit)]
        for i in 0..<8 { hdr[i] = header[i] }
        guard let enc = timedComputeEncoder(cb, "cloudShadow.bake") else { return }
        var cfg = SIMD4<UInt32>(UInt32(steps), 0, 0, 0)
        enc.label = "Illuminatorama.cloudShadow.bake"
        enc.setComputePipelineState(pipe)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(skyU, offset: 0, index: 1)
        enc.setBytes(&cfg, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
        enc.setTexture(noise, index: 0)
        dispatch(enc, pipeline: pipe, width: res, height: res)
        enc.endEncoding()
        st.current = next
        st.key = h
        st.builds += 1
        let snap = st.completed
        nonisolated(unsafe) let baked = buf   // read only after the GPU finished writing it
        cb.addCompletedHandler { _ in snap.markCompleted(baked) }
    }
}
