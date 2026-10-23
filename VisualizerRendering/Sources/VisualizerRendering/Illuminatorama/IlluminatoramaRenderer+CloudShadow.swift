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
        else { st.key = nil; return }
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
    }
}
