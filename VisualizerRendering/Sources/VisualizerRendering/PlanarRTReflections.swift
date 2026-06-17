import Foundation
import Metal
import OSLog
import SceneKit
import VisualizerCore
import simd

/// ── PLANAR RT REFLECTIONS (revertable) ──────────────────────────
/// Shared hardware-ray-traced planar-reflection renderer used by the
/// `+`-tier scenes (Well Drop, Hotdog Drop+, Pizza Turntable+) and by
/// City Street's wall-mounted window reflections.
///
/// Revert: delete this file + `PlanarRTReflections.metal` (in the app
/// target's `Visualizer/Rendering/`) + every
/// `// ── RT REFLECTIONS (revertable) ──` block in each scene's
/// controller / settings / settings view.
@MainActor
public final class PlanarRTReflections {

    static let log = Logger(subsystem: AppLog.subsystem, category: "planarRT")

    // ── Reflective plane parameters ──────────────────────────────

    public struct Plane {
        public var origin: SIMD3<Float>
        public var uDir: SIMD3<Float>
        public var vDir: SIMD3<Float>
        public var normal: SIMD3<Float>
        public var uExtent: Float
        public var vExtent: Float

        public init(origin: SIMD3<Float>, uDir: SIMD3<Float>, vDir: SIMD3<Float>,
                    normal: SIMD3<Float>, uExtent: Float, vExtent: Float) {
            self.origin = origin
            self.uDir = uDir
            self.vDir = vDir
            self.normal = normal
            self.uExtent = uExtent
            self.vExtent = vExtent
        }

        public static func floor(y: Float, halfX: Float, halfZ: Float) -> Plane {
            Plane(
                origin: SIMD3(0, y, 0),
                uDir: SIMD3(1, 0, 0),
                vDir: SIMD3(0, 0, 1),
                normal: SIMD3(0, 1, 0),
                uExtent: halfX,
                vExtent: halfZ
            )
        }
    }

    // ── Public surface ───────────────────────────────────────────

    public let reflectionTexture: MTLTexture
    public let plane: Plane

    public var surfaceY: Float { plane.origin.y }
    public var xzExtentX: Float { plane.uExtent }
    public var xzExtentZ: Float { plane.vExtent }

    // ── Metal plumbing ───────────────────────────────────────────

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    private var primitiveAS: MTLAccelerationStructure?
    private let vertexBuffer: MTLBuffer
    private let indexBuffer: MTLBuffer
    private let triangleCount: Int

    private var instanceAS: MTLAccelerationStructure?
    private var instanceDescriptorBuffer: MTLBuffer?
    private var instanceCapacity: Int = 0

    private var tintBuffer: MTLBuffer?
    private var tintCapacity: Int = 0

    private var primitiveScratch: MTLBuffer?
    private var instanceScratch: MTLBuffer?

    private let uniformBuffer: MTLBuffer

    // ── Geometry constants ───────────────────────────────────────

    public let textureSize: Int
    public let maxRayDistance: Float

    // ── Init ─────────────────────────────────────────────────────

    public init?(
        meshVertices: [SIMD3<Float>],
        meshIndices: [UInt32],
        plane: Plane,
        textureSize: Int = 256,
        maxRayDistance: Float = 40.0
    ) {
        guard
            GPUCapabilities.supportsRaytracing,
            let device = GPUCapabilities.device,
            device.supportsRaytracing
        else {
            Self.log.notice("ray tracing not supported on this GPU; skipping init")
            return nil
        }
        guard let queue = device.makeCommandQueue() else {
            Self.log.warning("could not create command queue")
            return nil
        }

        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .main)
        } catch {
            Self.log.error("default Metal library missing: \(error.localizedDescription)")
            return nil
        }
        guard let function = library.makeFunction(name: "planarRTReflect") else {
            Self.log.error("kernel `planarRTReflect` not found in default library")
            return nil
        }
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            Self.log.error("failed to build compute pipeline: \(error.localizedDescription)")
            return nil
        }

        guard
            meshIndices.count % 3 == 0, !meshIndices.isEmpty,
            let vb = device.makeBuffer(
                bytes: meshVertices,
                length: MemoryLayout<SIMD3<Float>>.stride * meshVertices.count,
                options: [.storageModeShared]
            ),
            let ib = device.makeBuffer(
                bytes: meshIndices,
                length: MemoryLayout<UInt32>.stride * meshIndices.count,
                options: [.storageModeShared]
            )
        else {
            Self.log.error("could not allocate canonical mesh buffers")
            return nil
        }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: textureSize,
            height: textureSize,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead, .shaderWrite]
        texDesc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: texDesc) else {
            Self.log.error("could not allocate reflection texture")
            return nil
        }
        tex.label = "PlanarRT.reflection"

        guard let uniforms = device.makeBuffer(
            length: MemoryLayout<RTUniforms>.stride,
            options: [.storageModeShared]
        ) else {
            Self.log.error("could not allocate uniform buffer")
            return nil
        }
        uniforms.label = "PlanarRT.uniforms"

        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        self.vertexBuffer = vb
        self.indexBuffer = ib
        self.triangleCount = meshIndices.count / 3
        self.reflectionTexture = tex
        self.uniformBuffer = uniforms
        self.plane = plane
        self.textureSize = textureSize
        self.maxRayDistance = maxRayDistance

        do {
            try buildPrimitiveAS()
        } catch {
            Self.log.error("primitive AS build failed: \(error.localizedDescription)")
            return nil
        }

        Self.log.notice("""
            PlanarRT initialised \
            tex=\(textureSize)×\(textureSize) \
            tri=\(self.triangleCount) \
            extent=(\(plane.uExtent), \(plane.vExtent)) \
            origin=(\(plane.origin.x), \(plane.origin.y), \(plane.origin.z)) \
            normal=(\(plane.normal.x), \(plane.normal.y), \(plane.normal.z))
            """)
    }

    public convenience init?(
        meshVertices: [SIMD3<Float>],
        meshIndices: [UInt32],
        surfaceY: Float,
        xzExtentX: Float,
        xzExtentZ: Float,
        textureSize: Int = 256,
        maxRayDistance: Float = 40.0
    ) {
        self.init(
            meshVertices: meshVertices,
            meshIndices: meshIndices,
            plane: .floor(y: surfaceY, halfX: xzExtentX, halfZ: xzExtentZ),
            textureSize: textureSize,
            maxRayDistance: maxRayDistance
        )
    }

    // ── Per-tick update ──────────────────────────────────────────

    public func update(
        transforms: [simd_float4x4],
        tints: [SIMD4<Float>],
        cameraWorldPosition: SIMD3<Float>
    ) {
        precondition(transforms.count == tints.count, "transform/tint count mismatch")

        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        commandBuffer.label = "PlanarRT.frame"

        let actualTransforms = transforms.isEmpty ? [Self.farAwayTransform] : transforms
        let actualTints      = transforms.isEmpty ? [SIMD4<Float>(repeating: 0)] : tints

        buildInstanceAS(transforms: actualTransforms, in: commandBuffer)
        dispatchReflectionPass(
            tints: actualTints,
            cameraWorldPosition: cameraWorldPosition,
            in: commandBuffer
        )

        commandBuffer.commit()
    }

    private static let farAwayTransform: simd_float4x4 = {
        var m = matrix_identity_float4x4
        m.columns.3.y = -1_000_000
        return m
    }()

    // ── Primitive AS (canonical mesh) ────────────────────────────

    private func buildPrimitiveAS() throws {
        let geomDesc = MTLAccelerationStructureTriangleGeometryDescriptor()
        geomDesc.vertexBuffer = vertexBuffer
        geomDesc.vertexBufferOffset = 0
        geomDesc.vertexStride = MemoryLayout<SIMD3<Float>>.stride
        geomDesc.vertexFormat = .float3
        geomDesc.indexBuffer = indexBuffer
        geomDesc.indexBufferOffset = 0
        geomDesc.indexType = .uint32
        geomDesc.triangleCount = triangleCount

        let asDesc = MTLPrimitiveAccelerationStructureDescriptor()
        asDesc.geometryDescriptors = [geomDesc]

        let sizes = device.accelerationStructureSizes(descriptor: asDesc)
        guard
            let storage = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
            let scratch = device.makeBuffer(
                length: max(sizes.buildScratchBufferSize, 16),
                options: [.storageModePrivate]
            ),
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeAccelerationStructureCommandEncoder()
        else {
            throw NSError(
                domain: "PlanarRT", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "AS storage / scratch alloc failed"]
            )
        }
        storage.label = "PlanarRT.primitiveAS"
        enc.build(
            accelerationStructure: storage,
            descriptor: asDesc,
            scratchBuffer: scratch,
            scratchBufferOffset: 0
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        self.primitiveAS = storage
        self.primitiveScratch = scratch
    }

    // ── Instance AS (per-frame) ──────────────────────────────────

    private func buildInstanceAS(transforms: [simd_float4x4], in commandBuffer: MTLCommandBuffer) {
        guard let primitive = primitiveAS else { return }

        let stride = MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride
        if instanceCapacity < transforms.count || instanceDescriptorBuffer == nil {
            let cap = max(transforms.count, 32)
            instanceDescriptorBuffer = device.makeBuffer(
                length: cap * stride,
                options: [.storageModeShared]
            )
            instanceDescriptorBuffer?.label = "PlanarRT.instanceDesc"
            instanceCapacity = cap
        }
        guard let descBuffer = instanceDescriptorBuffer else { return }

        let descPtr = descBuffer.contents().bindMemory(
            to: MTLAccelerationStructureInstanceDescriptor.self,
            capacity: transforms.count
        )
        for (i, m) in transforms.enumerated() {
            var inst = MTLAccelerationStructureInstanceDescriptor()
            inst.accelerationStructureIndex = 0
            inst.mask = 0xFF
            inst.options = MTLAccelerationStructureInstanceOptions(rawValue: 0)
            inst.intersectionFunctionTableOffset = 0
            inst.transformationMatrix = Self.pack(m)
            descPtr[i] = inst
        }

        let asDesc = MTLInstanceAccelerationStructureDescriptor()
        asDesc.instancedAccelerationStructures = [primitive]
        asDesc.instanceCount = transforms.count
        asDesc.instanceDescriptorBuffer = descBuffer
        asDesc.instanceDescriptorBufferOffset = 0
        asDesc.instanceDescriptorStride = stride

        let sizes = device.accelerationStructureSizes(descriptor: asDesc)

        if instanceAS == nil || instanceAS!.size < sizes.accelerationStructureSize {
            instanceAS = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
            instanceAS?.label = "PlanarRT.instanceAS"
        }
        if instanceScratch == nil || instanceScratch!.length < sizes.buildScratchBufferSize {
            instanceScratch = device.makeBuffer(
                length: max(sizes.buildScratchBufferSize, 16),
                options: [.storageModePrivate]
            )
        }
        guard
            let target = instanceAS,
            let scratch = instanceScratch,
            let enc = commandBuffer.makeAccelerationStructureCommandEncoder()
        else { return }
        enc.build(
            accelerationStructure: target,
            descriptor: asDesc,
            scratchBuffer: scratch,
            scratchBufferOffset: 0
        )
        enc.endEncoding()

        if tintCapacity < transforms.count || tintBuffer == nil {
            let cap = max(transforms.count, 32)
            tintBuffer = device.makeBuffer(
                length: cap * MemoryLayout<SIMD4<Float>>.stride,
                options: [.storageModeShared]
            )
            tintBuffer?.label = "PlanarRT.tints"
            tintCapacity = cap
        }
    }

    // ── Reflection dispatch ──────────────────────────────────────

    private func dispatchReflectionPass(
        tints: [SIMD4<Float>],
        cameraWorldPosition: SIMD3<Float>,
        in commandBuffer: MTLCommandBuffer
    ) {
        guard
            let instance = instanceAS,
            let tintsBuf = tintBuffer
        else { return }

        let tintPtr = tintsBuf.contents().bindMemory(
            to: SIMD4<Float>.self,
            capacity: tints.count
        )
        for (i, t) in tints.enumerated() {
            tintPtr[i] = t
        }

        let uPtr = uniformBuffer.contents().bindMemory(to: RTUniforms.self, capacity: 1)
        uPtr.pointee = RTUniforms(
            origin: (plane.origin.x, plane.origin.y, plane.origin.z),
            pad0: 0,
            uDir: (plane.uDir.x, plane.uDir.y, plane.uDir.z),
            uExtent: plane.uExtent,
            vDir: (plane.vDir.x, plane.vDir.y, plane.vDir.z),
            vExtent: plane.vExtent,
            normal: (plane.normal.x, plane.normal.y, plane.normal.z),
            maxDistance: maxRayDistance,
            cameraPos: (
                cameraWorldPosition.x,
                cameraWorldPosition.y,
                cameraWorldPosition.z
            ),
            pad1: 0
        )

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "PlanarRT.dispatch"
        enc.setComputePipelineState(pipeline)
        enc.setTexture(reflectionTexture, index: 0)
        enc.setAccelerationStructure(instance, bufferIndex: 0)
        enc.setBuffer(tintsBuf, offset: 0, index: 1)
        enc.setBuffer(uniformBuffer, offset: 0, index: 2)
        if let primitive = primitiveAS {
            enc.useResource(primitive, usage: .read)
        }

        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(
            width: reflectionTexture.width,
            height: reflectionTexture.height,
            depth: 1
        )
        enc.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
    }

    // ── Math helpers ─────────────────────────────────────────────

    private static func pack(_ m: simd_float4x4) -> MTLPackedFloat4x3 {
        var p = MTLPackedFloat4x3()
        p.columns.0 = MTLPackedFloat3Make(m.columns.0.x, m.columns.0.y, m.columns.0.z)
        p.columns.1 = MTLPackedFloat3Make(m.columns.1.x, m.columns.1.y, m.columns.1.z)
        p.columns.2 = MTLPackedFloat3Make(m.columns.2.x, m.columns.2.y, m.columns.2.z)
        p.columns.3 = MTLPackedFloat3Make(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        return p
    }
}

// ── Shader modifier helpers ──────────────────────────────────────

extension PlanarRTReflections {
    public static func surfaceModifierSource(normalYThreshold: Float) -> String {
        let threshold = String(format: "%.4f", normalYThreshold)
        return """
        #pragma arguments
        texture2d<float> rtReflection;
        float rtBlend;
        float rtExtentX;
        float rtExtentZ;

        #pragma body
        if (rtBlend > 0.0001) {
            float3 worldNormal = (u_inverseViewTransform * float4(_surface.normal, 0.0)).xyz;
            if (worldNormal.y > \(threshold)) {
                constexpr sampler rtSampler(filter::linear, address::clamp_to_edge);
                float4 worldPos = u_inverseViewTransform * float4(_surface.position, 1.0);
                float2 uv = float2(
                    (worldPos.x / (2.0 * rtExtentX)) + 0.5,
                    (worldPos.z / (2.0 * rtExtentZ)) + 0.5
                );
                uv = clamp(uv, float2(0.001), float2(0.999));
                float4 rt = rtReflection.sample(rtSampler, uv);
                _surface.emission.rgb += rt.rgb * rtBlend;
            }
        }
        """
    }

    public static func surfaceModifierSource(forWallNormalAlongX rtNormalXSign: Float) -> String {
        let signLiteral = String(format: "%.1f", rtNormalXSign)
        return """
        #pragma arguments
        texture2d<float> rtReflection;
        float rtBlend;
        float rtExtentU;
        float rtExtentV;
        float rtOriginY;
        float rtOriginZ;

        #pragma body
        if (rtBlend > 0.0001) {
            float3 worldNormal = (u_inverseViewTransform * float4(_surface.normal, 0.0)).xyz;
            if (worldNormal.x * \(signLiteral) > 0.5) {
                constexpr sampler rtSampler(filter::linear, address::clamp_to_edge);
                float4 worldPos = u_inverseViewTransform * float4(_surface.position, 1.0);
                float u = ((worldPos.z - rtOriginZ) / (2.0 * rtExtentU)) + 0.5;
                float v = ((worldPos.y - rtOriginY) / (2.0 * rtExtentV)) + 0.5;
                float2 uv = clamp(float2(u, v), float2(0.001), float2(0.999));
                float4 rt = rtReflection.sample(rtSampler, uv);
                _surface.emission.rgb += rt.rgb * rtBlend;
            }
        }
        """
    }
}

// ── Shared mesh helpers ──────────────────────────────────────────

extension PlanarRTReflections {
    public static func makeCylinderMesh(
        length: Float,
        radius: Float,
        segments: Int = 24
    ) -> (vertices: [SIMD3<Float>], indices: [UInt32]) {
        let halfL = length * 0.5
        var verts: [SIMD3<Float>] = []
        var idx: [UInt32] = []

        for i in 0...segments {
            let theta = Float(i) / Float(segments) * 2 * .pi
            let c = cos(theta), s = sin(theta)
            verts.append(SIMD3(radius * c, -halfL, radius * s))
            verts.append(SIMD3(radius * c, +halfL, radius * s))
        }
        for i in 0..<segments {
            let i0 = UInt32(i * 2)
            let i1 = UInt32(i * 2 + 1)
            let i2 = UInt32(i * 2 + 2)
            let i3 = UInt32(i * 2 + 3)
            idx.append(contentsOf: [i0, i2, i3, i0, i3, i1])
        }

        let topCentre = UInt32(verts.count)
        verts.append(SIMD3(0, +halfL, 0))
        let botCentre = UInt32(verts.count)
        verts.append(SIMD3(0, -halfL, 0))

        for i in 0..<segments {
            let theta1 = Float(i) / Float(segments) * 2 * .pi
            let theta2 = Float(i + 1) / Float(segments) * 2 * .pi
            let topA = UInt32(verts.count)
            verts.append(SIMD3(radius * cos(theta1), +halfL, radius * sin(theta1)))
            let topB = UInt32(verts.count)
            verts.append(SIMD3(radius * cos(theta2), +halfL, radius * sin(theta2)))
            idx.append(contentsOf: [topCentre, topB, topA])

            let botA = UInt32(verts.count)
            verts.append(SIMD3(radius * cos(theta1), -halfL, radius * sin(theta1)))
            let botB = UInt32(verts.count)
            verts.append(SIMD3(radius * cos(theta2), -halfL, radius * sin(theta2)))
            idx.append(contentsOf: [botCentre, botA, botB])
        }

        return (verts, idx)
    }

    public static func makeCapsuleXMesh(
        length: Float,
        radius: Float,
        radialSegments: Int = 16,
        capSegments: Int = 6
    ) -> (vertices: [SIMD3<Float>], indices: [UInt32]) {
        let halfL = max(0.001, length * 0.5 - radius)
        var verts: [SIMD3<Float>] = []
        var idx: [UInt32] = []

        for i in 0...radialSegments {
            let theta = Float(i) / Float(radialSegments) * 2 * .pi
            let y = radius * cos(theta)
            let z = radius * sin(theta)
            verts.append(SIMD3(-halfL, y, z))
            verts.append(SIMD3(+halfL, y, z))
        }
        for i in 0..<radialSegments {
            let i0 = UInt32(i * 2)
            let i1 = UInt32(i * 2 + 1)
            let i2 = UInt32(i * 2 + 2)
            let i3 = UInt32(i * 2 + 3)
            idx.append(contentsOf: [i0, i2, i3, i0, i3, i1])
        }

        func addCap(centreX: Float, side: Float) {
            let base = UInt32(verts.count)
            for lat in 0...capSegments {
                let phi = Float(lat) / Float(capSegments) * (.pi * 0.5)
                let xOff = side * radius * sin(phi)
                let ringR = radius * cos(phi)
                for lon in 0...radialSegments {
                    let theta = Float(lon) / Float(radialSegments) * 2 * .pi
                    verts.append(SIMD3(centreX + xOff, ringR * cos(theta), ringR * sin(theta)))
                }
            }
            let ringStride = UInt32(radialSegments + 1)
            for lat in 0..<capSegments {
                for lon in 0..<radialSegments {
                    let a = base + UInt32(lat) * ringStride + UInt32(lon)
                    let b = a + 1
                    let c = a + ringStride
                    let d = c + 1
                    idx.append(contentsOf: [a, c, d, a, d, b])
                }
            }
        }
        addCap(centreX: -halfL, side: -1)
        addCap(centreX: +halfL, side: +1)

        return (verts, idx)
    }

    public static func makeUnitSphereMesh(
        latitudeSegments: Int = 10,
        longitudeSegments: Int = 14
    ) -> (vertices: [SIMD3<Float>], indices: [UInt32]) {
        var verts: [SIMD3<Float>] = []
        var idx: [UInt32] = []
        for lat in 0...latitudeSegments {
            let phi = Float.pi * Float(lat) / Float(latitudeSegments) - .pi / 2
            let y = sin(phi)
            let r = cos(phi)
            for lon in 0...longitudeSegments {
                let theta = 2 * Float.pi * Float(lon) / Float(longitudeSegments)
                verts.append(SIMD3(r * cos(theta), y, r * sin(theta)))
            }
        }
        let stride = UInt32(longitudeSegments + 1)
        for lat in 0..<latitudeSegments {
            for lon in 0..<longitudeSegments {
                let a = UInt32(lat) * stride + UInt32(lon)
                let b = a + 1
                let c = a + stride
                let d = c + 1
                idx.append(contentsOf: [a, c, d, a, d, b])
            }
        }
        return (verts, idx)
    }

    public static func makeLampPostMesh(
        postHeight: Float,
        postRadius: Float,
        globeY: Float,
        globeRadius: Float,
        postSegments: Int = 10,
        latitudeSegments: Int = 8,
        longitudeSegments: Int = 12
    ) -> (vertices: [SIMD3<Float>], indices: [UInt32]) {
        var verts: [SIMD3<Float>] = []
        var idx: [UInt32] = []

        for i in 0...postSegments {
            let theta = Float(i) / Float(postSegments) * 2 * .pi
            let c = cos(theta), s = sin(theta)
            verts.append(SIMD3(postRadius * c, 0,          postRadius * s))
            verts.append(SIMD3(postRadius * c, postHeight, postRadius * s))
        }
        for i in 0..<postSegments {
            let i0 = UInt32(i * 2)
            let i1 = UInt32(i * 2 + 1)
            let i2 = UInt32(i * 2 + 2)
            let i3 = UInt32(i * 2 + 3)
            idx.append(contentsOf: [i0, i2, i3, i0, i3, i1])
        }

        let sphereBase = UInt32(verts.count)
        for lat in 0...latitudeSegments {
            let phi = Float.pi * Float(lat) / Float(latitudeSegments) - .pi / 2
            let y = sin(phi)
            let r = cos(phi)
            for lon in 0...longitudeSegments {
                let theta = 2 * Float.pi * Float(lon) / Float(longitudeSegments)
                verts.append(SIMD3(
                    globeRadius * r * cos(theta),
                    globeY + globeRadius * y,
                    globeRadius * r * sin(theta)
                ))
            }
        }
        let stride = UInt32(longitudeSegments + 1)
        for lat in 0..<latitudeSegments {
            for lon in 0..<longitudeSegments {
                let a = sphereBase + UInt32(lat) * stride + UInt32(lon)
                let b = a + 1
                let c = a + stride
                let d = c + 1
                idx.append(contentsOf: [a, c, d, a, d, b])
            }
        }
        return (verts, idx)
    }
}

private struct RTUniforms {
    var origin: (Float, Float, Float)
    var pad0: Float
    var uDir: (Float, Float, Float)
    var uExtent: Float
    var vDir: (Float, Float, Float)
    var vExtent: Float
    var normal: (Float, Float, Float)
    var maxDistance: Float
    var cameraPos: (Float, Float, Float)
    var pad1: Float
}
