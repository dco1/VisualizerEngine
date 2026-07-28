import AppKit
import CoreGraphics
import ImageIO
import Metal
import simd
import UniformTypeIdentifiers
import VisualizerHumans
import VisualizerRendering

// Headless contact-sheet renderer for the human generator:
//   swift run humans-preview [outputDir]
// Renders a sweep of HumanSpecs through the Illuminatorama pipeline offscreen
// and writes one PNG per spec — quality gets judged from pixels, never imagination.

@MainActor
func render(specs: [(String, HumanSpec)], to dir: URL) throws {
    let width = 1280, height = 1600
    let camera = IlluminatoramaCamera(
        position: SIMD3(0.55, 1.35, 2.6), target: SIMD3(0, 0.9, 0), up: SIMD3(0, 1, 0),
        fovYRadians: .pi / 4.2, aspect: Float(width) / Float(height), zNear: 0.05, zFar: 200)
    let renderer = try IlluminatoramaRenderer(engine: .shared, width: width, height: height, camera: camera)
    renderer.internalRenderScale = 1.5
    renderer.resize(width: width, height: height)
    renderer.sharedTAAOverride = false
    renderer.appliesSharedLensFX = false
    renderer.autoExposureEnabled = false
    renderer.exposure = 0.9

    // Portrait-studio three-point: warm key, cool fill, rim; soft sky IBL.
    renderer.directionalLightDirection = simd_normalize(SIMD3<Float>(0.5, 0.75, 0.55))
    renderer.directionalLightColor = SIMD3(2.4, 2.2, 1.9)
    renderer.extraDirectionals = [
        IlluminatoramaDirectionalLight(dir: simd_normalize(SIMD3(-0.6, 0.3, 0.4)), color: SIMD3(0.5, 0.55, 0.7)),
        IlluminatoramaDirectionalLight(dir: simd_normalize(SIMD3(-0.2, 0.4, -0.9)), color: SIMD3(0.6, 0.55, 0.5)),
    ]
    renderer.ambientColor = SIMD3(0.14, 0.15, 0.17)

    let cloudSky = VolumetricCloudRenderer(engine: .shared, resolution: SIMD2<Int>(1024, 512))
    var skyParams = VolumetricCloudRenderer.Params()
    skyParams.time = 40
    skyParams.cameraPos = camera.position
    skyParams.sunDir = -renderer.directionalLightDirection
    skyParams.coverage = 0.35
    skyParams.sunIntensity = 8
    cloudSky.render(params: skyParams)
    renderer.equirectSky = cloudSky.outputTexture
    renderer.iblEnabled = true
    renderer.iblIntensity = 0.6

    let ground = IlluminatoramaRenderer.InstanceRef(
        meshKind: .ground,
        data: IlluminatoramaInstance(
            modelMatrix: float4x4(diagonal: SIMD4(400, 1, 400, 1)),
            albedo: SIMD3(0.34, 0.34, 0.36), metallic: 0, roughness: 0.9),
        lightEmission: .zero, superquadricShape: nil)

    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    var meshRetainer: [IlluminatoramaMesh] = []

    for (index, (label, spec)) in specs.enumerated() {
        let human = GeneratedHuman(spec: spec)
        let clothed = !label.hasSuffix("-nude")
        let outfit = Outfit.random(seed: spec.seed &+ UInt64(index))

        // Walk-pose entries validate the full rig: CPU LBS reference over the
        // bind mesh at a mid-stride gait phase. Garments skin through the
        // identical path, so they follow the pose too.
        let posed: PosedHuman? = label.hasSuffix("-walking")
            ? PosedHuman(human: human, pose: WalkCycle.pose(phase: 0.18, human: human))
            : nil

        func makeMesh(
            positions: [SIMD3<Float>], normals: [SIMD3<Float>], uvs: [SIMD2<Float>],
            triangles: [UInt32], doubleSided: Bool, colors: [SIMD4<Float>]? = nil
        ) -> IlluminatoramaMesh {
            var vertices: [IlluminatoramaVertex] = []
            vertices.reserveCapacity(positions.count)
            for i in 0..<positions.count {
                vertices.append(IlluminatoramaVertex(
                    position: positions[i], normal: normals[i], uv: uvs[i],
                    color: colors?[i] ?? SIMD4(1, 1, 1, 1)))
            }
            let mesh = IlluminatoramaMesh(device: renderer.device, vertices: vertices, indices: triangles)
            mesh.doubleSided = doubleSided
            return mesh
        }

        var refs: [IlluminatoramaRenderer.InstanceRef] = [ground]

        var bodyPositions = human.positions
        var bodyNormals = human.normals
        if let posed { (bodyPositions, bodyNormals) = posed.skinned(human) }
        let bodyMesh = makeMesh(positions: bodyPositions, normals: bodyNormals,
                                uvs: human.uvs, triangles: human.triangles, doubleSided: false,
                                colors: SkinPainter.paint(human: human))
        let bodyKind = IlluminatoramaRenderer.MeshKind.custom("human-\(index)")
        renderer.setMesh(bodyKind, bodyMesh)
        meshRetainer.append(bodyMesh)

        let albedo = SkinPainter.baseTone(spec)
        refs.append(IlluminatoramaRenderer.InstanceRef(
            meshKind: bodyKind,
            data: IlluminatoramaInstance(
                modelMatrix: matrix_identity_float4x4,
                albedo: albedo, metallic: 0, roughness: 0.55),
            lightEmission: .zero, superquadricShape: nil))

        var garmentList = clothed ? ClothingBuilder.garments(for: human, outfit: outfit) : []
        var hairStyle = HairStyle.random(seed: spec.seed &+ 99, ageYears: spec.ageYears)
        if label.hasSuffix("-portrait") { hairStyle.bald = false }   // portraits judge the face
        if let hair = HairBuilder.hair(for: human, style: hairStyle) {
            garmentList.append(hair)
        }
        if let strands = StrandHairBuilder.strands(for: human, style: hairStyle) {
            garmentList.append(strands)
        }
        do {
            for (gi, garment) in garmentList.enumerated() {
                var gPositions = garment.positions
                var gNormals = garment.normals
                if let posed {
                    (gPositions, gNormals) = posed.skin(
                        positions: garment.positions, normals: garment.normals,
                        joints: garment.skinJoints, weights: garment.skinWeights)
                }
                let mesh = makeMesh(positions: gPositions, normals: gNormals,
                                    uvs: garment.uvs, triangles: garment.triangles, doubleSided: true)
                let kind = IlluminatoramaRenderer.MeshKind.custom("human-\(index)-garment-\(gi)")
                renderer.setMesh(kind, mesh)
                meshRetainer.append(mesh)
                refs.append(IlluminatoramaRenderer.InstanceRef(
                    meshKind: kind,
                    data: IlluminatoramaInstance(
                        modelMatrix: matrix_identity_float4x4,
                        albedo: garment.color, metallic: 0, roughness: garment.roughness),
                    lightEmission: .zero, superquadricShape: nil))
            }
        }

        for eye in EyeBuilder.eyes(for: human) {
            var center4 = SIMD4(eye.center, 1)
            if let posed { center4 = posed.boneWorlds[eye.boneIndex] * center4 }
            refs.append(IlluminatoramaRenderer.InstanceRef(
                meshKind: .sphere,
                data: IlluminatoramaInstance(
                    modelMatrix: translateScale(SIMD3(center4.x, center4.y, center4.z), eye.radius),
                    albedo: eye.color, metallic: 0, roughness: eye.roughness),
                lightEmission: .zero, superquadricShape: nil))
        }

        // Frame each subject by its actual height (kids and giants both fill the
        // shot); portraits frame the head to judge the face paint.
        let h = human.heightMeters
        if label.hasSuffix("-portrait") {
            renderer.camera.target = SIMD3(0, h * 0.915, 0)
            renderer.camera.position = SIMD3(h * 0.10, h * 0.93, h * 0.34)
        } else {
            renderer.camera.target = SIMD3(0, h * 0.52, 0)
            renderer.camera.position = SIMD3(h * 0.35, h * 0.62, h * 1.55)
        }

        renderer.instances = refs
        // A few frames so temporal passes settle before the readback.
        for _ in 0..<8 { _ = renderer.render(blocking: true) }

        let url = dir.appendingPathComponent(String(format: "%02d-%@.png", index, label))
        try writePNG(renderer.outputTexture, queue: renderer.commandQueue, to: url)
        print("wrote \(url.lastPathComponent)  height=\(String(format: "%.2f", human.heightMeters))m")
    }
}

func translateScale(_ t: SIMD3<Float>, _ s: Float) -> float4x4 {
    var m = float4x4(diagonal: SIMD4(s, s, s, 1))
    m.columns.3 = SIMD4(t, 1)
    return m
}

func writePNG(_ texture: MTLTexture, queue: MTLCommandQueue, to url: URL) throws {
    enum E: Error { case blit, image, write }
    let w = texture.width, h = texture.height, bpr = w * 4
    guard let buffer = queue.device.makeBuffer(length: bpr * h, options: .storageModeShared),
          let cb = queue.makeCommandBuffer(),
          let blit = cb.makeBlitCommandEncoder() else { throw E.blit }
    blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1),
              to: buffer, destinationOffset: 0,
              destinationBytesPerRow: bpr, destinationBytesPerImage: bpr * h)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    guard let provider = CGDataProvider(data: Data(bytes: buffer.contents(), count: bpr * h) as CFData),
          let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                              bytesPerRow: bpr, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                  | CGBitmapInfo.byteOrder32Little.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false,
                              intent: .defaultIntent) else { throw E.image }
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw E.write }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw E.write }
}

// Headless video: several clothed runners jog toward a fixed camera; frames are
// rendered offscreen and assembled with ffmpeg. `humans-preview --video [outdir]`.
@MainActor
func renderRunVideo(to dir: URL) throws {
    let width = 1280, height = 720
    let fps = 30
    let seconds: Float = 4.5
    let camera = IlluminatoramaCamera(
        position: SIMD3(0, 1.7, 6.0), target: SIMD3(0, 1.05, -3), up: SIMD3(0, 1, 0),
        fovYRadians: 0.75, aspect: Float(width) / Float(height), zNear: 0.05, zFar: 200)
    let renderer = try IlluminatoramaRenderer(engine: .shared, width: width, height: height, camera: camera)
    renderer.internalRenderScale = 1.5
    renderer.resize(width: width, height: height)
    renderer.sharedTAAOverride = false
    renderer.appliesSharedLensFX = false
    renderer.autoExposureEnabled = false
    renderer.exposure = 0.9
    renderer.directionalLightDirection = simd_normalize(SIMD3<Float>(0.5, 0.75, 0.55))
    renderer.directionalLightColor = SIMD3(2.4, 2.2, 1.9)
    renderer.extraDirectionals = [
        IlluminatoramaDirectionalLight(dir: simd_normalize(SIMD3(-0.6, 0.3, 0.4)), color: SIMD3(0.5, 0.55, 0.7)),
        IlluminatoramaDirectionalLight(dir: simd_normalize(SIMD3(-0.2, 0.4, -0.9)), color: SIMD3(0.6, 0.55, 0.5)),
    ]
    renderer.ambientColor = SIMD3(0.14, 0.15, 0.17)

    let cloudSky = VolumetricCloudRenderer(engine: .shared, resolution: SIMD2<Int>(1024, 512))
    var skyParams = VolumetricCloudRenderer.Params()
    skyParams.time = 40
    skyParams.cameraPos = camera.position
    skyParams.sunDir = -renderer.directionalLightDirection
    skyParams.coverage = 0.35
    skyParams.sunIntensity = 8
    cloudSky.render(params: skyParams)
    renderer.equirectSky = cloudSky.outputTexture
    renderer.iblEnabled = true
    renderer.iblIntensity = 0.6

    let ground = IlluminatoramaRenderer.InstanceRef(
        meshKind: .ground,
        data: IlluminatoramaInstance(
            modelMatrix: float4x4(diagonal: SIMD4(400, 1, 400, 1)),
            albedo: SIMD3(0.34, 0.34, 0.36), metallic: 0, roughness: 0.9),
        lightEmission: .zero, superquadricShape: nil)

    // The LIVE path: each runner's body + garments are GPU-skinned meshes
    // registered ONCE via registerGPUMesh; per frame only the matrix palette
    // changes and one compute pass runs. This is exactly how a game hosts them.
    struct Runner {
        let human: GeneratedHuman
        let body: GPUSkinnedHuman
        let bodyHandle: IlluminatoramaMeshHandle
        let garments: [(gpu: GPUSkinnedHuman, handle: IlluminatoramaMeshHandle, color: SIMD3<Float>, roughness: Float)]
        let lane: Float          // x offset
        let startZ: Float
        let phaseOffset: Float
        let speed: Float         // m/s toward +Z (the camera)
        let strideHz: Float
    }
    let specs: [(HumanSpec, Float, Float, Float)] = [   // spec, lane, z0, speed
        (HumanSpec(ageYears: 30, gender: 1, muscle: 0.7, skinTone: 0.2, seed: 11), -2.2, -11.0, 3.1),
        (HumanSpec(ageYears: 26, gender: 0, skinTone: 0.55, seed: 22), -0.75, -9.5, 3.0),
        (HumanSpec(ageYears: 45, gender: 1, weight: 0.85, skinTone: 0.75, seed: 33), 0.75, -10.5, 2.7),
        (HumanSpec(ageYears: 24, gender: 0, weight: 0.25, height: 0.8, skinTone: 0.9, seed: 44), 2.2, -8.8, 3.2),
    ]
    // Hand-picked outfits: random rolls can land skin-toned fabric, which reads
    // as missing clothing on camera.
    let outfits = [
        Outfit(shirtColor: SIMD3(0.15, 0.45, 0.70), pantsColor: SIMD3(0.12, 0.12, 0.16)),
        Outfit(shirtColor: SIMD3(0.85, 0.25, 0.20), pantsColor: SIMD3(0.10, 0.10, 0.12)),
        Outfit(shirtColor: SIMD3(0.90, 0.75, 0.15), pantsColor: SIMD3(0.15, 0.25, 0.45)),
        Outfit(shirtColor: SIMD3(0.30, 0.65, 0.35), pantsColor: SIMD3(0.30, 0.12, 0.35)),
    ]
    let runners: [Runner] = try specs.enumerated().map { i, s in
        let human = GeneratedHuman(spec: s.0)
        var garmentMeshes = ClothingBuilder.garments(for: human, outfit: outfits[i % outfits.count])
        if let hair = HairBuilder.hair(
            for: human, style: .random(seed: s.0.seed &+ 99, ageYears: s.0.ageYears)) {
            garmentMeshes.append(hair)
        }
        let body = try GPUSkinnedHuman(device: renderer.device, human: human)
        guard let bodyHandle = renderer.registerGPUMesh(body.gpuMeshDescriptor()) else {
            fatalError("registerGPUMesh failed for runner \(i) body")
        }
        let garments = try garmentMeshes.map { garment in
            let gpu = try GPUSkinnedHuman(device: renderer.device, garment: garment, boneCount: human.bones.count)
            guard let handle = renderer.registerGPUMesh(gpu.gpuMeshDescriptor()) else {
                fatalError("registerGPUMesh failed for runner \(i) garment")
            }
            return (gpu: gpu, handle: handle, color: garment.color, roughness: garment.roughness)
        }
        return Runner(
            human: human, body: body, bodyHandle: bodyHandle, garments: garments,
            lane: s.1, startZ: s.2, phaseOffset: Float(i) * 0.31,
            speed: s.3,
            // Stride rate scales with leg length: shorter runners take quicker steps.
            strideHz: 1.45 * (1.75 / max(1.2, human.heightMeters)))
    }

    let framesDir = dir.appendingPathComponent("frames")
    try? FileManager.default.removeItem(at: framesDir)
    try FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)

    func translation(_ t: SIMD3<Float>) -> float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4(t, 1)
        return m
    }

    guard let skinQueue = renderer.device.makeCommandQueue() else {
        fatalError("no command queue for skinning")
    }
    let frameCount = Int(seconds * Float(fps))
    for frame in -8..<frameCount {         // negative frames warm the temporal passes
        let t = Float(max(0, frame)) / Float(fps)
        var refs: [IlluminatoramaRenderer.InstanceRef] = [ground]

        // One command buffer skins every character; offline we block, live the
        // vsync gap covers it.
        guard let skinCB = skinQueue.makeCommandBuffer() else { fatalError("no skin CB") }
        for runner in runners {
            let phase = t * runner.strideHz + runner.phaseOffset
            let pose = WalkCycle.pose(phase: phase, vigor: 1.1, human: runner.human, running: true)
            let posed = PosedHuman(human: runner.human, pose: pose)
            runner.body.encode(pose: posed, into: skinCB)
            for garment in runner.garments {
                garment.gpu.encode(pose: posed, into: skinCB)
            }
        }
        skinCB.commit()
        skinCB.waitUntilCompleted()

        for runner in runners {
            let human = runner.human
            let phase = t * runner.strideHz + runner.phaseOffset
            let pose = WalkCycle.pose(phase: phase, vigor: 1.1, human: human, running: true)
            let posed = PosedHuman(human: human, pose: pose)
            let model = translation(SIMD3(
                runner.lane, WalkCycle.runBob(phase: phase), runner.startZ + runner.speed * t))

            refs.append(IlluminatoramaRenderer.InstanceRef(
                meshKind: runner.bodyHandle.kind,
                data: IlluminatoramaInstance(
                    modelMatrix: model, albedo: SkinPainter.baseTone(human.spec),
                    metallic: 0, roughness: 0.55),
                lightEmission: .zero, superquadricShape: nil))
            for garment in runner.garments {
                refs.append(IlluminatoramaRenderer.InstanceRef(
                    meshKind: garment.handle.kind,
                    data: IlluminatoramaInstance(
                        modelMatrix: model, albedo: garment.color,
                        metallic: 0, roughness: garment.roughness),
                    lightEmission: .zero, superquadricShape: nil))
            }

            for eye in EyeBuilder.eyes(for: human) {
                let c4 = model * posed.boneWorlds[eye.boneIndex] * SIMD4(eye.center, 1)
                refs.append(IlluminatoramaRenderer.InstanceRef(
                    meshKind: .sphere,
                    data: IlluminatoramaInstance(
                        modelMatrix: translateScale(SIMD3(c4.x, c4.y, c4.z), eye.radius),
                        albedo: eye.color, metallic: 0, roughness: eye.roughness),
                    lightEmission: .zero, superquadricShape: nil))
            }
        }

        renderer.instances = refs
        _ = renderer.render(blocking: true)
        if frame >= 0 {
            try writePNG(renderer.outputTexture, queue: renderer.commandQueue,
                         to: framesDir.appendingPathComponent(String(format: "frame-%04d.png", frame)))
        }
        if frame % 30 == 0 { print("frame \(max(0, frame))/\(frameCount)") }
    }

    let movie = dir.appendingPathComponent("runners.mp4")
    let ffmpeg = Process()
    ffmpeg.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    ffmpeg.arguments = [
        "ffmpeg", "-y", "-framerate", "\(fps)",
        "-i", framesDir.appendingPathComponent("frame-%04d.png").path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", movie.path,
    ]
    ffmpeg.standardOutput = FileHandle.nullDevice
    ffmpeg.standardError = FileHandle.nullDevice
    try ffmpeg.run()
    ffmpeg.waitUntilExit()
    guard ffmpeg.terminationStatus == 0 else {
        fatalError("ffmpeg failed (\(ffmpeg.terminationStatus))")
    }
    print("wrote \(movie.path)")
}

let cliArgs = CommandLine.arguments.dropFirst()
let videoMode = cliArgs.contains("--video")
let pathArg = cliArgs.first { !$0.hasPrefix("--") }
let out = URL(fileURLWithPath: pathArg ?? "build/human-previews")

let sweep: [(String, HumanSpec)] = [
    ("female-young-average-nude", HumanSpec(ageYears: 28, gender: 0)),
    ("male-young-average-nude", HumanSpec(ageYears: 28, gender: 1)),
    ("male-heavy", HumanSpec(ageYears: 45, gender: 1, weight: 0.9, skinTone: 0.15)),
    ("female-muscular", HumanSpec(ageYears: 32, gender: 0, muscle: 0.9, skinTone: 0.6)),
    ("child", HumanSpec(ageYears: 7, gender: 0.5, skinTone: 0.4)),
    ("elderly-male", HumanSpec(ageYears: 80, gender: 1, muscle: 0.25, skinTone: 0.75)),
    ("tall-thin-female", HumanSpec(ageYears: 24, gender: 0, weight: 0.2, height: 0.85, skinTone: 0.9)),
    ("male-minweight", HumanSpec(ageYears: 30, gender: 1, weight: 0.0)),
    ("male-maxweight", HumanSpec(ageYears: 30, gender: 1, weight: 1.0)),
    ("male-walking", HumanSpec(ageYears: 30, gender: 1)),
    ("female-walking", HumanSpec(ageYears: 26, gender: 0, skinTone: 0.55)),
    ("female-portrait", HumanSpec(ageYears: 27, gender: 0, skinTone: 0.3, seed: 5)),
    ("elderly-portrait", HumanSpec(ageYears: 76, gender: 1, skinTone: 0.6, seed: 6)),
    ("male-potbelly", HumanSpec(ageYears: 55, gender: 1, weight: 0.7, belly: 1.0, seed: 7)),
    ("female-pear", HumanSpec(ageYears: 35, gender: 0, skinTone: 0.5, archetype: .pear,
                              archetypeAmount: 1.0, glutes: 0.85, hips: 0.8, seed: 8)),
]

try await MainActor.run {
    if videoMode {
        try renderRunVideo(to: out)
    } else {
        try render(specs: sweep, to: out)
    }
}
