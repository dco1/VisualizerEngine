import XCTest
import Metal
import SceneKit
import simd
import AppKit
@testable import VisualizerRendering

/// A self-contained VISUAL demo of `RigidPileField`: drops a heap of mixed coins,
/// marbles, and dice into a bin, settles it, and snapshots the pile to a PNG via
/// SceneKit (offscreen) — proving the whole chain (mixed-shape solver → per-type
/// CoinInstancedRenderer → instanced SCNGeometry) renders end-to-end. Isolated from
/// the app (no SceneManifest / Illuminatorama), so it needs nothing registered.
///
/// Skipped unless VIZ_PILE_DEMO=1 (offscreen SceneKit snapshot is slow for CI).
/// Run:  VIZ_PILE_DEMO=1 swift test --filter PileOfMessDemoTests
/// Out:  build/renders/pile-of-mess.png
@MainActor
final class PileOfMessDemoTests: XCTestCase {

    private static func makeLibrary(_ device: MTLDevice) throws -> MTLLibrary {
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/CoinDEM.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("CoinDEM.metal not found")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    func testRenderPileOfMess() throws {
        guard ProcessInfo.processInfo.environment["VIZ_PILE_DEMO"] != nil else {
            throw XCTSkip("set VIZ_PILE_DEMO=1 to render the demo PNG")
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal device") }
        let engine = SimEngine(device: device)
        let lib = try Self.makeLibrary(device)
        guard let expandFn = lib.makeFunction(name: "coin_expand_instances"),
              let expandPipe = try? device.makeComputePipelineState(function: expandFn),  // gpu-ok: test
              let queue = device.makeCommandQueue()                                        // gpu-ok: test
        else { throw XCTSkip("expand pipeline / queue") }

        // ── Solver: a mixed heap in a bin ──────────────────────────────────────
        let maxBodies = 220
        var cfg = RigidPileField.Config(maxBodies: maxBodies, bodyScale: 0.09,
                                        bounds: (SIMD3(-1.2, -0.5, -1.2), SIMD3(1.2, 2.5, 1.2)))
        cfg.floorY = 0
        cfg.restitution = 0.18
        cfg.friction = 0.86
        cfg.maxHorizontalSpeed = 2.5
        let inner = SIMD2<Float>(0.55, 0.55)
        guard let pile = RigidPileField(engine: engine, library: lib, config: cfg,
                                        colliders: RigidPileField.bin(innerHalf: inner, floorY: 0))
        else { throw XCTSkip("pile init") }

        // ── Three type-filtered renderers sharing the one pile ─────────────────
        let coinMesh = CoinMesh.make(radius: 0.085, halfThickness: 0.011, radialSegments: 28, rings: 4, text: "")
        let marbleMesh = SphereMesh.make(radius: 0.07)
        let dieMesh = BoxMesh.unit()
        func renderer(_ mesh: CoinMesh.Mesh, type: UInt32) -> CoinInstancedRenderer? {
            let r = CoinInstancedRenderer(engine: engine, pipeline: expandPipe, mesh: mesh, maxInstances: maxBodies)
            r?.instanceType = type; r?.filterByType = true; r?.setActiveInstanceCount(maxBodies)
            return r
        }
        guard let coinR = renderer(coinMesh, type: 0),
              let marbleR = renderer(marbleMesh, type: 1),
              let dieR = renderer(dieMesh, type: 2) else { throw XCTSkip("renderer init") }

        // ── Drop a mess: coins, marbles, dice in distinct colours ──────────────
        var seed: UInt64 = 0xD1CE
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1; return Float(seed >> 40) / Float(1 << 24) }
        let marblePalette: [SIMD4<Float>] = [SIMD4(0.10,0.45,0.95,1), SIMD4(0.95,0.25,0.30,1),
                                             SIMD4(0.15,0.80,0.45,1), SIMD4(0.95,0.75,0.10,1),
                                             SIMD4(0.65,0.30,0.90,1)]
        let diePalette: [SIMD4<Float>] = [SIMD4(0.92,0.90,0.85,1), SIMD4(0.85,0.20,0.20,1), SIMD4(0.15,0.18,0.22,1)]
        func drop(frameStart: Int) {
            for k in 0..<3 {
                let p = SIMD3<Float>((rnd()-0.5)*0.8, 1.4 + rnd()*0.6, (rnd()-0.5)*0.8)
                let tumble = SIMD3<Float>((rnd()-0.5)*5, (rnd()-0.5)*5, (rnd()-0.5)*5)
                let pick = (frameStart + k) % 3
                if pick == 0, let id = pile.dropDisc(at: p, radius: 0.085, halfThickness: 0.011, tumble: tumble, type: 0) {
                    coinR.setInstanceColor(id, SIMD4(1.0, 0.80, 0.32, 1))            // gold
                } else if pick == 1, let id = pile.dropSphere(at: p, radius: 0.07, tumble: tumble, type: 1) {
                    marbleR.setInstanceColor(id, marblePalette[Int(rnd()*5) % 5])
                } else {
                    let he = 0.045 + rnd()*0.02
                    if let id = pile.dropBox(at: p, halfExtents: SIMD3(he,he,he), tumble: tumble, type: 2) {
                        dieR.setInstanceColor(id, diePalette[Int(rnd()*3) % 3])
                    }
                }
            }
        }

        // ── Step: drop in waves, settle ────────────────────────────────────────
        let renderers = [coinR, marbleR, dieR]
        for f in 0..<520 {
            if f < 200 && f % 6 == 0 { drop(frameStart: f) }     // ~100 bodies over the first waves
            guard let cb = queue.makeCommandBuffer() else { break }
            pile.encode(to: cb, dt: 1.0/60.0)
            for r in renderers {
                r.setActiveInstanceCount(pile.solver.highWater)
                r.dispatchExpand(transforms: pile.solver.transformBuffer,
                                 bodyType: pile.solver.bodyTypeBuffer, in: cb)
            }
            cb.commit(); cb.waitUntilCompleted()                 // gpu-ok: test render harness
        }
        print("PILE_DEMO bodies=\(pile.activeCount)")

        // ── Build the SceneKit scene around the live instanced geometries ──────
        let scene = SCNScene()
        func node(_ r: CoinInstancedRenderer, metalness: Float, roughness: Float) -> SCNNode {
            let m = SCNMaterial()
            m.lightingModel = .physicallyBased
            m.diffuse.contents = NSColor.white                   // vertex colour carries the hue
            m.metalness.contents = metalness
            m.roughness.contents = roughness
            r.geometry.materials = [m]
            return SCNNode(geometry: r.geometry)
        }
        scene.rootNode.addChildNode(node(coinR, metalness: 1.0, roughness: 0.28))
        scene.rootNode.addChildNode(node(marbleR, metalness: 0.0, roughness: 0.12))
        scene.rootNode.addChildNode(node(dieR, metalness: 0.0, roughness: 0.45))

        // Floor.
        let floor = SCNFloor(); floor.reflectivity = 0.08
        let fm = SCNMaterial(); fm.lightingModel = .physicallyBased
        fm.diffuse.contents = NSColor(calibratedWhite: 0.22, alpha: 1); fm.roughness.contents = 0.7
        floor.materials = [fm]
        scene.rootNode.addChildNode(SCNNode(geometry: floor))

        // Lighting: a warm key + cool fill + a procedural gradient environment for PBR.
        // Kept deliberately dim — instanced PBR + a bright env blows out fast.
        let env = Self.gradient(top: NSColor(calibratedRed: 0.34, green: 0.40, blue: 0.52, alpha: 1),
                                bottom: NSColor(calibratedRed: 0.08, green: 0.08, blue: 0.10, alpha: 1))
        scene.lightingEnvironment.contents = env
        scene.lightingEnvironment.intensity = 0.7
        scene.background.contents = env
        let key = SCNLight(); key.type = .directional; key.intensity = 620; key.castsShadow = true
        key.temperature = 5200; key.shadowMode = .deferred; key.shadowRadius = 5
        let keyNode = SCNNode(); keyNode.light = key; keyNode.eulerAngles = SCNVector3(-1.0, 0.5, 0)
        scene.rootNode.addChildNode(keyNode)
        let fill = SCNLight(); fill.type = .omni; fill.intensity = 130; fill.temperature = 7200
        let fillNode = SCNNode(); fillNode.light = fill; fillNode.position = SCNVector3(-1.5, 1.2, 1.5)
        scene.rootNode.addChildNode(fillNode)

        // Camera looking down into the bin.
        let cam = SCNCamera(); cam.fieldOfView = 46; cam.wantsHDR = true; cam.bloomIntensity = 0.08
        cam.wantsExposureAdaptation = false; cam.exposureOffset = -0.7; cam.bloomThreshold = 0.9
        let camNode = SCNNode(); camNode.camera = cam
        camNode.position = SCNVector3(0.0, 0.62, 1.25)
        camNode.look(at: SCNVector3(0, 0.12, 0))
        scene.rootNode.addChildNode(camNode)

        // ── Offscreen snapshot ─────────────────────────────────────────────────
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene
        renderer.pointOfView = camNode
        let img = renderer.snapshot(atTime: 0, with: CGSize(width: 1100, height: 720),
                                    antialiasingMode: .multisampling4X)

        let outDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()   // repo root
            .appendingPathComponent("build/renders")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let outURL = outDir.appendingPathComponent("pile-of-mess.png")
        if let tiff = img.tiffRepresentation, let bm = NSBitmapImageRep(data: tiff),
           let png = bm.representation(using: .png, properties: [:]) {
            try png.write(to: outURL)
            print("PILE_DEMO wrote \(outURL.path)")
        }
        XCTAssertGreaterThan(pile.activeCount, 50, "the demo dropped a real heap")
    }

    /// A simple vertical-gradient equirect-ish image for the PBR environment + bg.
    private static func gradient(top: NSColor, bottom: NSColor) -> NSImage {
        let w = 16, h = 256
        let img = NSImage(size: NSSize(width: w, height: h))
        img.lockFocus()
        let g = NSGradient(starting: bottom, ending: top)
        g?.draw(in: NSRect(x: 0, y: 0, width: w, height: h), angle: 90)
        img.unlockFocus()
        return img
    }
}
