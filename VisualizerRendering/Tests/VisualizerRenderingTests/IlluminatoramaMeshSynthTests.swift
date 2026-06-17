import XCTest
import Metal
import simd
@testable import VisualizerRendering

/// Deterministic GPU test for the Phase 4.21 mesh normal/tangent synthesis
/// kernels (`illumi_mesh_build_adjacency` + `illumi_mesh_synth`). These replaced
/// the CPU `synthesiseNormals` / `synthesiseTangents` passes that used to run on
/// the main thread inside `IlluminatoramaMesh.from` and hung the run loop on a
/// cold-cache scene (~9 s in `accumulateTangents`).
///
/// The test drives the real kernels with the SAME 3-step encoding as
/// `IlluminatoramaRenderer.synthesiseMeshGeometry` (blit-clear → per-triangle
/// adjacency → per-vertex gather, the adjacency dispatch in its own encoder so
/// the gather observes its writes), on an analytically-known planar quad, then
/// reads the vertex buffer back and checks the synthesised TBN frame — no flaky
/// full-app harness in the loop.
///
/// The SwiftPM CLI does NOT compile the package's `.metal` into a metallib (an
/// Xcode-only build rule — it only copies the sources), so the kernels are
/// compiled from source at runtime via `makeLibrary(source:)`.
final class IlluminatoramaMeshSynthTests: XCTestCase {

    private static let maxValence: UInt32 = 24

    private static func makeShaderLibrary(device: MTLDevice) throws -> MTLLibrary {
        // #filePath: …/Tests/VisualizerRenderingTests/IlluminatoramaMeshSynthTests.swift
        // shader:    …/Sources/VisualizerRendering/Shaders/Illuminatorama.metal
        let shader = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // VisualizerRenderingTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // package root
            .appendingPathComponent("Sources/VisualizerRendering/Shaders/Illuminatorama.metal")
        guard let src = try? String(contentsOf: shader, encoding: .utf8) else {
            throw XCTSkip("shader source not found at \(shader.path)")
        }
        return try device.makeLibrary(source: src, options: nil)
    }

    /// Run the real two-kernel synth on `verts` / `indices` and return the
    /// rewritten vertex array.
    @MainActor
    private func runSynth(verts: [IlluminatoramaVertex],
                          indices: [UInt16],
                          doNormals: Bool,
                          doTangents: Bool) throws -> [IlluminatoramaVertex] {
        let engine = SimEngine.shared
        let device = engine.device
        let queue = engine.commandQueue

        let lib = try Self.makeShaderLibrary(device: device)
        guard let adjFn = lib.makeFunction(name: "illumi_mesh_build_adjacency"),
              let synthFn = lib.makeFunction(name: "illumi_mesh_synth") else {
            throw XCTSkip("synth kernels missing from compiled library: \(lib.functionNames.sorted())")
        }
        let adjPipe = try device.makeComputePipelineState(function: adjFn)
        let synthPipe = try device.makeComputePipelineState(function: synthFn)

        let vertCount = verts.count
        let triCount = indices.count / 3
        let stride = MemoryLayout<IlluminatoramaVertex>.stride
        guard let vbuf = device.makeBuffer(bytes: verts, length: stride * vertCount,
                                           options: .storageModeShared),
              let ibuf = device.makeBuffer(bytes: indices,
                                           length: MemoryLayout<UInt16>.stride * indices.count,
                                           options: .storageModeShared),
              let cb = queue.makeCommandBuffer() else {
            throw XCTSkip("buffer / command-buffer allocation failed")
        }
        let countBytes = MemoryLayout<UInt32>.stride * vertCount
        let listBytes = MemoryLayout<UInt32>.stride * vertCount * Int(Self.maxValence)
        guard let countBuf = device.makeBuffer(length: countBytes, options: .storageModeShared),
              let listBuf = device.makeBuffer(length: listBytes, options: .storageModeShared) else {
            throw XCTSkip("scratch allocation failed")
        }

        var isU32: UInt32 = 0
        var triCountU = UInt32(triCount)
        var vertCountU = UInt32(vertCount)
        var maxValenceU = Self.maxValence
        var doN: UInt32 = doNormals ? 1 : 0
        var doT: UInt32 = doTangents ? 1 : 0

        if let blit = cb.makeBlitCommandEncoder() {
            blit.fill(buffer: countBuf, range: 0..<countBytes, value: 0)
            blit.endEncoding()
        }
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(adjPipe)
            enc.setBuffer(ibuf, offset: 0, index: 0)
            enc.setBuffer(ibuf, offset: 0, index: 1)
            enc.setBytes(&isU32, length: 4, index: 2)
            enc.setBytes(&triCountU, length: 4, index: 3)
            enc.setBytes(&maxValenceU, length: 4, index: 4)
            enc.setBuffer(countBuf, offset: 0, index: 5)
            enc.setBuffer(listBuf, offset: 0, index: 6)
            enc.setBytes(&vertCountU, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: triCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
        }
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(synthPipe)
            enc.setBuffer(vbuf, offset: 0, index: 0)
            enc.setBuffer(ibuf, offset: 0, index: 1)
            enc.setBuffer(ibuf, offset: 0, index: 2)
            enc.setBytes(&isU32, length: 4, index: 3)
            enc.setBytes(&vertCountU, length: 4, index: 4)
            enc.setBytes(&maxValenceU, length: 4, index: 5)
            enc.setBuffer(countBuf, offset: 0, index: 6)
            enc.setBuffer(listBuf, offset: 0, index: 7)
            enc.setBytes(&doN, length: 4, index: 8)
            enc.setBytes(&doT, length: 4, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: vertCount, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted() // gpu-ok: test-only synchronous readback of the synth result
        XCTAssertNil(cb.error, "GPU command buffer errored: \(String(describing: cb.error))")

        let p = vbuf.contents().bindMemory(to: IlluminatoramaVertex.self, capacity: vertCount)
        return (0..<vertCount).map { p[$0] }
    }

    /// Planar quad in the XZ plane; UVs map u→+X, v→+Z. Standard fixture.
    private func quad() -> (verts: [IlluminatoramaVertex], indices: [UInt16]) {
        let v: [IlluminatoramaVertex] = [
            IlluminatoramaVertex(position: [0, 0, 0], normal: [0, 1, 0], uv: [0, 0]),
            IlluminatoramaVertex(position: [1, 0, 0], normal: [0, 1, 0], uv: [1, 0]),
            IlluminatoramaVertex(position: [1, 0, 1], normal: [0, 1, 0], uv: [1, 1]),
            IlluminatoramaVertex(position: [0, 0, 1], normal: [0, 1, 0], uv: [0, 1]),
        ]
        return (v, [0, 1, 2, 0, 2, 3])
    }

    /// Source ships no normals (`doNormals = 1`): kernel recomputes ±Y and a
    /// valid +X tangent.
    @MainActor
    func testNormalAndTangentSynthesis() throws {
        let (verts, indices) = quad()
        let out = try runSynth(verts: verts, indices: indices, doNormals: true, doTangents: true)
        for (i, v) in out.enumerated() {
            let n = v.normal
            let t = SIMD3<Float>(v.tangent.x, v.tangent.y, v.tangent.z)
            XCTAssertEqual(simd_length(n), 1, accuracy: 1e-3, "vert \(i) normal not unit: \(n)")
            XCTAssertEqual(abs(n.y), 1, accuracy: 1e-3, "vert \(i) normal not ±Y: \(n)")
            XCTAssertEqual(simd_length(t), 1, accuracy: 1e-3, "vert \(i) tangent not unit: \(t)")
            XCTAssertEqual(simd_dot(t, n), 0, accuracy: 1e-3, "vert \(i) tangent not ⟂ N: t=\(t) n=\(n)")
            XCTAssertEqual(abs(t.x), 1, accuracy: 1e-3, "vert \(i) tangent not along X: \(t)")
            XCTAssertEqual(v.tangent.w, 1, accuracy: 1e-6, "vert \(i) handedness != 1")
        }
    }

    /// The dominant production path: SCN geometry ships normals
    /// (`doNormals = 0`) and only tangents are synthesised — e.g. the room's
    /// walls. The supplied normal must be PRESERVED, and the tangent must come
    /// out valid and orthogonal to it. This is the path the blank-render risk
    /// hinged on.
    @MainActor
    func testTangentOnlyPreservesShippedNormal() throws {
        var (verts, indices) = quad()
        // Give a non-axis-aligned but unit normal the kernel must keep verbatim.
        let shipped = simd_normalize(SIMD3<Float>(0.2, 0.95, 0.1))
        for i in verts.indices { verts[i].normal = shipped }
        let out = try runSynth(verts: verts, indices: indices, doNormals: false, doTangents: true)
        for (i, v) in out.enumerated() {
            let n = v.normal
            let t = SIMD3<Float>(v.tangent.x, v.tangent.y, v.tangent.z)
            XCTAssertEqual(simd_distance(n, shipped), 0, accuracy: 1e-4,
                           "vert \(i) shipped normal not preserved: \(n) vs \(shipped)")
            XCTAssertEqual(simd_length(t), 1, accuracy: 1e-3, "vert \(i) tangent not unit: \(t)")
            XCTAssertEqual(simd_dot(t, n), 0, accuracy: 1e-3, "vert \(i) tangent not ⟂ N: t=\(t) n=\(n)")
            XCTAssertEqual(v.tangent.w, 1, accuracy: 1e-6, "vert \(i) handedness != 1")
        }
    }
}
