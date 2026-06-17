import Foundation
import Metal
import simd

// ── EggMotionKernel ──────────────────────────────────────────────────────────
//
// GPU-resident per-egg motion advance + LUT-sampled rail position +
// orientation matrix build. Replaces the CPU `placeEgg` + `network.sample`
// hot path. One thread per egg.
//
// Caller responsibility:
//   • Allocate the LUT buffer once per network rebuild from
//     `EggsRails.Network.bakeLUT()`.
//   • Allocate the per-egg state buffer at egg-population time. The buffer
//     is shared-storage so CPU passes (clampSpacing, POV knock-off) can
//     read/write `s` directly.
//   • Allocate the transform output buffer (also shared storage so the
//     controller can copy results into SCNNode `simdTransform`s without
//     a readback hop).
//   • Per tick: write the uniforms (dt, globalSpeed, povBoost), dispatch,
//     wait for completion, copy transforms into nodes.

/// Mirror of `EggMotionState` in EggMotion.metal.
public struct EggMotionState {
    public var s: Float
    public var speed: Float
    public var liftCurrent: Float
    public var rollOffset: Float
    public var rollAccum: Float
    public var rollingRadius: Float
    public var motionMode: UInt32   // 0=roll, 1=glide
    public var povBoost: UInt32     // 1 = apply pov-boost uniform
    public var scale: Float         // uniform per-egg scale; baked into transform
    public var _pad0: Float
    public var _pad1: Float
    public var _pad2: Float

    public init(
        s: Float, speed: Float, liftCurrent: Float,
        rollOffset: Float, rollAccum: Float, rollingRadius: Float,
        motionMode: UInt32, povBoost: UInt32, scale: Float
    ) {
        self.s = s
        self.speed = speed
        self.liftCurrent = liftCurrent
        self.rollOffset = rollOffset
        self.rollAccum = rollAccum
        self.rollingRadius = rollingRadius
        self.motionMode = motionMode
        self.povBoost = povBoost
        self.scale = scale
        self._pad0 = 0
        self._pad1 = 0
        self._pad2 = 0
    }
}

/// Mirror of `EggMotionUniforms` in EggMotion.metal.
public struct EggMotionUniforms {
    public var dt: Float
    public var globalSpeed: Float
    public var povBoost: Float
    public var loopLength: Float
    public var lutStep: Float
    public var lutSlotCount: UInt32
    public var eggCount: UInt32
    public var _pad0: Float

    public init(
        dt: Float, globalSpeed: Float, povBoost: Float,
        loopLength: Float, lutStep: Float,
        lutSlotCount: UInt32, eggCount: UInt32
    ) {
        self.dt = dt
        self.globalSpeed = globalSpeed
        self.povBoost = povBoost
        self.loopLength = loopLength
        self.lutStep = lutStep
        self.lutSlotCount = lutSlotCount
        self.eggCount = eggCount
        self._pad0 = 0
    }
}

/// 3 columns of rotation + 1 column of position, packed as 4 × float4 to
/// avoid alignment pitfalls in the shared struct. `EggMotionKernel.readMatrix`
/// converts to `simd_float4x4` for SceneKit.
public struct EggTransform {
    public var col0: SIMD4<Float>
    public var col1: SIMD4<Float>
    public var col2: SIMD4<Float>
    public var col3: SIMD4<Float>

    public init() {
        self.col0 = SIMD4<Float>(1, 0, 0, 0)
        self.col1 = SIMD4<Float>(0, 1, 0, 0)
        self.col2 = SIMD4<Float>(0, 0, 1, 0)
        self.col3 = SIMD4<Float>(0, 0, 0, 1)
    }

    public var matrix: simd_float4x4 {
        simd_float4x4(columns: (col0, col1, col2, col3))
    }
}

@MainActor
public final class EggMotionKernel {

    private let engine: SimEngine
    private let pipeline: MTLComputePipelineState
    private let threadgroupSize: Int

    public init?(engine: SimEngine = .shared) {
        self.engine = engine
        guard let pipeline = engine.pipeline("eggs_advance_motion") else {
            return nil
        }
        self.pipeline = pipeline
        self.threadgroupSize = min(64, pipeline.maxTotalThreadsPerThreadgroup)
    }

    /// Build a buffer holding the network LUT for the egg motion kernel.
    /// The returned buffer is the exact layout the kernel reads.
    public func makeLUTBuffer(from lut: [SIMD4<Float>]) -> MTLBuffer? {
        guard !lut.isEmpty else { return nil }
        let size = lut.count * MemoryLayout<SIMD4<Float>>.stride
        guard let buf = engine.device.makeBuffer(length: size, options: .storageModeShared)
        else { return nil }
        buf.label = "EggMotion.lut"
        lut.withUnsafeBufferPointer { src in
            memcpy(buf.contents(), src.baseAddress, size)
        }
        return buf
    }

    /// Allocate a state buffer for `eggCount` eggs.
    public func makeStateBuffer(eggCount: Int) -> MTLBuffer? {
        let size = max(1, eggCount) * MemoryLayout<EggMotionState>.stride
        guard let buf = engine.device.makeBuffer(length: size, options: .storageModeShared)
        else { return nil }
        buf.label = "EggMotion.state"
        memset(buf.contents(), 0, size)
        return buf
    }

    /// Allocate a transform output buffer for `eggCount` eggs.
    public func makeTransformBuffer(eggCount: Int) -> MTLBuffer? {
        let size = max(1, eggCount) * MemoryLayout<EggTransform>.stride
        guard let buf = engine.device.makeBuffer(length: size, options: .storageModeShared)
        else { return nil }
        buf.label = "EggMotion.transforms"
        // Initialise to identity so a pre-first-tick read produces no NaNs.
        let ptr = buf.contents().bindMemory(to: EggTransform.self, capacity: eggCount)
        for i in 0..<eggCount {
            ptr[i] = EggTransform()
        }
        return buf
    }

    /// Dispatch the motion advance. Returns the command buffer so the
    /// caller can chain other work (e.g., RT emissive) into the same submit.
    public func dispatch(
        uniforms: EggMotionUniforms,
        state: MTLBuffer,
        transforms: MTLBuffer,
        lut: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) {
        guard uniforms.eggCount > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "EggMotionKernel.dispatch"
        enc.setComputePipelineState(pipeline)
        var u = uniforms
        enc.setBytes(&u, length: MemoryLayout<EggMotionUniforms>.stride, index: 0)
        enc.setBuffer(state, offset: 0, index: 1)
        enc.setBuffer(transforms, offset: 0, index: 2)
        enc.setBuffer(lut, offset: 0, index: 3)
        let grid = MTLSize(width: Int(uniforms.eggCount), height: 1, depth: 1)
        let group = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        enc.endEncoding()
    }
}
