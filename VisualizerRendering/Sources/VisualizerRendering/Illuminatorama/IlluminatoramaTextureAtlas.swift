import AppKit
import CoreGraphics
import Foundation
import Metal
import OSLog
import VisualizerCore

// ── ILLUMINATORAMA TEXTURE ATLAS (Phase 4.0 — albedo only) ──────────────────
//
// The Phase 2.6 scene extractor surfaces `SCNMaterial.diffuse.contents` as
// an *average colour* whenever the contents are an `NSImage` / `CGImage` —
// it has no path to actually sample the texture in the G-buffer shader. The
// effect on real scenes (Eggs's corrugated warehouse walls, Forest's bark,
// VintageDiner's tile + tabletop) is a uniform mid-grey wash where SceneKit
// would have shown rich textured surfaces.
//
// This file plugs that gap with a tiny atlas: a single `texture2d_array`
// holding up to `capacity` slices, each rendered from an `NSImage` /
// `CGImage` via Core Graphics. The G-buffer fragment shader samples the
// atlas at `instance.albedoTextureSlice` when `>= 0`; otherwise the
// existing per-instance `albedo` colour is used (the old code path is
// fully preserved for solid-colour materials).
//
// Format choice: `.bgra8Unorm_srgb` lets the GPU's texture unit decode
// sRGB → linear at sample time, which is exactly what the BRDF wants. The
// uploaded bytes are raw sRGB BGRA from NSImage's `bitmapImageRep`; no
// per-sample math needed in the shader.
//
// Size choice: 256×256 × 32 slices = 8 MB. Conservative first cut; bump
// capacity / slice size once Eggs / Forest / VintageDiner are profiled.
// `register(image:)` returns nil when the atlas is full — caller falls
// back to `extractMaterial`'s average-colour path.
//
// Phase 4.25 (issue #60 item 5) — ASPECT-CORRECT storage. The slices are a
// uniform SQUARE `texture2d_array`, but the G-buffer sampler is
// `address::repeat` and tiled materials lean on HARDWARE repeat over the full
// square. Aspect-correctness and hardware tiling are in genuine tension on a
// uniform-slice array, so we resolve it as follows:
//   • Each source image is LETTERBOXED into its slice preserving aspect, and
//     the filled fraction is recorded per-slice in `uvScaleBuffer` (a `float2`
//     table indexed by slice). A square source ⇒ uvScale (1,1).
//   • Square slices keep the hardware-`repeat` fast path — bit-identical to the
//     pre-aspect code, no seam.
//   • Non-square slices are tiled MANUALLY in the shader as `fract(uv)*uvScale`
//     (with a half-texel inset so the bilinear footprint never bleeds into the
//     empty letterbox band). This reintroduces a thin filter seam at tile
//     boundaries — but ONLY on the aspect-corrected (formerly squished)
//     textures, which previously rendered horizontally-squished AND
//     downsampled. The seam is the declared cost of keeping one uniform-size
//     slice array; the heavier alternatives (per-aspect bucketed arrays / a
//     packed 2D atlas with per-object UV rects) preserve hardware tiling but
//     cost a real packer + per-object UV rects — deferred as a follow-on.

@MainActor
public final class IlluminatoramaTextureAtlas {

    private static let log = Logger(subsystem: AppLog.subsystem,
                                     category: "illuminatoramaTextureAtlas")

    /// Side of each slice, in pixels. Source images are downsampled (or
    /// upsampled) to this resolution at registration time.
    public let sliceSize: Int
    /// Current number of slices the atlas can hold. Grows on demand —
    /// reads via the public property; the underlying texture is
    /// reallocated when the cap is exceeded.
    public private(set) var capacity: Int
    /// The underlying `texture2d_array` bound into the G-buffer fragment
    /// shader. Mutable because growth re-allocates the texture and blits
    /// existing slices over. The renderer re-binds it each draw so
    /// reassignment is safe.
    public private(set) var texture: MTLTexture
    /// Phase 4.25 (issue #60 item 5) — per-slice UV scale, indexed by slice.
    /// Each entry is the fraction of the square slice the *aspect-preserved*
    /// (letterboxed) image fills: `(1, 1)` for a square source, `(1, 0.25)`
    /// for a 4:1 wide brick/tile, `(0.25, 1)` for a 1:4 tall plank. The
    /// G-buffer shader reads `uvScale[slice]` and, when it isn't `(1,1)`,
    /// tiles manually with `fract(uv) * uvScale` so a non-square texture reads
    /// at its true aspect instead of being squished into the square. Square
    /// slices keep the hardware-`repeat` fast path (no seam) — see the file
    /// header for the aspect-vs-hardware-tiling tradeoff. `SIMD2<Float>` is an
    /// 8-byte stride that matches Metal `float2` exactly (no SIMD3/float3
    /// padding trap). Mutable: `grow()` re-allocates it alongside the texture.
    public private(set) var uvScaleBuffer: MTLBuffer

    private let device: MTLDevice
    private let blitQueue: MTLCommandQueue
    private let pixelFormat: MTLPixelFormat
    private var nextSlice: Int = 0
    /// Cache keyed by the SCNMaterialProperty contents' identity, so a
    /// single shared `NSImage` referenced by many materials gets uploaded
    /// once. `slice == nil` means "we tried this content and it failed to
    /// convert" — skip future attempts.
    ///
    /// The entry RETAINS the keyed object: `ObjectIdentifier` is just the
    /// object's address, so caching it for a transient image (a procedural
    /// bake's fresh `CGImage`, released right after registration) left a
    /// stale key that a LATER image could collide with when the allocator
    /// recycled the address — a false cache HIT returning the dead image's
    /// slice. Symptom: a material randomly wearing another material's
    /// texture, stable within a run but flipping across runs with allocator
    /// state (Daydream's bistable washed-white lawn / "two stable exterior
    /// regimes"). Retaining the object pins the address for the cache's
    /// lifetime, so a live key can never alias. `reset()` releases them.
    private struct SliceCacheEntry { let retained: AnyObject; let slice: Int32? }
    private var sliceForObject: [ObjectIdentifier: SliceCacheEntry] = [:]

    /// Pixel format passed at init. Use `.bgra8Unorm_srgb` for colour
    /// (diffuse / albedo / emission) so the GPU sampler decodes sRGB → linear
    /// automatically. Use `.bgra8Unorm` (no sRGB) for non-colour data —
    /// metallic, roughness, normal maps — where applying a gamma curve
    /// would distort the values.
    public init(device: MTLDevice,
                pixelFormat: MTLPixelFormat = .bgra8Unorm_srgb,
                sliceSize: Int = 256,
                capacity: Int = 64) throws {
        self.pixelFormat = pixelFormat
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = pixelFormat
        d.width = sliceSize
        d.height = sliceSize
        d.arrayLength = capacity
        d.usage = [.shaderRead]
        // `.managed` would let us upload via blit on Intel; on Apple Silicon
        // `.shared` and `.managed` perform identically. Use `.shared` so
        // `texture.replace(region:…)` works directly without a blit encoder.
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d),
              let q = device.makeCommandQueue(), // gpu-ok: one-time setup; atlas is not a solver, no SimEngine needed
              let uv = Self.makeUVScaleBuffer(device: device, capacity: capacity) else {
            throw IlluminatoramaError.bufferAllocationFailed("albedoAtlas")
        }
        t.label = "Illuminatorama.albedoAtlas"
        q.label = "Illuminatorama.blitQueue"
        self.texture = t
        self.blitQueue = q
        self.uvScaleBuffer = uv
        self.device = device
        self.sliceSize = sliceSize
        self.capacity = capacity
    }

    /// Allocate a per-slice `float2` UV-scale buffer initialised to `(1, 1)`
    /// (square / hardware-repeat fast path) for every slot. Slots are
    /// overwritten by `upload(cgImage:to:)` as real images register; the
    /// default keeps any slot that's bound-but-never-written reading as a
    /// plain square sample.
    private static func makeUVScaleBuffer(device: MTLDevice, capacity: Int) -> MTLBuffer? {
        let stride = MemoryLayout<SIMD2<Float>>.stride   // 8 bytes == Metal float2
        guard let buf = device.makeBuffer(length: stride * capacity,
                                          options: .storageModeShared) else { return nil }
        buf.label = "Illuminatorama.atlasUVScale"
        let p = buf.contents().bindMemory(to: SIMD2<Float>.self, capacity: capacity)
        for i in 0..<capacity { p[i] = SIMD2<Float>(1, 1) }
        return buf
    }

    /// Drop the slice-allocation table. Underlying texture stays allocated
    /// for re-use; the old slices simply become "free" for future
    /// `register(image:)` calls to overwrite. Called on scene switch by
    /// the extractor.
    public func reset() {
        nextSlice = 0
        sliceForObject.removeAll(keepingCapacity: true)
        freeSlices.removeAll(keepingCapacity: true)
    }

    /// Slots explicitly RETURNED by the host — e.g. a slider-customized material whose
    /// params changed, leaving the old bake unreachable. `register` reuses these before
    /// consuming a fresh slot, so churned custom materials stop growing the atlas
    /// unboundedly (the terrazzo-slider issue). Any content-cache entry pointing at the
    /// freed slot is dropped (releasing its retained image).
    private var freeSlices: [Int32] = []
    /// TEST-OBSERVABLE: how many freed slots are currently awaiting reuse.
    public var freeSliceCount: Int { freeSlices.count }

    public func freeSlice(_ slice: Int32) {
        guard slice >= 0, Int(slice) < nextSlice, !freeSlices.contains(slice) else { return }
        sliceForObject = sliceForObject.filter { $0.value.slice != slice }
        freeSlices.append(slice)
    }

    /// Try to surface a slice index for the supplied SCNMaterialProperty
    /// contents. Returns `nil` if the contents can't be converted, if the
    /// atlas is full, or if no contents were supplied. Cached per content
    /// identity — repeat calls for the same image are O(1).
    public func register(contents: Any?) -> Int32? {
        guard let contents = contents else { return nil }
        // Pull a CGImage out of the typed-erased contents. The common
        // cases that arrive from SCNMaterial:
        //   • NSImage           — typical Asset Catalog / file loads
        //   • CGImage           — programmatic; Core Graphics output
        //   • String (path)     — file URL or asset name (rare here)
        //   • MTLTexture        — already on the GPU; not handled in 4.0
        //   • NSColor / CGColor — solid-colour material; not our path
        let key: ObjectIdentifier
        let keyObject: AnyObject
        let cg: CGImage
        if let img = contents as? NSImage {
            key = ObjectIdentifier(img)
            keyObject = img
            if let cached = sliceForObject[key] { return cached.slice }
            guard let extracted = Self.cgImage(from: img) else {
                sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: nil)
                return nil
            }
            cg = extracted
        } else if CFGetTypeID(contents as CFTypeRef) == CGImage.typeID {
            // `contents as? CGImage` is rejected by Swift 6 (CF conditional
            // downcasts always succeed at the type system level, which the
            // compiler flags as an error). Bridge through `CFTypeRef` and
            // then force-cast once the type ID confirms we have a CGImage.
            let img = contents as! CGImage
            key = ObjectIdentifier(img as AnyObject)
            keyObject = img as AnyObject
            if let cached = sliceForObject[key] { return cached.slice }
            cg = img
        } else {
            return nil
        }
        // Phase 4.4 — when the atlas runs out of slots, double its capacity
        // by blitting existing slices into a larger texture2d_array. The
        // renderer rebinds `albedoAtlas.texture` each draw, so the swap
        // is invisible from outside. Bounded by Metal's per-device limit
        // (typically 2048 array layers); we hard-cap below to keep VRAM
        // sane on small machines.
        // Reuse an explicitly-freed slot before consuming a fresh one (see `freeSlice`).
        let slice: Int32
        let reusedFreeSlot: Bool
        if let reused = freeSlices.popLast() {
            slice = reused; reusedFreeSlot = true
        } else {
            if nextSlice >= capacity {
                let maxCap = 1024
                guard capacity < maxCap, grow(to: min(capacity * 2, maxCap)) else {
                    Self.log.warning("Atlas exhausted at capacity=\(self.capacity, privacy: .public); falling back to average colour")
                    sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: nil)
                    return nil
                }
            }
            slice = Int32(nextSlice)
            nextSlice += 1
            reusedFreeSlot = false
        }
        if upload(cgImage: cg, to: Int(slice)) {
            sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: slice)
            return slice
        } else {
            // Failed upload — return the slot and remember the failure.
            if reusedFreeSlot { freeSlices.append(slice) } else { nextSlice -= 1 }
            sliceForObject[key] = SliceCacheEntry(retained: keyObject, slice: nil)
            return nil
        }
    }

    /// Reserve a fresh, uninitialised slice and return its index — for a
    /// LIVE (per-frame-updated) texture binding rather than a one-shot CGImage
    /// upload. The caller owns refreshing it each frame via
    /// `blitLiveSlice(_:from:on:)`. The slot is left at the default `uvScale`
    /// (1,1) so the shader samples it as a plain square. Returns nil only if
    /// the atlas is full and can't grow.
    ///
    /// Used by the Tennis Ball Painter to bind its GPU paint-accumulation
    /// textures (the wall canvas, the floor spatter sheet, and each ball's own
    /// localized stain texture) directly into the Illuminatorama material, so
    /// the accumulating painting and the per-ball contact-patch stains are
    /// sampled by the renderer with no CPU round-trip.
    public func reserveLiveSlice() -> Int32? {
        if nextSlice >= capacity {
            let maxCap = 1024
            guard capacity < maxCap, grow(to: min(capacity * 2, maxCap)) else {
                Self.log.warning("Atlas exhausted reserving live slice at capacity=\(self.capacity, privacy: .public)")
                return nil
            }
        }
        let slice = Int32(nextSlice)
        nextSlice += 1
        // Default this slice to mid-grey so a binding that hasn't been blitted
        // yet (the one-frame warmup) reads as a neutral surface, not garbage.
        let region = MTLRegionMake3D(0, 0, 0, sliceSize, sliceSize, 1)
        let bytesPerRow = sliceSize * 4
        let mid = [UInt8](repeating: 160, count: sliceSize * sliceSize * 4)
        mid.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            texture.replace(region: region, mipmapLevel: 0, slice: Int(slice),
                            withBytes: base, bytesPerRow: bytesPerRow,
                            bytesPerImage: bytesPerRow * sliceSize)
        }
        return slice
    }

    /// Copy a live source texture (same width/height as `sliceSize`, a
    /// blit-compatible BGRA8 format) into atlas `slice` on the supplied command
    /// buffer. `bgra8Unorm` ↔ `bgra8Unorm_srgb` are blit-compatible (identical
    /// byte layout; the sRGB-ness is a sampling-time view), so a kernel that
    /// writes encoded sRGB bytes into a `bgra8Unorm` source resolves correctly
    /// when sampled through this `bgra8Unorm_srgb` atlas. No `waitUntilCompleted`
    /// — the caller's render command buffer carries the blit.
    public func blitLiveSlice(_ slice: Int32, from src: MTLTexture,
                              on cb: MTLCommandBuffer) {
        guard slice >= 0, Int(slice) < capacity,
              src.width == sliceSize, src.height == sliceSize,
              let blit = cb.makeBlitCommandEncoder() else { return }
        blit.label = "Illuminatorama.atlas.liveBlit"
        blit.copy(from: src, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: sliceSize, height: sliceSize, depth: 1),
                  to: texture, destinationSlice: Int(slice), destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
    }

    /// Render a CGImage into a transient BGRA buffer at `sliceSize`,
    /// uploading it into the atlas at `slice`. Returns `true` on success.
    private func upload(cgImage: CGImage, to slice: Int) -> Bool {
        let bytesPerPixel = 4
        let bytesPerRow = sliceSize * bytesPerPixel
        let bufferSize = sliceSize * sliceSize * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        // bgra8Unorm_srgb expects BGRA channel order. CG's bitmap info uses
        // .premultipliedFirst + byteOrder32Little to lay out BGRA on a
        // little-endian host.
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
                       | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = buffer.withUnsafeMutableBytes({ raw -> CGContext? in
            guard let base = raw.baseAddress else { return nil }
            return CGContext(
                data: base,
                width: sliceSize,
                height: sliceSize,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: bitmapInfo
            )
        }) else {
            Self.log.error("CGContext init failed for slice \(slice, privacy: .public)")
            return false
        }
        // Phase 4.25 — aspect-preserving letterbox. Instead of squishing a
        // non-square source into the full square (which distorted Forest bark,
        // Eggs warehouse walls, VintageDiner tile), fit the image to the slice
        // preserving its aspect and record the filled fraction as `uvScale`.
        // A 4:1 wide tile fills the full width and the top quarter of the
        // height → uvScale (1, 0.25); the empty band stays zeroed and the
        // shader's half-texel-inset manual tiling never samples into it.
        //
        // Anchor at CG-top (y = sliceSize − drawH): a CGBitmapContext stores
        // row 0 at the image top, and Metal texture v=0 is buffer row 0, so
        // drawing at high CG-y lands the image in texture v ∈ [0, uvScale.y] —
        // exactly the sub-rect the shader samples. (origin x = 0 → u ∈ [0,
        // uvScale.x].)
        let aspect = Double(cgImage.width) / Double(max(cgImage.height, 1))  // w / h
        var sx: CGFloat = 1, sy: CGFloat = 1
        if aspect >= 1 { sy = CGFloat(1.0 / aspect) } else { sx = CGFloat(aspect) }
        let drawW = CGFloat(sliceSize) * sx
        let drawH = CGFloat(sliceSize) * sy
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: CGFloat(sliceSize) - drawH,
                                     width: drawW, height: drawH))
        uvScaleBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: capacity)[slice]
            = SIMD2<Float>(Float(sx), Float(sy))
        let region = MTLRegionMake3D(0, 0, 0, sliceSize, sliceSize, 1)
        buffer.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            texture.replace(
                region: region,
                mipmapLevel: 0,
                slice: slice,
                withBytes: base,
                bytesPerRow: bytesPerRow,
                bytesPerImage: bufferSize
            )
        }
        return true
    }

    /// Reallocate the underlying texture at a larger `arrayLength`,
    /// blitting existing slices into the new texture. Returns `false` if
    /// allocation or blit setup fails (caller falls back to "atlas full"
    /// behaviour). Synchronous because in-flight command buffers retain
    /// the old texture; the new texture takes over on subsequent draws.
    private func grow(to newCapacity: Int) -> Bool {
        guard newCapacity > capacity else { return false }
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = pixelFormat
        d.width = sliceSize
        d.height = sliceSize
        d.arrayLength = newCapacity
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let newTex = device.makeTexture(descriptor: d),
              let cb = blitQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            return false
        }
        newTex.label = "Illuminatorama.albedoAtlas[grown=\(newCapacity)]"
        // Copy slice 0..nextSlice from old texture into new texture at the
        // same indices. `nextSlice` is "the next slot to write", so it's
        // also the count of in-use slices.
        for slice in 0..<nextSlice {
            blit.copy(from: texture,
                      sourceSlice: slice, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: sliceSize, height: sliceSize, depth: 1),
                      to: newTex,
                      destinationSlice: slice, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted() // gpu-ok: atlas-grow is a one-shot setup, not per-frame
        // Grow the per-slice UV-scale table in lockstep with the texture:
        // carry the in-use entries (0..<nextSlice) forward, default the rest to
        // (1,1). CPU-side copy of a tiny shared buffer — no GPU blit needed.
        guard let newUV = Self.makeUVScaleBuffer(device: device, capacity: newCapacity) else {
            return false
        }
        let oldP = uvScaleBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: capacity)
        let newP = newUV.contents().bindMemory(to: SIMD2<Float>.self, capacity: newCapacity)
        for i in 0..<nextSlice { newP[i] = oldP[i] }
        uvScaleBuffer = newUV
        Self.log.info("Atlas grew \(self.capacity, privacy: .public) → \(newCapacity, privacy: .public) slices")
        texture = newTex
        capacity = newCapacity
        return true
    }

    private static func cgImage(from nsImage: NSImage) -> CGImage? {
        var rect = CGRect(origin: .zero, size: nsImage.size)
        return nsImage.cgImage(forProposedRect: &rect,
                                context: nil,
                                hints: nil)
    }
}
