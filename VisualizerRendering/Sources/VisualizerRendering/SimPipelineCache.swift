import Foundation
import Metal
import OSLog
import VisualizerCore

// ── SIM PIPELINE CACHE ───────────────────────────────────────────────────────
//
// Process-wide cache of compiled MTLComputePipelineState (and the MTLLibrary
// each pipeline was built from), keyed by device. Pipeline-state construction
// is the slowest part of solver init by an order of magnitude — a hot-dog
// spawn at 4/sec was paying ~25 ms on a cold device every quarter-second
// before this cache went in. With caching, every solver after the first on a
// given device is essentially free to construct on the Metal side.
//
// Keyed by MTLDevice: in normal use there's one device, so most entries have
// a single slot — but the per-device shape stays correct on machines with
// both an integrated and a discrete GPU.
//
// Two layers:
//
// 1. `pipelineState(name:device:)` — the generic, kernel-name-keyed entry
//    point. New solvers (an MLS-MPM fluid kernel, an SPH foam kernel) look
//    their pipelines up by Metal function name and get back a memoised state.
//
// 2. `pbdPipelines(for:)` — a typed bundle of the seven PBD kernels the
//    XPBD solver + tube renderer use. Returns a struct of named pipeline
//    states so callers don't repeat the lookup boilerplate. Memoised per
//    device just like the individual states; lives in this file so the PBD
//    code stays decoupled from "how the cache is structured."
//
// The old name was `PBDPipelineCache`. Renamed when promoting from a
// PBD-specific helper to the foundation for other GPU solvers (fluid, SPH,
// grass) sharing the same cache.

@MainActor
public final class SimPipelineCache {

    private static let log = Logger(subsystem: AppLog.subsystem, category: "SimPipelineCache")

    /// Process-wide default cache. New solvers and SimEngine.shared use this
    /// transparently so cache hits survive across scene rebuilds, asset spawns,
    /// and any path that constructs a new solver without threading an engine
    /// through. Matches the prior behaviour of `PBDPipelineCache.shared`.
    public static let shared = SimPipelineCache()

    public init() {}

    // ── Generic per-(device, name) pipeline cache ────────────────────────────

    private struct Key: Hashable {
        let device: ObjectIdentifier
        let name: String
    }

    private var states: [Key: MTLComputePipelineState] = [:]
    private var libraries: [ObjectIdentifier: MTLLibrary] = [:]

    /// The default Metal library shipped with this package
    /// (`device.makeDefaultLibrary(bundle: Bundle.module)`), cached per device.
    /// Returns nil if loading fails — useful for callers that want to do their
    /// own function lookups (e.g. SCNTechnique).
    ///
    /// The library is built by SwiftPM from `Sources/VisualizerRendering/Shaders/*.metal`
    /// (declared as `.process("Shaders")` in `Package.swift`) into a
    /// `default.metallib` placed inside the generated
    /// `VisualizerRendering_VisualizerRendering` resource bundle. That means
    /// every solver in this package ships its own kernels — host apps don't
    /// need to add the `.metal` files to their main target. Apps with their
    /// own scene-specific `.metal` files (e.g. caustics, RT reflections) keep
    /// loading those from `Bundle.main` directly; this cache only knows about
    /// the package's own library.
    public func library(for device: MTLDevice) -> MTLLibrary? {
        let key = ObjectIdentifier(device)
        if let lib = libraries[key] { return lib }
        do {
            let lib = try device.makeDefaultLibrary(bundle: Bundle.module)
            libraries[key] = lib
            return lib
        } catch {
            Self.log.error("makeDefaultLibrary(bundle: .module) failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Look up (and memoise) a compute pipeline state by its Metal function name.
    /// Returns nil if the function is missing from the library or if pipeline
    /// construction fails — both are programmer errors (typo / missing kernel
    /// in the compiled .metal source), logged at error level.
    public func pipelineState(name: String,
                              device: MTLDevice) -> MTLComputePipelineState? {
        let key = Key(device: ObjectIdentifier(device), name: name)
        if let s = states[key] { return s }
        guard let lib = library(for: device) else { return nil }
        guard let fn = lib.makeFunction(name: name) else {
            Self.log.error("Missing Metal function: \(name)")
            return nil
        }
        do {
            let state = try device.makeComputePipelineState(function: fn)
            states[key] = state
            return state
        } catch {
            Self.log.error("makeComputePipelineState(\(name)) failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Concurrently pre-compile a batch of compute pipelines into the cache, then
    /// block until they're all ready — WWDC23 #10127 ("async pipeline
    /// compilation") applied to STARTUP, not just runtime variants. A renderer
    /// that compiles ~26 init pipelines via serial `pipelineState(name:)` calls
    /// blocks one `makeComputePipelineState` after another; this fires them all
    /// through the async API across the driver's compile threads, so a cold
    /// pipeline cache warms in parallel (a `makeComputePipelineState(function:)`
    /// returns a Metal-disk-cached pipeline fast on warm launches, so the win is
    /// the first launch after a shader change). Already-cached names are skipped;
    /// every subsequent `pipelineState(name:)` then hits the warm cache instantly.
    ///
    /// The completion handlers run on background compile threads and write into a
    /// lock-guarded LOCAL box — NOT the MainActor `states` — so blocking the init
    /// thread on the group is not a deadlock; the results land in `states`
    /// afterwards on the calling actor. The `names` list is an optimisation hint:
    /// a name not listed just compiles serially later (correct, slightly slower),
    /// so it never has to be kept perfectly in sync with the call sites.
    public func precompile(_ names: [String], device: MTLDevice) {
        guard let lib = library(for: device) else { return }
        #if !targetEnvironment(macCatalyst)
        if #available(macOS 13.3, *) { device.shouldMaximizeConcurrentCompilation = true }
        #endif
        final class Box: @unchecked Sendable {
            let lock = NSLock()
            var dict: [String: MTLComputePipelineState] = [:]
        }
        let box = Box()
        let group = DispatchGroup()
        for name in names {
            let key = Key(device: ObjectIdentifier(device), name: name)
            if states[key] != nil { continue }
            guard let fn = lib.makeFunction(name: name) else { continue }
            group.enter()
            device.makeComputePipelineState(function: fn) { state, _ in // gpu-ok: async non-blocking startup batch compile (WWDC23 #10127); the group.wait below is one-time init, not per-frame
                // A failed/skipped name simply isn't cached here → it compiles
                // (and logs any real error) on its serial `pipelineState` call.
                if let state { box.lock.lock(); box.dict[name] = state; box.lock.unlock() }
                group.leave()
            }
        }
        group.wait()
        for (name, state) in box.dict {
            states[Key(device: ObjectIdentifier(device), name: name)] = state
        }
    }

    /// Specialized variant of `pipelineState(name:device:)`: builds the kernel
    /// with `[[function_constant]]` values bound, so the Metal compiler can
    /// constant-fold the gated branches and dead-code-eliminate the untaken
    /// paths (WWDC23 #10127, "Optimize GPU renderers with Metal"). Memoised per
    /// `(device, name, variantKey)` — `variantKey` must uniquely encode the
    /// constant combination the caller bound into `constants` (e.g. a bit-string
    /// of the bool flags), so distinct combinations don't collide in the cache
    /// and identical ones hit it. A given combination compiles once; every frame
    /// after is a dictionary lookup.
    ///
    /// Returns nil on a missing function or a build failure (logged); callers
    /// that have a non-specialized fallback should use it on nil.
    public func pipelineState(name: String,
                              device: MTLDevice,
                              constants: MTLFunctionConstantValues,
                              variantKey: String) -> MTLComputePipelineState? {
        let key = Key(device: ObjectIdentifier(device), name: "\(name)#\(variantKey)")
        if let s = states[key] { return s }
        guard let lib = library(for: device) else { return nil }
        let fn: MTLFunction
        do {
            fn = try lib.makeFunction(name: name, constantValues: constants)
        } catch {
            Self.log.error("makeFunction(\(name), constants \(variantKey)) failed: \(error.localizedDescription)")
            return nil
        }
        do {
            // Setup-time, memoised specialized-pipeline construction: one compile per
            // constant combination, then cached — same shape as the base path above.
            let state = try device.makeComputePipelineState(function: fn) // gpu-ok: setup-time memoised specialized pipeline (WWDC23 #10127)
            states[key] = state
            return state
        } catch {
            Self.log.error("makeComputePipelineState(\(name)#\(variantKey)) failed: \(error.localizedDescription)") // gpu-ok: log string, not a call — the real construction on line 137 is the suppressed setup-time one
            return nil
        }
    }

    // ── Async (non-blocking) specialized-pipeline compile ────────────────────

    /// Variants currently compiling in the background, so `pipelineStateAsync`
    /// kicks each `(device, name, variantKey)` exactly once instead of every
    /// frame the flag stays on.
    private var inFlightVariants: Set<Key> = []
    /// Devices that already had `shouldMaximizeConcurrentCompilation` set, so we
    /// only flip it once per device.
    private var maximizedDevices: Set<ObjectIdentifier> = []

    /// Non-blocking sibling of `pipelineState(name:device:constants:variantKey:)`
    /// — WWDC23 #10127 lever 2 (async pipeline compilation).
    ///
    /// On a cache hit returns the ready specialized state. On a miss it kicks off
    /// background compilation **once** and returns nil immediately, so the caller
    /// uses its non-specialized fallback (the uber-shader) instead of blocking
    /// the frame that first needs the variant. When compilation finishes the
    /// state lands in the cache and the next call hits it. This converts the
    /// one-frame hitch that used to happen the first time a renderer feature flag
    /// toggled (a new function-constant combination → a fresh compile) into a few
    /// frames on the slightly-slower uber-shader with no stall.
    ///
    /// The completion handler fires off the main thread; the cache mutation hops
    /// back to the MainActor (`MTLComputePipelineState` is `Sendable`, so it
    /// crosses cleanly). Callers MUST have a fallback for the nil return — see
    /// `IlluminatoramaRenderer.currentLightingPipeline()` for the reference use.
    public func pipelineStateAsync(name: String,
                                   device: MTLDevice,
                                   constants: MTLFunctionConstantValues,
                                   variantKey: String) -> MTLComputePipelineState? {
        let key = Key(device: ObjectIdentifier(device), name: "\(name)#\(variantKey)")
        if let s = states[key] { return s }              // ready → use the variant
        if inFlightVariants.contains(key) { return nil } // compiling → fallback

        guard let lib = library(for: device) else { return nil }
        let fn: MTLFunction
        do {
            fn = try lib.makeFunction(name: name, constantValues: constants)
        } catch {
            Self.log.error("makeFunction(\(name), constants \(variantKey)) failed: \(error.localizedDescription)")
            return nil
        }

        // Hint the driver to parallelise background compiles (macOS 13.3+), once
        // per device.
        let devKey = ObjectIdentifier(device)
        if !maximizedDevices.contains(devKey) {
            #if !targetEnvironment(macCatalyst)
        if #available(macOS 13.3, *) { device.shouldMaximizeConcurrentCompilation = true }
        #endif
            maximizedDevices.insert(devKey)
        }

        inFlightVariants.insert(key)
        device.makeComputePipelineState(function: fn) { [weak self] state, error in // gpu-ok: async, non-blocking setup-time variant compile (WWDC23 #10127 lever 2) — not the per-frame CPU-stall antipattern
            let errMsg = error?.localizedDescription
            Task { @MainActor in
                guard let self else { return }
                self.inFlightVariants.remove(key)
                if let state {
                    self.states[key] = state
                } else {
                    Self.log.error("async makeComputePipelineState(\(name)#\(variantKey)) failed: \(errMsg ?? "unknown")") // gpu-ok: log string, not a call — the real async construction on line 198 is the suppressed setup-time one
                }
            }
        }
        return nil   // not ready this frame — caller falls back to the uber-shader
    }

    // ── PBD pipeline bundle ──────────────────────────────────────────────────

    /// Bundle of the seven compute pipelines the XPBD solver and tube renderer
    /// share, plus three new batch-SDF pipelines. Built lazily on first request
    /// per device, then memoised.
    public struct PBDPipelines {
        public let integrate:        MTLComputePipelineState
        public let constraint:       MTLComputePipelineState
        public let floor:            MTLComputePipelineState
        public let collide:          MTLComputePipelineState
        public let velocity:         MTLComputePipelineState
        public let buildColliders:   MTLComputePipelineState
        public let tubeExpand:       MTLComputePipelineState
        // ── Batch SDF pipelines (issue #45 — one dispatch for all tubes) ──────
        public let copyToFlat:       MTLComputePipelineState
        public let sdfBatch:         MTLComputePipelineState
        public let copyFromFlat:     MTLComputePipelineState
        // ── Particle self-collision (liquid-rope coiling) ────────────────────
        public let selfCollide:      MTLComputePipelineState
        // ── Endless-feed recycle (treadmill respool) ─────────────────────────
        public let recycle:          MTLComputePipelineState
        // ── Sinking-floor conveyor + closed-loop column probe ────────────────
        public let conveyorSink:     MTLComputePipelineState
        public let probeColumn:      MTLComputePipelineState
    }

    private var pbdBundles: [ObjectIdentifier: PBDPipelines] = [:]

    public func pbdPipelines(for device: MTLDevice) -> PBDPipelines? {
        let key = ObjectIdentifier(device)
        if let bundle = pbdBundles[key] { return bundle }
        guard
            let integ       = pipelineState(name: "pbdIntegrate",             device: device),
            let constr      = pipelineState(name: "pbdConstraint",            device: device),
            let flr         = pipelineState(name: "pbdFloorCollide",          device: device),
            let coll        = pipelineState(name: "pbdSDFCollide",            device: device),
            let vel         = pipelineState(name: "pbdVelocitySolve",         device: device),
            let build       = pipelineState(name: "pbdBuildCapsuleColliders", device: device),
            let expand      = pipelineState(name: "pbdTubeExpand",            device: device),
            let copyToFlat  = pipelineState(name: "pbdCopyToFlat",            device: device),
            let sdfBatch    = pipelineState(name: "pbdSDFCollideBatch",       device: device),
            let copyFromFlat = pipelineState(name: "pbdCopyFromFlat",         device: device),
            let selfColl    = pipelineState(name: "pbdSelfCollide",           device: device),
            let recycle     = pipelineState(name: "pbdRecycle",               device: device),
            let conveyor    = pipelineState(name: "pbdConveyorSink",          device: device),
            let probe       = pipelineState(name: "pbdProbeColumn",           device: device)
        else { return nil }
        let bundle = PBDPipelines(
            integrate: integ, constraint: constr, floor: flr, collide: coll,
            velocity: vel, buildColliders: build, tubeExpand: expand,
            copyToFlat: copyToFlat, sdfBatch: sdfBatch, copyFromFlat: copyFromFlat,
            selfCollide: selfColl, recycle: recycle,
            conveyorSink: conveyor, probeColumn: probe
        )
        pbdBundles[key] = bundle
        return bundle
    }
}
