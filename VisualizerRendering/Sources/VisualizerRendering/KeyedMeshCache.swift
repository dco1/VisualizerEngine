/// A keyed cache of baked GPU meshes that **prunes what is no longer on screen, on every miss**.
///
/// A host that bakes geometry per element — a tree, a hedge, a planter, a deck — caches the result
/// by a key derived from that element's parameters, so identical elements share one mesh and moving
/// one is a transform change rather than a re-bake. The part that is easy to write and easy to
/// forget is the other half: when a parameter CHANGES, the old key is never asked for again, and
/// its registered mesh stays alive for as long as the host does.
///
/// That is not a slow leak. Keys are usually built from raw, unquantized parameter values, so
/// **dragging one inspector slider mints a fresh key on every tick** and pins a full set of
/// registered meshes per intermediate value. Daydream Home shipped this twice: once on trees
/// (fixed 2026-07-05, multi-MB per slider tick), and once on potted plants, which was the only one
/// of six caches whose author did not copy the eviction block along with the lookup (DH-0659).
///
/// Six hand-written copies of a three-line idiom, one of them silently missing, is the argument for
/// this type. There is no way to read through it without the prune happening: `value(forKey:live:build:)`
/// is the only accessor, and it prunes before it stores.
///
/// ## The live set
///
/// `live` is a closure, not a value, because it is only needed on a miss — computing it walks every
/// element on every floor, which is pure waste on the hot path where the key is already cached. It
/// must return the keys of every element **currently rendered**, which for an editor means the
/// in-progress edit state *unioned with* the committed document: an element being dragged is in one
/// and not the other, and pruning against either alone thrashes a re-bake every frame.
///
/// The key being fetched is always retained, whether or not `live` names it — a bake in progress is
/// by definition about to be used.
public struct KeyedMeshCache<Value> {

    private var storage: [String: Value] = [:]

    public init() {}

    /// Direct access, for TESTS AND DIAGNOSTICS. Does not prune and does not build.
    ///
    /// The setter exists because a few GPU test harnesses swap a mesh in, render, and swap it back
    /// to compare two bakes of the same element (`+FidelityContactSheet`), and because one cache is
    /// a namespace shared by four subsystems. **Production code must never use it** — storing
    /// without pruning is the whole defect this type exists to prevent, so
    /// `MeshCacheEvictionCensusTests` fails the build on any `…Handles[key] = …` under
    /// `DaydreamHome/Render/`. Fetch through `value(forKey:live:build:)` instead.
    public subscript(key: String) -> Value? {
        get { storage[key] }
        set { storage[key] = newValue }
    }

    /// Drop one entry. Same audience and same rule as the subscript setter.
    @discardableResult
    public mutating func removeValue(forKey key: String) -> Value? {
        storage.removeValue(forKey: key)
    }

    /// Whether the cache holds nothing. Mirrors `Dictionary.isEmpty` — `Sequence` does not supply
    /// one, and the diagnostic call sites that read the raw dictionary expect it.
    public var isEmpty: Bool { storage.isEmpty }

    /// How many entries are held. Test-observable: a cache that grows across a slider drag or a
    /// camera orbit is the defect this type exists to prevent, and that is asserted as a number.
    public var count: Int { storage.count }

    /// Every key currently held, for diagnostics.
    public var keys: some Collection<String> { storage.keys }

    /// Every value currently held — the mirror of `keys`, for the census tests that walk every
    /// registered mesh in a cache (RT instance counts, single-sidedness) without caring which key
    /// produced it. Read-only: mutating a stored mesh has to go through `value(forKey:live:build:)`
    /// so the prune cannot be skipped.
    public var values: some Collection<Value> { storage.values }

    /// Fetch the mesh for `key`, building it if absent.
    ///
    /// On a HIT this is a dictionary lookup and nothing else. On a MISS it evicts every entry whose
    /// key is not in `live()` (retaining `key` itself), then builds and stores. A `build` returning
    /// nil — a degenerate element with no geometry — is not cached, so a later fix to that element's
    /// parameters is not shadowed by a stored nil.
    public mutating func value(forKey key: String,
                               live: () -> Set<String>,
                               build: () -> Value?) -> Value? {
        if let hit = storage[key] { return hit }
        let liveKeys = live()
        storage = storage.filter { liveKeys.contains($0.key) || $0.key == key }
        guard let made = build() else { return nil }
        storage[key] = made
        return made
    }

    /// Fetch for a cache whose key space is FINITE BY CONSTRUCTION — nothing is ever evicted.
    ///
    /// Not every cache has a live set worth computing. A socket face keyed by
    /// `"\(plugStandard)|\(part)"` can only ever hold a handful of entries no matter what the user
    /// does, so pruning it would be ceremony: there is nothing to prune. Saying that here, in a
    /// named method, is honest — the alternative was to hand `live` a set containing everything,
    /// which reads like an oversight rather than a decision.
    ///
    /// Use this ONLY when the key is built entirely from enum cases or small integers. The moment a
    /// key folds in a slider value, the key space is unbounded and this is the wrong method.
    public mutating func valueRetainingAll(forKey key: String,
                                           build: () -> Value?) -> Value? {
        if let hit = storage[key] { return hit }
        guard let made = build() else { return nil }
        storage[key] = made
        return made
    }

    /// Fetch for a cache keyed by TRANSIENT state whose live set the caller cannot name, bounded
    /// instead by capacity.
    ///
    /// A cord is keyed by its quantized endpoints, which change every frame of a drag; nothing
    /// upstream knows which endpoint keys are "current" without recomputing the routing. So the
    /// bound is a cap: past `capacity` the cache is emptied wholesale and the visible few
    /// re-register on the next frame. Cruder than pruning, correct for this shape, and — crucially —
    /// still a bound, which is what the raw dictionary never had.
    ///
    /// Prefer `value(forKey:live:build:)` whenever the live set IS derivable; a cap re-bakes work
    /// that pruning would have kept.
    public mutating func value(forKey key: String,
                               capacity: Int,
                               build: () -> Value?) -> Value? {
        if let hit = storage[key] { return hit }
        if storage.count > capacity { storage.removeAll() }
        guard let made = build() else { return nil }
        storage[key] = made
        return made
    }

    /// Drop everything. For a document swap or a teardown, where no key survives.
    public mutating func removeAll() { storage.removeAll() }
}

/// Iterating a cache yields its `(key, value)` pairs, so the census and diagnostic call sites that
/// walked the raw dictionary — `for (key, handle) in cache`, `cache.compactMap { (k, v) in … }`,
/// `cache.contains { $0.key.hasPrefix(…) }` — read exactly as they did before.
extension KeyedMeshCache: Sequence {
    public func makeIterator() -> Dictionary<String, Value>.Iterator { storage.makeIterator() }
}
