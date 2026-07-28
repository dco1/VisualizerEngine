import Foundation
import simd

// Reader for HumanDataset.bin (format "MHB1"), baked by tools/humans/import_makehuman.py
// from the CC0 MakeHuman data. Keep the layout in sync with that script.
//
// Geometry model: `origPositions` is the full MakeHuman vertex table (body +
// helper + joint-cube vertices) — morphs, skin weights, and skeleton joints all
// index into it. Render vertices (deduplicated position/UV pairs of the "body"
// group only) reference orig vertices through `renderOrigIndex`.
public struct HumanDataset: Sendable {

    public struct Morph: Sendable {
        public let name: String
        public let indices: [UInt32]
        public let deltas: [SIMD3<Float>]
    }

    public struct Joint: Sendable {
        public let name: String
        public let vertices: [UInt32]
    }

    public struct Bone: Sendable {
        public let name: String
        public let parent: Int          // -1 = root
        public let headJoint: Int
        public let tailJoint: Int
        public let planeJoints: SIMD3<Int32>   // -1s when absent
        public let weightIndices: [UInt32]     // orig vertex indices
        public let weightValues: [Float]
    }

    public let origPositions: [SIMD3<Float>]
    public let renderOrigIndex: [UInt32]
    public let renderUV: [SIMD2<Float>]
    public let triangles: [UInt32]
    public let morphs: [Morph]
    public let joints: [Joint]
    public let bones: [Bone]
    public let morphIndex: [String: Int]
    public let boneIndex: [String: Int]

    public enum LoadError: Error { case missingResource, badMagic, truncated }

    // The dataset is ~50 MB; load once and share.
    public static let shared: HumanDataset = {
        do {
            return try HumanDataset()
        } catch {
            fatalError("HumanDataset.bin failed to load: \(error)")
        }
    }()

    public init() throws {
        guard let url = Bundle.module.url(forResource: "HumanDataset", withExtension: "bin") else {
            throw LoadError.missingResource
        }
        try self.init(data: try Data(contentsOf: url))
    }

    public init(data: Data) throws {
        var cursor = 0

        func read<T>(_ type: T.Type) throws -> T {
            let size = MemoryLayout<T>.size
            guard cursor + size <= data.count else { throw LoadError.truncated }
            defer { cursor += size }
            return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: T.self) }
        }

        func readArray<T>(_ type: T.Type, count: Int) throws -> [T] {
            let size = MemoryLayout<T>.stride * count
            guard cursor + size <= data.count else { throw LoadError.truncated }
            defer { cursor += size }
            return data.withUnsafeBytes { raw in
                (0..<count).map { raw.loadUnaligned(fromByteOffset: cursor + $0 * MemoryLayout<T>.stride, as: T.self) }
            }
        }

        func readString() throws -> String {
            let len = Int(try read(UInt16.self))
            guard cursor + len <= data.count else { throw LoadError.truncated }
            defer { cursor += len }
            return String(decoding: data[data.startIndex + cursor ..< data.startIndex + cursor + len], as: UTF8.self)
        }

        func readVec3() throws -> SIMD3<Float> {
            SIMD3(try read(Float.self), try read(Float.self), try read(Float.self))
        }

        guard try read(UInt32.self) == 0x3142_484D else { throw LoadError.badMagic }   // "MHB1" LE
        _ = try read(UInt32.self)   // version
        let origCount = Int(try read(UInt32.self))
        let renderCount = Int(try read(UInt32.self))
        let triCount = Int(try read(UInt32.self))
        let morphCount = Int(try read(UInt32.self))
        let jointCount = Int(try read(UInt32.self))
        let boneCount = Int(try read(UInt32.self))

        var orig: [SIMD3<Float>] = []
        orig.reserveCapacity(origCount)
        for _ in 0..<origCount { orig.append(try readVec3()) }
        origPositions = orig

        renderOrigIndex = try readArray(UInt32.self, count: renderCount)
        let uvFlat = try readArray(Float.self, count: renderCount * 2)
        renderUV = (0..<renderCount).map { SIMD2(uvFlat[$0 * 2], uvFlat[$0 * 2 + 1]) }
        triangles = try readArray(UInt32.self, count: triCount)

        var ms: [Morph] = []
        ms.reserveCapacity(morphCount)
        for _ in 0..<morphCount {
            let name = try readString()
            let n = Int(try read(UInt32.self))
            var idx: [UInt32] = []
            var deltas: [SIMD3<Float>] = []
            idx.reserveCapacity(n)
            deltas.reserveCapacity(n)
            for _ in 0..<n {
                idx.append(try read(UInt32.self))
                let x = try read(Float16.self)
                let y = try read(Float16.self)
                let z = try read(Float16.self)
                deltas.append(SIMD3(Float(x), Float(y), Float(z)))
            }
            ms.append(Morph(name: name, indices: idx, deltas: deltas))
        }
        morphs = ms

        var js: [Joint] = []
        js.reserveCapacity(jointCount)
        for _ in 0..<jointCount {
            let name = try readString()
            let n = Int(try read(UInt32.self))
            js.append(Joint(name: name, vertices: try readArray(UInt32.self, count: n)))
        }
        joints = js

        var bs: [Bone] = []
        bs.reserveCapacity(boneCount)
        for _ in 0..<boneCount {
            let name = try readString()
            let parent = Int(try read(Int32.self))
            let head = Int(try read(UInt32.self))
            let tail = Int(try read(UInt32.self))
            let plane = SIMD3<Int32>(try read(Int32.self), try read(Int32.self), try read(Int32.self))
            let n = Int(try read(UInt32.self))
            var wi: [UInt32] = []
            var wv: [Float] = []
            wi.reserveCapacity(n)
            wv.reserveCapacity(n)
            for _ in 0..<n {
                wi.append(try read(UInt32.self))
                wv.append(try read(Float.self))
            }
            bs.append(Bone(name: name, parent: parent, headJoint: head, tailJoint: tail,
                           planeJoints: plane, weightIndices: wi, weightValues: wv))
        }
        bones = bs

        morphIndex = Dictionary(uniqueKeysWithValues: morphs.enumerated().map { ($0.element.name, $0.offset) })
        boneIndex = Dictionary(uniqueKeysWithValues: bones.enumerated().map { ($0.element.name, $0.offset) })
    }
}
