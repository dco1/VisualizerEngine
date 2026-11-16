//  Modifier+GPUOps.swift — encoding of each modifier as Metal dispatches on a resident `GPUMeshState` (item M2).
//
//  Every vertex-moving modifier is the same three steps, all in the caller's one compute encoder: (1) welded
//  normals if the modifier needs them, (2) its own kernel old positions -> new positions, (3) `turn_normals`
//  (`Mesh3.moved`). Solidify instead builds a new mesh (two layers + rim) and a new topology for whatever follows.

#if canImport(Metal)
import Metal
import simd

/// A modifier the GPU can run, in the form the evaluator needs.
enum GPUOperation {
    case noise(NoiseDisplaceModifier), deform(SimpleDeformModifier), lattice(LatticeDeformModifier)
    case shrinkwrap(ShrinkwrapModifier), solidify(SolidifyModifier)

    init?(_ m: any Modifier) {
        if let x = m as? NoiseDisplaceModifier { self = .noise(x) }
        else if let x = m as? SimpleDeformModifier { self = .deform(x) }
        else if let x = m as? LatticeDeformModifier { self = .lattice(x) }
        else if let x = m as? ShrinkwrapModifier { self = .shrinkwrap(x) }
        else if let x = m as? SolidifyModifier { self = .solidify(x) }
        else { return nil }          // incl. closure-based DisplaceModifier, Mirror, Array, Bevel, Subdivide, Boolean
    }

    var modifier: any Modifier {
        switch self {
        case .noise(let m): return m
        case .deform(let m): return m
        case .lattice(let m): return m
        case .shrinkwrap(let m): return m
        case .solidify(let m): return m
        }
    }
    var name: String { String(describing: type(of: modifier)) }
}

private struct WPar { var nW: UInt32; var even: UInt32; var pad0: UInt32 = 0, pad1: UInt32 = 0 }
private struct TPar { var n: UInt32; var pad0: UInt32 = 0, pad1: UInt32 = 0, pad2: UInt32 = 0 }
private struct NPar { var fixedDir: SIMD4<Float>; var n, dirMode, octaves, seedLo, seedHi: UInt32
                      var strength, mid, freq, lac, gain, pad0, pad1: Float }
private struct SDPar { var origin, w, u, bv, w2: SIMD4<Float>; var mode, n: UInt32
                       var angle, lower, upper, factor, radius, pad: Float }
private struct LPar { var boxMin, size: SIMD4<Float>; var dx, dy, dz, n, cubic, pad0, pad1, pad2: UInt32 }
private struct SPar { var nt, ne, hasN, hasUV: UInt32; var aOff, bOff, pad0, pad1: Float }
private struct SWPar { var fixedDir: SIMD4<Float>; var n, mode, tryPos, tryNeg: UInt32; var offset, factor, pad0, pad1: Float }

private func f4(_ v: Vec3, _ w: Double = 0) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), Float(w)) }

extension ModifierGPU {
    func encode(_ op: GPUOperation, _ s: GPUMeshState, _ enc: MTLComputeCommandEncoder) -> GPUMeshState {
        switch op {
        case .noise(let m): return moveVertices(s, enc, needsNormals: true) { newP, wN in
            let sd = m.seed
            var fixed = SIMD4<Float>.zero
            if case .fixed(let v) = m.direction { fixed = f4(normalize3(v)) }
            let mode: UInt32 = { if case .normal = m.direction { return 0 } else { return 1 } }()
            let p = NPar(fixedDir: fixed, n: UInt32(s.n), dirMode: mode, octaves: UInt32(Swift.max(1, m.octaves)),
                         seedLo: UInt32(truncatingIfNeeded: sd), seedHi: UInt32(truncatingIfNeeded: sd >> 32),
                         strength: Float(m.strength), mid: Float(m.midlevel), freq: Float(m.frequency),
                         lac: Float(m.lacunarity), gain: Float(m.gain), pad0: 0, pad1: 0)
            dispatch(enc, "displace_noise", s.n, [s.pos, newP, s.topo.buf("cid", self), s.topo.buf("rep", self), wN!], p)
        }
        case .deform(let m): return moveVertices(s, enc, needsNormals: false) { newP, _ in
            let fr = frame(for: m.axis)
            var v = m.bendDirection.map { $0 - fr.w * dot3($0, fr.w) } ?? fr.u
            v = len3(v) > 1e-9 ? normalize3(v) : fr.u
            var mode: UInt32 = 0; var angle = 0.0, factor = 0.0
            switch m.mode {
            case .bend(let a): mode = 0; angle = a
            case .twist(let a): mode = 1; angle = a
            case .taper(let f): mode = 2; factor = f
            case .stretch(let f): mode = 3; factor = f
            }
            let radius = angle != 0 ? (m.upper - m.lower) / angle : 1
            let p = SDPar(origin: f4(m.origin), w: f4(fr.w), u: f4(fr.u), bv: f4(v), w2: f4(cross3(fr.w, v)), mode: mode, n: UInt32(s.n),
                          angle: Float(angle), lower: Float(m.lower), upper: Float(m.upper), factor: Float(factor), radius: Float(radius), pad: 0)
            dispatch(enc, "simple_deform", s.n, [s.pos, newP], p)
        }
        case .lattice(let m): return moveVertices(s, enc, needsNormals: false) { newP, _ in
            let size = Vec3(Swift.max(m.boxMax.x - m.boxMin.x, 1e-12), Swift.max(m.boxMax.y - m.boxMin.y, 1e-12), Swift.max(m.boxMax.z - m.boxMin.z, 1e-12))
            let pts = buffer(m.points.map { f4($0) })
            let p = LPar(boxMin: f4(m.boxMin), size: f4(size), dx: UInt32(m.divisions.x), dy: UInt32(m.divisions.y), dz: UInt32(m.divisions.z),
                         n: UInt32(s.n), cubic: m.interpolation == .cubic ? 1 : 0, pad0: 0, pad1: 0, pad2: 0)
            dispatch(enc, "lattice", s.n, [s.pos, newP, pts], p)
        }
        case .shrinkwrap(let m):
            guard !m.target.isEmpty else { return s }
            return moveVertices(s, enc, needsNormals: true) { newP, wN in
                let bvh = FlatBVH(m.target)
                var mode: UInt32 = 0, tp: UInt32 = 1, tn: UInt32 = 0
                var fixed = SIMD4<Float>.zero
                switch m.method {
                case .nearestSurface: mode = 0
                case .projectAlongNormal(let sg): mode = 1; tp = sg == .negative ? 0 : 1; tn = sg == .positive ? 0 : 1
                case .projectAlong(let d): mode = 2; tp = 1; tn = 1; fixed = f4(d)
                }
                let p = SWPar(fixedDir: fixed, n: UInt32(s.n), mode: mode, tryPos: tp, tryNeg: tn,
                              offset: Float(m.offset), factor: Float(m.factor), pad0: 0, pad1: 0)
                dispatch(enc, "shrinkwrap", s.n, [s.pos, newP, s.topo.buf("cid", self), s.topo.buf("rep", self), wN!,
                                                  buffer(bvh.nodes), buffer(bvh.a), buffer(bvh.b), buffer(bvh.c), buffer(bvh.n)], p)
            }
        case .solidify(let m): return solidify(m, s, enc)
        }
    }

    /// Welded normals: one thread per welded vertex.
    func weldedNormals(_ s: GPUMeshState, _ enc: MTLComputeCommandEncoder) -> MTLBuffer {
        let t = s.topo
        let wN = empty(t.nW * 16)
        dispatch(enc, "weld_normals", t.nW, [s.pos, t.buf("rep", self), t.buf("wStart", self), t.buf("wEnt", self), t.buf("wtris", self), wN],
                 WPar(nW: UInt32(t.nW), even: 0))
        return wN
    }

    private func moveVertices(_ s: GPUMeshState, _ enc: MTLComputeCommandEncoder, needsNormals: Bool,
                              _ body: (MTLBuffer, MTLBuffer?) -> Void) -> GPUMeshState {
        let newP = empty(s.n * 16)
        body(newP, needsNormals ? weldedNormals(s, enc) : nil)
        var newN = s.nrm
        if s.hasN {
            newN = empty(s.n * 16)
            let t = s.topo
            dispatch(enc, "turn_normals", s.n, [s.pos, newP, s.nrm, newN, s.idx, t.buf("vStart", self), t.buf("vTris", self)], TPar(n: UInt32(s.n)))
        }
        return GPUMeshState(gpu: s.gpu, n: s.n, pos: newP, nrm: newN, uv: s.uv, idx: s.idx, hasN: s.hasN, hasUV: s.hasUV,
                            topo: { s.topo }, keepNormals: s.keepNormals, keepUVs: s.keepUVs, indices: s.indicesCPU)
    }

    private func solidify(_ m: SolidifyModifier, _ s: GPUMeshState, _ enc: MTLComputeCommandEncoder) -> GPUMeshState {
        guard m.thickness != 0 else { return s }
        let t = s.topo
        let wN = weldedNormals(s, enc)
        let scale = empty(t.nW * 4)
        dispatch(enc, "solid_scale", t.nW, [s.pos, t.buf("rep", self), t.buf("wStart", self), t.buf("wEnt", self), t.buf("wtris", self), wN, scale],
                 WPar(nW: UInt32(t.nW), even: m.evenThickness ? 1 : 0))
        let nt = t.wtriSrc.count, ne = m.rim ? t.rimEdges.count / 2 : 0
        let M = 6 * (nt + ne)
        let oP = empty(M * 16), oN = empty(M * 16), oU = empty(M * 8)
        let par = SPar(nt: UInt32(nt), ne: UInt32(ne), hasN: s.hasN ? 1 : 0, hasUV: s.hasUV ? 1 : 0,
                       aOff: Float(m.thickness * (1 + m.offset) / 2), bOff: Float(m.thickness * (m.offset - 1) / 2), pad0: 0, pad1: 0)
        dispatch(enc, "solid_tris", nt, [s.pos, s.nrm, s.uv, s.idx, t.buf("cid", self), t.buf("rep", self), wN, scale, t.buf("wsrc", self), oP, oN, oU], par)
        if ne > 0 { dispatch(enc, "solid_rim", ne, [s.pos, t.buf("rep", self), wN, scale, t.buf("rim", self), oP, oN, oU], par) }

        // The new mesh's topology, built only if something follows: A layer keeps the welded ids, B layer is offset by
        // nW; rim corners are A/B of its edge.
        let iota = (0..<UInt32(M)).map { $0 }
        let makeTopo: () -> GPUTopology = {
            var cid = [UInt32](); cid.reserveCapacity(M)
            let nW = UInt32(t.nW)
            for j in 0..<nt {
                let src = Int(t.wtriSrc[j])
                let c = (0..<3).map { t.cornerID[Int(t.indices[3 * src + $0])] }
                cid += [c[0], c[1], c[2], nW + c[0], nW + c[2], nW + c[1]]
            }
            for e in 0..<ne {
                let a = t.rimEdges[2 * e], b = t.rimEdges[2 * e + 1]
                cid += [a, nW + a, nW + b, a, nW + b, b]
            }
            return GPUTopology(cornerID: cid, nW: 2 * t.nW, indices: iota)
        }
        return GPUMeshState(gpu: s.gpu, n: M, pos: oP, nrm: oN, uv: oU, idx: buffer(iota), hasN: true, hasUV: true,
                            topo: makeTopo, keepNormals: nil, keepUVs: nil, indices: iota)
    }
}
#endif
