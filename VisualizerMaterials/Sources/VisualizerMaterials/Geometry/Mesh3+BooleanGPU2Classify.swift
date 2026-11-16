//  The GPU-heavy boolean pipeline, classification and assembly half (item M6b).
//
//  Classification: the faces of each mesh fall into patches (faces joined across edges that are not part of the
//  intersection curve). On the GPU: a vertex→face CSR, connected components by hooking and pointer jumping,
//  each patch's largest face (atomic max, then lowest index among the ties), and the winding number of one
//  representative point per patch against the other mesh — ALL of the patches' points in one batch, one
//  threadgroup per point, so it fires for every input the pipeline takes. Coplanar patches are found by a grid
//  query and confirmed in double on the CPU, as in `BooleanJob.classify`. A winding number near the 0.5
//  decision is recomputed in double on the CPU.
//
//  Assembly: keep flags, stream compaction of the kept faces, vertex remap (flags + scan), and the barycentric
//  normals / UVs per corner with fixed-point atomic sums (deterministic), all on the GPU. Positions are the
//  pool's doubles, gathered on the CPU. `weld: .keepCreases` assembles on the CPU from the GPU's labels.

import Foundation
import simd

#if canImport(Metal)
import Metal

extension GPU2Run {
    // MARK: classify

    /// Per side: the components' parent array and the relation of each patch, left on the GPU for assembly.
    func classify() -> (parent: [MTLBuffer], rel: [MTLBuffer], relCPU: [[UInt32]])? {
        guard let pb = g.buf(floats(job.pool.points)) else { return nil }
        posBuf = pb
        var parents: [MTLBuffer] = [], rels: [MTLBuffer] = [], relCPU: [[UInt32]] = []
        for s in 0..<2 {
            guard let (par, roots) = patches(s) else { return nil }
            lap("g.patches\(s)")
            guard let rel = relations(s, roots) else { return nil }
            guard let rb = g.buf(rel) else { return nil }
            parents.append(par); rels.append(rb); relCPU.append(rel)
            lap("g.relations\(s)")
        }
        return (parents, rels, relCPU)
    }

    private func patches(_ s: Int) -> (MTLBuffer, [SIMD2<UInt32>])? {
        let faces = job.faces[s]
        let nF = faces.count, nV = job.pool.points.count
        guard let fb = g.buf(faces), let counts = g.zeros((nV + 1) * 4), let list = g.zeros(nF * 12),
              let cursor = g.zeros(nV * 4), let parent = g.zeros(nF * 4) else { return nil }
        lap("g.patchPrep")
        let cutKeys = job.cut.sorted().map { SIMD2<UInt32>(UInt32($0 >> 32), UInt32($0 & 0xFFFF_FFFF)) }
        guard let keyBuf = g.buf(cutKeys) else { return nil }
        guard g.run({ enc in
            g.go(enc, "csr_count", nF, [.buf(fb), .buf(counts), .val(UInt32(nF))])
            g.scan(enc, counts, nV + 1)
        }) else { return nil }
        guard g.run({ enc in
            g.go(enc, "csr_fill", nF, [.buf(fb), .buf(counts), .buf(cursor), .buf(list), .val(UInt32(nF))])
            g.go(enc, "cc_init", nF, [.buf(parent), .val(UInt32(nF))])
        }) else { return nil }
        lap("g.csr")
        // components: batches of rounds until the last round changes nothing
        var rounds = 0
        while true {
            var flags: [MTLBuffer] = []
            for _ in 0..<8 { guard let b = g.zeros(16) else { return nil }; flags.append(b) }
            guard g.run({ enc in
                for r in 0..<8 {
                    g.go(enc, "cc_hook", nF, [.buf(fb), .buf(counts), .buf(list), .buf(keyBuf),
                                               .val(SIMD2<UInt32>(UInt32(nF), UInt32(cutKeys.count))), .buf(parent), .buf(flags[r])])
                    for _ in 0..<3 { g.go(enc, "cc_jump", nF, [.buf(parent), .buf(flags[r]), .val(UInt32(nF))]) }
                }
            }) else { return nil }
            rounds += 8
            if g.u32(flags[7], 0) == 0 || rounds > 4000 { break }
        }
        // a final full compression
        while true {
            guard let flag = g.zeros(16) else { return nil }
            guard g.run({ enc in for _ in 0..<4 { g.go(enc, "cc_jump", nF, [.buf(parent), .buf(flag), .val(UInt32(nF))]) } }) else { return nil }
            if g.u32(flag, 0) == 0 { break }
        }
        lap("g.ccLoop")
        job.stats.seconds["g.ccRounds"] = Double(rounds)
        guard let maxA = g.zeros(nF * 4), let win = g.zeros(nF * 4), let rflags = g.zeros((nF + 1) * 4), let posB = Optional(posBuf!) else { return nil }
        g.fill(win, UInt32.max)
        guard g.run({ enc in
            g.go(enc, "area_max", nF, [.buf(fb), .buf(posB), .buf(parent), .buf(maxA), .val(UInt32(nF))])
            g.go(enc, "area_win", nF, [.buf(fb), .buf(posB), .buf(parent), .buf(maxA), .buf(win), .val(UInt32(nF))])
            g.go(enc, "root_flags", nF, [.buf(parent), .buf(rflags), .val(UInt32(nF))])
            g.scan(enc, rflags, nF + 1)
        }) else { return nil }
        let nR = g.u32(rflags, nF)
        guard let roots = g.zeros(nR * 8) else { return nil }
        guard g.run({ enc in
            g.go(enc, "compact_roots", nF, [.buf(parent), .buf(rflags), .buf(win), .buf(roots), .val(UInt32(nF))])
        }) else { return nil }
        job.stats.patches += nR
        return (parent, g.read(roots, SIMD2<UInt32>.self, nR))
    }

    /// 0 inside, 1 outside, 2 same, 3 opposite, at each root face's index (other entries 0).
    private func relations(_ s: Int, _ roots: [SIMD2<UInt32>]) -> [UInt32]? {
        let o = 1 - s
        let P = job.pool.points
        let fs = job.faces[s]
        var rel = [UInt32](repeating: 0, count: fs.count)
        guard !roots.isEmpty else { return rel }
        // representative points
        var cents: [Vec3] = [], normals: [Vec3] = []
        for r in roots {
            let f = fs[Int(r.y)]
            cents.append((P[Int(f.x)] + P[Int(f.y)] + P[Int(f.z)]) / 3)
            normals.append(cross3(P[Int(f.y)] - P[Int(f.x)], P[Int(f.z)] - P[Int(f.x)]))
        }
        // coplanar candidates: a point-box query against the other mesh's grid
        let m = margin
        let f4 = floats(cents)
        let lo = f4.map { SIMD4<Float>($0.x - m, $0.y - m, $0.z - m, 0) }, hi = f4.map { SIMD4<Float>($0.x + m, $0.y + m, $0.z + m, 0) }
        guard let lb = g.buf(lo), let hb = g.buf(hi),
              let q = g.query(grid: grids[o], loQ: lb, hiQ: hb, nQ: cents.count, mode: 0, margin: 0, tris: triBuf[o], pos: posBuf)
        else { return nil }
        let cand = g.read(q.pairs, SIMD2<UInt32>.self, q.count)
        let to = job.tris[o]
        let eps = job.eps
        var cop = [UInt32?](repeating: nil, count: cents.count)
        for pr in cand {
            let k = Int(pr.x)
            if cop[k] != nil { continue }
            let c = cents[k], nF = normals[k]
            let v = to[Int(pr.y)]
            let a = P[Int(v.x)], b = P[Int(v.y)], cc = P[Int(v.z)]
            let nS = cross3(b - a, cc - a)
            let nu = normalize3(nS)
            if abs(dot3(nu, c - a)) > eps { continue }
            let pts = [a, b, cc]
            var inside = true
            for e in 0..<3 {
                let mm = normalize3(cross3(nu, pts[(e + 1) % 3] - pts[e]))
                if dot3(mm, c - pts[e]) < -eps { inside = false; break }
            }
            if inside { cop[k] = dot3(nS, nF) > 0 ? 2 : 3 }
        }
        lap("g.coplanar")
        // winding numbers, one batch
        var wIdx: [Int] = []
        for k in 0..<cents.count where cop[k] == nil { wIdx.append(k) }
        var w = [Double](repeating: 0, count: wIdx.count)
        if !wIdx.isEmpty {
            let pts = wIdx.map { f4[$0] }
            guard let pb = g.buf(pts), let ob = g.zeros(pts.count * 4) else { return nil }
            guard g.run({ enc in
                g.goGroups(enc, "winding_tg", groups: pts.count, tg: 256,
                           [.buf(pb), .buf(triBuf[o]), .buf(posBuf), .buf(ob), .val(UInt32(to.count))])
            }) else { return nil }
            let gw = g.read(ob, Float.self, pts.count)
            var redo: [Int] = []
            for (i, v) in gw.enumerated() {
                w[i] = Double(v)
                if abs(abs(Double(v)) - 0.5) < 0.2 { redo.append(i) }
            }
            if !redo.isEmpty {
                var corners: [Vec3] = []
                corners.reserveCapacity(to.count * 3)
                for v in to { corners.append(P[Int(v.x)]); corners.append(P[Int(v.y)]); corners.append(P[Int(v.z)]) }
                let exact = BooleanJob.windingCPU(redo.map { cents[wIdx[$0]] }, corners)
                for (j, i) in redo.enumerated() { w[i] = exact[j] }
                job.stats.seconds["g.windingRedoneOnCPU", default: 0] += Double(redo.count)
            }
            job.stats.windingOnGPU = true
        }
        var wi = 0
        for (k, r) in roots.enumerated() {
            if let c = cop[k] { rel[Int(r.x)] = c } else { rel[Int(r.x)] = abs(w[wi]) > 0.5 ? 0 : 1; wi += 1 }
        }
        lap("g.winding")
        return rel
    }

    // MARK: assemble

    func assemble(_ op: BooleanOp, parent: [MTLBuffer], rel: [MTLBuffer], relCPU: [[UInt32]]) -> Mesh3? {
        let P = job.pool.points
        let nA = job.faces[0].count, nB = job.faces[1].count, nF = nA + nB
        let opCode: UInt32 = op == .union ? 0 : (op == .difference ? 1 : 2)
        guard nF > 0 else { return Mesh3() }
        guard let fb = g.buf(job.faces[0] + job.faces[1]), let flags = g.zeros((nF + 1) * 4) else { return nil }
        struct AP { var nFaces: UInt32; var nFacesA: UInt32; var op: UInt32; var nPool: UInt32; var nVertA: UInt32; var idxOffB: UInt32; var hasN: UInt32; var hasUV: UInt32 }
        let mA = job.meshes[0], mB = job.meshes[1]
        let hasN = [mA, mB].enumerated().reduce(UInt32(0)) { $0 | (($1.element.normals.count == $1.element.positions.count ? 1 : 0) << UInt32($1.offset)) }
        let hasUV = [mA, mB].enumerated().reduce(UInt32(0)) { $0 | (($1.element.uvs.count == $1.element.positions.count ? 1 : 0) << UInt32($1.offset)) }
        let ap = AP(nFaces: UInt32(nF), nFacesA: UInt32(nA), op: opCode, nPool: UInt32(P.count), nVertA: UInt32(mA.positions.count),
                    idxOffB: UInt32(mA.indices.count), hasN: hasN, hasUV: hasUV)
        guard g.run({ enc in
            g.go(enc, "asm_flags", nF, [.buf(fb), .buf(parent[0]), .buf(rel[0]), .val(ap), .buf(flags), .buf(parent[1]), .buf(rel[1])])
            g.scan(enc, flags, nF + 1)
        }) else { return nil }
        let nKept = g.u32(flags, nF)
        if nKept == 0 { return Mesh3() }
        guard let kept = g.zeros(nKept * 16), let keptFace = g.zeros(nKept * 4), let used = g.zeros((P.count + 1) * 4) else { return nil }
        guard g.run({ enc in
            g.go(enc, "asm_compact", nF, [.buf(fb), .buf(parent[0]), .buf(rel[0]), .val(ap), .buf(flags), .buf(flags),
                                           .buf(kept), .buf(keptFace), .buf(used), .buf(parent[1]), .buf(rel[1])])
        }) else { return nil }
        lap("g.asmCompact")
        // vertex remap: scan of the used flags
        guard let usedFlags = g.zeros((P.count + 1) * 4) else { return nil }
        memcpy(usedFlags.contents(), used.contents(), P.count * 4)
        guard g.run({ enc in g.scan(enc, usedFlags, P.count + 1) }) else { return nil }
        let nUsed = g.u32(usedFlags, P.count)
        guard let usedList = g.zeros(nUsed * 4) else { return nil }
        // pool and source tables, double-single
        func hiLo(_ v: [Vec3]) -> ([SIMD4<Float>], [SIMD4<Float>]) {
            (pmap(v) { SIMD4(Float($0.x), Float($0.y), Float($0.z), 0) },
             pmap(v) { p in SIMD4(Float(p.x - Double(Float(p.x))), Float(p.y - Double(Float(p.y))), Float(p.z - Double(Float(p.z))), 0) })
        }
        let (ph, pl) = hiLo(P)
        let (sh, sl) = hiLo(mA.positions + mB.positions)
        var sn = [SIMD4<Float>](), suv = [SIMD2<Float>]()
        for m in [mA, mB] {
            if m.normals.count == m.positions.count { sn += pmap(m.normals) { SIMD4(Float($0.x), Float($0.y), Float($0.z), 0) } }
            else { sn += [SIMD4<Float>](repeating: .zero, count: m.positions.count) }
            if m.uvs.count == m.positions.count { suv += pmap(m.uvs) { SIMD2(Float($0.x), Float($0.y)) } }
            else { suv += [SIMD2<Float>](repeating: .zero, count: m.positions.count) }
        }
        let srcIdx = mA.indices + mB.indices
        let faceSrc = job.faceSrc[0] + job.faceSrc[1]
        guard let phB = g.buf(ph), let plB = g.buf(pl), let shB = g.buf(sh), let slB = g.buf(sl), let snB = g.buf(sn),
              let suvB = g.buf(suv), let siB = g.buf(srcIdx), let fsB = g.buf(faceSrc),
              let nsum = g.zeros(nUsed * 16), let first = g.zeros(nUsed * 4), let cornerUV = g.zeros(nKept * 24),
              let outIdx = g.zeros(nKept * 12), let outN = g.zeros(nUsed * 16), let outUV = g.zeros(nUsed * 8) else { return nil }
        g.fill(first, UInt32.max)
        lap("g.asmPrep")
        // newIndex[pid] = exclusive prefix of the used flags
        guard g.run({ enc in
            g.go(enc, "asm_used_list", P.count, [.buf(used), .buf(usedFlags), .buf(usedList), .val(UInt32(P.count))])
            g.go(enc, "asm_corners", nKept, [.buf(kept), .buf(keptFace), .buf(fsB), .val(ap), .buf(siB), .buf(shB), .buf(slB),
                                              .buf(snB), .buf(suvB), .buf(phB), .buf(plB), .buf(usedFlags), .buf(nsum), .buf(first),
                                              .buf(cornerUV), .buf(outIdx), .val(UInt32(nKept))])
            g.go(enc, "asm_vertices", nUsed, [.buf(nsum), .buf(first), .buf(cornerUV), .buf(outN), .buf(outUV), .val(UInt32(nUsed))])
        }) else { return nil }
        lap("g.asmKernels")
        var out = Mesh3()
        let ul = g.read(usedList, UInt32.self, nUsed)
        out.positions = pmap(ul) { P[Int($0)] }
        let nn = g.read(outN, SIMD4<Float>.self, nUsed), uu = g.read(outUV, SIMD2<Float>.self, nUsed)
        out.normals = pmap(nn) { Vec3(Double($0.x), Double($0.y), Double($0.z)) }
        out.uvs = pmap(uu) { Vec2(Double($0.x), Double($0.y)) }
        out.indices = g.read(outIdx, UInt32.self, nKept * 3)
        lap("g.asmReadback")
        return out
    }
}
#endif

extension BooleanJob {
    /// The whole boolean on the GPU-heavy pipeline, or nil (no device, a GPU error, `.keepCreases` weld) and the
    /// caller runs the CPU path on a fresh job.
    func runGPU2(_ op: BooleanOp) -> Mesh3? {
        #if canImport(Metal)
        guard let g = BooleanGPU2.shared else { return nil }
        let run = GPU2Run(job: self, g: g)
        let t0 = Date()
        guard run.weld() else { return nil }
        stats.seconds["weld"] = Date().timeIntervalSince(t0)
        let t1 = Date()
        guard run.cutAll() else { return nil }
        stats.seconds["cut"] = Date().timeIntervalSince(t1)
        let t2 = Date()
        guard let c = run.classify() else { return nil }
        stats.seconds["classify"] = Date().timeIntervalSince(t2)
        let t3 = Date()
        var mesh: Mesh3?
        if options.weld == .byPosition {
            mesh = run.assemble(op, parent: c.parent, rel: c.rel, relCPU: c.relCPU)
        } else {
            return nil
        }
        stats.seconds["assemble"] = Date().timeIntervalSince(t3)
        return mesh
        #else
        return nil
        #endif
    }
}
