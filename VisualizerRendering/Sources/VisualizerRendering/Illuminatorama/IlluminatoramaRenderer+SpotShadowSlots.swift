import simd

// STICKY SPOT-ATLAS SLICES (Daydream DH-1064)
// ===========================================
//
// `updateSpotShadows` hands the atlas's slices out compactly, in caster order: area lights
// (window portals, skylights) first, then the casting cones, each taking the next free slice. A
// static light's map only changes when the light or a caster in its frustum moves, and the
// encoder reuses a slice that still holds the map for this frame's matrix — but with compact
// order a slice's OWNER changes whenever the caster set does. A host that rations a scarce atlas
// by camera distance (Daydream's portal and cone budgets) changes the set as the camera travels,
// and one portal joining the front of the list shifted every later caster onto a new slice:
// measured on 4000 Sunset's daytime orbit, 8 of 120 orbit frames re-rendered ALL 8 slices
// although only 2–5 matrices had moved.
//
// So after the compact assignment, each caster whose matrix one of last frame's slices already
// holds is moved back onto that slice, and the newcomers fill the slices that freed up. The slice
// set stays exactly 0..<count (the encoder, `shadowViewsThisFrame` and the pass-descriptor cache
// all index casters by slice), and only a light's `shadowSliceIndex` — which the lighting reads
// per light — learns the permutation. A slice beyond this frame's count cannot be kept; its
// caster lands on a free slice and re-renders, exactly as before.
extension IlluminatoramaRenderer {

    func stickSpotShadowSlicesToLastFrame() {
        let last = shd.lastSpotShadowMats
        let n = shd.spotShadowCasters.count
        guard spotShadowStickySlicesEnabled, n > 1, !last.isEmpty else { return }
        let casters = shd.spotShadowCasters
        // Already where they were (the common, static case): nothing to permute.
        if n <= last.count && (0..<n).allSatisfy({ casters[$0].matrix == last[$0] }) { return }

        let keepable = min(n, last.count)
        var bySlice = [ShadowCaster?](repeating: nil, count: n)
        var newcomers: [ShadowCaster] = []
        newcomers.reserveCapacity(n)
        for c in casters {
            if let s = (0..<keepable).first(where: { bySlice[$0] == nil && last[$0] == c.matrix }) {
                bySlice[s] = c
            } else {
                newcomers.append(c)
            }
        }
        var next = newcomers.makeIterator()
        for s in 0..<n where bySlice[s] == nil { bySlice[s] = next.next() }

        shd.spotShadowCasters.removeAll(keepingCapacity: true)
        for (slice, c) in bySlice.enumerated() {
            guard let c else { continue }   // unreachable: newcomers exactly fill the holes
            shd.spotShadowCasters.append(c)
            switch c.owner {
            case .spot(let i): spotLights[i].shadowSliceIndex = Int32(slice)
            case .area(let i): areaLights[i].shadowSliceIndex = Int32(slice)
            }
        }
    }
}
