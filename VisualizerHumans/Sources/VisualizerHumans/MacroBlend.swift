import Foundation

// HumanSpec → sparse morph weights, reproducing MakeHuman's macro interpolation:
// each macro axis resolves to anchor weights, and a target's weight is the
// product of its axes' anchor weights. Targets missing from the dataset
// (MakeHuman ships no file when a combination equals the base) contribute 0.
enum MacroBlend {

    // MakeHuman age slider: 1y…25y maps to 0…0.5, 25y…90y maps to 0.5…1,
    // with anchors baby=0, child=0.1875, young=0.5, old=1.
    static func ageSlider(years: Float) -> Float {
        if years < 25 {
            return (years - 1) / ((25 - 1) * 2)
        }
        return (years - 25) / ((90 - 25) * 2) + 0.5
    }

    struct Anchor { let name: String; let weight: Float }

    static func ageAnchors(_ a: Float) -> [Anchor] {
        if a < 0.1875 {
            return [Anchor(name: "baby", weight: (0.1875 - a) / 0.1875),
                    Anchor(name: "child", weight: a / 0.1875)]
        }
        if a < 0.5 {
            return [Anchor(name: "child", weight: (0.5 - a) / 0.3125),
                    Anchor(name: "young", weight: (a - 0.1875) / 0.3125)]
        }
        return [Anchor(name: "young", weight: (1 - a) / 0.5),
                Anchor(name: "old", weight: (a - 0.5) / 0.5)]
    }

    static func genderAnchors(_ g: Float) -> [Anchor] {
        [Anchor(name: "female", weight: 1 - g), Anchor(name: "male", weight: g)]
    }

    // min / average / max triple used by muscle and weight axes. The "average"
    // anchor exists as a real target file in the universal grid.
    static func triAnchors(_ v: Float, min minName: String, avg avgName: String, max maxName: String) -> [Anchor] {
        if v < 0.5 {
            return [Anchor(name: minName, weight: (0.5 - v) / 0.5),
                    Anchor(name: avgName, weight: v / 0.5)]
        }
        return [Anchor(name: avgName, weight: (1 - v) / 0.5),
                Anchor(name: maxName, weight: (v - 0.5) / 0.5)]
    }

    // Height and proportions have no "average" target — the base mesh IS the
    // average, so mid-slider contributes nothing.
    static func bipolarAnchors(_ v: Float, min minName: String, max maxName: String) -> [Anchor] {
        if v < 0.5 {
            return [Anchor(name: minName, weight: (0.5 - v) / 0.5)]
        }
        return [Anchor(name: maxName, weight: (v - 0.5) / 0.5)]
    }

    static func weights(for spec: HumanSpec, dataset: HumanDataset) -> [Int: Float] {
        let age = ageAnchors(ageSlider(years: spec.ageYears))
        let gender = genderAnchors(spec.gender)
        let muscle = triAnchors(spec.muscle, min: "minmuscle", avg: "averagemuscle", max: "maxmuscle")
        let weight = triAnchors(spec.weight, min: "minweight", avg: "averageweight", max: "maxweight")
        let height = bipolarAnchors(spec.height, min: "minheight", max: "maxheight")
        let proportions = bipolarAnchors(spec.proportions, min: "uncommonproportions", max: "idealproportions")

        let ethnicitySum = max(spec.african + spec.asian + spec.caucasian, 0.0001)
        let ethnicity = [
            Anchor(name: "african", weight: spec.african / ethnicitySum),
            Anchor(name: "asian", weight: spec.asian / ethnicitySum),
            Anchor(name: "caucasian", weight: spec.caucasian / ethnicitySum),
        ]

        var out: [Int: Float] = [:]
        func add(_ name: String, _ w: Float) {
            guard w > 0.0005, let idx = dataset.morphIndex[name] else { return }
            out[idx, default: 0] += w
        }

        // Ethnic head/feature targets: {ethnicity}-{gender}-{age}
        for e in ethnicity where e.weight > 0 {
            for g in gender where g.weight > 0 {
                for a in age where a.weight > 0 {
                    add("\(e.name)-\(g.name)-\(a.name)", e.weight * g.weight * a.weight)
                }
            }
        }

        // Universal body-mass grid: universal-{gender}-{age}-{muscle}-{weight}
        for g in gender where g.weight > 0 {
            for a in age where a.weight > 0 {
                for m in muscle where m.weight > 0 {
                    for w in weight where w.weight > 0 {
                        add("universal-\(g.name)-\(a.name)-\(m.name)-\(w.name)",
                            g.weight * a.weight * m.weight * w.weight)
                    }
                }
            }
        }

        // Height: height/{gender}-{age}-{muscle}-{weight}-{min|max}height
        for g in gender where g.weight > 0 {
            for a in age where a.weight > 0 {
                for m in muscle where m.weight > 0 {
                    for w in weight where w.weight > 0 {
                        for h in height where h.weight > 0 {
                            add("height/\(g.name)-\(a.name)-\(m.name)-\(w.name)-\(h.name)",
                                g.weight * a.weight * m.weight * w.weight * h.weight)
                        }
                    }
                }
            }
        }

        // Proportions: proportions/{gender}-{age}-{muscle}-{weight}-{ideal|uncommon}proportions
        for g in gender where g.weight > 0 {
            for a in age where a.weight > 0 {
                for m in muscle where m.weight > 0 {
                    for w in weight where w.weight > 0 {
                        for p in proportions where p.weight > 0 {
                            add("proportions/\(g.name)-\(a.name)-\(m.name)-\(w.name)-\(p.name)",
                                g.weight * a.weight * m.weight * w.weight * p.weight)
                        }
                    }
                }
            }
        }

        // ── Detail axes ─────────────────────────────────────────────────────

        // Silhouette archetype, blended across the gendered elvs variants.
        if spec.archetype != .none, spec.archetypeAmount > 0 {
            if let fem = spec.archetype.feminineTarget {
                add(fem, spec.archetypeAmount * (1 - spec.gender))
            }
            if let masc = spec.archetype.masculineTarget {
                add(masc, spec.archetypeAmount * spec.gender)
            }
        }

        // Simple bipolar pairs: mid-slider is the base mesh.
        func bipolar(_ v: Float, low: String, high: String) {
            if v < 0.5 { add(low, (0.5 - v) / 0.5) } else { add(high, (v - 0.5) / 0.5) }
        }
        bipolar(spec.belly, low: "stomach/stomach-tone-incr", high: "stomach/stomach-pregnant-incr")
        bipolar(spec.glutes, low: "buttocks/buttocks-volume-decr", high: "buttocks/buttocks-volume-incr")
        bipolar(spec.hips, low: "hip/hip-scale-horiz-decr", high: "hip/hip-scale-horiz-incr")

        // Female-weighted chest grid: breast/female-{age}-{muscle}-{weight}-{cup}-{firmness}
        // (child/young/old only — no baby tier ships).
        let femW = 1 - spec.gender
        if femW > 0.001 {
            let cup = triAnchors(spec.cup, min: "mincup", avg: "averagecup", max: "maxcup")
            let firm = triAnchors(spec.firmness, min: "minfirmness", avg: "averagefirmness", max: "maxfirmness")
            for a in age where a.weight > 0 && a.name != "baby" {
                for m in muscle where m.weight > 0 {
                    for w in weight where w.weight > 0 {
                        for c in cup where c.weight > 0 {
                            for f in firm where f.weight > 0 {
                                add("breast/female-\(a.name)-\(m.name)-\(w.name)-\(c.name)-\(f.name)",
                                    femW * a.weight * m.weight * w.weight * c.weight * f.weight)
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
