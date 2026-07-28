import simd

// Deterministic procedural gait on the MakeHuman default rig. phase is the gait
// cycle in [0,1); left heel-strike at 0, right at 0.5. Amplitudes tuned for
// mid-shot plausibility, scaled by `vigor` (0 idle … 1 brisk walk).
public enum WalkCycle {

    // filmingArm: right arm holds a camera up at eye level (the auditor stance);
    // swing/flex apply to the left arm only.
    public static func pose(phase: Float, vigor: Float = 1, human: GeneratedHuman,
                            running: Bool = false, filmingArm: Bool = false) -> HumanPose {
        var p = HumanPose()
        let tau = 2 * Float.pi
        let t = phase * tau
        let swing = sin(t)              // +1: left leg forward
        let counter = -swing
        let double = sin(2 * t)         // per-step bounce terms

        // Legs: hip swing, knee flexion peaking mid-swing, ankle compensation.
        // A run is not a fast walk: bigger hip arc, much higher heel lift.
        let hip: Float = (running ? 0.58 : 0.42) * vigor
        let knee: Float = (running ? 1.35 : 0.65) * vigor
        p.bendX("upperleg01.L", -swing * hip)
        p.bendX("upperleg01.R", -counter * hip)
        p.bendX("lowerleg01.L", max(0, sin(t + 0.9)) * knee)
        p.bendX("lowerleg01.R", max(0, sin(t + .pi + 0.9)) * knee)
        p.bendX("foot.L", swing * 0.18 * vigor - 0.05)
        p.bendX("foot.R", counter * 0.18 * vigor - 0.05)

        // Arms: aim-solved (shortest arc to target hang directions) — no per-rig
        // sign guessing. Swing is sagittal; elbows flex forward; wrists relax and
        // fingers curl gently out of the bind splay. Runners pump bent arms.
        let arm: Float = (running ? 0.55 : 0.30) * vigor
        let flexBase: Float = running ? 0.9 : 0
        aimArm(&p, human: human, side: "L", swing: counter * arm,
               flex: flexBase + max(0, counter) * 0.35 * vigor, running: running)
        if filmingArm {
            aimArmAt(&p, human: human, side: "R",
                     upperTarget: simd_normalize(SIMD3(-0.10, -0.42, 0.90)),
                     lowerTarget: simd_normalize(SIMD3(-0.05, 0.42, 0.90)))
        } else {
            aimArm(&p, human: human, side: "R", swing: swing * arm,
                   flex: flexBase + max(0, swing) * 0.35 * vigor, running: running)
        }

        // Torso: forward lean (strong for a run), lateral sway, counter-twist.
        let lean: Float = running ? 0.20 : 0.05 * vigor + 0.02
        p.set("spine02", simd_quatf(angle: lean, axis: SIMD3(1, 0, 0)))
        p.set("spine03",
              simd_quatf(angle: swing * (running ? 0.10 : 0.06) * vigor, axis: SIMD3(0, 1, 0))
              * simd_quatf(angle: double * 0.03 * vigor, axis: SIMD3(0, 0, 1)))
        p.set("neck01", simd_quatf(angle: -swing * 0.04 * vigor - (running ? 0.10 : 0), axis: SIMD3(1, 0, 0)))
        return p
    }

    // Whole-body vertical bob for a run (applied by the host via model matrix):
    // two bounces per gait cycle, landing dip just after each foot strike.
    public static func runBob(phase: Float) -> Float {
        0.035 * (0.5 - 0.5 * cos(4 * .pi * phase))
    }

    // Aim the arm chain at explicit world directions: upper arm hangs nearly
    // straight down (slight outward), forearm continues with a relaxed forward
    // bend; `swing` pitches the whole target sagittally, `flex` adds elbow bend.
    // Solved with shortest-arc quats against the bind directions, so it works
    // for every body shape without per-rig sign tuning.
    private static func aimArm(_ p: inout HumanPose, human: GeneratedHuman, side: String, swing: Float, flex: Float, running: Bool = false) {
        let out: Float = side == "L" ? 1 : -1
        let qSwing = simd_quatf(angle: -swing, axis: SIMD3(1, 0, 0))   // -X pitches forward
        let qFlex = simd_quatf(angle: -(swing + flex), axis: SIMD3(1, 0, 0))
        aimArmAt(&p, human: human, side: side,
                 upperTarget: qSwing.act(simd_normalize(SIMD3(0.10 * out, -0.99, 0.0))),
                 lowerTarget: qFlex.act(simd_normalize(SIMD3(0.08 * out, -0.92, 0.38))))
    }

    // Aim an arm chain at explicit world directions (bind-relative FK form).
    public static func aimArmAt(_ p: inout HumanPose, human: GeneratedHuman, side: String,
                                upperTarget: SIMD3<Float>, lowerTarget: SIMD3<Float>) {
        guard let ui = human.boneIndex["upperarm01.\(side)"],
              let li = human.boneIndex["lowerarm01.\(side)"] else { return }
        let upper = human.bones[ui]
        let lower = human.bones[li]

        let upperBind = simd_normalize(upper.tail - upper.head)
        let qUpper = simd_quatf(from: upperBind, to: upperTarget)
        p.set("upperarm01.\(side)", qUpper)

        let lowerBind = simd_normalize(lower.tail - lower.head)
        // Child rotations premultiply under the parent in PosedHuman, so solve in
        // the parent-posed frame: q_lower = qU⁻¹ · arc(qU·bind → target) · qU.
        let arc = simd_quatf(from: qUpper.act(lowerBind), to: lowerTarget)
        p.set("lowerarm01.\(side)", qUpper.inverse * arc * qUpper)

        p.bendX("wrist.\(side)", 0.06)
        for finger in 1...5 {
            for seg in 1...3 {
                p.bendX("finger\(finger)-\(seg).\(side)", -0.16)
            }
        }
    }

    // Subtle breathing/weight-shift idle.
    public static func idle(time: Float) -> HumanPose {
        var p = HumanPose()
        let breathe = sin(time * 0.9)
        let sway = sin(time * 0.35)
        p.set("spine02", simd_quatf(angle: 0.015 * breathe + 0.02, axis: SIMD3(1, 0, 0)))
        p.set("spine03", simd_quatf(angle: 0.02 * sway, axis: SIMD3(0, 0, 1)))
        p.bendX("upperarm01.L", 0.06 + 0.01 * breathe)
        p.bendX("upperarm01.R", 0.06 + 0.01 * breathe)
        p.set("head", simd_quatf(angle: 0.02 * sway, axis: SIMD3(0, 1, 0)))
        return p
    }
}
