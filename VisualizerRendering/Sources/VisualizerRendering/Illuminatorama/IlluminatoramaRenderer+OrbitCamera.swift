import simd
import VisualizerCore

public extension IlluminatoramaRenderer {
    /// Drive `camera` from an eased `OrbitCameraRig`: aspect from the current
    /// output canvas, position/target from the orbit + pivot, FOV from the rig.
    ///
    /// This is the single per-tick camera write that every orbit-style native
    /// Illuminatorama scene used to hand-roll (AspectTest, EmissiveMonitor, …).
    /// Call it right after `rig.advance(...)`:
    /// ```
    /// rig.advance(distance: s.cameraDistance, …, easing: s.cameraEasing, dt: dt)
    /// renderer.applyOrbitCamera(rig, pivot: pivot, minY: 0.2)
    /// ```
    /// `pivot` is the orbit centre AND the look target (they coincide for an
    /// orbit camera). `minY` floors the camera's world Y — pass
    /// `-.greatestFiniteMagnitude` for no clamp.
    func applyOrbitCamera(_ rig: OrbitCameraRig, pivot: SIMD3<Float>,
                          minY: Float = -.greatestFiniteMagnitude) {
        camera.aspect = Float(outputWidth) / Float(max(1, outputHeight))
        camera.position = rig.easedPosition(pivot: pivot, minY: minY)
        camera.target = pivot
        camera.fovYRadians = rig.easedFovYRadians
    }
}
