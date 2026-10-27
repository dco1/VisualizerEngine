import simd

/// The one table of a unit cube's six faces — normal plus four corners, CCW seen from outside —
/// shared by `IlluminatoramaMesh.unitBox` and `BoxMesh.unit()` so the renderer's front-face
/// convention for a cube is declared once.
enum UnitCube {
    static let faces: [(normal: SIMD3<Float>, corners: [SIMD3<Float>])] = {
        let h: Float = 0.5
        return [
            (SIMD3( 1, 0, 0), [SIMD3( h,-h, h), SIMD3( h,-h,-h), SIMD3( h, h,-h), SIMD3( h, h, h)]),
            (SIMD3(-1, 0, 0), [SIMD3(-h,-h,-h), SIMD3(-h,-h, h), SIMD3(-h, h, h), SIMD3(-h, h,-h)]),
            (SIMD3( 0, 1, 0), [SIMD3(-h, h, h), SIMD3( h, h, h), SIMD3( h, h,-h), SIMD3(-h, h,-h)]),
            (SIMD3( 0,-1, 0), [SIMD3(-h,-h,-h), SIMD3( h,-h,-h), SIMD3( h,-h, h), SIMD3(-h,-h, h)]),
            (SIMD3( 0, 0, 1), [SIMD3(-h,-h, h), SIMD3( h,-h, h), SIMD3( h, h, h), SIMD3(-h, h, h)]),
            (SIMD3( 0, 0,-1), [SIMD3( h,-h,-h), SIMD3(-h,-h,-h), SIMD3(-h, h,-h), SIMD3( h, h,-h)]),
        ]
    }()
}
