import Foundation
import Metal
import OSLog
import QuartzCore
import simd
import VisualizerCore

// Forwarders for `MiscKnobs` — storage lives in IlluminatoramaMiscKnobs.swift.
extension IlluminatoramaRenderer {
    public var buildRTFromExtractedScene: Bool { get { kMisc.buildRTFromExtractedScene } set { kMisc.buildRTFromExtractedScene = newValue } _modify { yield &kMisc.buildRTFromExtractedScene } }
    var meshes: [MeshKind: IlluminatoramaMesh] {
        get { kMisc.meshes }
        set { kMisc.meshes = newValue; kMisc.meshesVersion &+= 1 }
        _modify { yield &kMisc.meshes; kMisc.meshesVersion &+= 1 }
    }
}
