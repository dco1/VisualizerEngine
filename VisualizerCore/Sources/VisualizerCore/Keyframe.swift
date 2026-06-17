import Foundation

public struct Keyframe: Identifiable, Hashable {
    public let id: UUID
    public var time: Double
    public var value: Double
    public var easing: Easing

    public init(
        id: UUID = UUID(),
        time: Double,
        value: Double,
        easing: Easing = .easeInOut
    ) {
        self.id = id
        self.time = time
        self.value = value
        self.easing = easing
    }
}
