import Foundation

/// Protocol implemented by all capture sources (file video, live stream).
public protocol CaptureSession: Actor {
    /// Start processing. Throws on fatal setup errors.
    func start() async throws

    /// Cancel processing. Idempotent.
    func cancel()

    /// AsyncStream of progress updates (at least 4 Hz during active processing).
    var progress: AsyncStream<CaptureProgress> { get }
}
