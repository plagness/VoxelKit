import Foundation

/// Progress snapshot from an active capture session.
public struct CaptureProgress: Sendable {
    /// Frames processed so far.
    public var processedFrames: Int
    /// Total frames (0 if unknown, e.g. live stream).
    public var totalFrames: Int
    /// Occupied voxels inserted into the world.
    public var insertedVoxelCount: Int
    /// Processing rate (frames per second).
    public var fps: Double
    /// Current camera pose estimate.
    public var currentPose: Pose3D
    /// Stream-mode extras (nil for file capture).
    public var stream: StreamStats?

    public init(processedFrames: Int, totalFrames: Int, insertedVoxelCount: Int,
                fps: Double, currentPose: Pose3D, stream: StreamStats? = nil) {
        self.processedFrames = processedFrames
        self.totalFrames = totalFrames
        self.insertedVoxelCount = insertedVoxelCount
        self.fps = fps
        self.currentPose = currentPose
        self.stream = stream
    }

    /// Progress fraction 0–1 (nil if total unknown).
    public var fraction: Double? {
        guard totalFrames > 0 else { return nil }
        return Double(processedFrames) / Double(totalFrames)
    }

    /// Estimated seconds remaining (nil if unknown).
    public var estimatedSecondsRemaining: Double? {
        guard let f = fraction, f > 0, fps > 0 else { return nil }
        let remaining = Double(totalFrames - processedFrames)
        return remaining / fps
    }
}

/// Statistics for stream-mode capture.
public struct StreamStats: Sendable {
    public var latencyMs: Double
    public var droppedFrames: Int
    public var bytesPerSecond: Double

    public init(latencyMs: Double, droppedFrames: Int, bytesPerSecond: Double) {
        self.latencyMs = latencyMs
        self.droppedFrames = droppedFrames
        self.bytesPerSecond = bytesPerSecond
    }
}
