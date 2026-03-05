import Foundation
import AVFoundation
import CoreVideo

/// Processes a `.mov` / `.mp4` file through the mapping pipeline.
///
/// Pipeline:
/// ```
/// AVAssetReader (VideoToolbox HW decode)
///   → CVPixelBuffer (420YpCbCr8BiPlanarVideoRange)
///   → VideoFrameCallback (passed to VoxelInserter in VoxelKitCompute)
/// ```
///
/// This actor manages reading and sequencing frames. The actual depth estimation
/// and voxel insertion is handled by `VoxelInsertionPipeline` (VoxelKitCompute).
public actor VideoCaptureSession: CaptureSession {

    // MARK: - Configuration

    public enum PlaybackRate: Sendable {
        /// Process at real-time speed (1× wall clock).
        case realTime
        /// Process as fast as hardware allows (no throttle).
        case fast
        /// Custom multiplier (0.5×, 2×, etc.)
        case custom(Double)
    }

    // MARK: - State

    private let url: URL
    private let rate: PlaybackRate
    private var cancelled = false
    private var _progress: AsyncStream<CaptureProgress>.Continuation?

    public let progress: AsyncStream<CaptureProgress>
    private var progressContinuation: AsyncStream<CaptureProgress>.Continuation

    /// Callback invoked for each decoded frame. Set before calling `start()`.
    /// Receives: CVPixelBuffer (YUV 4:2:0), frame index, total frame count, frame timestamp.
    public var onFrame: (@Sendable (CVPixelBuffer, Int, Int, CMTime) async -> Void)?

    /// Actor-isolated setter for `onFrame` (call with `await` from outside the actor).
    public func setOnFrame(_ callback: @Sendable @escaping (CVPixelBuffer, Int, Int, CMTime) async -> Void) {
        self.onFrame = callback
    }

    public init(url: URL, rate: PlaybackRate = .fast) {
        self.url = url
        self.rate = rate

        var continuation: AsyncStream<CaptureProgress>.Continuation!
        self.progress = AsyncStream { continuation = $0 }
        self.progressContinuation = continuation
    }

    // MARK: - CaptureSession

    public func start() async throws {
        cancelled = false

        let asset = AVAsset(url: url)
        let reader = try AVAssetReader(asset: asset)

        // Video track
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VideoCaptureError.noVideoTrack
        }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        trackOutput.alwaysCopiesSampleData = false

        guard reader.canAdd(trackOutput) else { throw VideoCaptureError.cannotAddTrackOutput }
        reader.add(trackOutput)
        guard reader.startReading() else {
            throw VideoCaptureError.readerFailed(reader.error)
        }

        // Frame count estimate
        let duration = try await asset.load(.duration)
        let frameRate = try await track.load(.nominalFrameRate)
        let totalFrames = max(1, Int(CMTimeGetSeconds(duration) * Double(frameRate)))

        // Timing for real-time mode
        let nominalFPS = Double(frameRate)
        let frameDuration: Duration = .nanoseconds(Int64(1_000_000_000 / nominalFPS))

        var processedFrames = 0
        var lastProgressTime = Date.now
        var lastFPSTime = Date.now
        var fpsFrameCount = 0

        while !cancelled {
            guard let sampleBuffer = trackOutput.copyNextSampleBuffer() else { break }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            processedFrames += 1
            fpsFrameCount += 1

            // Invoke frame callback
            if let cb = onFrame {
                await cb(pixelBuffer, processedFrames, totalFrames, presentationTime)
            }

            // Real-time throttle
            if case .realTime = rate {
                try await Task.sleep(for: frameDuration)
            } else if case .custom(let multiplier) = rate, multiplier < 10 {
                let ns = Int64(1_000_000_000 / (nominalFPS * multiplier))
                try await Task.sleep(for: .nanoseconds(ns))
            }

            // Emit progress ~4 Hz
            let now = Date.now
            if now.timeIntervalSince(lastProgressTime) >= 0.25 {
                let elapsed = now.timeIntervalSince(lastFPSTime)
                let fps = elapsed > 0 ? Double(fpsFrameCount) / elapsed : 0
                fpsFrameCount = 0
                lastFPSTime = now
                lastProgressTime = now

                let prog = CaptureProgress(
                    processedFrames: processedFrames,
                    totalFrames: totalFrames,
                    insertedVoxelCount: 0,  // updated by VoxelInsertionPipeline
                    fps: fps,
                    currentPose: .identity
                )
                progressContinuation.yield(prog)
            }
        }

        reader.cancelReading()
        progressContinuation.finish()
    }

    public func cancel() {
        cancelled = true
    }
}

// MARK: - Errors

public enum VideoCaptureError: Error, LocalizedError {
    case noVideoTrack
    case cannotAddTrackOutput
    case readerFailed(Error?)

    public var errorDescription: String? {
        switch self {
        case .noVideoTrack:           return "No video track found in file"
        case .cannotAddTrackOutput:   return "Cannot add track output to AVAssetReader"
        case .readerFailed(let e):    return "AVAssetReader failed: \(e?.localizedDescription ?? "unknown")"
        }
    }
}
