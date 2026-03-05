import Foundation
import Vision
import simd
import CoreVideo
import CoreImage

/// Estimates camera pose frame-by-frame using Vision optical flow.
///
/// Algorithm:
/// 1. `VNGenerateOpticalFlowRequest` between consecutive frames (1/4 resolution).
/// 2. Estimate rotation from dominant flow angle (rotation matrix around Z axis approximation).
/// 3. Estimate translation magnitude from mean flow magnitude × depth scale.
/// 4. Accumulate pose with complementary filter.
///
/// For monocular video, depth scale is ambiguous — we use a fixed metric scale
/// calibrated to roughly 1 m/s of camera motion per unit of mean flow.
/// Accuracy is sufficient for small-room indoor mapping.
public actor OpticalFlowPoseEstimator {

    // MARK: - State

    private var previousImage: CIImage?
    private var currentPose: Pose3D = .identity

    /// Smoothing factor for complementary filter (0 = all new, 1 = all old).
    public var alpha: Float = 0.85

    /// Metric scale factor: world metres per pixel of optical flow magnitude.
    /// Tune empirically: for a handheld iPhone at ~arm's length (~0.5 m from scene),
    /// typical flow is ~10–50 px, corresponding to 0.02–0.1 m motion → scale ~0.002.
    public var depthScale: Float = 0.002

    public init() {}

    // MARK: - Public API

    /// Process a new frame and return the updated absolute pose.
    ///
    /// - Parameter pixelBuffer: YUV or RGB CVPixelBuffer from AVAssetReader.
    /// - Returns: Updated `Pose3D` in world space.
    public func process(pixelBuffer: CVPixelBuffer) async -> Pose3D {
        let image = CIImage(cvPixelBuffer: pixelBuffer)

        defer { previousImage = image }

        guard let prev = previousImage else {
            // First frame: pose is identity
            return currentPose
        }

        // Compute optical flow
        let delta = await computeFlowDelta(current: image, previous: prev)
        applyDelta(delta)

        return currentPose
    }

    /// Reset pose to identity (call between capture sessions).
    public func reset() {
        currentPose = .identity
        previousImage = nil
    }

    // MARK: - Private: Flow computation

    private func computeFlowDelta(current: CIImage, previous: CIImage) async -> FlowDelta {
        return await withCheckedContinuation { continuation in
            // Scale down to 1/4 resolution for speed
            let scale = 0.25
            let scaledCurrent  = current.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            let scaledPrevious = previous.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

            let request = VNGenerateOpticalFlowRequest(targetedCIImage: scaledPrevious)
            request.computationAccuracy = .medium
            request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float

            let handler = VNImageRequestHandler(ciImage: scaledCurrent)

            do {
                try handler.perform([request])
                if let result = request.results?.first as? VNPixelBufferObservation {
                    let delta = extractFlowDelta(from: result.pixelBuffer)
                    continuation.resume(returning: delta)
                } else {
                    continuation.resume(returning: .zero)
                }
            } catch {
                continuation.resume(returning: .zero)
            }
        }
    }

    private func extractFlowDelta(from flowBuffer: CVPixelBuffer) -> FlowDelta {
        CVPixelBufferLockBaseAddress(flowBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowBuffer, .readOnly) }

        guard let baseAddr = CVPixelBufferGetBaseAddress(flowBuffer) else { return .zero }
        let width  = CVPixelBufferGetWidth(flowBuffer)
        let height = CVPixelBufferGetHeight(flowBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(flowBuffer)
        let ptr = baseAddr.bindMemory(to: Float.self, capacity: height * bytesPerRow / 4)

        var sumX: Double = 0, sumY: Double = 0
        var count = 0

        // Sample every 4th pixel for speed
        let step = 4
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let idx = y * (bytesPerRow / 4) + x * 2
                let fx = ptr[idx]
                let fy = ptr[idx + 1]
                // Filter out large/invalid flow vectors
                if abs(fx) < 100 && abs(fy) < 100 {
                    sumX += Double(fx)
                    sumY += Double(fy)
                    count += 1
                }
            }
        }

        guard count > 0 else { return .zero }
        return FlowDelta(dx: Float(sumX / Double(count)),
                         dy: Float(sumY / Double(count)))
    }

    // MARK: - Private: Pose update

    private func applyDelta(_ delta: FlowDelta) {
        guard delta.magnitude > 0.1 else { return }  // ignore sub-pixel jitter

        // Translate: flow in XY pixel space → world XZ translation
        // (camera moves in XZ plane when horizontal, XY when tilted)
        let tx = delta.dx * depthScale
        let ty = delta.dy * depthScale

        // Apply translation in camera-local space, then transform to world
        let localTranslation = SIMD3<Float>(tx, 0, ty)
        let worldTranslation = currentPose.rotation.act(localTranslation)

        // Complementary filter: blend new estimate with previous
        let newPosition = currentPose.position + worldTranslation
        currentPose.position = currentPose.position * alpha + newPosition * (1 - alpha)

        // Yaw estimation from flow direction (simplified)
        let angle = atan2(delta.dy, delta.dx) * 0.01  // very small rotation per frame
        let yawDelta = simd_quatf(angle: angle, axis: SIMD3<Float>(0, 1, 0))
        currentPose.rotation = simd_normalize(simd_mul(currentPose.rotation, yawDelta))
    }
}

// MARK: - Flow Delta

private struct FlowDelta {
    let dx: Float
    let dy: Float
    static let zero = FlowDelta(dx: 0, dy: 0)
    var magnitude: Float { sqrt(dx*dx + dy*dy) }
}
