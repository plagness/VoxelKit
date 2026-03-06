import Foundation
import Vision
import simd
import CoreVideo
import CoreImage

/// Estimates camera pose and per-pixel depth from consecutive video frames
/// using Vision optical flow + essential matrix decomposition.
///
/// **Pose algorithm**:
/// 1. Between keyframes: simple flow-based delta for smooth tracking.
/// 2. At keyframe boundaries: essential matrix (8-point + RANSAC) → proper R, t.
/// 3. Essential matrix "corrects" accumulated drift at each keyframe.
///
/// **Depth algorithm (parallax)**:
/// Subtracts mean flow (rotation) to isolate parallax; `depth = K / |parallax|`.
///
/// **Resource management**:
/// Images scaled on entry (1/8 for 4K, 1/4 otherwise). Keyframes stored as
/// scaled copies. Adjacent flow uses `.low` accuracy.
public actor OpticalFlowPoseEstimator {

    // MARK: - State

    private var previousScaledImage: CIImage?
    private var currentPose: Pose3D = .identity

    /// Keyframe: pre-scaled image + pose at capture time.
    private var keyframe: (pose: Pose3D, image: CIImage)?

    private var accumulatedDistance: Float = 0
    private var framesSinceKeyframe: Int = 0
    private var totalFrameCount: Int = 0

    // MARK: - Configuration

    public var alpha: Float = 0.85
    public var depthScale: Float = 0.002
    public var minDepth: Float = 0.3
    public var maxDepth: Float = 8.0
    public var minParallax: Float = 0.15
    public var keyframeDistanceThreshold: Float = 0.15
    public var frameSkip: Int = 1

    public init() {}

    public func setFrameSkip(_ skip: Int) {
        frameSkip = max(1, skip)
    }

    // MARK: - Public API

    public func process(pixelBuffer: CVPixelBuffer) async -> Pose3D {
        totalFrameCount += 1
        guard totalFrameCount % frameSkip == 0 else { return currentPose }

        let scaled = scaleImage(CIImage(cvPixelBuffer: pixelBuffer),
                                width: CVPixelBufferGetWidth(pixelBuffer))
        defer { previousScaledImage = scaled }
        guard let prev = previousScaledImage else { return currentPose }
        let (delta, _) = await computeFlow(currentScaled: scaled, previousScaled: prev)
        applySimpleDelta(delta)
        return currentPose
    }

    public func processWithDepth(
        pixelBuffer: CVPixelBuffer,
        intrinsics: CameraIntrinsics,
        samplingStep: Int = 2
    ) async -> (pose: Pose3D, positions: [SIMD3<Float>]) {
        totalFrameCount += 1
        guard totalFrameCount % frameSkip == 0 else { return (currentPose, []) }

        let videoW = CVPixelBufferGetWidth(pixelBuffer)
        let videoH = CVPixelBufferGetHeight(pixelBuffer)
        let scaled = scaleImage(CIImage(cvPixelBuffer: pixelBuffer), width: videoW)
        defer { previousScaledImage = scaled }

        // First frame: set keyframe
        guard let prev = previousScaledImage else {
            keyframe = (pose: currentPose, image: scaled)
            return (currentPose, [])
        }

        // Adjacent-frame flow for smooth inter-keyframe tracking
        let (delta, _) = await computeFlow(currentScaled: scaled, previousScaled: prev, accuracy: .low)
        applySimpleDelta(delta)

        accumulatedDistance += delta.meanMagnitude * depthScale
        framesSinceKeyframe += 1

        if keyframe == nil {
            keyframe = (pose: currentPose, image: scaled)
            framesSinceKeyframe = 0
            return (currentPose, [])
        }

        // Trigger at distance threshold or frame count
        let shouldEmit = accumulatedDistance >= keyframeDistanceThreshold
                      || framesSinceKeyframe >= 30
        guard shouldEmit else {
            return (currentPose, [])
        }

        // Keyframe flow (medium accuracy for essential matrix + depth)
        let (_, keyflowBuf) = await computeFlow(currentScaled: scaled, previousScaled: keyframe!.image)

        let kfPose = keyframe!.pose
        keyframe = (pose: currentPose, image: scaled)
        accumulatedDistance = 0
        framesSinceKeyframe = 0

        guard let flow = keyflowBuf else { return (currentPose, []) }

        let scaledIntrinsics = intrinsics.scaled(toWidth: videoW, height: videoH)

        // --- Essential matrix: get proper R, t ---
        if let (R, t) = EssentialMatrixEstimator.estimateMotion(
            flowBuffer: flow,
            fx: scaledIntrinsics.fx, fy: scaledIntrinsics.fy,
            cx: scaledIntrinsics.cx, cy: scaledIntrinsics.cy,
            videoW: videoW, videoH: videoH
        ) {
            // R rotates from keyframe camera to current camera
            let relQuat = simd_quatf(R)

            // Apply relative rotation: current = keyframe * relative
            currentPose.rotation = simd_normalize(simd_mul(kfPose.rotation, relQuat))

            // Apply translation (t is unit vector — scale by accumulated flow)
            let translationScale = accumulatedFlowScale(delta)
            let worldT = kfPose.rotation.act(t * translationScale)
            currentPose.position = kfPose.position + worldT
        }

        // --- Parallax depth ---
        let positions = parallaxDepth(
            flowBuffer: flow,
            pose: currentPose,
            intrinsics: scaledIntrinsics,
            videoW: videoW, videoH: videoH,
            samplingStep: samplingStep
        )
        return (currentPose, positions)
    }

    public func reset() {
        currentPose = .identity
        previousScaledImage = nil
        keyframe = nil
        accumulatedDistance = 0
        framesSinceKeyframe = 0
        totalFrameCount = 0
    }

    // MARK: - Private: Helpers

    /// Estimate translation scale from flow magnitude.
    private func accumulatedFlowScale(_ delta: FlowDelta) -> Float {
        return delta.meanMagnitude * depthScale * 10
    }

    private func scaleImage(_ image: CIImage, width: Int) -> CIImage {
        let scale: CGFloat = width > 2000 ? 0.125 : 0.25
        return image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }

    // MARK: - Private: Simple pose delta (inter-keyframe smoothing)

    private func applySimpleDelta(_ delta: FlowDelta) {
        guard delta.meanMagnitude > 0.05 else { return }

        let localX =  delta.dx * depthScale
        let localY = -delta.dy * depthScale
        let lateralMag = delta.magnitude
        let forwardMag = sqrt(max(0, delta.meanMagnitude * delta.meanMagnitude - lateralMag * lateralMag))
        let localZ = forwardMag * depthScale

        let worldTranslation = currentPose.rotation.act(SIMD3<Float>(localX, localY, localZ))
        let newPosition = currentPose.position + worldTranslation
        currentPose.position = currentPose.position * alpha + newPosition * (1 - alpha)

        // Simple yaw for inter-keyframe smoothness (will be corrected at keyframe)
        if lateralMag > 0.5 {
            let angle = atan2(delta.dy, delta.dx) * 0.003
            let yawDelta = simd_quatf(angle: angle, axis: SIMD3<Float>(0, 1, 0))
            currentPose.rotation = simd_normalize(simd_mul(currentPose.rotation, yawDelta))
        }
    }

    // MARK: - Private: Parallax depth

    private func parallaxDepth(
        flowBuffer: CVPixelBuffer,
        pose: Pose3D,
        intrinsics: CameraIntrinsics,
        videoW: Int, videoH: Int,
        samplingStep: Int
    ) -> [SIMD3<Float>] {
        CVPixelBufferLockBaseAddress(flowBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowBuffer, .readOnly) }

        guard let addr = CVPixelBufferGetBaseAddress(flowBuffer) else { return [] }
        let flowW = CVPixelBufferGetWidth(flowBuffer)
        let flowH = CVPixelBufferGetHeight(flowBuffer)
        let bpr   = CVPixelBufferGetBytesPerRow(flowBuffer)
        let ptr   = addr.bindMemory(to: Float.self, capacity: flowH * bpr / 4)

        // Pass 1: mean flow (rotational component)
        var sumFx: Double = 0, sumFy: Double = 0
        var meanCount = 0
        for fy in stride(from: 0, to: flowH, by: 4) {
            for fx in stride(from: 0, to: flowW, by: 4) {
                let idx = fy * (bpr / 4) + fx * 2
                let fxv = ptr[idx], fyv = ptr[idx + 1]
                if abs(fxv) < 100 && abs(fyv) < 100 {
                    sumFx += Double(fxv); sumFy += Double(fyv); meanCount += 1
                }
            }
        }
        let meanFx = meanCount > 0 ? Float(sumFx / Double(meanCount)) : 0
        let meanFy = meanCount > 0 ? Float(sumFy / Double(meanCount)) : 0

        // Pass 2: parallax depth
        let scaleX = Float(videoW) / Float(flowW)
        let scaleY = Float(videoH) / Float(flowH)
        let K: Float = intrinsics.fx * depthScale * 20

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity((flowW / samplingStep) * (flowH / samplingStep))

        for fy in stride(from: 0, to: flowH, by: samplingStep) {
            for fx in stride(from: 0, to: flowW, by: samplingStep) {
                let idx = fy * (bpr / 4) + fx * 2
                let dx  = ptr[idx]
                let dy  = ptr[idx + 1]

                let px = dx - meanFx
                let py = dy - meanFy
                let parallax = sqrt(px * px + py * py)
                guard parallax >= minParallax else { continue }

                let depth = min(maxDepth, max(minDepth, K / parallax))

                let u = Float(fx) * scaleX + scaleX * 0.5
                let v = Float(fy) * scaleY + scaleY * 0.5
                let dir = pose.rotation.act(simd_normalize(SIMD3<Float>(
                    (u - intrinsics.cx) / intrinsics.fx,
                    (v - intrinsics.cy) / intrinsics.fy,
                    1.0
                )))
                positions.append(pose.position + dir * depth)
            }
        }
        return positions
    }

    // MARK: - Private: Flow computation

    private func computeFlow(
        currentScaled: CIImage,
        previousScaled: CIImage,
        accuracy: VNGenerateOpticalFlowRequest.ComputationAccuracy = .medium
    ) async -> (FlowDelta, CVPixelBuffer?) {
        return await withCheckedContinuation { continuation in
            let request = VNGenerateOpticalFlowRequest(targetedCIImage: previousScaled)
            request.computationAccuracy = accuracy
            request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float

            let handler = VNImageRequestHandler(ciImage: currentScaled)
            do {
                try handler.perform([request])
                if let result = request.results?.first as? VNPixelBufferObservation {
                    let delta = extractFlowDelta(from: result.pixelBuffer)
                    continuation.resume(returning: (delta, result.pixelBuffer))
                } else {
                    continuation.resume(returning: (.zero, nil))
                }
            } catch {
                continuation.resume(returning: (.zero, nil))
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

        var sumX: Double = 0, sumY: Double = 0, sumMag: Double = 0
        var count = 0
        let step = 4
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let idx = y * (bytesPerRow / 4) + x * 2
                let fx = ptr[idx], fy = ptr[idx + 1]
                if abs(fx) < 100 && abs(fy) < 100 {
                    sumX   += Double(fx)
                    sumY   += Double(fy)
                    sumMag += Double(sqrt(fx * fx + fy * fy))
                    count  += 1
                }
            }
        }
        guard count > 0 else { return .zero }
        return FlowDelta(
            dx: Float(sumX / Double(count)),
            dy: Float(sumY / Double(count)),
            meanMagnitude: Float(sumMag / Double(count))
        )
    }
}

// MARK: - Flow Delta

struct FlowDelta {
    let dx: Float
    let dy: Float
    let meanMagnitude: Float
    static let zero = FlowDelta(dx: 0, dy: 0, meanMagnitude: 0)
    var magnitude: Float { sqrt(dx * dx + dy * dy) }
}
