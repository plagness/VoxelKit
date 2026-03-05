import Foundation
import Vision
import simd
import CoreVideo
import CoreImage

/// Estimates camera pose and per-pixel depth from consecutive video frames
/// using Vision optical flow.
///
/// **Pose algorithm**:
/// 1. `VNGenerateOpticalFlowRequest` between consecutive frames (1/4 resolution).
/// 2. Mean flow → translation direction in camera-local XZ plane.
/// 3. Complementary filter accumulates pose over time.
///
/// **Depth algorithm (inverse-flow)**:
/// Depth ∝ 1/flow_magnitude. For lateral camera motion: depth ≈ fx × v / flow.
/// K = mean_flow × assumedMeanDepthM gives the proportionality constant.
/// This gives approximate metric depth for indoor scenes.
public actor OpticalFlowPoseEstimator {

    // MARK: - State

    private var previousImage: CIImage?
    private var currentPose: Pose3D = .identity
    private var lastFlowBuffer: CVPixelBuffer?
    private var lastDeltaMagnitude: Float = 0

    /// Smoothing factor for complementary filter (0 = all new, 1 = all old).
    public var alpha: Float = 0.85

    /// Metric scale factor: world metres per pixel of optical flow magnitude.
    public var depthScale: Float = 0.002

    /// Assumed mean scene depth for inverse-flow calibration (metres).
    public var assumedMeanDepthM: Float = 2.5

    /// Minimum depth to keep (metres).
    public var minDepth: Float = 0.3

    /// Maximum depth to keep (metres).
    public var maxDepth: Float = 8.0

    /// Minimum flow magnitude at 1/4-scale to compute depth (avoids static regions).
    public var minFlowForDepth: Float = 0.3

    public init() {}

    // MARK: - Public API

    /// Process a new frame — returns updated pose only.
    public func process(pixelBuffer: CVPixelBuffer) async -> Pose3D {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        defer { previousImage = image }
        guard let prev = previousImage else { return currentPose }
        let (delta, flowBuf) = await computeFlow(current: image, previous: prev)
        lastFlowBuffer = flowBuf
        lastDeltaMagnitude = delta.magnitude
        applyDelta(delta)
        return currentPose
    }

    /// Process a new frame and return per-pixel back-projected world positions
    /// derived from inverse-flow depth. Suitable for direct insertion into BotMapWorld.
    ///
    /// - Parameters:
    ///   - intrinsics: Camera intrinsics for back-projection.
    ///   - samplingStep: Process every Nth pixel of the flow buffer (default 2).
    /// - Returns: (pose, worldPositions) — positions are metric-approximate.
    public func processWithDepth(
        pixelBuffer: CVPixelBuffer,
        intrinsics: CameraIntrinsics,
        samplingStep: Int = 2
    ) async -> (pose: Pose3D, positions: [SIMD3<Float>]) {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        defer { previousImage = image }
        guard let prev = previousImage else { return (currentPose, []) }

        let (delta, flowBuf) = await computeFlow(current: image, previous: prev)
        lastFlowBuffer = flowBuf
        lastDeltaMagnitude = delta.magnitude
        applyDelta(delta)

        guard let flow = flowBuf, delta.magnitude > 0.05 else {
            return (currentPose, [])
        }

        // K = assumed_mean_depth × mean_flow_at_quarter_scale
        // per-pixel: depth_m = K / flow_magnitude_quarter
        let K = assumedMeanDepthM * delta.magnitude

        let videoW = CVPixelBufferGetWidth(pixelBuffer)
        let videoH = CVPixelBufferGetHeight(pixelBuffer)
        let scaled = intrinsics.scaled(toWidth: videoW, height: videoH)

        let positions = backProjectFromFlow(
            flowBuffer: flow, K: K,
            pose: currentPose, intrinsics: scaled,
            videoW: videoW, videoH: videoH,
            samplingStep: samplingStep
        )
        return (currentPose, positions)
    }

    /// Reset pose to identity.
    public func reset() {
        currentPose = .identity
        previousImage = nil
        lastFlowBuffer = nil
        lastDeltaMagnitude = 0
    }

    // MARK: - Private: Flow computation

    private func computeFlow(current: CIImage, previous: CIImage) async -> (FlowDelta, CVPixelBuffer?) {
        return await withCheckedContinuation { continuation in
            let scale: CGFloat = 0.25
            let sc = current.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            let sp = previous.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

            let request = VNGenerateOpticalFlowRequest(targetedCIImage: sp)
            request.computationAccuracy = .medium
            request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float

            let handler = VNImageRequestHandler(ciImage: sc)
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

        var sumX: Double = 0, sumY: Double = 0
        var count = 0
        let step = 4
        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let idx = y * (bytesPerRow / 4) + x * 2
                let fx = ptr[idx], fy = ptr[idx + 1]
                if abs(fx) < 100 && abs(fy) < 100 {
                    sumX += Double(fx); sumY += Double(fy); count += 1
                }
            }
        }
        guard count > 0 else { return .zero }
        return FlowDelta(dx: Float(sumX / Double(count)), dy: Float(sumY / Double(count)))
    }

    // MARK: - Private: Back-projection from flow

    private func backProjectFromFlow(
        flowBuffer: CVPixelBuffer,
        K: Float,
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

        // Flow buffer is at 1/4 video resolution → multiply coords × 4 for video pixels.
        let scaleX = Float(videoW) / Float(flowW)
        let scaleY = Float(videoH) / Float(flowH)

        let R = float3x3(pose.rotation)
        let t = pose.position

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity((flowW / samplingStep) * (flowH / samplingStep))

        for fy in stride(from: 0, to: flowH, by: samplingStep) {
            for fx in stride(from: 0, to: flowW, by: samplingStep) {
                let idx = fy * (bpr / 4) + fx * 2
                let dx  = ptr[idx]
                let dy  = ptr[idx + 1]
                let mag = sqrt(dx * dx + dy * dy)

                guard mag >= minFlowForDepth else { continue }

                let depth = K / mag
                guard depth >= minDepth && depth <= maxDepth else { continue }

                // Video pixel coordinates (centre of 4×4 flow cell)
                let u = Float(fx) * scaleX + scaleX * 0.5
                let v = Float(fy) * scaleY + scaleY * 0.5

                // Back-project to camera space, then world space
                let camX = (u - intrinsics.cx) / intrinsics.fx * depth
                let camY = (v - intrinsics.cy) / intrinsics.fy * depth
                let camZ = depth
                let world = R * SIMD3<Float>(camX, camY, camZ) + t
                positions.append(world)
            }
        }
        return positions
    }

    // MARK: - Private: Pose update

    private func applyDelta(_ delta: FlowDelta) {
        guard delta.magnitude > 0.1 else { return }

        let tx = delta.dx * depthScale
        let ty = delta.dy * depthScale
        let localTranslation = SIMD3<Float>(tx, 0, ty)
        let worldTranslation = currentPose.rotation.act(localTranslation)

        let newPosition = currentPose.position + worldTranslation
        currentPose.position = currentPose.position * alpha + newPosition * (1 - alpha)

        let angle    = atan2(delta.dy, delta.dx) * 0.01
        let yawDelta = simd_quatf(angle: angle, axis: SIMD3<Float>(0, 1, 0))
        currentPose.rotation = simd_normalize(simd_mul(currentPose.rotation, yawDelta))
    }
}

// MARK: - Flow Delta

private struct FlowDelta {
    let dx: Float
    let dy: Float
    static let zero = FlowDelta(dx: 0, dy: 0)
    var magnitude: Float { sqrt(dx * dx + dy * dy) }
}
