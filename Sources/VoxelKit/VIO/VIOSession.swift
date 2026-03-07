import Foundation
import simd
import CoreVideo

/// Visual-Inertial Odometry session — the main public API for sensor fusion.
///
/// Usage:
/// ```swift
/// let vio = VIOSession(profile: .go2Air)
/// // Feed sensor data as it arrives:
/// await vio.feedVideo(pixelBuffer)
/// vio.feedIMU(imuSample)
/// vio.feedEncoder(encoderSample)
/// vio.feedLiDAR(lidarFrame)
/// // Read fused pose:
/// let pose = vio.currentPose
/// ```
///
/// The session internally manages:
/// - `OpticalFlowPoseEstimator` for visual odometry
/// - `ScaleEstimator` for metric scale recovery
/// - Complementary fusion of visual + encoder + IMU
/// - Optional ICP correction when LiDAR data is available
public actor VIOSession {

    // MARK: - Configuration

    public let profile: DeviceProfile

    // MARK: - State

    private var state: VIOState = .uninitialized
    private var fusedPose: Pose3D = .identity
    private let visualOdometry = OpticalFlowPoseEstimator()
    private var scaleEstimator: ScaleEstimator

    // Sensor state
    private var lastIMU: IMUSample?
    private var lastEncoder: EncoderSample?
    private var lastVideoTimestamp: Date = .distantPast
    private var videoFrameCount: Int = 0

    // VO tracking
    private var lastVOPose: Pose3D = .identity
    private var voQuality: Float = 0

    // Encoder dead-reckoning
    private var prevEncoderPos: SIMD3<Float> = .zero
    private var prevEncoderVelocity: SIMD3<Float> = .zero
    private var deadReckonPos: SIMD3<Float> = .zero

    // Latest video buffer (for colorization)
    private var latestPixelBuffer: CVPixelBuffer?

    // MARK: - Init

    private var voConfigured = false

    public init(profile: DeviceProfile) {
        self.profile = profile
        self.scaleEstimator = ScaleEstimator()
    }

    // MARK: - Public Output

    /// Current fused 6-DoF pose.
    public var currentPose: Pose3D { fusedPose }

    /// Whether the session has been initialized with sensor data.
    public var isInitialized: Bool { state != .uninitialized }

    /// Runtime diagnostics for debugging.
    public var diagnostics: VIODiagnostics {
        VIODiagnostics(
            state: state,
            voQuality: voQuality,
            scaleConverged: scaleEstimator.isConverged,
            scaleFactor: scaleEstimator.scale,
            icpScore: 0,
            framesProcessed: videoFrameCount
        )
    }

    /// Latest captured video frame (for external colorization).
    public var latestVideoFrame: CVPixelBuffer? { latestPixelBuffer }

    /// Colorize 3D points using the latest video frame and current fused pose.
    /// Returns empty array if no video frame is available.
    public func colorize(points: [SIMD3<Float>]) -> [ColoredPosition] {
        guard let buffer = latestPixelBuffer else { return [] }
        return VideoColorSampler.colorize(
            points: points,
            pixelBuffer: buffer,
            cameraPose: fusedPose,
            intrinsics: profile.camera
        )
    }

    // MARK: - Sensor Feeds

    /// Feed a video frame for visual odometry processing.
    public func feedVideo(_ pixelBuffer: CVPixelBuffer) async {
        if !voConfigured {
            await visualOdometry.setFrameSkip(profile.voFrameSkip)
            voConfigured = true
        }
        latestPixelBuffer = pixelBuffer
        videoFrameCount += 1

        let voPose = await visualOdometry.process(pixelBuffer: pixelBuffer)

        // Compute VO displacement for scale estimation
        let voDisplacement = simd_length(voPose.position - lastVOPose.position)
        if let enc = lastEncoder, voDisplacement > 0.01 {
            let encDisplacement = simd_length(enc.position - prevEncoderPos)
            if encDisplacement > 0.01 {
                scaleEstimator.addEncoderSample(
                    encoderDisplacement: encDisplacement,
                    voDisplacement: voDisplacement
                )
            }
        }

        // Estimate VO quality from pose change magnitude (crude but functional)
        voQuality = min(1.0, voDisplacement * 10)

        lastVOPose = voPose

        // Fuse visual with current state
        fuseVisualUpdate(voPose)

        if state == .uninitialized { state = .initializing }
        if state == .initializing && scaleEstimator.isConverged { state = .tracking }
    }

    /// Feed an IMU measurement.
    public func feedIMU(_ sample: IMUSample) {
        lastIMU = sample

        if state == .uninitialized {
            // Initialize orientation from first IMU
            fusedPose.rotation = sample.quaternion
            state = .initializing
            return
        }

        // Fuse IMU orientation
        let w = profile.fusionWeights
        fusedPose.rotation = simd_normalize(
            simd_slerp(fusedPose.rotation, sample.quaternion, w.imuOrientationTrust)
        )
    }

    /// Feed an encoder measurement (ground robots only).
    public func feedEncoder(_ sample: EncoderSample) {
        if state == .uninitialized {
            fusedPose.position = sample.position
            prevEncoderPos = sample.position
            prevEncoderVelocity = sample.velocity
            deadReckonPos = sample.position
            state = .initializing
            lastEncoder = sample
            return
        }

        let w = profile.fusionWeights
        guard w.encoderPositionTrust > 0 else {
            lastEncoder = sample
            return
        }

        // Dead-reckon from previous velocity
        if let prev = lastEncoder {
            let dt = Float(sample.timestamp.timeIntervalSince(prev.timestamp))
            if dt > 0, dt < 0.5 {
                deadReckonPos = prevEncoderPos + prevEncoderVelocity * dt

                // Complementary blend: encoder (low-freq) + dead-reckoning (high-freq)
                let alpha = exp(-dt / 0.15)
                let blended = (1 - alpha) * sample.position + alpha * deadReckonPos

                // Blend encoder into fused position
                let trust = w.encoderPositionTrust
                fusedPose.position = fusedPose.position * (1 - trust) + blended * trust
            }
        }

        prevEncoderPos = sample.position
        prevEncoderVelocity = sample.velocity
        deadReckonPos = fusedPose.position
        lastEncoder = sample
    }

    /// Feed a LiDAR frame for ICP correction.
    public func feedLiDAR(_ frame: LiDARFrame) {
        // ICP correction will be added in Phase 3 when ScanMatcher moves to VoxelKit.
        // For now, LiDAR data can be used for scale estimation via depth comparison.
    }

    /// Apply ICP scan-match correction (called externally or from feedLiDAR when ScanMatcher is available).
    public func applyICPCorrection(position: SIMD3<Float>, orientation: simd_quatf, score: Float) {
        let w = profile.fusionWeights
        let alpha = w.icpCorrectionWeight * max(0, 1 - score * 2)
        guard alpha > 0.01 else { return }

        fusedPose.position = fusedPose.position * (1 - alpha) + position * alpha
        fusedPose.rotation = simd_normalize(simd_slerp(fusedPose.rotation, orientation, alpha))
        deadReckonPos = fusedPose.position
    }

    /// Reset the session to uninitialized state.
    public func reset() async {
        state = .uninitialized
        fusedPose = .identity
        await visualOdometry.reset()
        scaleEstimator.reset()
        lastIMU = nil
        lastEncoder = nil
        lastVOPose = .identity
        voQuality = 0
        prevEncoderPos = .zero
        prevEncoderVelocity = .zero
        deadReckonPos = .zero
        latestPixelBuffer = nil
        videoFrameCount = 0
    }

    // MARK: - Private Fusion

    private func fuseVisualUpdate(_ voPose: Pose3D) {
        let w = profile.fusionWeights
        let voWeight = min(w.voMaxWeight, voQuality * 0.5)
        guard voWeight > 0.01 else { return }

        // Scale VO translation to metric
        let scaledVOPos = voPose.position * scaleEstimator.scale

        // Blend VO position into fused
        fusedPose.position = fusedPose.position * (1 - voWeight) + scaledVOPos * voWeight

        // Blend VO orientation (secondary to IMU)
        let voOrientWeight = (1 - w.imuOrientationTrust) * voWeight / w.voMaxWeight
        if voOrientWeight > 0.01 {
            fusedPose.rotation = simd_normalize(
                simd_slerp(fusedPose.rotation, voPose.rotation, voOrientWeight)
            )
        }
    }
}
