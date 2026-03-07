import Foundation
import simd

// MARK: - Device Category

/// Category of robotic device for VIO pipeline tuning.
public enum DeviceCategory: String, Sendable, Codable {
    case robodog    // legged robot with encoders (Go2, Spot, etc.)
    case drone      // aerial vehicle, no ground contact encoders
    case handheld   // handheld scanner (phone-like, no encoders)
    case custom
}

// MARK: - Fusion Weights

/// Tuning parameters for sensor fusion in VIOSession.
public struct FusionWeights: Sendable {
    /// How much to trust IMU orientation (0=ignore, 1=fully trust IMU over VO).
    /// Go2 Air = 0.85 (excellent internal MCU fusion).
    /// Cheap drone IMU = 0.5.
    public var imuOrientationTrust: Float

    /// How much to trust encoder position (0=no encoders, 0.7=good encoders).
    /// Go2 Air = 0.7. Drones = 0.
    public var encoderPositionTrust: Float

    /// Maximum weight for visual odometry contribution (0..1).
    /// Higher = VO contributes more to fused pose.
    public var voMaxWeight: Float

    /// Maximum weight for ICP scan-match correction (0..1).
    /// Applied as `icpCorrectionWeight * (1 - icpScore)`.
    public var icpCorrectionWeight: Float

    public init(
        imuOrientationTrust: Float = 0.85,
        encoderPositionTrust: Float = 0.7,
        voMaxWeight: Float = 0.4,
        icpCorrectionWeight: Float = 0.5
    ) {
        self.imuOrientationTrust = imuOrientationTrust
        self.encoderPositionTrust = encoderPositionTrust
        self.voMaxWeight = voMaxWeight
        self.icpCorrectionWeight = icpCorrectionWeight
    }
}

// MARK: - Device Profile

/// Hardware description of a robotic device for VIO pipeline configuration.
///
/// Contains camera intrinsics, sensor rates, mounting offsets, and fusion weights.
/// Use built-in presets (`.go2Air`, `.genericDrone`) or create a custom profile.
public struct DeviceProfile: Sendable {
    public let name: String
    public let category: DeviceCategory
    public let camera: CameraIntrinsics
    /// Sensor (camera/LiDAR) mounting offset from robot body center (metres).
    public let sensorMountingOffset: SIMD3<Float>
    /// IMU telemetry rate in Hz.
    public let imuRate: Float
    /// Video frame rate in Hz.
    public let videoRate: Float
    /// Process every N-th video frame for VO (skip others).
    public let voFrameSkip: Int
    /// Device has wheel/leg encoders providing position + velocity.
    public let hasEncoder: Bool
    /// Device has LiDAR for ICP correction + depth.
    public let hasLiDAR: Bool
    /// Fusion algorithm weights.
    public let fusionWeights: FusionWeights

    public init(
        name: String,
        category: DeviceCategory,
        camera: CameraIntrinsics,
        sensorMountingOffset: SIMD3<Float> = .zero,
        imuRate: Float = 10,
        videoRate: Float = 30,
        voFrameSkip: Int = 3,
        hasEncoder: Bool = false,
        hasLiDAR: Bool = false,
        fusionWeights: FusionWeights = FusionWeights()
    ) {
        self.name = name
        self.category = category
        self.camera = camera
        self.sensorMountingOffset = sensorMountingOffset
        self.imuRate = imuRate
        self.videoRate = videoRate
        self.voFrameSkip = voFrameSkip
        self.hasEncoder = hasEncoder
        self.hasLiDAR = hasLiDAR
        self.fusionWeights = fusionWeights
    }
}

// MARK: - Built-in Presets

extension DeviceProfile {

    /// Unitree Go2 Air quadruped robot.
    /// Camera: 1280x720 wide-angle (~120 FOV), LiDAR: 128x128x40 5cm, IMU+encoders at 10Hz.
    public static let go2Air = DeviceProfile(
        name: "Unitree Go2 Air",
        category: .robodog,
        camera: .go2Air,
        sensorMountingOffset: SIMD3<Float>(0.15, 0, 0.10),
        imuRate: 10,
        videoRate: 30,
        voFrameSkip: 3,
        hasEncoder: true,
        hasLiDAR: true,
        fusionWeights: FusionWeights(
            imuOrientationTrust: 0.85,
            encoderPositionTrust: 0.7,
            voMaxWeight: 0.4,
            icpCorrectionWeight: 0.5
        )
    )

    /// Generic quadcopter drone with camera + optional LiDAR, no ground encoders.
    /// IMU typically at 100Hz+, camera 30fps.
    public static let genericDrone = DeviceProfile(
        name: "Generic Drone",
        category: .drone,
        camera: CameraIntrinsics(fx: 500, fy: 500, cx: 640, cy: 360,
                                 width: 1280, height: 720),
        sensorMountingOffset: SIMD3<Float>(0, 0, -0.05),
        imuRate: 100,
        videoRate: 30,
        voFrameSkip: 3,
        hasEncoder: false,
        hasLiDAR: false,
        fusionWeights: FusionWeights(
            imuOrientationTrust: 0.5,
            encoderPositionTrust: 0,
            voMaxWeight: 0.8,
            icpCorrectionWeight: 0
        )
    )

    /// Generic ground robot with encoders, camera, and optional LiDAR.
    public static let genericRobot = DeviceProfile(
        name: "Generic Robot",
        category: .robodog,
        camera: CameraIntrinsics(fx: 600, fy: 600, cx: 320, cy: 240,
                                 width: 640, height: 480),
        sensorMountingOffset: SIMD3<Float>(0.1, 0, 0.15),
        imuRate: 10,
        videoRate: 30,
        voFrameSkip: 3,
        hasEncoder: true,
        hasLiDAR: true,
        fusionWeights: FusionWeights(
            imuOrientationTrust: 0.7,
            encoderPositionTrust: 0.6,
            voMaxWeight: 0.5,
            icpCorrectionWeight: 0.5
        )
    )
}
