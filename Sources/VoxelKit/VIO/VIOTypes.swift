import Foundation
import simd

// MARK: - IMU Sample

/// Raw IMU measurement from any device.
public struct IMUSample: Sendable {
    /// Orientation quaternion (device-fused or raw).
    public let quaternion: simd_quatf
    /// Gyroscope reading (rad/s).
    public let gyroscope: SIMD3<Float>
    /// Accelerometer reading (m/s^2).
    public let accelerometer: SIMD3<Float>
    /// Sample timestamp.
    public let timestamp: Date

    public init(
        quaternion: simd_quatf,
        gyroscope: SIMD3<Float> = .zero,
        accelerometer: SIMD3<Float> = .zero,
        timestamp: Date = .now
    ) {
        self.quaternion = quaternion
        self.gyroscope = gyroscope
        self.accelerometer = accelerometer
        self.timestamp = timestamp
    }
}

// MARK: - Encoder Sample

/// Wheel/leg encoder measurement (ground robots only).
public struct EncoderSample: Sendable {
    /// World-space position from encoders.
    public let position: SIMD3<Float>
    /// World-space velocity from encoders.
    public let velocity: SIMD3<Float>
    /// Sample timestamp.
    public let timestamp: Date

    public init(
        position: SIMD3<Float>,
        velocity: SIMD3<Float>,
        timestamp: Date = .now
    ) {
        self.position = position
        self.velocity = velocity
        self.timestamp = timestamp
    }
}

// MARK: - VIO State

/// Current state of the VIO fusion pipeline.
public enum VIOState: String, Sendable {
    case uninitialized  // waiting for first sensor data
    case initializing   // collecting initial samples for scale calibration
    case tracking       // normal operation, fusing all available sensors
    case lost           // visual tracking lost, falling back to IMU/encoder
}

// MARK: - VIO Diagnostics

/// Runtime diagnostics from VIOSession for monitoring and debugging.
public struct VIODiagnostics: Sendable {
    /// Current pipeline state.
    public let state: VIOState
    /// Visual odometry quality (0 = lost, 1 = excellent).
    public let voQuality: Float
    /// Whether metric scale has converged.
    public let scaleConverged: Bool
    /// Current scale factor (visual → metric).
    public let scaleFactor: Float
    /// Last ICP residual score (lower = better, 0 = no ICP).
    public let icpScore: Float
    /// Total video frames processed.
    public let framesProcessed: Int

    public static let empty = VIODiagnostics(
        state: .uninitialized,
        voQuality: 0,
        scaleConverged: false,
        scaleFactor: 1,
        icpScore: 0,
        framesProcessed: 0
    )
}
