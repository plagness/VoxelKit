import Foundation
import simd

/// A single frame of data streamed from iPhone ARKit to Mac.
public struct VoxelStreamFrame: Sendable {
    /// Sequence number indicating a stop-marker (iPhone stopped recording).
    public static let stopMarkerSequence: UInt32 = 0xFFFF_FFFF

    /// Monotonically increasing frame counter.
    public let sequence: UInt32
    /// ARFrame.timestamp (seconds since device boot).
    public let timestamp: Double
    /// Camera transform (4x4) from ARKit — world-space pose.
    public let pose: simd_float4x4
    /// Camera intrinsics (3x3) from ARKit.
    public let intrinsics: simd_float3x3
    /// JPEG-compressed camera image.
    public let imageJPEG: Data
    /// Image dimensions in pixels.
    public let imageWidth: UInt16
    public let imageHeight: UInt16
    /// LiDAR depth map as Float16 data (nil on non-LiDAR devices like iPhone 13).
    public let depthFloat16: Data?
    /// Depth map dimensions (0x0 if no depth).
    public let depthWidth: UInt16
    public let depthHeight: UInt16
    /// World-space 3D points from ARKit (plane vertices + feature points).
    /// Packed as Float32×3 per point. nil if no points available.
    public let worldPoints: Data?
    /// Number of world-space 3D points.
    public let worldPointCount: UInt32

    public init(
        sequence: UInt32,
        timestamp: Double,
        pose: simd_float4x4,
        intrinsics: simd_float3x3,
        imageJPEG: Data,
        imageWidth: UInt16,
        imageHeight: UInt16,
        depthFloat16: Data? = nil,
        depthWidth: UInt16 = 0,
        depthHeight: UInt16 = 0,
        worldPoints: Data? = nil,
        worldPointCount: UInt32 = 0
    ) {
        self.sequence = sequence
        self.timestamp = timestamp
        self.pose = pose
        self.intrinsics = intrinsics
        self.imageJPEG = imageJPEG
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.depthFloat16 = depthFloat16
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.worldPoints = worldPoints
        self.worldPointCount = worldPointCount
    }
}
