import Foundation
import simd

// MARK: - Map Layers

/// Semantic map layer for voxel classification.
public enum MapLayer: UInt8, Codable, CaseIterable, Sendable {
    case structure = 0  // floor, walls, ceiling — permanent
    case furniture = 1  // furniture, large objects — semi-permanent
    case dynamic   = 2  // people, animals — TTL 30s
}

// MARK: - Voxel Position

public typealias VoxelPosition = SIMD3<Float>

// MARK: - Packed Voxel (GPU-ready, 16 bytes)

/// Packed voxel for Metal buffer.
/// Layout: x(4) y(4) z(4) colorAndFlags(4) = 16 bytes — matches Metal float3+uint.
/// `colorAndFlags` layout: [R8][G8][B8][layer 2bit | classId 6bit]
public struct PackedVoxel: Sendable {
    public var x: Float
    public var y: Float
    public var z: Float
    public var colorAndFlags: UInt32

    public init(position: SIMD3<Float>, color: (UInt8, UInt8, UInt8) = (128, 128, 128),
                layer: MapLayer = .structure, classId: UInt8 = 0) {
        self.x = position.x
        self.y = position.y
        self.z = position.z
        let flags = (UInt32(layer.rawValue) << 6) | UInt32(classId & 0x3F)
        self.colorAndFlags = (UInt32(color.0) << 24) | (UInt32(color.1) << 16)
                           | (UInt32(color.2) << 8) | flags
    }

    public var position: SIMD3<Float> { SIMD3<Float>(x, y, z) }

    public var layer: MapLayer {
        MapLayer(rawValue: UInt8((colorAndFlags & 0xFF) >> 6)) ?? .structure
    }

    public var classId: UInt8 {
        UInt8(colorAndFlags & 0x3F)
    }
}

// MARK: - LiDAR Frame

/// Decoded LiDAR packet: world-space voxel positions from one scan.
public struct LiDARFrame: Sendable {
    public let origin: SIMD3<Float>
    public let resolution: Float
    public let voxels: [VoxelPosition]
    public let timestamp: Date

    public init(origin: SIMD3<Float>, resolution: Float,
                voxels: [VoxelPosition], timestamp: Date = .now) {
        self.origin = origin
        self.resolution = resolution
        self.voxels = voxels
        self.timestamp = timestamp
    }
}

// MARK: - Axis-Aligned Bounding Box

public struct AxisAlignedBoundingBox: Sendable {
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>

    public static let zero = AxisAlignedBoundingBox(min: .zero, max: .zero)

    public init(min: SIMD3<Float>, max: SIMD3<Float>) {
        self.min = min
        self.max = max
    }

    public var size: SIMD3<Float> { max - min }
    public var center: SIMD3<Float> { (min + max) * 0.5 }
    public var volume: Float { let s = size; return s.x * s.y * s.z }

    public func contains(_ point: SIMD3<Float>) -> Bool {
        all(point .>= min) && all(point .<= max)
    }

    public func intersects(_ other: AxisAlignedBoundingBox) -> Bool {
        all(min .<= other.max) && all(max .>= other.min)
    }

    public mutating func expand(toInclude point: SIMD3<Float>) {
        min = simd_min(min, point)
        max = simd_max(max, point)
    }

    public mutating func expand(toInclude other: AxisAlignedBoundingBox) {
        min = simd_min(min, other.min)
        max = simd_max(max, other.max)
    }
}

public typealias AABB = AxisAlignedBoundingBox
