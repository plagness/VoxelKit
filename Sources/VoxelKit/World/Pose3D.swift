import simd
import Foundation

/// 6-DoF camera pose: position + rotation.
public struct Pose3D: Sendable, Codable {
    public var position: SIMD3<Float>
    public var rotation: simd_quatf

    public static let identity = Pose3D(
        position: .zero,
        rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    )

    public init(position: SIMD3<Float>, rotation: simd_quatf) {
        self.position = position
        self.rotation = rotation
    }

    /// Transform a camera-space point to world space.
    public func transform(_ point: SIMD3<Float>) -> SIMD3<Float> {
        rotation.act(point) + position
    }

    /// 4×4 model matrix (for Metal shaders).
    public var matrix: float4x4 {
        let r = float4x4(rotation)
        return float4x4(
            SIMD4<Float>(r.columns.0.x, r.columns.0.y, r.columns.0.z, 0),
            SIMD4<Float>(r.columns.1.x, r.columns.1.y, r.columns.1.z, 0),
            SIMD4<Float>(r.columns.2.x, r.columns.2.y, r.columns.2.z, 0),
            SIMD4<Float>(position.x, position.y, position.z, 1)
        )
    }

    // MARK: - Codable (simd_quatf is not Codable by default)

    enum CodingKeys: String, CodingKey {
        case px, py, pz, qx, qy, qz, qw
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        position = SIMD3<Float>(
            try c.decode(Float.self, forKey: .px),
            try c.decode(Float.self, forKey: .py),
            try c.decode(Float.self, forKey: .pz)
        )
        rotation = simd_quatf(
            ix: try c.decode(Float.self, forKey: .qx),
            iy: try c.decode(Float.self, forKey: .qy),
            iz: try c.decode(Float.self, forKey: .qz),
            r:  try c.decode(Float.self, forKey: .qw)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(position.x, forKey: .px)
        try c.encode(position.y, forKey: .py)
        try c.encode(position.z, forKey: .pz)
        try c.encode(rotation.imag.x, forKey: .qx)
        try c.encode(rotation.imag.y, forKey: .qy)
        try c.encode(rotation.imag.z, forKey: .qz)
        try c.encode(rotation.real,   forKey: .qw)
    }
}
