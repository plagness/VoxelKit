import simd

/// Protocol for 3D spatial point index supporting nearest-neighbor queries.
///
/// Used by ICP scan matching: for each LiDAR point, find the nearest point
/// in the existing map to compute point-to-point correspondences.
public protocol SpatialIndex {
    var count: Int { get }
    mutating func build(from points: [SIMD3<Float>])
    func nearest(to query: SIMD3<Float>) -> SIMD3<Float>?
    func nearestK(_ k: Int, to query: SIMD3<Float>) -> [SIMD3<Float>]
    func radiusSearch(center: SIMD3<Float>, radius: Float) -> [SIMD3<Float>]
}
