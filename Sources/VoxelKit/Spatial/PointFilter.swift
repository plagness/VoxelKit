import simd

/// Statistical outlier removal for noisy 3D point clouds.
///
/// Uses a spatial hash grid for O(N) neighbor lookup, then rejects points
/// whose mean neighbor distance exceeds (globalMean + multiplier * globalStdDev).
public enum PointFilter {

    /// Remove statistical outliers from a point cloud.
    ///
    /// - Parameters:
    ///   - points: Input 3D points.
    ///   - k: Number of nearest neighbors to consider.
    ///   - stddevMultiplier: Rejection threshold in standard deviations.
    ///   - cellSize: Spatial hash grid cell size (should be ~2-5x expected point spacing).
    /// - Returns: Filtered points with outliers removed.
    public static func removeOutliers(
        from points: [(SIMD3<Float>, (UInt8, UInt8, UInt8))],
        k: Int = 6,
        stddevMultiplier: Float = 1.5,
        cellSize: Float = 0.1
    ) -> [(SIMD3<Float>, (UInt8, UInt8, UInt8))] {
        guard points.count > k + 1 else { return points }

        // Build spatial hash grid
        let invCell = 1.0 / cellSize
        var grid: [SIMD3<Int32>: [Int]] = [:]
        grid.reserveCapacity(points.count)

        for (i, pt) in points.enumerated() {
            let cell = cellKey(pt.0, invCell: invCell)
            grid[cell, default: []].append(i)
        }

        // Compute mean neighbor distance for each point
        var meanDists = [Float](repeating: 0, count: points.count)

        for (i, pt) in points.enumerated() {
            let center = cellKey(pt.0, invCell: invCell)
            var dists: [Float] = []
            dists.reserveCapacity(27 * 4)

            // Search 3x3x3 neighborhood
            for dz: Int32 in -1...1 {
                for dy: Int32 in -1...1 {
                    for dx: Int32 in -1...1 {
                        let neighborCell = SIMD3<Int32>(center.x + dx, center.y + dy, center.z + dz)
                        guard let indices = grid[neighborCell] else { continue }
                        for j in indices {
                            if j == i { continue }
                            dists.append(simd_distance(pt.0, points[j].0))
                        }
                    }
                }
            }

            // Take k nearest
            if dists.count > k {
                dists.sort()
                dists = Array(dists.prefix(k))
            }

            meanDists[i] = dists.isEmpty ? Float.greatestFiniteMagnitude
                         : dists.reduce(0, +) / Float(dists.count)
        }

        // Compute global mean and stddev of mean distances
        let validDists = meanDists.filter { $0 < Float.greatestFiniteMagnitude }
        guard !validDists.isEmpty else { return points }

        let globalMean = validDists.reduce(0, +) / Float(validDists.count)
        let variance = validDists.reduce(0.0) { $0 + ($1 - globalMean) * ($1 - globalMean) }
            / Float(validDists.count)
        let globalStdDev = variance.squareRoot()

        let threshold = globalMean + stddevMultiplier * globalStdDev

        // Filter
        return points.enumerated().compactMap { (i, pt) in
            meanDists[i] <= threshold ? pt : nil
        }
    }

    private static func cellKey(_ pos: SIMD3<Float>, invCell: Float) -> SIMD3<Int32> {
        SIMD3<Int32>(
            Int32(floor(pos.x * invCell)),
            Int32(floor(pos.y * invCell)),
            Int32(floor(pos.z * invCell))
        )
    }
}
