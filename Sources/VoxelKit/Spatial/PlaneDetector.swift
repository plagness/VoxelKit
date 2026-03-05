import Foundation
import simd
import os

private let logger = Logger(subsystem: "com.voxelkit", category: "PlaneDetector")

/// Detects dominant horizontal planes in a LiDAR frame and fills them solid.
///
/// The Go2 Air LiDAR publishes sparse surface hits (~1–5% of the voxel grid).
/// PlaneDetector fills large flat surfaces (floor, table, shelves) using a
/// Z-histogram to find prominent horizontal levels and fills their bounding area.
///
/// Algorithm (O(N log N), single-pass per frame):
/// 1. Build a histogram of voxel Z-values with 5 cm bins.
/// 2. Find peaks where bin density > `minDensityPerM2` voxels / m².
/// 3. For each peak: collect the XY bounding box of voxels in Z ± 2.5 cm.
/// 4. Fill all 5 cm cells within that XY bbox at that Z level.
public enum PlaneDetector {

    public static let binSize: Float = 0.05
    public static let minDensityPerM2: Float = 4.0
    public static let planeHalfThickness: Float = 0.025
    public static let maxPlanes: Int = 6
    public static let maxPlaneExtent: Float = 8.0

    /// Detect horizontal planes in `frame` and return fill positions.
    public static func detectFillPositions(in frame: LiDARFrame,
                                           resolution: Float = 0.05) -> [SIMD3<Float>] {
        guard frame.voxels.count > 20 else { return [] }

        var histogram: [Int: Int] = [:]
        for v in frame.voxels {
            let bin = Int(floor(v.z / binSize))
            histogram[bin, default: 0] += 1
        }

        var peaks: [(z: Float, count: Int)] = []
        for (bin, count) in histogram {
            let zCenter = (Float(bin) + 0.5) * binSize
            let area = Float(count) / minDensityPerM2
            if Float(count) >= minDensityPerM2 * max(1.0, area) {
                peaks.append((z: zCenter, count: count))
            }
        }

        peaks.sort { $0.count > $1.count }
        let topPeaks = Array(peaks.prefix(maxPlanes))
        guard !topPeaks.isEmpty else { return [] }

        var fillPositions: [SIMD3<Float>] = []

        for peak in topPeaks {
            let zLow  = peak.z - planeHalfThickness
            let zHigh = peak.z + planeHalfThickness

            var minX = Float.greatestFiniteMagnitude, minY = Float.greatestFiniteMagnitude
            var maxX = -Float.greatestFiniteMagnitude, maxY = -Float.greatestFiniteMagnitude
            var inPlane = 0

            for v in frame.voxels {
                guard v.z >= zLow && v.z <= zHigh else { continue }
                if v.x < minX { minX = v.x }; if v.y < minY { minY = v.y }
                if v.x > maxX { maxX = v.x }; if v.y > maxY { maxY = v.y }
                inPlane += 1
            }

            guard inPlane >= 4 else { continue }

            let extentX = min(maxX - minX, maxPlaneExtent)
            let extentY = min(maxY - minY, maxPlaneExtent)
            let centerX = (minX + maxX) * 0.5
            let centerY = (minY + maxY) * 0.5

            guard extentX > resolution && extentY > resolution else { continue }

            let clampedMinX = centerX - extentX * 0.5
            let clampedMinY = centerY - extentY * 0.5
            let clampedMaxX = centerX + extentX * 0.5
            let clampedMaxY = centerY + extentY * 0.5

            var x = clampedMinX
            while x <= clampedMaxX {
                var y = clampedMinY
                while y <= clampedMaxY {
                    fillPositions.append(SIMD3<Float>(x, y, peak.z))
                    y += resolution
                }
                x += resolution
            }
        }

        logger.debug("PlaneDetector: \(topPeaks.count) planes, \(fillPositions.count) fill voxels")
        return fillPositions
    }

    /// Lowest-Z peak in the histogram (for auto floor detection).
    public static func lowestPlaneZ(in frame: LiDARFrame) -> Float? {
        guard frame.voxels.count > 20 else { return nil }
        var histogram: [Int: Int] = [:]
        for v in frame.voxels {
            let bin = Int(floor(v.z / binSize))
            histogram[bin, default: 0] += 1
        }
        let qualified = histogram.filter { $0.value >= 10 }
        guard let lowestBin = qualified.keys.min() else { return nil }
        return (Float(lowestBin) + 0.5) * binSize
    }
}
