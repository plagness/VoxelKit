import Foundation
import simd
import Accelerate
import os

private let logger = Logger(subsystem: "com.voxelkit", category: "ScanMatcher")

/// Point-to-point ICP scan matcher for LiDAR-based pose correction.
///
/// Aligns a new LiDAR frame against the current voxel map using Iterative Closest Point
/// with SVD-based rigid transform estimation (Apple Accelerate, no external deps).
///
/// **Usage:**
/// ```swift
/// let matcher = ScanMatcher()
/// let result = matcher.match(
///     framePoints: frame.voxels,
///     mapPoints: mapPointsInRadius,
///     initialPosition: pose.position,
///     initialOrientation: pose.orientation
/// )
/// ```
///
/// **Performance target:** < 50ms per frame (500 source points, 5m radius map).
public struct ScanMatcher: Sendable {

    // MARK: - Parameters

    /// Maximum ICP iterations before stopping.
    public var maxIterations: Int = 15

    /// Convergence threshold: translation change (meters) below which ICP stops.
    public var convergenceThreshold: Float = 0.001

    /// Maximum correspondence distance (meters). Pairs further apart are rejected.
    public var maxCorrespondenceDistance: Float = 0.5

    /// Minimum fraction of source points that must have inlier correspondences.
    public var minInlierRatio: Float = 0.3

    /// Maximum number of source points used (stride downsampling).
    public var maxSourcePoints: Int = 500

    public init() {}

    // MARK: - Result

    public struct Result: Sendable {
        /// Corrected world-space position.
        public let position: SIMD3<Float>
        /// Corrected world-space orientation.
        public let orientation: simd_quatf
        /// Mean point-to-point residual error in meters (lower = better match).
        public let score: Float
        /// True if ICP converged within `convergenceThreshold`.
        public let converged: Bool
        public let iterations: Int
        public let inlierCount: Int
        public let totalSourcePoints: Int
    }

    // MARK: - Match

    /// Align `framePoints` (robot-local coordinates) against world-space `mapPoints`.
    ///
    /// The initial pose transforms frame points to world space.
    /// Returns refined pose, or `nil` if alignment fails.
    public func match(
        framePoints: [SIMD3<Float>],
        mapPoints: [SIMD3<Float>],
        initialPosition: SIMD3<Float>,
        initialOrientation: simd_quatf
    ) -> Result? {
        let srcFull = downsample(framePoints, maxCount: maxSourcePoints)
        guard srcFull.count >= 6, mapPoints.count >= 6 else {
            return nil
        }

        var tree = KDTree()
        tree.build(from: mapPoints)

        var pos = initialPosition
        var rot = simd_normalize(initialOrientation)

        var worldSrc = srcFull.map { simd_float3x3(rot) * $0 + pos }

        var finalScore: Float = .greatestFiniteMagnitude
        var finalInliers: Int = 0
        var converged = false
        var iter = 0

        for iteration in 0..<maxIterations {
            iter = iteration + 1

            var srcCorr: [SIMD3<Float>] = []
            var dstCorr: [SIMD3<Float>] = []
            var residualSum: Float = 0

            for wp in worldSrc {
                guard let nearest = tree.nearest(to: wp) else { continue }
                let dist = simd_distance(wp, nearest)
                guard dist < maxCorrespondenceDistance else { continue }
                srcCorr.append(wp)
                dstCorr.append(nearest)
                residualSum += dist
            }

            guard srcCorr.count >= 6 else { break }

            finalScore = residualSum / Float(srcCorr.count)
            finalInliers = srcCorr.count

            guard let (deltaR, deltaT) = rigidTransformSVD(src: srcCorr, dst: dstCorr) else { break }

            worldSrc = worldSrc.map { deltaR * $0 + deltaT }

            pos = deltaR * pos + deltaT
            rot = simd_normalize(simd_mul(simd_quatf(deltaR), rot))

            if simd_length(deltaT) < convergenceThreshold {
                converged = true
                break
            }
        }

        let inlierRatio = Float(finalInliers) / Float(srcFull.count)
        guard inlierRatio >= minInlierRatio, finalScore < maxCorrespondenceDistance else {
            return nil
        }

        logger.debug("ICP score=\(finalScore, format: .fixed(precision: 3))m inliers=\(finalInliers)/\(srcFull.count) iter=\(iter)\(converged ? "" : " (no converge)")")

        return Result(
            position: pos,
            orientation: rot,
            score: finalScore,
            converged: converged,
            iterations: iter,
            inlierCount: finalInliers,
            totalSourcePoints: srcFull.count
        )
    }

    // MARK: - SVD Rigid Transform

    private func rigidTransformSVD(
        src: [SIMD3<Float>],
        dst: [SIMD3<Float>]
    ) -> (simd_float3x3, SIMD3<Float>)? {
        let n = src.count
        guard n > 0 else { return nil }
        let fn = Float(n)

        let cSrc = src.reduce(.zero, +) / fn
        let cDst = dst.reduce(.zero, +) / fn

        var H = simd_float3x3(0)
        for i in 0..<n {
            let a = src[i] - cSrc
            let b = dst[i] - cDst
            H.columns.0 += a * b.x
            H.columns.1 += a * b.y
            H.columns.2 += a * b.z
        }

        guard let (U, Vt) = svd3x3(H) else { return nil }

        var R = simd_transpose(Vt) * simd_transpose(U)

        if simd_determinant(R) < 0 {
            var Vfix = simd_transpose(Vt)
            Vfix.columns.2 = -Vfix.columns.2
            R = Vfix * simd_transpose(U)
        }

        let t = cDst - R * cSrc
        return (R, t)
    }

    // MARK: - LAPACK SVD 3x3

    private func svd3x3(_ H: simd_float3x3) -> (U: simd_float3x3, Vt: simd_float3x3)? {
        var A: [Float] = [
            H.columns.0.x, H.columns.0.y, H.columns.0.z,
            H.columns.1.x, H.columns.1.y, H.columns.1.z,
            H.columns.2.x, H.columns.2.y, H.columns.2.z
        ]
        var jobu  = Int8(UInt8(ascii: "A"))
        var jobvt = Int8(UInt8(ascii: "A"))
        var m: Int32 = 3
        var n: Int32 = 3
        var lda: Int32 = 3
        var s   = [Float](repeating: 0, count: 3)
        var u   = [Float](repeating: 0, count: 9)
        var ldu: Int32 = 3
        var vt  = [Float](repeating: 0, count: 9)
        var ldvt: Int32 = 3
        var lwork: Int32 = 64
        var work = [Float](repeating: 0, count: 64)
        var info: Int32 = 0

        sgesvd_(&jobu, &jobvt, &m, &n, &A, &lda,
                &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &info)

        guard info == 0 else { return nil }

        let U = simd_float3x3(columns: (
            SIMD3<Float>(u[0], u[1], u[2]),
            SIMD3<Float>(u[3], u[4], u[5]),
            SIMD3<Float>(u[6], u[7], u[8])
        ))
        let Vt = simd_float3x3(columns: (
            SIMD3<Float>(vt[0], vt[1], vt[2]),
            SIMD3<Float>(vt[3], vt[4], vt[5]),
            SIMD3<Float>(vt[6], vt[7], vt[8])
        ))
        return (U, Vt)
    }

    // MARK: - Helpers

    private func downsample(_ points: [SIMD3<Float>], maxCount: Int) -> [SIMD3<Float>] {
        guard points.count > maxCount else { return points }
        let step = points.count / maxCount
        return Swift.stride(from: 0, to: points.count, by: step).map { points[$0] }
    }
}
