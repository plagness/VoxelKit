import Foundation
import Accelerate
import simd
import CoreVideo

/// Essential matrix estimation via 8-point algorithm + RANSAC.
///
/// Given dense optical flow, samples sparse correspondences in normalized
/// camera coordinates, estimates the essential matrix E, and decomposes it
/// into camera rotation R and translation direction t.
///
/// References:
/// - Hartley & Zisserman, "Multiple View Geometry", Chapter 11
/// - Nistér, "An efficient solution to the five-point relative pose problem" (2004)
enum EssentialMatrixEstimator {

    struct Correspondence {
        let p1: SIMD2<Float>  // normalized camera coords in frame 1 (keyframe)
        let p2: SIMD2<Float>  // normalized camera coords in frame 2 (current)
    }

    // MARK: - Public API

    /// Estimate relative camera motion from optical flow.
    ///
    /// - Parameters:
    ///   - flowBuffer: Dense optical flow (VNGenerateOpticalFlowRequest output).
    ///     Convention: at pixel (x,y) in current frame, (dx,dy) points to keyframe.
    ///   - fx, fy, cx, cy: Camera intrinsics at VIDEO resolution.
    ///   - videoW, videoH: Video dimensions.
    /// - Returns: (R, t) where R is rotation from keyframe→current, t is unit translation.
    static func estimateMotion(
        flowBuffer: CVPixelBuffer,
        fx: Float, fy: Float, cx: Float, cy: Float,
        videoW: Int, videoH: Int
    ) -> (R: simd_float3x3, t: SIMD3<Float>)? {
        let correspondences = sampleCorrespondences(
            flowBuffer: flowBuffer,
            fx: fx, fy: fy, cx: cx, cy: cy,
            videoW: videoW, videoH: videoH
        )
        guard correspondences.count >= 20 else { return nil }

        guard let E = ransac(correspondences: correspondences) else { return nil }
        return decompose(E, correspondences: correspondences)
    }

    // MARK: - Correspondence sampling

    static func sampleCorrespondences(
        flowBuffer: CVPixelBuffer,
        fx: Float, fy: Float, cx: Float, cy: Float,
        videoW: Int, videoH: Int,
        maxCount: Int = 300,
        minFlowMag: Float = 0.2
    ) -> [Correspondence] {
        CVPixelBufferLockBaseAddress(flowBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowBuffer, .readOnly) }

        guard let addr = CVPixelBufferGetBaseAddress(flowBuffer) else { return [] }
        let flowW = CVPixelBufferGetWidth(flowBuffer)
        let flowH = CVPixelBufferGetHeight(flowBuffer)
        let bpr = CVPixelBufferGetBytesPerRow(flowBuffer)
        let ptr = addr.bindMemory(to: Float.self, capacity: flowH * bpr / 4)

        let scaleX = Float(videoW) / Float(flowW)
        let scaleY = Float(videoH) / Float(flowH)

        let totalPixels = flowW * flowH
        let step = max(1, Int(sqrt(Double(totalPixels / maxCount))))

        var result: [Correspondence] = []
        result.reserveCapacity(maxCount)

        for fy_idx in stride(from: step / 2, to: flowH, by: step) {
            for fx_idx in stride(from: step / 2, to: flowW, by: step) {
                let idx = fy_idx * (bpr / 4) + fx_idx * 2
                let dx = ptr[idx]
                let dy = ptr[idx + 1]
                let mag = sqrt(dx * dx + dy * dy)
                guard mag >= minFlowMag, mag < 100 else { continue }

                // Current-frame pixel (video coords)
                let u2 = Float(fx_idx) * scaleX + scaleX * 0.5
                let v2 = Float(fy_idx) * scaleY + scaleY * 0.5

                // Keyframe pixel (flow points current → keyframe)
                let u1 = u2 + dx * scaleX
                let v1 = v2 + dy * scaleY

                guard u1 >= 0, u1 < Float(videoW), v1 >= 0, v1 < Float(videoH) else { continue }

                // Normalize to camera coordinates
                let p1 = SIMD2<Float>((u1 - cx) / fx, (v1 - cy) / fy)
                let p2 = SIMD2<Float>((u2 - cx) / fx, (v2 - cy) / fy)

                result.append(Correspondence(p1: p1, p2: p2))
            }
        }
        return result
    }

    // MARK: - RANSAC

    static func ransac(
        correspondences: [Correspondence],
        iterations: Int = 200,
        inlierThreshold: Float = 0.005
    ) -> simd_float3x3? {
        let n = correspondences.count
        guard n >= 8 else { return nil }

        var bestE: simd_float3x3? = nil
        var bestInliers = 0

        for _ in 0..<iterations {
            // Random 8-point sample
            var sample = [Correspondence]()
            var indices = Set<Int>()
            while sample.count < 8 {
                let idx = Int.random(in: 0..<n)
                if indices.insert(idx).inserted {
                    sample.append(correspondences[idx])
                }
            }

            guard let E = eightPoint(sample) else { continue }

            // Count inliers: |x2^T * E * x1| < threshold
            var inliers = 0
            for c in correspondences {
                let x1 = SIMD3<Float>(c.p1.x, c.p1.y, 1)
                let x2 = SIMD3<Float>(c.p2.x, c.p2.y, 1)
                let err = abs(simd_dot(x2, E * x1))
                if err < inlierThreshold { inliers += 1 }
            }

            if inliers > bestInliers {
                bestInliers = inliers
                bestE = E
            }
        }

        // Refit with all inliers of best model
        guard let bestModel = bestE else { return nil }
        var inlierSet = [Correspondence]()
        for c in correspondences {
            let x1 = SIMD3<Float>(c.p1.x, c.p1.y, 1)
            let x2 = SIMD3<Float>(c.p2.x, c.p2.y, 1)
            if abs(simd_dot(x2, bestModel * x1)) < inlierThreshold * 2 {
                inlierSet.append(c)
            }
        }

        if inlierSet.count >= 8 {
            return eightPoint(inlierSet) ?? bestModel
        }
        return bestModel
    }

    // MARK: - 8-Point Algorithm

    /// Estimate essential matrix from ≥8 normalized correspondences.
    static func eightPoint(_ correspondences: [Correspondence]) -> simd_float3x3? {
        let n = correspondences.count
        guard n >= 8 else { return nil }

        // Build constraint matrix A (n × 9), column-major.
        // Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        // Constraint: vec(E)^T * row = 0
        var A = [Float](repeating: 0, count: n * 9)
        for i in 0..<n {
            let x1 = correspondences[i].p1.x, y1 = correspondences[i].p1.y
            let x2 = correspondences[i].p2.x, y2 = correspondences[i].p2.y
            // Column-major: A[row + col * n]
            A[i + 0 * n] = x2 * x1
            A[i + 1 * n] = x2 * y1
            A[i + 2 * n] = x2
            A[i + 3 * n] = y2 * x1
            A[i + 4 * n] = y2 * y1
            A[i + 5 * n] = y2
            A[i + 6 * n] = x1
            A[i + 7 * n] = y1
            A[i + 8 * n] = 1
        }

        // Null vector of A = eigenvector of A^T*A with smallest eigenvalue
        guard let e = nullVectorOfATA(A: A, m: n, n: 9) else { return nil }
        guard e.count == 9 else { return nil }

        // e is row-major vectorization of E: [E00, E01, E02, E10, E11, E12, E20, E21, E22]
        // Enforce rank-2 constraint via SVD
        return enforceRank2(e)
    }

    // MARK: - Essential Matrix Decomposition

    /// Decompose E into (R, t) with chirality check.
    ///
    /// Returns the unique (R, t) where most triangulated points have positive
    /// depth in both cameras. t is a unit vector (scale is unknown).
    static func decompose(
        _ E: simd_float3x3,
        correspondences: [Correspondence]
    ) -> (R: simd_float3x3, t: SIMD3<Float>)? {
        guard let svdResult = svd3x3(E) else { return nil }

        let U = svdResult.U
        let Vt = svdResult.Vt

        // W matrix for rotation extraction
        // W rotates 90° around z-axis
        let W = simd_float3x3(
            SIMD3<Float>( 0, 1, 0),
            SIMD3<Float>(-1, 0, 0),
            SIMD3<Float>( 0, 0, 1)
        )

        // Two candidate rotations
        var R1 = U * W * Vt
        var R2 = U * W.transpose * Vt

        // Ensure proper rotation (det = +1)
        if R1.determinant < 0 { R1 = R1 * -1 }
        if R2.determinant < 0 { R2 = R2 * -1 }

        // Translation = ± third column of U
        let t1 = SIMD3<Float>(U.columns.2.x, U.columns.2.y, U.columns.2.z)
        let t2 = -t1

        // Four candidates — pick by chirality (most points in front of both cameras)
        let candidates: [(simd_float3x3, SIMD3<Float>)] = [
            (R1, t1), (R1, t2), (R2, t1), (R2, t2)
        ]

        let sampleCount = min(50, correspondences.count)
        let sampleStep = max(1, correspondences.count / sampleCount)

        var bestScore = 0
        var bestIdx = 0

        for (ci, (R, t)) in candidates.enumerated() {
            var score = 0
            for i in stride(from: 0, to: correspondences.count, by: sampleStep) {
                let c = correspondences[i]
                if chiralityCheck(R: R, t: t, p1: c.p1, p2: c.p2) {
                    score += 1
                }
            }
            if score > bestScore {
                bestScore = score
                bestIdx = ci
            }
        }

        // Require at least 25% of samples to pass chirality
        guard bestScore > sampleCount / 4 else { return nil }
        return candidates[bestIdx]
    }

    // MARK: - Chirality Check

    /// Check if a point pair is in front of both cameras.
    ///
    /// Camera 1: [I | 0], Camera 2: [R | t]
    private static func chiralityCheck(
        R: simd_float3x3,
        t: SIMD3<Float>,
        p1: SIMD2<Float>,
        p2: SIMD2<Float>
    ) -> Bool {
        let d1 = SIMD3<Float>(p1.x, p1.y, 1)
        let d2_cam2 = SIMD3<Float>(p2.x, p2.y, 1)
        let d2_cam1 = R.transpose * d2_cam2  // ray in camera 1 frame

        // Triangulate: X = s * d1 = t + u * R^T * d2_cam2
        // s * d1 - u * d2_cam1 = t
        // Solve: [d1 | -d2_cam1] * [s; u] = t
        let a = simd_dot(d1, d1)
        let b = -simd_dot(d1, d2_cam1)
        let c = simd_dot(d2_cam1, d2_cam1)
        let d = simd_dot(d1, t)
        let e = -simd_dot(d2_cam1, t)
        let det = a * c - b * b
        guard abs(det) > 1e-8 else { return false }

        let s = (c * d - b * e) / det  // depth in camera 1
        let u = (a * e - b * d) / det  // depth in camera 2

        return s > 0 && u > 0
    }

    // MARK: - Linear Algebra

    /// Find null vector of A via eigendecomposition of A^T*A.
    ///
    /// Returns the eigenvector corresponding to the smallest eigenvalue of A^T*A.
    /// More efficient than full SVD when m >> n.
    private static func nullVectorOfATA(A: [Float], m: Int, n: Int) -> [Float]? {
        // B = A^T * A (n×n symmetric)
        var B = [Float](repeating: 0, count: n * n)
        var aCopy = A
        var alpha: Float = 1.0, beta: Float = 0.0
        var m_ = Int32(m), n_ = Int32(n)
        var n2 = Int32(n), m2 = Int32(m), m3 = Int32(m), n3 = Int32(n)
        var aCopy2 = aCopy
        var transA = Int8(UInt8(ascii: "T"))
        var transN = Int8(UInt8(ascii: "N"))
        sgemm_(&transA, &transN, &n_, &n2, &m_, &alpha, &aCopy, &m2, &aCopy2, &m3, &beta, &B, &n3)

        // Eigendecompose B
        var jobz = Int8(UInt8(ascii: "V"))
        var uplo = Int8(UInt8(ascii: "U"))
        var nn = Int32(n)
        var nn2 = Int32(n)
        var eigenvalues = [Float](repeating: 0, count: n)
        var work = [Float](repeating: 0, count: 1)
        var lwork = Int32(-1)
        var info = Int32(0)

        // Query work size
        ssyev_(&jobz, &uplo, &nn, &B, &nn2, &eigenvalues, &work, &lwork, &info)
        guard info == 0 else { return nil }

        lwork = Int32(work[0])
        work = [Float](repeating: 0, count: Int(lwork))

        // Compute
        nn = Int32(n); nn2 = Int32(n)
        ssyev_(&jobz, &uplo, &nn, &B, &nn2, &eigenvalues, &work, &lwork, &info)
        guard info == 0 else { return nil }

        // ssyev returns eigenvalues ascending → first column of B = smallest eigenvector
        return Array(B[0..<n])
    }

    /// Enforce rank-2 constraint: set smallest singular value to 0.
    private static func enforceRank2(_ e: [Float]) -> simd_float3x3? {
        // e is row-major: [E00, E01, E02, E10, E11, E12, E20, E21, E22]
        // Convert to simd_float3x3 (column-major)
        let E = simd_float3x3(
            SIMD3<Float>(e[0], e[3], e[6]),  // column 0
            SIMD3<Float>(e[1], e[4], e[7]),  // column 1
            SIMD3<Float>(e[2], e[5], e[8])   // column 2
        )

        guard let svdResult = svd3x3(E) else { return nil }

        // Average the two larger singular values, zero out the third
        let avg = (svdResult.S.x + svdResult.S.y) / 2.0
        let S = simd_float3x3(diagonal: SIMD3<Float>(avg, avg, 0))

        return svdResult.U * S * svdResult.Vt
    }

    /// SVD of a 3×3 matrix using LAPACK.
    private static func svd3x3(_ M: simd_float3x3) -> (U: simd_float3x3, S: SIMD3<Float>, Vt: simd_float3x3)? {
        // Column-major storage
        var a: [Float] = [
            M.columns.0.x, M.columns.0.y, M.columns.0.z,
            M.columns.1.x, M.columns.1.y, M.columns.1.z,
            M.columns.2.x, M.columns.2.y, M.columns.2.z
        ]

        var m: Int32 = 3
        var n: Int32 = 3
        var lda: Int32 = 3
        var s = [Float](repeating: 0, count: 3)
        var u = [Float](repeating: 0, count: 9)
        var ldu: Int32 = 3
        var vt = [Float](repeating: 0, count: 9)
        var ldvt: Int32 = 3
        var work = [Float](repeating: 0, count: 1)
        var lwork: Int32 = -1
        var iwork = [Int32](repeating: 0, count: 24)
        var info: Int32 = 0
        var jobz = Int8(UInt8(ascii: "A"))

        // Query
        sgesdd_(&jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &iwork, &info)
        guard info == 0 else { return nil }

        lwork = Int32(work[0])
        work = [Float](repeating: 0, count: Int(lwork))

        // Compute
        sgesdd_(&jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt,
                &work, &lwork, &iwork, &info)
        guard info == 0 else { return nil }

        let U = simd_float3x3(
            SIMD3<Float>(u[0], u[1], u[2]),
            SIMD3<Float>(u[3], u[4], u[5]),
            SIMD3<Float>(u[6], u[7], u[8])
        )
        let Vt = simd_float3x3(
            SIMD3<Float>(vt[0], vt[1], vt[2]),
            SIMD3<Float>(vt[3], vt[4], vt[5]),
            SIMD3<Float>(vt[6], vt[7], vt[8])
        )

        return (U, SIMD3<Float>(s[0], s[1], s[2]), Vt)
    }
}
