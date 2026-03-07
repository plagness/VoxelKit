import Foundation

/// Estimates metric scale for visual odometry using encoder displacement or LiDAR depth.
///
/// Visual odometry from essential matrix decomposition produces unit-scale translation.
/// ScaleEstimator maintains a running median of `encoder_displacement / vo_displacement`
/// to recover the metric scale factor.
public struct ScaleEstimator: Sendable {

    private var samples: [Float] = []
    private let windowSize: Int
    private let outlierLow: Float
    private let outlierHigh: Float
    private let minSamplesForConvergence: Int

    /// Current estimated scale factor (visual units → metres).
    public private(set) var scale: Float = 1.0

    /// Whether scale has converged (enough consistent samples).
    public var isConverged: Bool { samples.count >= minSamplesForConvergence }

    public init(windowSize: Int = 20, minSamplesForConvergence: Int = 5) {
        self.windowSize = windowSize
        self.minSamplesForConvergence = minSamplesForConvergence
        self.outlierLow = 0.33
        self.outlierHigh = 3.0
    }

    /// Add a new scale observation from encoder vs visual displacement.
    ///
    /// - Parameters:
    ///   - encoderDisplacement: Distance moved according to encoders (metres).
    ///   - voDisplacement: Distance moved according to visual odometry (arbitrary units).
    public mutating func addEncoderSample(encoderDisplacement: Float, voDisplacement: Float) {
        guard voDisplacement > 0.01 else { return } // ignore near-zero VO displacement
        let ratio = encoderDisplacement / voDisplacement
        addRatio(ratio)
    }

    /// Add a new scale observation from LiDAR depth vs visual parallax depth.
    ///
    /// - Parameters:
    ///   - lidarDepth: Measured depth from LiDAR (metres).
    ///   - visualDepth: Estimated depth from parallax (arbitrary units).
    public mutating func addDepthSample(lidarDepth: Float, visualDepth: Float) {
        guard visualDepth > 0.01 else { return }
        let ratio = lidarDepth / visualDepth
        addRatio(ratio)
    }

    private mutating func addRatio(_ ratio: Float) {
        // Outlier rejection based on current median
        if !samples.isEmpty {
            let currentMedian = scale
            if ratio < currentMedian * outlierLow || ratio > currentMedian * outlierHigh {
                return
            }
        }

        samples.append(ratio)
        if samples.count > windowSize {
            samples.removeFirst()
        }

        // Update scale to median of window
        let sorted = samples.sorted()
        scale = sorted[sorted.count / 2]
    }

    /// Reset all collected samples.
    public mutating func reset() {
        samples.removeAll()
        scale = 1.0
    }
}
