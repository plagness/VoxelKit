import Foundation
import CoreML
import CoreImage
import simd

/// Depth estimation mode controlling model selection and scheduling.
public enum DepthMode: String, Sendable, Codable {
    /// Depth Anything V2 Small (~30ms ANE) — realtime exploration.
    case explore
    /// Apple Depth Pro (~200ms) — metric accuracy for refinement.
    case refine
    /// V2 realtime + Pro in background queue for re-rendering visited chunks.
    case hybrid
}

/// Result of depth estimation from a single frame.
public struct DepthEstimate: Sendable {
    /// Depth map as Float array (row-major, top-left origin).
    public let depthMap: [Float]
    /// Width of the depth map.
    public let width: Int
    /// Height of the depth map.
    public let height: Int
    /// Whether this is metric depth (Depth Pro) or relative (V2).
    public let isMetric: Bool
    /// Scale factor to convert relative depth to meters (for V2, calibrated).
    public let metersPerUnit: Float

    /// Get depth at pixel coordinates.
    public func depth(atX x: Int, y: Int) -> Float {
        guard x >= 0, x < width, y >= 0, y < height else { return .nan }
        return depthMap[y * width + x] * metersPerUnit
    }
}

/// Neural Engine depth estimator supporting two models:
/// - Depth Anything V2 Small: fast relative depth (~30ms on ANE)
/// - Apple Depth Pro: accurate metric depth (~200ms)
///
/// CoreML models must be bundled or downloaded separately.
/// This class provides the inference API — model loading is the caller's responsibility.
public final class DepthEstimator: @unchecked Sendable {

    /// Current depth mode.
    public var mode: DepthMode

    /// V2 model (fast, relative depth).
    private var v2Model: MLModel?

    /// Depth Pro model (slow, metric depth).
    private var proModel: MLModel?

    /// V2 calibration: meters per unit (updated by known reference points).
    public var v2MetersPerUnit: Float = 1.0

    /// Expected input size for V2 model.
    public let v2InputSize: (width: Int, height: Int) = (518, 518)

    /// Expected input size for Depth Pro model.
    public let proInputSize: (width: Int, height: Int) = (1536, 1536)

    private let ciContext = CIContext()

    public init(mode: DepthMode = .explore) {
        self.mode = mode
    }

    // MARK: - Model Loading

    /// Load the Depth Anything V2 Small model from a compiled .mlmodelc URL.
    public func loadV2Model(from url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Prefer ANE
        v2Model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Load the Apple Depth Pro model from a compiled .mlmodelc URL.
    public func loadProModel(from url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        proModel = try MLModel(contentsOf: url, configuration: config)
    }

    /// Whether V2 model is loaded and ready.
    public var isV2Ready: Bool { v2Model != nil }

    /// Whether Depth Pro model is loaded and ready.
    public var isProReady: Bool { proModel != nil }

    // MARK: - Inference

    /// Estimate depth from a pixel buffer (camera frame).
    /// Uses the current `mode` to select model.
    /// Returns nil if the required model is not loaded.
    public func estimateDepth(from pixelBuffer: CVPixelBuffer) -> DepthEstimate? {
        switch mode {
        case .explore:
            return estimateV2(pixelBuffer)
        case .refine:
            return estimatePro(pixelBuffer) ?? estimateV2(pixelBuffer)
        case .hybrid:
            // In hybrid, primary path uses V2 for speed
            return estimateV2(pixelBuffer)
        }
    }

    /// Estimate depth using Depth Anything V2 (fast path).
    public func estimateV2(_ pixelBuffer: CVPixelBuffer) -> DepthEstimate? {
        guard let model = v2Model else { return nil }

        // Resize to model input size
        guard let resized = resizePixelBuffer(pixelBuffer,
                                               width: v2InputSize.width,
                                               height: v2InputSize.height) else { return nil }

        // Create MLFeatureProvider from pixel buffer
        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: resized)]
        ) else { return nil }

        guard let output = try? model.prediction(from: input) else { return nil }

        // Extract depth map from output
        guard let depthMultiArray = output.featureValue(for: "depth")?.multiArrayValue else { return nil }

        let count = depthMultiArray.count
        let ptr = depthMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        let depthMap = Array(UnsafeBufferPointer(start: ptr, count: count))

        let h = depthMultiArray.shape[depthMultiArray.shape.count - 2].intValue
        let w = depthMultiArray.shape[depthMultiArray.shape.count - 1].intValue

        return DepthEstimate(
            depthMap: depthMap,
            width: w, height: h,
            isMetric: false,
            metersPerUnit: v2MetersPerUnit
        )
    }

    /// Estimate depth using Apple Depth Pro (accurate path).
    public func estimatePro(_ pixelBuffer: CVPixelBuffer) -> DepthEstimate? {
        guard let model = proModel else { return nil }

        guard let resized = resizePixelBuffer(pixelBuffer,
                                               width: proInputSize.width,
                                               height: proInputSize.height) else { return nil }

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: resized)]
        ) else { return nil }

        guard let output = try? model.prediction(from: input) else { return nil }

        guard let depthMultiArray = output.featureValue(for: "depth")?.multiArrayValue else { return nil }

        let count = depthMultiArray.count
        let ptr = depthMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        let depthMap = Array(UnsafeBufferPointer(start: ptr, count: count))

        let h = depthMultiArray.shape[depthMultiArray.shape.count - 2].intValue
        let w = depthMultiArray.shape[depthMultiArray.shape.count - 1].intValue

        return DepthEstimate(
            depthMap: depthMap,
            width: w, height: h,
            isMetric: true,
            metersPerUnit: 1.0
        )
    }

    // MARK: - Calibration

    /// Calibrate V2's relative depth using a known metric reference.
    /// E.g., ground plane at known height from ARKit/LiDAR.
    public func calibrateV2(knownDistance: Float, estimatedValue: Float) {
        guard estimatedValue > 0 else { return }
        v2MetersPerUnit = knownDistance / estimatedValue
    }

    // MARK: - Back-projection

    /// Back-project a depth estimate to world-space 3D positions.
    /// - Parameters:
    ///   - estimate: The depth estimate.
    ///   - intrinsics: Camera intrinsic matrix (3×3: fx, fy, cx, cy).
    ///   - extrinsics: Camera-to-world transform (4×4).
    ///   - stride: Pixel sampling stride (higher = fewer points, faster).
    /// - Returns: Array of (direction, hitDistance, color) tuples for ray carving.
    public func backProject(
        _ estimate: DepthEstimate,
        intrinsics: simd_float3x3,
        extrinsics: simd_float4x4,
        pixelStride: Int = 4,
        colorSampler: ((Int, Int) -> (UInt8, UInt8, UInt8))? = nil
    ) -> [(direction: SIMD3<Float>, hitDistance: Float?, color: (UInt8, UInt8, UInt8))] {
        var rays: [(direction: SIMD3<Float>, hitDistance: Float?, color: (UInt8, UInt8, UInt8))] = []
        rays.reserveCapacity((estimate.width / pixelStride) * (estimate.height / pixelStride))

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        let rotation = simd_float3x3(
            SIMD3<Float>(extrinsics[0][0], extrinsics[0][1], extrinsics[0][2]),
            SIMD3<Float>(extrinsics[1][0], extrinsics[1][1], extrinsics[1][2]),
            SIMD3<Float>(extrinsics[2][0], extrinsics[2][1], extrinsics[2][2])
        )

        for py in Swift.stride(from: 0, to: estimate.height, by: pixelStride) {
            for px in Swift.stride(from: 0, to: estimate.width, by: pixelStride) {
                let depth = estimate.depth(atX: px, y: py)
                guard depth > 0.1, depth < 100.0 else { continue }

                // Camera-space ray direction
                let camDir = SIMD3<Float>(
                    (Float(px) - cx) / fx,
                    (Float(py) - cy) / fy,
                    1.0
                )
                let worldDir = simd_normalize(rotation * camDir)

                let color = colorSampler?(px, py) ?? (128, 128, 128)
                rays.append((direction: worldDir, hitDistance: depth, color: color))
            }
        }

        return rays
    }

    // MARK: - Private

    private func resizePixelBuffer(_ buffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let scaleX = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(buffer))
        let scaleY = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(buffer))
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var output: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &output)
        guard let outBuffer = output else { return nil }
        ciContext.render(scaled, to: outBuffer)
        return outBuffer
    }
}
