import Foundation
import CoreML
import CoreImage
import simd

/// Detected object with 2D bounding box and estimated 3D position.
public struct DetectedObject: Sendable {
    /// Object class label (e.g., "car", "person", "building").
    public let label: String
    /// Detection confidence (0-1).
    public let confidence: Float
    /// 2D bounding box in normalized image coordinates (0-1).
    public let bbox: (x: Float, y: Float, width: Float, height: Float)
    /// Estimated 3D bounding box in world space (if depth available).
    public var worldAABB: AABB?
    /// Estimated distance from camera.
    public var estimatedDistance: Float?
    /// Object class identifier (0-63), from ObjectPrior catalog. 0 = unknown.
    public var classId: UInt8 = 0
    /// Map layer for this object (structure/furniture/dynamic).
    public var layer: MapLayer = .structure
}

/// Object detection for creating initial solid volumes in the subtractive pipeline.
///
/// Distant objects detected in camera frames → 3D bounding boxes → `initializeSolidRegion()`.
/// Uses CoreML YOLO or similar model on ANE.
public final class ObjectDetector: @unchecked Sendable {

    private var model: MLModel?
    private let ciContext = CIContext()

    /// Minimum confidence threshold for detections.
    public var confidenceThreshold: Float = 0.5

    /// Minimum distance (m) for creating solid regions.
    /// Nearby objects don't need solid initialization — direct scanning suffices.
    public var minDistance: Float = 5.0

    /// Maximum distance (m) for solid region creation.
    public var maxDistance: Float = 200.0

    /// Model input size.
    public let inputSize: (width: Int, height: Int) = (640, 640)

    /// COCO class labels (subset — common outdoor objects for robotic mapping).
    public static let outdoorLabels: Set<String> = [
        "car", "truck", "bus", "motorcycle", "bicycle",
        "person", "dog", "cat",
        "bench", "fire hydrant", "stop sign", "parking meter",
        "potted plant", "chair", "dining table",
    ]

    public init() {}

    // MARK: - Model Loading

    /// Load a compiled CoreML detection model (.mlmodelc).
    public func loadModel(from url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: config)
    }

    /// Whether the model is loaded.
    public var isReady: Bool { model != nil }

    // MARK: - Detection

    /// Detect objects in a camera frame.
    /// Returns detected objects with 2D bounding boxes.
    public func detect(in pixelBuffer: CVPixelBuffer) -> [DetectedObject] {
        guard let model = model else { return [] }

        guard let resized = resizePixelBuffer(pixelBuffer,
                                               width: inputSize.width,
                                               height: inputSize.height) else { return [] }

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: resized)]
        ) else { return [] }

        guard let output = try? model.prediction(from: input) else { return [] }

        return parseDetections(output)
    }

    // MARK: - 3D Projection

    /// Project 2D detections to 3D world-space bounding boxes using depth estimates.
    ///
    /// For each detection, estimates a 3D AABB using:
    /// 1. Median depth within the bbox from the depth estimate
    /// 2. Camera intrinsics for size projection
    /// 3. Camera extrinsics for world-space transform
    public func project(detections: [DetectedObject],
                        depthEstimate: DepthEstimate,
                        intrinsics: simd_float3x3,
                        extrinsics: simd_float4x4) -> [DetectedObject] {
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        let cameraPos = SIMD3<Float>(extrinsics[3][0], extrinsics[3][1], extrinsics[3][2])
        let rotation = simd_float3x3(
            SIMD3<Float>(extrinsics[0][0], extrinsics[0][1], extrinsics[0][2]),
            SIMD3<Float>(extrinsics[1][0], extrinsics[1][1], extrinsics[1][2]),
            SIMD3<Float>(extrinsics[2][0], extrinsics[2][1], extrinsics[2][2])
        )

        return detections.compactMap { det in
            // Sample depth within bbox
            let bboxPixelX = Int(det.bbox.x * Float(depthEstimate.width))
            let bboxPixelY = Int(det.bbox.y * Float(depthEstimate.height))
            let bboxW = Int(det.bbox.width * Float(depthEstimate.width))
            let bboxH = Int(det.bbox.height * Float(depthEstimate.height))

            var depths: [Float] = []
            let step = max(1, min(bboxW, bboxH) / 5)
            for dy in stride(from: 0, to: bboxH, by: step) {
                for dx in stride(from: 0, to: bboxW, by: step) {
                    let d = depthEstimate.depth(atX: bboxPixelX + dx, y: bboxPixelY + dy)
                    if d > 0.1 && d < maxDistance { depths.append(d) }
                }
            }

            guard !depths.isEmpty else { return nil }
            depths.sort()
            let medianDepth = depths[depths.count / 2]

            guard medianDepth >= minDistance && medianDepth <= maxDistance else { return nil }

            // Project bbox corners to world space
            let centerPixelX = Float(bboxPixelX) + Float(bboxW) * 0.5
            let centerPixelY = Float(bboxPixelY) + Float(bboxH) * 0.5

            let camCenter = SIMD3<Float>(
                (centerPixelX - cx) / fx * medianDepth,
                (centerPixelY - cy) / fy * medianDepth,
                medianDepth
            )
            let worldCenter = cameraPos + rotation * camCenter

            // Estimate world-space size from pixel bbox + depth
            var sizeW = (Float(bboxW) / fx) * medianDepth
            var sizeH = (Float(bboxH) / fy) * medianDepth
            var sizeD = min(sizeW, sizeH) // assume roughly cubic depth

            var det = det
            det.estimatedDistance = medianDepth

            // Apply object size priors — blend with known dimensions
            if let prior = ObjectPrior.lookup(det.label) {
                det.classId = prior.classId
                det.layer = prior.layer
                let depthSize = SIMD3<Float>(sizeW, sizeH, sizeD)
                let blended = prior.blendSize(depthSize: depthSize, distance: medianDepth)
                sizeW = blended.x
                sizeH = blended.y
                sizeD = blended.z
            } else {
                det.classId = ObjectPrior.classId(for: det.label)
            }

            let halfSize = SIMD3<Float>(sizeW, sizeH, sizeD) * 0.5
            det.worldAABB = AABB(
                min: worldCenter - halfSize,
                max: worldCenter + halfSize
            )
            return det
        }
    }

    // MARK: - Private

    private func parseDetections(_ output: MLFeatureProvider) -> [DetectedObject] {
        // Generic parsing — actual field names depend on the specific model
        // Common patterns: "coordinates", "confidence", "classLabel"
        // This is a best-effort parser for YOLO-style outputs

        var detections: [DetectedObject] = []

        // Try Vision-style output (MLFeatureValue with multiarray)
        if let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue,
           let confidence = output.featureValue(for: "confidence")?.multiArrayValue {
            let count = coordinates.shape[0].intValue
            let numClasses = confidence.shape.count > 1 ? confidence.shape[1].intValue : 1

            let coordPtr = coordinates.dataPointer.bindMemory(to: Float.self, capacity: count * 4)
            let confPtr = confidence.dataPointer.bindMemory(to: Float.self, capacity: count * numClasses)

            for i in 0..<count {
                // Find best class
                var bestClass = 0
                var bestConf: Float = 0
                for c in 0..<numClasses {
                    let conf = confPtr[i * numClasses + c]
                    if conf > bestConf { bestConf = conf; bestClass = c }
                }

                guard bestConf >= confidenceThreshold else { continue }

                let cx = coordPtr[i * 4]
                let cy = coordPtr[i * 4 + 1]
                let w = coordPtr[i * 4 + 2]
                let h = coordPtr[i * 4 + 3]

                let label = "class_\(bestClass)"
                let prior = ObjectPrior.lookup(label)
                detections.append(DetectedObject(
                    label: label,
                    confidence: bestConf,
                    bbox: (x: cx - w * 0.5, y: cy - h * 0.5, width: w, height: h),
                    classId: prior?.classId ?? 0,
                    layer: prior?.layer ?? .structure
                ))
            }
        }

        return detections
    }

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
