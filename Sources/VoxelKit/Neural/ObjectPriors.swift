import Foundation
import simd

/// Known real-world object dimensions for constraining detection-based AABBs.
///
/// When an object is detected by `ObjectDetector`, the depth-estimated bounding box
/// is blended with the prior's `typicalSize` — at longer distances the prior dominates
/// (sensor data is noisier), at short range the sensor dominates.
public struct ObjectPrior: Sendable {
    /// Class identifier (0-63), matching `PackedVoxel`'s 6-bit classId field.
    public let classId: UInt8
    /// Human-readable label (COCO class name).
    public let label: String
    /// Typical dimensions in meters: (width, height, depth).
    public let typicalSize: SIMD3<Float>
    /// How much the size can vary from typical (0 = exact, 1 = highly variable).
    public let sizeVariance: Float
    /// Default map layer for this object class.
    public let layer: MapLayer

    public init(classId: UInt8, label: String, typicalSize: SIMD3<Float>,
                sizeVariance: Float, layer: MapLayer) {
        self.classId = classId
        self.label = label
        self.typicalSize = typicalSize
        self.sizeVariance = sizeVariance
        self.layer = layer
    }

    // MARK: - Catalog

    /// Built-in catalog of common COCO object classes with metric size priors.
    public static let catalog: [ObjectPrior] = [
        // Dynamic (TTL-based expiry)
        ObjectPrior(classId: 1,  label: "person",       typicalSize: SIMD3(0.5, 1.7, 0.3),   sizeVariance: 0.3, layer: .dynamic),
        ObjectPrior(classId: 2,  label: "bicycle",      typicalSize: SIMD3(0.5, 1.0, 1.8),   sizeVariance: 0.2, layer: .dynamic),
        ObjectPrior(classId: 4,  label: "motorcycle",   typicalSize: SIMD3(0.8, 1.1, 2.2),   sizeVariance: 0.2, layer: .dynamic),
        ObjectPrior(classId: 7,  label: "dog",          typicalSize: SIMD3(0.3, 0.5, 0.7),   sizeVariance: 0.4, layer: .dynamic),
        ObjectPrior(classId: 8,  label: "cat",          typicalSize: SIMD3(0.2, 0.3, 0.4),   sizeVariance: 0.3, layer: .dynamic),

        // Structure (permanent)
        ObjectPrior(classId: 3,  label: "car",          typicalSize: SIMD3(1.8, 1.5, 4.5),   sizeVariance: 0.3, layer: .structure),
        ObjectPrior(classId: 5,  label: "bus",          typicalSize: SIMD3(2.5, 3.2, 12.0),  sizeVariance: 0.3, layer: .structure),
        ObjectPrior(classId: 6,  label: "truck",        typicalSize: SIMD3(2.5, 3.0, 8.0),   sizeVariance: 0.4, layer: .structure),
        ObjectPrior(classId: 15, label: "door",         typicalSize: SIMD3(0.8, 2.0, 0.1),   sizeVariance: 0.15, layer: .structure),
        ObjectPrior(classId: 16, label: "window",       typicalSize: SIMD3(1.0, 1.2, 0.1),   sizeVariance: 0.3, layer: .structure),
        ObjectPrior(classId: 17, label: "wall",         typicalSize: SIMD3(3.0, 2.8, 0.2),   sizeVariance: 0.5, layer: .structure),
        ObjectPrior(classId: 18, label: "fire hydrant",  typicalSize: SIMD3(0.3, 0.7, 0.3),   sizeVariance: 0.1, layer: .structure),
        ObjectPrior(classId: 19, label: "stop sign",    typicalSize: SIMD3(0.6, 0.6, 0.05),  sizeVariance: 0.1, layer: .structure),
        ObjectPrior(classId: 21, label: "parking meter", typicalSize: SIMD3(0.3, 1.2, 0.3),   sizeVariance: 0.1, layer: .structure),

        // Furniture (semi-permanent)
        ObjectPrior(classId: 9,  label: "chair",        typicalSize: SIMD3(0.5, 0.9, 0.5),   sizeVariance: 0.2, layer: .furniture),
        ObjectPrior(classId: 10, label: "dining table", typicalSize: SIMD3(1.2, 0.75, 0.8),  sizeVariance: 0.3, layer: .furniture),
        ObjectPrior(classId: 11, label: "couch",        typicalSize: SIMD3(2.0, 0.85, 0.9),  sizeVariance: 0.3, layer: .furniture),
        ObjectPrior(classId: 12, label: "bed",          typicalSize: SIMD3(1.6, 0.6, 2.0),   sizeVariance: 0.3, layer: .furniture),
        ObjectPrior(classId: 13, label: "potted plant", typicalSize: SIMD3(0.4, 0.6, 0.4),   sizeVariance: 0.5, layer: .furniture),
        ObjectPrior(classId: 14, label: "tv",           typicalSize: SIMD3(1.0, 0.6, 0.1),   sizeVariance: 0.4, layer: .furniture),
        ObjectPrior(classId: 20, label: "bench",        typicalSize: SIMD3(1.5, 0.8, 0.5),   sizeVariance: 0.2, layer: .furniture),
    ]

    // MARK: - Lookup

    private static let byLabel: [String: ObjectPrior] = {
        Dictionary(uniqueKeysWithValues: catalog.map { ($0.label, $0) })
    }()

    private static let byClassId: [UInt8: ObjectPrior] = {
        Dictionary(uniqueKeysWithValues: catalog.map { ($0.classId, $0) })
    }()

    /// Look up a prior by its COCO label.
    public static func lookup(_ label: String) -> ObjectPrior? {
        byLabel[label]
    }

    /// Look up a prior by classId.
    public static func lookup(classId: UInt8) -> ObjectPrior? {
        byClassId[classId]
    }

    /// Get the classId for a COCO label, or 0 (unknown) if not in catalog.
    public static func classId(for label: String) -> UInt8 {
        byLabel[label]?.classId ?? 0
    }

    // MARK: - AABB Blending

    /// Blend a depth-estimated AABB size with the prior's typical size.
    ///
    /// At short range (< 5m) the sensor dominates. At long range (> 50m) the prior dominates.
    /// The `sizeVariance` controls how quickly the prior takes over.
    ///
    /// - Parameters:
    ///   - depthSize: Size estimated from depth sensor (width, height, depth).
    ///   - distance: Distance from camera to object center.
    /// - Returns: Blended size in meters.
    public func blendSize(depthSize: SIMD3<Float>, distance: Float) -> SIMD3<Float> {
        // Prior weight increases with distance: 0 at ≤5m, up to (1-sizeVariance) at 50m
        let t = max(0, min((distance - 5.0) / 45.0, 1.0))
        let maxPriorWeight = 1.0 - sizeVariance
        let priorWeight = t * maxPriorWeight
        return depthSize * (1.0 - priorWeight) + typicalSize * priorWeight
    }
}
