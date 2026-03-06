import Foundation
import CoreML
import simd

/// Orchestrates depth estimation, object detection, and chunk refinement.
///
/// Manages the full neural pipeline:
/// 1. Depth estimation (V2 realtime / Pro refinement / hybrid)
/// 2. Object detection for distant solid region initialization
/// 3. Background refinement of low-quality chunks
///
/// Thread-safety: designed to be called from a single actor context.
public final class NeuralPipeline: @unchecked Sendable {

    public let depthEstimator: DepthEstimator
    public let objectDetector: ObjectDetector
    public let refinementQueue: RefinementQueue

    /// Current depth mode.
    public var depthMode: DepthMode {
        get { depthEstimator.mode }
        set { depthEstimator.mode = newValue }
    }

    /// Whether the pipeline is fully initialized (at least V2 depth loaded).
    public var isReady: Bool { depthEstimator.isV2Ready }

    /// Whether object detection is available.
    public var hasObjectDetection: Bool { objectDetector.isReady }

    /// Statistics.
    public private(set) var framesProcessed: Int = 0
    public private(set) var objectsDetected: Int = 0
    public private(set) var solidRegionsCreated: Int = 0

    /// Auto-mode tracking: frames since last new chunk discovery.
    private var framesSinceNewChunk: Int = 0
    private var lastChunkCount: Int = 0

    /// Auto-switch threshold: if no new chunks for N frames, switch to refine.
    public var autoSwitchThreshold: Int = 300 // ~10 seconds at 30fps

    public init(depthMode: DepthMode = .explore) {
        self.depthEstimator = DepthEstimator(mode: depthMode)
        self.objectDetector = ObjectDetector()
        self.refinementQueue = RefinementQueue()
    }

    // MARK: - Model Loading

    /// Load models from a directory containing .mlmodelc files.
    /// Expected files: "DepthAnythingV2Small.mlmodelc", "DepthPro.mlmodelc", "YOLOv8n.mlmodelc"
    public func loadModels(from directory: URL) throws {
        let v2URL = directory.appendingPathComponent("DepthAnythingV2Small.mlmodelc")
        if FileManager.default.fileExists(atPath: v2URL.path) {
            try depthEstimator.loadV2Model(from: v2URL)
        }

        let proURL = directory.appendingPathComponent("DepthPro.mlmodelc")
        if FileManager.default.fileExists(atPath: proURL.path) {
            try depthEstimator.loadProModel(from: proURL)
        }

        let yoloURL = directory.appendingPathComponent("YOLOv8n.mlmodelc")
        if FileManager.default.fileExists(atPath: yoloURL.path) {
            try objectDetector.loadModel(from: yoloURL)
        }
    }

    // MARK: - Frame Processing

    /// Process a camera frame through the neural pipeline.
    ///
    /// - Parameters:
    ///   - pixelBuffer: Camera frame (BGRA or NV12).
    ///   - intrinsics: Camera intrinsic matrix.
    ///   - extrinsics: Camera-to-world transform.
    ///   - world: The BotMapWorld to insert into.
    ///   - stride: Pixel sampling stride for depth back-projection.
    /// - Returns: Processing result with statistics.
    public func processFrame(
        pixelBuffer: CVPixelBuffer,
        intrinsics: simd_float3x3,
        extrinsics: simd_float4x4,
        world: BotMapWorld,
        pixelStride: Int = 4
    ) async -> NeuralFrameResult {
        var result = NeuralFrameResult()

        // 1. Depth estimation
        guard let depthEstimate = depthEstimator.estimateDepth(from: pixelBuffer) else {
            return result
        }
        result.hasDepth = true
        result.depthWidth = depthEstimate.width
        result.depthHeight = depthEstimate.height
        result.isMetricDepth = depthEstimate.isMetric

        // 2. Back-project to rays and carve
        let cameraPos = SIMD3<Float>(extrinsics[3][0], extrinsics[3][1], extrinsics[3][2])
        let rays = depthEstimator.backProject(
            depthEstimate,
            intrinsics: intrinsics,
            extrinsics: extrinsics,
            pixelStride: pixelStride
        )

        if !rays.isEmpty {
            let carveResult = await world.carveRays(rays, origin: cameraPos, maxDistance: 50.0)
            result.raysCarved = carveResult.carvedCount
            result.hitCount = carveResult.hitCount
        }

        // 3. Object detection (every N frames to save compute)
        if objectDetector.isReady && framesProcessed % 10 == 0 {
            let detections = objectDetector.detect(in: pixelBuffer)
            let projected = objectDetector.project(
                detections: detections,
                depthEstimate: depthEstimate,
                intrinsics: intrinsics,
                extrinsics: extrinsics
            )

            for det in projected {
                guard let aabb = det.worldAABB else { continue }
                await world.initializeSolidRegion(aabb)
                solidRegionsCreated += 1
                result.solidRegions += 1
            }
            objectsDetected += projected.count
            result.objectsDetected = projected.count
        }

        // 4. Auto-mode switching (hybrid only)
        if depthMode == .hybrid {
            let currentChunks = await world.chunkCount
            if currentChunks > lastChunkCount {
                framesSinceNewChunk = 0
                lastChunkCount = currentChunks
            } else {
                framesSinceNewChunk += 1
            }

            // If idle, process refinement queue
            if framesSinceNewChunk > autoSwitchThreshold {
                if let task = refinementQueue.dequeue() {
                    result.refinedChunk = task.chunkKey
                }
            }
        }

        framesProcessed += 1
        return result
    }

    // MARK: - Refinement

    /// Scan the world for chunks needing refinement and enqueue them.
    public func scanForRefinement(world: BotMapWorld) async {
        let store = await world.store
        refinementQueue.scanForRefinement(store: store)
    }
}

/// Result of processing a single frame through the neural pipeline.
public struct NeuralFrameResult: Sendable {
    public var hasDepth: Bool = false
    public var depthWidth: Int = 0
    public var depthHeight: Int = 0
    public var isMetricDepth: Bool = false
    public var raysCarved: Int = 0
    public var hitCount: Int = 0
    public var objectsDetected: Int = 0
    public var solidRegions: Int = 0
    public var refinedChunk: ChunkKey? = nil
}
