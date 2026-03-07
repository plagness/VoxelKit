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

    /// Run object detection every N frames (1 = every frame, 10 = every 10th).
    public var detectionFrequency: Int = 10

    public init(depthMode: DepthMode = .explore) {
        self.depthEstimator = DepthEstimator(mode: depthMode)
        self.objectDetector = ObjectDetector()
        self.refinementQueue = RefinementQueue()
    }

    // MARK: - Model Loading

    /// Load models from a directory containing .mlmodelc or .mlpackage files.
    /// Expected files: "DepthAnythingV2Small", "DepthPro", "YOLOv8n"
    public func loadModels(from directory: URL) throws {
        if let url = Self.findModel(named: "DepthAnythingV2Small", in: directory) {
            try depthEstimator.loadV2Model(from: url)
        }

        if let url = Self.findModel(named: "DepthPro", in: directory) {
            try depthEstimator.loadProModel(from: url)
        }

        if let url = Self.findModel(named: "YOLOv8n", in: directory) {
            try objectDetector.loadModel(from: url)
        }
    }

    /// Find a CoreML model by name, checking .mlmodelc first, then .mlpackage.
    private static func findModel(named name: String, in directory: URL) -> URL? {
        let fm = FileManager.default
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        if fm.fileExists(atPath: compiled.path) { return compiled }
        let package = directory.appendingPathComponent("\(name).mlpackage")
        if fm.fileExists(atPath: package.path) { return package }
        return nil
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
        if objectDetector.isReady && framesProcessed % max(1, detectionFrequency) == 0 {
            let detections = objectDetector.detect(in: pixelBuffer)
            let projected = objectDetector.project(
                detections: detections,
                depthEstimate: depthEstimate,
                intrinsics: intrinsics,
                extrinsics: extrinsics
            )

            for det in projected {
                guard let aabb = det.worldAABB else { continue }
                await world.initializeSolidRegion(aabb, classId: det.classId, layer: det.layer)
                refinementQueue.enqueueRegion(aabb, priority: .newDetection, store: world.store)
                solidRegionsCreated += 1
                result.solidRegions += 1
            }
            objectsDetected += projected.count
            result.objectsDetected = projected.count
            result.detectedObjects = projected
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
    public func scanForRefinement(world: BotMapWorld) {
        refinementQueue.scanForRefinement(store: world.store)
    }

    /// Whether there are refinement tasks available.
    public var hasRefinementWork: Bool { !refinementQueue.isEmpty }

    /// Process the next refinement task: re-estimate depth for a chunk
    /// using Depth Pro (if available) and update the world.
    ///
    /// Returns the chunk key that was refined, or nil if nothing to do.
    public func processNextRefinement(
        world: BotMapWorld,
        lastPixelBuffer: CVPixelBuffer?,
        intrinsics: simd_float3x3,
        extrinsics: simd_float4x4
    ) async -> ChunkKey? {
        guard depthEstimator.isProReady else { return nil }
        guard let task = refinementQueue.dequeue() else { return nil }
        guard let pixelBuffer = lastPixelBuffer else { return nil }

        // Use Depth Pro for high-quality metric depth
        guard let estimate = depthEstimator.estimatePro(pixelBuffer) else { return nil }

        // Back-project with larger stride (refinement = quality, not speed)
        let cameraPos = SIMD3<Float>(extrinsics[3][0], extrinsics[3][1], extrinsics[3][2])
        let rays = depthEstimator.backProject(
            estimate,
            intrinsics: intrinsics,
            extrinsics: extrinsics,
            pixelStride: 2  // finer stride for refinement
        )

        if !rays.isEmpty {
            await world.carveRays(rays, origin: cameraPos, maxDistance: 50.0)
        }

        return task.chunkKey
    }

    // MARK: - Background Scheduling

    /// Start a background refinement loop that processes chunks when idle.
    ///
    /// This task runs indefinitely until cancelled. It:
    /// 1. Periodically scans for low-quality chunks
    /// 2. Processes them through Depth Pro when available
    /// 3. Sleeps when no work is available or when in explore mode
    ///
    /// - Parameters:
    ///   - world: The world to refine.
    ///   - interval: Seconds between scan/process cycles.
    ///   - frameProvider: Closure that returns the latest camera frame data, or nil if unavailable.
    /// - Returns: A Task that can be cancelled to stop the loop.
    @discardableResult
    public func startBackgroundRefinement(
        world: BotMapWorld,
        interval: TimeInterval = 5.0,
        frameProvider: @escaping @Sendable () async -> (pixelBuffer: CVPixelBuffer, intrinsics: simd_float3x3, extrinsics: simd_float4x4)?
    ) -> Task<Void, Never> {
        Task.detached(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }

                // Only refine in hybrid or refine mode
                guard self.depthMode == .hybrid || self.depthMode == .refine else {
                    try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                    continue
                }

                // Scan for new refinement targets periodically
                if self.refinementQueue.isEmpty {
                    self.scanForRefinement(world: world)
                }

                // Process next chunk if we have frame data
                if self.hasRefinementWork, let frame = await frameProvider() {
                    let _ = await self.processNextRefinement(
                        world: world,
                        lastPixelBuffer: frame.pixelBuffer,
                        intrinsics: frame.intrinsics,
                        extrinsics: frame.extrinsics
                    )
                    // Short pause between refinements to avoid starving the main pipeline
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s
                } else {
                    // No work or no frames — sleep longer
                    try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                }
            }
        }
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
    public var detectedObjects: [DetectedObject] = []
}
