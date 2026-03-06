import Foundation
import simd

/// Primary container for a spatial map built from capture sessions.
///
/// An actor that owns a `ChunkedOctreeStore` and provides the single point
/// of insertion for voxel data arriving from `VideoCaptureSession` or LiDAR.
/// Thread-safe: all mutations go through the actor.
/// Pipeline mode: additive (classic) or subtractive (sculpting).
public enum PipelineMode: String, Sendable, Codable {
    /// Classic: add voxels one-by-one from sensor data.
    case additive
    /// Sculpting: start with solid volumes, carve free space.
    case subtractive
}

public actor BotMapWorld {

    // MARK: - Public state

    /// Underlying voxel store (access only from actor context).
    public let store: ChunkedOctreeStore

    /// Pipeline mode: additive (classic) or subtractive (sculpting).
    public private(set) var pipelineMode: PipelineMode = .additive

    /// Display name for this map (e.g. filename without extension).
    public var name: String

    /// When the map was created.
    public let createdAt: Date

    /// Total number of occupied voxels across all chunks.
    public var voxelCount: Int { store.voxelCount }

    /// Number of 1m³ chunks.
    public var chunkCount: Int { store.chunkCount }

    /// World-space bounding box of all occupied voxels.
    public var boundingBox: AABB { store.boundingBox }

    // MARK: - Init

    public init(name: String = "Untitled") {
        self.store = ChunkedOctreeStore()
        self.name = name
        self.createdAt = .now
    }

    // Internal init for deserialization (see BotMapSerializer).
    init(store: ChunkedOctreeStore, name: String, createdAt: Date) {
        self.store = store
        self.name = name
        self.createdAt = createdAt
    }

    // MARK: - Insertion

    /// Insert a batch of world-space occupied positions (from depth back-projection).
    ///
    /// Called by `VoxelInserter` after converting a depth map + pose to world positions.
    public func insertVoxelBatch(_ positions: [SIMD3<Float>], robotIndex: Int = 0) {
        store.insertPositions(positions, robotIndex: robotIndex)
    }

    /// Insert a batch of colored voxels (from camera-sampled depth back-projection).
    public func insertColoredVoxelBatch(_ voxels: [ColoredPosition], robotIndex: Int = 0) {
        store.insertColoredPositions(voxels, robotIndex: robotIndex)
    }

    /// Insert a decoded LiDAR frame (for LiDAR-based capture sessions).
    public func insertLiDARFrame(_ frame: LiDARFrame, robotIndex: Int = 0) {
        store.insertFrame(frame, robotIndex: robotIndex)
    }

    /// Set the pipeline mode.
    public func setPipelineMode(_ mode: PipelineMode) {
        pipelineMode = mode
    }

    // MARK: - Subtractive Insertion

    /// Insert a subtractive observation at LOD-appropriate depth.
    /// Hits mark surface, misses carve free space.
    public func insertSubtractive(at pos: SIMD3<Float>, hit: Bool,
                                   cameraPos: SIMD3<Float>,
                                   robotIndex: Int = 0,
                                   color: (UInt8, UInt8, UInt8) = (128, 128, 128)) {
        store.insertSubtractive(at: pos, hit: hit, cameraPos: cameraPos,
                                robotIndex: robotIndex, color: color)
    }

    /// Carve a ray through the world: free space before hitDistance, surface at hitDistance.
    @discardableResult
    public func carveRay(origin: SIMD3<Float>, direction: SIMD3<Float>,
                          hitDistance: Float?, maxDistance: Float = 10.0,
                          robotIndex: Int = 0,
                          color: (UInt8, UInt8, UInt8) = (128, 128, 128)) -> RayCarveResult {
        OctreeRayCaster.carveRay(origin: origin, direction: direction,
                                  hitDistance: hitDistance, maxDistance: maxDistance,
                                  store: store, cameraPos: origin,
                                  robotIndex: robotIndex, color: color)
    }

    /// Batch carve multiple rays (from depth back-projection).
    @discardableResult
    public func carveRays(
        _ rays: [(direction: SIMD3<Float>, hitDistance: Float?, color: (UInt8, UInt8, UInt8))],
        origin: SIMD3<Float>, maxDistance: Float = 10.0,
        robotIndex: Int = 0
    ) -> RayCarveResult {
        OctreeRayCaster.carveRays(rays, origin: origin, maxDistance: maxDistance,
                                   store: store, robotIndex: robotIndex)
    }

    /// Initialize a bounding box as "unknown-solid" for the subtractive pipeline.
    /// Used by object detection: distant silhouette → large solid block.
    public func initializeSolidRegion(_ aabb: AABB, color: (UInt8, UInt8, UInt8) = (128, 128, 128)) {
        store.initializeSolidRegion(aabb, color: color)
    }

    // MARK: - Maintenance

    /// Prune uniform octree subtrees (reduces memory after dense capture).
    public func prune() {
        store.pruneAll()
    }

    /// Expire dynamic-layer voxels older than TTL seconds.
    public func expireDynamic(ttlSeconds: UInt32 = 30) {
        store.expireDynamic(ttlSeconds: ttlSeconds)
    }

    /// Clear all voxels.
    public func clear() {
        store.clear()
    }

    // MARK: - Persistence

    /// Return up to `maxCount` occupied positions sampled uniformly (for live preview).
    public func samplePositions(maxCount: Int = 4000) -> [SIMD3<Float>] {
        store.samplePositions(maxCount: maxCount)
    }

    /// Return up to `maxCount` colored voxels sampled uniformly (position + RGB).
    public func sampleColoredPositions(maxCount: Int = 4000) -> [(SIMD3<Float>, (UInt8, UInt8, UInt8))] {
        store.sampleColoredPositions(maxCount: maxCount)
    }

    /// Collect greedy-merged voxels for rendering (cuboids with center + halfSize + color).
    /// Automatically respects `pipelineMode`: subtractive includes unobserved-solid nodes.
    public func collectMergedVoxels(cameraPos: SIMD3<Float> = .zero,
                                     frustumPlanes: [SIMD4<Float>] = []) -> [MergedVoxel] {
        store.collectMergedVoxels(cameraPos: cameraPos, frustumPlanes: frustumPlanes,
                                  subtractiveMode: pipelineMode == .subtractive)
    }

    // MARK: - Persistence

    /// Save this world to a `.botmap` file.
    public func save(to url: URL) async throws {
        try await BotMapSerializer.save(world: self, to: url)
    }

    /// Load a world from a `.botmap` file.
    public static func load(from url: URL) async throws -> BotMapWorld {
        try await BotMapSerializer.load(from: url)
    }
}
