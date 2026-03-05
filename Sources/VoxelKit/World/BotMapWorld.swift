import Foundation
import simd

/// Primary container for a spatial map built from capture sessions.
///
/// An actor that owns a `ChunkedOctreeStore` and provides the single point
/// of insertion for voxel data arriving from `VideoCaptureSession` or LiDAR.
/// Thread-safe: all mutations go through the actor.
public actor BotMapWorld {

    // MARK: - Public state

    /// Underlying voxel store (access only from actor context).
    public let store: ChunkedOctreeStore

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

    /// Insert a decoded LiDAR frame (for LiDAR-based capture sessions).
    public func insertLiDARFrame(_ frame: LiDARFrame, robotIndex: Int = 0) {
        store.insertFrame(frame, robotIndex: robotIndex)
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
