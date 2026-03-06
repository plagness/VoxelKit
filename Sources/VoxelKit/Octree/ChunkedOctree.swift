import Foundation
import simd
import os

private let logger = Logger(subsystem: "com.voxelkit", category: "ChunkedOctree")

// MARK: - Chunk Key

/// Key for a 1m³ spatial chunk.
public struct ChunkKey: Hashable, Codable, CustomStringConvertible, Sendable {
    public let x: Int16
    public let y: Int16
    public let z: Int16

    public init(x: Int16, y: Int16, z: Int16) {
        self.x = x; self.y = y; self.z = z
    }

    public var description: String { "(\(x),\(y),\(z))" }

    /// World-space min corner of this chunk.
    public var worldOrigin: SIMD3<Float> {
        SIMD3<Float>(Float(x), Float(y), Float(z))
    }
}

// MARK: - Octree Chunk

/// A 1m³ chunk containing a local octree at 5cm resolution.
public final class OctreeChunk: @unchecked Sendable {
    public let key: ChunkKey
    public let tree: Octree
    public var isDirty: Bool = false
    public var version: UInt32 = 0
    public var lastModified: Date = .now

    public init(key: ChunkKey) {
        self.key = key
        self.tree = Octree(resolution: 0.03125, origin: key.worldOrigin, rootSize: 1.0)
    }
}

// MARK: - Chunked Octree Store

/// Top-level spatial map: dictionary of 1m³ octree chunks.
///
/// Not an actor — isolation is provided by the owning `BotMapWorld` actor.
/// Supports Bayesian log-odds occupancy fusion, multi-resolution queries,
/// greedy-merged voxel collection for GPU rendering, and delta tracking.
public final class ChunkedOctreeStore: @unchecked Sendable {
    private var chunks: [ChunkKey: OctreeChunk] = [:]
    private var dirtyChunks: Set<ChunkKey> = []

    // CPU-side greedy mesh cache: invalidated on insert.
    private var mergedCache: [ChunkKey: [MergedVoxel]] = [:]
    private var mergedCacheDepth: [ChunkKey: Int] = [:]

    // LOD bands (maxDistFromCamera → octreeDepth)
    private static let lodBands: [(maxDist: Float, depth: Int)] = [
        (20,  5),
        (100, 3),
        (.greatestFiniteMagnitude, 1)
    ]

    private let sessionStart = Date.now

    /// Optional chunk streaming manager for large-scale worlds.
    public var streamManager: ChunkStreamManager?

    public init() {}

    /// Initialize with chunk streaming for 10+ km² worlds.
    public init(streamManager: ChunkStreamManager) {
        self.streamManager = streamManager
    }

    // MARK: - Insert

    /// Insert a batch of world-space occupied positions (video depth back-projection).
    public func insertPositions(_ positions: [SIMD3<Float>], robotIndex: Int = 0) {
        let timestamp = UInt32(Date.now.timeIntervalSince(sessionStart))
        for pos in positions {
            insertOccupied(pos, robotIndex: robotIndex, timestamp: timestamp)
        }
        evictIfNeeded()
    }

    /// Insert a batch of colored voxels (camera-sampled RGB).
    public func insertColoredPositions(_ voxels: [ColoredPosition], robotIndex: Int = 0) {
        let timestamp = UInt32(Date.now.timeIntervalSince(sessionStart))
        for v in voxels {
            let key = chunkKeyFor(v.position)
            let chunk = getOrCreateChunk(key)
            chunk.tree.updateOccupancy(at: v.position, hit: true, robotIndex: robotIndex,
                                       timestamp: timestamp, color: v.color)
            chunk.isDirty = true
            chunk.version &+= 1
            dirtyChunks.insert(key)
            mergedCache.removeValue(forKey: key)
            mergedCacheDepth.removeValue(forKey: key)
        }
        evictIfNeeded()
    }

    /// Insert a LiDAR frame with free-space ray casting and plane detection.
    public func insertFrame(_ frame: LiDARFrame, robotIndex: Int = 0) {
        let timestamp = UInt32(Date.now.timeIntervalSince(sessionStart))
        let res = frame.resolution
        let origin = frame.origin

        for pos in frame.voxels {
            insertOccupied(pos, robotIndex: robotIndex, timestamp: timestamp)

            // Free-space misses along the ray
            let delta = pos - origin
            let dist = simd_length(delta)
            let maxSteps = min(Int(dist / res) - 1, 25)
            guard maxSteps > 0 else { continue }
            let step = delta / dist * res
            for s in 1...maxSteps {
                let freePos = origin + step * Float(s)
                let fKey = chunkKeyFor(freePos)
                let fChunk = getOrCreateChunk(fKey)
                fChunk.tree.updateOccupancy(at: freePos, hit: false,
                                            robotIndex: robotIndex, timestamp: timestamp)
            }
        }

        // Fill dominant horizontal planes to close scan-line gaps
        let fillPositions = PlaneDetector.detectFillPositions(in: frame, resolution: res)
        for pos in fillPositions {
            insertOccupied(pos, robotIndex: robotIndex, timestamp: timestamp)
        }
        evictIfNeeded()
    }

    private func insertOccupied(_ pos: SIMD3<Float>, robotIndex: Int, timestamp: UInt32) {
        let key = chunkKeyFor(pos)
        let chunk = getOrCreateChunk(key)
        chunk.tree.updateOccupancy(at: pos, hit: true, robotIndex: robotIndex,
                                   timestamp: timestamp, color: heightColor(z: pos.z))
        chunk.isDirty = true
        chunk.version &+= 1
        dirtyChunks.insert(key)
        mergedCache.removeValue(forKey: key)
        mergedCacheDepth.removeValue(forKey: key)
    }

    // MARK: - Subtractive Insert

    /// Insert a subtractive observation (surface hit or free-space carve) at LOD-appropriate depth.
    public func insertSubtractive(at pos: SIMD3<Float>, hit: Bool,
                                   cameraPos: SIMD3<Float>,
                                   robotIndex: Int = 0, timestamp: UInt32 = 0,
                                   color: (UInt8, UInt8, UInt8) = (128, 128, 128)) {
        let key = chunkKeyFor(pos)
        let chunk = getOrCreateChunk(key)
        let (depth, _) = lodParams(for: key.worldOrigin, cameraPos: cameraPos)
        chunk.tree.updateSubtractive(at: pos, hit: hit, targetDepth: depth,
                                      robotIndex: robotIndex, timestamp: timestamp, color: color)
        chunk.isDirty = true
        chunk.version &+= 1
        dirtyChunks.insert(key)
        mergedCache.removeValue(forKey: key)
        mergedCacheDepth.removeValue(forKey: key)
    }

    /// Initialize a region as "unknown-solid" for the subtractive pipeline.
    /// Used by object detection: distant silhouette → large solid bounding box.
    public func initializeSolidRegion(_ aabb: AABB, color: (UInt8, UInt8, UInt8) = (128, 128, 128)) {
        // Fill the AABB with chunks. Each chunk's root starts as unobserved (conservatively solid).
        let minKey = chunkKeyFor(aabb.min)
        let maxKey = chunkKeyFor(aabb.max)
        for z in minKey.z...maxKey.z {
            for y in minKey.y...maxKey.y {
                for x in minKey.x...maxKey.x {
                    let key = ChunkKey(x: x, y: y, z: z)
                    let chunk = getOrCreateChunk(key)
                    // Set root color for unobserved rendering
                    chunk.tree.root.color = color
                    chunk.isDirty = true
                    dirtyChunks.insert(key)
                    mergedCache.removeValue(forKey: key)
                    mergedCacheDepth.removeValue(forKey: key)
                }
            }
        }
    }

    // MARK: - Query

    public var voxelCount: Int {
        chunks.values.reduce(0) { $0 + $1.tree.occupiedLeafCount }
    }

    public var chunkCount: Int { chunks.count }

    public var boundingBox: AABB {
        guard !chunks.isEmpty else { return .zero }
        var bb = AABB(
            min: SIMD3<Float>(repeating: .greatestFiniteMagnitude),
            max: SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        )
        for (key, _) in chunks {
            let origin = key.worldOrigin
            bb.expand(toInclude: origin)
            bb.expand(toInclude: origin + SIMD3<Float>(repeating: 1.0))
        }
        return bb
    }

    public func popDirtyChunks() -> [ChunkKey] {
        let result = Array(dirtyChunks)
        dirtyChunks.removeAll()
        return result
    }

    /// Collect all occupied positions (for serialization and ICP).
    public func allOccupiedPositions() -> [SIMD3<Float>] {
        var result = [SIMD3<Float>]()
        for (_, chunk) in chunks {
            result.append(contentsOf: chunk.tree.collectOccupiedPositions())
        }
        return result
    }

    /// Sample up to `maxCount` occupied positions (uniform stride across chunks).
    public func samplePositions(maxCount: Int) -> [SIMD3<Float>] {
        let all = allOccupiedPositions()
        guard all.count > maxCount else { return all }
        let stride = max(1, all.count / maxCount)
        return stride == 1 ? all : (0..<maxCount).map { all[$0 * stride] }
    }

    /// Sample up to `maxCount` colored voxels (position + stored RGB).
    public func sampleColoredPositions(maxCount: Int) -> [(SIMD3<Float>, (UInt8, UInt8, UInt8))] {
        let all = collectPackedVoxels()
        let selected: [PackedVoxel]
        if all.count > maxCount {
            let stride = max(1, all.count / maxCount)
            selected = (0..<min(maxCount, all.count)).map { all[$0 * stride] }
        } else {
            selected = all
        }
        return selected.map { pv in
            let r = UInt8((pv.colorAndFlags >> 24) & 0xFF)
            let g = UInt8((pv.colorAndFlags >> 16) & 0xFF)
            let b = UInt8((pv.colorAndFlags >> 8)  & 0xFF)
            return (SIMD3<Float>(pv.x, pv.y, pv.z), (r, g, b))
        }
    }

    /// Collect all occupied voxels as PackedVoxel (for GPU upload).
    public func collectPackedVoxels(frustumPlanes: [SIMD4<Float>] = []) -> [PackedVoxel] {
        var result = [PackedVoxel]()
        let cull = !frustumPlanes.isEmpty
        for (key, chunk) in chunks {
            if cull, !chunkInFrustum(origin: key.worldOrigin, planes: frustumPlanes) { continue }
            result.append(contentsOf: chunk.tree.collectOccupiedVoxels())
        }
        return result
    }

    /// Collect greedy-merged voxels for instanced rendering.
    /// When `subtractiveMode` is true, unobserved nodes are included as solid.
    public func collectMergedVoxels(cameraPos: SIMD3<Float> = .zero,
                                    frustumPlanes: [SIMD4<Float>] = [],
                                    subtractiveMode: Bool = false) -> [MergedVoxel] {
        var result = [MergedVoxel]()
        let cull = !frustumPlanes.isEmpty

        for (key, chunk) in chunks {
            if cull, !chunkInFrustum(origin: key.worldOrigin, planes: frustumPlanes) { continue }

            let (depth, voxelSize) = lodParams(for: key.worldOrigin, cameraPos: cameraPos)
            if let cached = mergedCache[key], mergedCacheDepth[key] == depth {
                result.append(contentsOf: cached)
            } else {
                let raw = depth >= chunk.tree.maxDepth
                    ? chunk.tree.collectOccupiedVoxels(includeUnobserved: subtractiveMode)
                    : chunk.tree.collectAtDepth(depth, includeUnobserved: subtractiveMode)
                let merged = GreedyMesher.merge(voxels: raw, voxelSize: voxelSize,
                                                chunkOrigin: key.worldOrigin)
                mergedCache[key] = merged
                mergedCacheDepth[key] = depth
                result.append(contentsOf: merged)
            }
        }
        return result
    }

    /// Nearest occupied points within radius (for ICP scan matching).
    public func occupiedPoints(near center: SIMD3<Float>, radius: Float) -> [SIMD3<Float>] {
        let r2 = radius * radius
        var result = [SIMD3<Float>]()
        for (key, chunk) in chunks {
            let chunkMin = key.worldOrigin
            let chunkMax = chunkMin + SIMD3<Float>(repeating: 1.0)
            let closest = simd_clamp(center, chunkMin, chunkMax)
            guard simd_distance_squared(center, closest) <= r2 else { continue }
            for pv in chunk.tree.collectOccupiedPositions() {
                if simd_distance_squared(center, pv) <= r2 { result.append(pv) }
            }
        }
        return result
    }

    // MARK: - Maintenance

    public func pruneAll() {
        for (_, chunk) in chunks { chunk.tree.prune() }
        mergedCache.removeAll()
    }

    public func expireDynamic(ttlSeconds: UInt32 = 30) {
        let threshold = UInt32(Date.now.timeIntervalSince(sessionStart)) - ttlSeconds
        for (_, chunk) in chunks { chunk.tree.expireDynamicLayer(olderThan: threshold) }
    }

    public func clear() {
        chunks.removeAll()
        dirtyChunks.removeAll()
        mergedCache.removeAll()
        mergedCacheDepth.removeAll()
    }

    // MARK: - Serialization

    public var allChunks: [ChunkKey: OctreeChunk] { chunks }

    public func loadChunks(_ loaded: [ChunkKey: OctreeChunk]) {
        chunks = loaded
        dirtyChunks.removeAll()
        mergedCache.removeAll()
    }

    public func loadChunk(_ key: ChunkKey, chunk: OctreeChunk) {
        chunks[key] = chunk
    }

    /// Remove a single chunk (used by ChunkStreamManager for LRU eviction).
    public func removeChunk(_ key: ChunkKey) {
        chunks.removeValue(forKey: key)
        mergedCache.removeValue(forKey: key)
        mergedCacheDepth.removeValue(forKey: key)
    }

    // MARK: - Frustum culling (public for VoxelKitCompute)

    public func chunkInFrustum(origin: SIMD3<Float>, planes: [SIMD4<Float>]) -> Bool {
        let aabbMax = origin + SIMD3<Float>(repeating: 1.0)
        for plane in planes {
            let n = SIMD3<Float>(plane.x, plane.y, plane.z)
            let px = n.x >= 0 ? aabbMax.x : origin.x
            let py = n.y >= 0 ? aabbMax.y : origin.y
            let pz = n.z >= 0 ? aabbMax.z : origin.z
            if n.x * px + n.y * py + n.z * pz + plane.w < 0 { return false }
        }
        return true
    }

    // MARK: - Private

    private func heightColor(z: Float) -> (UInt8, UInt8, UInt8) {
        switch z {
        case ..<(-0.15):   return (220,  40,  40)
        case ..<0.04:      return ( 80, 100, 160)
        case 0.04..<0.45:  return ( 55, 185,  90)
        case 0.45..<1.1:   return (225, 185,  45)
        case 1.1..<2.0:    return (220,  90,  45)
        default:           return (110, 120, 210)
        }
    }

    private func chunkKeyFor(_ pos: SIMD3<Float>) -> ChunkKey {
        ChunkKey(x: Int16(floor(pos.x)), y: Int16(floor(pos.y)), z: Int16(floor(pos.z)))
    }

    private func getOrCreateChunk(_ key: ChunkKey) -> OctreeChunk {
        if let existing = chunks[key] { return existing }

        // Try loading from disk (streaming manager)
        if let sm = streamManager, let loaded = sm.loadFromDisk(key) {
            chunks[key] = loaded
            sm.touch(key)
            return loaded
        }

        let chunk = OctreeChunk(key: key)
        chunks[key] = chunk
        streamManager?.touch(key)
        return chunk
    }

    /// Evict LRU chunks to disk if streaming is enabled.
    /// Call after batch insertions to keep memory bounded.
    public func evictIfNeeded() {
        streamManager?.evictIfNeeded(store: self)
    }

    /// Flush all chunks to disk (for shutdown / save).
    public func flushToDisk() {
        streamManager?.flushAll(store: self)
    }

    private func lodParams(for chunkOrigin: SIMD3<Float>,
                           cameraPos: SIMD3<Float>) -> (depth: Int, voxelSize: Float) {
        let chunkCenter = chunkOrigin + SIMD3<Float>(repeating: 0.5)
        let dist = simd_length(chunkCenter - cameraPos)
        for band in Self.lodBands {
            if dist < band.maxDist {
                return (band.depth, 1.0 / Float(1 << band.depth))
            }
        }
        return (1, 0.5)
    }
}
