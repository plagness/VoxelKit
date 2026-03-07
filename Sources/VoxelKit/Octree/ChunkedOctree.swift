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
    // Tuned for Go2 Air sensor (128×128×40 voxels/frame, ~5m effective range):
    //   0-10m: full detail (3.125cm leaves) — immediate surroundings
    //  10-30m: medium detail (6.25cm) — near environment
    //  30-80m: coarse detail (12.5cm) — mid-range
    //  80m+:   minimal (50cm) — distant, handled mostly by SuperChunks
    private static let lodBands: [(maxDist: Float, depth: Int)] = [
        (10,  5),   // 3.125cm — full resolution
        (30,  4),   // 6.25cm — half resolution
        (80,  3),   // 12.5cm — quarter resolution
        (.greatestFiniteMagnitude, 1)  // 50cm — coarse
    ]

    private let sessionStart = Date.now

    /// Rendering quality filters.
    /// Minimum log-odds for a voxel to be rendered (default 0 = any occupied).
    /// Set higher (e.g. 0.4) to filter low-confidence voxels.
    public var renderMinLogOdds: Float = 0.4

    /// Minimum observation count for a voxel to be rendered (default 0 = no filter).
    /// Set to 2+ to filter single-observation noise from ARKit feature points.
    public var renderMinObservationCount: UInt16 = 2

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
        insertColoredPositions(voxels, cameraPos: nil, robotIndex: robotIndex)
    }

    /// Insert colored world points with optional ray casting from camera position.
    ///
    /// When `cameraPos` is provided, uses the subtractive pipeline:
    /// - Hit points inserted via LOD-aware `updateSubtractive`
    /// - Free-space carved along rays from camera to each point
    /// When nil, falls back to simple additive insertion.
    public func insertColoredPositions(_ voxels: [ColoredPosition], cameraPos: SIMD3<Float>?,
                                       robotIndex: Int = 0) {
        let timestamp = UInt32(Date.now.timeIntervalSince(sessionStart))
        var touchedKeys = Set<ChunkKey>()

        if let cameraPos {
            // Subtractive path: LOD-aware hits + free-space ray carving
            let resolution: Float = 0.05
            for v in voxels {
                let key = chunkKeyFor(v.position)
                let chunk = getOrCreateChunk(key)
                let (depth, _) = lodParams(for: key.worldOrigin, cameraPos: cameraPos)
                chunk.tree.updateSubtractive(at: v.position, hit: true, targetDepth: depth,
                                              robotIndex: robotIndex, timestamp: timestamp,
                                              color: v.color)
                chunk.isDirty = true
                chunk.version &+= 1
                touchedKeys.insert(key)

                // Free-space ray carving from camera to hit point (every 5cm cell)
                let delta = v.position - cameraPos
                let dist = simd_length(delta)
                let dir = delta / dist
                // Carve from 30cm out to 1 cell before hit, stepping every cell
                let startDist = max(resolution, 0.3)
                let endDist = dist - resolution
                guard endDist > startDist else { continue }
                let stepCount = min(Int((endDist - startDist) / resolution), 50)
                for s in 0..<stepCount {
                    let d = startDist + Float(s) * resolution
                    let freePos = cameraPos + dir * d
                    let fKey = chunkKeyFor(freePos)
                    let fChunk = getOrCreateChunk(fKey)
                    let (fDepth, _) = lodParams(for: fKey.worldOrigin, cameraPos: cameraPos)
                    fChunk.tree.updateSubtractive(at: freePos, hit: false, targetDepth: fDepth,
                                                   robotIndex: robotIndex, timestamp: timestamp)
                    fChunk.isDirty = true
                    touchedKeys.insert(fKey)
                }
            }
        } else {
            // Additive fallback (no camera pose)
            for v in voxels {
                let key = chunkKeyFor(v.position)
                let chunk = getOrCreateChunk(key)
                chunk.tree.updateOccupancy(at: v.position, hit: true, robotIndex: robotIndex,
                                           timestamp: timestamp, color: v.color)
                chunk.isDirty = true
                chunk.version &+= 1
                touchedKeys.insert(key)
            }
        }

        for key in touchedKeys {
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
    public func initializeSolidRegion(_ aabb: AABB,
                                      color: (UInt8, UInt8, UInt8) = (128, 128, 128),
                                      classId: UInt8 = 0,
                                      layer: MapLayer = .structure) {
        // Fill the AABB with chunks. Each chunk's root starts as unobserved (conservatively solid).
        let minKey = chunkKeyFor(aabb.min)
        let maxKey = chunkKeyFor(aabb.max)
        for z in minKey.z...maxKey.z {
            for y in minKey.y...maxKey.y {
                for x in minKey.x...maxKey.x {
                    let key = ChunkKey(x: x, y: y, z: z)
                    let chunk = getOrCreateChunk(key)
                    // Set root properties for unobserved rendering
                    chunk.tree.root.color = color
                    chunk.tree.root.classId = classId
                    chunk.tree.root.layer = layer
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
    /// Uses coarse depth (2) per chunk for speed — avoids full tree traversal.
    public func sampleColoredPositions(maxCount: Int) -> [(SIMD3<Float>, (UInt8, UInt8, UInt8))] {
        guard !chunks.isEmpty else { return [] }
        let budgetPerChunk = max(1, maxCount / chunks.count)
        var result = [(SIMD3<Float>, (UInt8, UInt8, UInt8))]()
        result.reserveCapacity(maxCount)

        for (_, chunk) in chunks {
            // Collect at depth 2 (max 64 leaves) — much faster than full maxDepth
            let voxels = chunk.tree.collectAtDepth(2)
            let selected: ArraySlice<PackedVoxel>
            if voxels.count > budgetPerChunk {
                let stride = max(1, voxels.count / budgetPerChunk)
                selected = voxels[...].enumerated().compactMap { (i, v) in
                    i % stride == 0 ? v : nil
                }.prefix(budgetPerChunk)[...]
            } else {
                selected = voxels[...]
            }
            for pv in selected {
                let r = UInt8((pv.colorAndFlags >> 24) & 0xFF)
                let g = UInt8((pv.colorAndFlags >> 16) & 0xFF)
                let b = UInt8((pv.colorAndFlags >> 8)  & 0xFF)
                result.append((SIMD3<Float>(pv.x, pv.y, pv.z), (r, g, b)))
                if result.count >= maxCount { return result }
            }
        }
        return result
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
                    ? chunk.tree.collectOccupiedVoxels(minLogOdds: renderMinLogOdds,
                                                        minObservationCount: renderMinObservationCount,
                                                        includeUnobserved: subtractiveMode)
                    : chunk.tree.collectAtDepth(depth, minLogOdds: renderMinLogOdds,
                                                 minObservationCount: renderMinObservationCount,
                                                 includeUnobserved: subtractiveMode)
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
        let elapsed = Date.now.timeIntervalSince(sessionStart)
        guard elapsed > Double(ttlSeconds) else { return }
        let threshold = UInt32(elapsed) - ttlSeconds
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

    public func chunkKeyFor(_ pos: SIMD3<Float>) -> ChunkKey {
        ChunkKey(x: Int16(clamping: Int32(floor(pos.x))),
                 y: Int16(clamping: Int32(floor(pos.y))),
                 z: Int16(clamping: Int32(floor(pos.z))))
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
