import Foundation
import simd

/// A 16×16×16 m³ macro-LOD chunk for distant terrain rendering (100-500m).
///
/// Stores a low-resolution octree (depth 3 → 2m leaves) built by downsampling
/// the underlying 1m³ OctreeChunks. Purely for rendering — never written to by sensor data.
public final class SuperChunk: @unchecked Sendable {
    public let key: SuperChunkKey
    public let tree: Octree
    public var version: UInt32 = 0
    public var isDirty: Bool = true

    public init(key: SuperChunkKey) {
        self.key = key
        self.tree = Octree(resolution: 2.0, origin: key.worldOrigin, rootSize: 16.0)
    }
}

/// Key for a 16m³ spatial super-chunk.
public struct SuperChunkKey: Hashable, Codable, CustomStringConvertible, Sendable {
    public let x: Int16
    public let y: Int16
    public let z: Int16

    public init(x: Int16, y: Int16, z: Int16) {
        self.x = x; self.y = y; self.z = z
    }

    public var description: String { "SC(\(x),\(y),\(z))" }

    /// World-space min corner of this super-chunk.
    public var worldOrigin: SIMD3<Float> {
        SIMD3<Float>(Float(x) * 16.0, Float(y) * 16.0, Float(z) * 16.0)
    }
}

/// Manages super-chunks for macro-LOD rendering of distant terrain.
///
/// Rebuilds super-chunks from dirty regular chunks on demand.
/// Thread-safety: access only from the owning actor (BotMapWorld).
public final class SuperChunkManager: @unchecked Sendable {

    private var superChunks: [SuperChunkKey: SuperChunk] = [:]
    private var dirtySuperChunks: Set<SuperChunkKey> = []

    /// Distance beyond which super-chunks are used instead of regular chunks.
    /// Matches the last regular LOD band boundary (80m).
    public var superChunkMinDistance: Float = 80.0

    /// Distance beyond which nothing is rendered.
    public var maxRenderDistance: Float = 500.0

    public init() {}

    // MARK: - Key mapping

    /// Map a regular ChunkKey to its parent SuperChunkKey.
    public func superChunkKey(for chunkKey: ChunkKey) -> SuperChunkKey {
        SuperChunkKey(
            x: Int16(floor(Float(chunkKey.x) / 16.0)),
            y: Int16(floor(Float(chunkKey.y) / 16.0)),
            z: Int16(floor(Float(chunkKey.z) / 16.0))
        )
    }

    /// Mark a super-chunk as needing rebuild when its child chunks change.
    public func markDirty(_ chunkKey: ChunkKey) {
        dirtySuperChunks.insert(superChunkKey(for: chunkKey))
    }

    // MARK: - Rebuild

    /// Rebuild dirty super-chunks by downsampling from the regular chunk store.
    /// Call periodically (e.g., every few seconds) — NOT every frame.
    public func rebuildDirty(from store: ChunkedOctreeStore) {
        for scKey in dirtySuperChunks {
            rebuild(scKey, from: store)
        }
        dirtySuperChunks.removeAll()
    }

    private func rebuild(_ scKey: SuperChunkKey, from store: ChunkedOctreeStore) {
        let sc = getOrCreate(scKey)
        // Clear existing data
        sc.tree.replaceRoot(OctreeNode())

        let baseX = Int32(scKey.x) * 16
        let baseY = Int32(scKey.y) * 16
        let baseZ = Int32(scKey.z) * 16

        // Sample each child chunk at depth 1 (coarsest) and insert into super-chunk
        let allChunks = store.allChunks
        for dz: Int32 in 0..<16 {
            for dy: Int32 in 0..<16 {
                for dx: Int32 in 0..<16 {
                    let cKey = ChunkKey(x: Int16(clamping: baseX + dx),
                                        y: Int16(clamping: baseY + dy),
                                        z: Int16(clamping: baseZ + dz))
                    guard let chunk = allChunks[cKey] else { continue }

                    // Get the dominant color and occupancy from this chunk
                    let voxels = chunk.tree.collectAtDepth(1)
                    guard !voxels.isEmpty else { continue }

                    // Use first voxel's color as representative
                    let representative = voxels[0]
                    let r = UInt8((representative.colorAndFlags >> 24) & 0xFF)
                    let g = UInt8((representative.colorAndFlags >> 16) & 0xFF)
                    let b = UInt8((representative.colorAndFlags >> 8)  & 0xFF)

                    // Insert at the center of the child chunk in super-chunk space
                    let pos = SIMD3<Float>(
                        Float(scKey.x) * 16.0 + Float(dx) + 0.5,
                        Float(scKey.y) * 16.0 + Float(dy) + 0.5,
                        Float(scKey.z) * 16.0 + Float(dz) + 0.5
                    )
                    sc.tree.updateOccupancy(at: pos, hit: true, color: (r, g, b))
                }
            }
        }

        sc.version &+= 1
        sc.isDirty = false
    }

    // MARK: - Query

    /// Collect merged voxels from super-chunks for distant rendering.
    public func collectMergedVoxels(cameraPos: SIMD3<Float>,
                                     frustumPlanes: [SIMD4<Float>] = []) -> [MergedVoxel] {
        var result = [MergedVoxel]()
        let cull = !frustumPlanes.isEmpty

        for (key, sc) in superChunks {
            let origin = key.worldOrigin
            let center = origin + SIMD3<Float>(repeating: 8.0)
            let dist = simd_length(center - cameraPos)

            // Only render super-chunks in their LOD band
            guard dist >= superChunkMinDistance && dist < maxRenderDistance else { continue }

            // Frustum culling (16m³ AABB)
            if cull {
                let aabbMax = origin + SIMD3<Float>(repeating: 16.0)
                var visible = true
                for plane in frustumPlanes {
                    let n = SIMD3<Float>(plane.x, plane.y, plane.z)
                    let px = n.x >= 0 ? aabbMax.x : origin.x
                    let py = n.y >= 0 ? aabbMax.y : origin.y
                    let pz = n.z >= 0 ? aabbMax.z : origin.z
                    if n.x * px + n.y * py + n.z * pz + plane.w < 0 { visible = false; break }
                }
                guard visible else { continue }
            }

            let voxels = sc.tree.collectOccupiedVoxels()
            let merged = GreedyMesher.merge(voxels: voxels, voxelSize: 2.0,
                                             chunkOrigin: origin)
            result.append(contentsOf: merged)
        }
        return result
    }

    public var superChunkCount: Int { superChunks.count }

    // MARK: - Private

    private func getOrCreate(_ key: SuperChunkKey) -> SuperChunk {
        if let existing = superChunks[key] { return existing }
        let sc = SuperChunk(key: key)
        superChunks[key] = sc
        return sc
    }
}
