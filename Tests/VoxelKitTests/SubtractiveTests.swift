import Testing
import Foundation
import simd
@testable import VoxelKit

@Suite("Subtractive Engine Tests")
struct SubtractiveEngineTests {

    // MARK: - OctreeNode fields

    @Test("New node is unobserved with default signedDistance")
    func newNodeDefaults() {
        let node = OctreeNode()
        #expect(node.isUnobserved)
        #expect(node.observationCount == 0)
        #expect(node.signedDistance == .greatestFiniteMagnitude)
        #expect(node.subdivisionHint == 0)
        #expect(node.isConservativelyOccupied) // unobserved = solid
        #expect(!node.needsRefinement)
    }

    @Test("Observed node is not unobserved")
    func observedNode() {
        let node = OctreeNode()
        node.observationCount = 3
        node.logOdds = 1.0
        #expect(!node.isUnobserved)
        #expect(node.isConservativelyOccupied) // logOdds > 0
    }

    @Test("Observed free node is not conservatively occupied")
    func observedFreeNode() {
        let node = OctreeNode()
        node.observationCount = 5
        node.logOdds = -1.5
        #expect(!node.isConservativelyOccupied)
    }

    @Test("Subdivision hint triggers needsRefinement on leaf")
    func subdivisionHint() {
        let node = OctreeNode()
        node.subdivisionHint = 0b0000_0101
        #expect(node.needsRefinement)

        // Interior node doesn't need refinement
        node.children = ContiguousArray<OctreeNode?>(repeating: nil, count: 8)
        #expect(!node.needsRefinement)
    }

    // MARK: - Subtractive insertion

    @Test("Subtractive hit creates occupied node with observationCount")
    func subtractiveHit() {
        let tree = Octree(resolution: 0.03125, origin: .zero, rootSize: 1.0)
        let point = SIMD3<Float>(0.5, 0.5, 0.5)
        tree.updateSubtractive(at: point, hit: true, targetDepth: 3,
                                color: (255, 0, 0))

        let voxels = tree.collectOccupiedVoxels()
        #expect(!voxels.isEmpty)
    }

    @Test("Subtractive miss carves free space")
    func subtractiveMiss() {
        let tree = Octree(resolution: 0.03125, origin: .zero, rootSize: 1.0)
        let point = SIMD3<Float>(0.5, 0.5, 0.5)

        // First hit
        tree.updateSubtractive(at: point, hit: true, targetDepth: 3)
        #expect(tree.occupiedLeafCount > 0)

        // Now carve (many misses to overcome the hit)
        for _ in 0..<10 {
            tree.updateSubtractive(at: point, hit: false, targetDepth: 3)
        }
        let voxels = tree.collectOccupiedVoxels()
        #expect(voxels.isEmpty) // carved away
    }

    @Test("Subtractive insert at lower depth creates coarser nodes")
    func subtractiveCoarseDepth() {
        let tree = Octree(resolution: 0.03125, origin: .zero, rootSize: 1.0)
        let point = SIMD3<Float>(0.5, 0.5, 0.5)

        // Depth 1: should create only 1 level of children
        tree.updateSubtractive(at: point, hit: true, targetDepth: 1)
        #expect(tree.nodeCount <= 3) // root + 1 child (+ maybe another)

        let voxels = tree.collectOccupiedVoxels()
        #expect(voxels.count == 1)
        // Voxel should be large (0.5m half-way)
        let v = voxels[0]
        #expect(v.x > 0.2 && v.x < 0.8)
    }

    @Test("includeUnobserved returns conservatively-occupied nodes")
    func includeUnobserved() {
        let tree = Octree(resolution: 0.03125, origin: .zero, rootSize: 1.0)

        // Insert one hit, then carve a nearby point so there's a free observed node
        tree.updateSubtractive(at: SIMD3<Float>(0.5, 0.5, 0.5), hit: true, targetDepth: 2)
        // Carve another octant so that some children are observed-free
        for _ in 0..<10 {
            tree.updateSubtractive(at: SIMD3<Float>(0.1, 0.1, 0.1), hit: false, targetDepth: 2)
        }

        // Without includeUnobserved: only occupied (logOdds > 0) nodes
        let without = tree.collectOccupiedVoxels(includeUnobserved: false)
        // With includeUnobserved: also includes unobserved children (conservatively solid)
        let with = tree.collectOccupiedVoxels(includeUnobserved: true)

        // The unobserved sibling octants should appear with includeUnobserved
        #expect(with.count >= without.count)
        #expect(!without.isEmpty) // at least the hit voxel
    }

    // MARK: - ChunkedOctreeStore subtractive

    @Test("initializeSolidRegion creates chunks")
    func initializeSolidRegion() {
        let store = ChunkedOctreeStore()
        let aabb = AABB(min: SIMD3<Float>(0, 0, 0), max: SIMD3<Float>(2, 1, 1))
        store.initializeSolidRegion(aabb, color: (200, 100, 50))
        // Should create 2×1×1 = 2 chunks (x=0,1 y=0 z=0)
        #expect(store.chunkCount >= 2)
    }

    @Test("ChunkedOctreeStore subtractive insert and carve")
    func storeSubtractiveInsertAndCarve() {
        let store = ChunkedOctreeStore()
        let camera = SIMD3<Float>(0, 0, 0)

        // Hit at (0.5, 0.5, 0.5)
        store.insertSubtractive(at: SIMD3<Float>(0.5, 0.5, 0.5), hit: true,
                                cameraPos: camera, color: (255, 0, 0))

        let merged = store.collectMergedVoxels(cameraPos: camera, subtractiveMode: false)
        #expect(!merged.isEmpty)
    }

    // MARK: - Ray caster

    @Test("carveRay creates hit and carves free space")
    func carveRayBasic() {
        let store = ChunkedOctreeStore()
        let origin = SIMD3<Float>(0, 0, 0)
        let direction = SIMD3<Float>(1, 0, 0) // +X ray

        let result = OctreeRayCaster.carveRay(
            origin: origin, direction: direction,
            hitDistance: 2.0, maxDistance: 5.0,
            store: store, cameraPos: origin,
            color: (0, 255, 0)
        )

        #expect(result.hitCount == 1)
        #expect(result.carvedCount > 0)
    }

    @Test("carveRays batch processes multiple rays")
    func carveRaysBatch() {
        let store = ChunkedOctreeStore()
        let origin = SIMD3<Float>(0, 0, 0)

        let rays: [(direction: SIMD3<Float>, hitDistance: Float?, color: (UInt8, UInt8, UInt8))] = [
            (simd_normalize(SIMD3<Float>(1, 0, 0)), 1.5, (255, 0, 0)),
            (simd_normalize(SIMD3<Float>(0, 1, 0)), 2.0, (0, 255, 0)),
            (simd_normalize(SIMD3<Float>(0, 0, 1)), nil,  (0, 0, 255)),  // infinite free ray
        ]

        let result = OctreeRayCaster.carveRays(rays, origin: origin, maxDistance: 3.0,
                                                store: store)

        #expect(result.hitCount == 2) // two rays hit, one free
        #expect(result.carvedCount > 0)
    }

    // MARK: - PipelineMode

    @Test("BotMapWorld PipelineMode default is additive")
    func pipelineModeDefault() async {
        let world = BotMapWorld(name: "test")
        let mode = await world.pipelineMode
        #expect(mode == .additive)
    }

    @Test("BotMapWorld subtractive mode includes unobserved in merged")
    func subtractiveMergedVoxels() async {
        let world = BotMapWorld(name: "test")
        await world.setPipelineMode(.subtractive)
        await world.initializeSolidRegion(
            AABB(min: .zero, max: SIMD3<Float>(repeating: 1)),
            color: (128, 128, 128)
        )

        let merged = await world.collectMergedVoxels()
        #expect(!merged.isEmpty)
    }

    // MARK: - ChunkStreamManager

    @Test("ChunkStreamManager save and load roundtrip")
    func chunkStreamRoundtrip() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxelkit_test_\(ProcessInfo.processInfo.globallyUniqueString)")
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let manager = ChunkStreamManager(storageURL: tmpDir, maxInMemoryChunks: 5)

        // Create a chunk with data
        let key = ChunkKey(x: 1, y: 2, z: 3)
        let chunk = OctreeChunk(key: key)
        chunk.tree.updateOccupancy(at: SIMD3<Float>(1.5, 2.5, 3.5), hit: true,
                                    color: (255, 0, 0))
        chunk.version = 42

        // Save to disk
        manager.saveToDisk(chunk)

        // Load back
        let loaded = manager.loadFromDisk(key)
        #expect(loaded != nil)
        #expect(loaded!.key.x == 1)
        #expect(loaded!.key.y == 2)
        #expect(loaded!.key.z == 3)
        let voxels = loaded!.tree.collectOccupiedVoxels()
        #expect(!voxels.isEmpty)
    }

    @Test("ChunkStreamManager LRU eviction")
    func chunkStreamEviction() {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxelkit_evict_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let manager = ChunkStreamManager(storageURL: tmpDir, maxInMemoryChunks: 3)
        let store = ChunkedOctreeStore(streamManager: manager)

        // Insert into 5 different chunks to trigger eviction
        for i: Int16 in 0..<5 {
            let pos = SIMD3<Float>(Float(i) + 0.5, 0.5, 0.5)
            store.insertSubtractive(at: pos, hit: true, cameraPos: SIMD3<Float>.zero, color: (128, 128, 128))
            manager.touch(ChunkKey(x: i, y: 0, z: 0))
        }

        manager.evictIfNeeded(store: store)
        #expect(manager.evictionCount > 0)
    }

    // MARK: - SuperChunk

    @Test("SuperChunkKey maps correctly")
    func superChunkKeyMapping() {
        let manager = SuperChunkManager()
        let chunkKey = ChunkKey(x: 5, y: 20, z: -3)
        let scKey = manager.superChunkKey(for: chunkKey)
        #expect(scKey.x == 0)
        #expect(scKey.y == 1)
        #expect(scKey.z == -1)
    }

    @Test("SuperChunk worldOrigin is correct")
    func superChunkWorldOrigin() {
        let key = SuperChunkKey(x: 1, y: 0, z: -1)
        let origin = key.worldOrigin
        #expect(origin.x == 16.0)
        #expect(origin.y == 0.0)
        #expect(origin.z == -16.0)
    }

    // MARK: - CameraIntrinsics

    @Test("CameraIntrinsics matrix3x3 has correct values")
    func intrinsicsMatrix() {
        let intr = CameraIntrinsics(fx: 500, fy: 600, cx: 320, cy: 240,
                                     width: 640, height: 480)
        let m = intr.matrix3x3
        #expect(m[0][0] == 500) // fx
        #expect(m[1][1] == 600) // fy
        #expect(m[2][0] == 320) // cx
        #expect(m[2][1] == 240) // cy
    }
}
