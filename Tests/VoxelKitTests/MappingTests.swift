import Testing
import Foundation
import simd
@testable import VoxelKit

// MARK: - Octree Tests

@Suite("Octree Tests")
struct OctreeTests {

    @Test("Single hit makes voxel occupied")
    func singleHitOccupied() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        octree.updateOccupancy(at: SIMD3<Float>(0.5, 0.5, 0.5), hit: true, robotIndex: 0)
        #expect(octree.occupiedLeafCount == 1)
    }

    @Test("Single miss does not occupy")
    func singleMissNotOccupied() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        octree.updateOccupancy(at: SIMD3<Float>(0.5, 0.5, 0.5), hit: false, robotIndex: 0)
        #expect(octree.occupiedLeafCount == 0)
    }

    @Test("Multiple hits then misses flip occupancy")
    func hitsAndMissesFlip() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        let point = SIMD3<Float>(0.1, 0.1, 0.1)
        for _ in 0..<5 { octree.updateOccupancy(at: point, hit: true, robotIndex: 0) }
        #expect(octree.occupiedLeafCount == 1)
        for _ in 0..<10 { octree.updateOccupancy(at: point, hit: false, robotIndex: 0) }
        #expect(octree.occupiedLeafCount == 0)
    }

    @Test("Log-odds clamped at maximum")
    func logOddsMaxClamp() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        for _ in 0..<20 { octree.updateOccupancy(at: SIMD3<Float>(0.5, 0.5, 0.5), hit: true, robotIndex: 0) }
        let voxels = octree.collectOccupiedVoxels()
        #expect(voxels.count == 1)
    }

    @Test("Out-of-bounds point ignored")
    func outOfBoundsIgnored() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        octree.updateOccupancy(at: SIMD3<Float>(1.5, 0.5, 0.5), hit: true, robotIndex: 0)
        octree.updateOccupancy(at: SIMD3<Float>(-0.1, 0.5, 0.5), hit: true, robotIndex: 0)
        #expect(octree.occupiedLeafCount == 0)
    }

    @Test("Two distinct voxels tracked independently")
    func twoVoxelsIndependent() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        octree.updateOccupancy(at: SIMD3<Float>(0.1, 0.1, 0.1), hit: true, robotIndex: 0)
        octree.updateOccupancy(at: SIMD3<Float>(0.9, 0.9, 0.9), hit: true, robotIndex: 0)
        #expect(octree.occupiedLeafCount == 2)
        #expect(octree.collectOccupiedVoxels().count == 2)
    }

    @Test("Prune collapses uniform subtree")
    func pruneCollapsesUniform() {
        let octree = Octree(resolution: 0.5, origin: .zero, rootSize: 1.0)
        let centers: [SIMD3<Float>] = [
            [0.25, 0.25, 0.25], [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25], [0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75], [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75], [0.75, 0.75, 0.75]
        ]
        for c in centers { octree.updateOccupancy(at: c, hit: true, robotIndex: 0) }
        let before = octree.nodeCount
        let pruned = octree.prune()
        #expect(pruned > 0)
        #expect(octree.nodeCount < before)
    }

    @Test("collectAtDepth returns fewer voxels than maxDepth")
    func collectAtDepthCoarser() {
        let octree = Octree(resolution: 0.05, origin: .zero, rootSize: 1.0)
        for x in stride(from: 0.025, to: 0.5, by: 0.05) {
            for y in stride(from: 0.025, to: 0.5, by: 0.05) {
                octree.updateOccupancy(at: SIMD3<Float>(Float(x), Float(y), 0.025),
                                       hit: true, robotIndex: 0)
            }
        }
        let full   = octree.collectOccupiedVoxels()
        let coarse = octree.collectAtDepth(2)
        #expect(coarse.count <= full.count)
    }
}

// MARK: - ChunkKey Tests

@Suite("ChunkKey Tests")
struct ChunkKeyTests {

    @Test("worldOrigin matches chunk coordinates")
    func worldOriginPositive() {
        let key = ChunkKey(x: 3, y: 7, z: -2)
        #expect(key.worldOrigin.x == 3.0)
        #expect(key.worldOrigin.y == 7.0)
        #expect(key.worldOrigin.z == -2.0)
    }

    @Test("Negative chunk coordinates have correct worldOrigin")
    func worldOriginNegative() {
        let key = ChunkKey(x: -5, y: -1, z: -10)
        #expect(key.worldOrigin.x == -5.0)
        #expect(key.worldOrigin.y == -1.0)
        #expect(key.worldOrigin.z == -10.0)
    }

    @Test("ChunkKey equality and hashability")
    func hashableEquality() {
        let k1 = ChunkKey(x: 1, y: 2, z: 3)
        let k2 = ChunkKey(x: 1, y: 2, z: 3)
        let k3 = ChunkKey(x: 1, y: 2, z: 4)
        #expect(k1 == k2)
        #expect(k1 != k3)
        var set = Set<ChunkKey>()
        set.insert(k1); set.insert(k2); set.insert(k3)
        #expect(set.count == 2)
    }
}

// MARK: - ChunkedOctreeStore Tests

@Suite("ChunkedOctreeStore Tests")
struct ChunkedOctreeStoreTests {

    @Test("Insert positions and count voxels")
    func insertAndCount() {
        let store = ChunkedOctreeStore()
        let positions: [SIMD3<Float>] = [
            [0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]
        ]
        store.insertPositions(positions)
        #expect(store.voxelCount >= 1)
    }

    @Test("Clear removes all voxels")
    func clearRemovesAll() {
        let store = ChunkedOctreeStore()
        store.insertPositions([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        store.clear()
        #expect(store.voxelCount == 0)
        #expect(store.chunkCount == 0)
    }

    @Test("BoundingBox contains inserted voxels")
    func boundingBoxContains() {
        let store = ChunkedOctreeStore()
        store.insertPositions([[0.5, 0.5, 0.5], [5.5, 5.5, 5.5]])
        let bb = store.boundingBox
        #expect(bb.min.x <= 0.0)
        #expect(bb.max.x >= 6.0)
    }

    @Test("allOccupiedPositions returns inserted positions")
    func allOccupiedPositions() {
        let store = ChunkedOctreeStore()
        store.insertPositions([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        let positions = store.allOccupiedPositions()
        #expect(positions.count >= 2)
    }
}

// MARK: - GreedyMesher Tests

@Suite("GreedyMesher Tests")
struct GreedyMesherTests {

    private func voxel(_ x: Float, _ y: Float, _ z: Float,
                        color: (UInt8, UInt8, UInt8) = (255, 0, 0)) -> PackedVoxel {
        PackedVoxel(position: SIMD3<Float>(x, y, z), color: color)
    }

    @Test("Empty input returns empty result")
    func emptyInput() {
        #expect(GreedyMesher.merge(voxels: [], voxelSize: 0.05, chunkOrigin: .zero).isEmpty)
    }

    @Test("Single voxel produces one MergedVoxel")
    func singleVoxel() {
        let leaf: Float = 1.0 / 32.0
        let result = GreedyMesher.merge(voxels: [voxel(leaf * 0.5, leaf * 0.5, leaf * 0.5)],
                                         voxelSize: 0.05, chunkOrigin: .zero)
        #expect(result.count == 1)
        let halfLeaf = leaf * 0.5
        #expect(abs(result[0].hx - halfLeaf) < 1e-5)
    }

    @Test("Four adjacent same-color voxels merge")
    func fourAdjacentMerge() {
        let leaf: Float = 1.0 / 32.0
        let voxels = (0..<4).map { i in
            voxel(leaf * (Float(i) + 0.5), 0.5 * leaf, 0.5 * leaf, color: (200, 100, 50))
        }
        let result = GreedyMesher.merge(voxels: voxels, voxelSize: 0.05, chunkOrigin: .zero)
        #expect(result.count == 1)
        #expect(abs(result[0].hx - 4 * leaf * 0.5) < 1e-5)
    }

    @Test("Different colors stay separate")
    func differentColorsNotMerged() {
        let leaf: Float = 1.0 / 32.0
        let voxels = [
            voxel(leaf * 0.5, leaf * 0.5, leaf * 0.5, color: (255, 0, 0)),
            voxel(leaf * 1.5, leaf * 0.5, leaf * 0.5, color: (0, 255, 0))
        ]
        let result = GreedyMesher.merge(voxels: voxels, voxelSize: 0.05, chunkOrigin: .zero)
        #expect(result.count == 2)
    }

    @Test("2x2 same-color voxels merge into single cuboid")
    func twoByTwoGrid() {
        let leaf: Float = 1.0 / 32.0
        let voxels = [
            voxel(leaf * 0.5, leaf * 0.5, leaf * 0.5),
            voxel(leaf * 1.5, leaf * 0.5, leaf * 0.5),
            voxel(leaf * 0.5, leaf * 1.5, leaf * 0.5),
            voxel(leaf * 1.5, leaf * 1.5, leaf * 0.5)
        ]
        let result = GreedyMesher.merge(voxels: voxels, voxelSize: 0.05, chunkOrigin: .zero)
        #expect(result.count == 1)
        #expect(abs(result[0].hx - 2 * leaf * 0.5) < 1e-5)
        #expect(abs(result[0].hy - 2 * leaf * 0.5) < 1e-5)
    }

    @Test("Zero colorAndFlags remapped to avoid empty confusion")
    func zeroColorRemapped() {
        let leaf: Float = 1.0 / 32.0
        let v = PackedVoxel(position: SIMD3<Float>(leaf * 0.5, leaf * 0.5, leaf * 0.5),
                            color: (0, 0, 0), layer: .structure, classId: 0)
        let result = GreedyMesher.merge(voxels: [v], voxelSize: 0.05, chunkOrigin: .zero)
        #expect(result.count == 1)
        #expect(result[0].colorAndFlags == 1)
    }
}

// MARK: - PackedVoxel Tests

@Suite("PackedVoxel Tests")
struct PackedVoxelTests {

    @Test("PackedVoxel packs color and layer correctly")
    func packsColorAndLayer() {
        let v = PackedVoxel(position: .zero, color: (255, 128, 64), layer: .furniture, classId: 5)
        let r = UInt8((v.colorAndFlags >> 24) & 0xFF)
        let g = UInt8((v.colorAndFlags >> 16) & 0xFF)
        let b = UInt8((v.colorAndFlags >> 8) & 0xFF)
        #expect(r == 255)
        #expect(g == 128)
        #expect(b == 64)
        #expect(v.layer == .furniture)
        #expect(v.classId == 5)
    }

    @Test("PackedVoxel is 16 bytes")
    func sizeIs16Bytes() {
        #expect(MemoryLayout<PackedVoxel>.size == 16)
        #expect(MemoryLayout<PackedVoxel>.stride == 16)
    }

    @Test("MergedVoxel is 32 bytes")
    func mergedVoxelSize() {
        #expect(MemoryLayout<MergedVoxel>.size == 32)
        #expect(MemoryLayout<MergedVoxel>.stride == 32)
    }
}

// MARK: - KDTree Tests

@Suite("KDTree Tests")
struct KDTreeTests {

    @Test("Empty tree returns nil")
    func emptyTree() {
        var tree = KDTree()
        tree.build(from: [])
        #expect(tree.nearest(to: .zero) == nil)
        #expect(tree.nearestK(5, to: .zero).isEmpty)
        #expect(tree.radiusSearch(center: .zero, radius: 10).isEmpty)
    }

    @Test("Single point tree")
    func singlePoint() {
        var tree = KDTree()
        let p = SIMD3<Float>(1, 2, 3)
        tree.build(from: [p])
        #expect(tree.count == 1)
        #expect(tree.nearest(to: .zero) == p)
    }

    @Test("Nearest of two points")
    func nearestOfTwo() {
        var tree = KDTree()
        tree.build(from: [SIMD3<Float>(1, 0, 0), SIMD3<Float>(10, 0, 0)])
        #expect(tree.nearest(to: SIMD3<Float>(0.5, 0, 0)) == SIMD3<Float>(1, 0, 0))
    }

    @Test("Nearest among 100 deterministic points")
    func nearestAmong100() {
        var tree = KDTree()
        var pts = [SIMD3<Float>]()
        for i in 0..<100 { pts.append(SIMD3<Float>(Float(i), Float(i % 10), Float(i % 5))) }
        tree.build(from: pts)
        let query = SIMD3<Float>(15.1, 5.1, 0.1)
        let result = tree.nearest(to: query)
        let expected = pts.min(by: {
            simd_distance_squared($0, query) < simd_distance_squared($1, query)
        })
        #expect(result == expected)
    }

    @Test("K-nearest returns k results sorted by distance")
    func kNearest() {
        var tree = KDTree()
        tree.build(from: [[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[10,0,0],[20,0,0]])
        let result = tree.nearestK(3, to: .zero)
        #expect(result.count == 3)
        #expect(result[0] == SIMD3<Float>(1, 0, 0))
        #expect(result[1] == SIMD3<Float>(2, 0, 0))
        #expect(result[2] == SIMD3<Float>(3, 0, 0))
    }

    @Test("Radius search finds correct points")
    func radiusSearch() {
        var tree = KDTree()
        tree.build(from: [[0.5,0,0],[1.0,0,0],[1.5,0,0],[5.0,0,0]])
        let found = tree.radiusSearch(center: .zero, radius: 1.5)
        #expect(found.count == 3)
        #expect(!found.contains(SIMD3<Float>(5, 0, 0)))
    }
}

// MARK: - BotMapWorld Tests

@Suite("BotMapWorld Tests")
struct BotMapWorldTests {

    @Test("Initial voxel count is zero")
    func initialCountZero() async {
        let world = BotMapWorld(name: "test")
        let count = await world.voxelCount
        #expect(count == 0)
    }

    @Test("Inserting positions increases voxel count")
    func insertIncreasesCount() async {
        let world = BotMapWorld(name: "test")
        let positions: [SIMD3<Float>] = stride(from: Float(0), to: 10, by: 0.5).map {
            SIMD3<Float>($0, $0, 0.5)
        }
        await world.insertVoxelBatch(positions)
        let count = await world.voxelCount
        #expect(count > 0)
    }

    @Test("Clear removes all voxels")
    func clearRemovesAll() async {
        let world = BotMapWorld(name: "test")
        await world.insertVoxelBatch([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        await world.clear()
        let count = await world.voxelCount
        #expect(count == 0)
    }
}

// MARK: - BotMapSerializer Round-trip Tests

@Suite("BotMapSerializer Tests")
struct BotMapSerializerTests {

    @Test("Round-trip save and load preserves voxel count")
    func roundTrip() async throws {
        let world = BotMapWorld(name: "roundtrip-test")
        // Insert some voxels in a grid
        var positions = [SIMD3<Float>]()
        for x in stride(from: Float(0.5), to: 5.0, by: 1.0) {
            for y in stride(from: Float(0.5), to: 5.0, by: 1.0) {
                positions.append(SIMD3<Float>(x, y, 0.5))
            }
        }
        await world.insertVoxelBatch(positions)
        let originalCount = await world.voxelCount
        #expect(originalCount > 0)

        // Save
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxelkit-test-\(UUID().uuidString).botmap")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await world.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        // Load
        let loaded = try await BotMapWorld.load(from: tempURL)
        let loadedCount = await loaded.voxelCount
        let loadedName  = await loaded.name

        #expect(loadedCount == originalCount)
        #expect(loadedName == "roundtrip-test")
    }

    @Test("Load from invalid file throws error")
    func loadInvalidFile() async {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("invalid-\(UUID().uuidString).botmap")
        try? "not a botmap file".write(to: tempURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        do {
            _ = try await BotMapWorld.load(from: tempURL)
            #expect(Bool(false), "Should have thrown an error")
        } catch BotMapError.invalidMagic {
            // Expected
        } catch {
            #expect(Bool(false), "Wrong error type: \(error)")
        }
    }
}
