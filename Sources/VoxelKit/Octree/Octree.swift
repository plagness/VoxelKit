import Foundation
import simd

/// Multi-resolution octree for 3D occupancy mapping.
///
/// Probabilistic occupancy using Bayesian log-odds fusion (OctoMap approach).
/// Each leaf stores a log-odds value updated via Bayesian fusion.
public final class Octree: @unchecked Sendable {
    public let maxDepth: Int
    public let resolution: Float    // leaf size in metres
    public let origin: SIMD3<Float> // min corner of the root volume
    public let rootSize: Float      // root node edge length

    public private(set) var root = OctreeNode()
    public private(set) var nodeCount: Int = 1
    public private(set) var occupiedLeafCount: Int = 0

    // Log-odds constants (OctoMap defaults)
    public static let logOddsHit: Float  =  0.85
    public static let logOddsMiss: Float = -0.7
    public static let logOddsMin: Float  = -2.0
    public static let logOddsMax: Float  =  3.5

    public init(resolution: Float = 0.05, origin: SIMD3<Float> = .zero, rootSize: Float = 1.0) {
        self.resolution = resolution
        self.origin = origin
        self.rootSize = rootSize
        self.maxDepth = max(1, Int(ceil(log2(rootSize / resolution))))
    }

    // MARK: - Update

    /// Bayesian log-odds occupancy update at a world-space point.
    public func updateOccupancy(at point: SIMD3<Float>, hit: Bool, robotIndex: Int = 0,
                                timestamp: UInt32 = 0, layer: MapLayer = .structure,
                                color: (UInt8, UInt8, UInt8) = (128, 128, 128)) {
        let localPoint = point - origin
        guard all(localPoint .>= .zero) && all(localPoint .< SIMD3<Float>(repeating: rootSize)) else {
            return
        }

        var node = root
        var nodeOrigin = SIMD3<Float>.zero
        var nodeSize = rootSize

        for _ in 0..<maxDepth {
            let halfSize = nodeSize * 0.5
            let childIdx = childIndex(point: localPoint, nodeOrigin: nodeOrigin, halfSize: halfSize)

            if node.children == nil {
                node.children = ContiguousArray<OctreeNode?>(repeating: nil, count: 8)
            }

            if node.children![childIdx] == nil {
                node.children![childIdx] = OctreeNode()
                nodeCount += 1
            }

            nodeOrigin = nodeOrigin + childOffset(index: childIdx, halfSize: halfSize)
            nodeSize = halfSize
            node = node.children![childIdx]!
        }

        let wasOccupied = node.isOccupied
        let delta = hit ? Self.logOddsHit : Self.logOddsMiss
        node.logOdds = max(Self.logOddsMin, min(Self.logOddsMax, node.logOdds + delta))
        node.layer = layer
        node.color = color
        node.lastObserved = timestamp
        if hit { node.observationCount = node.observationCount < UInt16.max ? node.observationCount &+ 1 : .max }
        if robotIndex < 64 { node.observerMask |= (1 << robotIndex) }

        if !wasOccupied && node.isOccupied {
            occupiedLeafCount += 1
        } else if wasOccupied && !node.isOccupied {
            occupiedLeafCount -= 1
        }
    }

    // MARK: - Subtractive Update

    /// Subtractive occupancy update: inserts at `targetDepth` (not always maxDepth).
    /// Supports lazy subdivision — only creates children when conflicting evidence exists.
    ///
    /// - Parameters:
    ///   - point: World-space position.
    ///   - hit: true = surface detected, false = free space (carving).
    ///   - targetDepth: How deep to insert (LOD-aware, may be < maxDepth).
    ///   - robotIndex: Robot observer index.
    ///   - timestamp: Session-relative timestamp.
    ///   - color: RGB color for hits.
    public func updateSubtractive(at point: SIMD3<Float>, hit: Bool,
                                   targetDepth: Int, robotIndex: Int = 0,
                                   timestamp: UInt32 = 0,
                                   color: (UInt8, UInt8, UInt8) = (128, 128, 128)) {
        let localPoint = point - origin
        guard all(localPoint .>= .zero) && all(localPoint .< SIMD3<Float>(repeating: rootSize)) else {
            return
        }

        let effectiveDepth = min(targetDepth, maxDepth)
        var node = root
        var nodeOrigin = SIMD3<Float>.zero
        var nodeSize = rootSize

        for depth in 0..<effectiveDepth {
            let halfSize = nodeSize * 0.5
            let childIdx = childIndex(point: localPoint, nodeOrigin: nodeOrigin, halfSize: halfSize)

            if node.children == nil {
                // Only subdivide if this node has conflicting evidence
                // or if we haven't reached target depth yet
                let shouldSubdivide: Bool
                if node.isUnobserved {
                    // Unobserved node — subdivide to reach target depth
                    shouldSubdivide = true
                } else if (hit && !node.isOccupied) || (!hit && node.isConservativelyOccupied) {
                    // Conflicting evidence — need finer resolution
                    shouldSubdivide = true
                    node.subdivisionHint |= UInt8(1 << (childIdx & 7))
                } else if depth < effectiveDepth - 1 {
                    shouldSubdivide = true
                } else {
                    shouldSubdivide = false
                }

                guard shouldSubdivide else { break }
                node.children = ContiguousArray<OctreeNode?>(repeating: nil, count: 8)
            }

            if node.children![childIdx] == nil {
                let child = OctreeNode()
                // Inherit parent state for subtractive: new children start as parent's state
                child.logOdds = node.logOdds
                child.color = node.color
                child.layer = node.layer
                child.classId = node.classId
                child.observationCount = node.observationCount
                node.children![childIdx] = child
                nodeCount += 1
            }

            nodeOrigin = nodeOrigin + childOffset(index: childIdx, halfSize: halfSize)
            nodeSize = halfSize
            node = node.children![childIdx]!
        }

        // Apply update at the reached node
        let wasOccupied = node.isOccupied
        let delta = hit ? Self.logOddsHit : Self.logOddsMiss
        node.logOdds = max(Self.logOddsMin, min(Self.logOddsMax, node.logOdds + delta))
        node.observationCount = node.observationCount < UInt16.max ? node.observationCount &+ 1 : .max
        node.lastObserved = timestamp
        if robotIndex < 64 { node.observerMask |= (1 << robotIndex) }

        if hit {
            node.color = color
            node.signedDistance = 0  // Surface
        } else {
            node.signedDistance = -1  // Free space marker
        }

        if !wasOccupied && node.isOccupied {
            occupiedLeafCount += 1
        } else if wasOccupied && !node.isOccupied {
            occupiedLeafCount -= 1
        }
    }

    // MARK: - Query

    /// Collect all occupied voxels as PackedVoxel array.
    /// When `includeUnobserved` is true, unobserved nodes are treated as conservatively
    /// occupied (subtractive mode: unknown = solid).
    public func collectOccupiedVoxels(minLogOdds: Float = 0.0,
                                       minObservationCount: UInt16 = 0,
                                       includeUnobserved: Bool = false) -> [PackedVoxel] {
        var result = [PackedVoxel]()
        result.reserveCapacity(max(occupiedLeafCount, 64))
        collectRecursive(node: root, origin: origin, size: rootSize,
                         minLogOdds: minLogOdds, minObsCount: minObservationCount,
                         includeUnobserved: includeUnobserved, result: &result)
        return result
    }

    /// Collect voxels at a specific depth for LOD queries.
    public func collectAtDepth(_ targetDepth: Int, minLogOdds: Float = 0.0,
                                minObservationCount: UInt16 = 0,
                                includeUnobserved: Bool = false) -> [PackedVoxel] {
        var result = [PackedVoxel]()
        collectAtDepthRecursive(node: root, origin: origin, size: rootSize,
                                currentDepth: 0, targetDepth: targetDepth,
                                minLogOdds: minLogOdds, minObsCount: minObservationCount,
                                includeUnobserved: includeUnobserved, result: &result)
        return result
    }

    /// Collect occupied voxel world-space positions (for ICP / serialization).
    public func collectOccupiedPositions(minLogOdds: Float = 0.0) -> [SIMD3<Float>] {
        var result = [SIMD3<Float>]()
        result.reserveCapacity(max(occupiedLeafCount, 64))
        collectPositionsRecursive(node: root, origin: origin, size: rootSize,
                                  minLogOdds: minLogOdds, result: &result)
        return result
    }

    // MARK: - Pruning

    @discardableResult
    public func prune() -> Int {
        var pruned = 0
        pruneRecursive(node: root, pruned: &pruned)
        return pruned
    }

    public func expireDynamicLayer(olderThan threshold: UInt32) {
        expireRecursive(node: root, threshold: threshold)
    }

    // MARK: - Serialization

    public func replaceRoot(_ newRoot: OctreeNode) {
        root = newRoot
        var nodes = 0, occupied = 0
        countRecursive(node: root, nodes: &nodes, occupied: &occupied)
        nodeCount = nodes
        occupiedLeafCount = occupied
    }

    // MARK: - Private: Child math

    private func childIndex(point: SIMD3<Float>, nodeOrigin: SIMD3<Float>, halfSize: Float) -> Int {
        let mid = nodeOrigin + SIMD3<Float>(repeating: halfSize)
        var idx = 0
        if point.x >= mid.x { idx |= 1 }
        if point.y >= mid.y { idx |= 2 }
        if point.z >= mid.z { idx |= 4 }
        return idx
    }

    private func childOffset(index: Int, halfSize: Float) -> SIMD3<Float> {
        SIMD3<Float>(
            (index & 1) != 0 ? halfSize : 0,
            (index & 2) != 0 ? halfSize : 0,
            (index & 4) != 0 ? halfSize : 0
        )
    }

    // MARK: - Private: Collection

    private func collectRecursive(node: OctreeNode, origin: SIMD3<Float>, size: Float,
                                  minLogOdds: Float, minObsCount: UInt16 = 0,
                                  includeUnobserved: Bool = false,
                                  result: inout [PackedVoxel]) {
        if node.isLeaf {
            let passesObs = minObsCount == 0 || node.observationCount >= minObsCount
            let include = (node.logOdds >= minLogOdds && passesObs) ||
                          (includeUnobserved && node.isConservativelyOccupied)
            if include {
                let center = origin + SIMD3<Float>(repeating: size * 0.5)
                result.append(PackedVoxel(position: center, color: node.color, layer: node.layer, classId: node.classId))
            }
            return
        }
        guard let children = node.children else { return }
        let halfSize = size * 0.5
        for i in 0..<8 {
            guard let child = children[i] else { continue }
            collectRecursive(node: child, origin: origin + childOffset(index: i, halfSize: halfSize),
                             size: halfSize, minLogOdds: minLogOdds, minObsCount: minObsCount,
                             includeUnobserved: includeUnobserved, result: &result)
        }
    }

    private func collectAtDepthRecursive(node: OctreeNode, origin: SIMD3<Float>, size: Float,
                                         currentDepth: Int, targetDepth: Int,
                                         minLogOdds: Float, minObsCount: UInt16 = 0,
                                         includeUnobserved: Bool = false,
                                         result: inout [PackedVoxel]) {
        if currentDepth >= targetDepth || node.isLeaf {
            let passesObs = minObsCount == 0 || node.observationCount >= minObsCount
            let include = (node.logOdds >= minLogOdds && passesObs) ||
                          (includeUnobserved && node.isConservativelyOccupied)
            if include {
                let center = origin + SIMD3<Float>(repeating: size * 0.5)
                result.append(PackedVoxel(position: center, color: node.color, layer: node.layer, classId: node.classId))
            }
            return
        }
        guard let children = node.children else { return }
        let halfSize = size * 0.5
        for i in 0..<8 {
            guard let child = children[i] else { continue }
            collectAtDepthRecursive(node: child, origin: origin + childOffset(index: i, halfSize: halfSize),
                                    size: halfSize, currentDepth: currentDepth + 1, targetDepth: targetDepth,
                                    minLogOdds: minLogOdds, minObsCount: minObsCount,
                                    includeUnobserved: includeUnobserved, result: &result)
        }
    }

    private func collectPositionsRecursive(node: OctreeNode, origin: SIMD3<Float>, size: Float,
                                           minLogOdds: Float, result: inout [SIMD3<Float>]) {
        if node.isLeaf {
            if node.logOdds >= minLogOdds {
                result.append(origin + SIMD3<Float>(repeating: size * 0.5))
            }
            return
        }
        guard let children = node.children else { return }
        let halfSize = size * 0.5
        for i in 0..<8 {
            guard let child = children[i] else { continue }
            collectPositionsRecursive(node: child, origin: origin + childOffset(index: i, halfSize: halfSize),
                                      size: halfSize, minLogOdds: minLogOdds, result: &result)
        }
    }

    // MARK: - Private: Pruning

    private func pruneRecursive(node: OctreeNode, pruned: inout Int) {
        guard let children = node.children else { return }
        for i in 0..<8 { if let child = children[i] { pruneRecursive(node: child, pruned: &pruned) } }

        var allLeaves = true, allSameState = true
        var firstOccupied: Bool?
        for i in 0..<8 {
            guard let child = children[i] else { continue }
            if !child.isLeaf { allLeaves = false; break }
            if let first = firstOccupied { if child.isOccupied != first { allSameState = false } }
            else { firstOccupied = child.isOccupied }
        }

        if allLeaves && allSameState, let firstChild = children.compactMap({ $0 }).first {
            node.logOdds = firstChild.logOdds
            node.layer = firstChild.layer
            node.color = firstChild.color
            node.observerMask = firstChild.observerMask
            node.lastObserved = firstChild.lastObserved
            node.observationCount = firstChild.observationCount
            node.signedDistance = firstChild.signedDistance
            node.subdivisionHint = 0
            let childCount = children.compactMap({ $0 }).count
            node.children = nil
            nodeCount -= childCount
            pruned += childCount
        }
    }

    private func countRecursive(node: OctreeNode, nodes: inout Int, occupied: inout Int) {
        nodes += 1
        if node.isLeaf { if node.isOccupied { occupied += 1 }; return }
        guard let children = node.children else { return }
        for i in 0..<8 { if let child = children[i] { countRecursive(node: child, nodes: &nodes, occupied: &occupied) } }
    }

    private func expireRecursive(node: OctreeNode, threshold: UInt32) {
        if node.isLeaf {
            if node.layer == .dynamic && node.lastObserved < threshold {
                node.logOdds = 0
                if node.isOccupied { occupiedLeafCount -= 1 }
            }
            return
        }
        guard let children = node.children else { return }
        for i in 0..<8 { if let child = children[i] { expireRecursive(node: child, threshold: threshold) } }
    }
}
