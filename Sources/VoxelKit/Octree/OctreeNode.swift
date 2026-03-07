import Foundation

/// A single node in the occupancy octree.
///
/// Either a **leaf** (no children, holds occupancy data) or an **interior** node
/// with up to 8 children. Uses Bayesian log-odds for probabilistic occupancy.
///
/// Child index layout: `z*4 + y*2 + x` — each bit selects high/low half of parent.
public final class OctreeNode: @unchecked Sendable {
    /// Log-odds occupancy. 0 = unknown, positive = occupied, negative = free.
    public var logOdds: Float = 0.0

    /// Semantic classification layer
    public var layer: MapLayer = .structure

    /// Object class identifier (0-63), matching PackedVoxel's 6-bit classId field.
    /// See `ObjectPrior.catalog` for known class mappings.
    public var classId: UInt8 = 0

    /// Packed color (R, G, B)
    public var color: (r: UInt8, g: UInt8, b: UInt8) = (128, 128, 128)

    /// Bitmask: which robots/agents have observed this node (up to 64)
    public var observerMask: UInt64 = 0

    /// Seconds since mapping session start (for dynamic layer TTL)
    public var lastObserved: UInt32 = 0

    // MARK: - Subtractive fields

    /// Number of sensor observations that contributed to this node.
    /// 0 = never observed ("unknown-solid" in subtractive mode).
    public var observationCount: UInt16 = 0

    /// Truncated signed distance to nearest surface.
    /// +∞ = no observation, 0 = surface, + = inside solid, - = free space.
    public var signedDistance: Float = .greatestFiniteMagnitude

    /// Bitmask of octant directions with conflicting occupancy evidence.
    /// Non-zero = candidate for subdivision in next refinement pass.
    public var subdivisionHint: UInt8 = 0

    /// Children: nil = leaf, 8-element array = interior node.
    public var children: ContiguousArray<OctreeNode?>?

    public init() {}

    // MARK: - Computed

    public var isLeaf: Bool { children == nil }

    /// Occupancy probability from log-odds
    public var occupancyProbability: Float {
        1.0 / (1.0 + exp(-logOdds))
    }

    /// Whether this voxel is considered occupied (positive log-odds)
    public var isOccupied: Bool { logOdds > 0 }

    /// Whether this node has never been observed by any sensor.
    public var isUnobserved: Bool { observationCount == 0 }

    /// Subtractive model: unobserved = conservatively solid, or positively occupied.
    public var isConservativelyOccupied: Bool { isUnobserved || logOdds > 0 }

    /// Whether this leaf needs subdivision (conflicting evidence at current scale).
    public var needsRefinement: Bool { subdivisionHint != 0 && children == nil }

    /// Number of observers that have seen this node
    public var observerCount: Int { observerMask.nonzeroBitCount }

    // MARK: - DFS Serialization

    /// Serialize this node and its subtree to binary data (DFS).
    ///
    /// Leaf data: [Float16 logOdds (2B)] [UInt8 flags (1B)] [RGB (3B)] [Float16 signedDistance (2B)] = 8 bytes.
    /// Interior: [UInt8 childBitmap] + recursive children.
    /// At maxDepth: leaf data only (no bitmap).
    public func serialize(depth: Int, maxDepth: Int, into data: inout Data) {
        if depth == maxDepth || isLeaf {
            if depth < maxDepth {
                data.append(0x00) // pruned leaf — bitmap = 0
            }
            writeLeafData(into: &data)
            return
        }

        guard let children = children else {
            if depth < maxDepth { data.append(0x00) }
            writeLeafData(into: &data)
            return
        }

        var bitmap: UInt8 = 0
        for i in 0..<8 {
            if children[i] != nil { bitmap |= (1 << i) }
        }
        data.append(bitmap)

        for i in 0..<8 {
            if let child = children[i] {
                child.serialize(depth: depth + 1, maxDepth: maxDepth, into: &data)
            }
        }
    }

    /// Deserialize a node and its subtree from binary data (DFS).
    public static func deserialize(from buffer: UnsafeRawBufferPointer, offset: inout Int,
                                    depth: Int, maxDepth: Int) -> OctreeNode? {
        let node = OctreeNode()

        if depth == maxDepth {
            guard readLeafData(from: buffer, offset: &offset, into: node) else { return nil }
            return node
        }

        guard offset < buffer.count else { return nil }
        let bitmap = buffer[offset]
        offset += 1

        if bitmap == 0x00 {
            guard readLeafData(from: buffer, offset: &offset, into: node) else { return nil }
            return node
        }

        node.children = ContiguousArray<OctreeNode?>(repeating: nil, count: 8)
        for i in 0..<8 {
            if bitmap & (1 << i) != 0 {
                guard let child = deserialize(from: buffer, offset: &offset,
                                              depth: depth + 1, maxDepth: maxDepth) else {
                    return nil
                }
                node.children![i] = child
            }
        }
        return node
    }

    // MARK: - Leaf Data (8 bytes)

    private func writeLeafData(into data: inout Data) {
        var halfLogOdds = Float16(logOdds)
        withUnsafeBytes(of: &halfLogOdds) { data.append(contentsOf: $0) }
        let flags = (UInt8(layer.rawValue) << 6) | (classId & 0x3F)
        data.append(flags)
        data.append(color.r)
        data.append(color.g)
        data.append(color.b)
        // signedDistance as Float16 (2 bytes) — preserves subtractive refinement state
        var halfSD: Float16 = signedDistance.isFinite ? Float16(signedDistance) : .greatestFiniteMagnitude
        withUnsafeBytes(of: &halfSD) { data.append(contentsOf: $0) }
    }

    private static func readLeafData(from buffer: UnsafeRawBufferPointer,
                                     offset: inout Int, into node: OctreeNode) -> Bool {
        guard offset + 8 <= buffer.count else { return false }
        let halfLogOdds = buffer.loadUnaligned(fromByteOffset: offset, as: Float16.self)
        node.logOdds = Float(halfLogOdds)
        offset += 2
        let flags = buffer[offset]
        node.layer = MapLayer(rawValue: flags >> 6) ?? .structure
        node.classId = flags & 0x3F
        offset += 1
        node.color = (buffer[offset], buffer[offset + 1], buffer[offset + 2])
        offset += 3
        let halfSD = buffer.loadUnaligned(fromByteOffset: offset, as: Float16.self)
        node.signedDistance = Float(halfSD)
        offset += 2
        return true
    }
}
