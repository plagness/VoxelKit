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
}
