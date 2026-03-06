import Foundation
import simd

/// A chunk needing refinement via high-quality depth estimation.
public struct RefinementTask: Sendable {
    public let chunkKey: ChunkKey
    /// Average observation count — lower = needs more refinement.
    public let avgObservationCount: Float
    /// Priority (lower = more urgent).
    public var priority: Float { avgObservationCount }

    public init(chunkKey: ChunkKey, avgObservationCount: Float) {
        self.chunkKey = chunkKey
        self.avgObservationCount = avgObservationCount
    }
}

/// Background queue that schedules low-quality chunks for re-processing
/// through Apple Depth Pro when the robot is idle or revisiting areas.
///
/// Thread-safety: access from the owning actor only.
public final class RefinementQueue: @unchecked Sendable {

    /// Maximum tasks in the queue.
    public var maxQueueSize: Int = 500

    /// Minimum observation count below which a chunk is considered for refinement.
    public var refinementThreshold: Float = 3.0

    /// Queued tasks, sorted by priority (most urgent first).
    private var queue: [RefinementTask] = []

    /// Set of keys already in queue (for dedup).
    private var queuedKeys: Set<ChunkKey> = []

    /// Statistics.
    public private(set) var totalEnqueued: Int = 0
    public private(set) var totalProcessed: Int = 0

    public init() {}

    // MARK: - Enqueue

    /// Scan chunks and enqueue those needing refinement.
    public func scanForRefinement(store: ChunkedOctreeStore) {
        for (key, chunk) in store.allChunks {
            guard !queuedKeys.contains(key) else { continue }

            // Estimate average observation quality
            let voxels = chunk.tree.collectOccupiedVoxels()
            guard !voxels.isEmpty else { continue }

            // Use voxel count as rough proxy — sparse chunks need refinement
            let density = Float(voxels.count)
            if density < refinementThreshold {
                enqueue(RefinementTask(chunkKey: key, avgObservationCount: density))
            }
        }
    }

    /// Enqueue a specific task.
    public func enqueue(_ task: RefinementTask) {
        guard !queuedKeys.contains(task.chunkKey) else { return }
        guard queue.count < maxQueueSize else { return }

        queue.append(task)
        queuedKeys.insert(task.chunkKey)
        queue.sort { $0.priority < $1.priority }
        totalEnqueued += 1
    }

    // MARK: - Dequeue

    /// Get the next task to process.
    public func dequeue() -> RefinementTask? {
        guard !queue.isEmpty else { return nil }
        let task = queue.removeFirst()
        queuedKeys.remove(task.chunkKey)
        totalProcessed += 1
        return task
    }

    /// Peek at the next task without removing it.
    public var next: RefinementTask? { queue.first }

    /// Number of pending tasks.
    public var count: Int { queue.count }

    /// Whether the queue is empty.
    public var isEmpty: Bool { queue.isEmpty }

    // MARK: - Maintenance

    /// Remove tasks for chunks that no longer exist.
    public func prune(existingKeys: Set<ChunkKey>) {
        queue.removeAll { !existingKeys.contains($0.chunkKey) }
        queuedKeys = Set(queue.map(\.chunkKey))
    }

    /// Clear the queue.
    public func clear() {
        queue.removeAll()
        queuedKeys.removeAll()
    }
}
