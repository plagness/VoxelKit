import simd
import Foundation

/// Balanced static KD-tree for 3D nearest-neighbor queries.
///
/// - Build: O(N log N)
/// - Query: O(log N) average, O(N) worst case
/// - Space: O(N)
///
/// Not thread-safe. Build once, then query concurrently (read-only).
public struct KDTree: SpatialIndex, Sendable {

    private enum Node: Sendable {
        case leaf(SIMD3<Float>)
        case split(axis: Int, median: Float, left: Int, right: Int)
    }

    private var nodes: [Node] = []
    public private(set) var count: Int = 0

    public init() {}

    // MARK: - Build

    public mutating func build(from points: [SIMD3<Float>]) {
        nodes = []
        count = points.count
        guard !points.isEmpty else { return }
        var pts = points
        nodes.reserveCapacity(2 * points.count)
        _ = buildNode(pts: &pts, start: 0, end: pts.count)
    }

    // MARK: - Queries

    public func nearest(to query: SIMD3<Float>) -> SIMD3<Float>? {
        guard !nodes.isEmpty else { return nil }
        var best = (distSq: Float.greatestFiniteMagnitude, point: SIMD3<Float>.zero)
        searchNearest(nodeIdx: 0, query: query, best: &best)
        return best.distSq < .greatestFiniteMagnitude ? best.point : nil
    }

    public func nearestK(_ k: Int, to query: SIMD3<Float>) -> [SIMD3<Float>] {
        guard !nodes.isEmpty, k > 0 else { return [] }
        var heap: [(Float, SIMD3<Float>)] = []
        heap.reserveCapacity(k + 1)
        searchKNearest(nodeIdx: 0, query: query, k: k, heap: &heap)
        return heap.sorted(by: { $0.0 < $1.0 }).map(\.1)
    }

    public func radiusSearch(center: SIMD3<Float>, radius: Float) -> [SIMD3<Float>] {
        guard !nodes.isEmpty else { return [] }
        var result = [SIMD3<Float>]()
        searchRadius(nodeIdx: 0, center: center, radiusSq: radius * radius, result: &result)
        return result
    }

    // MARK: - Private: Build

    private mutating func buildNode(pts: inout [SIMD3<Float>], start: Int, end: Int) -> Int {
        if end - start == 1 {
            let idx = nodes.count
            nodes.append(.leaf(pts[start]))
            return idx
        }

        let axis = longestAxis(pts: pts, start: start, end: end)
        let mid = (start + end) / 2
        partialSort(&pts, start: start, end: end, k: mid, axis: axis)
        let medianVal = pts[mid][axis]

        let nodeIdx = nodes.count
        nodes.append(.split(axis: 0, median: 0, left: 0, right: 0))

        let leftIdx  = buildNode(pts: &pts, start: start, end: mid)
        let rightIdx = buildNode(pts: &pts, start: mid,   end: end)

        nodes[nodeIdx] = .split(axis: axis, median: medianVal, left: leftIdx, right: rightIdx)
        return nodeIdx
    }

    private func longestAxis(pts: [SIMD3<Float>], start: Int, end: Int) -> Int {
        var minV = pts[start], maxV = pts[start]
        for i in (start+1)..<end { minV = simd_min(minV, pts[i]); maxV = simd_max(maxV, pts[i]) }
        let range = maxV - minV
        if range.x >= range.y && range.x >= range.z { return 0 }
        if range.y >= range.z { return 1 }
        return 2
    }

    private func partialSort(_ pts: inout [SIMD3<Float>], start: Int, end: Int, k: Int, axis: Int) {
        var lo = start, hi = end - 1
        while lo < hi {
            let pivot = pts[(lo + hi) / 2][axis]
            var i = lo, j = hi
            while i <= j {
                while pts[i][axis] < pivot { i += 1 }
                while pts[j][axis] > pivot { j -= 1 }
                if i <= j { pts.swapAt(i, j); i += 1; j -= 1 }
            }
            if j < k { lo = i }
            if i > k { hi = j }
            if lo >= hi { break }
        }
    }

    // MARK: - Private: Nearest

    private func searchNearest(nodeIdx: Int, query: SIMD3<Float>,
                                best: inout (distSq: Float, point: SIMD3<Float>)) {
        guard nodeIdx < nodes.count else { return }
        switch nodes[nodeIdx] {
        case .leaf(let p):
            let d = simd_distance_squared(p, query)
            if d < best.distSq { best = (d, p) }
        case .split(let axis, let median, let left, let right):
            let diff = query[axis] - median
            let (near, far) = diff < 0 ? (left, right) : (right, left)
            searchNearest(nodeIdx: near, query: query, best: &best)
            if diff * diff < best.distSq {
                searchNearest(nodeIdx: far, query: query, best: &best)
            }
        }
    }

    // MARK: - Private: K-nearest

    private func searchKNearest(nodeIdx: Int, query: SIMD3<Float>,
                                 k: Int, heap: inout [(Float, SIMD3<Float>)]) {
        guard nodeIdx < nodes.count else { return }
        switch nodes[nodeIdx] {
        case .leaf(let p):
            heapPush(&heap, value: (simd_distance_squared(p, query), p), maxSize: k)
        case .split(let axis, let median, let left, let right):
            let diff = query[axis] - median
            let (near, far) = diff < 0 ? (left, right) : (right, left)
            searchKNearest(nodeIdx: near, query: query, k: k, heap: &heap)
            let maxDist = heap.count < k ? Float.greatestFiniteMagnitude : (heap.first?.0 ?? .greatestFiniteMagnitude)
            if diff * diff < maxDist {
                searchKNearest(nodeIdx: far, query: query, k: k, heap: &heap)
            }
        }
    }

    private func heapPush(_ heap: inout [(Float, SIMD3<Float>)],
                           value: (Float, SIMD3<Float>), maxSize: Int) {
        heap.append(value)
        var i = heap.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            if heap[parent].0 < heap[i].0 { heap.swapAt(parent, i); i = parent } else { break }
        }
        if heap.count > maxSize {
            heap.swapAt(0, heap.count - 1)
            heap.removeLast()
            var idx = 0
            while true {
                let l = 2 * idx + 1, r = 2 * idx + 2
                var largest = idx
                if l < heap.count && heap[l].0 > heap[largest].0 { largest = l }
                if r < heap.count && heap[r].0 > heap[largest].0 { largest = r }
                if largest == idx { break }
                heap.swapAt(idx, largest)
                idx = largest
            }
        }
    }

    // MARK: - Private: Radius

    private func searchRadius(nodeIdx: Int, center: SIMD3<Float>,
                               radiusSq: Float, result: inout [SIMD3<Float>]) {
        guard nodeIdx < nodes.count else { return }
        switch nodes[nodeIdx] {
        case .leaf(let p):
            if simd_distance_squared(p, center) <= radiusSq { result.append(p) }
        case .split(let axis, let median, let left, let right):
            let diff = center[axis] - median
            searchRadius(nodeIdx: diff < 0 ? left : right, center: center, radiusSq: radiusSq, result: &result)
            if diff * diff <= radiusSq {
                searchRadius(nodeIdx: diff < 0 ? right : left, center: center, radiusSq: radiusSq, result: &result)
            }
        }
    }
}

// MARK: - SIMD3<Float> axis subscript

private extension SIMD3<Float> {
    subscript(axis: Int) -> Float {
        switch axis {
        case 0: return x
        case 1: return y
        default: return z
        }
    }
}
