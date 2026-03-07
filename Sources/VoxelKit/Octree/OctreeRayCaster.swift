import simd

/// Result of a ray carving operation.
public struct RayCarveResult: Sendable {
    /// Number of nodes carved (marked as free space).
    public var carvedCount: Int = 0
    /// Number of nodes marked as occupied (surface hit).
    public var hitCount: Int = 0
}

/// Casts rays through an octree to carve free space and mark surface hits.
///
/// Uses a simplified DDA-style traversal: steps along the ray at the resolution
/// of the target LOD depth, updating nodes along the path.
/// Before hitDistance: carve (mark free). At hitDistance: mark occupied.
public enum OctreeRayCaster {

    /// Carve a single ray through a chunked octree store.
    ///
    /// - Parameters:
    ///   - origin: Ray origin (camera position in world space).
    ///   - direction: Normalized ray direction.
    ///   - hitDistance: Distance to surface hit (nil = infinite free ray, carve to maxDistance).
    ///   - maxDistance: Maximum ray length to process.
    ///   - store: The chunked octree store to modify.
    ///   - cameraPos: Camera position for LOD computation.
    ///   - robotIndex: Observer robot index.
    ///   - timestamp: Session timestamp.
    ///   - color: Color for hit voxel.
    /// - Returns: Carving statistics.
    @discardableResult
    public static func carveRay(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        hitDistance: Float?,
        maxDistance: Float = 10.0,
        store: ChunkedOctreeStore,
        cameraPos: SIMD3<Float>,
        robotIndex: Int = 0,
        timestamp: UInt32 = 0,
        color: (UInt8, UInt8, UInt8) = (128, 128, 128)
    ) -> RayCarveResult {
        var result = RayCarveResult()

        let effectiveMax = hitDistance ?? maxDistance
        guard effectiveMax > 0 else { return result }

        // Adaptive step size matched to LOD bands (same as ChunkedOctreeStore):
        //   0-10m: 3.125cm (depth 5) — full detail carving
        //  10-30m: 6.25cm (depth 4)  — half resolution
        //  30-80m: 12.5cm (depth 3)  — quarter resolution
        //  80m+:   50cm (depth 1)    — coarse only
        let midpointDist = simd_length(origin + direction * (effectiveMax * 0.5) - cameraPos)
        let stepSize: Float
        if midpointDist < 10 {
            stepSize = 0.03125  // depth 5: 3.125cm
        } else if midpointDist < 30 {
            stepSize = 0.0625   // depth 4: 6.25cm
        } else if midpointDist < 80 {
            stepSize = 0.125    // depth 3: 12.5cm
        } else {
            stepSize = 0.5      // depth 1: 50cm
        }

        // Carve free space along the ray (before hit).
        // Stop 2 voxels before the surface to avoid punching holes through thin walls.
        let stopMargin = stepSize * 2
        let carveEnd = hitDistance.map { max(0, $0 - stopMargin) } ?? maxDistance
        let carveSteps = max(0, Int(carveEnd / stepSize))
        // Limit steps: 512 for close rays (fine detail), 128 for distant
        let stepLimit = midpointDist < 10 ? 512 : 256
        let maxSteps = min(carveSteps, stepLimit)

        if maxSteps > 0 {
            let strideCount = max(1, carveSteps / maxSteps)
            for i in stride(from: 1, through: carveSteps, by: strideCount) {
                let t = Float(i) * stepSize
                let pos = origin + direction * t
                store.insertSubtractive(at: pos, hit: false, cameraPos: cameraPos,
                                        robotIndex: robotIndex, timestamp: timestamp)
                result.carvedCount += 1
            }
        }

        // Mark hit point as occupied
        if let hitDist = hitDistance, hitDist <= maxDistance {
            let hitPos = origin + direction * hitDist
            store.insertSubtractive(at: hitPos, hit: true, cameraPos: cameraPos,
                                    robotIndex: robotIndex, timestamp: timestamp, color: color)
            result.hitCount += 1
        }

        return result
    }

    /// Batch carve multiple rays through a chunked octree store.
    ///
    /// - Parameters:
    ///   - rays: Array of (direction, hitDistance?) pairs. Direction must be normalized.
    ///   - origin: Common ray origin (camera position).
    ///   - maxDistance: Maximum ray length.
    ///   - store: The chunked octree store.
    ///   - robotIndex: Observer robot index.
    ///   - timestamp: Session timestamp.
    ///   - colors: Per-ray hit colors (same count as rays).
    /// - Returns: Aggregate carving statistics.
    @discardableResult
    public static func carveRays(
        _ rays: [(direction: SIMD3<Float>, hitDistance: Float?, color: (UInt8, UInt8, UInt8))],
        origin: SIMD3<Float>,
        maxDistance: Float = 10.0,
        store: ChunkedOctreeStore,
        robotIndex: Int = 0,
        timestamp: UInt32 = 0
    ) -> RayCarveResult {
        var total = RayCarveResult()
        for ray in rays {
            let r = carveRay(origin: origin, direction: ray.direction,
                             hitDistance: ray.hitDistance, maxDistance: maxDistance,
                             store: store, cameraPos: origin,
                             robotIndex: robotIndex, timestamp: timestamp,
                             color: ray.color)
            total.carvedCount += r.carvedCount
            total.hitCount += r.hitCount
        }
        return total
    }
}
