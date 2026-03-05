import simd

// MARK: - Merged Voxel

/// Per-instance GPU data for greedy-merged voxel rendering.
/// 32 bytes — must match Metal `MergedVoxel` struct exactly.
/// Layout: center(3×4=12) + halfSize(3×4=12) + colorAndFlags(4) + pad(4) = 32 bytes.
public struct MergedVoxel: Sendable {
    public var cx: Float
    public var cy: Float
    public var cz: Float
    public var hx: Float
    public var hy: Float
    public var hz: Float
    public var colorAndFlags: UInt32
    public var _pad: Float = 0

    public init(cx: Float, cy: Float, cz: Float,
                hx: Float, hy: Float, hz: Float,
                colorAndFlags: UInt32) {
        self.cx = cx; self.cy = cy; self.cz = cz
        self.hx = hx; self.hy = hy; self.hz = hz
        self.colorAndFlags = colorAndFlags
    }
}

// MARK: - Greedy Mesher

/// Merges adjacent same-color voxels into larger axis-aligned cuboids.
///
/// Algorithm: sweeps Z → Y → X, greedily extending runs/rows/slabs
/// of cells with the same `colorAndFlags`. Adjacent cells with the same color
/// collapse into a single instance, achieving 5–50× instance reduction.
public enum GreedyMesher {

    /// Merge PackedVoxels from one 1m³ chunk into MergedVoxels.
    ///
    /// - Parameters:
    ///   - voxels: World-space voxel centers from `Octree.collectOccupiedVoxels()`.
    ///   - voxelSize: Requested octree resolution (typically 0.05 m).
    ///   - chunkOrigin: World-space min corner of the chunk (`ChunkKey.worldOrigin`).
    public static func merge(
        voxels: [PackedVoxel],
        voxelSize: Float,
        chunkOrigin: SIMD3<Float>
    ) -> [MergedVoxel] {
        guard !voxels.isEmpty else { return [] }

        let maxDepth = max(1, Int(ceil(log2(1.0 / voxelSize))))
        let gridN = 1 << maxDepth
        let leafSize = 1.0 / Float(gridN)
        let invLeafSize = Float(gridN)
        let halfLeaf = leafSize * 0.5
        let gridN2 = gridN * gridN
        let total = gridN2 * gridN

        var grid = [UInt32](repeating: 0, count: total)
        var used = [Bool](repeating: false, count: total)

        for v in voxels {
            let lx = v.x - chunkOrigin.x
            let ly = v.y - chunkOrigin.y
            let lz = v.z - chunkOrigin.z
            let ix = Int(lx * invLeafSize)
            let iy = Int(ly * invLeafSize)
            let iz = Int(lz * invLeafSize)
            guard ix >= 0, iy >= 0, iz >= 0, ix < gridN, iy < gridN, iz < gridN else { continue }
            let cf = v.colorAndFlags == 0 ? UInt32(1) : v.colorAndFlags
            grid[iz * gridN2 + iy * gridN + ix] = cf
        }

        var result = [MergedVoxel]()
        result.reserveCapacity(max(voxels.count / 8, 8))

        for iz in 0..<gridN {
            let zBase = iz * gridN2
            for iy in 0..<gridN {
                let yzBase = zBase + iy * gridN
                for ix in 0..<gridN {
                    let baseIdx = yzBase + ix
                    guard !used[baseIdx] else { continue }
                    let color = grid[baseIdx]
                    guard color != 0 else { continue }

                    var nx = 1
                    while ix + nx < gridN {
                        let i = yzBase + ix + nx
                        guard grid[i] == color, !used[i] else { break }
                        nx += 1
                    }

                    var ny = 1
                    outerY: while iy + ny < gridN {
                        let rowBase = zBase + (iy + ny) * gridN + ix
                        for dx in 0..<nx {
                            let i = rowBase + dx
                            guard grid[i] == color, !used[i] else { break outerY }
                        }
                        ny += 1
                    }

                    var nz = 1
                    outerZ: while iz + nz < gridN {
                        let slabBase = (iz + nz) * gridN2 + iy * gridN + ix
                        for dy in 0..<ny {
                            let rowBase = slabBase + dy * gridN
                            for dx in 0..<nx {
                                let i = rowBase + dx
                                guard grid[i] == color, !used[i] else { break outerZ }
                            }
                        }
                        nz += 1
                    }

                    for dz in 0..<nz {
                        let dzBase = (iz + dz) * gridN2
                        for dy in 0..<ny {
                            let dyBase = dzBase + (iy + dy) * gridN + ix
                            for dx in 0..<nx { used[dyBase + dx] = true }
                        }
                    }

                    let wx = chunkOrigin.x + (Float(ix) + Float(nx) * 0.5) * leafSize
                    let wy = chunkOrigin.y + (Float(iy) + Float(ny) * 0.5) * leafSize
                    let wz = chunkOrigin.z + (Float(iz) + Float(nz) * 0.5) * leafSize
                    result.append(MergedVoxel(
                        cx: wx, cy: wy, cz: wz,
                        hx: Float(nx) * halfLeaf,
                        hy: Float(ny) * halfLeaf,
                        hz: Float(nz) * halfLeaf,
                        colorAndFlags: color
                    ))
                }
            }
        }

        return result
    }
}
