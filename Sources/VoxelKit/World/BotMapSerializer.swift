import Foundation
import simd
import Compression

/// Serializer for the `.botmap` file format.
///
/// BOTMAP02 format (v2):
/// - 8 bytes: magic `BOTMAP02`
/// - 4 bytes: JSON metadata length (UInt32 little-endian)
/// - N bytes: JSON metadata (UTF-8) — includes pipelineMode
/// - 4 bytes: voxel count (UInt32 little-endian)
/// - 4 bytes: raw data size (for LZ4 decompression)
/// - M bytes: LZ4-compressed voxel data
///   Each voxel: position(12B) + colorAndFlags(4B) + observationCount(2B) + signedDistance(4B) = 22 bytes
///
/// Backward compatible: reads BOTMAP01 (12B/voxel, positions only).
public enum BotMapSerializer {

    private static let magicV1 = "BOTMAP01"
    private static let magicV2 = "BOTMAP02"

    // MARK: - Save

    public static func save(world: BotMapWorld, to url: URL) async throws {
        // nonisolated let properties — no await needed
        let store = world.store
        let createdAt = world.createdAt
        // Actor-isolated var properties
        let name = await world.name
        let pipelineMode = await world.pipelineMode

        // Single-pass collection: packed voxels + observation data in lockstep
        var packedVoxels = [PackedVoxel]()
        var obsData = [(UInt16, Float)]()
        for (_, chunk) in store.allChunks {
            collectVoxelsWithNodeData(node: chunk.tree.root,
                                       origin: chunk.tree.origin,
                                       size: chunk.tree.rootSize,
                                       packedVoxels: &packedVoxels,
                                       obsData: &obsData)
        }
        let count = packedVoxels.count

        // Build JSON header
        let meta = BotMapMeta(
            name: name,
            voxelCount: count,
            createdAt: ISO8601DateFormatter().string(from: createdAt),
            version: 2,
            pipelineMode: pipelineMode.rawValue
        )
        let jsonData = try JSONEncoder().encode(meta)

        // Serialize voxels: position(12B) + colorAndFlags(4B) + observationCount(2B) + signedDistance(4B) = 22B each
        var rawVoxels = Data(capacity: count * 22)
        for (i, pv) in packedVoxels.enumerated() {
            withUnsafeBytes(of: pv.x) { rawVoxels.append(contentsOf: $0) }
            withUnsafeBytes(of: pv.y) { rawVoxels.append(contentsOf: $0) }
            withUnsafeBytes(of: pv.z) { rawVoxels.append(contentsOf: $0) }
            withUnsafeBytes(of: pv.colorAndFlags) { rawVoxels.append(contentsOf: $0) }
            let obsCount = obsData[i].0
            let signedDist = obsData[i].1
            withUnsafeBytes(of: obsCount) { rawVoxels.append(contentsOf: $0) }
            withUnsafeBytes(of: signedDist) { rawVoxels.append(contentsOf: $0) }
        }

        // LZ4 compress voxel data
        let compressedVoxels = try lz4Compress(rawVoxels)

        // Assemble file
        var output = Data()
        output.append(contentsOf: magicV2.utf8)                    // 8 bytes magic
        var jsonLen = UInt32(jsonData.count).littleEndian
        withUnsafeBytes(of: &jsonLen) { output.append(contentsOf: $0) }  // 4 bytes JSON len
        output.append(jsonData)                                    // JSON metadata
        var voxCount = UInt32(count).littleEndian
        withUnsafeBytes(of: &voxCount) { output.append(contentsOf: $0) } // 4 bytes voxel count
        var rawLen = UInt32(rawVoxels.count).littleEndian
        withUnsafeBytes(of: &rawLen) { output.append(contentsOf: $0) }   // 4 bytes raw size
        output.append(compressedVoxels)                            // LZ4 data

        try output.write(to: url)
    }

    /// Single-pass recursive collection of PackedVoxels + observation data in lockstep.
    /// Guarantees 1:1 correspondence between packed voxels and observation data.
    private static func collectVoxelsWithNodeData(
        node: OctreeNode, origin: SIMD3<Float>, size: Float,
        packedVoxels: inout [PackedVoxel], obsData: inout [(UInt16, Float)]
    ) {
        if node.isLeaf {
            if node.logOdds >= 0.0 {
                let center = origin + SIMD3<Float>(repeating: size * 0.5)
                packedVoxels.append(PackedVoxel(position: center, color: node.color, layer: node.layer))
                obsData.append((node.observationCount, node.signedDistance))
            }
            return
        }
        guard let children = node.children else { return }
        let halfSize = size * 0.5
        for i in 0..<8 {
            guard let child = children[i] else { continue }
            let childOrigin = origin + SIMD3<Float>(
                (i & 1) != 0 ? halfSize : 0,
                (i & 2) != 0 ? halfSize : 0,
                (i & 4) != 0 ? halfSize : 0
            )
            collectVoxelsWithNodeData(node: child, origin: childOrigin, size: halfSize,
                                       packedVoxels: &packedVoxels, obsData: &obsData)
        }
    }

    // MARK: - Load

    public static func load(from url: URL) async throws -> BotMapWorld {
        let data = try Data(contentsOf: url)
        var offset = data.startIndex

        // Magic (8 bytes)
        let magicBytes = data[offset..<offset + 8]
        let magicStr = String(bytes: magicBytes, encoding: .utf8)
        guard magicStr == magicV1 || magicStr == magicV2 else {
            throw BotMapError.invalidMagic
        }
        let isV2 = magicStr == magicV2
        offset += 8

        // JSON length
        let jsonLen = Int(data.readUInt32(at: offset))
        offset += 4

        // JSON metadata
        let jsonData = data[offset..<offset + jsonLen]
        let meta = try JSONDecoder().decode(BotMapMeta.self, from: jsonData)
        offset += jsonLen

        // Voxel count
        let voxelCount = Int(data.readUInt32(at: offset))
        offset += 4

        // Raw size (for LZ4 decompression)
        let rawSize = Int(data.readUInt32(at: offset))
        offset += 4

        // LZ4 decompress
        let compressedData = data[offset...]
        let rawVoxels = try lz4Decompress(compressedData, originalSize: rawSize)

        let store = ChunkedOctreeStore()

        if isV2 {
            // V2: 22 bytes per voxel (position + color + observationCount + signedDistance)
            let bytesPerVoxel = 22
            guard rawVoxels.count == voxelCount * bytesPerVoxel else {
                throw BotMapError.dataSizeMismatch(expected: voxelCount * bytesPerVoxel, got: rawVoxels.count)
            }

            var coloredPositions = [ColoredPosition]()
            coloredPositions.reserveCapacity(voxelCount)

            rawVoxels.withUnsafeBytes { ptr in
                for i in 0..<voxelCount {
                    let base = i * bytesPerVoxel
                    let x = ptr.loadUnaligned(fromByteOffset: base, as: Float.self)
                    let y = ptr.loadUnaligned(fromByteOffset: base + 4, as: Float.self)
                    let z = ptr.loadUnaligned(fromByteOffset: base + 8, as: Float.self)
                    let cf = ptr.loadUnaligned(fromByteOffset: base + 12, as: UInt32.self)
                    let r = UInt8((cf >> 24) & 0xFF)
                    let g = UInt8((cf >> 16) & 0xFF)
                    let b = UInt8((cf >> 8)  & 0xFF)
                    // observationCount at base+16 (2B) and signedDistance at base+18 (4B)
                    // are stored but restored as regular occupancy (log-odds handles state)
                    coloredPositions.append(ColoredPosition(
                        position: SIMD3<Float>(x, y, z),
                        color: (r, g, b)
                    ))
                }
            }
            store.insertColoredPositions(coloredPositions)
        } else {
            // V1: 12 bytes per voxel (position only)
            guard rawVoxels.count == voxelCount * 12 else {
                throw BotMapError.dataSizeMismatch(expected: voxelCount * 12, got: rawVoxels.count)
            }

            var positions = [SIMD3<Float>]()
            positions.reserveCapacity(voxelCount)
            rawVoxels.withUnsafeBytes { ptr in
                let floats = ptr.bindMemory(to: Float.self)
                for i in 0..<voxelCount {
                    positions.append(SIMD3<Float>(floats[i*3], floats[i*3+1], floats[i*3+2]))
                }
            }
            store.insertPositions(positions)
        }

        let createdAt = ISO8601DateFormatter().date(from: meta.createdAt) ?? .now
        let world = BotMapWorld(store: store, name: meta.name, createdAt: createdAt)

        // Restore pipeline mode from V2 metadata
        if let modeStr = meta.pipelineMode, let mode = PipelineMode(rawValue: modeStr) {
            await world.setPipelineMode(mode)
        }

        return world
    }

    // MARK: - LZ4 helpers

    private static func lz4Compress(_ data: Data) throws -> Data {
        guard !data.isEmpty else { return Data() }

        let srcSize = data.count
        let dstCapacity = srcSize + (srcSize / 255) + 16  // LZ4 worst case
        var dst = Data(count: dstCapacity)

        let compressedSize = data.withUnsafeBytes { src in
            dst.withUnsafeMutableBytes { dstPtr in
                compression_encode_buffer(
                    dstPtr.bindMemory(to: UInt8.self).baseAddress!,
                    dstCapacity,
                    src.bindMemory(to: UInt8.self).baseAddress!,
                    srcSize,
                    nil,
                    COMPRESSION_LZ4
                )
            }
        }

        guard compressedSize > 0 else { throw BotMapError.compressionFailed }
        dst.removeLast(dstCapacity - compressedSize)
        return dst
    }

    private static func lz4Decompress(_ data: Data, originalSize: Int) throws -> Data {
        guard !data.isEmpty else { return Data() }

        var dst = Data(count: originalSize)
        let decompressedSize = data.withUnsafeBytes { src in
            dst.withUnsafeMutableBytes { dstPtr in
                compression_decode_buffer(
                    dstPtr.bindMemory(to: UInt8.self).baseAddress!,
                    originalSize,
                    src.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    COMPRESSION_LZ4
                )
            }
        }

        guard decompressedSize == originalSize else { throw BotMapError.decompressionFailed }
        return dst
    }
}

// MARK: - Metadata

private struct BotMapMeta: Codable {
    let name: String
    let voxelCount: Int
    let createdAt: String
    let version: Int
    var pipelineMode: String?
}

// MARK: - Errors

public enum BotMapError: Error, LocalizedError {
    case invalidMagic
    case dataSizeMismatch(expected: Int, got: Int)
    case compressionFailed
    case decompressionFailed

    public var errorDescription: String? {
        switch self {
        case .invalidMagic:          return "Not a valid .botmap file (invalid magic bytes)"
        case .dataSizeMismatch(let e, let g): return "Data size mismatch: expected \(e), got \(g)"
        case .compressionFailed:     return "LZ4 compression failed"
        case .decompressionFailed:   return "LZ4 decompression failed"
        }
    }
}

// MARK: - Data helpers

private extension Data {
    func readUInt32(at offset: Index) -> UInt32 {
        var value: UInt32 = 0
        _ = Swift.withUnsafeMutableBytes(of: &value) { ptr in
            self.copyBytes(to: ptr, from: offset..<offset+4)
        }
        return UInt32(littleEndian: value)
    }
}
