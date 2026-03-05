import Foundation
import simd
import Compression

/// Serializer for the `.botmap` file format.
///
/// Format:
/// - 8 bytes: magic `BOTMAP01`
/// - 4 bytes: JSON metadata length (UInt32 little-endian)
/// - N bytes: JSON metadata (UTF-8)
/// - 4 bytes: voxel count (UInt32 little-endian)
/// - M bytes: LZ4-compressed voxel positions (each: 3 × Float32 = 12 bytes)
public enum BotMapSerializer {

    private static let magic = "BOTMAP01"

    // MARK: - Save

    public static func save(world: BotMapWorld, to url: URL) async throws {
        // Collect data on actor
        let positions = await world.store.allOccupiedPositions()
        let name = await world.name
        let createdAt = await world.createdAt
        let count = positions.count

        // Build JSON header
        let meta = BotMapMeta(
            name: name,
            voxelCount: count,
            createdAt: ISO8601DateFormatter().string(from: createdAt),
            version: 1
        )
        let jsonData = try JSONEncoder().encode(meta)

        // Serialize voxel positions: [x, y, z, x, y, z, ...]
        var rawVoxels = Data(capacity: count * 12)
        for pos in positions {
            withUnsafeBytes(of: pos.x) { rawVoxels.append(contentsOf: $0) }
            withUnsafeBytes(of: pos.y) { rawVoxels.append(contentsOf: $0) }
            withUnsafeBytes(of: pos.z) { rawVoxels.append(contentsOf: $0) }
        }

        // LZ4 compress voxel data
        let compressedVoxels = try lz4Compress(rawVoxels)

        // Assemble file
        var output = Data()
        output.append(contentsOf: magic.utf8)                      // 8 bytes magic
        var jsonLen = UInt32(jsonData.count).littleEndian
        withUnsafeBytes(of: &jsonLen) { output.append(contentsOf: $0) }  // 4 bytes JSON len
        output.append(jsonData)                                    // JSON metadata
        var voxCount = UInt32(count).littleEndian
        withUnsafeBytes(of: &voxCount) { output.append(contentsOf: $0) } // 4 bytes voxel count
        var rawLen = UInt32(rawVoxels.count).littleEndian
        withUnsafeBytes(of: &rawLen) { output.append(contentsOf: $0) }   // 4 bytes raw size (for decompress)
        output.append(compressedVoxels)                            // LZ4 data

        try output.write(to: url)
    }

    // MARK: - Load

    public static func load(from url: URL) async throws -> BotMapWorld {
        let data = try Data(contentsOf: url)
        var offset = data.startIndex

        // Magic
        let magicBytes = data[offset..<offset + 8]
        guard String(bytes: magicBytes, encoding: .utf8) == magic else {
            throw BotMapError.invalidMagic
        }
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

        // Parse positions
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

        // Build world
        let store = ChunkedOctreeStore()
        store.insertPositions(positions)

        let createdAt = ISO8601DateFormatter().date(from: meta.createdAt) ?? .now
        return BotMapWorld(store: store, name: meta.name, createdAt: createdAt)
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
