import Foundation
import Compression

/// Manages streaming chunks to/from disk for large-scale worlds (10+ km²).
///
/// Uses an LRU cache to keep a limited number of chunks in memory.
/// Evicted chunks are serialized to `.vxchunk` files on disk.
/// Thread-safety: all access must go through the owning actor (BotMapWorld).
public final class ChunkStreamManager: @unchecked Sendable {

    /// Maximum chunks held in memory before LRU eviction.
    public var maxInMemoryChunks: Int

    /// Base directory for chunk storage.
    public let storageURL: URL

    /// LRU access order (most recently accessed at end).
    private var accessOrder: [ChunkKey] = []

    /// Fast membership check for accessOrder (avoids O(n) firstIndex calls).
    private var accessSet: Set<ChunkKey> = []

    /// Set of chunk keys known to be on disk.
    private var onDiskKeys: Set<ChunkKey> = []

    /// Statistics.
    public private(set) var evictionCount: Int = 0
    public private(set) var loadCount: Int = 0

    private static let chunkMagic: [UInt8] = [0x56, 0x58, 0x43, 0x31] // "VXC1"

    public init(storageURL: URL, maxInMemoryChunks: Int = 10_000) {
        self.storageURL = storageURL
        self.maxInMemoryChunks = maxInMemoryChunks
        try? FileManager.default.createDirectory(at: storageURL, withIntermediateDirectories: true)
    }

    // MARK: - Access tracking

    /// Mark a chunk as recently accessed (move to end of LRU).
    public func touch(_ key: ChunkKey) {
        if accessSet.contains(key) {
            if let idx = accessOrder.firstIndex(of: key) {
                accessOrder.remove(at: idx)
            }
        } else {
            accessSet.insert(key)
        }
        accessOrder.append(key)
    }

    // MARK: - Load

    /// Load a chunk from disk if it exists, or return nil.
    public func loadFromDisk(_ key: ChunkKey) -> OctreeChunk? {
        let url = chunkFileURL(for: key)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }

        guard let data = try? Data(contentsOf: url) else { return nil }
        guard data.count > 10 else { return nil }

        // Verify magic
        guard data[0] == 0x56, data[1] == 0x58, data[2] == 0x43, data[3] == 0x31 else { return nil }

        // Read ChunkKey (6 bytes) — use loadUnaligned for safety
        let x = Int16(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: Int16.self) })
        let y = Int16(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 6, as: Int16.self) })
        let z = Int16(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: Int16.self) })

        guard x == key.x, y == key.y, z == key.z else { return nil }

        // Read version
        let version = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 10, as: UInt32.self) }

        // Read raw size (for decompression)
        let rawSize = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 14, as: UInt32.self) })

        // Decompress octree data
        let compressedData = data[18...]
        guard let rawData = try? lz4Decompress(compressedData, originalSize: rawSize) else { return nil }

        // Rebuild octree from serialized leaf positions + colors
        let chunk = OctreeChunk(key: key)
        chunk.version = UInt32(littleEndian: version)
        deserializeLeaves(rawData, into: chunk.tree)

        loadCount += 1
        onDiskKeys.insert(key)
        return chunk
    }

    // MARK: - Eviction

    /// Evict least-recently-used chunks from the provided store to disk.
    /// Returns the set of evicted keys.
    @discardableResult
    public func evictIfNeeded(store: ChunkedOctreeStore) -> Set<ChunkKey> {
        var evicted = Set<ChunkKey>()
        let allChunks = store.allChunks

        while accessOrder.count > maxInMemoryChunks, let oldest = accessOrder.first {
            accessOrder.removeFirst()
            accessSet.remove(oldest)
            if let chunk = allChunks[oldest] {
                saveToDisk(chunk)
                store.removeChunk(oldest)
                evicted.insert(oldest)
                evictionCount += 1
            }
        }
        return evicted
    }

    /// Save a single chunk to disk.
    public func saveToDisk(_ chunk: OctreeChunk) {
        let url = chunkFileURL(for: chunk.key)
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var data = Data()

        // Magic "VXC1"
        data.append(contentsOf: Self.chunkMagic)

        // ChunkKey (6 bytes)
        var x = chunk.key.x.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        var y = chunk.key.y.littleEndian; withUnsafeBytes(of: &y) { data.append(contentsOf: $0) }
        var z = chunk.key.z.littleEndian; withUnsafeBytes(of: &z) { data.append(contentsOf: $0) }

        // Version (4 bytes)
        var ver = chunk.version.littleEndian; withUnsafeBytes(of: &ver) { data.append(contentsOf: $0) }

        // Serialize leaves
        let rawData = serializeLeaves(chunk.tree)

        // Raw size (4 bytes)
        var rawLen = UInt32(rawData.count).littleEndian
        withUnsafeBytes(of: &rawLen) { data.append(contentsOf: $0) }

        // LZ4 compress
        if let compressed = try? lz4Compress(rawData) {
            data.append(compressed)
        } else {
            data.append(rawData) // fallback: uncompressed
        }

        try? data.write(to: url)
        onDiskKeys.insert(chunk.key)
    }

    /// Flush all chunks in the store to disk (for shutdown).
    public func flushAll(store: ChunkedOctreeStore) {
        for (_, chunk) in store.allChunks {
            saveToDisk(chunk)
        }
    }

    // MARK: - File paths

    private func chunkFileURL(for key: ChunkKey) -> URL {
        storageURL
            .appendingPathComponent("\(key.x)")
            .appendingPathComponent("\(key.y)")
            .appendingPathComponent("\(key.z).vxchunk")
    }

    // MARK: - Leaf serialization

    // Format: each leaf = 16 bytes: position(12B) + colorAndFlags(4B)
    private func serializeLeaves(_ tree: Octree) -> Data {
        let voxels = tree.collectOccupiedVoxels()
        var data = Data(capacity: voxels.count * 16)
        for v in voxels {
            withUnsafeBytes(of: v.x) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: v.y) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: v.z) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: v.colorAndFlags) { data.append(contentsOf: $0) }
        }
        return data
    }

    private func deserializeLeaves(_ data: Data, into tree: Octree) {
        let count = data.count / 16
        data.withUnsafeBytes { ptr in
            for i in 0..<count {
                let offset = i * 16
                let x = ptr.loadUnaligned(fromByteOffset: offset, as: Float.self)
                let y = ptr.loadUnaligned(fromByteOffset: offset + 4, as: Float.self)
                let z = ptr.loadUnaligned(fromByteOffset: offset + 8, as: Float.self)
                let cf = ptr.loadUnaligned(fromByteOffset: offset + 12, as: UInt32.self)
                let r = UInt8((cf >> 24) & 0xFF)
                let g = UInt8((cf >> 16) & 0xFF)
                let b = UInt8((cf >> 8) & 0xFF)
                let pos = SIMD3<Float>(x, y, z)
                tree.updateOccupancy(at: pos, hit: true, color: (r, g, b))
            }
        }
    }

    // MARK: - LZ4

    private func lz4Compress(_ data: Data) throws -> Data {
        guard !data.isEmpty else { return Data() }
        let srcSize = data.count
        let dstCapacity = srcSize + (srcSize / 255) + 16
        var dst = Data(count: dstCapacity)
        let compressedSize = data.withUnsafeBytes { src in
            dst.withUnsafeMutableBytes { dstPtr in
                compression_encode_buffer(
                    dstPtr.bindMemory(to: UInt8.self).baseAddress!, dstCapacity,
                    src.bindMemory(to: UInt8.self).baseAddress!, srcSize,
                    nil, COMPRESSION_LZ4
                )
            }
        }
        guard compressedSize > 0 else { throw ChunkStreamError.compressionFailed }
        dst.removeLast(dstCapacity - compressedSize)
        return dst
    }

    private func lz4Decompress(_ data: Data, originalSize: Int) throws -> Data {
        guard !data.isEmpty else { return Data() }
        var dst = Data(count: originalSize)
        let decompressedSize = data.withUnsafeBytes { src in
            dst.withUnsafeMutableBytes { dstPtr in
                compression_decode_buffer(
                    dstPtr.bindMemory(to: UInt8.self).baseAddress!, originalSize,
                    src.bindMemory(to: UInt8.self).baseAddress!, data.count,
                    nil, COMPRESSION_LZ4
                )
            }
        }
        guard decompressedSize == originalSize else { throw ChunkStreamError.decompressionFailed }
        return dst
    }
}

public enum ChunkStreamError: Error {
    case compressionFailed
    case decompressionFailed
    case invalidChunkFile
}
