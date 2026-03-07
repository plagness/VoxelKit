import Foundation
import Compression
import simd

/// Encodes `VoxelStreamFrame` into a binary `Data` blob for network transmission.
///
/// Wire format (little-endian):
/// ```
/// Header (146 bytes):
///   [0..3]     magic "VXSF"          (4B)
///   [4..7]     sequence UInt32       (4B)
///   [8..15]    timestamp Float64     (8B)
///   [16..79]   pose 16×Float32       (64B)
///   [80..115]  intrinsics 9×Float32  (36B)
///   [116..117] imageWidth UInt16     (2B)
///   [118..119] imageHeight UInt16    (2B)
///   [120..123] jpegSize UInt32       (4B)
///   [124..125] depthWidth UInt16     (2B)
///   [126..127] depthHeight UInt16    (2B)
///   [128..131] depthSize UInt32      (4B)
///   [132..135] worldPointCount UInt32(4B)
///   [136..139] worldPointsSize UInt32(4B) = count × 12
///   [140]      detectionCount UInt8  (1B)
///   [141..144] detectionsSize UInt32 (4B) = count × 29
///   [145]      reserved UInt8        (1B)
///
/// Payload:
///   [146 ..< 146+jpegSize]           JPEG data
///   [+jpegSize ..< +depthSize]       Float16 depth (if depthSize > 0)
///   [+depthSize ..< +worldPtsSize]   Float32×3 world points (if worldPointsSize > 0)
///   [+worldPtsSize ..< +detSize]     Detection data (if detectionsSize > 0)
/// ```
public enum VoxelStreamEncoder {

    public static let headerSize = 146
    public static let magic: [UInt8] = [0x56, 0x58, 0x53, 0x46] // "VXSF"

    public static func encode(_ frame: VoxelStreamFrame) -> Data {
        let jpegSize = UInt32(frame.imageJPEG.count)
        var depthData = frame.depthFloat16 ?? Data()
        var frameFlags = frame.flags
        // LZ4 compress depth if present and large enough to benefit
        if depthData.count > 64 {
            if let compressed = lz4Compress(depthData), compressed.count < depthData.count {
                depthData = compressed
                frameFlags |= 0x02 // bit 1 = depth is LZ4 compressed
            }
        }
        let depthSize = UInt32(depthData.count)
        let worldPtsData = frame.worldPoints ?? Data()
        let worldPtsSize = UInt32(worldPtsData.count)
        let detData = frame.detections ?? Data()
        let detSize = UInt32(detData.count)

        var data = Data(capacity: headerSize + Int(jpegSize) + Int(depthSize) + Int(worldPtsSize) + Int(detSize))

        // Magic
        data.append(contentsOf: magic)

        // Sequence
        appendLE(&data, frame.sequence)

        // Timestamp
        appendLE(&data, frame.timestamp)

        // Pose (column-major 4x4 = 16 floats)
        let p = frame.pose
        for col in [p.columns.0, p.columns.1, p.columns.2, p.columns.3] {
            appendLE(&data, col.x)
            appendLE(&data, col.y)
            appendLE(&data, col.z)
            appendLE(&data, col.w)
        }

        // Intrinsics (column-major 3x3 = 9 floats)
        let m = frame.intrinsics
        for col in [m.columns.0, m.columns.1, m.columns.2] {
            appendLE(&data, col.x)
            appendLE(&data, col.y)
            appendLE(&data, col.z)
        }

        // Image dimensions + JPEG size
        appendLE(&data, frame.imageWidth)
        appendLE(&data, frame.imageHeight)
        appendLE(&data, jpegSize)

        // Depth dimensions + depth size
        appendLE(&data, frame.depthWidth)
        appendLE(&data, frame.depthHeight)
        appendLE(&data, depthSize)

        // World points header
        appendLE(&data, frame.worldPointCount)
        appendLE(&data, worldPtsSize)

        // Detections header
        data.append(frame.detectionCount)
        appendLE(&data, detSize)
        data.append(frameFlags)

        // Payload
        data.append(frame.imageJPEG)
        if !depthData.isEmpty {
            data.append(depthData)
        }
        if !worldPtsData.isEmpty {
            data.append(worldPtsData)
        }
        if !detData.isEmpty {
            data.append(detData)
        }

        return data
    }

    /// Encode with a 4-byte length prefix (for framing over TCP).
    public static func encodeLengthPrefixed(_ frame: VoxelStreamFrame) -> Data {
        let payload = encode(frame)
        var framed = Data(capacity: 4 + payload.count)
        appendLE(&framed, UInt32(payload.count))
        framed.append(payload)
        return framed
    }

    /// Encode a minimal stop-marker frame (length-prefixed) to signal end of recording.
    public static func encodeStopMarker() -> Data {
        let frame = VoxelStreamFrame(
            sequence: VoxelStreamFrame.stopMarkerSequence,
            timestamp: 0,
            pose: .init(1), // identity
            intrinsics: .init(1),
            imageJPEG: Data([0xFF, 0xD8, 0xFF, 0xD9]), // minimal valid JPEG
            imageWidth: 1,
            imageHeight: 1
        )
        return encodeLengthPrefixed(frame)
    }

    // MARK: - DeviceStatusMessage

    public static func encodeStatus(_ status: DeviceStatusMessage) -> Data {
        var data = Data(capacity: DeviceStatusMessage.messageSize)
        data.append(contentsOf: DeviceStatusMessage.magic)
        data.append(status.cameraActive ? 1 : 0)
        data.append(status.gyroActive ? 1 : 0)
        data.append(status.selectedCamera)
        data.append(status.batteryLevel)
        data.append(status.thermalState)
        data.append(status.trackingState)
        // Reserved 4 bytes
        data.append(contentsOf: [0, 0, 0, 0])
        return data
    }

    public static func encodeStatusLengthPrefixed(_ status: DeviceStatusMessage) -> Data {
        let payload = encodeStatus(status)
        var framed = Data(capacity: 4 + payload.count)
        appendLE(&framed, UInt32(payload.count))
        framed.append(payload)
        return framed
    }

    // MARK: - Helpers

    @inline(__always)
    private static func appendLE<T>(_ data: inout Data, _ value: T) {
        withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }

    /// LZ4 compress data. Returns nil on failure.
    static func lz4Compress(_ input: Data) -> Data? {
        let capacity = input.count // LZ4 output ≤ input for compressible data
        var output = Data(count: capacity)
        let compressedSize = input.withUnsafeBytes { srcBuf in
            output.withUnsafeMutableBytes { dstBuf in
                compression_encode_buffer(
                    dstBuf.baseAddress!.assumingMemoryBound(to: UInt8.self), capacity,
                    srcBuf.baseAddress!.assumingMemoryBound(to: UInt8.self), input.count,
                    nil, COMPRESSION_LZ4
                )
            }
        }
        guard compressedSize > 0 else { return nil }
        output.count = compressedSize
        return output
    }

    /// LZ4 decompress data to expected size. Returns nil on failure.
    static func lz4Decompress(_ input: Data, decompressedSize: Int) -> Data? {
        var output = Data(count: decompressedSize)
        let actualSize = input.withUnsafeBytes { srcBuf in
            output.withUnsafeMutableBytes { dstBuf in
                compression_decode_buffer(
                    dstBuf.baseAddress!.assumingMemoryBound(to: UInt8.self), decompressedSize,
                    srcBuf.baseAddress!.assumingMemoryBound(to: UInt8.self), input.count,
                    nil, COMPRESSION_LZ4
                )
            }
        }
        guard actualSize == decompressedSize else { return nil }
        return output
    }
}
