import Foundation
import simd

/// Encodes `VoxelStreamFrame` into a binary `Data` blob for network transmission.
///
/// Wire format (little-endian):
/// ```
/// Header (140 bytes):
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
///
/// Payload:
///   [140 ..< 140+jpegSize]           JPEG data
///   [+jpegSize ..< +depthSize]       Float16 depth (if depthSize > 0)
///   [+depthSize ..< +worldPtsSize]   Float32×3 world points (if worldPointsSize > 0)
/// ```
public enum VoxelStreamEncoder {

    public static let headerSize = 140
    public static let magic: [UInt8] = [0x56, 0x58, 0x53, 0x46] // "VXSF"

    public static func encode(_ frame: VoxelStreamFrame) -> Data {
        let jpegSize = UInt32(frame.imageJPEG.count)
        let depthData = frame.depthFloat16 ?? Data()
        let depthSize = UInt32(depthData.count)
        let worldPtsData = frame.worldPoints ?? Data()
        let worldPtsSize = UInt32(worldPtsData.count)

        var data = Data(capacity: headerSize + Int(jpegSize) + Int(depthSize) + Int(worldPtsSize))

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

        // Payload
        data.append(frame.imageJPEG)
        if !depthData.isEmpty {
            data.append(depthData)
        }
        if !worldPtsData.isEmpty {
            data.append(worldPtsData)
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

    // MARK: - Helpers

    @inline(__always)
    private static func appendLE<T>(_ data: inout Data, _ value: T) {
        withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }
}
