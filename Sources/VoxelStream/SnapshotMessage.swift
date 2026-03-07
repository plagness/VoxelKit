import Foundation

/// JPEG snapshot sent from Mac to iPhone over the reverse TCP channel.
///
/// Wire format "VXSS" (12 bytes header + JPEG payload):
/// ```
/// [0..3]   magic 0x56 0x58 0x53 0x53   4B
/// [4..5]   width UInt16                 2B
/// [6..7]   height UInt16                2B
/// [8..11]  jpegSize UInt32              4B
/// [12..]   JPEG data                    variable
/// ```
public struct SnapshotMessage: Sendable {
    public static let magic: [UInt8] = [0x56, 0x58, 0x53, 0x53] // "VXSS"
    public static let headerSize = 12

    public let width: UInt16
    public let height: UInt16
    public let jpegData: Data

    public init(width: UInt16, height: UInt16, jpegData: Data) {
        self.width = width
        self.height = height
        self.jpegData = jpegData
    }
}

// MARK: - Encoder

public enum SnapshotMessageEncoder {

    public static func encode(_ msg: SnapshotMessage) -> Data {
        var data = Data(capacity: SnapshotMessage.headerSize + msg.jpegData.count)
        data.append(contentsOf: SnapshotMessage.magic)
        appendLE(&data, msg.width)
        appendLE(&data, msg.height)
        appendLE(&data, UInt32(msg.jpegData.count))
        data.append(msg.jpegData)
        return data
    }

    public static func encodeLengthPrefixed(_ msg: SnapshotMessage) -> Data {
        let payload = encode(msg)
        var framed = Data(capacity: 4 + payload.count)
        appendLE(&framed, UInt32(payload.count))
        framed.append(payload)
        return framed
    }

    @inline(__always)
    private static func appendLE<T>(_ data: inout Data, _ value: T) {
        withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }
}

// MARK: - Decoder

public enum SnapshotMessageDecoder {

    public static func decode(_ data: Data) -> SnapshotMessage? {
        guard data.count >= SnapshotMessage.headerSize else { return nil }
        let s = data.startIndex
        guard data[s] == 0x56, data[s+1] == 0x58,
              data[s+2] == 0x53, data[s+3] == 0x53 else { return nil }

        var offset = s + 4
        let width: UInt16 = readLE(data, &offset)
        let height: UInt16 = readLE(data, &offset)
        let jpegSize: UInt32 = readLE(data, &offset)

        let expectedSize = SnapshotMessage.headerSize + Int(jpegSize)
        guard data.count >= expectedSize else { return nil }

        let jpegData = data[s + SnapshotMessage.headerSize ..< s + expectedSize]
        return SnapshotMessage(width: width, height: height, jpegData: Data(jpegData))
    }

    public static func decodeLengthPrefixed(_ data: Data) -> (SnapshotMessage, Int)? {
        guard data.count >= 4 else { return nil }
        let payloadLength = data[data.startIndex..<data.startIndex + 4].withUnsafeBytes {
            $0.loadUnaligned(as: UInt32.self)
        }
        let totalLength = 4 + Int(payloadLength)
        guard data.count >= totalLength else { return nil }
        let payload = data[data.startIndex + 4 ..< data.startIndex + totalLength]
        guard let msg = decode(Data(payload)) else { return nil }
        return (msg, totalLength)
    }

    @inline(__always)
    private static func readLE<T>(_ data: Data, _ offset: inout Data.Index) -> T {
        let size = MemoryLayout<T>.size
        let value = data[offset..<offset + size].withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        offset += size
        return value
    }
}
