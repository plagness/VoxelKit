import Foundation

/// A single voxel in a preview message sent from Mac back to iPhone.
public struct PreviewVoxel: Sendable {
    public var x: Float
    public var y: Float
    public var z: Float
    public var colorAndFlags: UInt32

    public init(x: Float, y: Float, z: Float, colorAndFlags: UInt32) {
        self.x = x; self.y = y; self.z = z
        self.colorAndFlags = colorAndFlags
    }
}

/// Preview message sent from Mac to iPhone over the reverse channel.
public struct PreviewMessage: Sendable {
    public static let magic: [UInt8] = [0x56, 0x58, 0x50, 0x56] // "VXPV"
    public static let headerSize = 8

    public let voxels: [PreviewVoxel]

    public init(voxels: [PreviewVoxel]) {
        self.voxels = voxels
    }
}

// MARK: - Encoder

public enum PreviewMessageEncoder {

    /// Encode a preview message into binary data.
    public static func encode(_ msg: PreviewMessage) -> Data {
        let voxelBytes = msg.voxels.count * 16
        var data = Data(capacity: PreviewMessage.headerSize + voxelBytes)

        // Magic "VXPV"
        data.append(contentsOf: PreviewMessage.magic)

        // Voxel count
        appendLE(&data, UInt32(msg.voxels.count))

        // Voxel payload (16 bytes each: x, y, z, colorAndFlags)
        for v in msg.voxels {
            appendLE(&data, v.x)
            appendLE(&data, v.y)
            appendLE(&data, v.z)
            appendLE(&data, v.colorAndFlags)
        }

        return data
    }

    /// Encode with 4-byte length prefix for TCP framing.
    public static func encodeLengthPrefixed(_ msg: PreviewMessage) -> Data {
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

public enum PreviewMessageDecoder {

    /// Decode from length-prefixed TCP data. Returns (message, bytesConsumed) or nil if incomplete.
    public static func decodeLengthPrefixed(_ data: Data) -> (PreviewMessage, Int)? {
        guard data.count >= 4 else { return nil }
        var offset = data.startIndex
        let payloadLength: UInt32 = readLE(data, &offset)
        let totalLength = 4 + Int(payloadLength)
        guard data.count >= totalLength else { return nil }

        let payload = data[data.startIndex + 4 ..< data.startIndex + totalLength]
        guard let msg = decode(Data(payload)) else { return nil }
        return (msg, totalLength)
    }

    /// Decode a preview message from raw binary data (no length prefix).
    public static func decode(_ data: Data) -> PreviewMessage? {
        guard data.count >= PreviewMessage.headerSize else { return nil }

        // Verify magic
        guard data[data.startIndex] == 0x56,
              data[data.startIndex + 1] == 0x58,
              data[data.startIndex + 2] == 0x50,
              data[data.startIndex + 3] == 0x56 else { return nil }

        var offset = data.startIndex + 4
        let voxelCount: UInt32 = readLE(data, &offset)

        let expectedSize = PreviewMessage.headerSize + Int(voxelCount) * 16
        guard data.count >= expectedSize else { return nil }

        var voxels = [PreviewVoxel]()
        voxels.reserveCapacity(Int(voxelCount))

        for _ in 0..<voxelCount {
            let x: Float = readLE(data, &offset)
            let y: Float = readLE(data, &offset)
            let z: Float = readLE(data, &offset)
            let c: UInt32 = readLE(data, &offset)
            voxels.append(PreviewVoxel(x: x, y: y, z: z, colorAndFlags: c))
        }

        return PreviewMessage(voxels: voxels)
    }

    @inline(__always)
    private static func readLE<T>(_ data: Data, _ offset: inout Data.Index) -> T {
        let size = MemoryLayout<T>.size
        let value = data[offset..<offset + size].withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        offset += size
        return value
    }
}
