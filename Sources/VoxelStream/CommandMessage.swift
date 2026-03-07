import Foundation

/// Command message sent from Mac to iPhone over the reverse TCP channel.
///
/// Wire format "VXCM" (8 bytes):
/// ```
/// [0..3]   magic 0x56 0x58 0x43 0x4D   4B
/// [4]      commandType UInt8            1B
/// [5]      payload UInt8                1B
/// [6..7]   reserved                     2B
/// ```
public struct CommandMessage: Sendable {
    public static let magic: [UInt8] = [0x56, 0x58, 0x43, 0x4D] // "VXCM"
    public static let messageSize = 8

    public enum CommandType: UInt8, Sendable {
        case switchCamera = 0
    }

    public let commandType: CommandType
    public let payload: UInt8

    public init(commandType: CommandType, payload: UInt8) {
        self.commandType = commandType
        self.payload = payload
    }
}

// MARK: - Encoder

public enum CommandMessageEncoder {

    public static func encode(_ msg: CommandMessage) -> Data {
        var data = Data(capacity: CommandMessage.messageSize)
        data.append(contentsOf: CommandMessage.magic)
        data.append(msg.commandType.rawValue)
        data.append(msg.payload)
        data.append(contentsOf: [0, 0]) // reserved
        return data
    }

    public static func encodeLengthPrefixed(_ msg: CommandMessage) -> Data {
        let payload = encode(msg)
        var framed = Data(capacity: 4 + payload.count)
        withUnsafeBytes(of: UInt32(payload.count)) { framed.append(contentsOf: $0) }
        framed.append(payload)
        return framed
    }
}

// MARK: - Decoder

public enum CommandMessageDecoder {

    public static func decode(_ data: Data) -> CommandMessage? {
        guard data.count >= CommandMessage.messageSize else { return nil }
        let s = data.startIndex
        guard data[s] == 0x56, data[s+1] == 0x58,
              data[s+2] == 0x43, data[s+3] == 0x4D else { return nil }
        guard let cmdType = CommandMessage.CommandType(rawValue: data[s + 4]) else { return nil }
        return CommandMessage(commandType: cmdType, payload: data[s + 5])
    }

    public static func decodeLengthPrefixed(_ data: Data) -> (CommandMessage, Int)? {
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
}
