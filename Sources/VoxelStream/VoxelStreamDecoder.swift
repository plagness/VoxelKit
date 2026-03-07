import Foundation
import Compression
import simd

/// Decodes binary `Data` back into `VoxelStreamFrame`.
public enum VoxelStreamDecoder {

    public static func decode(_ data: Data) -> VoxelStreamFrame? {
        guard data.count >= VoxelStreamEncoder.headerSize else { return nil }

        // Verify magic
        guard data[data.startIndex] == 0x56,
              data[data.startIndex + 1] == 0x58,
              data[data.startIndex + 2] == 0x53,
              data[data.startIndex + 3] == 0x46 else { return nil }

        var offset = data.startIndex + 4

        let sequence: UInt32 = readLE(data, &offset)
        let timestamp: Float64 = readLE(data, &offset)

        // Pose: 16 floats → simd_float4x4
        var poseFloats = [Float](repeating: 0, count: 16)
        for i in 0..<16 { poseFloats[i] = readLE(data, &offset) }
        let pose = simd_float4x4(
            SIMD4(poseFloats[0], poseFloats[1], poseFloats[2], poseFloats[3]),
            SIMD4(poseFloats[4], poseFloats[5], poseFloats[6], poseFloats[7]),
            SIMD4(poseFloats[8], poseFloats[9], poseFloats[10], poseFloats[11]),
            SIMD4(poseFloats[12], poseFloats[13], poseFloats[14], poseFloats[15])
        )

        // Intrinsics: 9 floats → simd_float3x3
        var intFloats = [Float](repeating: 0, count: 9)
        for i in 0..<9 { intFloats[i] = readLE(data, &offset) }
        let intrinsics = simd_float3x3(
            SIMD3(intFloats[0], intFloats[1], intFloats[2]),
            SIMD3(intFloats[3], intFloats[4], intFloats[5]),
            SIMD3(intFloats[6], intFloats[7], intFloats[8])
        )

        let imageWidth: UInt16 = readLE(data, &offset)
        let imageHeight: UInt16 = readLE(data, &offset)
        let jpegSize: UInt32 = readLE(data, &offset)
        let depthWidth: UInt16 = readLE(data, &offset)
        let depthHeight: UInt16 = readLE(data, &offset)
        let depthSize: UInt32 = readLE(data, &offset)

        // World points header
        let worldPointCount: UInt32 = readLE(data, &offset)
        let worldPtsSize: UInt32 = readLE(data, &offset)

        // Detections header (v2 — 6 extra bytes after world points header)
        // Backward-compatible: if data is only 140 bytes header, detections = 0
        var detectionCount: UInt8 = 0
        var detSize: UInt32 = 0
        var flags: UInt8 = 0
        if offset + 6 <= data.startIndex + VoxelStreamEncoder.headerSize {
            detectionCount = data[offset]
            offset += 1
            detSize = readLE(data, &offset)
            flags = data[offset]
            offset += 1
        }

        // Validate payload size
        let expectedSize = VoxelStreamEncoder.headerSize + Int(jpegSize) + Int(depthSize) + Int(worldPtsSize) + Int(detSize)
        guard data.count >= expectedSize else { return nil }

        let jpegStart = offset
        let jpegEnd = jpegStart + Int(jpegSize)
        let imageJPEG = data[jpegStart..<jpegEnd]

        var depthFloat16: Data? = nil
        if depthSize > 0 {
            let depthStart = jpegEnd
            let depthEnd = depthStart + Int(depthSize)
            let rawDepth = Data(data[depthStart..<depthEnd])
            if flags & 0x02 != 0 {
                // LZ4 compressed depth — decompress
                let expectedSize = Int(depthWidth) * Int(depthHeight) * 2 // Float16
                depthFloat16 = VoxelStreamEncoder.lz4Decompress(rawDepth, decompressedSize: expectedSize)
            } else {
                depthFloat16 = rawDepth
            }
        }

        var worldPoints: Data? = nil
        if worldPtsSize > 0 {
            let wpStart = jpegEnd + Int(depthSize)
            let wpEnd = wpStart + Int(worldPtsSize)
            worldPoints = data[wpStart..<wpEnd]
        }

        var detections: Data? = nil
        if detSize > 0 {
            let detStart = jpegEnd + Int(depthSize) + Int(worldPtsSize)
            let detEnd = detStart + Int(detSize)
            detections = data[detStart..<detEnd]
        }

        return VoxelStreamFrame(
            sequence: sequence,
            timestamp: timestamp,
            pose: pose,
            intrinsics: intrinsics,
            imageJPEG: Data(imageJPEG),
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            depthFloat16: depthFloat16.map { Data($0) },
            depthWidth: depthWidth,
            depthHeight: depthHeight,
            worldPoints: worldPoints.map { Data($0) },
            worldPointCount: worldPointCount,
            detections: detections.map { Data($0) },
            detectionCount: detectionCount,
            flags: flags
        )
    }

    /// Decode from length-prefixed TCP data. Returns (frame, bytesConsumed) or nil if incomplete.
    public static func decodeLengthPrefixed(_ data: Data) -> (VoxelStreamFrame, Int)? {
        guard data.count >= 4 else { return nil }
        var offset = data.startIndex
        let payloadLength: UInt32 = readLE(data, &offset)
        let totalLength = 4 + Int(payloadLength)
        guard data.count >= totalLength else { return nil }
        let payload = data[data.startIndex + 4 ..< data.startIndex + totalLength]
        guard let frame = decode(Data(payload)) else { return nil }
        return (frame, totalLength)
    }

    // MARK: - DeviceStatusMessage

    public static func decodeStatus(_ data: Data) -> DeviceStatusMessage? {
        guard data.count >= DeviceStatusMessage.messageSize else { return nil }
        let s = data.startIndex
        guard data[s] == 0x56, data[s+1] == 0x58,
              data[s+2] == 0x53, data[s+3] == 0x54 else { return nil }
        return DeviceStatusMessage(
            cameraActive: data[s + 4] != 0,
            gyroActive: data[s + 5] != 0,
            selectedCamera: data[s + 6],
            batteryLevel: data[s + 7],
            thermalState: data[s + 8],
            trackingState: data[s + 9]
        )
    }

    // MARK: - Unified Message Dispatch

    /// Multiplexed message type for the VoxelStream TCP channel.
    public enum VoxelMessage {
        case frame(VoxelStreamFrame)
        case status(DeviceStatusMessage)
    }

    /// Decode any length-prefixed message by inspecting the magic bytes.
    /// Returns (message, bytesConsumed) or nil if data is incomplete.
    public static func decodeAnyLengthPrefixed(_ data: Data) -> (VoxelMessage, Int)? {
        guard data.count >= 8 else { return nil } // 4 length + 4 magic minimum
        var offset = data.startIndex
        let payloadLength: UInt32 = readLE(data, &offset)
        let totalLength = 4 + Int(payloadLength)
        guard data.count >= totalLength else { return nil }

        let payload = data[data.startIndex + 4 ..< data.startIndex + totalLength]
        let ms = payload.startIndex

        // VXSF — frame
        if payload[ms] == 0x56 && payload[ms+1] == 0x58 &&
           payload[ms+2] == 0x53 && payload[ms+3] == 0x46 {
            if let frame = decode(Data(payload)) {
                return (.frame(frame), totalLength)
            }
            return nil
        }

        // VXST — status
        if payload[ms] == 0x56 && payload[ms+1] == 0x58 &&
           payload[ms+2] == 0x53 && payload[ms+3] == 0x54 {
            if let status = decodeStatus(Data(payload)) {
                return (.status(status), totalLength)
            }
            return nil
        }

        // Unknown magic — skip this message
        return nil
    }

    // MARK: - Helpers

    @inline(__always)
    private static func readLE<T>(_ data: Data, _ offset: inout Data.Index) -> T {
        let size = MemoryLayout<T>.size
        let value = data[offset..<offset + size].withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        offset += size
        return value
    }
}
