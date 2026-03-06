import Foundation
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

        // Validate payload size
        let expectedSize = VoxelStreamEncoder.headerSize + Int(jpegSize) + Int(depthSize) + Int(worldPtsSize)
        guard data.count >= expectedSize else { return nil }

        let jpegStart = offset
        let jpegEnd = jpegStart + Int(jpegSize)
        let imageJPEG = data[jpegStart..<jpegEnd]

        var depthFloat16: Data? = nil
        if depthSize > 0 {
            let depthStart = jpegEnd
            let depthEnd = depthStart + Int(depthSize)
            depthFloat16 = data[depthStart..<depthEnd]
        }

        var worldPoints: Data? = nil
        if worldPtsSize > 0 {
            let wpStart = jpegEnd + Int(depthSize)
            let wpEnd = wpStart + Int(worldPtsSize)
            worldPoints = data[wpStart..<wpEnd]
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
            worldPointCount: worldPointCount
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

    // MARK: - Helpers

    @inline(__always)
    private static func readLE<T>(_ data: Data, _ offset: inout Data.Index) -> T {
        let size = MemoryLayout<T>.size
        let value = data[offset..<offset + size].withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        offset += size
        return value
    }
}
