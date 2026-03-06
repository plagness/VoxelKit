import Foundation
import Testing
import simd
@testable import VoxelStream

@Suite struct VoxelStreamEncoderDecoderTests {

    @Test func roundTripWithoutDepth() {
        let frame = VoxelStreamFrame(
            sequence: 42,
            timestamp: 1234.567,
            pose: simd_float4x4(
                SIMD4(1, 0, 0, 0),
                SIMD4(0, 1, 0, 0),
                SIMD4(0, 0, 1, 0),
                SIMD4(0.5, 1.2, -3.0, 1)
            ),
            intrinsics: simd_float3x3(
                SIMD3(1552, 0, 0),
                SIMD3(0, 1552, 0),
                SIMD3(960, 540, 1)
            ),
            imageJPEG: Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00]),
            imageWidth: 1920,
            imageHeight: 1080
        )

        let encoded = VoxelStreamEncoder.encode(frame)
        #expect(encoded.count == VoxelStreamEncoder.headerSize + 5)

        let decoded = VoxelStreamDecoder.decode(encoded)
        #expect(decoded != nil)

        let d = decoded!
        #expect(d.sequence == 42)
        #expect(abs(d.timestamp - 1234.567) < 0.001)
        #expect(d.imageWidth == 1920)
        #expect(d.imageHeight == 1080)
        #expect(d.imageJPEG.count == 5)
        #expect(d.depthFloat16 == nil)
        #expect(d.depthWidth == 0)
        #expect(d.depthHeight == 0)

        // Pose round-trip
        #expect(abs(d.pose.columns.3.x - 0.5) < 0.001)
        #expect(abs(d.pose.columns.3.y - 1.2) < 0.001)
        #expect(abs(d.pose.columns.3.z - (-3.0)) < 0.001)

        // Intrinsics round-trip
        #expect(abs(d.intrinsics.columns.0.x - 1552) < 0.01)
        #expect(abs(d.intrinsics.columns.1.y - 1552) < 0.01)
        #expect(abs(d.intrinsics.columns.2.x - 960) < 0.01)
        #expect(abs(d.intrinsics.columns.2.y - 540) < 0.01)
    }

    @Test func roundTripWithDepth() {
        let depthData = Data(repeating: 0x42, count: 256 * 192 * 2) // Float16

        let frame = VoxelStreamFrame(
            sequence: 100,
            timestamp: 0.0,
            pose: matrix_identity_float4x4,
            intrinsics: simd_float3x3(1),
            imageJPEG: Data([0xFF, 0xD8]),
            imageWidth: 1920,
            imageHeight: 1440,
            depthFloat16: depthData,
            depthWidth: 256,
            depthHeight: 192
        )

        let encoded = VoxelStreamEncoder.encode(frame)
        let expectedSize = VoxelStreamEncoder.headerSize + 2 + depthData.count
        #expect(encoded.count == expectedSize)

        let decoded = VoxelStreamDecoder.decode(encoded)!
        #expect(decoded.depthFloat16 != nil)
        #expect(decoded.depthFloat16!.count == depthData.count)
        #expect(decoded.depthWidth == 256)
        #expect(decoded.depthHeight == 192)
    }

    @Test func lengthPrefixedRoundTrip() {
        let frame = VoxelStreamFrame(
            sequence: 1,
            timestamp: 99.9,
            pose: matrix_identity_float4x4,
            intrinsics: simd_float3x3(1),
            imageJPEG: Data([0xAB, 0xCD]),
            imageWidth: 640,
            imageHeight: 480
        )

        let prefixed = VoxelStreamEncoder.encodeLengthPrefixed(frame)
        #expect(prefixed.count == 4 + VoxelStreamEncoder.headerSize + 2)

        let result = VoxelStreamDecoder.decodeLengthPrefixed(prefixed)
        #expect(result != nil)
        #expect(result!.0.sequence == 1)
        #expect(result!.1 == prefixed.count)
    }

    @Test func decoderRejectsGarbage() {
        let garbage = Data([0x00, 0x01, 0x02, 0x03])
        #expect(VoxelStreamDecoder.decode(garbage) == nil)
    }

    @Test func decoderRejectsTruncated() {
        let frame = VoxelStreamFrame(
            sequence: 1,
            timestamp: 0,
            pose: matrix_identity_float4x4,
            intrinsics: simd_float3x3(1),
            imageJPEG: Data(repeating: 0xFF, count: 100),
            imageWidth: 640,
            imageHeight: 480
        )
        let encoded = VoxelStreamEncoder.encode(frame)
        // Truncate payload
        let truncated = encoded[0..<VoxelStreamEncoder.headerSize + 10]
        #expect(VoxelStreamDecoder.decode(Data(truncated)) == nil)
    }
}
