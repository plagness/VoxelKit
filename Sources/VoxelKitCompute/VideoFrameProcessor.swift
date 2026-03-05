import Foundation
import Metal
import MetalKit
import CoreVideo
import simd

/// Converts CVPixelBuffer (YUV 4:2:0 biplanar) to RGB MTLTexture using Metal compute.
///
/// Used by the video capture pipeline to convert AVAssetReader output
/// to a format suitable for Vision requests and VoxelInserter.
public final class VideoFrameProcessor: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw VideoProcessorError.noCommandQueue
        }
        self.commandQueue = queue

        // Load compute pipeline from inline Metal source (works in CLI + swift test)
        let library = try MetalShaders.makeLibrary(device: device)
        guard let fn = library.makeFunction(name: "yuv_to_rgb") else {
            throw VideoProcessorError.functionNotFound("yuv_to_rgb")
        }
        self.pipeline = try device.makeComputePipelineState(function: fn)

        // Create texture cache for zero-copy CVPixelBuffer → MTLTexture
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    /// Convert a YUV pixel buffer to an RGB MTLTexture.
    public func process(_ pixelBuffer: CVPixelBuffer) throws -> MTLTexture {
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        guard let cache = textureCache else { throw VideoProcessorError.noTextureCache }

        // Luma plane (Y)
        var yMTLTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            .r8Unorm, width, height, 0, &yMTLTexture
        )

        // Chroma plane (CbCr)
        var cbcrMTLTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            .rg8Unorm, width / 2, height / 2, 1, &cbcrMTLTexture
        )

        guard let yTex = yMTLTexture.flatMap({ CVMetalTextureGetTexture($0) }),
              let cbcrTex = cbcrMTLTexture.flatMap({ CVMetalTextureGetTexture($0) })
        else { throw VideoProcessorError.textureCreationFailed }

        // Output RGB texture
        let rgbDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false
        )
        rgbDesc.usage = [.shaderRead, .shaderWrite]
        rgbDesc.storageMode = .private
        guard let rgbTexture = device.makeTexture(descriptor: rgbDesc) else {
            throw VideoProcessorError.textureCreationFailed
        }

        // Encode compute pass
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder()
        else { throw VideoProcessorError.encoderFailed }

        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(yTex,      index: 0)
        encoder.setTexture(cbcrTex,   index: 1)
        encoder.setTexture(rgbTexture, index: 2)

        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let threadGroups = MTLSize(
            width:  (width  + w - 1) / w,
            height: (height + h - 1) / h,
            depth:  1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        return rgbTexture
    }
}

public enum VideoProcessorError: Error {
    case noCommandQueue
    case functionNotFound(String)
    case noTextureCache
    case textureCreationFailed
    case encoderFailed
}
