import Foundation
import Metal
import CoreVideo
import CoreImage
import simd
import VoxelKit

/// GPU-accelerated depth back-projection: converts a depth estimate + camera pose
/// into world-space voxel positions and inserts them into `BotMapWorld`.
///
/// MVP mode: uses synthetic flat depth at `defaultDepth` metres.
/// Pose changes from `OpticalFlowPoseEstimator` create the map as the camera moves.
///
/// Upgrade path: inject a real depth buffer (e.g. from ARKit or Vision depth API)
/// via `processFrame(pixelBuffer:depthBuffer:pose:intrinsics:world:)`.
public final class VoxelInserter: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    // MARK: - Config

    /// Flat depth used in MVP mode (no depth buffer).
    public var defaultDepth: Float = 2.0
    /// Scale factor: depth buffer value × depthScale = world metres.
    public var depthScale: Float = 0.05
    public var minDepth: Float = 0.3
    public var maxDepth: Float = 8.0
    /// Process every Nth pixel (1 = full res, 8 = 1/64 pixels for speed).
    public var samplingStep: Int = 8

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw VoxelInserterError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try MetalShaders.makeLibrary(device: device)
        guard let fn = library.makeFunction(name: "voxel_insert") else {
            throw VoxelInserterError.functionNotFound("voxel_insert")
        }
        self.pipeline = try device.makeComputePipelineState(function: fn)
    }

    // MARK: - Public API

    /// Process one frame in MVP flat-depth mode.
    @discardableResult
    public func processFrame(
        pixelBuffer: CVPixelBuffer,
        pose: Pose3D,
        intrinsics: CameraIntrinsics,
        world: BotMapWorld
    ) async throws -> Int {
        let positions = try backProjectFlat(pixelBuffer: pixelBuffer, pose: pose, intrinsics: intrinsics)
        await world.insertVoxelBatch(positions)
        return positions.count
    }

    /// Process one frame with an explicit depth buffer.
    @discardableResult
    public func processFrame(
        pixelBuffer: CVPixelBuffer,
        depthBuffer: CVPixelBuffer,
        pose: Pose3D,
        intrinsics: CameraIntrinsics,
        world: BotMapWorld
    ) async throws -> Int {
        let positions = try backProject(depthBuffer: depthBuffer, pose: pose, intrinsics: intrinsics)
        await world.insertVoxelBatch(positions)
        return positions.count
    }

    // MARK: - Flat depth back-projection

    private func backProjectFlat(pixelBuffer: CVPixelBuffer, pose: Pose3D,
                                  intrinsics: CameraIntrinsics) throws -> [SIMD3<Float>] {
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false
        )
        desc.storageMode = .shared
        desc.usage = [.shaderRead]
        guard let depthTexture = device.makeTexture(descriptor: desc) else {
            throw VoxelInserterError.textureCreationFailed
        }

        let normalizedDepth = defaultDepth / depthScale
        var flatData = [Float](repeating: normalizedDepth, count: width * height)
        depthTexture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: &flatData,
            bytesPerRow: width * MemoryLayout<Float>.stride
        )

        return try runKernel(depthTexture: depthTexture, pose: pose, intrinsics: intrinsics,
                             width: width, height: height)
    }

    // MARK: - Depth buffer back-projection

    private func backProject(depthBuffer: CVPixelBuffer, pose: Pose3D,
                             intrinsics: CameraIntrinsics) throws -> [SIMD3<Float>] {
        let width  = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        let depthTexture = try makeDepthTexture(from: depthBuffer)
        return try runKernel(depthTexture: depthTexture, pose: pose, intrinsics: intrinsics,
                             width: width, height: height)
    }

    // MARK: - Metal kernel dispatch

    private func runKernel(depthTexture: MTLTexture, pose: Pose3D,
                            intrinsics: CameraIntrinsics,
                            width: Int, height: Int) throws -> [SIMD3<Float>] {
        let pixelCount = width * height
        let bufferSize = pixelCount * MemoryLayout<SIMD3<Float>>.stride
        guard let outBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw VoxelInserterError.bufferAllocationFailed
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder()
        else { throw VoxelInserterError.encoderFailed }

        let scaledIntrinsics = intrinsics.scaled(toWidth: width, height: height)

        struct MetalIntrinsics {
            var fx: Float; var fy: Float; var cx: Float; var cy: Float
            var depthScale: Float; var minDepth: Float; var maxDepth: Float; var _pad: Float
        }
        var mi = MetalIntrinsics(
            fx: scaledIntrinsics.fx, fy: scaledIntrinsics.fy,
            cx: scaledIntrinsics.cx, cy: scaledIntrinsics.cy,
            depthScale: depthScale, minDepth: minDepth, maxDepth: maxDepth, _pad: 0
        )

        let rm = float3x3(pose.rotation)
        struct MetalPose {
            var r0x: Float; var r0y: Float; var r0z: Float; var _p0: Float
            var r1x: Float; var r1y: Float; var r1z: Float; var _p1: Float
            var r2x: Float; var r2y: Float; var r2z: Float; var _p2: Float
            var px: Float; var py: Float; var pz: Float; var _p3: Float
        }
        var mp = MetalPose(
            r0x: rm.columns.0.x, r0y: rm.columns.0.y, r0z: rm.columns.0.z, _p0: 0,
            r1x: rm.columns.1.x, r1y: rm.columns.1.y, r1z: rm.columns.1.z, _p1: 0,
            r2x: rm.columns.2.x, r2y: rm.columns.2.y, r2z: rm.columns.2.z, _p2: 0,
            px: pose.position.x, py: pose.position.y, pz: pose.position.z, _p3: 0
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setBytes(&mi, length: MemoryLayout<MetalIntrinsics>.size, index: 0)
        encoder.setBytes(&mp, length: MemoryLayout<MetalPose>.size,       index: 1)
        encoder.setBuffer(outBuffer, offset: 0, index: 2)

        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        encoder.dispatchThreadgroups(
            MTLSize(width: (width + w - 1) / w, height: (height + h - 1) / h, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1)
        )
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        let ptr = outBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: pixelCount)
        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(pixelCount / (samplingStep * samplingStep))

        for y in stride(from: 0, to: height, by: samplingStep) {
            for x in stride(from: 0, to: width, by: samplingStep) {
                let pos = ptr[y * width + x]
                if !pos.x.isNaN && !pos.y.isNaN && !pos.z.isNaN {
                    positions.append(pos)
                }
            }
        }

        return positions
    }

    private func makeDepthTexture(from buffer: CVPixelBuffer) throws -> MTLTexture {
        let width  = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead]; desc.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: desc) else {
            throw VoxelInserterError.textureCreationFailed
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        if let addr = CVPixelBufferGetBaseAddress(buffer) {
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: addr, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer))
        }
        return texture
    }
}

public enum VoxelInserterError: Error {
    case noCommandQueue
    case functionNotFound(String)
    case bufferAllocationFailed
    case encoderFailed
    case textureCreationFailed
}
