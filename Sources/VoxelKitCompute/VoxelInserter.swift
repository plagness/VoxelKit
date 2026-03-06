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
    private let coloredPipeline: MTLComputePipelineState
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Config

    /// Flat depth used in MVP mode (no depth buffer).
    public var defaultDepth: Float = 2.0
    /// Scale factor: depth buffer value × depthScale = world metres.
    public var depthScale: Float = 1.0
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

        guard let coloredFn = library.makeFunction(name: "voxel_insert_colored") else {
            throw VoxelInserterError.functionNotFound("voxel_insert_colored")
        }
        self.coloredPipeline = try device.makeComputePipelineState(function: coloredFn)
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

    /// Process one frame with depth + camera color sampling.
    @discardableResult
    public func processFrameColored(
        pixelBuffer: CVPixelBuffer,
        depthBuffer: CVPixelBuffer,
        pose: Pose3D,
        intrinsics: CameraIntrinsics,
        world: BotMapWorld
    ) async throws -> Int {
        let voxels = try backProjectColored(
            depthBuffer: depthBuffer, colorBuffer: pixelBuffer,
            pose: pose, intrinsics: intrinsics
        )
        await world.insertColoredVoxelBatch(voxels)
        return voxels.count
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

    // MARK: - Colored depth back-projection

    private func backProjectColored(depthBuffer: CVPixelBuffer, colorBuffer: CVPixelBuffer,
                                     pose: Pose3D, intrinsics: CameraIntrinsics) throws -> [ColoredPosition] {
        let depthW = CVPixelBufferGetWidth(depthBuffer)
        let depthH = CVPixelBufferGetHeight(depthBuffer)
        let colorW = CVPixelBufferGetWidth(colorBuffer)
        let colorH = CVPixelBufferGetHeight(colorBuffer)

        let depthTexture = try makeDepthTexture(from: depthBuffer)
        let colorTexture = try makeColorTexture(from: colorBuffer)

        return try runColoredKernel(
            depthTexture: depthTexture, colorTexture: colorTexture,
            pose: pose, intrinsics: intrinsics,
            depthWidth: depthW, depthHeight: depthH,
            colorWidth: colorW, colorHeight: colorH
        )
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

    private func runColoredKernel(depthTexture: MTLTexture, colorTexture: MTLTexture,
                                   pose: Pose3D, intrinsics: CameraIntrinsics,
                                   depthWidth: Int, depthHeight: Int,
                                   colorWidth: Int, colorHeight: Int) throws -> [ColoredPosition] {
        let pixelCount = depthWidth * depthHeight
        let posBufferSize = pixelCount * MemoryLayout<SIMD3<Float>>.stride
        let colBufferSize = pixelCount * 4 // uchar4
        guard let posBuffer = device.makeBuffer(length: posBufferSize, options: .storageModeShared),
              let colBuffer = device.makeBuffer(length: colBufferSize, options: .storageModeShared)
        else { throw VoxelInserterError.bufferAllocationFailed }

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder()
        else { throw VoxelInserterError.encoderFailed }

        let scaledIntrinsics = intrinsics.scaled(toWidth: depthWidth, height: depthHeight)

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

        struct ColoredInsertParams {
            var colorWidth: UInt32; var colorHeight: UInt32
            var depthWidth: UInt32; var depthHeight: UInt32
        }
        var params = ColoredInsertParams(
            colorWidth: UInt32(colorWidth), colorHeight: UInt32(colorHeight),
            depthWidth: UInt32(depthWidth), depthHeight: UInt32(depthHeight)
        )

        encoder.setComputePipelineState(coloredPipeline)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(colorTexture, index: 1)
        encoder.setBytes(&mi, length: MemoryLayout<MetalIntrinsics>.size, index: 0)
        encoder.setBytes(&mp, length: MemoryLayout<MetalPose>.size,       index: 1)
        encoder.setBuffer(posBuffer, offset: 0, index: 2)
        encoder.setBuffer(colBuffer, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<ColoredInsertParams>.size, index: 4)

        let w = coloredPipeline.threadExecutionWidth
        let h = coloredPipeline.maxTotalThreadsPerThreadgroup / w
        encoder.dispatchThreadgroups(
            MTLSize(width: (depthWidth + w - 1) / w, height: (depthHeight + h - 1) / h, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1)
        )
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        let posPtr = posBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: pixelCount)
        let colPtr = colBuffer.contents().bindMemory(to: UInt8.self, capacity: pixelCount * 4)
        var result = [ColoredPosition]()
        result.reserveCapacity(pixelCount / (samplingStep * samplingStep))

        for y in stride(from: 0, to: depthHeight, by: samplingStep) {
            for x in stride(from: 0, to: depthWidth, by: samplingStep) {
                let idx = y * depthWidth + x
                let pos = posPtr[idx]
                if !pos.x.isNaN && !pos.y.isNaN && !pos.z.isNaN {
                    let ci = idx * 4
                    result.append(ColoredPosition(
                        position: pos,
                        color: (colPtr[ci], colPtr[ci + 1], colPtr[ci + 2])
                    ))
                }
            }
        }

        return result
    }

    private func makeColorTexture(from buffer: CVPixelBuffer) throws -> MTLTexture {
        let width  = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead]; desc.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: desc) else {
            throw VoxelInserterError.textureCreationFailed
        }

        // Render CIImage into RGBA8 pixel buffer then upload to texture
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        // Create a temporary RGBA buffer
        var rgbaBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                           kCVPixelFormatType_32BGRA, nil, &rgbaBuffer)
        guard let rgba = rgbaBuffer else { throw VoxelInserterError.textureCreationFailed }

        let ciImage = CIImage(cvPixelBuffer: buffer)
        ciContext.render(ciImage, to: rgba)

        CVPixelBufferLockBaseAddress(rgba, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(rgba, .readOnly) }
        guard let addr = CVPixelBufferGetBaseAddress(rgba) else { return texture }

        // BGRA → we'll read as RGBA in shader, but since we use rgba8Unorm Metal format
        // and the data is BGRA, we need to swizzle. Simpler: just upload BGRA and swizzle in shader.
        // Actually, let's use bgra8Unorm format instead.
        // Re-create with bgra8Unorm
        let desc2 = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc2.usage = [.shaderRead]; desc2.storageMode = .shared
        guard let texture2 = device.makeTexture(descriptor: desc2) else {
            throw VoxelInserterError.textureCreationFailed
        }
        texture2.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                         withBytes: addr, bytesPerRow: CVPixelBufferGetBytesPerRow(rgba))
        return texture2
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
        guard let addr = CVPixelBufferGetBaseAddress(buffer) else { return texture }

        let fmt = CVPixelBufferGetPixelFormatType(buffer)
        if fmt == kCVPixelFormatType_DepthFloat32 || fmt == kCVPixelFormatType_DisparityFloat32 {
            // Native Float32 — copy directly
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: addr, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer))
        } else {
            // Float16 — convert to Float32
            let count = width * height
            let src = addr.bindMemory(to: UInt16.self, capacity: count)
            var f32 = [Float32](repeating: 0, count: count)
            for i in 0..<count {
                f32[i] = half16toFloat32(src[i])
            }
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: f32, bytesPerRow: width * 4)
        }
        return texture
    }

    private func half16toFloat32(_ h: UInt16) -> Float32 {
        let sign     = UInt32(h & 0x8000) << 16
        let exp      = UInt32(h & 0x7C00)
        let mantissa = UInt32(h & 0x03FF)
        let bits: UInt32
        if exp == 0           { bits = sign }
        else if exp == 0x7C00 { bits = sign | 0x7F800000 | (mantissa << 13) }
        else                  { bits = sign | ((exp + 0x1C000) << 13) | (mantissa << 13) }
        return Float32(bitPattern: bits)
    }
}

public enum VoxelInserterError: Error {
    case noCommandQueue
    case functionNotFound(String)
    case bufferAllocationFailed
    case encoderFailed
    case textureCreationFailed
}
