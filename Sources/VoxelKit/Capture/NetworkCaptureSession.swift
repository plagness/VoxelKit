import Foundation
import Network
import CoreImage
import CoreVideo
import simd
import os
import VoxelStream

private let logger = Logger(subsystem: "com.voxelkit", category: "NetworkCapture")

/// Receives live ARKit frames from an iPhone over Network.framework (USB/Wi-Fi).
///
/// Discovers the iPhone via Bonjour (`_voxelkit._tcp`), connects, receives
/// length-prefixed `VoxelStreamFrame` blobs, decodes them, and invokes
/// the `onFrame` callback with the pixel buffer, intrinsics, pose, and
/// optional depth buffer.
public actor NetworkCaptureSession: CaptureSession {

    // MARK: - Types

    /// Callback for each received frame.
    /// Parameters: pixelBuffer (from JPEG), intrinsics, pose, depthBuffer (nil if no LiDAR).
    public typealias FrameCallback = @Sendable (
        CVPixelBuffer, CameraIntrinsics, Pose3D, CVPixelBuffer?
    ) async -> Void

    /// Callback for world-space 3D points with sampled camera colors.
    /// Parameters: array of (worldPosition, RGB color).
    public typealias WorldPointsCallback = @Sendable (
        [(SIMD3<Float>, (UInt8, UInt8, UInt8))]
    ) async -> Void

    public enum ConnectionState: Sendable {
        case searching
        case connecting
        case connected
        case disconnected
    }

    // MARK: - State

    public let progress: AsyncStream<CaptureProgress>
    private let progressContinuation: AsyncStream<CaptureProgress>.Continuation

    public var onFrame: FrameCallback?
    public var onWorldPoints: WorldPointsCallback?
    public var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?

    private var browser: NWBrowser?
    private var connection: NWConnection?
    private var cancelled = false
    private var processedFrames = 0
    private var receiveBuffer = Data()

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Init

    public init() {
        var cont: AsyncStream<CaptureProgress>.Continuation!
        self.progress = AsyncStream { cont = $0 }
        self.progressContinuation = cont
    }

    public func setOnFrame(_ callback: @escaping FrameCallback) {
        self.onFrame = callback
    }

    public func setOnWorldPoints(_ callback: @escaping WorldPointsCallback) {
        self.onWorldPoints = callback
    }

    public func setOnConnectionStateChanged(_ callback: @escaping @Sendable (ConnectionState) -> Void) {
        self.onConnectionStateChanged = callback
    }

    // MARK: - CaptureSession

    public func start() async throws {
        cancelled = false
        processedFrames = 0
        receiveBuffer = Data()

        onConnectionStateChanged?(.searching)

        // Browse for _voxelkit._tcp Bonjour service
        let params = NWParameters.tcp
        let browser = NWBrowser(for: .bonjour(type: "_voxelkit._tcp", domain: nil), using: params)
        self.browser = browser

        // Wait for endpoint discovery
        let endpoint = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<NWEndpoint, Error>) in
            let resumed = OSAllocatedUnfairLock(initialState: false)
            browser.browseResultsChangedHandler = { results, _ in
                guard let result = results.first else { return }
                let shouldResume = resumed.withLock { val -> Bool in
                    guard !val else { return false }
                    val = true
                    return true
                }
                guard shouldResume else { return }
                browser.cancel()
                cont.resume(returning: result.endpoint)
            }
            browser.stateUpdateHandler = { state in
                if case .failed(let error) = state {
                    let shouldResume = resumed.withLock { val -> Bool in
                        guard !val else { return false }
                        val = true
                        return true
                    }
                    guard shouldResume else { return }
                    cont.resume(throwing: error)
                }
            }
            browser.start(queue: .global(qos: .userInitiated))
        }

        guard !cancelled else { return }

        // Connect to the discovered endpoint
        onConnectionStateChanged?(.connecting)
        let conn = NWConnection(to: endpoint, using: .tcp)
        self.connection = conn

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let resumed = OSAllocatedUnfairLock(initialState: false)
            conn.stateUpdateHandler = { state in
                let shouldResume = resumed.withLock { val -> Bool in
                    guard !val else { return false }
                    val = true
                    return true
                }
                guard shouldResume else { return }
                switch state {
                case .ready:
                    cont.resume()
                case .failed(let error):
                    cont.resume(throwing: error)
                case .cancelled:
                    cont.resume(throwing: CancellationError())
                default:
                    // Not a terminal state — undo the lock
                    resumed.withLock { $0 = false }
                }
            }
            conn.start(queue: .global(qos: .userInitiated))
        }

        guard !cancelled else { return }
        onConnectionStateChanged?(.connected)

        // Receive loop
        var lastProgressTime = Date.now
        var fpsFrameCount = 0
        var lastFPSTime = Date.now

        while !cancelled {
            let chunk: Data
            do {
                chunk = try await receiveChunk(from: conn)
            } catch {
                break
            }

            receiveBuffer.append(chunk)

            // Process all complete frames in the buffer
            while let (frame, consumed) = VoxelStreamDecoder.decodeLengthPrefixed(receiveBuffer) {
                receiveBuffer.removeFirst(consumed)

                // Stop-marker: iPhone stopped recording
                if frame.sequence == VoxelStreamFrame.stopMarkerSequence {
                    cancelled = true
                    break
                }

                processedFrames += 1
                fpsFrameCount += 1

                let pose = Pose3D(arTransform: frame.pose)
                let intrinsics = CameraIntrinsics(
                    arIntrinsics: frame.intrinsics,
                    width: Int(frame.imageWidth),
                    height: Int(frame.imageHeight)
                )

                // Convert and deliver frame (depth pipeline)
                if let cb = onFrame {
                    if let pixelBuffer = jpegToPixelBuffer(
                        frame.imageJPEG,
                        width: Int(frame.imageWidth),
                        height: Int(frame.imageHeight)
                    ) {
                        let depthBuffer = frame.depthFloat16.flatMap {
                            depthDataToPixelBuffer(
                                $0,
                                width: Int(frame.depthWidth),
                                height: Int(frame.depthHeight)
                            )
                        }
                        await cb(pixelBuffer, intrinsics, pose, depthBuffer)
                    }
                }

                // World points pipeline (plane vertices + feature points)
                if let wpCb = onWorldPoints,
                   frame.worldPointCount > 0,
                   let wpData = frame.worldPoints {
                    if processedFrames <= 3 {
                        logger.info("Frame \(self.processedFrames): \(frame.worldPointCount) world points, \(wpData.count) bytes")
                    }
                    let coloredPts = projectWorldPointsToColor(
                        wpData,
                        count: Int(frame.worldPointCount),
                        jpegData: frame.imageJPEG,
                        imageWidth: Int(frame.imageWidth),
                        imageHeight: Int(frame.imageHeight),
                        pose: frame.pose,
                        intrinsics: frame.intrinsics
                    )
                    if !coloredPts.isEmpty {
                        await wpCb(coloredPts)
                    }
                }

                // Emit progress ~4 Hz
                let now = Date.now
                if now.timeIntervalSince(lastProgressTime) >= 0.25 {
                    let elapsed = now.timeIntervalSince(lastFPSTime)
                    let fps = elapsed > 0 ? Double(fpsFrameCount) / elapsed : 0
                    fpsFrameCount = 0
                    lastFPSTime = now
                    lastProgressTime = now

                    progressContinuation.yield(CaptureProgress(
                        processedFrames: processedFrames,
                        totalFrames: 0, // unknown for live stream
                        insertedVoxelCount: 0,
                        fps: fps,
                        currentPose: Pose3D(arTransform: frame.pose),
                        stream: StreamStats(
                            latencyMs: 0,
                            droppedFrames: 0,
                            bytesPerSecond: 0
                        )
                    ))
                }
            }
        }

        onConnectionStateChanged?(.disconnected)
        progressContinuation.finish()
    }

    public func cancel() {
        cancelled = true
        connection?.cancel()
        browser?.cancel()
    }

    // MARK: - Network helpers

    private func receiveChunk(from conn: NWConnection) async throws -> Data {
        try await withCheckedThrowingContinuation { cont in
            conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, _, error in
                if let error {
                    cont.resume(throwing: error)
                } else if let data, !data.isEmpty {
                    cont.resume(returning: data)
                } else {
                    cont.resume(throwing: NetworkCaptureError.connectionClosed)
                }
            }
        }
    }

    // MARK: - World points → colored positions

    private func projectWorldPointsToColor(
        _ wpData: Data,
        count: Int,
        jpegData: Data,
        imageWidth: Int,
        imageHeight: Int,
        pose: simd_float4x4,
        intrinsics: simd_float3x3
    ) -> [(SIMD3<Float>, (UInt8, UInt8, UInt8))] {
        // Decode JPEG to get pixel data for color sampling
        guard let ciImage = CIImage(data: jpegData) else { return [] }
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: imageWidth,
            kCVPixelBufferHeightKey as String: imageHeight,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, imageWidth, imageHeight,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pb = pixelBuffer else { return [] }
        ciContext.render(ciImage, to: pb)

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let baseAddr = CVPixelBufferGetBaseAddress(pb) else { return [] }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)

        // View matrix = inverse of camera pose
        let viewMatrix = pose.inverse

        let fx = intrinsics.columns.0.x
        let fy = intrinsics.columns.1.y
        let cx = intrinsics.columns.2.x
        let cy = intrinsics.columns.2.y

        var result: [(SIMD3<Float>, (UInt8, UInt8, UInt8))] = []
        result.reserveCapacity(count)

        let elemStride = MemoryLayout<SIMD3<Float>>.stride // 16 bytes (SIMD3<Float> is padded)
        wpData.withUnsafeBytes { rawBuf in
            let ptr = rawBuf.baseAddress!
            for i in 0..<count {
                let worldPt = ptr.advanced(by: i * elemStride).assumingMemoryBound(to: SIMD3<Float>.self).pointee

                // Project world point to camera space
                // ARKit camera looks along -Z, so visible points have z < 0
                let camPt4 = viewMatrix * SIMD4<Float>(worldPt.x, worldPt.y, worldPt.z, 1.0)
                let depth = -camPt4.z  // positive for visible points
                guard depth > 0.01 else { continue }

                // ARKit intrinsics: u = fx * X/depth + cx, v = fy * (-Y)/depth + cy
                // Y is flipped because camera Y is up but image v goes down
                let u = fx * camPt4.x / depth + cx
                let v = fy * (-camPt4.y) / depth + cy

                let px = Int(u)
                let py = Int(v)
                guard px >= 0, px < imageWidth, py >= 0, py < imageHeight else { continue }

                // Sample BGRA pixel
                let pixel = baseAddr.advanced(by: py * bytesPerRow + px * 4)
                    .assumingMemoryBound(to: UInt8.self)
                let b = pixel[0]
                let g = pixel[1]
                let r = pixel[2]

                result.append((worldPt, (r, g, b)))
            }
        }

        if count > 0 && result.isEmpty {
            logger.warning("projectWorldPointsToColor: \(count) input points, 0 projected — check pose/intrinsics")
        }

        return result
    }

    // MARK: - Image conversion

    private func jpegToPixelBuffer(_ jpegData: Data, width: Int, height: Int) -> CVPixelBuffer? {
        guard let ciImage = CIImage(data: jpegData) else { return nil }

        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pb = pixelBuffer else { return nil }

        ciContext.render(ciImage, to: pb)
        return pb
    }

    private func depthDataToPixelBuffer(_ data: Data, width: Int, height: Int) -> CVPixelBuffer? {
        guard width > 0, height > 0 else { return nil }

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_DepthFloat16, nil, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pb = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }

        guard let dest = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        let srcBytesPerRow = width * 2 // Float16 = 2 bytes

        data.withUnsafeBytes { src in
            for row in 0..<height {
                let srcOffset = row * srcBytesPerRow
                let dstOffset = row * bytesPerRow
                guard srcOffset + srcBytesPerRow <= data.count else { return }
                memcpy(dest + dstOffset, src.baseAddress! + srcOffset, srcBytesPerRow)
            }
        }

        return pb
    }
}

// MARK: - Extensions

extension Pose3D {
    /// Create from an ARKit camera transform (4x4 column-major).
    public init(arTransform m: simd_float4x4) {
        let t = m.columns.3
        self.init(
            position: SIMD3(t.x, t.y, t.z),
            rotation: simd_quatf(m)
        )
    }
}

extension CameraIntrinsics {
    /// Create from ARKit's 3x3 intrinsics matrix.
    public init(arIntrinsics m: simd_float3x3, width: Int, height: Int) {
        self.init(
            fx: m.columns.0.x,
            fy: m.columns.1.y,
            cx: m.columns.2.x,
            cy: m.columns.2.y,
            width: width,
            height: height
        )
    }
}

// MARK: - Errors

public enum NetworkCaptureError: Error, LocalizedError {
    case connectionClosed
    case browseTimeout

    public var errorDescription: String? {
        switch self {
        case .connectionClosed: return "Network connection closed"
        case .browseTimeout: return "Could not find iPhone (Bonjour browse timeout)"
        }
    }
}
