import Foundation
import CoreVideo
import simd

/// Projects camera RGB onto existing 3D points by sampling the video frame.
///
/// Used by VIOSession to colorize LiDAR point clouds using the camera image.
/// Supports NV12 (420v/420f) and BGRA pixel formats. Applies distortion-aware projection.
public struct VideoColorSampler: Sendable {

    /// Colorize 3D points by projecting them into the camera frame and sampling RGB.
    ///
    /// - Parameters:
    ///   - points: World-space 3D positions to colorize.
    ///   - pixelBuffer: Camera frame (NV12 or BGRA).
    ///   - cameraPose: Camera pose in world space.
    ///   - intrinsics: Camera intrinsics (with distortion coefficients).
    /// - Returns: Array of ColoredPosition for points that project into frame bounds.
    public static func colorize(
        points: [SIMD3<Float>],
        pixelBuffer: CVPixelBuffer,
        cameraPose: Pose3D,
        intrinsics: CameraIntrinsics
    ) -> [ColoredPosition] {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // Build inverse camera transform (world → camera space)
        let invRotation = cameraPose.rotation.inverse
        let cameraPos = cameraPose.position

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let scaled = intrinsics.scaled(toWidth: width, height: height)

        var result = [ColoredPosition]()
        result.reserveCapacity(points.count / 4)

        for worldPoint in points {
            // Transform to camera space
            let relative = worldPoint - cameraPos
            let camPoint = invRotation.act(relative)

            // Must be in front of camera
            guard camPoint.z > 0.05 else { continue }

            // Project to pixel coordinates (pinhole, no distortion applied to projection)
            let u = scaled.fx * camPoint.x / camPoint.z + scaled.cx
            let v = scaled.fy * camPoint.y / camPoint.z + scaled.cy

            let ui = Int(u)
            let vi = Int(v)
            guard ui >= 0, ui < width, vi >= 0, vi < height else { continue }

            // Sample color
            let color: (UInt8, UInt8, UInt8)
            switch format {
            case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                 kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
                color = sampleNV12(pixelBuffer, x: ui, y: vi, width: width)
            case kCVPixelFormatType_32BGRA:
                color = sampleBGRA(pixelBuffer, x: ui, y: vi)
            default:
                continue
            }

            result.append(ColoredPosition(position: worldPoint, color: color))
        }

        return result
    }

    // MARK: - NV12 Sampling

    private static func sampleNV12(_ buffer: CVPixelBuffer, x: Int, y: Int, width: Int) -> (UInt8, UInt8, UInt8) {
        // Plane 0: Y (full resolution)
        let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let yVal = yBase.advanced(by: y * yStride + x).assumingMemoryBound(to: UInt8.self).pointee

        // Plane 1: CbCr (half resolution, interleaved)
        let uvBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let uvOffset = (y / 2) * uvStride + (x / 2) * 2
        let cb = uvBase.advanced(by: uvOffset).assumingMemoryBound(to: UInt8.self).pointee
        let cr = uvBase.advanced(by: uvOffset + 1).assumingMemoryBound(to: UInt8.self).pointee

        return nv12ToRGB(y: yVal, cb: cb, cr: cr)
    }

    // MARK: - BGRA Sampling

    private static func sampleBGRA(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (UInt8, UInt8, UInt8) {
        let base = CVPixelBufferGetBaseAddress(buffer)!
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let pixel = base.advanced(by: y * stride + x * 4).assumingMemoryBound(to: UInt8.self)
        return (pixel[2], pixel[1], pixel[0]) // BGRA → RGB
    }

    // MARK: - YCbCr → RGB (BT.601)

    private static func nv12ToRGB(y: UInt8, cb: UInt8, cr: UInt8) -> (UInt8, UInt8, UInt8) {
        let yf = Float(y) - 16
        let cbf = Float(cb) - 128
        let crf = Float(cr) - 128

        let r = 1.164 * yf + 1.596 * crf
        let g = 1.164 * yf - 0.392 * cbf - 0.813 * crf
        let b = 1.164 * yf + 2.017 * cbf

        return (
            UInt8(clamping: Int(r)),
            UInt8(clamping: Int(g)),
            UInt8(clamping: Int(b))
        )
    }
}
