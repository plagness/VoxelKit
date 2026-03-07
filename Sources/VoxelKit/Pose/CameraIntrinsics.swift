import Foundation
import AVFoundation
import simd
import CoreMedia

/// Pinhole camera intrinsics: focal length and principal point.
///
/// Used to back-project depth pixels to camera-space 3D rays.
public struct CameraIntrinsics: Sendable {
    /// Focal length X (pixels)
    public var fx: Float
    /// Focal length Y (pixels)
    public var fy: Float
    /// Principal point X (pixels)
    public var cx: Float
    /// Principal point Y (pixels)
    public var cy: Float
    /// Image width (pixels)
    public var width: Int
    /// Image height (pixels)
    public var height: Int

    // MARK: - Distortion (Brown-Conrady model)

    /// Radial distortion coefficients.
    public var k1: Float
    public var k2: Float
    public var k3: Float
    /// Tangential distortion coefficients.
    public var p1: Float
    public var p2: Float

    public init(fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
                k1: Float = 0, k2: Float = 0, k3: Float = 0, p1: Float = 0, p2: Float = 0) {
        self.fx = fx; self.fy = fy
        self.cx = cx; self.cy = cy
        self.width = width; self.height = height
        self.k1 = k1; self.k2 = k2; self.k3 = k3
        self.p1 = p1; self.p2 = p2
    }

    // MARK: - Device defaults

    /// iPhone 14 rear camera (1× lens, 4K).
    public static let iPhone14Default = CameraIntrinsics(
        fx: 1440, fy: 1440, cx: 960, cy: 720, width: 1920, height: 1440
    )

    /// iPhone 14 Pro rear camera (1× lens, 4K).
    public static let iPhone14ProDefault = CameraIntrinsics(
        fx: 2073, fy: 2073, cx: 1512, cy: 1008, width: 3024, height: 2016
    )

    /// iPhone 13 rear camera (1× lens, 1080p video).
    /// Approximate — overridden by actual CMSampleBuffer metadata when available.
    public static let iPhone13Default = CameraIntrinsics(
        fx: 1552, fy: 1552, cx: 960, cy: 540, width: 1920, height: 1080
    )

    /// Unitree Go2 Air front camera (wide-angle ~120 FOV, 720p H.264 via WebRTC).
    /// Distortion coefficients are approximate for a typical action-cam lens.
    /// For best results, calibrate with a checkerboard pattern.
    public static let go2Air = CameraIntrinsics(
        fx: 460, fy: 460, cx: 640, cy: 360, width: 1280, height: 720,
        k1: -0.25, k2: 0.06, k3: 0, p1: 0, p2: 0
    )

    /// 3×3 intrinsic matrix for use with neural pipeline.
    public var matrix3x3: simd_float3x3 {
        simd_float3x3(
            SIMD3<Float>(fx, 0, 0),
            SIMD3<Float>(0, fy, 0),
            SIMD3<Float>(cx, cy, 1)
        )
    }

    // MARK: - Distortion Correction

    /// Whether this camera has non-zero distortion coefficients.
    public var hasDistortion: Bool {
        k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0
    }

    /// Remove lens distortion from a pixel coordinate using iterative Brown-Conrady model.
    /// Returns undistorted pixel (u, v). Runs 5 Newton iterations.
    public func undistort(u: Float, v: Float) -> (Float, Float) {
        guard hasDistortion else { return (u, v) }

        // Normalize to camera coordinates
        let x0 = (u - cx) / fx
        let y0 = (v - cy) / fy

        // Iterative undistortion (inverse of distortion model)
        var x = x0
        var y = y0
        for _ in 0..<5 {
            let r2 = x * x + y * y
            let r4 = r2 * r2
            let r6 = r4 * r2
            let radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            let dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            let dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = (x0 - dx) / radial
            y = (y0 - dy) / radial
        }

        return (x * fx + cx, y * fy + cy)
    }

    // MARK: - Back-projection

    /// Convert a pixel (u, v) and depth `d` (metres) to camera-space 3D position.
    /// Applies distortion correction if distortion coefficients are set.
    public func unproject(u: Float, v: Float, depth: Float) -> SIMD3<Float> {
        let (uu, uv) = undistort(u: u, v: v)
        return SIMD3<Float>(
            (uu - cx) * depth / fx,
            (uv - cy) * depth / fy,
            depth
        )
    }

    /// Project a camera-space 3D point to pixel coordinates.
    /// Does NOT apply distortion (returns ideal pinhole projection).
    public func project(_ point: SIMD3<Float>) -> (u: Float, v: Float)? {
        guard point.z > 0 else { return nil }
        let u = fx * point.x / point.z + cx
        let v = fy * point.y / point.z + cy
        guard u >= 0, u < Float(width), v >= 0, v < Float(height) else { return nil }
        return (u, v)
    }

    /// Scale intrinsics to a different image resolution. Preserves distortion coefficients.
    public func scaled(toWidth newWidth: Int, height newHeight: Int) -> CameraIntrinsics {
        let sx = Float(newWidth) / Float(width)
        let sy = Float(newHeight) / Float(height)
        return CameraIntrinsics(fx: fx * sx, fy: fy * sy,
                                cx: cx * sx, cy: cy * sy,
                                width: newWidth, height: newHeight,
                                k1: k1, k2: k2, k3: k3, p1: p1, p2: p2)
    }
}

// MARK: - AVFoundation extraction

extension CameraIntrinsics {
    /// Extract intrinsics from a CMSampleBuffer's camera intrinsic matrix attachment.
    /// Falls back to `iPhone14Default` if calibration data is unavailable.
    public static func from(track: AVAssetTrack,
                             sampleBuffer: CMSampleBuffer? = nil) -> CameraIntrinsics {
        if let sb = sampleBuffer {
            return from(sampleBuffer: sb)
        }
        return .iPhone14Default
    }

    /// Extract intrinsics from a CMSampleBuffer.
    /// Falls back to `iPhone14Default` if calibration data is unavailable.
    public static func from(sampleBuffer: CMSampleBuffer) -> CameraIntrinsics {
        // Try camera intrinsic matrix (iOS 11+ / macOS 10.13+)
        if let rawMatrix = CMGetAttachment(sampleBuffer,
               key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
               attachmentModeOut: nil),
           let matrix = rawMatrix as? Data {
            // Actual video pixel dimensions from the sample buffer
            if let imgBuf = CMSampleBufferGetImageBuffer(sampleBuffer) {
                let w = CVPixelBufferGetWidth(imgBuf)
                let h = CVPixelBufferGetHeight(imgBuf)
                return from(intrinsicMatrixData: matrix, width: w, height: h)
            }
        }
        return .iPhone14Default
    }

    /// Parse a 3×3 column-major Float32 intrinsic matrix from Data.
    static func from(intrinsicMatrixData data: Data, width: Int, height: Int) -> CameraIntrinsics {
        guard data.count >= 36 else { return .iPhone14Default }
        return data.withUnsafeBytes { ptr in
            let floats = ptr.bindMemory(to: Float.self)
            // Column-major 3x3: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
            return CameraIntrinsics(
                fx: floats[0], fy: floats[4],
                cx: floats[6], cy: floats[7],
                width: width, height: height
            )
        }
    }
}
