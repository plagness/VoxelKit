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

    public init(fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int) {
        self.fx = fx; self.fy = fy
        self.cx = cx; self.cy = cy
        self.width = width; self.height = height
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

    // MARK: - Back-projection

    /// Convert a pixel (u, v) and depth `d` (metres) to camera-space 3D position.
    public func unproject(u: Float, v: Float, depth: Float) -> SIMD3<Float> {
        SIMD3<Float>(
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth
        )
    }

    /// Scale intrinsics to a different image resolution.
    public func scaled(toWidth newWidth: Int, height newHeight: Int) -> CameraIntrinsics {
        let sx = Float(newWidth) / Float(width)
        let sy = Float(newHeight) / Float(height)
        return CameraIntrinsics(fx: fx * sx, fy: fy * sy,
                                cx: cx * sx, cy: cy * sy,
                                width: newWidth, height: newHeight)
    }
}

// MARK: - AVFoundation extraction

extension CameraIntrinsics {
    /// Extract intrinsics from an AVAssetTrack's formatDescription metadata.
    /// Falls back to `iPhone14Default` if calibration data is unavailable.
    public static func from(track: AVAssetTrack,
                             sampleBuffer: CMSampleBuffer? = nil) -> CameraIntrinsics {
        // Try CMSampleBuffer camera intrinsic matrix (iOS 11+ / macOS 10.13+)
        if let sb = sampleBuffer,
           let rawMatrix = CMGetAttachment(sb,
               key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
               attachmentModeOut: nil) {
            if let matrix = rawMatrix as? Data {
                return CameraIntrinsics.from(intrinsicMatrixData: matrix,
                                             width: 1920, height: 1080)
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
