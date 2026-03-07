import Foundation

/// Status message sent periodically from iPhone to Mac.
///
/// Wire format "VXST" (14 bytes, little-endian):
/// ```
/// [0..3]   magic "VXST"           4B
/// [4]      cameraActive UInt8     1B   — 0=off, 1=on
/// [5]      gyroActive UInt8       1B   — 0=off, 1=on
/// [6]      selectedCamera UInt8   1B   — 0=ARKit wide, 1=ultra-wide, 2=dog
/// [7]      batteryLevel UInt8     1B   — 0-100%
/// [8]      thermalState UInt8     1B   — 0=nominal..3=critical
/// [9]      trackingState UInt8    1B   — 0=notAvailable, 1=limited, 2=normal
/// [10..13] reserved               4B
/// ```
public struct DeviceStatusMessage: Sendable {

    public static let magic: [UInt8] = [0x56, 0x58, 0x53, 0x54] // "VXST"
    public static let messageSize = 14

    public let cameraActive: Bool
    public let gyroActive: Bool
    /// 0 = ARKit wide, 1 = ultra-wide, 2 = dog camera
    public let selectedCamera: UInt8
    public let batteryLevel: UInt8
    /// 0 = nominal, 1 = fair, 2 = serious, 3 = critical
    public let thermalState: UInt8
    /// 0 = notAvailable, 1 = limited, 2 = normal
    public let trackingState: UInt8

    public init(
        cameraActive: Bool,
        gyroActive: Bool,
        selectedCamera: UInt8,
        batteryLevel: UInt8,
        thermalState: UInt8,
        trackingState: UInt8
    ) {
        self.cameraActive = cameraActive
        self.gyroActive = gyroActive
        self.selectedCamera = selectedCamera
        self.batteryLevel = batteryLevel
        self.thermalState = thermalState
        self.trackingState = trackingState
    }

    /// Human-readable label for the selected camera.
    public var cameraLabel: String {
        switch selectedCamera {
        case 0: return "Wide"
        case 1: return "Ultra-Wide"
        case 2: return "Dog"
        default: return "Unknown"
        }
    }
}
