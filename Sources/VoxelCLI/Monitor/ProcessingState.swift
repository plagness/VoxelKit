import Foundation
import simd

/// Shared state between the processing pipeline and the monitor UI.
/// Plain class — updates written via DispatchQueue.main.async,
/// read by MonitorView on a 0.25 s Timer (bypasses @ObservedObject issues in CLI NSHostingController).
public final class ProcessingState: @unchecked Sendable {

    // MARK: - Processing
    public var voxelCount: Int = 0
    public var fps: Double = 0
    public var progress: Double = 0
    public var etaSeconds: Double? = nil
    public var isFinished: Bool = false

    // MARK: - File info (set before window opens, never changes)
    public var inputName: String = ""
    public var outputPath: String = ""
    public var gpuName: String = ""

    // MARK: - System metrics (written on main thread by SystemMonitor)
    public var cpuCoresLoad: [Double] = []
    public var cpuTotalLoad: Double = 0
    public var memoryUsedGB: Double = 0
    public var memoryTotalGB: Double = 0
    public var gpuUtilization: Double = 0

    // MARK: - Map preview (written on main thread by sampler task)
    public var previewPositions: [SIMD3<Float>] = []

    public init() {}
}
