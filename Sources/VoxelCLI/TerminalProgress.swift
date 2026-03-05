import Foundation

/// ANSI terminal progress bar, updated in-place via carriage return.
///
/// Example output:
/// ```
/// [████████████░░░░░░░░] 67% | 14823 voxels | 24.3 fps | ETA 0:12
/// ```
public struct TerminalProgress {

    private let barWidth: Int

    public init(barWidth: Int = 20) {
        self.barWidth = barWidth
    }

    /// Print a progress line, overwriting the current line.
    public func update(
        fraction: Double?,      // 0.0–1.0, nil = unknown
        voxelCount: Int,
        fps: Double,
        etaSeconds: Double?,
        status: String? = nil
    ) {
        let bar: String
        if let f = fraction {
            let filled = Int(Double(barWidth) * f.clamped(to: 0...1))
            let empty = barWidth - filled
            bar = "[" + String(repeating: "█", count: filled) + String(repeating: "░", count: empty) + "]"
        } else {
            // Spinner for unknown progress
            bar = "[" + String(repeating: "─", count: barWidth) + "]"
        }

        let pct = fraction.map { String(format: "%3.0f%%", $0 * 100) } ?? "  ?%"
        let voxStr = formatCount(voxelCount) + " voxels"
        let fpsStr = String(format: "%.1f fps", fps)
        let eta = etaSeconds.map { "ETA \(formatTime($0))" } ?? ""
        let statusStr = status.map { " [\($0)]" } ?? ""

        var parts = [bar, pct, "|", voxStr, "|", fpsStr]
        if !eta.isEmpty { parts += ["|", eta] }

        let line = parts.joined(separator: " ") + statusStr
        // Carriage return to overwrite line, no newline
        print("\r" + line, terminator: "")
        fflush(stdout)
    }

    /// Print a final newline after progress is complete.
    public func finish(message: String? = nil) {
        if let msg = message {
            print("\r\(msg)")
        } else {
            print("")
        }
    }

    // MARK: - Formatting

    private func formatCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    private func formatTime(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "0:\(String(format: "%02d", s))" }
        return "\(s / 60):\(String(format: "%02d", s % 60))"
    }
}

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        Swift.max(range.lowerBound, Swift.min(range.upperBound, self))
    }
}
