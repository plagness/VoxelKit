import SwiftUI
import Combine

/// Main window: left = top-down map preview, right = stats panel.
/// Uses a 0.25 s Timer (in .common runloop mode) to pull values from ProcessingState.
/// This is more reliable than @ObservedObject in NSHostingController CLI context.
struct MonitorView: View {

    let state: ProcessingState  // plain reference, we poll it on timer

    // Local @State snapshot — mutations here ALWAYS trigger SwiftUI re-render
    @State private var voxelCount: Int = 0
    @State private var fps: Double = 0
    @State private var progress: Double = 0
    @State private var etaSeconds: Double? = nil
    @State private var isFinished: Bool = false
    @State private var cpuCoresLoad: [Double] = []
    @State private var cpuTotalLoad: Double = 0
    @State private var memoryUsedGB: Double = 0
    @State private var memoryTotalGB: Double = 0
    @State private var gpuUtilization: Double = 0
    @State private var previewPositions: [SIMD3<Float>] = []

    private let timer = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 0) {
            // Left: map preview
            ZStack {
                Color.black
                VoxelPreviewView(positions: previewPositions)
                    .padding(8)

                if isFinished {
                    VStack {
                        Spacer()
                        Text("✓ Done — \(voxelCount.formatted()) voxels")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.green)
                            .padding(8)
                            .background(.black.opacity(0.7))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .padding(12)
                    }
                }
            }
            .frame(minWidth: 500, minHeight: 400)

            Divider()

            // Right: stats panel
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    fileInfoSection
                    processingSection
                    systemSection
                }
                .padding(16)
            }
            .frame(width: 260)
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(minWidth: 780, minHeight: 440)
        .onReceive(timer) { _ in pullFromState() }
    }

    // MARK: - Timer pull

    private func pullFromState() {
        voxelCount      = state.voxelCount
        fps             = state.fps
        progress        = state.progress
        etaSeconds      = state.etaSeconds
        isFinished      = state.isFinished
        cpuCoresLoad    = state.cpuCoresLoad
        cpuTotalLoad    = state.cpuTotalLoad
        memoryUsedGB    = state.memoryUsedGB
        memoryTotalGB   = state.memoryTotalGB
        gpuUtilization  = state.gpuUtilization
        previewPositions = state.previewPositions
    }

    // MARK: - Sections

    private var fileInfoSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader("INPUT")
            monoText(state.inputName.isEmpty ? "—" : state.inputName)
            if !state.gpuName.isEmpty {
                monoText("GPU: \(state.gpuName)", color: .secondary)
            }
        }
    }

    private var processingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("PROCESSING")
            statRow("Voxels", value: voxelCount.formatted())
            statRow("FPS",    value: String(format: "%.1f", fps))
            if let eta = etaSeconds {
                statRow("ETA", value: formatETA(eta))
            }
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(String(format: "%.0f%%", progress * 100))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Spacer()
                    if isFinished {
                        Text("Done").font(.system(size: 11)).foregroundStyle(.green)
                    }
                }
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .tint(isFinished ? .green : .accentColor)
            }
        }
    }

    private var systemSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("SYSTEM  (\(cpuCoresLoad.count) cores)")

            if !cpuCoresLoad.isEmpty {
                LazyVGrid(
                    columns: Array(repeating: GridItem(.fixed(22), spacing: 4), count: 5),
                    spacing: 4
                ) {
                    ForEach(Array(cpuCoresLoad.enumerated()), id: \.offset) { idx, load in
                        CoreBar(load: load, index: idx)
                    }
                }
                statRow("CPU avg", value: String(format: "%.0f%%", cpuTotalLoad * 100))
            }

            resourceBar(label: "MEM",
                        value: String(format: "%.1f / %.0f GB", memoryUsedGB, memoryTotalGB),
                        fraction: memoryTotalGB > 0 ? memoryUsedGB / memoryTotalGB : 0,
                        warningThreshold: 0.85)

            resourceBar(label: "GPU",
                        value: String(format: "%.0f%%", gpuUtilization * 100),
                        fraction: gpuUtilization,
                        warningThreshold: 0.90)
        }
    }

    // MARK: - Helpers

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .foregroundStyle(.secondary)
    }

    private func monoText(_ text: String, color: Color = .primary) -> some View {
        Text(text)
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(color)
            .lineLimit(1)
            .truncationMode(.middle)
    }

    private func statRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
        }
    }

    private func resourceBar(label: String, value: String, fraction: Double,
                              warningThreshold: Double) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label).font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
                Spacer()
                Text(value).font(.system(size: 11, design: .monospaced))
            }
            ProgressView(value: min(1, max(0, fraction)))
                .progressViewStyle(.linear)
                .tint(fraction > warningThreshold ? .orange : .accentColor)
        }
    }

    private func formatETA(_ seconds: Double) -> String {
        guard seconds.isFinite && seconds > 0 else { return "—" }
        let s = Int(seconds)
        return s > 60 ? "\(s / 60):\(String(format: "%02d", s % 60))" : "\(s)s"
    }
}

private struct CoreBar: View {
    let load: Double
    let index: Int

    var body: some View {
        VStack(spacing: 2) {
            GeometryReader { geo in
                ZStack(alignment: .bottom) {
                    RoundedRectangle(cornerRadius: 2).fill(Color.gray.opacity(0.2))
                    RoundedRectangle(cornerRadius: 2)
                        .fill(barColor)
                        .frame(height: geo.size.height * max(0, min(1, load)))
                }
            }
            .frame(height: 28)
            Text("\(index)")
                .font(.system(size: 7, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    private var barColor: Color {
        load > 0.8 ? .orange : load > 0.5 ? .yellow : .accentColor
    }
}
