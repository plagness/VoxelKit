import Foundation
import ArgumentParser
import VoxelKit
import AppKit
import SwiftUI

struct ViewCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "view",
        abstract: "View a .botmap file in 3D (interactive orbit/zoom/pan)."
    )

    @Argument(help: "Path to .botmap file")
    var input: String

    @Option(help: "Max points to display (default: 200000)")
    var maxPoints: Int = 200_000

    mutating func run() async throws {
        let url = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError.fileNotFound(url.path)
        }

        print("Loading \(url.lastPathComponent)…")
        let world = try await BotMapWorld.load(from: url)
        let voxelCount = await world.voxelCount
        let positions  = await world.samplePositions(maxCount: maxPoints)
        print("Loaded \(voxelCount.formatted()) voxels — displaying \(positions.count.formatted()) points")
        print("Controls: Left-drag = orbit  |  Right-drag = pan  |  Scroll = zoom")

        await MainActor.run {
            let app = NSApplication.shared
            app.setActivationPolicy(.regular)

            let w = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 1000, height: 750),
                styleMask:   [.titled, .closable, .resizable, .miniaturizable],
                backing:     .buffered,
                defer:       false
            )
            w.title = "\(url.lastPathComponent) — \(voxelCount.formatted()) voxels"
            w.contentViewController = NSHostingController(
                rootView: Voxel3DViewerView(positions: positions, filename: url.lastPathComponent)
            )
            w.center()
            w.makeKeyAndOrderFront(nil)
            app.activate(ignoringOtherApps: true)
        }

        // Keep running until all windows are closed (ArgumentParser RunLoop handles events)
        while await MainActor.run(body: { !NSApplication.shared.windows.filter { $0.isVisible }.isEmpty }) {
            try await Task.sleep(nanoseconds: 200_000_000)
        }
    }
}

// MARK: - Standalone viewer SwiftUI view

private struct Voxel3DViewerView: View {
    let positions: [SIMD3<Float>]
    let filename: String

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            VoxelScene3DView(positions: positions)

            HStack(spacing: 12) {
                Text(filename)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text("\(positions.count.formatted()) points")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("Left-drag: orbit  ·  Right-drag: pan  ·  Scroll: zoom")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.4))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.black.opacity(0.6))
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}
