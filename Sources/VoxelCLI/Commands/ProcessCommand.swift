import Foundation
import ArgumentParser
import VoxelKit
import VoxelKitCompute
import Metal
import AppKit
import SwiftUI

struct ProcessCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "process",
        abstract: "Process a video file into a .botmap spatial map."
    )

    @Argument(help: "Path to input video file (.mov or .mp4)")
    var input: String

    @Option(name: .shortAndLong, help: "Output .botmap file path")
    var output: String?

    @Flag(name: .shortAndLong, help: "Process as fast as possible")
    var fast: Bool = false

    @Flag(name: .shortAndLong, help: "Verbose progress output")
    var verbose: Bool = false

    @Option(help: "Maximum depth in metres (default: 8.0)")
    var maxDepth: Float = 8.0

    @Option(help: "Max CPU cores to use (default: half of available)")
    var maxCores: Int?

    @Option(help: "Pause when free RAM drops below N GB (default: 1.0)")
    var minFreeRamGb: Double = 1.0

    @Flag(name: .long, help: "Terminal-only, no monitor window")
    var noGui: Bool = false

    mutating func run() async throws {
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw CLIError.fileNotFound(inputURL.path)
        }
        let outputURL: URL = output.map { URL(fileURLWithPath: $0).standardizedFileURL }
            ?? inputURL.deletingPathExtension().appendingPathExtension("botmap")

        guard let device = MTLCreateSystemDefaultDevice() else { throw CLIError.noMetalDevice }

        let availCores = ProcessInfo.processInfo.activeProcessorCount
        let cores = max(1, min(maxCores ?? max(1, availCores / 2), availCores))
        let step  = max(8, 64 / cores)

        let world         = BotMapWorld(name: inputURL.deletingPathExtension().lastPathComponent)
        let capture       = VideoCaptureSession(url: inputURL, rate: .fast)
        let poseEstimator = OpticalFlowPoseEstimator()
        let inserter      = try VoxelInserter(device: device)
        inserter.maxDepth    = maxDepth
        inserter.samplingStep = step

        let state = ProcessingState()
        state.inputName  = inputURL.lastPathComponent
        state.outputPath = outputURL.path
        state.gpuName    = device.name
        let memLimit = minFreeRamGb

        print("VoxelKit — \(inputURL.lastPathComponent)")
        print("GPU: \(device.name) | cores: \(cores)/\(availCores) | pixel step: \(step)")

        await capture.setOnFrame { pixelBuffer, _, _, _ in
            if SystemMonitor.freeMemoryGB() < memLimit {
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            do {
                let pose = await poseEstimator.process(pixelBuffer: pixelBuffer)
                try await inserter.processFrame(
                    pixelBuffer: pixelBuffer, pose: pose,
                    intrinsics: .iPhone14Default, world: world)
            } catch { }
        }

        if noGui {
            try await terminalMode(capture: capture, world: world, outputURL: outputURL)
        } else {
            try await guiMode(capture: capture, world: world, state: state, outputURL: outputURL)
        }
    }

    // MARK: - Terminal mode

    private func terminalMode(
        capture: VideoCaptureSession,
        world: BotMapWorld,
        outputURL: URL
    ) async throws {
        let bar = TerminalProgress(barWidth: 24)
        let t0  = Date.now
        async let task: Void = capture.start()
        for await prog in await capture.progress {
            let v = await world.voxelCount
            bar.update(fraction: prog.fraction, voxelCount: v, fps: prog.fps,
                       etaSeconds: prog.estimatedSecondsRemaining)
        }
        try await task
        let v = await world.voxelCount
        bar.finish(message: "✓ \(String(format: "%.1f", Date.now.timeIntervalSince(t0)))s — \(v) voxels")
        try await saveAndPrint(world: world, outputURL: outputURL)
    }

    // MARK: - GUI mode
    //
    // Key insight: ArgumentParser's @main already runs RunLoop.main.run() to keep the process
    // alive. We must NOT call NSApp.run() — that creates RunLoop reentrancy from inside a
    // DispatchQueue.main.async callback, which blocks Timer.publish and @State updates.
    // Instead: create the window on MainActor and run processing inline in this task.
    // NSApp retains the window via NSApp.windows[]. ArgumentParser's RunLoop handles events.

    private func guiMode(
        capture: VideoCaptureSession,
        world: BotMapWorld,
        state: ProcessingState,
        outputURL: URL
    ) async throws {

        // System monitor
        let sysMonitor = SystemMonitor(interval: 1.0)
        sysMonitor.onUpdate = { cores, avg, usedGB, totalGB, gpu in
            DispatchQueue.main.async {
                state.cpuCoresLoad   = cores
                state.cpuTotalLoad   = avg
                state.memoryUsedGB   = usedGB
                state.memoryTotalGB  = totalGB
                state.gpuUtilization = gpu
            }
        }

        // Create and show window on main thread.
        // No app.run() — ArgumentParser's RunLoop keeps us alive.
        // NSApp.windows[] retains the NSWindow automatically.
        await MainActor.run {
            let app = NSApplication.shared
            app.setActivationPolicy(.regular)

            let w = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 820, height: 500),
                styleMask:   [.titled, .closable, .resizable, .miniaturizable],
                backing:     .buffered,
                defer:       false
            )
            w.title = "VoxelKit Monitor — \(state.inputName)"
            w.contentViewController = NSHostingController(rootView: MonitorView(state: state))
            w.center()
            w.makeKeyAndOrderFront(nil)
            app.activate(ignoringOtherApps: true)
        }

        sysMonitor.start()

        // Preview sampler: first sample immediately, then every 2 s
        let samplerTask = Task.detached(priority: .utility) {
            while !Task.isCancelled {
                let pts = await world.samplePositions(maxCount: 4000)
                DispatchQueue.main.async { state.previewPositions = pts }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }

        // Processing — runs inline (keeps this task, and thus the process, alive)
        let bar = TerminalProgress(barWidth: 24)
        let t0  = Date.now

        async let captureTask: Void = capture.start()
        for await prog in await capture.progress {
            let v = await world.voxelCount
            DispatchQueue.main.async {
                state.voxelCount = v
                state.fps        = prog.fps
                state.progress   = prog.fraction ?? 0
                state.etaSeconds = prog.estimatedSecondsRemaining
            }
            bar.update(fraction: prog.fraction, voxelCount: v, fps: prog.fps,
                       etaSeconds: prog.estimatedSecondsRemaining)
        }
        try await captureTask

        samplerTask.cancel()
        sysMonitor.stop()

        let v   = await world.voxelCount
        let pts = await world.samplePositions(maxCount: 6000)
        bar.finish(message: "✓ \(String(format: "%.1f", Date.now.timeIntervalSince(t0)))s — \(v) voxels")
        try await saveAndPrint(world: world, outputURL: outputURL)

        DispatchQueue.main.async {
            state.previewPositions = pts
            state.voxelCount       = v
            state.isFinished       = true
        }

        // Keep window visible for 5 s after completion
        try await Task.sleep(nanoseconds: 5_000_000_000)
    }

    // MARK: - Helpers

    private func saveAndPrint(world: BotMapWorld, outputURL: URL) async throws {
        print("Saving \(outputURL.lastPathComponent)…", terminator: "")
        try await world.save(to: outputURL)
        let size = (try? outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        print(" \(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))")
        print("Done. Run: voxelcli info \(outputURL.path)")
    }
}

enum CLIError: Error, LocalizedError {
    case fileNotFound(String)
    case noMetalDevice

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let p): return "File not found: \(p)"
        case .noMetalDevice:       return "No Metal device available"
        }
    }
}
