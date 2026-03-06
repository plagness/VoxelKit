import Testing
import Foundation
import simd
import Metal
@testable import VoxelKit
import VoxelKitCompute

/// Integration tests using the real IMG_7331.MOV file.
/// These are the primary end-to-end validation tests.
///
/// Path: /Users/plag/robodog/IMG_7331.MOV
@Suite("Integration Tests")
struct IntegrationTests {

    static let videoPath = "/Users/plag/robodog/IMG_7331.MOV"

    // MARK: - Pipeline integration

    @Test("Process first 100 frames of IMG_7331.MOV — >500 voxels expected")
    func processFirst100Frames() async throws {
        let videoURL = URL(fileURLWithPath: Self.videoPath)
        guard FileManager.default.fileExists(atPath: videoURL.path) else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let world = BotMapWorld(name: "integration-test")
        let capture = VideoCaptureSession(url: videoURL, rate: .fast)
        let poseEstimator = OpticalFlowPoseEstimator()
        let inserter = try VoxelInserter(device: device)
        inserter.samplingStep = 16  // 1/256 pixels for speed in tests

        await capture.setOnFrame { pixelBuffer, _, frameIdx, _, _ in
            guard frameIdx <= 100 else { return }
            let pose = await poseEstimator.process(pixelBuffer: pixelBuffer)
            try? await inserter.processFrame(
                pixelBuffer: pixelBuffer,
                pose: pose,
                intrinsics: .iPhone14Default,
                world: world
            )
        }

        try await capture.start()

        let count = await world.voxelCount
        #expect(count > 500, "Expected >500 voxels after 100 frames, got \(count)")
    }

    @Test("Process 50 frames and save to .botmap — file is created and loadable")
    func processAndSave() async throws {
        let videoURL = URL(fileURLWithPath: Self.videoPath)
        guard FileManager.default.fileExists(atPath: videoURL.path) else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let world = BotMapWorld(name: "save-test")
        let capture = VideoCaptureSession(url: videoURL, rate: .fast)
        let poseEstimator = OpticalFlowPoseEstimator()
        let inserter = try VoxelInserter(device: device)
        inserter.samplingStep = 32  // very coarse for speed

        await capture.setOnFrame { pixelBuffer, _, frameIdx, _, _ in
            guard frameIdx <= 50 else { return }
            let pose = await poseEstimator.process(pixelBuffer: pixelBuffer)
            try? await inserter.processFrame(
                pixelBuffer: pixelBuffer,
                pose: pose,
                intrinsics: .iPhone14Default,
                world: world
            )
        }

        try await capture.start()

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("integration-\(UUID().uuidString).botmap")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await world.save(to: tempURL)

        let loaded = try await BotMapWorld.load(from: tempURL)
        let originalCount = await world.voxelCount
        let loadedCount   = await loaded.voxelCount

        #expect(FileManager.default.fileExists(atPath: tempURL.path))
        #expect(loadedCount == originalCount)
        #expect(loadedCount > 0)
    }

    // MARK: - CaptureProgress integration

    @Test("VideoCaptureSession emits progress updates")
    func progressUpdates() async throws {
        let videoURL = URL(fileURLWithPath: Self.videoPath)
        guard FileManager.default.fileExists(atPath: videoURL.path) else { return }

        let capture = VideoCaptureSession(url: videoURL, rate: .fast)

        await capture.setOnFrame { _, _, _, _, _ in
            // just read frames for progress test, no processing
        }

        var progressCount = 0
        async let captureTask: Void = capture.start()

        for await _ in await capture.progress {
            progressCount += 1
            if progressCount >= 3 { await capture.cancel(); break }
        }

        try? await captureTask

        #expect(progressCount >= 1)
    }
}
