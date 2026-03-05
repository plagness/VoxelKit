import Foundation
import ArgumentParser
import VoxelKit

/// Display metadata from a `.botmap` file.
struct InfoCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Display metadata from a .botmap file."
    )

    @Argument(help: "Path to .botmap file")
    var input: String

    mutating func run() async throws {
        let url = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError.fileNotFound(url.path)
        }

        let world = try await BotMapWorld.load(from: url)

        let name = await world.name
        let voxelCount = await world.voxelCount
        let chunkCount = await world.chunkCount
        let bb = await world.boundingBox
        let createdAt = await world.createdAt

        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let sizeStr = ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)

        print("File:       \(url.lastPathComponent) (\(sizeStr))")
        print("Name:       \(name)")
        print("Created:    \(ISO8601DateFormatter().string(from: createdAt))")
        print("Voxels:     \(voxelCount)")
        print("Chunks:     \(chunkCount)")
        let dims = bb.size
        print("Dimensions: \(String(format: "%.1f × %.1f × %.1f m", dims.x, dims.y, dims.z))")
    }
}
