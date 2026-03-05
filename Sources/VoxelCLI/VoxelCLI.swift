import Foundation
import ArgumentParser

@main
struct VoxelCLI: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "voxelcli",
        abstract: "VoxelKit command-line tool — build 3D spatial maps from video and LiDAR.",
        version: "26.3.5.1",
        subcommands: [
            ProcessCommand.self,
            InfoCommand.self,
        ]
    )
}
