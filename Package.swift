// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VoxelKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VoxelKit", targets: ["VoxelKit"]),
        .library(name: "VoxelKitCompute", targets: ["VoxelKitCompute"]),
        .library(name: "VoxelKitUI", targets: ["VoxelKitUI"]),
        .executable(name: "voxelcli", targets: ["VoxelCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "VoxelKit",
            path: "Sources/VoxelKit"
        ),
        .target(
            name: "VoxelKitCompute",
            dependencies: ["VoxelKit"],
            path: "Sources/VoxelKitCompute",
            resources: [.process("Metal")]
        ),
        .target(
            name: "VoxelKitUI",
            dependencies: ["VoxelKit", "VoxelKitCompute"],
            path: "Sources/VoxelKitUI"
        ),
        .executableTarget(
            name: "VoxelCLI",
            dependencies: [
                "VoxelKit",
                "VoxelKitCompute",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/VoxelCLI"
        ),
        .testTarget(
            name: "VoxelKitTests",
            dependencies: ["VoxelKit", "VoxelKitCompute"],
            path: "Tests/VoxelKitTests"
        ),
    ]
)
