import SceneKit
import SwiftUI
import simd

/// Interactive 3D point cloud viewer using SceneKit.
/// Supports mouse orbit, zoom, and pan via `allowsCameraControl`.
/// Points are colored by Y height: blue (floor) → green → red (ceiling).
struct VoxelScene3DView: NSViewRepresentable {

    let positions: [SIMD3<Float>]

    // MARK: - NSViewRepresentable

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero)
        view.allowsCameraControl = true
        view.backgroundColor = NSColor(white: 0.05, alpha: 1)
        view.antialiasingMode = .multisampling4X
        view.autoenablesDefaultLighting = false

        let scene = SCNScene()

        // Ambient light
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light!.type = .ambient
        ambient.light!.color = NSColor(white: 1.0, alpha: 1)
        ambient.light!.intensity = 1000
        scene.rootNode.addChildNode(ambient)

        // Initial camera
        let camNode = SCNNode()
        camNode.name = "defaultCamera"
        camNode.camera = SCNCamera()
        camNode.camera!.zNear = 0.01
        camNode.camera!.zFar = 200
        camNode.position = SCNVector3(0, 4, 12)
        camNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(camNode)

        view.scene = scene
        view.pointOfView = camNode
        return view
    }

    func updateNSView(_ scnView: SCNView, context: Context) {
        // Only rebuild geometry when position count changes (sampler fires every 2s)
        guard positions.count != context.coordinator.lastCount else { return }
        context.coordinator.lastCount = positions.count

        scnView.scene?.rootNode.childNode(withName: "pointCloud", recursively: false)?
            .removeFromParentNode()

        guard !positions.isEmpty else { return }

        let node = buildPointCloud(positions: positions)
        scnView.scene?.rootNode.addChildNode(node)

        // First time: frame the cloud
        if !context.coordinator.hasCentered {
            context.coordinator.hasCentered = true
            centerCamera(scnView, on: positions)
        }
    }

    // MARK: - Coordinator

    final class Coordinator {
        var lastCount: Int = -1
        var hasCentered: Bool = false
    }

    // MARK: - Point cloud geometry

    private func buildPointCloud(positions: [SIMD3<Float>]) -> SCNNode {
        let count = positions.count

        // Compute Y range for color mapping
        var yMin = positions[0].y, yMax = positions[0].y
        for p in positions { yMin = min(yMin, p.y); yMax = max(yMax, p.y) }
        let yRange = max(0.001, yMax - yMin)

        // Vertex buffer
        var verts = [SCNVector3](repeating: .init(0, 0, 0), count: count)
        // Color buffer: 3×Float32 per vertex (RGB)
        var colors = [Float](repeating: 0, count: count * 3)

        for i in 0..<count {
            let p = positions[i]
            verts[i] = SCNVector3(p.x, p.y, p.z)

            // Height → hue: 0.66 (blue) at bottom → 0.0 (red) at top
            let t = (p.y - yMin) / yRange
            let hue = (1.0 - t) * 0.66
            let (r, g, b) = hsvToRgb(h: hue, s: 0.9, v: 0.95)
            colors[i * 3]     = r
            colors[i * 3 + 1] = g
            colors[i * 3 + 2] = b
        }

        let vertexSource = SCNGeometrySource(vertices: verts)
        let colorSource  = SCNGeometrySource(
            data:                 Data(bytes: colors, count: colors.count * 4),
            semantic:             .color,
            vectorCount:          count,
            usesFloatComponents:  true,
            componentsPerVector:  3,
            bytesPerComponent:    4,
            dataOffset:           0,
            dataStride:           12
        )

        var indices = (0..<count).map { Int32($0) }
        let element = SCNGeometryElement(
            data:            Data(bytes: &indices, count: count * 4),
            primitiveType:   .point,
            primitiveCount:  count,
            bytesPerIndex:   4
        )
        element.pointSize = 4.0
        element.minimumPointScreenSpaceRadius = 1.0
        element.maximumPointScreenSpaceRadius = 8.0

        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant  // no shading on points
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "pointCloud"
        return node
    }

    // MARK: - Camera framing

    private func centerCamera(_ scnView: SCNView, on positions: [SIMD3<Float>]) {
        var minX = positions[0].x, maxX = positions[0].x
        var minY = positions[0].y, maxY = positions[0].y
        var minZ = positions[0].z, maxZ = positions[0].z
        for p in positions {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
            minZ = min(minZ, p.z); maxZ = max(maxZ, p.z)
        }
        let cx = (minX + maxX) / 2
        let cy = (minY + maxY) / 2
        let cz = (minZ + maxZ) / 2
        let size = max(maxX - minX, maxY - minY, maxZ - minZ)
        let dist = size * 1.5 + 2

        scnView.pointOfView?.position = SCNVector3(cx, cy + size * 0.3, cz + dist)
        scnView.pointOfView?.look(at: SCNVector3(cx, cy, cz))
    }

    // MARK: - HSV → RGB

    private func hsvToRgb(h: Float, s: Float, v: Float) -> (Float, Float, Float) {
        let i = Int(h * 6)
        let f = h * 6 - Float(i)
        let p = v * (1 - s)
        let q = v * (1 - f * s)
        let t = v * (1 - (1 - f) * s)
        switch i % 6 {
        case 0: return (v, t, p)
        case 1: return (q, v, p)
        case 2: return (p, v, t)
        case 3: return (p, q, v)
        case 4: return (t, p, v)
        default: return (v, p, q)
        }
    }
}
