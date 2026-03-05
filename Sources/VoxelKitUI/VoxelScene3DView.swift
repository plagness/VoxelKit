import SceneKit
import SwiftUI
import simd

/// Interactive 3D point cloud viewer using SceneKit.
/// Custom orbit/pan/zoom via OrbitSCNView (SwiftUI intercepts NSGestureRecognizers,
/// so we override mouseDown/mouseDragged/scrollWheel directly instead of allowsCameraControl).
/// Points are colored by Y height: blue (floor) → green → red (ceiling).
public struct VoxelScene3DView: NSViewRepresentable {

    public let positions: [SIMD3<Float>]

    public init(positions: [SIMD3<Float>]) {
        self.positions = positions
    }

    // MARK: - NSViewRepresentable

    public func makeCoordinator() -> Coordinator { Coordinator() }

    public func makeNSView(context: Context) -> OrbitSCNView {
        let view = OrbitSCNView(frame: .zero)
        view.allowsCameraControl = false
        view.backgroundColor = NSColor(white: 0.05, alpha: 1)
        view.antialiasingMode = .multisampling4X
        view.autoenablesDefaultLighting = false

        let scene = SCNScene()

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light!.type = .ambient
        ambient.light!.color = NSColor(white: 1.0, alpha: 1)
        ambient.light!.intensity = 1000
        scene.rootNode.addChildNode(ambient)

        let camNode = SCNNode()
        camNode.name = "cam"
        camNode.camera = SCNCamera()
        camNode.camera!.zNear = 0.01
        camNode.camera!.zFar = 500
        scene.rootNode.addChildNode(camNode)

        view.scene = scene
        view.pointOfView = camNode
        view.updateCamera()
        return view
    }

    public func updateNSView(_ scnView: OrbitSCNView, context: Context) {
        guard positions.count != context.coordinator.lastCount else { return }
        context.coordinator.lastCount = positions.count

        scnView.scene?.rootNode.childNode(withName: "pointCloud", recursively: false)?
            .removeFromParentNode()

        guard !positions.isEmpty else { return }

        let node = buildPointCloud(positions: positions)
        scnView.scene?.rootNode.addChildNode(node)

        if !context.coordinator.hasCentered {
            context.coordinator.hasCentered = true
            centerCamera(scnView, on: positions)
        }
    }

    // MARK: - Coordinator

    public final class Coordinator {
        public var lastCount: Int = -1
        public var hasCentered: Bool = false
        public init() {}
    }

    // MARK: - Point cloud geometry

    private func buildPointCloud(positions: [SIMD3<Float>]) -> SCNNode {
        let count = positions.count

        var yMin = positions[0].y, yMax = positions[0].y
        for p in positions { yMin = min(yMin, p.y); yMax = max(yMax, p.y) }
        let yRange = max(0.001, yMax - yMin)

        var verts  = [SCNVector3](repeating: .init(0, 0, 0), count: count)
        var colors = [Float](repeating: 0, count: count * 3)

        for i in 0..<count {
            let p = positions[i]
            verts[i] = SCNVector3(p.x, p.y, p.z)
            let t   = (p.y - yMin) / yRange
            let hue = (1.0 - t) * 0.66
            let (r, g, b) = hsvToRgb(h: hue, s: 0.9, v: 0.95)
            colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b
        }

        let vertexSource = SCNGeometrySource(vertices: verts)
        let colorSource  = SCNGeometrySource(
            data:                Data(bytes: colors, count: colors.count * 4),
            semantic:            .color,
            vectorCount:         count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent:   4,
            dataOffset:          0,
            dataStride:          12
        )

        var indices = (0..<count).map { Int32($0) }
        let element = SCNGeometryElement(
            data:           Data(bytes: &indices, count: count * 4),
            primitiveType:  .point,
            primitiveCount: count,
            bytesPerIndex:  4
        )
        element.pointSize = 4.0
        element.minimumPointScreenSpaceRadius = 1.0
        element.maximumPointScreenSpaceRadius = 8.0

        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "pointCloud"
        return node
    }

    // MARK: - Camera framing

    private func centerCamera(_ scnView: OrbitSCNView, on positions: [SIMD3<Float>]) {
        var minX = positions[0].x, maxX = positions[0].x
        var minY = positions[0].y, maxY = positions[0].y
        var minZ = positions[0].z, maxZ = positions[0].z
        for p in positions {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
            minZ = min(minZ, p.z); maxZ = max(maxZ, p.z)
        }
        let cx   = (minX + maxX) / 2
        let cy   = (minY + maxY) / 2
        let cz   = (minZ + maxZ) / 2
        let size = max(maxX - minX, maxY - minY, maxZ - minZ)

        scnView.pivot          = SIMD3<Float>(cx, cy, cz)
        scnView.orbitRadius    = size * 1.5 + 2
        scnView.orbitElevation = 0.3
        scnView.orbitAzimuth   = 0
        scnView.updateCamera()
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

// MARK: - Custom SCNView: orbit / pan / zoom

public final class OrbitSCNView: SCNView {

    public var orbitAzimuth:   Float = 0
    public var orbitElevation: Float = 0.3
    public var orbitRadius:    Float = 15
    public var pivot:          SIMD3<Float> = .zero

    private var lastDrag: CGPoint = .zero

    override public var acceptsFirstResponder: Bool { true }

    override public func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    public func updateCamera() {
        let cosEl = cos(orbitElevation)
        let sinEl = sin(orbitElevation)
        let x = pivot.x + orbitRadius * cosEl * sin(orbitAzimuth)
        let y = pivot.y + orbitRadius * sinEl
        let z = pivot.z + orbitRadius * cosEl * cos(orbitAzimuth)
        pointOfView?.position = SCNVector3(x, y, z)
        pointOfView?.look(at: SCNVector3(pivot.x, pivot.y, pivot.z))
    }

    // MARK: Left-drag → orbit

    override public func mouseDown(with event: NSEvent) {
        lastDrag = convert(event.locationInWindow, from: nil)
        window?.makeFirstResponder(self)
    }

    override public func mouseDragged(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        let dx  = Float(loc.x - lastDrag.x)
        let dy  = Float(loc.y - lastDrag.y)
        lastDrag = loc

        let sens: Float = 0.008
        orbitAzimuth   -= dx * sens
        orbitElevation  = max(-.pi / 2 + 0.01,
                              min(.pi / 2 - 0.01, orbitElevation + dy * sens))
        updateCamera()
    }

    // MARK: Right-drag → pan

    override public func rightMouseDown(with event: NSEvent) {
        lastDrag = convert(event.locationInWindow, from: nil)
    }

    override public func rightMouseDragged(with event: NSEvent) {
        let loc  = convert(event.locationInWindow, from: nil)
        let dx   = Float(loc.x - lastDrag.x)
        let dy   = Float(loc.y - lastDrag.y)
        lastDrag = loc

        let sens  = orbitRadius * 0.0015
        let az    = orbitAzimuth
        let el    = orbitElevation
        let right = SIMD3<Float>( cos(az),           0,           -sin(az))
        let up    = SIMD3<Float>(-sin(el) * sin(az), cos(el), -sin(el) * cos(az))
        pivot    -= right * (dx * sens)
        pivot    += up    * (dy * sens)
        updateCamera()
    }

    // MARK: Scroll → zoom

    override public func scrollWheel(with event: NSEvent) {
        let factor: Float = event.hasPreciseScrollingDeltas ? 0.02 : 0.5
        orbitRadius = max(0.5, orbitRadius - Float(event.scrollingDeltaY) * factor)
        updateCamera()
    }
}
