#if os(macOS)
import SceneKit
import SwiftUI
import simd
import VoxelKit

// MARK: - VoxelSceneController

/// Controller that exposes snapshot capability for the underlying OrbitSCNView.
@MainActor
public final class VoxelSceneController: ObservableObject {
    weak var scnView: OrbitSCNView?
    public init() {}

    public func snapshot() -> NSImage? { scnView?.snapshot() }
    public var scene: SCNScene? { scnView?.scene }

    @discardableResult
    public func saveSnapshotToDesktop(prefix: String = "VoxelUI") -> URL? {
        guard let img = snapshot() else { return nil }
        guard let tiff = img.tiffRepresentation,
              let bmp  = NSBitmapImageRep(data: tiff),
              let png  = bmp.representation(using: .png, properties: [:]) else { return nil }
        let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
        let dest = desktop.appendingPathComponent("\(prefix)-\(Int(Date().timeIntervalSince1970)).png")
        try? png.write(to: dest)
        return dest
    }
}

// MARK: - VoxelScene3DView

/// Interactive 3D voxel viewer using SceneKit.
/// Custom orbit/pan/zoom via OrbitSCNView (SwiftUI intercepts NSGestureRecognizers,
/// so we override mouseDown/mouseDragged/scrollWheel directly instead of allowsCameraControl).
/// Supports both raw colored positions (legacy) and greedy-merged voxels (preferred).
public struct VoxelScene3DView: NSViewRepresentable {

    public let positions: [SIMD3<Float>]
    public let colors: [(UInt8, UInt8, UInt8)]?
    public let mergedVoxels: [MergedVoxel]?
    public var controller: VoxelSceneController?

    public init(positions: [SIMD3<Float>], controller: VoxelSceneController? = nil) {
        self.positions  = positions
        self.colors     = nil
        self.mergedVoxels = nil
        self.controller = controller
    }

    public init(coloredPositions: [(SIMD3<Float>, (UInt8, UInt8, UInt8))],
                controller: VoxelSceneController? = nil) {
        self.positions  = coloredPositions.map(\.0)
        self.colors     = coloredPositions.map(\.1)
        self.mergedVoxels = nil
        self.controller = controller
    }

    public init(mergedVoxels: [MergedVoxel], controller: VoxelSceneController? = nil) {
        self.mergedVoxels = mergedVoxels
        self.positions = mergedVoxels.map { SIMD3<Float>($0.cx, $0.cy, $0.cz) }
        self.colors = nil
        self.controller = controller
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
        ambient.light!.color = NSColor(white: 0.4, alpha: 1)
        ambient.light!.intensity = 800
        scene.rootNode.addChildNode(ambient)

        let directional = SCNNode()
        directional.light = SCNLight()
        directional.light!.type = .directional
        directional.light!.color = NSColor(white: 1.0, alpha: 1)
        directional.light!.intensity = 800
        directional.eulerAngles = SCNVector3(-Float.pi / 4, Float.pi / 6, 0)
        scene.rootNode.addChildNode(directional)

        let camNode = SCNNode()
        camNode.name = "cam"
        camNode.camera = SCNCamera()
        camNode.camera!.zNear = 0.01
        camNode.camera!.zFar = 500
        scene.rootNode.addChildNode(camNode)

        view.scene = scene
        view.pointOfView = camNode
        view.updateCamera()

        // Register with controller so caller can call snapshot()
        Task { @MainActor in controller?.scnView = view }

        return view
    }

    public func updateNSView(_ scnView: OrbitSCNView, context: Context) {
        let itemCount = mergedVoxels?.count ?? positions.count
        guard itemCount != context.coordinator.lastCount else { return }
        context.coordinator.lastCount = itemCount

        scnView.scene?.rootNode.childNode(withName: "voxelMesh", recursively: false)?
            .removeFromParentNode()

        guard itemCount > 0 else { return }

        let node: SCNNode
        if let merged = mergedVoxels, !merged.isEmpty {
            node = buildMergedMesh(merged)
        } else {
            node = buildPointCloud(positions: positions)
        }
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

    // MARK: - Voxel cube geometry

    /// Voxel size in meters. Matches octree leaf size (1.0/32 = 3.125cm).
    private static let voxelSize: Float = 0.03125

    private func buildPointCloud(positions: [SIMD3<Float>]) -> SCNNode {
        let count = positions.count
        let h = Self.voxelSize * 0.5 // half-size

        // 24 vertices per cube (4 per face × 6 faces, separate normals)
        // 12 triangles per cube (2 per face × 6 faces)
        let vertsPerCube = 24
        let trisPerCube = 12
        let vertCount = count * vertsPerCube
        let triCount = count * trisPerCube

        var verts = [Float](repeating: 0, count: vertCount * 3)
        var normals = [Float](repeating: 0, count: vertCount * 3)
        var colorData = [Float](repeating: 0, count: vertCount * 3)
        var indices = [Int32](repeating: 0, count: triCount * 3)

        // Per-face vertex offsets (dx, dy, dz) and normal, 4 verts per face
        // Face order: +Y, -Y, +X, -X, +Z, -Z
        let faceData: [(offsets: [(Float, Float, Float)], normal: (Float, Float, Float))] = [
            // +Y (top)
            (offsets: [(-h, h, -h), ( h, h, -h), ( h, h,  h), (-h, h,  h)],
             normal: (0, 1, 0)),
            // -Y (bottom)
            (offsets: [(-h, -h,  h), ( h, -h,  h), ( h, -h, -h), (-h, -h, -h)],
             normal: (0, -1, 0)),
            // +X (right)
            (offsets: [( h, -h, -h), ( h,  h, -h), ( h,  h,  h), ( h, -h,  h)],
             normal: (1, 0, 0)),
            // -X (left)
            (offsets: [(-h, -h,  h), (-h,  h,  h), (-h,  h, -h), (-h, -h, -h)],
             normal: (-1, 0, 0)),
            // +Z (front)
            (offsets: [(-h, -h,  h), ( h, -h,  h), ( h,  h,  h), (-h,  h,  h)],
             normal: (0, 0, 1)),
            // -Z (back)
            (offsets: [( h, -h, -h), (-h, -h, -h), (-h,  h, -h), ( h,  h, -h)],
             normal: (0, 0, -1)),
        ]

        for i in 0..<count {
            let p = positions[i]

            // Determine color
            let r, g, b: Float
            if let storedColors = colors, storedColors.count == count {
                let c = storedColors[i]
                r = Float(c.0) / 255.0; g = Float(c.1) / 255.0; b = Float(c.2) / 255.0
            } else {
                r = 0.5; g = 0.5; b = 0.5
            }

            let vBase = i * vertsPerCube
            let iBase = i * trisPerCube * 3

            for (faceIdx, face) in faceData.enumerated() {
                let fvBase = vBase + faceIdx * 4
                for (vi, off) in face.offsets.enumerated() {
                    let idx = (fvBase + vi) * 3
                    verts[idx]     = p.x + off.0
                    verts[idx + 1] = p.y + off.1
                    verts[idx + 2] = p.z + off.2
                    normals[idx]     = face.normal.0
                    normals[idx + 1] = face.normal.1
                    normals[idx + 2] = face.normal.2
                    colorData[idx]     = r
                    colorData[idx + 1] = g
                    colorData[idx + 2] = b
                }
                // Two triangles: (0,1,2) and (0,2,3)
                let fi = iBase + faceIdx * 6
                let fb = Int32(fvBase)
                indices[fi]     = fb
                indices[fi + 1] = fb + 1
                indices[fi + 2] = fb + 2
                indices[fi + 3] = fb
                indices[fi + 4] = fb + 2
                indices[fi + 5] = fb + 3
            }
        }

        let vertexSource = SCNGeometrySource(
            data: Data(bytes: verts, count: verts.count * 4),
            semantic: .vertex,
            vectorCount: vertCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: 4,
            dataOffset: 0,
            dataStride: 12
        )
        let normalSource = SCNGeometrySource(
            data: Data(bytes: normals, count: normals.count * 4),
            semantic: .normal,
            vectorCount: vertCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: 4,
            dataOffset: 0,
            dataStride: 12
        )
        let colorSource = SCNGeometrySource(
            data: Data(bytes: colorData, count: colorData.count * 4),
            semantic: .color,
            vectorCount: vertCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: 4,
            dataOffset: 0,
            dataStride: 12
        )

        let element = SCNGeometryElement(
            data: Data(bytes: indices, count: indices.count * 4),
            primitiveType: .triangles,
            primitiveCount: triCount,
            bytesPerIndex: 4
        )

        let geometry = SCNGeometry(sources: [vertexSource, normalSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .lambert
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "voxelMesh"
        return node
    }

    // MARK: - Merged voxel mesh (GreedyMesher output)

    private func buildMergedMesh(_ voxels: [MergedVoxel]) -> SCNNode {
        let count = voxels.count
        let vertsPerBox = 24
        let trisPerBox = 12
        let vertCount = count * vertsPerBox
        let triCount = count * trisPerBox

        var verts = [Float](repeating: 0, count: vertCount * 3)
        var normals = [Float](repeating: 0, count: vertCount * 3)
        var colorData = [Float](repeating: 0, count: vertCount * 3)
        var indices = [Int32](repeating: 0, count: triCount * 3)

        // Unit cube face templates: offsets from center as (sx*hx, sy*hy, sz*hz)
        // sign multipliers for each face's 4 vertices + normal direction
        typealias FaceTemplate = (verts: [(sx: Float, sy: Float, sz: Float)], nx: Float, ny: Float, nz: Float)
        let faces: [FaceTemplate] = [
            // +Y top
            (verts: [(-1, 1,-1), ( 1, 1,-1), ( 1, 1, 1), (-1, 1, 1)], nx: 0, ny: 1, nz: 0),
            // -Y bottom
            (verts: [(-1,-1, 1), ( 1,-1, 1), ( 1,-1,-1), (-1,-1,-1)], nx: 0, ny:-1, nz: 0),
            // +X right
            (verts: [( 1,-1,-1), ( 1, 1,-1), ( 1, 1, 1), ( 1,-1, 1)], nx: 1, ny: 0, nz: 0),
            // -X left
            (verts: [(-1,-1, 1), (-1, 1, 1), (-1, 1,-1), (-1,-1,-1)], nx:-1, ny: 0, nz: 0),
            // +Z front
            (verts: [(-1,-1, 1), ( 1,-1, 1), ( 1, 1, 1), (-1, 1, 1)], nx: 0, ny: 0, nz: 1),
            // -Z back
            (verts: [( 1,-1,-1), (-1,-1,-1), (-1, 1,-1), ( 1, 1,-1)], nx: 0, ny: 0, nz:-1),
        ]

        for i in 0..<count {
            let v = voxels[i]
            let cf = v.colorAndFlags
            let r = Float((cf >> 24) & 0xFF) / 255.0
            let g = Float((cf >> 16) & 0xFF) / 255.0
            let b = Float((cf >> 8)  & 0xFF) / 255.0

            let vBase = i * vertsPerBox
            let iBase = i * trisPerBox * 3

            for (fi, face) in faces.enumerated() {
                let fvBase = vBase + fi * 4
                for (vi, sv) in face.verts.enumerated() {
                    let idx = (fvBase + vi) * 3
                    verts[idx]     = v.cx + sv.sx * v.hx
                    verts[idx + 1] = v.cy + sv.sy * v.hy
                    verts[idx + 2] = v.cz + sv.sz * v.hz
                    normals[idx]     = face.nx
                    normals[idx + 1] = face.ny
                    normals[idx + 2] = face.nz
                    colorData[idx]     = r
                    colorData[idx + 1] = g
                    colorData[idx + 2] = b
                }
                let idxOff = iBase + fi * 6
                let fb = Int32(fvBase)
                indices[idxOff]     = fb
                indices[idxOff + 1] = fb + 1
                indices[idxOff + 2] = fb + 2
                indices[idxOff + 3] = fb
                indices[idxOff + 4] = fb + 2
                indices[idxOff + 5] = fb + 3
            }
        }

        let vertexSource = SCNGeometrySource(
            data: Data(bytes: verts, count: verts.count * 4),
            semantic: .vertex, vectorCount: vertCount,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: 12
        )
        let normalSource = SCNGeometrySource(
            data: Data(bytes: normals, count: normals.count * 4),
            semantic: .normal, vectorCount: vertCount,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: 12
        )
        let colorSource = SCNGeometrySource(
            data: Data(bytes: colorData, count: colorData.count * 4),
            semantic: .color, vectorCount: vertCount,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: 4, dataOffset: 0, dataStride: 12
        )
        let element = SCNGeometryElement(
            data: Data(bytes: indices, count: indices.count * 4),
            primitiveType: .triangles, primitiveCount: triCount, bytesPerIndex: 4
        )

        let geometry = SCNGeometry(sources: [vertexSource, normalSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .lambert
        material.isDoubleSided = true
        geometry.materials = [material]

        let node = SCNNode(geometry: geometry)
        node.name = "voxelMesh"
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

#endif
