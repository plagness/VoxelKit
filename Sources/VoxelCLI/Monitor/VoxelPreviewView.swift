import SwiftUI
import simd

/// Top-down (XZ) projection of occupied voxels, colored by height (Y).
struct VoxelPreviewView: View {

    let positions: [SIMD3<Float>]
    var dotSize: CGFloat = 3

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                guard !positions.isEmpty else {
                    // "Waiting for data…" placeholder
                    let text = Text("Waiting for voxels…")
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(Color.gray)
                    ctx.draw(text, at: CGPoint(x: size.width / 2, y: size.height / 2),
                             anchor: .center)
                    return
                }

                // Compute XZ bounds
                let xs = positions.map { $0.x }
                let zs = positions.map { $0.z }
                let ys = positions.map { $0.y }
                let xMin = xs.min()!, xMax = xs.max()!
                let zMin = zs.min()!, zMax = zs.max()!
                let yMin = ys.min()!, yMax = ys.max()!
                let yRange = yMax - yMin

                let padding: CGFloat = 16
                let drawW = size.width  - padding * 2
                let drawH = size.height - padding * 2
                let xRange = CGFloat(xMax - xMin)
                let zRange = CGFloat(zMax - zMin)
                let scale = xRange > 0 && zRange > 0
                    ? min(drawW / xRange, drawH / zRange)
                    : 1

                // Center the projection
                let offsetX = padding + (drawW - xRange * scale) / 2
                let offsetZ = padding + (drawH - zRange * scale) / 2

                for pos in positions {
                    let px = offsetX + CGFloat(pos.x - xMin) * scale
                    let pz = offsetZ + CGFloat(pos.z - zMin) * scale

                    // Color: cold (blue) at floor → warm (red/orange) at ceiling
                    let t = yRange > 0 ? Double((pos.y - yMin) / yRange) : 0.5
                    let color = heightColor(t: t)

                    let rect = CGRect(x: px - dotSize / 2, y: pz - dotSize / 2,
                                      width: dotSize, height: dotSize)
                    ctx.fill(Path(ellipseIn: rect), with: .color(color))
                }
            }
        }
    }

    /// Map t ∈ [0,1] to a perceptually-smooth hue: blue→cyan→green→yellow→red
    private func heightColor(t: Double) -> Color {
        let hue = (1 - t) * 0.66   // 0.66 = blue, 0 = red
        return Color(hue: hue, saturation: 0.85, brightness: 0.9)
    }
}
