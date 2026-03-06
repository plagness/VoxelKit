import Metal

/// Inline Metal shader source for SPM compatibility.
/// SPM doesn't compile .metal files in all contexts (swift build, swift test).
/// Inlining the source and using `makeLibrary(source:)` works universally.
enum MetalShaders {

    static let source: String = """
#include <metal_stdlib>
using namespace metal;

// MARK: - YUV → RGB conversion

kernel void yuv_to_rgb(
    texture2d<float, access::read>  yTexture    [[ texture(0) ]],
    texture2d<float, access::read>  cbcrTexture [[ texture(1) ]],
    texture2d<float, access::write> rgbTexture  [[ texture(2) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint2 size = uint2(rgbTexture.get_width(), rgbTexture.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    float y    = yTexture.read(gid).r;
    float2 cbcr = cbcrTexture.read(gid / 2).rg;

    float cb = cbcr.x - 0.5;
    float cr = cbcr.y - 0.5;
    float yN = (y - 16.0/255.0) * (255.0/219.0);

    float r = clamp(yN + 1.402 * cr, 0.0, 1.0);
    float g = clamp(yN - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0);
    float b = clamp(yN + 1.772 * cb, 0.0, 1.0);

    rgbTexture.write(float4(r, g, b, 1.0), gid);
}

// MARK: - Voxel insertion (depth back-projection)

struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
    float depthScale;
    float minDepth;
    float maxDepth;
    float _pad;
};

struct CameraPose {
    float3x3 rotation;
    float3   position;
    float    _pad;
};

kernel void voxel_insert(
    texture2d<float, access::read> depthTexture [[ texture(0) ]],
    constant CameraIntrinsics& intrinsics       [[ buffer(0) ]],
    constant CameraPose& pose                   [[ buffer(1) ]],
    device float3* outPositions                 [[ buffer(2) ]],
    uint2 gid                                   [[ thread_position_in_grid ]],
    uint2 gridSize                              [[ threads_per_grid ]]
) {
    uint2 size = uint2(depthTexture.get_width(), depthTexture.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    float depthRaw = depthTexture.read(gid).r;
    float depth = depthRaw * intrinsics.depthScale;

    uint linearIdx = gid.y * size.x + gid.x;

    if (depth < intrinsics.minDepth || depth > intrinsics.maxDepth || isnan(depth) || isinf(depth)) {
        outPositions[linearIdx] = float3(NAN, NAN, NAN);
        return;
    }

    float u = float(gid.x);
    float v = float(gid.y);
    float3 camPoint = float3(
        (u - intrinsics.cx) * depth / intrinsics.fx,
        (v - intrinsics.cy) * depth / intrinsics.fy,
        depth
    );

    float3 worldPoint = pose.rotation * camPoint + pose.position;
    outPositions[linearIdx] = worldPoint;
}

// MARK: - Voxel insertion with color sampling

struct ColoredInsertParams {
    uint colorWidth;
    uint colorHeight;
    uint depthWidth;
    uint depthHeight;
};

kernel void voxel_insert_colored(
    texture2d<float, access::read> depthTexture  [[ texture(0) ]],
    texture2d<float, access::read> colorTexture  [[ texture(1) ]],
    constant CameraIntrinsics& intrinsics        [[ buffer(0) ]],
    constant CameraPose& pose                    [[ buffer(1) ]],
    device float3* outPositions                  [[ buffer(2) ]],
    device uchar4* outColors                     [[ buffer(3) ]],
    constant ColoredInsertParams& params         [[ buffer(4) ]],
    uint2 gid                                    [[ thread_position_in_grid ]]
) {
    uint2 depthSize = uint2(depthTexture.get_width(), depthTexture.get_height());
    if (gid.x >= depthSize.x || gid.y >= depthSize.y) return;

    float depthRaw = depthTexture.read(gid).r;
    float depth = depthRaw * intrinsics.depthScale;

    uint linearIdx = gid.y * depthSize.x + gid.x;

    if (depth < intrinsics.minDepth || depth > intrinsics.maxDepth || isnan(depth) || isinf(depth)) {
        outPositions[linearIdx] = float3(NAN, NAN, NAN);
        outColors[linearIdx] = uchar4(0, 0, 0, 0);
        return;
    }

    float u = float(gid.x);
    float v = float(gid.y);
    float3 camPoint = float3(
        (u - intrinsics.cx) * depth / intrinsics.fx,
        (v - intrinsics.cy) * depth / intrinsics.fy,
        depth
    );

    float3 worldPoint = pose.rotation * camPoint + pose.position;
    outPositions[linearIdx] = worldPoint;

    // Sample color from camera image (depth and camera have different resolutions)
    float2 colorUV = float2(gid) * float2(float(params.colorWidth) / float(params.depthWidth),
                                           float(params.colorHeight) / float(params.depthHeight));
    uint2 colorCoord = uint2(clamp(colorUV, float2(0), float2(float(params.colorWidth - 1),
                                                                float(params.colorHeight - 1))));
    float4 color = colorTexture.read(colorCoord);
    outColors[linearIdx] = uchar4(uchar(color.r * 255.0), uchar(color.g * 255.0),
                                   uchar(color.b * 255.0), 255);
}
"""

    /// Create a MTLLibrary from inline source (works in CLI, tests, and apps).
    static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        return try device.makeLibrary(source: source, options: options)
    }
}
