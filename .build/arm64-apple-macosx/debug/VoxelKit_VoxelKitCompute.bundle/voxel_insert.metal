#include <metal_stdlib>
using namespace metal;

/// Camera intrinsics passed as uniform.
struct CameraIntrinsics {
    float fx, fy;
    float cx, cy;
    float depthScale;   // world metres per depth unit
    float minDepth;     // ignore below this (metres)
    float maxDepth;     // ignore above this (metres)
    float _pad;
};

/// Camera pose: position + rotation matrix (column-major).
struct CameraPose {
    float3x3 rotation;  // world = pose.rotation * cameraPoint + position
    float3   position;
    float    _pad;
};

/// Back-project depth map pixels to world-space positions.
///
/// Each thread handles one pixel. If depth is valid, writes a world-space
/// SIMD3<Float> position to the output buffer. Invalid pixels write NaN.
///
/// - texture(0): depth map (R32Float, values in depth units)
/// - buffer(0): CameraIntrinsics
/// - buffer(1): CameraPose
/// - buffer(2): output world positions (float3 array)
/// - buffer(3): output valid count (atomic_uint, one per threadgroup)
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

    // Back-project to camera space
    float u = float(gid.x);
    float v = float(gid.y);
    float3 camPoint = float3(
        (u - intrinsics.cx) * depth / intrinsics.fx,
        (v - intrinsics.cy) * depth / intrinsics.fy,
        depth
    );

    // Transform to world space: worldPoint = R * camPoint + t
    float3 worldPoint = pose.rotation * camPoint + pose.position;
    outPositions[linearIdx] = worldPoint;
}
