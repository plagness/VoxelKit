#include <metal_stdlib>
using namespace metal;

/// Convert NV12 (YCbCr 4:2:0 biplanar) to RGBA8 texture.
/// Luma plane: texture2d<float> at binding 0
/// Chroma plane: texture2d<float> at binding 1
/// Output: texture2d<float, access::write> at binding 2
kernel void yuv_to_rgb(
    texture2d<float, access::read>  yTexture   [[ texture(0) ]],
    texture2d<float, access::read>  cbcrTexture [[ texture(1) ]],
    texture2d<float, access::write> rgbTexture  [[ texture(2) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint2 size = uint2(rgbTexture.get_width(), rgbTexture.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    float y    = yTexture.read(gid).r;
    float2 cbcr = cbcrTexture.read(gid / 2).rg;

    // BT.601 video range (16-235 luma, 16-240 chroma)
    float cb = cbcr.x - 0.5;
    float cr = cbcr.y - 0.5;
    float yN = (y - 16.0/255.0) * (255.0/219.0);  // normalize to [0,1]

    float r = clamp(yN + 1.402 * cr, 0.0, 1.0);
    float g = clamp(yN - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0);
    float b = clamp(yN + 1.772 * cb, 0.0, 1.0);

    rgbTexture.write(float4(r, g, b, 1.0), gid);
}
