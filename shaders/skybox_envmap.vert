#version 450
precision highp float;

// Fullscreen triangle vertex shader for the sky envmap background pass.
//
// The triangle covers the entire screen.  All three vertices are generated
// from gl_VertexIndex so no vertex buffer is needed.
//
// Depth is set to 1.0 (the far plane in standard Vulkan z=[0,1]).  Combined
// with a depth-test of LESS_OR_EQUAL and no depth write, this causes the sky
// to appear only on pixels where no geometry has been drawn (where the depth
// buffer still holds the cleared value of 1.0).

layout(location = 0) out vec2 v_UV;

void main()
{
    float x = -1.0 + float((gl_VertexIndex & 1) << 2);
    float y = -1.0 + float((gl_VertexIndex & 2) << 1);
    v_UV.x = (x + 1.0) * 0.5;
    v_UV.y = (y + 1.0) * 0.5;
    // z = 1.0 → after perspective divide depth = 1.0 = far plane.
    gl_Position = vec4(x, y, 1.0, 1.0);
}
