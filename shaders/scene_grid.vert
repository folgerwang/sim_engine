#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- scene_grid.vert ---------------------------------------------------
// Reference grid / ruler for the editor's "Create Scene" mode.  Draws a
// single flat ground quad in the XZ plane (Y up), centred at the origin.
// The actual grid lines are computed analytically in the fragment shader
// from the interpolated world position, which keeps every line ~1 pixel
// wide at any distance (no aliasing / moire like raw GPU line primitives).
//
// Vertex layout fed by helper::SceneGrid:
//   binding 0, location VINPUT_POSITION = 0 : vec3 position (quad corner)
// ----------------------------------------------------------------------

layout(push_constant) uniform SceneGridUniformBufferObject {
    ClusterDebugParams params;   // reused: { mat4 transform }
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
        ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;

layout(location = 0) out vec3 v_world;   // world-space position of this corner

void main() {
    vec3 position_ws = (params.transform * vec4(in_position, 1.0f)).xyz;
    v_world = position_ws;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
}
