#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- scene_axis.vert ---------------------------------------------------
// Origin coordinate-system gizmo for the editor's "Create Scene" mode.
// Solid colour-per-vertex geometry (three arrows = X/Y/Z) drawn at the
// world origin.  Pairs with scene_axis.frag.
//
// Vertex layout fed by helper::SceneGrid:
//   binding 0, location VINPUT_POSITION = 0 : vec3 position (world space)
//   binding 1, location VINPUT_COLOR    = 6 : vec3 colour (linear)
// ----------------------------------------------------------------------

layout(push_constant) uniform SceneAxisUniformBufferObject {
    ClusterDebugParams params;   // reused: { mat4 transform }
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
        ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_COLOR)    in vec3 in_color;

layout(location = 0) out vec3 v_color;

void main() {
    vec3 position_ws = (params.transform * vec4(in_position, 1.0f)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    v_color = in_color;
}
