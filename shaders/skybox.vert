#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(location = 0) in vec3 in_position;

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = 0) out VsPsData {
    vec3 vertex_position;
} out_data;

void main() {
    vec4 position_ws = vec4(in_position * 4000.0f + camera_info.position.xyz, 1.0);
    gl_Position = camera_info.view_proj * position_ws;

    out_data.vertex_position = position_ws.xyz;
}