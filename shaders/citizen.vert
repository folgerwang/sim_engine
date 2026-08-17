#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(push_constant) uniform CitizenDrawUniformBufferObject {
    CitizenDrawParams params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_NORMAL) in vec3 in_normal;

layout(location = 0) out vec3 out_normal_ws;
layout(location = 1) out vec3 out_position_ws;

void main() {
    vec3 position_ws = (params.transform * vec4(in_position, 1.0)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_position_ws = position_ws;
    mat3 nm = transpose(inverse(mat3(params.transform)));
    out_normal_ws = normalize(nm * in_normal);
}
