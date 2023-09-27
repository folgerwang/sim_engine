#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_TEXCOORD0) in vec2 in_tex_coord;
layout(location = VINPUT_NORMAL) in vec3 in_normal;
layout(location = VINPUT_TANGENT) in vec3 in_tangent;

layout(location = 0) out ObjectVsPsData vs_out_data;

void main() {
    mat4 matrix_ws = model_params.model_mat;

    vec3 position_ws = (matrix_ws * vec4(in_position, 1.0f)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    vs_out_data.vertex_position = position_ws;
    vs_out_data.vertex_tex_coord = vec4(in_tex_coord, 0, 0);
    vs_out_data.vertex_normal =
        (matrix_ws * vec4(in_normal, 0.0f)).xyz;
    vs_out_data.vertex_tangent =
        (matrix_ws * vec4(in_tangent, 0.0f)).xyz;
    vs_out_data.vertex_binormal = cross(vs_out_data.vertex_normal, vs_out_data.vertex_tangent);
}