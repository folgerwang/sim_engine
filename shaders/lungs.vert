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

layout(location = 0) out VsPsData {
    vec3 vertex_position;
    vec2 vertex_tex_coord;
    vec3 vertex_normal;
} out_data;

void main() {
    mat4 matrix_ls = model_params.model_mat;
    vec3 position_ls = (matrix_ls * vec4(in_position, 1.0f)).xyz;
    out_data.vertex_normal = mat3(matrix_ls) * in_normal.xyz;
    out_data.vertex_tex_coord = vec2(in_tex_coord.x, 1.0f - in_tex_coord.y);
    out_data.vertex_position = position_ls;

    gl_Position = camera_info.view_proj * vec4(position_ls, 1.0);
}