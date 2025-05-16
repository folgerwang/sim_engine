#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(push_constant) uniform BaseShapeDrawUniformBufferObject {
    BaseShapeDrawParams params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_TEXCOORD0) in vec2 in_tex_coord;
layout(location = VINPUT_NORMAL) in vec3 in_normal;
layout(location = VINPUT_TANGENT) in vec4 in_tangent;

layout(location = 0) out BaseShapeVsPsData out_data;

void main() {
    vec3 position_ws = (params.transform * vec4(in_position, 1.0f)).xyz;

    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_data.vertex_position = position_ws;
    out_data.vertex_tex_coord = in_tex_coord;

    mat3 normal_mat = transpose(inverse(mat3(params.transform)));
    out_data.vertex_normal = normalize(normal_mat * in_normal);
    out_data.vertex_tangent = normalize(normal_mat * in_tangent.xyz);
    out_data.vertex_binormal = cross(out_data.vertex_normal, out_data.vertex_tangent) * in_tangent.w;
}