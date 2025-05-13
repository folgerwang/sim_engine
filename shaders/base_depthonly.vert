#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

#if defined(HAS_SKIN_SET_0) || defined(HAS_SKIN_SET_1)
layout(std430, set = SKIN_PARAMS_SET, binding = JOINT_CONSTANT_INDEX) readonly buffer JointMatrices {
	mat4 joint_matrices[];
};
#endif

layout(location = VINPUT_POSITION) in vec3 in_position;

#ifdef HAS_UV_SET0
layout(location = VINPUT_TEXCOORD0) in vec2 in_tex_coord;
#endif

#ifdef HAS_SKIN_SET_0
layout(location = VINPUT_JOINTS_0) in uvec4 in_joints_0;
layout(location = VINPUT_WEIGHTS_0) in vec4 in_weights_0;
#endif

#ifdef HAS_SKIN_SET_1
layout(location = VINPUT_JOINTS_1) in uvec4 in_joints_1;
layout(location = VINPUT_WEIGHTS_1) in vec4 in_weights_1;
#endif

layout(location = IINPUT_MAT_ROT_0) in vec3 in_loc_rot_mat_0;
layout(location = IINPUT_MAT_ROT_1) in vec3 in_loc_rot_mat_1;
layout(location = IINPUT_MAT_ROT_2) in vec3 in_loc_rot_mat_2;
layout(location = IINPUT_MAT_POS_SCALE) in vec4 in_loc_pos_scale;

layout(location = 0) out ObjectVsPsData out_data;

void main() {
	// Calculate skinned matrix from weights and joint indices of the current vertex
    mat4 matrix_ls = model_params.model_mat;
#if defined(HAS_SKIN_SET_0) || defined(HAS_SKIN_SET_1)
#ifdef HAS_SKIN_SET_0
	mat4 skin_matrix =
		in_weights_0.x * joint_matrices[int(in_joints_0.x)] +
		in_weights_0.y * joint_matrices[int(in_joints_0.y)] +
		in_weights_0.z * joint_matrices[int(in_joints_0.z)] +
		in_weights_0.w * joint_matrices[int(in_joints_0.w)];
#endif
#ifdef HAS_SKIN_SET_1
	skin_matrix +=
		in_weights_1.x * joint_matrices[int(in_joints_1.x)] +
		in_weights_1.y * joint_matrices[int(in_joints_1.y)] +
		in_weights_1.z * joint_matrices[int(in_joints_1.z)] +
		in_weights_1.w * joint_matrices[int(in_joints_1.w)];
#endif
    matrix_ls = matrix_ls * skin_matrix;
#endif
    vec3 position_ls = (matrix_ls * vec4(in_position, 1.0f)).xyz;

    mat3 local_world_rot_mat =
        mat3x3(in_loc_rot_mat_0,
               in_loc_rot_mat_1,
               in_loc_rot_mat_2);
    vec3 position_ws =
        local_world_rot_mat *
        position_ls +
        in_loc_pos_scale.xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_data.vertex_position = position_ws;
    out_data.vertex_tex_coord = vec4(0);
#ifdef HAS_UV_SET0
    out_data.vertex_tex_coord.xy = in_tex_coord.xy;
    if ((model_params.flip_uv_coord & 0x01) != 0)
        out_data.vertex_tex_coord.x = 1.0f - in_tex_coord.x;
    if ((model_params.flip_uv_coord & 0x02) != 0)
        out_data.vertex_tex_coord.y = 1.0f - in_tex_coord.y;
#endif
}