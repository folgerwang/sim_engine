#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

#if defined(HAS_SKIN_SET_0) || defined(HAS_SKIN_SET_1)
layout(std430, set = MODEL_PARAMS_SET, binding = JOINT_CONSTANT_INDEX) readonly buffer JointMatrices {
	mat4 joint_matrices[];
};
#endif

layout(location = VINPUT_POSITION) in vec3 in_position;

#ifdef HAS_UV_SET0
layout(location = VINPUT_TEXCOORD0) in vec2 in_tex_coord;
#endif

#ifdef HAS_UV_SET1
layout(location = VINPUT_TEXCOORD1) in vec2 in_tex_coord;
#endif

#ifdef HAS_NORMALS
layout(location = VINPUT_NORMAL) in vec3 in_normal;
#ifdef HAS_TANGENT
layout(location = VINPUT_TANGENT) in vec4 in_tangent;
#endif
#endif

#ifdef HAS_VERTEX_COLOR_VEC3
layout(location = VINPUT_COLOR) in vec3 v_color;
#endif

#ifdef HAS_VERTEX_COLOR_VEC4
layout(location = VINPUT_COLOR) in vec4 v_color;
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

layout(location = 0) out VsPsData {
    vec3 vertex_position;
    vec4 vertex_tex_coord;
#ifdef HAS_NORMALS
    vec3 vertex_normal;
#ifdef HAS_TANGENT
    vec3 vertex_tangent;
    vec3 vertex_binormal;
#endif
#endif
#ifdef HAS_VERTEX_COLOR_VEC3
    vec3 vertex_color;
#endif
#ifdef HAS_VERTEX_COLOR_VEC4
    vec4 vertex_color;
#endif
} out_data;

#ifdef HAS_NORMALS
vec3 getNormal()
{
    vec3 normal = in_normal;

#ifdef USE_MORPHING
    normal += getTargetNormal();
#endif

#ifdef USE_SKINNING
    normal = mat3(getSkinningNormalMatrix()) * normal;
#endif

    return normalize(normal);
}
#endif

#ifdef HAS_TANGENT
vec3 getTangent()
{
    vec3 tangent = in_tangent.xyz;

#ifdef USE_MORPHING
    tangent += getTargetTangent();
#endif

#ifdef USE_SKINNING
    tangent = mat3(getSkinningMatrix()) * tangent;
#endif

    return normalize(tangent);
}
#endif

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
    out_data.vertex_tex_coord.xy = in_tex_coord;
#endif
#ifdef HAS_UV_SET1
    out_data.vertex_tex_coord.zw = in_tex_coord;
#endif

#ifdef HAS_NORMALS
    mat3 normal_mat = transpose(inverse(local_world_rot_mat * mat3(matrix_ls)));
    out_data.vertex_normal = normalize(normal_mat * getNormal());
#ifdef HAS_TANGENT
    out_data.vertex_tangent = normalize(normal_mat * getTangent());
    out_data.vertex_binormal = cross(out_data.vertex_normal, out_data.vertex_tangent) * in_tangent.w;
#endif
#endif // !HAS_NORMALS


#if defined(HAS_VERTEX_COLOR_VEC3) || defined(HAS_VERTEX_COLOR_VEC4)
    out_data.vertex_color = v_color;
#endif
}