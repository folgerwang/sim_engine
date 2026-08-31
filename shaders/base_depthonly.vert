#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "veg_sway.glsl.h"

layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

#ifdef CSM_PER_CASCADE
// ─── CSM_PER_CASCADE permutation ──────────────────────────────────────────
// "Regular" CSM mode: the host loops over CSM_CASCADE_COUNT cascades and
// renders each into a single-layer view of the shadow array.  Each draw
// pushes a per-cascade cascade_idx into ModelParams; the VS uses it to
// index lights_params.light_view_proj[] instead of reading the main
// shadow camera's view_proj from VIEW_PARAMS_SET.  This is the no-GS,
// no-mesh-shader baseline path on the shadow draw-mode menu.
//
// RUNTIME_LIGHTS_PARAMS_SET is already bound to the shadow pass for the
// GS path (base_depthonly_csm.geom reads light_view_proj[gl_Layer]).
// The pipeline layout owner adds VERTEX_BIT to that set's stage flags so
// this VS permutation can read the same UBO.
layout(std140, set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUBO {
    RuntimeLightsParams lights_params;
};
#endif

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

// 48 B instance layout: three vec4 columns of the rotation*scale basis
// with the world translation spread across the .w lanes (see
// BakedInstanceXform in drawable_object.h).  The old separate
// IINPUT_MAT_POS_SCALE attribute (location 14) no longer exists.
layout(location = IINPUT_MAT_ROT_0) in vec4 in_loc_rot_mat_0;
layout(location = IINPUT_MAT_ROT_1) in vec4 in_loc_rot_mat_1;
layout(location = IINPUT_MAT_ROT_2) in vec4 in_loc_rot_mat_2;

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
        mat3x3(in_loc_rot_mat_0.xyz,
               in_loc_rot_mat_1.xyz,
               in_loc_rot_mat_2.xyz);
    vec3 position_ws =
        local_world_rot_mat *
        position_ls +
        vec3(in_loc_rot_mat_0.w,
             in_loc_rot_mat_1.w,
             in_loc_rot_mat_2.w);
    // Vegetation wind sway — SAME function and SAME inputs as
    // base.vert, so the shadow of a swaying tree stays under the tree.
    if ((model_params.flip_uv_coord & MODEL_FLAG_VEGETATION_SWAY) != 0u) {
        vec3 inst_t = vec3(in_loc_rot_mat_0.w, in_loc_rot_mat_1.w,
                           in_loc_rot_mat_2.w);
        // Instance up axis: gates out the deadfall trunks that lie
        // pitched ~90 deg on the ground (see kVegFallenCos).
        position_ws += vegSwayOffset(inst_t, position_ls.y,
                                     camera_info.time_s,
                                     local_world_rot_mat * vec3(0.0f, 1.0f,
                                                                0.0f));
    }
#ifdef CSM_PER_CASCADE
    // Per-cascade VP picked by model_params.cascade_idx (written by
    // drawMesh before each cascade pass).  The host has already uploaded
    // light_view_proj[0..CSM_CASCADE_COUNT-1] this frame for the GS path,
    // so we re-use the same UBO without any extra upload work.
    gl_Position =
        lights_params.light_view_proj[model_params.cascade_idx]
        * vec4(position_ws, 1.0);
#else
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
#endif
    out_data.vertex_position = position_ws;
    out_data.vertex_ilod_fade = 1.0;
    // ── Per-instance LOD hard pick (depth / shadow passes) ───────────
    // Same packed band window as base.vert, but collapsed to a hard
    // midpoint ownership test — the depth pipelines don't run the
    // dissolve, and drawing both bands would double every caster in
    // the transition (the same reason node-level LOD uses
    // lod_node_owner_ here).  Exactly one band keeps each instance:
    // the fading-out side owns to the midpoint (w > 0.5), the
    // fading-in side beyond it (w < -0.5).  A culled instance emits a
    // degenerate clip position AND a far-underground vertex_position,
    // so the CSM geometry-shader path (which re-projects
    // vertex_position, not gl_Position) drops it too.
    {
        uint ilod_bits = floatBitsToUint(model_params.model_params_pad0);
        if (ilod_bits != 0u) {
            vec3 inst_t = vec3(in_loc_rot_mat_0.w, in_loc_rot_mat_1.w,
                               in_loc_rot_mat_2.w);
            float ilod_d = distance(inst_t.xz, camera_info.position.xz);
            float ilod_near = float((ilod_bits >> 16) & 0x7FFFu) * 0.25;
            float ilod_far = float(ilod_bits & 0xFFFFu) * 0.25;
            float ilod_k_out = clamp(ilod_far * 0.05, 2.0, 24.0);
            float ilod_k_in = clamp(ilod_near * 0.05, 2.0, 24.0);
            float ilod_w_out = ((ilod_bits & 0x80000000u) != 0u)
                ? clamp((ilod_far + ilod_k_out - ilod_d) /
                            (2.0 * ilod_k_out), 0.0, 1.0)
                : (ilod_d < ilod_far ? 1.0 : 0.0);
            float ilod_w_in = (ilod_near > 0.01)
                ? clamp((ilod_d - (ilod_near - ilod_k_in)) /
                            (2.0 * ilod_k_in), 0.0, 1.0)
                : 1.0;
            float ilod_w =
                (ilod_w_in < ilod_w_out) ? -ilod_w_in : ilod_w_out;
            out_data.vertex_ilod_fade = ilod_w;
            if (!(ilod_w > 0.5 || ilod_w < -0.5)) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                out_data.vertex_position = vec3(0.0, -4.0e8, 0.0);
            }
        }
    }
    out_data.vertex_tex_coord = vec4(0);
#ifdef HAS_UV_SET0
    out_data.vertex_tex_coord.xy = in_tex_coord.xy;
    if ((model_params.flip_uv_coord & 0x01) != 0)
        out_data.vertex_tex_coord.x = 1.0f - in_tex_coord.x;
    if ((model_params.flip_uv_coord & 0x02) != 0)
        out_data.vertex_tex_coord.y = 1.0f - in_tex_coord.y;
#endif
}