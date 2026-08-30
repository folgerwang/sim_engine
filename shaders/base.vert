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

#if defined(HAS_SKIN_SET_0) || defined(HAS_SKIN_SET_1)
layout(std430, set = SKIN_PARAMS_SET, binding = JOINT_CONSTANT_INDEX) readonly buffer JointMatrices {
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

// 48 B InstanceDataInfo: three vec4s, .xyz = basis COLUMN,
// .w = one translation component (the old IINPUT_MAT_POS_SCALE
// attribute is gone).  Must match the attribute setup in
// drawable_object.cpp (createDrawableGraphicsPipeline).
layout(location = IINPUT_MAT_ROT_0) in vec4 in_loc_rot_mat_0;
layout(location = IINPUT_MAT_ROT_1) in vec4 in_loc_rot_mat_1;
layout(location = IINPUT_MAT_ROT_2) in vec4 in_loc_rot_mat_2;

layout(location = 0) out ObjectVsPsData out_data;

// ── Skinning helpers ─────────────────────────────────────────────────────────
// getNormal() / getTangent() are called before skin_matrix is computed in
// main(), so they cannot reference that local variable.  These functions
// recompute the weighted joint matrix on demand.
#ifdef USE_SKINNING
mat4 getSkinningMatrix()
{
    mat4 skin = mat4(0);
    float wsum = 0.0;
#ifdef HAS_SKIN_SET_0
    skin += in_weights_0.x * joint_matrices[int(in_joints_0.x)]
          + in_weights_0.y * joint_matrices[int(in_joints_0.y)]
          + in_weights_0.z * joint_matrices[int(in_joints_0.z)]
          + in_weights_0.w * joint_matrices[int(in_joints_0.w)];
    wsum += in_weights_0.x + in_weights_0.y + in_weights_0.z + in_weights_0.w;
#endif
#ifdef HAS_SKIN_SET_1
    skin += in_weights_1.x * joint_matrices[int(in_joints_1.x)]
          + in_weights_1.y * joint_matrices[int(in_joints_1.y)]
          + in_weights_1.z * joint_matrices[int(in_joints_1.z)]
          + in_weights_1.w * joint_matrices[int(in_joints_1.w)];
    wsum += in_weights_1.x + in_weights_1.y + in_weights_1.z + in_weights_1.w;
#endif
    if (wsum > 1e-4) skin *= (1.0 / wsum);   // match the position path
    else             skin = mat4(1.0);
    return skin;
}

mat4 getSkinningNormalMatrix()
{
    return transpose(inverse(getSkinningMatrix()));
}
#endif // USE_SKINNING

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
    // ── Runtime skip-skinning bypass ──────────────────────────────
    // model_params.debug_skip_skinning != 0 forces this primitive to
    // render in its glTF bind pose (skin_matrix collapses to identity
    // i.e. matrix_ls stays = model_params.model_mat).  Used as a smoke
    // test for whether a "missing" skinned drawable is failing due to
    // bad skin math (degenerate joint matrices, broken bind matrices,
    // etc.) versus the draw not running at all.  We still consume the
    // joint / weight vertex inputs because they're declared in the
    // pipeline layout; we just don't index into joint_matrices when
    // skipping, which also keeps the path safe if joint_matrices is
    // empty or mis-sized.
    if (model_params.debug_skip_skinning == 0u) {
        mat4 skin_matrix = mat4(0);
        float wsum = 0.0;
#ifdef HAS_SKIN_SET_0
        skin_matrix +=
            in_weights_0.x * joint_matrices[int(in_joints_0.x)] +
            in_weights_0.y * joint_matrices[int(in_joints_0.y)] +
            in_weights_0.z * joint_matrices[int(in_joints_0.z)] +
            in_weights_0.w * joint_matrices[int(in_joints_0.w)];
        wsum += in_weights_0.x + in_weights_0.y + in_weights_0.z + in_weights_0.w;
#endif
#ifdef HAS_SKIN_SET_1
        skin_matrix +=
            in_weights_1.x * joint_matrices[int(in_joints_1.x)] +
            in_weights_1.y * joint_matrices[int(in_joints_1.y)] +
            in_weights_1.z * joint_matrices[int(in_joints_1.z)] +
            in_weights_1.w * joint_matrices[int(in_joints_1.w)];
        wsum += in_weights_1.x + in_weights_1.y + in_weights_1.z + in_weights_1.w;
#endif
        // Renormalize so weights that don't sum to 1 don't drag the vertex toward
        // the origin; an unweighted vertex (wsum~0) stays at bind pose instead of
        // collapsing to the object origin.  Mirrors the CPU preview path.
        if (wsum > 1e-4) skin_matrix *= (1.0 / wsum);
        else             skin_matrix = mat4(1.0);
        matrix_ls = matrix_ls * skin_matrix;
    }
#endif
    vec3 position_ls = (matrix_ls * vec4(in_position, 1.0f)).xyz;

    mat3 local_world_rot_mat =
        mat3x3(in_loc_rot_mat_0.xyz,
               in_loc_rot_mat_1.xyz,
               in_loc_rot_mat_2.xyz);
    // Translation rides the .w lanes of the three basis columns
    // (48 B InstanceDataInfo layout).
    vec3 position_ws =
        local_world_rot_mat *
        position_ls +
        vec3(in_loc_rot_mat_0.w,
             in_loc_rot_mat_1.w,
             in_loc_rot_mat_2.w);
    // ── Vegetation wind sway ────────────────────────────────────────
    // Flagged per draw (bit 3, set only for the _pcg_trees /
    // _pcg_clutter groups), so the branch is uniform across the draw
    // and costs nothing where it is off.  Displacement in WORLD space
    // after the instance transform; the bend height is the LOCAL
    // (pre-instance) height, so a rotated or scaled instance still
    // bends from its own roots.  Same function runs in
    // base_depthonly.vert — shadows track the canopy.
    if ((model_params.flip_uv_coord & MODEL_FLAG_VEGETATION_SWAY) != 0u) {
        vec3 inst_t = vec3(in_loc_rot_mat_0.w, in_loc_rot_mat_1.w,
                           in_loc_rot_mat_2.w);
        position_ws += vegSwayOffset(inst_t, position_ls.y,
                                     camera_info.time_s);
    }
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_data.vertex_position = position_ws;

    // ── Per-instance LOD band (dense ground cover) ───────────────────
    // See ModelParams::model_params_pad0.  Weight mirrors the CPU's
    // per-tile cross-fade, but keyed on THIS INSTANCE's translation —
    // the transforms are shared byte-for-byte across the bands, so both
    // sides of a boundary compute identical distances and their
    // screen-door halves partition every pixel: clumps swap band by
    // their own camera distance, not by 256 m tile.
    out_data.vertex_ilod_fade = 1.0;
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
            out_data.vertex_ilod_fade =
                (ilod_w_in < ilod_w_out) ? -ilod_w_in : ilod_w_out;
        }
    }

    // ── Weight-sum debug varying ─────────────────────────────────────────────
    // Carry the raw (pre-normalization) sum of all skin influences so the
    // WEIGHT_SUM render-debug mode can colour the mesh by it.  -1 marks a
    // non-skinned vertex; computed independently of debug_skip_skinning so the
    // visualization reflects the actual uploaded weights either way.
    out_data.vertex_weight_sum = -1.0;
#if defined(HAS_SKIN_SET_0) || defined(HAS_SKIN_SET_1)
    {
        float ws = 0.0;
#ifdef HAS_SKIN_SET_0
        ws += in_weights_0.x + in_weights_0.y + in_weights_0.z + in_weights_0.w;
#endif
#ifdef HAS_SKIN_SET_1
        ws += in_weights_1.x + in_weights_1.y + in_weights_1.z + in_weights_1.w;
#endif
        out_data.vertex_weight_sum = ws;
    }
#endif

    out_data.vertex_tex_coord = vec4(0);
#ifdef HAS_UV_SET0
    out_data.vertex_tex_coord.xy = in_tex_coord.xy;
    if ((model_params.flip_uv_coord & 0x01) != 0)
        out_data.vertex_tex_coord.x = 1.0f - in_tex_coord.x;
    if ((model_params.flip_uv_coord & 0x02) != 0)
        out_data.vertex_tex_coord.y = 1.0f - in_tex_coord.y;
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