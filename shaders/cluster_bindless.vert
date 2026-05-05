#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ─── cluster_bindless.vert ──────────────────────────────────────────
// Bindless cluster rendering vertex shader.
//
// Vertices are already in WORLD SPACE (baked during uploadMeshClusters).
// The only transform needed is view-proj from the camera UBO.
//
// gl_InstanceIndex carries the cluster index (set as firstInstance by
// the culling compute shader), which indexes into ClusterDrawInfo[]
// for material lookup in the fragment stage.
// ─────────────────────────────────────────────────────────────────────

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

// Vertex attributes from the merged vertex buffer (BindlessVertex layout).
layout(location = 0) in vec3 in_position;   // world-space position
layout(location = 1) in vec3 in_normal;      // world-space normal
layout(location = 2) in vec2 in_uv;          // texture UV
// in_tangent.xyz = world-space tangent (orthogonalised against in_normal,
// computed at upload by computeMeshTangents in cluster_renderer.cpp);
// in_tangent.w   = bitangent handedness sign so the fragment shader can
//                  recover B = cross(N, T) * w without storing B explicitly.
layout(location = 3) in vec4 in_tangent;

// Outputs to fragment shader.
layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) flat out uint v_cluster_idx;
layout(location = 4) out vec4 v_tangent;
// Current and previous-frame homogeneous clip positions.  The fragment
// shader's GBUFFER_OUTPUT branch divides each by .w to get NDC and
// writes (curNDC.xy - prevNDC.xy) into the RT3 velocity attachment.
// Cluster vertices are baked in WORLD space and the geometry is static,
// so velocity here purely reflects camera motion — that's enough for
// camera-driven TAA/motion blur on cluster meshes.  Animated/skinned
// instances would also need their previous-frame model transform fed in
// here, but the current cluster pipeline doesn't have that.
layout(location = 5) out vec4 v_cur_clip;
layout(location = 6) out vec4 v_prev_clip;

void main() {
    v_world_pos   = in_position;
    v_normal      = in_normal;
    v_uv          = in_uv;
    v_cluster_idx = gl_InstanceIndex;
    v_tangent     = in_tangent;

    vec4 world = vec4(in_position, 1.0);
    v_cur_clip  = camera_info.view_proj      * world;
    v_prev_clip = camera_info.prev_view_proj * world;
    gl_Position = v_cur_clip;
}
