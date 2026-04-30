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

// Outputs to fragment shader.
layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) flat out uint v_cluster_idx;

void main() {
    v_world_pos   = in_position;
    v_normal      = in_normal;
    v_uv          = in_uv;
    v_cluster_idx = gl_InstanceIndex;

    gl_Position = camera_info.view_proj * vec4(in_position, 1.0);
}
