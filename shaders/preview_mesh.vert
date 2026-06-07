#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- preview_mesh.vert --------------------------------------------------
// Editor Debug Display GPU preview pass (helper::MeshPreview).  Renders the
// selected object's triangles into a small offscreen target; PBR lighting
// (a three-spot studio rig) happens per-fragment in preview_mesh.frag.
//
// Vertex layout fed by helper::MeshPreview:
//   binding 0, location VINPUT_POSITION  = 0 : vec3 position (mesh space)
//   binding 1, location VINPUT_NORMAL    = 2 : vec3 normal
//   binding 2, location VINPUT_TEXCOORD0 = 1 : vec2 uv (zeros when absent)
// ------------------------------------------------------------------------

layout(push_constant) uniform PreviewMeshUniformBufferObject {
    PreviewMeshParams params;
};

layout(location = VINPUT_POSITION)  in vec3 in_position;
layout(location = VINPUT_NORMAL)    in vec3 in_normal;
layout(location = VINPUT_TEXCOORD0) in vec2 in_uv;

layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;

void main() {
    v_world  = in_position;
    v_normal = in_normal;
    v_uv     = in_uv;
    gl_Position = params.view_proj * vec4(in_position, 1.0);
}
