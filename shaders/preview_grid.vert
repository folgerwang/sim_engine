#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- preview_grid.vert --------------------------------------------------
// Ground-grid quad for the Debug Display preview (helper::MeshPreview).
// Same push-constant camera as preview_mesh.*; the analytic grid lines are
// computed in preview_grid.frag with the ORIGIN AT THE OBJECT CENTRE.
// ------------------------------------------------------------------------

layout(push_constant) uniform PreviewMeshUniformBufferObject {
    PreviewMeshParams params;
};

layout(location = VINPUT_POSITION) in vec3 in_position;

layout(location = 0) out vec3 v_world;

void main() {
    v_world = in_position;
    gl_Position = params.view_proj * vec4(in_position, 1.0);
}
