#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// --- scene_axis.frag ---------------------------------------------------
// Flat per-vertex colour for the origin XYZ gizmo (no lighting, so each
// axis reads as a solid, unambiguous colour).  Linear colour authored on
// the C++ side, converted to sRGB to match the forward output convention.
// ----------------------------------------------------------------------

layout(push_constant) uniform SceneAxisUniformBufferObject {
    ClusterDebugParams params;
};

layout(location = 0) in vec3 v_color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(linearTosRGB(v_color), 1.0);
}
