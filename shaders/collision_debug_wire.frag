#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// --- collision_debug_wire.frag ----------------------------------------
// Wireframe-overlay fragment shader for the collision-mesh debug pass.
// Pairs with collision_debug.vert (so the vertex layout matches the
// solid-fill pipeline 1:1) and emits a flat opaque white. Drawn after
// collision_debug.frag has filled each mesh with its hashed
// segmentation colour, with depth_bias pushing lines toward the camera
// so they win the LESS depth test against the fill they ride on. The
// `v_triangle_id` flat input is left declared but unused -- keeping
// the input layout identical to the solid-fill shader avoids needing
// a second vertex shader.
// ----------------------------------------------------------------------

layout(push_constant) uniform CollisionDebugWireUniformBufferObject {
    ClusterDebugParams params;
};

layout(location = 0) flat in uint v_triangle_id;

layout(location = 0) out vec4 outColor;

void main() {
    // Slightly off-pure-white so it reads cleanly on saturated hash
    // colours without burning out against pale segmentation tints.
    outColor = vec4(linearTosRGB(vec3(0.92)), 1.0);
}
