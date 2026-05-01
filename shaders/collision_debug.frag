#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// --- collision_debug.frag ---------------------------------------------
// Hashes the flat-interpolated segmentation id from the vertex stage
// into an RGB colour so adjacent CollisionMeshes get visibly different
// solid tints, giving an "instance segmentation" view of the physics
// world. The vertex stage feeds the same mesh_id through every vertex
// of a given mesh, so the whole mesh resolves to one flat colour.
// ----------------------------------------------------------------------

layout(push_constant) uniform CollisionDebugUniformBufferObject {
    ClusterDebugParams params;
};

layout(location = 0) flat in uint v_triangle_id;

layout(location = 0) out vec4 outColor;

// Cheap 32-bit integer hash (Thomas Wang) — same one cluster_debug.frag
// uses. Adjacent triangle ids differ by 1 which the multiplier and
// shifts spread across all 24 colour bits.
uint hash32(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

vec3 triangleColor(uint id) {
    uint h = hash32(id + 0x9E3779B9u);
    float r = float((h      ) & 0xFFu) * (1.0 / 255.0);
    float g = float((h >>  8) & 0xFFu) * (1.0 / 255.0);
    float b = float((h >> 16) & 0xFFu) * (1.0 / 255.0);
    // Bias upward so no triangle reads as pure black (which is hard to
    // distinguish from a culled pixel).
    return mix(vec3(0.30), vec3(r, g, b), 0.85);
}

void main() {
    vec3 color = triangleColor(v_triangle_id);
    outColor = vec4(linearTosRGB(color), 1.0);
}
