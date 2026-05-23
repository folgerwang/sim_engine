#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// --- collision_debug.frag ---------------------------------------------
// Per-category solid colouring of static collision meshes. The vertex
// stage forwards a flat uint seg_id, which the host fills with the
// MeshCategory enum value (Floor=1, Wall=2, Door=3, Object=4, Glass=5,
// Ceiling=6, Stairs=7, Vegetation=8, Unknown=0). The switch below
// assigns each category a hand-picked distinct colour; anything
// outside the known range falls back to a procedural hash so future
// categories still produce a visible tint before this shader is
// updated.
//
// Colour key (linear RGB, sRGB-encoded at output):
//   Floor       soft green     – walkable surfaces, the navigation plane
//   Wall        dim red        – blocking vertical surfaces
//   Door        amber yellow   – traversable openings
//   Object      warm purple    – gameplay props (tables, chairs, bottles)
//   Glass       pale cyan      – see-through-but-blocking
//   Ceiling     muted blue     – blocks-from-below (interior roofing)
//   Stairs      bright orange  – walkable but step-aware navigation
//   Vegetation  olive          – foliage / non-collidable greenery
//   Elevator    hot pink       – walkable + vertical lift traversal
//   Ladder      tan            – vertical hand-over-hand traversal
//   Unknown     neutral grey   – classifier couldn't decide; investigate
// ----------------------------------------------------------------------

layout(push_constant) uniform CollisionDebugUniformBufferObject {
    ClusterDebugParams params;
};

layout(location = 0) flat in uint v_triangle_id;

layout(location = 0) out vec4 outColor;

// Cheap 32-bit integer hash (Thomas Wang) — kept around for the fallback
// path so unknown / new category ids still produce a visible tint
// rather than reading as a pure black mesh.
uint hash32(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

vec3 hashedColor(uint id) {
    uint h = hash32(id + 0x9E3779B9u);
    float r = float((h      ) & 0xFFu) * (1.0 / 255.0);
    float g = float((h >>  8) & 0xFFu) * (1.0 / 255.0);
    float b = float((h >> 16) & 0xFFu) * (1.0 / 255.0);
    return mix(vec3(0.30), vec3(r, g, b), 0.85);
}

// Must stay in sync with the MeshCategory enum in collision_mesh.h.
// If the C++ side gets a new value, add a case here and the host code
// gets a default-fallback hash colour until you do.
vec3 categoryColor(uint id) {
    switch (id) {
        case 1u:  return vec3(0.20, 0.75, 0.30);  // Floor      – green
        case 2u:  return vec3(0.80, 0.20, 0.20);  // Wall       – red
        case 3u:  return vec3(0.95, 0.75, 0.10);  // Door       – amber
        case 4u:  return vec3(0.55, 0.25, 0.80);  // Object     – purple
        case 5u:  return vec3(0.40, 0.85, 0.95);  // Glass      – cyan
        case 6u:  return vec3(0.25, 0.45, 0.85);  // Ceiling    – blue
        case 7u:  return vec3(0.95, 0.50, 0.10);  // Stairs     – orange
        case 8u:  return vec3(0.55, 0.65, 0.20);  // Vegetation – olive
        case 9u:  return vec3(0.95, 0.20, 0.65);  // Elevator   – hot pink
        case 10u: return vec3(0.75, 0.55, 0.35);  // Ladder     – tan
        case 0u:  return vec3(0.55, 0.55, 0.55);  // Unknown    – grey
        default:  return hashedColor(id);
    }
}

void main() {
    vec3 color = categoryColor(v_triangle_id);
    // Alpha 0.8: the solid-fill pipeline is alpha-blended for the
    // translucent collision-LOD overlay drawn over the textured
    // scene.  (Ignored when the bound pipeline has blending off.)
    outColor = vec4(linearTosRGB(color), 0.8);
}
