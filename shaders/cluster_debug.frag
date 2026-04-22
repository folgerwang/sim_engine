#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// --- cluster_debug.frag ------------------------------------------------
// Hashes the flat-interpolated cluster_id from the vertex stage into an
// RGB color so neighbouring clusters get visibly different tints. The
// transform is in the push-constant block for consistency with the vertex
// shader (even though this stage does not consume it) -- the renderer
// pushes a single contiguous struct to VERTEX|FRAGMENT.
// ----------------------------------------------------------------------

layout(push_constant) uniform ClusterDebugUniformBufferObject {
    ClusterDebugParams params;
};

layout(location = 0) flat in uint v_cluster_id;

layout(location = 0) out vec4 outColor;

// Cheap 32-bit integer hash (Thomas Wang). Good enough for eyeballing
// cluster boundaries -- we don't need cryptographic uniformity, we just
// need adjacent ids to land on very different colors.
uint hash32(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

vec3 clusterColor(uint id) {
    uint h = hash32(id + 0x9E3779B9u);
    // Pull three low-entropy byte slices out of the 32-bit hash. Bias the
    // channels upward so every cluster is at least mid-tone -- pure black
    // looks like a culled pixel and confuses the eye.
    float r = float((h      ) & 0xFFu) * (1.0 / 255.0);
    float g = float((h >>  8) & 0xFFu) * (1.0 / 255.0);
    float b = float((h >> 16) & 0xFFu) * (1.0 / 255.0);
    return mix(vec3(0.35), vec3(r, g, b), 0.85);
}

void main() {
    vec3 color = clusterColor(v_cluster_id);
    outColor = vec4(linearTosRGB(color), 1.0);
}
