#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ── THE CHARACTER MESH (citizens v33) ───────────────────────────────
// A baked scan (tools/terrain/npc_bake.py) skinned to the citizen
// skeleton: four bones a vertex, the joint matrices in a per-frame
// palette SSBO as 3x4 rows (kNpcRows = 19 joints x 3 rows an
// instance; the instance's first row rides the instance stream).
// The matrices take the asset's unit-height rest space to the world,
// scale included, so the normal only needs the upper 3x3.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};
layout(std430, set = VIEW_PARAMS_SET + 1, binding = 0)
    readonly buffer NpcPalette {
    vec4 rows[];
};

layout(location = 0) in vec3  in_position;
layout(location = 1) in vec3  in_normal;
layout(location = 2) in vec2  in_uv;
layout(location = 3) in uvec4 in_joints;
layout(location = 4) in vec4  in_weights;
layout(location = 5) in vec4  in_inst;     // (first row, seed, lift, 0)

layout(location = 0) out vec3 out_normal_ws;
layout(location = 1) out vec3 out_position_ws;
layout(location = 2) out vec2 out_uv;
layout(location = 3) flat out vec4 out_inst;

void main() {
    uint base = uint(in_inst.x + 0.5);
    vec4 h = vec4(in_position, 1.0);
    vec3 p = vec3(0.0);
    vec3 n = vec3(0.0);
    float wsum = 0.0;
    for (int k = 0; k < 4; ++k) {
        float w = in_weights[k];
        if (w <= 0.0) continue;
        uint r = base + in_joints[k] * 3u;
        vec4 r0 = rows[r], r1 = rows[r + 1u], r2 = rows[r + 2u];
        p += w * vec3(dot(r0, h), dot(r1, h), dot(r2, h));
        n += w * vec3(dot(r0.xyz, in_normal), dot(r1.xyz, in_normal),
                      dot(r2.xyz, in_normal));
        wsum += w;
    }
    if (wsum > 0.0) { p /= wsum; n /= wsum; }
    else {
        uint r = base;
        p = vec3(dot(rows[r], h), dot(rows[r + 1u], h), dot(rows[r + 2u], h));
        n = in_normal;
    }
    gl_Position = camera_info.view_proj * vec4(p, 1.0);
    out_position_ws = p;
    out_normal_ws = normalize(n);
    out_uv = in_uv;
    out_inst = in_inst;
}
