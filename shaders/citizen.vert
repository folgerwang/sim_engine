#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ── ONE DRAW FOR THE WHOLE TOWN ─────────────────────────────────────
// This used to take its transform and colour from a PUSH CONSTANT, which
// meant one pushConstants + one drawIndexed PER BOX — seven per detailed
// citizen, one per distant one.  That is what capped the far tier at
// kMaxFarParts and is why anybody past the budget simply vanished: the
// limit was never the triangles (a box is 12), it was the draw calls.
//
// The same data now arrives as PER-INSTANCE VERTEX ATTRIBUTES, so the
// whole visible population is one bindVertexBuffers + one instanced
// drawIndexed (two, with the fine tier's rounded mesh).  Locations
// 10-15 are free in this pipeline (it declares only POSITION and
// NORMAL) and deliberately clear of the VINPUT_* block, which runs 0-9.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_NORMAL) in vec3 in_normal;

// per-instance: the 4 columns of the part transform, its colour, and
// the painting record (kind, style, packed accent colour, seed)
layout(location = 10) in vec4 in_xform0;
layout(location = 11) in vec4 in_xform1;
layout(location = 12) in vec4 in_xform2;
layout(location = 13) in vec4 in_xform3;
layout(location = 14) in vec4 in_color;
layout(location = 15) in vec4 in_extra;

layout(location = 0) out vec3 out_normal_ws;
layout(location = 1) out vec3 out_position_ws;
layout(location = 2) out vec4 out_color;
// the part's own unit-space position (-1..1 on every axis): what the
// garment painting in citizen.frag is drawn in
layout(location = 3) out vec3 out_local;
layout(location = 4) flat out vec4 out_extra;

void main() {
    mat4 xform = mat4(in_xform0, in_xform1, in_xform2, in_xform3);
    vec3 position_ws = (xform * vec4(in_position, 1.0)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_position_ws = position_ws;
    // The parts are rotations times NON-UNIFORM scales (a limb is long
    // and thin), never shears: M = R*S, so the normal matrix R*S^-1 is
    // M*S^-2 -- divide the local normal by the squared column lengths
    // and the upper 3x3 does the rest.  Cheaper than a mat3 inverse,
    // and exact; on the cube's axis normals it changes nothing, on the
    // rounded mesh it is what keeps a thin limb shading like a cylinder
    // rather than a squashed sphere.
    mat3 m3 = mat3(xform);
    vec3 s2 = vec3(dot(m3[0], m3[0]), dot(m3[1], m3[1]), dot(m3[2], m3[2]));
    out_normal_ws = normalize(m3 * (in_normal / max(s2, vec3(1e-8))));
    out_color = in_color;
    out_local = in_position;
    out_extra = in_extra;
}
