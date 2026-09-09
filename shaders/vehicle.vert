#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ── THE VEHICLE VERTEX SHADER ───────────────────────────────────────
// Cars used to be drawn through citizen.vert, because a car and a
// citizen are the same kind of draw: a unit-space mesh repeated by a
// per-instance transform and a colour.  They stop being the same kind
// of draw once the car's geometry comes from the LIBRARY, because the
// library's vertices carry two things a citizen's do not:
//
//   THE PART ID.  Every triangle of a baked car knows what it is —
//   body, glass, trim, chrome, lamp, grille, tyre, rim, interior,
//   livery, light bar.  That is the whole interface to the renderer:
//   the fragment shader asks what a fragment IS instead of deducing it
//   from where the fragment sits and which way it faces.
//
//   THE PAINT.  A paint is not a colour, it is a basecoat under a clear
//   coat: how metallic the base is, how strongly its flake sparkles,
//   how sharp the coat's own specular is.  Those ride per instance so
//   one mesh serves every car on the street.
//
// Giving vehicles their own vertex shader rather than adding both to
// citizen.vert is what keeps the citizens' pipelines out of it: an
// attribute a pipeline does not bind but a shader reads is undefined,
// so sharing would have meant every citizen pipeline growing two
// attributes it has no data for.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_NORMAL) in vec3 in_normal;
// The part id, as a float because that is what a vertex stream is: one
// value per vertex, constant across a triangle by construction (the
// library never shares a vertex between two parts).
layout(location = VINPUT_TEXCOORD0) in float in_part;

layout(location = 10) in vec4 in_xform0;
layout(location = 11) in vec4 in_xform1;
layout(location = 12) in vec4 in_xform2;
layout(location = 13) in vec4 in_xform3;
layout(location = 14) in vec4 in_color;
layout(location = 15) in vec4 in_extra;
// (metal, flake, coat, pearl) — the paint's parameters, per instance
layout(location = 16) in vec4 in_paint;

layout(location = 0) out vec3 out_normal_ws;
layout(location = 1) out vec3 out_position_ws;
layout(location = 2) out vec4 out_color;
layout(location = 3) out vec3 out_local;
layout(location = 4) flat out vec4 out_extra;
layout(location = 5) out vec3 out_normal_ls;
layout(location = 6) flat out float out_part;
layout(location = 7) flat out vec4 out_paint;

void main() {
    mat4 xform = mat4(in_xform0, in_xform1, in_xform2, in_xform3);
    vec3 position_ws = (xform * vec4(in_position, 1.0)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_position_ws = position_ws;
    // The instance transform is a rotation times a uniform scale (a car
    // is placed, never squashed), so the normal needs no inverse
    // transpose — the upper 3x3 renormalised is exact.
    mat3 m3 = mat3(xform);
    vec3 s2 = vec3(dot(m3[0], m3[0]), dot(m3[1], m3[1]), dot(m3[2], m3[2]));
    out_normal_ws = normalize(m3 * (in_normal / max(s2, vec3(1e-8))));
    out_normal_ls = normalize(in_normal);
    out_color = in_color;
    out_local = in_position;
    out_extra = in_extra;
    out_part = in_part;
    out_paint = in_paint;
}
