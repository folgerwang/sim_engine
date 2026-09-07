#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ── THE SKINNED CITIZEN PART ────────────────────────────────────────
// A part of the near-tier figure is a tube / blob / ball in a unit box
// (-1..1, +y along the limb) hung off one joint, and its two ends are
// skinned to the neighbouring joints: the instance carries THREE 3x4
// transforms of the same unit box -- hung off the part's own joint
// (self), off the joint above (up), off the joint below (lo) -- which
// agree exactly at the pivots the joints share.  A vertex past the
// upper pivot follows `up` entirely, one past the lower pivot follows
// `lo`, and the ones within `blend` of a pivot mix, so an elbow or a
// knee is one surface bending rather than two blocks meeting at a
// hinge.  `taper` narrows the bottom end so adjacent parts match
// cross-sections at their shared pivot (thigh to knee, chest to
// waist).  Shading data and the paint record go to citizen.frag
// exactly as the cube pipeline's do.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = VINPUT_NORMAL) in vec3 in_normal;

layout(location = 3)  in vec4 in_self0;    // rows of the 3x4 transforms
layout(location = 4)  in vec4 in_self1;
layout(location = 5)  in vec4 in_self2;
layout(location = 6)  in vec4 in_up0;
layout(location = 7)  in vec4 in_up1;
layout(location = 8)  in vec4 in_up2;
layout(location = 9)  in vec4 in_lo0;
layout(location = 10) in vec4 in_lo1;
layout(location = 11) in vec4 in_lo2;
layout(location = 12) in vec4 in_color;
layout(location = 13) in vec4 in_extra;
// (unit y of the upper pivot, unit y of the lower pivot, blend
// half-width in unit y, taper = bottom width / top width)
layout(location = 14) in vec4 in_shape;

layout(location = 0) out vec3 out_normal_ws;
layout(location = 1) out vec3 out_position_ws;
layout(location = 2) out vec4 out_color;
layout(location = 3) out vec3 out_local;
layout(location = 4) flat out vec4 out_extra;

vec3 xf_point(vec4 r0, vec4 r1, vec4 r2, vec3 p) {
    vec4 h = vec4(p, 1.0);
    return vec3(dot(r0, h), dot(r1, h), dot(r2, h));
}

// The transforms are rotations times non-uniform scales (a limb is
// long and thin): the normal matrix is M * S^-2 -- divide by the
// squared column lengths before the upper 3x3.
vec3 xf_normal(vec4 r0, vec4 r1, vec4 r2, vec3 n) {
    vec3 c0 = vec3(r0.x, r1.x, r2.x);
    vec3 c1 = vec3(r0.y, r1.y, r2.y);
    vec3 c2 = vec3(r0.z, r1.z, r2.z);
    vec3 s2 = max(vec3(dot(c0, c0), dot(c1, c1), dot(c2, c2)),
                  vec3(1e-8));
    vec3 q = n / s2;
    return c0 * q.x + c1 * q.y + c2 * q.z;
}

void main() {
    // the taper, in the part's own space: full width at the top end,
    // `taper` of it at the bottom end
    float y = in_position.y;
    float f = mix(in_shape.w, 1.0, (y + 1.0) * 0.5);
    vec3 p = vec3(in_position.x * f, y, in_position.z * f);
    vec3 n = in_normal;
    n.y += (1.0 - in_shape.w) * 0.5 * length(in_normal.xz);
    n = normalize(n);

    // the skinning weights, by unit height about the two pivots
    float bl = in_shape.z;
    float wu = 0.0, wl = 0.0;
    if (bl > 0.0) {
        wu = clamp(0.5 + (y - in_shape.x) / (2.0 * bl), 0.0, 1.0);
        wl = clamp(0.5 - (y - in_shape.y) / (2.0 * bl), 0.0, 1.0);
    }
    float ws = max(0.0, 1.0 - wu - wl);

    vec3 position_ws =
        ws * xf_point(in_self0, in_self1, in_self2, p) +
        wu * xf_point(in_up0, in_up1, in_up2, p) +
        wl * xf_point(in_lo0, in_lo1, in_lo2, p);
    vec3 normal_ws =
        ws * xf_normal(in_self0, in_self1, in_self2, n) +
        wu * xf_normal(in_up0, in_up1, in_up2, n) +
        wl * xf_normal(in_lo0, in_lo1, in_lo2, n);

    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    out_position_ws = position_ws;
    out_normal_ws = normalize(normal_ws);
    out_color = in_color;
    out_local = in_position;       // the paint is drawn in unit space
    out_extra = in_extra;
}
