#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- preview_grid.frag ----------------------------------------------------
// Analytic reference grid for the Debug Display preview — the same
// "pristine grid" technique as the scene view's scene_grid.frag (1 m minor /
// 10 m major lines, red X / blue Z axes), but self-contained: the camera
// comes from the push constants and the grid ORIGIN SITS AT THE OBJECT
// CENTRE (params.center_radius.xyz), with the fade radius scaled to the
// object's bounding radius.  Alpha-blended over the preview backdrop;
// gamma-encoded to match the RGBA8 target.
// ----------------------------------------------------------------------------

layout(push_constant) uniform PreviewMeshUniformBufferObject {
    PreviewMeshParams params;
};

layout(location = 0) in vec3 v_world;

layout(location = 0) out vec4 outColor;

float pristineGrid(vec2 uv, vec2 line_width) {
    vec2 ddx = dFdx(uv);
    vec2 ddy = dFdy(uv);
    vec2 uv_deriv = vec2(length(vec2(ddx.x, ddy.x)),
                         length(vec2(ddx.y, ddy.y)));
    vec2 target_width = line_width;
    vec2 draw_width = clamp(target_width, uv_deriv, vec2(0.5));
    vec2 line_aa    = uv_deriv * 2.5;
    vec2 grid_uv = 1.0 - abs(fract(uv) * 2.0 - 1.0);
    vec2 g2 = 1.0 - smoothstep(draw_width - line_aa,
                               draw_width + line_aa, grid_uv);
    g2 *= clamp(target_width / draw_width, 0.0, 1.0);
    g2 = mix(g2, target_width, clamp(uv_deriv * 2.0 - 1.0, 0.0, 1.0));
    return mix(g2.x, 1.0, g2.y);
}

float axisLine(float coord, float half_width_world) {
    float d  = fwidth(coord);
    float w  = max(half_width_world, d);
    float aa = d * 2.5;
    return 1.0 - smoothstep(w, w + aa, abs(coord));
}

void main() {
    const vec3  center = params.center_radius.xyz;
    const float r      = max(params.center_radius.w, 1e-4);

    // Grid coordinates relative to the OBJECT CENTRE — the axes cross
    // exactly at the object's origin.
    vec2 w = v_world.xz - center.xz;

    // Dark lines — the preview backdrop is a neutral gray (0.45), so the
    // grid must sit BELOW it in brightness to stay readable.
    const vec3 kMinorColor = vec3(0.075, 0.075, 0.080);
    const vec3 kMajorColor = vec3(0.030, 0.030, 0.034);
    const vec3 kAxisXColor = vec3(0.70, 0.08, 0.08);
    const vec3 kAxisZColor = vec3(0.08, 0.20, 0.75);

    float minorC = pristineGrid(w,        vec2(0.020));
    float majorC = pristineGrid(w / 10.0, vec2(0.012));
    float axisX  = axisLine(w.y, 0.03);
    float axisZ  = axisLine(w.x, 0.03);

    vec3  col = kMinorColor;
    float a   = minorC;
    col = mix(col, kMajorColor, majorC);
    a   = max(a, majorC);
    col = mix(col, kAxisZColor, axisZ);
    a   = max(a, axisZ);
    col = mix(col, kAxisXColor, axisX);
    a   = max(a, axisX);

    // Fade with distance from the object centre, scaled to its size so the
    // grid reads identically for tiny props and whole buildings.
    const float fade_end   = max(r * 3.0, 4.0);
    const float fade_start = fade_end * 0.55;
    float fade = 1.0 - smoothstep(fade_start, fade_end, length(w));

    a *= fade;
    if (a < 0.004) discard;

    // Perceptual ramp + overall opacity, then gamma to match the RGBA8
    // preview target (same conventions as the scene grid).
    a = pow(a, 1.0 / 2.2) * 0.22;
    outColor = vec4(pow(col, vec3(1.0 / 2.2)), a);
}
