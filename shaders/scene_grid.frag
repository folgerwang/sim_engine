#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// --- scene_grid.frag ---------------------------------------------------
// Analytic, anti-aliased reference grid drawn across a single ground quad.
//
// Uses the "pristine grid" technique (Ben Golus): line width is expressed
// as a fraction of a cell, the on-screen draw width is clamped to never go
// below one pixel (so lines stay crisp instead of staircasing), and when a
// line WOULD be thinner than a pixel its opacity is scaled down so the far
// field fades out smoothly instead of dissolving into a moire of converging
// lines.  This is what keeps the grid from looking pixelated at distance and
// at grazing angles.
//
// Three tiers are composited:
//   * minor lines  every 1 m   — dim grey
//   * major lines  every 10 m  — brighter grey
//   * centre axes  (X red at z=0, Z blue at x=0)
// ----------------------------------------------------------------------

layout(push_constant) uniform SceneGridUniformBufferObject {
    ClusterDebugParams params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
        ViewCameraInfo camera_info;
};

layout(location = 0) in vec3 v_world;

layout(location = 0) out vec4 outColor;

// Coverage of a periodic line set on `uv` (one line per integer unit).
// `line_width` is the desired half-width as a fraction of a cell.  Returns
// [0,1] coverage with proper per-axis anti-aliasing and sub-pixel fade.
float pristineGrid(vec2 uv, vec2 line_width) {
    // Per-axis screen-space rate of change of uv (≈ how many cells one
    // pixel spans on each axis).  Using length() of the full gradient is
    // what makes this stable at grazing angles.
    vec2 ddx = dFdx(uv);
    vec2 ddy = dFdy(uv);
    vec2 uv_deriv = vec2(length(vec2(ddx.x, ddy.x)),
                         length(vec2(ddx.y, ddy.y)));

    vec2 target_width = line_width;
    // Never draw a line thinner than one pixel — clamp the draw width up to
    // the derivative.  Cap at half a cell so lines can't fill the whole cell.
    vec2 draw_width = clamp(target_width, uv_deriv, vec2(0.5));
    // ~2.5 px feather: wider than the classic 1.5 so the edge gradient spans
    // enough pixels to read as genuinely smooth rather than a 1-px stipple.
    vec2 line_aa    = uv_deriv * 2.5;

    vec2 grid_uv = 1.0 - abs(fract(uv) * 2.0 - 1.0);   // 0 on the line, →1 between
    // Well-defined edge order (edge0 < edge1); the 1.0 - … inverts it so
    // coverage is high when grid_uv is small (i.e. on a line).
    vec2 g2 = 1.0 - smoothstep(draw_width - line_aa, draw_width + line_aa, grid_uv);

    // Fade lines that the clamp forced wider than intended (i.e. sub-pixel
    // lines) so distant rows fade out instead of aliasing.
    g2 *= clamp(target_width / draw_width, 0.0, 1.0);
    // Once a pixel spans more than ~one cell, collapse to the average tint.
    g2 = mix(g2, target_width, clamp(uv_deriv * 2.0 - 1.0, 0.0, 1.0));

    // Combine the two axes (a fragment on either an X-line or a Z-line lights).
    return mix(g2.x, 1.0, g2.y);
}

// Coverage of a single axis line sitting at world coordinate == 0.
float axisLine(float coord, float half_width_world) {
    float d  = fwidth(coord);
    float w  = max(half_width_world, d);   // at least ~one pixel wide
    float aa = d * 2.5;                    // match the grid's wide feather
    return 1.0 - smoothstep(w, w + aa, abs(coord));
}

void main() {
    vec2 w = v_world.xz;

    const vec3 kMinorColor = vec3(0.20, 0.20, 0.22);
    const vec3 kMajorColor = vec3(0.45, 0.45, 0.50);
    const vec3 kAxisXColor = vec3(0.85, 0.18, 0.18);  // z = 0 line (runs along X)
    const vec3 kAxisZColor = vec3(0.18, 0.34, 0.90);  // x = 0 line (runs along Z)

    // Line widths as a fraction of their cell (1 m and 10 m respectively).
    // A touch thicker than minimal: ~1.5-2 px lines with the wide feather
    // read far smoother than razor-thin 1 px ones.
    float minorC = pristineGrid(w,        vec2(0.020));
    float majorC = pristineGrid(w / 10.0, vec2(0.012));

    float axisX = axisLine(w.y, 0.04);   // w.y == world Z -> X axis (red)
    float axisZ = axisLine(w.x, 0.04);   // w.x == world X -> Z axis (blue)

    // Composite: minor under major under the coloured axes.
    vec3  col = kMinorColor;
    float a   = minorC;

    col = mix(col, kMajorColor, majorC);
    a   = max(a, majorC);

    col = mix(col, kAxisZColor, axisZ);
    a   = max(a, axisZ);
    col = mix(col, kAxisXColor, axisX);
    a   = max(a, axisX);

    // Gentle camera-distance fade so the ±100 m plane edge dissolves instead
    // of ending in a hard rectangle.  (The pristine-grid sub-pixel fade above
    // already handles the far-field aliasing.)
    float cam_dist = length(w - camera_info.position.xz);
    float fade = 1.0 - smoothstep(70.0, 100.0, cam_dist);

    a *= fade;
    if (a < 0.002) {
        discard;
    }

    // Perceptual alpha shaping: this pass alpha-blends into the linear HDR
    // buffer, which is tone-mapped + gamma-encoded later.  A linear coverage
    // ramp therefore gets compressed on its dark side and the edge reads as
    // a hard 1-px step ("crunchy").  Lifting the ramp by ~1/2.2 makes the
    // falloff perceptually even after the display transform — this is the
    // single biggest contributor to the lines reading as smooth.
    a = pow(a, 1.0 / 2.2);

    // Overall grid opacity: applied AFTER the perceptual shaping so it acts
    // as a clean 50 % cap without distorting the AA ramp.
    const float kGridOpacity = 0.5;
    a *= kGridOpacity;

    outColor = vec4(linearTosRGB(col), a);
}
