#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "tonemap.glsl.h"

// Colour arrives as a varying now, not a push constant: the whole
// population is drawn in ONE instanced call, so per-part data has to
// ride the vertex stream.  See citizen.vert.
layout(location = 0) in vec3 in_normal_ws;
layout(location = 1) in vec3 in_position_ws;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec4 outColor;

// Camera UBO (the vertex stage binds the same set): only the exposure
// scale is read here, so citizens follow the Camera & Lens exposure.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

// Simple wrapped-lambert + ambient: citizens are gameplay markers first
// and PBR objects second, so a stable readable shade beats a full BRDF.
// Finished through the shared scene tonemap so they sit in the same
// exposure as the world around them.
const vec3 kSunDir = normalize(vec3(-0.62, 0.62, -0.48));

void main() {
    vec3 n = normalize(in_normal_ws);
    float nl = dot(n, kSunDir) * 0.5 + 0.5;          // wrapped
    vec3 lit = in_color.rgb * (0.35 + 0.85 * nl);
    lit += in_color.rgb * in_color.a;         // readability lift
    outColor = vec4(sceneTonemapExposed(lit,
        sceneExposureScaleOf(camera_info.exposure_scale)), 1.0);
}
