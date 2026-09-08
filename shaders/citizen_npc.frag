#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "tonemap.glsl.h"

// The character mesh's albedo, shaded the way citizen.frag shades the
// puppet (wrapped lambert + ambient, the readability lift, the scene
// tonemap) so a scan and a puppet side by side sit in the same light.
// The scan's texture carries its own soft shading already, so the
// direct term is a little flatter than the puppet's.
layout(location = 0) in vec3 in_normal_ws;
layout(location = 1) in vec3 in_position_ws;
layout(location = 2) in vec2 in_uv;
layout(location = 3) flat in vec4 in_inst;

#include "citizen_gbuffer.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};
layout(set = VIEW_PARAMS_SET + 1, binding = 1) uniform sampler2D albedo_tex;

const vec3 kSunDir = normalize(vec3(-0.62, 0.62, -0.48));

void main() {
    vec3 n = normalize(in_normal_ws);
    // a scan's winding is not guaranteed: light the side that faces us
    if (!gl_FrontFacing) n = -n;
    vec3 albedo = texture(albedo_tex, in_uv).rgb;
    // a touch of per-person variation, so twins in one street differ
    albedo *= 0.90 + 0.20 * in_inst.y;
#ifdef GBUFFER_OUTPUT
    if (dot(n, camera_info.position - in_position_ws) < 0.0) n = -n;
    citizenGbuffer(albedo, n, in_position_ws, camera_info.view_proj,
                   camera_info.prev_view_proj, 0.9);
#else
    float nl = dot(n, kSunDir) * 0.5 + 0.5;
    vec3 lit = albedo * (0.45 + 0.70 * nl);
    lit += albedo * in_inst.z;                // readability lift
    outColor = vec4(sceneTonemapExposed(lit,
        sceneExposureScaleOf(camera_info.exposure_scale)), 1.0);
#endif
}
