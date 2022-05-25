#version 450
#extension GL_ARB_separate_shader_objects : enable

#define RENDER_SUN 1
#define RENDER_SUNLIGHT_SCATTERING  1
const int iSteps = 32;
const int jSteps = 8;

#include "global_definition.glsl.h"
#include "sun.glsl.h"
#include "sunlight_scattering.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = 0) in VsPsData {
    vec3 vertex_position;
} in_data;

layout(push_constant) uniform SunSkyUniformBufferObject {
    SunSkyParams params;
};

layout(location = 0) out vec4 outColor;

void main() {
    vec3 view_dir = normalize(in_data.vertex_position - camera_info.position.xyz);

    vec3 color = vec3(0.0, 0.0, 0.0);
    vec3 sun_pos = normalize(params.sun_pos);

#if RENDER_SUNLIGHT_SCATTERING
    vec3 r = view_dir;
    vec3 r0 = vec3(0, kPlanetRadius, 0) + camera_info.position.xyz;
    float g = params.g;
    float cast_range = rsi(r0, r, kAtmosphereRadius);
#if ATMOSPHERE_USE_LUT
    color += atmosphereLut(
        r,                               // normalized ray direction
        r0,                              // ray origin
        cast_range,
        sun_pos,                        // position of the sun
        22.0,                           // intensity of the sun
        kPlanetRadius,                  // radius of the planet in meters
        kAtmosphereRadius,              // radius of the atmosphere in meters
        vec3(5.5e-6, 13.0e-6, 22.4e-6), // Rayleigh scattering coefficient
        21e-6,                          // Mie scattering coefficient
        params.inv_rayleigh_scale_height,           // Rayleigh scale height
        params.inv_mie_scale_height,                // Mie scale height
        g,                              // Mie preferred scattering direction
        vec2(1.0f, 1.0f),
        iSteps).xyz;
#else
    color += atmosphere(
        r,                              // normalized ray direction
        r0,                             // ray origin
        cast_range,
        sun_pos,                        // position of the sun
        22.0,                           // intensity of the sun
        kPlanetRadius,                  // radius of the planet in meters
        kAtmosphereRadius,              // radius of the atmosphere in meters
        vec3(5.5e-6, 13.0e-6, 22.4e-6), // Rayleigh scattering coefficient
        21e-6,                          // Mie scattering coefficient
        params.inv_rayleigh_scale_height,           // Rayleigh scale height
        params.inv_mie_scale_height,                // Mie scale height
        g                               // Mie preferred scattering direction
        );
#endif
#endif

#if RENDER_SUN
  color += renderSun(view_dir, sun_pos);
#endif


    outColor = vec4(color/*textureLod(skybox_tex, view_dir, 0).xyz*/, 1.0);
}