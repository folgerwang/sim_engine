#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#include "ibl.glsl.h"
#include "tile_common.glsl.h"

layout(push_constant) uniform DebugDrawUniformBufferObject {
    DebugDrawParams params;
};

layout(location = 0) in VsPsData {
    vec4 debug_info;
} in_data;

layout(location = 0) out vec4 outColor;

void main() {

    vec3 color = vec3(0);
    if (params.debug_type == DEBUG_DRAW_TEMPRETURE) {
        color = mix(vec3(0, 0, 1), vec3(1, 0, 0), clamp((in_data.debug_info.x - 0.0f) / 30.0f, 0, 1));
    }
    else {
        color = vec3(in_data.debug_info.y);
    }

    float alpha = color.x;
    outColor = vec4(linearTosRGB(color), alpha);
}