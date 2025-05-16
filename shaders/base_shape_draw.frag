#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"
#include "ibl.glsl.h"

layout(push_constant) uniform BaseShapeDrawUniformBufferObject {
    BaseShapeDrawParams params;
};

layout(location = 0) in BaseShapeVsPsData in_data;

layout(location = 0) out vec4 outColor;

void main() {

    vec3 color = vec3(0);

    float alpha = color.x;
    outColor = vec4(linearTosRGB(color), alpha);
}