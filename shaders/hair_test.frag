#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(location = 0) out vec4 outColor;

layout(location = 0) in HairVsPsData ps_in_data;

void main() {
    float hair = (fract(ps_in_data.vertex_tex_coord.x * 500.0f) - 0.9f) * 10.0f;
    outColor = vec4(vec3(hair), 0.0f);
}