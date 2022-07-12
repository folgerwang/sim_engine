#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(location = 0) out vec4 outColor;

layout(location = 0) in VsPsData {
    vec2 tex_coord;
} in_data;

void main() {
    outColor = vec4(in_data.tex_coord, 0, 1.0);
}