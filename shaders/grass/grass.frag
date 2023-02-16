#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"
#include "..\functions.glsl.h"
#include "..\brdf.glsl.h"
#include "..\punctual.glsl.h"

layout(location = 0) out vec4 outColor;

layout(location = 0) in VsPsData {
    vec2 tex_coord;
} in_data;

void main() {
    float a = pow(max((abs(in_data.tex_coord.x - 0.5) - 0.003), 0.0), 0.05);
    outColor = vec4(vec3(a * 0.1 + 0.9) * vec3(0.5, 1.0, 0.45), 1.0);
}