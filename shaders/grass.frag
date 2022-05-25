#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0, 1, 0, 1.0);
}