#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(location = 0) out vec4 outColor;

layout(location = 0) in HairVsPsData ps_in_data;

void main() {
    const float num_hairs = 50.0f;
    const float hair_width = 0.1f;
    const float inv_hair_width = 1.0f / hair_width;

// layer 0
    float per_hair_0 = fract(ps_in_data.vertex_tex_coord.x * num_hairs);
    float layer_hair_color_0 = clamp(1.0f - abs(per_hair_0 - 0.5f) * inv_hair_width, 0.0f, 1.0f);

// layer 1
    float per_hair_1 = fract(ps_in_data.vertex_tex_coord.x * num_hairs + 0.5f);
    float layer_hair_color_1 = clamp(1.0f - abs(per_hair_1 - 0.5f) * inv_hair_width, 0.0f, 1.0f);

// layer 2
    float per_hair_2 = fract(ps_in_data.vertex_tex_coord.x * num_hairs + 1.0f);
    float layer_hair_color_2 = clamp(1.0f - abs(per_hair_2 - 0.5f) * inv_hair_width, 0.0f, 1.0f);

    float hair_color = layer_hair_color_0 + layer_hair_color_1 + layer_hair_color_2;
    outColor = vec4(vec3(hair_color), 0.0f);

    if (hair_color < 0.01f)
        discard;
}