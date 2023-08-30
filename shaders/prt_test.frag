#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = 0) in VsPsData {
    vec3 vertex_position;
    vec2 vertex_tex_coord;
    vec3 vertex_normal;
} in_data;

layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_BASE_TEX_INDEX) uniform sampler2D prt_base_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_BUMP_TEX_INDEX) uniform sampler2D prt_bump_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_CONEMAP_TEX_INDEX) uniform sampler2D prt_conemap_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_PRT_TEX_INDEX) uniform sampler2D prt_prt_tex;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(texture(prt_bump_tex, in_data.vertex_tex_coord).aaa, 1);
}