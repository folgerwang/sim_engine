#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"
#include "prt_core.glsl.h"

layout(push_constant) uniform PrtLightUniformBufferObject {
    PrtLightParams params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = 0) in VsPsData {
    vec3 vertex_position;
    vec2 vertex_tex_coord;
    vec3 vertex_normal;
    vec3 vertex_tangent;
} in_data;

layout(set = PBR_MATERIAL_PARAMS_SET, binding = BASE_COLOR_TEX_INDEX) uniform sampler2D base_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = NORMAL_TEX_INDEX) uniform sampler2D normal_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = METAL_ROUGHNESS_TEX_INDEX) uniform sampler2D orh_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = CONEMAP_TEX_INDEX) uniform sampler2D conemap_tex;
/*
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_0) uniform sampler2D prt_tex_0;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_1) uniform sampler2D prt_tex_1;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_2) uniform sampler2D prt_tex_2;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_3) uniform sampler2D prt_tex_3;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_4) uniform sampler2D prt_tex_4;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_5) uniform sampler2D prt_tex_5;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_6) uniform sampler2D prt_tex_6;*/

layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX, rgba32ui) uniform readonly uimage2D src_packed_img;
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = PRT_BUFFER_INDEX) readonly buffer PrtMinmaxBuffer {
	PrtMinmaxInfo info;
};

const int s_cone_steps = 15;
const int s_binary_steps = 8;

vec3 relaxedConeStepping(vec3 iv, vec3 ip, bool use_conserve_conemap)
{
    vec3 v = iv / iv.z;
    vec3 p0 = ip;

    float scale_x = 1.0f;
    if (v.x > 0.0f && p0.x + v.x > 1.0f)
    {
        scale_x = (1.0f - p0.x) / v.x;
    }

    if (v.x < 0.0f && p0.x + v.x < 0.0f)
    {
        scale_x = p0.x / (-v.x);
    }

    vec3 clamped_v = v * scale_x;

    float scale_y = 1.0f;
    if (clamped_v.y > 0.0f && p0.y + clamped_v.y > 1.0f)
    {
        scale_y = (1.0f - p0.y) / clamped_v.y;

    }

    if (clamped_v.y < 0.0f && p0.y + clamped_v.y < 0.0f)
    {
        scale_y = p0.y / (-clamped_v.y);
    }

    clamped_v = clamped_v * scale_y;

    float dist = length(vec2(v));

    vec4 relief_map_info = texture(conemap_tex, vec2(p0));
    float height = clamp(relief_map_info.z - p0.z, 0.0f, 1.0f);

    const float half_pi = PI / 2.0f;
    float tan_cone_angle = tan(relief_map_info.y * half_pi);
    float start_z = !use_conserve_conemap ? 0.0f : min(height / (dist * tan_cone_angle + 1.0f), clamped_v.z);
    float cast_z = start_z;

    vec3 p = p0;
    for (int i = 0; i < s_cone_steps; i++)
    {
        p = p0 + v * cast_z;
        vec4 relief_map_info = texture(conemap_tex, vec2(p));

        //The use of the saturate() function when calculating the distance to move guarantees that we stop on the first visited texel for which the viewing ray is under the relief surface.
        float height = clamp(relief_map_info.z - p.z, 0.0f, 1.0f);

        float tan_cone_angle = tan(relief_map_info.x * half_pi);
        cast_z = min(cast_z + height / (dist * tan_cone_angle + 1.0f), clamped_v.z);
    }

    float step_z = (cast_z - start_z) * 0.5f;
    float current_z = start_z + step_z;

    for (int i = 0; i < s_binary_steps; i++)
    {
        p = p0 + v * current_z;
        vec4 relief_map_info = texture(conemap_tex, vec2(p));
        step_z *= 0.5f;
        if (p.z < relief_map_info.z)
            current_z += step_z;
        else
            current_z -= step_z;
    }

    return p;
}

layout(location = 0) out vec4 outColor;

void main() {
    vec3 binormal = cross(in_data.vertex_normal, in_data.vertex_tangent);
    mat3 world2local = mat3(in_data.vertex_tangent, binormal, in_data.vertex_normal);

    // Run the relaxed cone stepping to find the new relief UV
    vec3 v = in_data.vertex_position - camera_info.position.xyz;
    v = world2local * v;
    v.z = abs(v.z);
    v.xy *= params.height_scale;

    vec3 intersect_pos = relaxedConeStepping(v, vec3(in_data.vertex_tex_coord, 0.0), false);

    vec2 pixel_coords = clamp(vec2(intersect_pos), 0.0f, 1.0f) * (params.buffer_size - 1);
    uvec4 prt_packed_info = imageLoad(src_packed_img, ivec2(pixel_coords));

    vec4 coeffs[6];
    float coeffs_6;
    coeffs[0].x = ((prt_packed_info.x >> 30) | ((prt_packed_info.y >> 30) << 2) | ((prt_packed_info.z >> 30) << 4) | ((prt_packed_info.w >> 30) << 6)) / 255.0f * info.prt_minmax[0].y + info.prt_minmax[0].x;
    coeffs[0].y = (prt_packed_info.x & 0x1f) / 31.0f * info.prt_minmax[1].y + info.prt_minmax[1].x;
    coeffs[0].z = ((prt_packed_info.x >> 5) & 0x1f) / 31.0f * info.prt_minmax[2].y + info.prt_minmax[2].x;
    coeffs[0].w = ((prt_packed_info.x >> 10) & 0x1f) / 31.0f * info.prt_minmax[3].y + info.prt_minmax[3].x;
    coeffs[1].x = ((prt_packed_info.x >> 15) & 0x1f) / 31.0f * info.prt_minmax[4].y + info.prt_minmax[4].x;
    coeffs[1].y = ((prt_packed_info.x >> 20) & 0x1f) / 31.0f * info.prt_minmax[5].y + info.prt_minmax[5].x;
    coeffs[1].z = ((prt_packed_info.x >> 25) & 0x1f) / 31.0f * info.prt_minmax[6].y + info.prt_minmax[6].x;
    coeffs[1].w = (prt_packed_info.y & 0x1f) / 31.0f * info.prt_minmax[7].y + info.prt_minmax[7].x;
    coeffs[2].x = ((prt_packed_info.y >> 5) & 0x1f) / 31.0f * info.prt_minmax[8].y + info.prt_minmax[8].x;
    coeffs[2].y = ((prt_packed_info.y >> 10) & 0x1f) / 31.0f * info.prt_minmax[9].y + info.prt_minmax[9].x;
    coeffs[2].z = ((prt_packed_info.y >> 15) & 0x1f) / 31.0f * info.prt_minmax[10].y + info.prt_minmax[10].x;
    coeffs[2].w = ((prt_packed_info.y >> 20) & 0x1f) / 31.0f * info.prt_minmax[11].y + info.prt_minmax[11].x;
    coeffs[3].x = ((prt_packed_info.y >> 25) & 0x1f) / 31.0f * info.prt_minmax[12].y + info.prt_minmax[12].x;
    coeffs[3].y = (prt_packed_info.z & 0x1f) / 31.0f * info.prt_minmax[13].y + info.prt_minmax[13].x;
    coeffs[3].z = ((prt_packed_info.z >> 5) & 0x1f) / 31.0f * info.prt_minmax[14].y + info.prt_minmax[14].x;
    coeffs[3].w = ((prt_packed_info.z >> 10) & 0x1f) / 31.0f * info.prt_minmax[15].y + info.prt_minmax[15].x;
    coeffs[4].x = ((prt_packed_info.z >> 15) & 0x1f) / 31.0f * info.prt_minmax[16].y + info.prt_minmax[16].x;
    coeffs[4].y = ((prt_packed_info.z >> 20) & 0x1f) / 31.0f * info.prt_minmax[17].y + info.prt_minmax[17].x;
    coeffs[4].z = ((prt_packed_info.z >> 25) & 0x1f) / 31.0f * info.prt_minmax[18].y + info.prt_minmax[18].x;
    coeffs[4].w = (prt_packed_info.w & 0x1f) / 31.0f * info.prt_minmax[19].y + info.prt_minmax[19].x;
    coeffs[5].x = ((prt_packed_info.w >> 5) & 0x1f) / 31.0f * info.prt_minmax[20].y + info.prt_minmax[20].x;
    coeffs[5].y = ((prt_packed_info.w >> 10) & 0x1f) / 31.0f * info.prt_minmax[21].y + info.prt_minmax[21].x;
    coeffs[5].z = ((prt_packed_info.w >> 15) & 0x1f) / 31.0f * info.prt_minmax[22].y + info.prt_minmax[22].x;
    coeffs[5].w = ((prt_packed_info.w >> 20) & 0x1f) / 31.0f * info.prt_minmax[23].y + info.prt_minmax[23].x;
    coeffs_6 = ((prt_packed_info.w >> 25) & 0x1f) / 31.0f * info.prt_minmax[24].y + info.prt_minmax[24].x;

    /*coeffs[0] = texture(prt_tex_0, vec2(intersect_pos));
    coeffs[1] = texture(prt_tex_1, vec2(intersect_pos));
    coeffs[2] = texture(prt_tex_2, vec2(intersect_pos));
    coeffs[3] = texture(prt_tex_3, vec2(intersect_pos));
    coeffs[4] = texture(prt_tex_4, vec2(intersect_pos));
    coeffs[5] = texture(prt_tex_5, vec2(intersect_pos));
    coeffs_6 = texture(prt_tex_6, vec2(intersect_pos)).x;*/

    float sum_visi = 0;
    sum_visi += dot(coeffs[0], vec4(params.coeffs[0], params.coeffs[1], params.coeffs[2], params.coeffs[3]));
    sum_visi += dot(coeffs[1], vec4(params.coeffs[4], params.coeffs[5], params.coeffs[6], params.coeffs[7]));
    sum_visi += dot(coeffs[2], vec4(params.coeffs[8], params.coeffs[9], params.coeffs[10], params.coeffs[11]));
    sum_visi += dot(coeffs[3], vec4(params.coeffs[12], params.coeffs[13], params.coeffs[14], params.coeffs[15]));
    sum_visi += dot(coeffs[4], vec4(params.coeffs[16], params.coeffs[17], params.coeffs[18], params.coeffs[19]));
    sum_visi += dot(coeffs[5], vec4(params.coeffs[20], params.coeffs[21], params.coeffs[22], params.coeffs[23]));
    sum_visi += coeffs_6 * params.coeffs[24];

    vec4 base_color = vec4(texture(base_tex, vec2(intersect_pos)).xyz, 1);
    outColor = vec4(base_color.xyz * vec3(sum_visi) * sqrt(4.0f * PI) * 2.0f, 1.0f);
}