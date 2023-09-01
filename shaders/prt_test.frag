#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"
#include "prt_core.glsl.h"

layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
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

layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_BASE_TEX_INDEX) uniform sampler2D prt_base_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_BUMP_TEX_INDEX) uniform sampler2D prt_bump_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_CONEMAP_TEX_INDEX) uniform sampler2D prt_conemap_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_0) uniform sampler2D prt_tex_0;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_1) uniform sampler2D prt_tex_1;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_2) uniform sampler2D prt_tex_2;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_3) uniform sampler2D prt_tex_3;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_4) uniform sampler2D prt_tex_4;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_5) uniform sampler2D prt_tex_5;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_TEX_INDEX_6) uniform sampler2D prt_tex_6;

const int s_cone_steps = 15;
const int s_binary_steps = 8;
const float s_depth_scale = 20.0f / 255.0f;

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

    vec4 relief_map_info = texture(prt_conemap_tex, vec2(p0));
    float height = clamp(relief_map_info.z - p0.z, 0.0f, 1.0f);

    const float half_pi = PI / 2.0f;
    float tan_cone_angle = tan(relief_map_info.y * half_pi);
    float start_z = !use_conserve_conemap ? 0.0f : min(height / (dist * tan_cone_angle + 1.0f), clamped_v.z);
    float cast_z = start_z;

    vec3 p = p0;
    for (int i = 0; i < s_cone_steps; i++)
    {
        p = p0 + v * cast_z;
        vec4 relief_map_info = texture(prt_conemap_tex, vec2(p));

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
        vec4 relief_map_info = texture(prt_conemap_tex, vec2(p));
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

    //Run the relaxed cone stepping to find the new relief UV
    vec3 v = in_data.vertex_position - camera_info.position.xyz;
    v = world2local * v;
    v.z = abs(v.z);
    v.xy *= s_depth_scale;

    float y_value[25];
    fillYVauleTablle(y_value, PI / 4.0f, PI / 4.0f);

    vec3 intersect_pos = relaxedConeStepping(v, vec3(in_data.vertex_tex_coord, 0.0), false);

    vec4 prt_0 = texture(prt_tex_0, vec2(intersect_pos));
    vec4 prt_1 = texture(prt_tex_1, vec2(intersect_pos));
    vec4 prt_2 = texture(prt_tex_2, vec2(intersect_pos));
    vec4 prt_3 = texture(prt_tex_3, vec2(intersect_pos));
    vec4 prt_4 = texture(prt_tex_4, vec2(intersect_pos));
    vec4 prt_5 = texture(prt_tex_5, vec2(intersect_pos));
    float prt_6 = texture(prt_tex_6, vec2(intersect_pos)).x;

    float sum_visi = 0;
    sum_visi += dot(prt_0, vec4(y_value[0], y_value[1], y_value[2], y_value[3]));
    sum_visi += dot(prt_1, vec4(y_value[4], y_value[5], y_value[6], y_value[7]));
    sum_visi += dot(prt_2, vec4(y_value[8], y_value[9], y_value[10], y_value[11]));
    sum_visi += dot(prt_3, vec4(y_value[12], y_value[13], y_value[14], y_value[15]));
    sum_visi += dot(prt_4, vec4(y_value[16], y_value[17], y_value[18], y_value[19]));
    sum_visi += dot(prt_5, vec4(y_value[20], y_value[21], y_value[22], y_value[23]));
    sum_visi += prt_6 * y_value[24];

    vec4 base_color = vec4(texture(prt_base_tex, vec2(intersect_pos)).xyz, 1);
    outColor = vec4(base_color.xyz * sum_visi, 1.0f);
}