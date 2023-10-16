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

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PBR_CONSTANT_INDEX) uniform MaterialUniformBufferObject {
    PbrMaterialParams material;
};
#endif

#include "ibl.glsl.h"
#include "pbr_lighting.glsl.h"

layout(location = 0) in ObjectVsPsData in_data;

layout(set = PBR_MATERIAL_PARAMS_SET, binding = CONEMAP_TEX_INDEX) uniform sampler2D conemap_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_PACK_TEX_INDEX, rgba32ui) uniform readonly uimage2D src_prt_pack_img;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PRT_PACK_INFO_TEX_INDEX, rgba32f) uniform readonly image2D src_prt_packed_info_img;

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
    ObjectVsPsData ps_in_data = in_data;
    mat3 world2local =
        mat3(
            ps_in_data.vertex_tangent,
            ps_in_data.vertex_binormal,
            ps_in_data.vertex_normal);

    // Run the relaxed cone stepping to find the new relief UV
    vec3 v = ps_in_data.vertex_position - camera_info.position.xyz;
    v = world2local * v;
    v.z = abs(v.z);
    v.xy *= params.height_scale;

    ps_in_data.vertex_tex_coord.xy =
        relaxedConeStepping(v, vec3(ps_in_data.vertex_tex_coord.xy, 0.0), false).xy;

    vec4 baseColor = getBaseColor(ps_in_data, material);

    v = normalize(camera_info.position.xyz - ps_in_data.vertex_position);
    NormalInfo normal_info = getNormalInfo(ps_in_data, material, v);

    MaterialInfo material_info =
        setupMaterialInfo(
            ps_in_data,
            material,
            normal_info,
            v,
            baseColor.xyz);

    // LIGHTING
    PbrLightsColorInfo color_info = initColorInfo();

    // Calculate lighting contribution from image based lighting source (IBL)

#ifdef USE_IBL
    iblLighting(
        color_info,
        material,
        material_info,
        normal_info, v);
#endif // USE_IBL

    ivec2 pixel_coords =
        ivec2(clamp(ps_in_data.vertex_tex_coord.xy, 0.0f, 1.0f) * (params.buffer_size - 1));

    ivec2 pack_info_pixel_coords = ivec2(0);
/*        pixel_coords /
        ivec2(kConemapGenBlockCacheSizeX, kConemapGenBlockCacheSizeY) *
        ivec2(4);*/

    uvec4 prt_packed_info =
        imageLoad(src_prt_pack_img, pixel_coords);

    vec4 pack_info_1[6], pack_info_2[6];
    pack_info_1[0] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords);
    pack_info_2[0] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(1, 0));
    pack_info_1[1] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(2, 0));
    pack_info_2[1] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(3, 0));
    pack_info_1[2] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(0, 1));
    pack_info_2[2] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(1, 1));
    pack_info_1[3] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(2, 1));
    pack_info_2[3] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(3, 1));
    pack_info_1[4] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(0, 2));
    pack_info_2[4] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(1, 2));
    pack_info_1[5] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(2, 2));
    pack_info_2[5] = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(3, 2));
    float pack_info_1_6 = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(0, 3)).x;
    float pack_info_2_6 = imageLoad(src_prt_packed_info_img, pack_info_pixel_coords + ivec2(1, 3)).x;

    vec4 coeffs[6];
    float coeffs_6;
    coeffs[0].x =
        ((prt_packed_info.x >> 30) |
        ((prt_packed_info.y >> 30) << 2) |
        ((prt_packed_info.z >> 30) << 4) |
        ((prt_packed_info.w >> 30) << 6)) /
        255.0f * pack_info_2_6 +
        pack_info_1_6;
    coeffs[0].y = (prt_packed_info.x & 0x1f) / 31.0f * pack_info_2[0].x + pack_info_1[0].x;
    coeffs[0].z = ((prt_packed_info.x >> 5) & 0x1f) / 31.0f * pack_info_2[0].y + pack_info_1[0].y;
    coeffs[0].w = ((prt_packed_info.x >> 10) & 0x1f) / 31.0f * pack_info_2[0].z + pack_info_1[0].z;
    coeffs[1].x = ((prt_packed_info.x >> 15) & 0x1f) / 31.0f * pack_info_2[0].w + pack_info_1[0].w;
    coeffs[1].y = ((prt_packed_info.x >> 20) & 0x1f) / 31.0f * pack_info_2[1].x + pack_info_1[1].x;
    coeffs[1].z = ((prt_packed_info.x >> 25) & 0x1f) / 31.0f * pack_info_2[1].y + pack_info_1[1].y;
    coeffs[1].w = (prt_packed_info.y & 0x1f) / 31.0f * pack_info_2[1].z + pack_info_1[1].z;
    coeffs[2].x = ((prt_packed_info.y >> 5) & 0x1f) / 31.0f * pack_info_2[1].w + pack_info_1[1].w;
    coeffs[2].y = ((prt_packed_info.y >> 10) & 0x1f) / 31.0f * pack_info_2[2].x + pack_info_1[2].x;
    coeffs[2].z = ((prt_packed_info.y >> 15) & 0x1f) / 31.0f * pack_info_2[2].y + pack_info_1[2].y;
    coeffs[2].w = ((prt_packed_info.y >> 20) & 0x1f) / 31.0f * pack_info_2[2].z + pack_info_1[2].z;
    coeffs[3].x = ((prt_packed_info.y >> 25) & 0x1f) / 31.0f * pack_info_2[2].w + pack_info_1[2].w;
    coeffs[3].y = (prt_packed_info.z & 0x1f) / 31.0f * pack_info_2[3].x + pack_info_1[3].x;
    coeffs[3].z = ((prt_packed_info.z >> 5) & 0x1f) / 31.0f * pack_info_2[3].y + pack_info_1[3].y;
    coeffs[3].w = ((prt_packed_info.z >> 10) & 0x1f) / 31.0f * pack_info_2[3].z + pack_info_1[3].z;
    coeffs[4].x = ((prt_packed_info.z >> 15) & 0x1f) / 31.0f * pack_info_2[3].w + pack_info_1[3].w;
    coeffs[4].y = ((prt_packed_info.z >> 20) & 0x1f) / 31.0f * pack_info_2[4].x + pack_info_1[4].x;
    coeffs[4].z = ((prt_packed_info.z >> 25) & 0x1f) / 31.0f * pack_info_2[4].y + pack_info_1[4].y;
    coeffs[4].w = (prt_packed_info.w & 0x1f) / 31.0f * pack_info_2[4].z + pack_info_1[4].z;
    coeffs[5].x = ((prt_packed_info.w >> 5) & 0x1f) / 31.0f * pack_info_2[4].w + pack_info_1[4].w;
    coeffs[5].y = ((prt_packed_info.w >> 10) & 0x1f) / 31.0f * pack_info_2[5].x + pack_info_1[5].x;
    coeffs[5].z = ((prt_packed_info.w >> 15) & 0x1f) / 31.0f * pack_info_2[5].y + pack_info_1[5].y;
    coeffs[5].w = ((prt_packed_info.w >> 20) & 0x1f) / 31.0f * pack_info_2[5].z + pack_info_1[5].z;
    coeffs_6 = ((prt_packed_info.w >> 25) & 0x1f) / 31.0f * pack_info_2[5].w + pack_info_1[5].w;

    float sum_visi = 0;
    sum_visi += dot(coeffs[0], vec4(params.coeffs[0], params.coeffs[1], params.coeffs[2], params.coeffs[3]));
    sum_visi += dot(coeffs[1], vec4(params.coeffs[4], params.coeffs[5], params.coeffs[6], params.coeffs[7]));
    sum_visi += dot(coeffs[2], vec4(params.coeffs[8], params.coeffs[9], params.coeffs[10], params.coeffs[11]));
    sum_visi += dot(coeffs[3], vec4(params.coeffs[12], params.coeffs[13], params.coeffs[14], params.coeffs[15]));
    sum_visi += dot(coeffs[4], vec4(params.coeffs[16], params.coeffs[17], params.coeffs[18], params.coeffs[19]));
    sum_visi += dot(coeffs[5], vec4(params.coeffs[20], params.coeffs[21], params.coeffs[22], params.coeffs[23]));
    sum_visi += coeffs_6 * params.coeffs[24];

	// Calculate lighting contribution from punctual light sources
#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        punctualLighting(
            color_info,
            ps_in_data,
            material,
            material_info,
            material.lights[i],
            normal_info,
            v,
            sum_visi);
    }
#endif // !USE_PUNCTUAL

    layerBlending(
        color_info,
        ps_in_data,
        material,
        material_info,
        normal_info,
        v);

    vec3 color =
        getFinalColor(
            color_info,
            ps_in_data,
            material,
            material_info,
            v);


    //outColor = vec4(vec3(sum_visi), 1.0f);
    //outColor = vec4(color.xyz * vec3(sum_visi) * sqrt(4.0f * PI) * 2.0f, 1.0f);
    //outColor = vec4(texture(conemap_tex, ps_in_data.vertex_tex_coord.xy).xxx, 1.0f);
    outColor = vec4(toneMap(material, color), 1.0f);
}