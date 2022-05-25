#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "weather_common.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#include "ibl.glsl.h"
#include "tile_common.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) in VsPsData {
    vec3 vertex_position;
    vec2 world_map_uv;
    vec3 test_color;
} in_data;

layout(location = 0) out vec4 outColor;

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

layout(set = TILE_PARAMS_SET, binding = SRC_COLOR_TEX_INDEX) uniform sampler2D src_tex;
layout(set = TILE_PARAMS_SET, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;
layout(set = TILE_PARAMS_SET, binding = SRC_MAP_MASK_INDEX) uniform sampler2D src_map_mask;
layout(set = TILE_PARAMS_SET, binding = SRC_TEMP_TEX_INDEX) uniform sampler3D src_temp;
layout(set = TILE_PARAMS_SET, binding = DETAIL_NOISE_TEXTURE_INDEX) uniform sampler3D src_detail_noise_tex;
layout(set = TILE_PARAMS_SET, binding = ROUGH_NOISE_TEXTURE_INDEX) uniform sampler3D src_rough_noise_tex;

struct MaterialInfo
{
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    vec3 f0;                        // full reflectance color (n incidence angle)

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])
    vec3 albedoColor;

    vec3 f90;                       // reflectance color at grazing angle
    float metallic;

    vec3 n;
    vec3 baseColor; // getBaseColor()

    float sheenIntensity;
    vec3 sheenColor;
    float sheenRoughness;

    float anisotropy;

    vec3 clearcoatF0;
    vec3 clearcoatF90;
    float clearcoatFactor;
    vec3 clearcoatNormal;
    float clearcoatRoughness;

    float subsurfaceScale;
    float subsurfaceDistortion;
    float subsurfacePower;
    vec3 subsurfaceColor;
    float subsurfaceThickness;

    float thinFilmFactor;
    float thinFilmThickness;

    float thickness;

    vec3 absorption;

    float transmission;
};

void main() {
    vec3 pos = in_data.vertex_position;
    vec3 tnor = terrainNormal(vec2(pos.x, pos.z), 0.0625f, 100.0f);
    // bump map
    vec4 tt = fbmd_8(pos * 0.3f * vec3(1.0f, 0.2f, 1.0f));
    vec3 normal = normalize(tnor + 0.8f*(1.0f - abs(tnor.y))*0.8f*vec3(tt.y, tt.z, tt.w));

    float uvw_y = getHeightToSample(pos.y);
    float c_temp = texture(src_temp, vec3(in_data.world_map_uv, uvw_y)).x;

#if 1
    vec3 uvw = vec3(in_data.world_map_uv, uvw_y) * 16.0f;
    ivec3 i_uvw = ivec3(uvw);
    bool show_cell = ((i_uvw.x + i_uvw.y + i_uvw.z) % 2) == 0;
    vec4 noise_value = /*(show_cell ? 1.0f : 0.0f) * */texture(src_rough_noise_tex, uvw);
#endif

    vec3 albedo = vec3(0.18, 0.11, 0.10)*.75f;
    albedo = 1.0f* mix(albedo, vec3(0.1, 0.1, 0.0)*0.2f, smoothstep(0.7f, 0.9f, normal.y));
    float cold_index = clamp((15.0f - c_temp) / 5.0f, 0.0f, 1.0f);

    // turn off right now, will fix flickering later.
    //albedo = mix(albedo, vec3(1.0, 1.0, 1.0), cold_index);

    albedo = texture(src_map_mask, in_data.world_map_uv).rgb;

    MaterialInfo material_info;
    material_info.baseColor = albedo;

    vec3 view_vec = camera_info.position.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

    vec3 f_diffuse = vec3(0);
    vec3 f_specular = vec3(0);

/*    float sha1 = 1.0f;
    float sha2 = 1.0f;

    float dif = clamp(dot(normal, kSunDir), 0.0f, 1.0f);
    dif *= sha1;
#ifndef LOWQUALITY
    dif *= sha2;
#endif

    float bac = clamp(dot(normalize(vec3(-kSunDir.x, 0.0, -kSunDir.z)), normal), 0.0f, 1.0f);
    float foc = clamp((pos.y + 100.0f) / 100.0f, 0.0f, 1.0f);
    float dom = clamp(0.5f + 0.5f*normal.y, 0.0f, 1.0f);
    vec3  lin = 1.0f*0.2f* mix(0.1f* vec3(0.1, 0.2, 0.1), vec3(0.7, 0.9, 1.5)*3.0f, dom)*foc;
    lin += 1.0f*8.5f* vec3(1.0, 0.9, 0.8)*dif;
    lin += 1.0f*0.27f* vec3(1.0)*bac*foc;

    color *= lin;*/

    float ior = 1.5;
    float f0_ior = 0.04;

    material_info.metallic = 0.3f;//material.metallic_factor;
    material_info.perceptualRoughness = 0.8f;//material.roughness_factor;

    // Achromatic f0 based on IOR.
    vec3 f0 = vec3(f0_ior);

    material_info.albedoColor = mix(material_info.baseColor.rgb * (vec3(1.0) - f0),  vec3(0), material_info.metallic);
    material_info.f0 = mix(f0, material_info.baseColor.rgb, material_info.metallic);

    #ifdef USE_IBL
    float mip_count = 10;
    f_specular += getIBLRadianceGGX(normal, view, material_info.perceptualRoughness, material_info.f0, mip_count);
    f_diffuse += getIBLRadianceLambertian(normal, material_info.albedoColor);
    #endif

    //vec3 color = vec3(noise_value.w);
    vec3 color = f_diffuse + f_specular;

    float alpha = 1.0f;
    outColor = vec4(linearTosRGB(color), alpha);
//    outColor.xyz *= in_data.test_color;
}