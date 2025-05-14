#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#define ALPHAMODE_MASK 1

#define DEBUG_BASE_COLOR 0
#define DEBUG_MIP_LEVEL 0

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PBR_CONSTANT_INDEX) uniform MaterialUniformBufferObject {
    PbrMaterialParams material;
};
#endif

layout(set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX) uniform RuntimeLightsUniformBufferObject {
    RuntimeLightsParams runtime_lights;
};

#include "ibl.glsl.h"

layout(location = 0) in ObjectVsPsData ps_in_data;

layout(location = 0) out vec4 outColor;

#include "pbr_lighting.glsl.h"

const float SHADOW_BIAS = 0.001;
float calculateShadowFactor(vec3 position_world) {
    vec4 position_light_clip = runtime_lights.light_view_proj * vec4(position_world, 1.0);

    vec3 position_light_NDC = position_light_clip.xyz / position_light_clip.w;

    vec2 shadow_map_texcoord = position_light_NDC.xy * 0.5f + 0.5f;
    shadow_map_texcoord.y = shadow_map_texcoord.y;

    float current_depth = position_light_NDC.z;

    float closest_depth = texture(direct_shadow_sampler, shadow_map_texcoord).r;

    float shadow = 1.0;
    if (closest_depth < 1.0 && current_depth > closest_depth + SHADOW_BIAS) {
        shadow = 0.0;
    }

    return shadow;
}

void main() {
    bool is_front_face = gl_FrontFacing;
#ifndef NO_MTL
    vec4 baseColor = getBaseColor(ps_in_data, material);
#ifdef ALPHAMODE_OPAQUE
    baseColor.a = 1.0;
#endif // ALPHAMODE_OPAQUE
#else
    vec4 baseColor = vec4(0);
#endif
    

#ifndef NO_MTL
#ifdef MATERIAL_UNLIT
    outColor = (vec4(linearTosRGB(baseColor.rgb), baseColor.a));
    return;
#endif // MATERIAL_UNLIT
    vec3 v = normalize(camera_info.position.xyz - ps_in_data.vertex_position);
    NormalInfo normal_info = getNormalInfo(ps_in_data, material, v, is_front_face);

    MaterialInfo material_info =
        setupMaterialInfo(
            ps_in_data,
            material,
            normal_info,
            v,
            baseColor.xyz);

    float shadow = calculateShadowFactor(ps_in_data.vertex_position);

#ifdef DOUBLE_SIDED
    // LIGHTING
    PbrLightsColorInfo back_color_info = initColorInfo();
    NormalInfo back_normal_info = normal_info;
    back_normal_info.ng = -normal_info.ng;
    back_normal_info.n = -normal_info.n;

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    iblLighting(
        back_color_info,
        material,
        material_info,
        back_normal_info, v);
#endif // USE_IBL

	// Calculate lighting contribution from punctual light sources
#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        punctualLighting(
            back_color_info,
            ps_in_data,
            material,
            material_info,
            runtime_lights.lights[i],
            back_normal_info,
            v,
            shadow);
    }
#endif // !USE_PUNCTUAL
#endif

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

	// Calculate lighting contribution from punctual light sources
#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        punctualLighting(
            color_info,
            ps_in_data,
            material,
            material_info,
            runtime_lights.lights[i],
            normal_info,
            v,
            shadow);
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
            v,
            1.0f);


#ifdef ALPHAMODE_MASK
    // Late discard to avoid samplig artifacts. See https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
    if(baseColor.a < material.alpha_cutoff)
    {
        discard;
    }
    baseColor.a = 1.0;
#endif // ALPHAMODE_MASK

    // regular shading
    outColor = vec4(toneMap(material, color), baseColor.a);
#else
    outColor = baseColor;
#endif // NO_MTL

#if DEBUG_BASE_COLOR
    outColor.xyz = baseColor.xyz;
#endif

#if DEBUG_MIP_LEVEL
#ifndef NO_MTL// Debugging: show mip level
    float lod = textureQueryLod(albedo_tex, ps_in_data.vertex_tex_coord.xy).x;
    outColor.xyz = vec3(lod / 10.0f);
#endif
#endif
}