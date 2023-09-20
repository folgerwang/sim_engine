#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PBR_CONSTANT_INDEX) uniform MaterialUniformBufferObject {
    PbrMaterialParams in_material;
};
#endif

#include "ibl.glsl.h"

layout(location = 0) in PbrVsPsData ps_in_data;

layout(location = 0) out vec4 outColor;

#include "pbr_lighting.glsl.h"

void main() {
#ifndef NO_MTL
    vec4 baseColor = getBaseColor(ps_in_data, in_material);
#else
    vec4 baseColor = vec4(0);
#endif
    

#ifndef NO_MTL
#ifdef ALPHAMODE_OPAQUE
    baseColor.a = 1.0;
#endif // ALPHAMODE_OPAQUE

#ifdef MATERIAL_UNLIT
    outColor = (vec4(linearTosRGB(baseColor.rgb), baseColor.a));
    return;
#endif // MATERIAL_UNLIT
    vec3 v = normalize(camera_info.position.xyz - ps_in_data.vertex_position);
    NormalInfo normal_info = getNormalInfo(ps_in_data, in_material, v);

    MaterialInfo material_info =
        setupMaterialInfo(
            ps_in_data,
            in_material,
            normal_info,
            v,
            baseColor.xyz);

    // LIGHTING
    PbrLightsColorInfo color_info = initColorInfo();

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    iblLighting(
        color_info,
        in_material,
        material_info,
        normal_info, v);
#endif // USE_IBL

	// Calculate lighting contribution from punctual light sources
#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        punctualLighting(
            color_info,
            ps_in_data,
            in_material,
            material_info,
            in_material.lights[i],
            normal_info,
            v);
#endif // !USE_PUNCTUAL

    layerBlending(
        color_info,
        ps_in_data,
        in_material,
        material_info,
        normal_info,
        v);

    vec3 color =
        getFinalColor(
            color_info,
            ps_in_data,
            in_material,
            material_info,
            v);

#ifdef ALPHAMODE_MASK
    // Late discard to avaoid samplig artifacts. See https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
    if(baseColor.a < in_material.alpha_cutoff)
    {
        discard;
    }
    baseColor.a = 1.0;
#endif // ALPHAMODE_MASK

    // regular shading
    outColor = vec4(toneMap(in_material, color), baseColor.a);
#else
    outColor = baseColor;
#endif // NO_MTL
}