#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#define ALPHAMODE_MASK 1

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PBR_CONSTANT_INDEX) uniform MaterialUniformBufferObject {
    PbrMaterialParams material;
};
#endif

#include "ibl.glsl.h"

layout(location = 0) in ObjectVsPsData ps_in_data;

#include "pbr_lighting.glsl.h"

void main() {
#ifndef NO_MTL
    vec4 baseColor = getBaseColor(ps_in_data, material);
#ifdef ALPHAMODE_OPAQUE
    baseColor.a = 1.0;
#endif // ALPHAMODE_OPAQUE
#ifdef ALPHAMODE_MASK
    // Late discard to avoid samplig artifacts. See https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
    if(baseColor.a < material.alpha_cutoff)
    {
        discard;
    }
#endif // ALPHAMODE_MASK
#endif
}