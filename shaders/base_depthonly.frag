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

// NOTE: An alpha-only R8 companion sampler is bound at
// ALPHA_ONLY_TEX_INDEX (see drawable_object.cpp), but we don't sample
// it here.  Reason: it can only be built when the source albedo has a
// CPU-side decoded pixel array (cpu_pixels), and the stb-loaded PNG/JPG
// path is the only one that fills cpu_pixels.  Bistro and most pro
// asset packs ship DDS / BC-compressed textures whose decode happens
// on the GPU only — cpu_pixels is null for those, no companion is
// built, and the binding becomes a 1×1 white fallback.  Sampling that
// would never discard, and cutout foliage would cast solid-quad
// shadows.
//
// Sampling getBaseColor(...).a instead pulls the alpha from the same
// (potentially compressed) GPU texture the forward pass uses — works
// uniformly for all source formats, at the cost of 4× more texture
// bandwidth than a dedicated R8 sample.  When DDS-aware companion
// extraction lands, this shader can switch to sampling the R8 view.

void main() {
#ifndef NO_MTL
    vec4 baseColor = getBaseColor(ps_in_data, material);
#ifdef ALPHAMODE_OPAQUE
    baseColor.a = 1.0;
#endif // ALPHAMODE_OPAQUE
#ifdef ALPHAMODE_MASK
    // Late discard (after the texture sample is complete) to avoid
    // sampling artifacts at neighbouring fragments.  See
    // https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
    if (baseColor.a < material.alpha_cutoff) {
        discard;
    }
#endif // ALPHAMODE_MASK
#endif
}