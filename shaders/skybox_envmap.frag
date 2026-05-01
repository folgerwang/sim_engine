#version 450
#extension GL_ARB_separate_shader_objects : enable

#include "global_definition.glsl.h"

// Sky envmap background fragment shader.
//
// Reconstructs the world-space view direction for each screen pixel from the
// push-constant inv_view_proj_relative matrix (= inverse of proj * view_without_translation),
// then samples mip 0 of the pre-convolved sky envmap cubemap.
//
// This is the final display path for the sky.  The envmap is kept fresh by
// the dithered mini-buffer compute system (cube_skybox_mini.comp) which
// updates 1/64 of the cubemap each frame at full atmospheric quality.

// Envmap cubemap: the full-res sky envmap kept live by the mini-buffer system.
layout(set = 0, binding = ENVMAP_TEX_INDEX) uniform samplerCube u_envmap;

// Push constant: rotation-only inverse view-projection for direction reconstruction.
layout(push_constant) uniform SkyboxEnvmapUniformBufferObject {
    SkyboxEnvmapParams params;
};

layout(location = 0) in vec2 v_UV;
layout(location = 0) out vec4 outColor;

// DEBUG_SKY: 0=normal (tone-mapped), 1=solid red, 3=view_dir RGB,
//            4=envmap raw (no tone-map), 5=envmap×10000
#define DEBUG_SKY 4

void main()
{
    // Convert UV [0, 1] → NDC [-1, 1].
    vec2 ndc = v_UV * 2.0 - 1.0;

    // Unproject to a world-space direction vector.
    vec4 dir_h = params.inv_view_proj_relative * vec4(ndc, 1.0, 1.0);
    vec3 view_dir = normalize(dir_h.xyz / dir_h.w);

#if DEBUG_SKY == 1
    outColor = vec4(1.0, 0.0, 0.0, 1.0); return;
#elif DEBUG_SKY == 3
    outColor = vec4(view_dir * 0.5 + 0.5, 1.0); return;
#elif DEBUG_SKY == 4
    outColor = vec4(textureLod(u_envmap, view_dir, 0).rgb, 1.0); return;
#elif DEBUG_SKY == 5
    outColor = vec4(textureLod(u_envmap, view_dir, 0).rgb * 10000.0, 1.0); return;
#endif

    // Sample the live sky envmap.  The envmap stores HDR radiance (atmosphere
    // output × 2.0), so apply an exposure + Reinhard tone-map to compress it
    // into displayable [0, 1] range for all sky conditions (noon to twilight).
    //
    // Convention note: cube_skybox_mini.comp uses uvToXYZ() where face index 2
    // maps to the -Y *direction* but writes to Vulkan layer 2 = +Y cubemap face.
    // The Y-flip in the write shaders compensates for that uvToXYZ/layer mismatch,
    // so textureLod here should use the un-flipped world-space view_dir directly.
    vec3 col = textureLod(u_envmap, view_dir, 0).rgb;
    const float kExposure = 8.0;
    col *= kExposure;
    col = col / (col + vec3(1.0));   // Reinhard tone-mapping
    outColor = vec4(col, 1.0);
}
