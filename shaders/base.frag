#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

#define ALPHAMODE_MASK 1

// Render-debug visualisation is now controlled at runtime via the
// FEATURE_INPUT_DEBUG_MODE bits of camera_info.input_features (set by the
// "Render Debug" combo in the menu) and dispatched at the bottom of main().
// The old compile-time DEBUG_BASE_COLOR / DEBUG_MIP_LEVEL toggles are
// removed because the runtime path covers their use cases.

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

// Push constant — shared with base.vert.  We only read debug_force_red
// here; the vertex shader is the canonical reader of the matrix +
// flip_uv_coord + cascade_idx fields.  Declaring the whole struct
// keeps the layout identical across stages so the driver doesn't
// complain about a layout mismatch.
layout(push_constant) uniform ModelUniformBufferObject {
    ModelParams model_params;
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

// PCSS soft shadow with cascade-consistent WORLD-SPACE blur radius.
// See deferred_resolve.comp for full tuning notes; constants MUST
// match across that file, cluster_bindless.frag, and base.frag.
const float CSM_NORMAL_BIAS_SCALE     = 0.05;
// Depth bias in WORLD units.  Converted per-cascade in
// calculateShadowFactor().  See deferred_resolve.comp for the full
// rationale.
const float CSM_DEPTH_BIAS_BASE_WORLD  = 0.05;
const float CSM_DEPTH_BIAS_SLOPE_WORLD = 0.20;
const float CSM_LIGHT_SIZE_WORLD      = 0.20;
const float CSM_BLOCKER_RADIUS_WORLD  = 0.40;
const float CSM_MIN_PCF_RADIUS_WORLD  = 0.02;
const float CSM_MAX_PCF_RADIUS_WORLD  = 0.80;
const int   CSM_BLOCKER_SAMPLES       = 16;
const int   CSM_PCF_SAMPLES           = 16;

vec2 csmVogelDisk(int i, int n, float phi) {
    const float GOLDEN_ANGLE = 2.39996323;
    float r     = sqrt((float(i) + 0.5) / float(n));
    float theta = GOLDEN_ANGLE * float(i) + phi;
    return vec2(r * cos(theta), r * sin(theta));
}

float csmIGN(vec2 pixel) {
    return fract(52.9829189 *
                 fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

float calculateShadowFactor(
    vec3 position_world, vec3 normal_world, vec2 screen_pixel) {
    // Cascade selection by view-space depth.
    vec4 position_view = camera_info.view * vec4(position_world, 1.0);
    float view_depth = -position_view.z;

    int cascade_idx = CSM_CASCADE_COUNT - 1;
    for (int i = 0; i < CSM_CASCADE_COUNT; ++i) {
        // cascade_splits is packed vec4[2]; index as [i/4][i%4].
        if (view_depth < runtime_lights.cascade_splits[i >> 2][i & 3]) {
            cascade_idx = i;
            break;
        }
    }

    // Normal-offset bias.
    vec3  N     = normalize(normal_world);
    vec3  L     = normalize(-runtime_lights.lights[0].direction);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    vec3  biased_world = position_world +
                         N * ((1.0 - NdotL) * CSM_NORMAL_BIAS_SCALE);

    vec4 position_light_clip =
        runtime_lights.light_view_proj[cascade_idx] * vec4(biased_world, 1.0);
    vec3 position_light_NDC = position_light_clip.xyz / position_light_clip.w;
    vec2 shadow_uv = position_light_NDC.xy * 0.5 + 0.5;
    float current_depth = position_light_NDC.z;

    // Per-cascade scale factors (see deferred_resolve.comp).
    float w2uv    = 0.5 *
        length(runtime_lights.light_view_proj[cascade_idx][0].xyz);
    float z_scale = length(runtime_lights.light_view_proj[cascade_idx][2].xyz);

    float depth_bias =
        (CSM_DEPTH_BIAS_BASE_WORLD +
         CSM_DEPTH_BIAS_SLOPE_WORLD * (1.0 - NdotL)) * z_scale;

    float blocker_radius_uv = CSM_BLOCKER_RADIUS_WORLD * w2uv;
    float light_size_uv     = CSM_LIGHT_SIZE_WORLD     * w2uv;
    float min_pcf_radius_uv = CSM_MIN_PCF_RADIUS_WORLD * w2uv;
    float max_pcf_radius_uv = CSM_MAX_PCF_RADIUS_WORLD * w2uv;

    float phi = csmIGN(screen_pixel) * 6.28318530718;

    // PCSS step 1: blocker search.
    float blocker_sum   = 0.0;
    int   blocker_count = 0;
    for (int i = 0; i < CSM_BLOCKER_SAMPLES; ++i) {
        vec2 off = csmVogelDisk(i, CSM_BLOCKER_SAMPLES, phi)
                       * blocker_radius_uv;
        float d = texture(direct_shadow_sampler,
                          vec3(shadow_uv + off, float(cascade_idx))).r;
        if (d < current_depth - depth_bias) {
            blocker_sum += d;
            ++blocker_count;
        }
    }
    if (blocker_count == 0) return 1.0;
    float avg_blocker_depth = blocker_sum / float(blocker_count);

    // PCSS step 2: penumbra estimate.
    float penumbra = (current_depth - avg_blocker_depth) /
                     max(avg_blocker_depth, 1e-4);
    float pcf_radius = clamp(penumbra * light_size_uv,
                             min_pcf_radius_uv,
                             max_pcf_radius_uv);

    // PCSS step 3: PCF at computed radius.
    float sum = 0.0;
    for (int i = 0; i < CSM_PCF_SAMPLES; ++i) {
        vec2 off = csmVogelDisk(i, CSM_PCF_SAMPLES, phi) * pcf_radius;
        float closest_depth =
            texture(direct_shadow_sampler,
                    vec3(shadow_uv + off, float(cascade_idx))).r;
        sum += (closest_depth < 1.0 &&
                current_depth > closest_depth + depth_bias)
               ? 0.0 : 1.0;
    }
    return sum * (1.0 / float(CSM_PCF_SAMPLES));
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

    // Skip shadow sampling when the pass is disabled (avoids stale/zero CSM
    // texture reads that would incorrectly shadow the whole scene).
    float shadow = 1.0;
    if ((camera_info.input_features & FEATURE_INPUT_SHADOW_DISABLED) == 0u) {
        // Pass the GEOMETRIC normal (normal_info.ng) — not the
        // normal-mapped one — for shadow biasing.  Normal-mapped detail
        // can produce inconsistent bias at texel scale and re-introduce
        // acne on bumpy surfaces.  gl_FragCoord.xy is the dither key
        // for per-pixel Vogel-disk rotation.
        shadow = calculateShadowFactor(
            ps_in_data.vertex_position,
            normal_info.ng,
            gl_FragCoord.xy);
    }

    bool light_from_back = false;
#ifdef USE_PUNCTUAL    
    if (dot(normal_info.ng, runtime_lights.lights[0].direction) > 0)
        light_from_back = true;
#endif

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

#ifdef DOUBLE_SIDED
    float translucent_ratio = 0.2f;
    color_info.f_diffuse += back_color_info.f_diffuse * translucent_ratio;
    color_info.f_specular += back_color_info.f_specular * translucent_ratio * 0.1f;
#endif

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

    // ── Runtime render-debug override ────────────────────────────────────────
    // Driven by the "Render Debug" menu (packed into camera_info.input_features
    // bits 16..23 by application.cpp).  Mode 0 = the shaded path above, all
    // other modes overwrite outColor with a single intermediate channel so we
    // can visually inspect what each part of the pipeline is contributing.
    uint dbg_mode =
        (camera_info.input_features & FEATURE_INPUT_DEBUG_MODE_MASK)
            >> FEATURE_INPUT_DEBUG_MODE_SHIFT;
    if (dbg_mode == DEBUG_RENDER_MODE_ALBEDO) {
        outColor = vec4(baseColor.rgb, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_NORMAL) {
        outColor = vec4(normal_info.n * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_GEOMETRIC_NORMAL) {
        outColor = vec4(normal_info.ng * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_DIFFUSE) {
        outColor = vec4(color_info.f_diffuse, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SPECULAR) {
        outColor = vec4(color_info.f_specular, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SHADOW) {
        outColor = vec4(vec3(shadow), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_ROUGHNESS) {
        outColor = vec4(vec3(material_info.perceptualRoughness), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_METALLIC) {
        outColor = vec4(vec3(material_info.metallic), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_TRANSLUCENT) {
        // Tint by AlphaMode so it's instantly clear which materials are
        // tagged translucent (glass / windows), alpha-tested, or opaque.
        // Magenta = blend / glass, yellow = mask, dark grey = opaque.
        if ((material.material_features & FEATURE_MATERIAL_BLEND) != 0u) {
            outColor = vec4(1.0, 0.2, 1.0, 1.0);
        } else if ((material.material_features & FEATURE_MATERIAL_ALPHA_MASK) != 0u) {
            outColor = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            outColor = vec4(0.1, 0.1, 0.1, 1.0);
        }
    } else if (dbg_mode == DEBUG_RENDER_MODE_SSAO) {
        // White → ssao_apply.comp multiplies by ao → vec3(ao) on screen.
        // See cluster_bindless.frag's matching branch for the rationale.
        outColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
#else
    outColor = baseColor;
#endif // NO_MTL

    // ── Debug "force red" override ─────────────────────────────────
    // Last thing in the shader so it wins over every shaded / debug
    // branch above.  Drives "is this drawable actually rendering?"
    // smoke tests — the application sets debug_force_red=1 on a
    // specific DrawableObject (currently the PlayerController player)
    // via setDebugForceRed(true); every other drawable keeps the
    // field at 0 and is unaffected.
    if (model_params.debug_force_red != 0u) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
}