#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// ─── cluster_bindless.frag ──────────────────────────────────────────
// Bindless cluster fragment shader.
//
// Lighting:
//   • Directional sun from RuntimeLightsParams (set RUNTIME_LIGHTS_PARAMS_SET)
//   • Diffuse uses Lambertian normalization (÷π) to match PBR pass magnitude
//   • Single CSM shadow sample (cheapest cascade, no PCF)
//   • Flat ambient term — no IBL
// ─────────────────────────────────────────────────────────────────────

// set 0 — shadow sampler
layout(set = PBR_GLOBAL_PARAMS_SET, binding = DIRECT_SHADOW_INDEX)
    uniform sampler2DArray direct_shadow_sampler;

// set 1 — camera
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

// set 2 — cluster SSBOs + texture array
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 0)
    readonly buffer DrawInfoBuffer {
    ClusterDrawInfo draw_infos[];
};
layout(std430, set = PBR_MATERIAL_PARAMS_SET, binding = 1)
    readonly buffer MaterialParamsBuffer {
    BindlessMaterialParams material_params[];
};
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 2)
    uniform sampler2D base_color_textures[MAX_CLUSTER_TEXTURES];

// set 4 — runtime lights
layout(set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUniformBufferObject {
    RuntimeLightsParams runtime_lights;
};

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) flat in uint v_cluster_idx;

layout(location = 0) out vec4 out_color;

// ── Cheap single-cascade shadow ──────────────────────────────────────
// Picks the tightest cascade in view-space depth and does one point
// sample — no PCF to keep the fragment cost low.
float shadowFactor(vec3 world_pos) {
    float view_depth = -(camera_info.view * vec4(world_pos, 1.0)).z;

    int cascade = CSM_CASCADE_COUNT - 1;
    for (int i = 0; i < CSM_CASCADE_COUNT; ++i) {
        if (view_depth < runtime_lights.cascade_splits[i]) {
            cascade = i;
            break;
        }
    }

    vec4 lclip = runtime_lights.light_view_proj[cascade] * vec4(world_pos, 1.0);
    vec3 lndc  = lclip.xyz / lclip.w;
    vec2 uv    = lndc.xy * 0.5 + 0.5;
    float ref  = lndc.z;
    float map  = texture(direct_shadow_sampler, vec3(uv, float(cascade))).r;
    return (map < 1.0 && ref > map + 0.001) ? 0.0 : 1.0;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(camera_info.position - v_world_pos);

    // Sun light direction (away from surface → sun) and normalised colour.
    // intensity is in physical units (lux) — apply Lambertian 1/π so the
    // diffuse magnitude matches what the PBR pass produces.
    vec3 L = normalize(-runtime_lights.lights[0].direction);
    // Clamp normalised radiance to avoid blown-out whites.
    vec3 light_col = runtime_lights.lights[0].color *
                     min(runtime_lights.lights[0].intensity, 10.0);

    float NdotL = max(dot(N, L), 0.0);
    float shad  = shadowFactor(v_world_pos);

    // Base colour
    uint mat_idx    = draw_infos[v_cluster_idx].material_idx;
    vec4 base_color = material_params[mat_idx].base_color_factor;
    int  tex_idx    = material_params[mat_idx].base_color_tex_idx;
    vec4 albedo4    = base_color;
    if (tex_idx >= 0) {
        albedo4 *= texture(base_color_textures[tex_idx], v_uv);
    }
    vec3 albedo = albedo4.rgb;

    // Lambertian diffuse (÷π matches PBR normalisation)
    const float INV_PI = 0.31831;
    vec3 ambient = albedo * light_col * (INV_PI * 0.15);
    vec3 diffuse = albedo * light_col * (INV_PI * NdotL * shad);

    // Blinn-Phong specular — small contribution, no texture lookup needed
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * shad;
    vec3  specular = light_col * (INV_PI * 0.05 * spec);

    out_color = vec4(ambient + diffuse + specular, base_color.a);
}
