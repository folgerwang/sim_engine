#version 450
#extension GL_ARB_separate_shader_objects : enable
// Required for bindless texture array with non-uniform (per-cluster) indices.
// Without this, the GPU assumes tex_idx is uniform across the subgroup and
// picks one texture for the whole wave, producing wrong textures at material
// boundaries between adjacent clusters.
#extension GL_EXT_nonuniform_qualifier : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"

// ─── cluster_bindless.frag ──────────────────────────────────────────
// Bindless cluster fragment shader.
//
// Lighting matches base.frag structure:
//   • Directional sun from RuntimeLightsParams (set RUNTIME_LIGHTS_PARAMS_SET)
//   • Diffuse: Lambertian ÷π (simplified, no Fresnel energy conservation)
//   • Single CSM shadow sample (cheapest cascade, no PCF)
//   • Flat ambient (0.03 × albedo) — no IBL
//   • linearTosRGB tone mapping — matches base.frag default path
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
// Matches calculateShadowFactor() in base.frag exactly.
const float SHADOW_BIAS = 0.001;
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
    return (map < 1.0 && ref > map + SHADOW_BIAS) ? 0.0 : 1.0;
}

void main() {
    // Base colour — fetch first so we can discard early.
    uint mat_idx    = draw_infos[v_cluster_idx].material_idx;
    vec4 base_color = material_params[mat_idx].base_color_factor;
    int  tex_idx    = material_params[mat_idx].base_color_tex_idx;
    int  mat_flags  = material_params[mat_idx].flags;
    vec4 albedo4    = base_color;
    if (tex_idx >= 0) {
        // Texture views are hardware sRGB format — GPU auto-linearizes on sample.
        // Do NOT apply manual sRGBToLinear here (would double-convert and desaturate).
        // nonuniformEXT: tex_idx varies per cluster — the GPU must not assume it is
        // uniform across the subgroup (wave). Without this, one texture is picked for
        // the entire wave and adjacent clusters with different materials get wrong textures.
        albedo4 *= texture(base_color_textures[nonuniformEXT(tex_idx)], v_uv);
    }

    // Alpha mask discard — matches base.frag: if(baseColor.a < alpha_cutoff) discard.
    if ((mat_flags & BINDLESS_MAT_ALPHA_MASK) != 0) {
        if (albedo4.a < material_params[mat_idx].alpha_cutoff) discard;
    }

    vec3 albedo = albedo4.rgb;

    // Normal — for double-sided materials flip N on back faces so lighting is
    // correct from both sides. Backface culling is disabled for the cluster pipeline.
    vec3 N = normalize(v_normal);
    if ((mat_flags & BINDLESS_MAT_DOUBLE_SIDED) != 0 && !gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(camera_info.position - v_world_pos);

    // Sun light direction (away from surface → sun).
    vec3 L = normalize(-runtime_lights.lights[0].direction);
    // Full physical intensity — tone mapping below handles highlight compression.
    vec3 light_col = runtime_lights.lights[0].color *
                     runtime_lights.lights[0].intensity;

    float NdotL = max(dot(N, L), 0.0);

    // Respect FEATURE_INPUT_SHADOW_DISABLED, matching base.frag.
    float shad = 1.0;
    if ((camera_info.input_features & FEATURE_INPUT_SHADOW_DISABLED) == 0u) {
        shad = shadowFactor(v_world_pos);
    }

    // Flat ambient (no IBL) — 3% albedo is a standard dark-room approximation.
    vec3 ambient = albedo * 0.03;

    // Lambertian diffuse (÷π matches PBR BRDF_lambertian normalisation).
    vec3 diffuse = albedo * light_col * (NdotL * shad / M_PI);

    // Blinn-Phong specular — cheap substitute for GGX, no texture lookup needed.
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * shad;
    vec3  specular = light_col * (0.05 * spec / M_PI);

    vec3 color = ambient + diffuse + specular;

    // Gamma-correct to sRGB — matches base.frag's TONEMAP_DEFAULT path (exposure=1.0,
    // linearTosRGB only). FBX materials set tonemap_type=TONEMAP_DEFAULT by default.
    out_color = vec4(linearTosRGB(color), base_color.a);
}
