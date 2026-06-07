#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- preview_mesh.frag ----------------------------------------------------
// PBR (Cook-Torrance GGX) three-spot studio lighting for the Debug Display
// preview.  The rig is derived from the mesh's bounding sphere and the
// camera position so every object gets the same framing:
//
//   * KEY   — warm, camera's upper-left, strongest
//   * FILL  — cool, camera's lower-right, reduced strength
//   * RIM   — white, behind / above, catches silhouette edges
//
// Material: base-colour texture (set 0 / binding 0 — a 1x1 white fallback
// when the asset has none) x base_color_factor, with metallic / roughness
// factors from the push constants.  Output is Reinhard tonemapped +
// gamma-encoded (RGBA8 target sampled directly by ImGui).
// ----------------------------------------------------------------------------

layout(push_constant) uniform PreviewMeshUniformBufferObject {
    PreviewMeshParams params;
};

layout(set = 0, binding = 0) uniform sampler2D base_color_tex;

layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 outColor;

// NOTE: PI comes from global_definition.glsl.h (macro) — do not redefine.

// ── Cook-Torrance helpers ────────────────────────────────────────────────
float D_GGX(float NdotH, float a) {
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float a) {
    float a2 = a * a;
    float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-6);
}

vec3 F_Schlick(vec3 f0, float VdotH) {
    float f = pow(1.0 - VdotH, 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

// Radiance arriving from one spot light aimed at the mesh centre.
vec3 spotRadiance(vec3 P, vec3 light_pos, vec3 color, float intensity,
                  float bound_radius, out vec3 L) {
    vec3  to_light = light_pos - P;
    float dist     = length(to_light);
    L = to_light / max(dist, 1e-5);

    // Wide, soft cone: the inner angle must cover the WHOLE object from a
    // light sitting only ~2.5 bounding-radii away — a narrow cone's edge
    // slices a visible bright/dark band straight across the mesh.
    vec3  axis    = normalize(params.center_radius.xyz - light_pos);
    float cos_dir = dot(-L, axis);
    float spot    = smoothstep(0.35, 0.75, cos_dir);

    float nd    = dist / max(bound_radius, 1e-5);
    float atten = 1.0 / (1.0 + 0.18 * nd * nd);

    return color * (intensity * spot * atten);
}

vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic,
          float roughness) {
    vec3  H     = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float a  = max(roughness * roughness, 0.002);
    vec3  f0 = mix(vec3(0.04), albedo, metallic);

    vec3  F    = F_Schlick(f0, VdotH);
    float D    = D_GGX(NdotH, a);
    float Vis  = V_SmithGGXCorrelated(NdotV, NdotL, a);
    vec3  spec = F * (D * Vis);

    vec3 kd   = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diff = kd * albedo / PI;

    return (diff + spec) * NdotL;
}

void main() {
    const vec3  center = params.center_radius.xyz;
    const float r      = max(params.center_radius.w, 1e-4);
    const vec3  cam    = params.camera_pos.xyz;
    const float metallic  = params.pbr_params.x;
    const float roughness = clamp(params.pbr_params.y, 0.03, 1.0);
    const float has_uv    = params.pbr_params.z;

    // Base colour: texture (authored sRGB → linearise) x factor.
    vec3 albedo = params.base_color_factor.rgb;
    if (has_uv > 0.5) {
        vec4 tex = texture(base_color_tex, v_uv);
        // Alpha cutout: foliage / fence / decal cards carve their
        // silhouette out of the quad instead of rendering it solid.
        if (tex.a < 0.5) discard;
        albedo *= pow(tex.rgb, vec3(2.2));
    }

    vec3 N = normalize(v_normal);
    vec3 V = normalize(cam - v_world);
    if (dot(N, V) < 0.0) N = -N;        // double-sided preview

    // Camera-relative basis for placing the rig.
    vec3 fwd   = normalize(center - cam);
    vec3 up_w  = abs(fwd.y) > 0.97 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(fwd, up_w));
    vec3 up    = cross(right, fwd);

    vec3 key_pos  = center + (-fwd * 1.8 - right * 1.5 + up * 1.7) * r;
    vec3 fill_pos = center + (-fwd * 1.6 + right * 1.7 + up * 0.4) * r;
    vec3 rim_pos  = center + ( fwd * 1.9             + up * 1.3) * r;

    const vec3 kKeyColor  = vec3(1.00, 0.95, 0.86);
    const vec3 kFillColor = vec3(0.62, 0.70, 0.88);
    const vec3 kRimColor  = vec3(1.00, 1.00, 1.00);

    vec3 L;
    vec3 lit = albedo * 0.04;   // ambient floor

    vec3 Li = spotRadiance(v_world, key_pos, kKeyColor, 9.0, r, L);
    lit += Li * brdf(N, V, L, albedo, metallic, roughness);

    Li = spotRadiance(v_world, fill_pos, kFillColor, 3.5, r, L);
    lit += Li * brdf(N, V, L, albedo, metallic, roughness);

    Li = spotRadiance(v_world, rim_pos, kRimColor, 6.0, r, L);
    lit += Li * brdf(N, V, L, albedo, metallic, roughness);

    // Global display exposure: +50%, applied BEFORE the tonemap so the
    // Reinhard curve absorbs it without clipping highlights.
    lit *= 1.5;

    // Reinhard tonemap + gamma 2.2 — the RGBA8 target is shown by ImGui
    // without any further display transform.
    lit = lit / (lit + vec3(1.0));
    lit = pow(lit, vec3(1.0 / 2.2));
    outColor = vec4(lit, 1.0);
}
