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
//   • Diffuse IBL ambient via lambertian_env_sampler (matches base.frag USE_IBL path)
//   • linearTosRGB tone mapping — matches base.frag default path
// ─────────────────────────────────────────────────────────────────────

// set 0 — IBL + shadow samplers (ibl.glsl.h declares all of set 0, including
// direct_shadow_sampler at DIRECT_SHADOW_INDEX — do not redeclare it below).
#include "ibl.glsl.h"

// set 1 — camera
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

// set 2 — cluster SSBOs + texture arrays
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
layout(set = PBR_MATERIAL_PARAMS_SET, binding = 3)
    uniform sampler2D normal_textures[MAX_CLUSTER_TEXTURES];

// set 4 — runtime lights
layout(set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUniformBufferObject {
    RuntimeLightsParams runtime_lights;
};

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) flat in uint v_cluster_idx;
// Interpolated world-space tangent + bitangent sign (see BindlessVertex doc
// in cluster_renderer.h).  Replaces the per-fragment dFdx/dFdy tangent
// reconstruction that produced sparkle on shaded surfaces.
layout(location = 4) in vec4 v_tangent;

// OIT_OUTPUT compiles a translucent variant that writes the McGuire-Bavoil
// "Weighted Blended OIT" pair (accum + reveal) instead of a single colour.
// The translucent cluster pipeline uses this variant; opaque uses the
// default single-attachment output.  See oit_composite.frag for the
// resolve shader and ClusterRenderer::initBindlessPipeline for the per-
// pipeline blend setup.
#ifdef OIT_OUTPUT
    layout(location = 0) out vec4 out_accum;   // RGBA16F: rgb = colour×alpha×w, a = alpha×w
    layout(location = 1) out float out_reveal; // R8/R16F: accumulates Π(1 − αᵢ)
#else
    layout(location = 0) out vec4 out_color;
#endif

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

#ifdef OIT_OUTPUT
    // Defense-in-depth: only AlphaMode::Blend materials should ever reach
    // the OIT translucent pipeline (cull shader routes by flag).  If
    // anything else sneaks through (e.g. a buggy material upload that left
    // the flag stale), discard rather than write to accum/reveal so it
    // can't darken the composite.
    if ((mat_flags & BINDLESS_MAT_TRANSLUCENT) == 0) {
        discard;
    }
#else
    // The opaque pipeline must NEVER draw translucent clusters — those go
    // through the OIT path.  Symmetric guard: if the cull shader miswrites
    // a translucent cluster into the opaque indirect bucket, drop it here.
    if ((mat_flags & BINDLESS_MAT_TRANSLUCENT) != 0) {
        discard;
    }
#endif
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

    // Geometry normal — flip for back faces BEFORE building TBN so the cotangent
    // frame is oriented correctly for double-sided materials.
    // Backface culling is disabled for the cluster pipeline.
    vec3 N = normalize(v_normal);
    if ((mat_flags & BINDLESS_MAT_DOUBLE_SIDED) != 0 && !gl_FrontFacing) {
        N = -N;
    }
    // Stash the un-perturbed (geometric) normal before the normal map below
    // overwrites N.  Used by the GEOMETRIC_NORMAL debug visualisation; if we
    // sampled `N` after the TBN multiplication it would be identical to the
    // NORMAL mode, which is exactly the "they look the same" bug.
    vec3 N_geom = N;

    // Normal map — sample and rotate into world space via the interpolated
    // vertex tangent frame.  Tangents are precomputed at upload time
    // (computeMeshTangents() in cluster_renderer.cpp) and stored on the
    // merged vertex stream as vec4(T_world, bitangent_sign), which removes
    // the per-fragment dFdx/dFdy reconstruction that previously produced
    // fine-grained sparkle on shaded surfaces (visible in render-debug
    // mode 2 / 3, and in the final shaded image as specular speckle).
    int norm_idx = material_params[mat_idx].normal_tex_idx;
    if (norm_idx >= 0) {
        // Sample and decode the normal map.
        // Bistro (and most DCC tools) export DirectX-convention normal maps where
        // the green channel is inverted relative to OpenGL/GLSL tangent space.
        // base.frag applies n.y = -n.y for the same reason — we must match it.
        // Z is reconstructed from XY (more robust than reading the stored Z which
        // can be degraded by DXT5nm / BC5 compression).
        vec4 raw_n = texture(normal_textures[nonuniformEXT(norm_idx)], v_uv);
        vec2 nxy   = raw_n.xy * 2.0 - 1.0;
        nxy.y      = -nxy.y;   // DX → GL convention flip
        vec3 ts_n  = vec3(nxy, sqrt(max(1.0 - dot(nxy, nxy), 0.0)));

        // Reconstruct an orthonormal TBN from the interpolated tangent.
        // The vertex tangent was orthogonalised against the OBJECT-space
        // normal at upload, but interpolation across the triangle and the
        // potentially-non-rigid model transform can leave a small component
        // along N — re-orthogonalise per fragment to keep TBN orthonormal.
        vec3 T = normalize(v_tangent.xyz - dot(v_tangent.xyz, N) * N);
        vec3 B = cross(N, T) * v_tangent.w;

        N = normalize(mat3(T, B, N) * ts_n);
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

    // Diffuse IBL ambient — samples the pre-convolved lambertian irradiance
    // cubemap with the surface normal, matching base.frag's USE_IBL path.
    // This gives direction-dependent ambient (brighter sky-facing surfaces,
    // darker ground-facing ones) instead of the old flat 3% approximation.
    vec3 ambient = getIBLRadianceLambertian(N, albedo);

    // Lambertian diffuse (÷π matches PBR BRDF_lambertian normalisation).
    vec3 diffuse = albedo * light_col * (NdotL * shad / M_PI);

    // Blinn-Phong specular — cheap substitute for GGX, no texture lookup needed.
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * shad;
    vec3  specular = light_col * (0.05 * spec / M_PI);

    vec3 color = ambient + diffuse + specular;

#ifdef OIT_OUTPUT
    // ── Image-based glass shading ───────────────────────────────────────
    // Glass surfaces aren't well represented by the diffuse + Blinn-Phong
    // model used for opaque cluster geometry.  In the OIT (translucent)
    // path we replace `color` with a Fresnel-blended mix of:
    //
    //   • Reflection : IBL GGX specular cube sampled in reflect(-V, N) at
    //                  low roughness (sharp environment reflection).
    //                  Standard getIBLRadianceGGX with F0 = 0.04 gives a
    //                  dielectric reflection that's faint at near-normal
    //                  angles and intensifies at grazing — exactly the
    //                  Fresnel behaviour real glass shows.
    //
    //   • Refraction : IBL specular cube sampled in a *slightly* bent
    //                  view direction.  Pure refract(-V, N, 1/η) at glass
    //                  IOR 1.5 produces a strong sphere-like warp; for a
    //                  thin pane we want the bend to be barely
    //                  perceptible — just enough for the eye to read
    //                  "this surface has thickness, what's behind shifts
    //                  with the surface curvature."  We blend the
    //                  geometric view direction (-V) with the fully-
    //                  refracted direction by glass_thickness ∈ [0, 1]:
    //                  0  = no refraction (looks like a flat decal),
    //                  1  = full IOR=1.5 refraction (sphere-style warp).
    //                  ~0.25 reads as "thick window pane" without making
    //                  the world warp obviously.
    //
    // Schlick Fresnel mixes refraction (near-normal) with reflection
    // (grazing).  base_color tints the refraction so stained-glass windows
    // and tinted bottles shift the colour of what's seen through them.
    // 25 % of the direct-lighting term is layered on top so a sun on
    // glass still leaves a Blinn-Phong specular highlight and the surface
    // isn't pitch-dark under unlit IBL.
    {
        // ── Glass parameters ─────────────────────────────────────────────
        const float glass_ior            = 1.5;    // standard window glass
        const float glass_thickness      = 0.30;   // 0 = no refraction, 1 = sphere-style bend
        const float glass_tint_strength  = 0.20;   // 0 = neutral, 1 = full albedo tint on transmission
        const float glass_F0             = 0.04;   // dielectric Fresnel reflectance at normal incidence
        const float glass_chromatic_aber = 0.015;  // R/B index spread (gives slight prism fringe)

        // ── Reflection — sharp env sample at the mirror direction ──────
        // textureLod(... , 0.0) bypasses the GGX-convolved mips and reads
        // the un-prefiltered sky cube, which is what real flat glass
        // reflects.  Deliberately skipping getIBLRadianceGGX here: that
        // helper multiplies by (f0·brdf.r + brdf.g) ≈ 0.04 for clear-glass
        // f0, which collapses the reflection to nearly nothing.  We want
        // the full sky to appear in the mirror direction with strength
        // controlled by Fresnel below.
        vec3 reflect_dir     = normalize(reflect(-V, N));
        vec3 reflection_term = textureLod(ggx_env_sampler, reflect_dir, 0.0).rgb;

        // ── Refraction — slightly bent ray to simulate pane thickness ──
        // refract() gives the direction the view ray continues *inside*
        // the glass.  At η = 1.5 a pure refract is a strong sphere-style
        // warp; for a thin pane we want a barely-perceptible bend, so we
        // interpolate from the straight-through view direction (-V) to
        // the fully-refracted direction by `glass_thickness`.
        // Per-channel IORs give a tiny chromatic split — red bends less
        // than blue, producing the subtle prism fringe real glass shows.
        vec3 ref_dir_r = refract(-V, N, 1.0 / (glass_ior - glass_chromatic_aber));
        vec3 ref_dir_g = refract(-V, N, 1.0 / glass_ior);
        vec3 ref_dir_b = refract(-V, N, 1.0 / (glass_ior + glass_chromatic_aber));
        if (dot(ref_dir_r, ref_dir_r) < 1e-6) ref_dir_r = -V;
        if (dot(ref_dir_g, ref_dir_g) < 1e-6) ref_dir_g = -V;
        if (dot(ref_dir_b, ref_dir_b) < 1e-6) ref_dir_b = -V;

        vec3 view_r = normalize(mix(-V, ref_dir_r, glass_thickness));
        vec3 view_g = normalize(mix(-V, ref_dir_g, glass_thickness));
        vec3 view_b = normalize(mix(-V, ref_dir_b, glass_thickness));

        vec3 refraction_term = vec3(
            textureLod(ggx_env_sampler, view_r, 0.0).r,
            textureLod(ggx_env_sampler, view_g, 0.0).g,
            textureLod(ggx_env_sampler, view_b, 0.0).b);

        // Albedo tints what's seen through the glass — a green pane
        // greens the world behind it, clear glass leaves it untouched.
        vec3 glass_tint   = mix(vec3(1.0), albedo, glass_tint_strength);
        refraction_term  *= glass_tint;

        // ── Schlick Fresnel ──────────────────────────────────────────────
        // F0 = 0.04 for dielectric glass.  At normal incidence the surface
        // transmits ~96 %; at grazing it reflects nearly 100 % — giving
        // the classic "looking up at a tall glass tower" effect where the
        // upper panels mirror the sky and the lower panels show what's
        // inside.
        float NdotV   = max(dot(N, V), 0.0);
        float fresnel = glass_F0 + (1.0 - glass_F0) * pow(1.0 - NdotV, 5.0);

        // ── Final composition ────────────────────────────────────────────
        // mix(refraction, reflection, fresnel) is the physically-motivated
        // glass term.  We layer a small fraction of the direct-lighting
        // result so a sun glint on glass still produces a Blinn-Phong
        // highlight on top of the IBL reflection.
        color = mix(refraction_term, reflection_term, fresnel)
              + specular * 0.5;
    }
#endif

    // Gamma-correct to sRGB — matches base.frag's TONEMAP_DEFAULT path (exposure=1.0,
    // linearTosRGB only). FBX materials set tonemap_type=TONEMAP_DEFAULT by default.
    //
    // Output alpha = albedo4.a (texture × factor) rather than just base_color.a
    // so semi-transparent textures contribute correctly to the translucent
    // pass blend equation.  For opaque materials albedo4.a ≈ 1.0, so the
    // translucent pipeline (drawn second) is a no-op for them — the cull
    // shader filters them out anyway.
    vec4 final_color = vec4(linearTosRGB(color), albedo4.a);

#ifdef OIT_OUTPUT
    // ── Weighted Blended OIT output ─────────────────────────────────────
    // McGuire & Bavoil 2013, "weight 7" formulation.  The weight emphasises
    // near-camera fragments (1 − z * 0.9) and high-alpha fragments (alpha
    // ramp), and the clamp keeps the magnitude inside FP16 range so the
    // RGBA16F accumulator never overflows.  Linear-space colour is fed in
    // (we pass linearTosRGB-encoded; for true OIT we'd skip sRGB encoding,
    // but matching the opaque path here keeps colour spaces consistent).
    //
    // Alpha is clamped into [0.2, 0.85] before the weight calc.  This:
    //   • prevents the upper-bound case (a == 1.0) from setting reveal to
    //     zero — that drives the composite to emit `vec4(rgb, 1.0)` which
    //     fully replaces the scene colour and reads as "all black" if the
    //     lit colour is small.  α = 0.85 also keeps Fresnel reflection
    //     punchy at grazing where glass should look quite opaque.
    //   • prevents the lower-bound case (a ~ 0) from contributing nothing
    //     visible to the composite — assets sometimes ship glass with
    //     base_color_factor.a ≈ 0.05, which after blending leaves only
    //     5 % of the IBL reflection / refraction visible.  A 0.2 floor
    //     keeps glass legible while still letting most of the scene show
    //     through.
    float a = clamp(final_color.a, 0.2, 0.85);
    float z = gl_FragCoord.z;
    float weight = clamp(
        pow(min(1.0, a * 10.0) + 0.01, 3.0) * 1e8 *
        pow(1.0 - z * 0.9, 3.0),
        1e-2, 3e3);

    out_accum  = vec4(final_color.rgb * a, a) * weight;
    out_reveal = a;  // pipeline blend is dst*(1−src) so this accumulates Π(1−αᵢ)

    // Debug-mode override is intentionally skipped in OIT_OUTPUT — the
    // visualisation only makes sense in the single-target opaque path.
    return;
#else
    out_color = final_color;

    // ── Runtime render-debug override ────────────────────────────────────────
    // Mirrors base.frag.  The cluster bindless path uses Blinn-Phong specular
    // (no IBL specular cube), so the SPECULAR mode here shows that term;
    // ROUGHNESS / METALLIC are not authored per-cluster and are emitted as
    // constants so the visualisation is still well-defined.
    uint dbg_mode =
        (camera_info.input_features & FEATURE_INPUT_DEBUG_MODE_MASK)
            >> FEATURE_INPUT_DEBUG_MODE_SHIFT;
    if (dbg_mode == DEBUG_RENDER_MODE_ALBEDO) {
        out_color = vec4(albedo, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_NORMAL) {
        out_color = vec4(N * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_GEOMETRIC_NORMAL) {
        out_color = vec4(N_geom * 0.5 + 0.5, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_DIFFUSE) {
        out_color = vec4(ambient + diffuse, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SPECULAR) {
        out_color = vec4(specular, 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SHADOW) {
        out_color = vec4(vec3(shad), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_ROUGHNESS) {
        out_color = vec4(vec3(0.5), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_METALLIC) {
        out_color = vec4(vec3(0.0), 1.0);
    } else if (dbg_mode == DEBUG_RENDER_MODE_TRANSLUCENT) {
        // Tint by alpha mode (matches base.frag's mapping):
        //   magenta = BINDLESS_MAT_TRANSLUCENT (glass / windows / Blend)
        //   yellow  = BINDLESS_MAT_ALPHA_MASK
        //   dark    = opaque
        // Lets you visually verify that glass-by-name overrides reached
        // the cluster path (since cluster bindless currently still draws
        // these without alpha blending, this debug mode is the only way
        // to see the tagging is correct upstream of the rendering).
        if ((mat_flags & BINDLESS_MAT_TRANSLUCENT) != 0) {
            out_color = vec4(1.0, 0.2, 1.0, 1.0);
        } else if ((mat_flags & BINDLESS_MAT_ALPHA_MASK) != 0) {
            out_color = vec4(1.0, 1.0, 0.0, 1.0);
        } else {
            out_color = vec4(0.1, 0.1, 0.1, 1.0);
        }
    }
#endif // OIT_OUTPUT
}
