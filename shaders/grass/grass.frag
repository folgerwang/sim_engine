#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"
#include "..\functions.glsl.h"
#include "..\brdf.glsl.h"
#include "..\punctual.glsl.h"
#include "..\ibl.glsl.h"
#include "..\tonemap.glsl.h"
#include "..\terrain\tile_common.glsl.h"
#include "grass_common.glsl.h"

// ─────────────────────────────────────────────────────────────────────
// What this used to be:
//
//     float a = pow(max(abs(tex_coord.x - 0.5) - 0.003, 0.0), 0.05);
//     outColor = vec4(vec3(a * 0.1 + 0.9) * vec3(0.5, 1.0, 0.45), 1.0);
//
// One constant, unlit, untonemapped, and emitted straight into the LDR
// forward target.  vec3(0.5, 1.0, 0.45) at ~0.99 intensity is a fully
// saturated primary green at nearly peak brightness — brighter than the
// SUNLIT terrain next to it and identical on every blade, in shadow and
// out.  That is the whole of the "colour palette" problem: there was no
// palette and no lighting, just one hard-coded green.
//
// Three things fix it, in order of how much they matter:
//
//  1. LIGHT the grass.  The GBUFFER_OUTPUT permutation writes the same
//     four attributes terrain and cluster geometry write, so
//     deferred_resolve.comp lights grass with the same sun, the same
//     CSM / RT shadows and the same tonemap as everything else — grass
//     inside a tree's shadow goes dark because it is in shadow, not
//     because a constant says so.  The forward branch below is the
//     no-deferred fallback and uses the sky IBL, matching what
//     tile.frag's forward branch does for terrain.
//
//  2. Bring the ALBEDO down to plant reflectance.  Fresh grass is
//     ~0.13-0.20 linear in green and ~0.06 in red — dark, and much less
//     saturated than the primary it was using.  Bright saturated green
//     is a pigment that does not exist outdoors.
//
//  3. Give it VARIATION with structure: dryness in patches (not
//     per-blade noise), a dark base and a bleached tip on each blade,
//     per-blade value jitter, and a hue pulled from the terrain albedo
//     underneath so grass agrees with the ground it grows out of
//     instead of floating over it as a separate colour.
// ─────────────────────────────────────────────────────────────────────

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

// The terrain's macro albedo map — the same sample tile.frag takes for
// its own surface colour, so the two cannot disagree.
layout(set = TILE_PARAMS_SET, binding = SRC_MAP_MASK_INDEX) uniform sampler2D src_map_mask;

layout(location = 0) in GrassVsPsData {
    vec4 pos_ws_v;   // xyz = world position, w = height fraction
    vec4 nrm_hash;   // xyz = world normal,   w = per-blade hash
    vec4 attribs;    // x = dryness, y = signed edge coord, z = blade height
} in_data;

#ifdef GBUFFER_OUTPUT
layout(location = 0) out vec4 out_albedo_ao;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_emissive_metal;
layout(location = 3) out vec2 out_velocity;

// Same encode as cluster_bindless.frag / tile.frag, paired with
// octDecode in deferred_resolve.comp.
vec2 octEncodeDir(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy
                            : (1.0 - abs(n.yx)) * vec2(
                                  n.x >= 0.0 ? 1.0 : -1.0,
                                  n.y >= 0.0 ? 1.0 : -1.0);
    return oct * 0.5 + 0.5;
}
#else
layout(location = 0) out vec4 outColor;
#endif

// Linear reflectances, measured-ish rather than picked:
//   live  — fresh leaf, green ~0.19, red ~0.13, blue ~0.06
//   dry   — cured straw, warm and a good deal brighter
//   shade — deep inside the sward, where almost nothing reaches
const vec3 kLeafLive  = vec3(0.128f, 0.188f, 0.062f);
const vec3 kLeafDry   = vec3(0.252f, 0.206f, 0.094f);
const vec3 kLeafShade = vec3(0.042f, 0.062f, 0.028f);

void main() {
    vec3  P = in_data.pos_ws_v.xyz;
    float v = clamp(in_data.pos_ws_v.w, 0.0f, 1.0f);
    float dry_patch = in_data.attribs.x;
    float hb = in_data.nrm_hash.w;
    float edge = clamp(abs(in_data.attribs.y), 0.0f, 1.0f);

    // ── Normal ───────────────────────────────────────────────────────
    // A blade is a two-sided ribbon and the strip alternates winding, so
    // gl_FrontFacing flips triangle to triangle and would band the
    // shading.  Orient toward the viewer instead: for geometry this thin
    // the lit side is whichever side you can see.
    vec3 V = normalize(camera_info.position - P);
    vec3 N = normalize(in_data.nrm_hash.xyz);
    if (dot(N, V) < 0.0f) N = -N;

    // ── Albedo ───────────────────────────────────────────────────────
    // Dryness is a PATCH property (grassDryField) plus a small per-blade
    // offset, so a meadow browns in drifts with individual blades still
    // ahead of or behind their neighbours.
    float d = clamp(dry_patch + (hb - 0.5f) * 0.38f, 0.0f, 1.0f);
    vec3 leaf = mix(kLeafLive, kLeafDry, d * d);

    // Per-blade value jitter — a field is never one dye lot.
    leaf *= 0.80f + 0.42f * fract(hb * 7.31f);

    // Agree with the ground.  Take a third of the terrain albedo's HUE
    // and part of its VALUE: grass on pale dry soil is pale and dry,
    // grass on dark river meadow is dark and deep, with neither able to
    // drag the blade away from being a plant.
    vec2 world_map_uv = (P.xz - tile_params.world_min) *
                        tile_params.inv_world_range;
    vec3 ground = texture(src_map_mask, world_map_uv).rgb;
    float g_luma = max(dot(ground, vec3(0.299f, 0.587f, 0.114f)), 1e-4f);
    vec3  g_hue  = ground / g_luma;
    leaf = mix(leaf, leaf * g_hue, 0.30f);
    leaf *= mix(1.0f, clamp(g_luma * 3.0f, 0.62f, 1.45f), 0.45f);

    // ── Along the blade ──────────────────────────────────────────────
    // The bottom of a sward is buried in its own neighbours and receives
    // almost nothing; the tip is exposed, sun-bleached and a little
    // translucent.  This gradient is what makes a field read as a
    // continuous mass instead of a bed of separate spikes, and it is
    // baked into albedo rather than into the AO channel because the
    // G-buffer's alpha is spoken for (see out_albedo_ao below).
    float sward = smoothstep(0.0f, 0.52f, v);
    vec3 albedo = mix(kLeafShade, leaf, sward);
    albedo = mix(albedo, mix(albedo, kLeafDry, 0.30f),
                 smoothstep(0.70f, 1.0f, v));

    // Thin translucent margin along the rim of the blade.
    albedo *= 1.0f + 0.12f * smoothstep(0.70f, 1.0f, edge);

    // Cured blades are matte; live ones keep a waxy cuticle sheen.
    float rough = mix(0.52f, 0.82f, d);

    // Ambient occlusion for the ambient term only: a blade near the
    // ground sees less sky.  Floored well clear of the G-buffer
    // sentinel (see below) — the real darkening is in the albedo.
    float ao = mix(0.70f, 1.0f, sward);

#ifdef GBUFFER_OUTPUT
    // Deferred: hand the same four attributes terrain hands over and let
    // deferred_resolve.comp do the lighting.  This is the whole reason
    // grass now sits in the scene's shadows instead of glowing through
    // them.
    //
    // Alpha is the resolve's forward/deferred sentinel as well as AO,
    // and the target is R8G8B8A8_UNORM, so the floor is 129/255 rather
    // than 0.5: 0.5 encodes on the rounding tie and can come back as
    // 127/255 = 0.498, which the resolve reads as "forward pixel, leave
    // it alone" (the same trap tile.frag documents).
    out_albedo_ao      = vec4(albedo, clamp(ao, 129.0f / 255.0f, 1.0f));
    out_normal_rough   = vec4(octEncodeDir(N), rough, 0.0f);
    // Grass is dielectric; the geometric normal a shadow ray should be
    // offset along is the blade's own face normal.
    out_emissive_metal = vec4(octEncodeDir(N), 0.0f, 0.0f);
    // Velocity is camera-only.  The blades DO move in wind, but the
    // sway is slow next to camera motion and reconstructing last
    // frame's wind would mean carrying the previous patch sample.
    vec4 cur_clip  = camera_info.view_proj      * vec4(P, 1.0f);
    vec4 prev_clip = camera_info.prev_view_proj * vec4(P, 1.0f);
    out_velocity   = cur_clip.xy / cur_clip.w - prev_clip.xy / prev_clip.w;
#else
    // Forward fallback (deferred off): sky IBL only, exactly what
    // tile.frag's forward branch gives the terrain, so the two agree.
    vec3 f0 = vec3(0.04f);
    vec3 diffuse_color = albedo * (vec3(1.0f) - f0);

    float mip_count = 10.0f;
    vec3 f_diffuse  = getIBLRadianceLambertian(N, diffuse_color) * ao;
    vec3 f_specular = getIBLRadianceGGX(N, V, rough, f0, mip_count) * ao;

    outColor = vec4(sceneTonemap(f_diffuse + f_specular), 1.0f);

    uint dbg_mode =
        (camera_info.input_features & FEATURE_INPUT_DEBUG_MODE_MASK)
            >> FEATURE_INPUT_DEBUG_MODE_SHIFT;
    if (dbg_mode == DEBUG_RENDER_MODE_ALBEDO) {
        outColor = vec4(albedo, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_NORMAL ||
               dbg_mode == DEBUG_RENDER_MODE_GEOMETRIC_NORMAL) {
        outColor = vec4(N * 0.5f + 0.5f, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_ROUGHNESS) {
        outColor = vec4(vec3(rough), 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_METALLIC) {
        outColor = vec4(vec3(0.0f), 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SSAO) {
        outColor = vec4(1.0f);
    }
#endif
}
