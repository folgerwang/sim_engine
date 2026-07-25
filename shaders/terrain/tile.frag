#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"
#include "..\weather\weather_common.glsl.h"
#include "..\functions.glsl.h"
#include "..\brdf.glsl.h"
#include "..\punctual.glsl.h"

#include "..\ibl.glsl.h"
#include "tile_common.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) in TileVsPsData in_data;

layout(location = 0) out vec4 outColor;

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

layout(set = TILE_PARAMS_SET, binding = SRC_COLOR_TEX_INDEX) uniform sampler2D src_tex;
layout(set = TILE_PARAMS_SET, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;
layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;

#include "tile_detail.glsl.h"

// 1 m albedo detail tiles (fragment-only binding — streamed with the
// height tiles; RGBA8 2048^2 per slot).
layout(set = TILE_PARAMS_SET, binding = TERRAIN_DETAIL_COLOR_INDEX)
    uniform sampler2DArray detail_color_tiles;

// Near-field surface colour from the 1 m tile if resident; `fallback`
// (global albedo) otherwise / beyond the fade band.
vec3 terrainDetailAlbedo(vec2 pos_xz_ws, vec3 fallback, float fade) {
    if (fade <= 0.0f) return fallback;
    vec2 rel = (pos_xz_ws + vec2(kTerrainMapMeters * 0.5f)) / kDetailTileMeters;
    ivec2 t = ivec2(floor(rel));
    if (any(lessThan(t, ivec2(0))) ||
        any(greaterThanEqual(t, ivec2(kDetailTilesPerSide))))
        return fallback;
    int slot = detail_color_slot[t.y * kDetailTilesPerSide + t.x];
    if (slot < 0) return fallback;
    vec2 uv = rel - vec2(t);
    vec3 c = texture(detail_color_tiles, vec3(uv, float(slot))).rgb;
    return mix(fallback, c, fade);
}

// ── Virtual-textured terrain albedo ─────────────────────────────────
// Same RVT pools/page-table/feedback as the cluster bindless path (one
// shared VirtualTextureManager) — declaration order matters, see
// vt_sample.glsl.h.  tile_params.vt_albedo_id gates the whole path.
#include "..\vt_types.glsl.h"

layout(set = TILE_PARAMS_SET, binding = TERRAIN_VT_POOL_ALBEDO_INDEX)
    uniform sampler2D vt_pool_albedo;
layout(set = TILE_PARAMS_SET, binding = TERRAIN_VT_POOL_NORMAL_INDEX)
    uniform sampler2D vt_pool_normal;
layout(set = TILE_PARAMS_SET, binding = TERRAIN_VT_POOL_MR_AO_INDEX)
    uniform sampler2D vt_pool_mr_ao;
layout(set = TILE_PARAMS_SET, binding = TERRAIN_VT_POOL_EMISSIVE_INDEX)
    uniform sampler2D vt_pool_emissive;
layout(std430, set = TILE_PARAMS_SET, binding = TERRAIN_VT_PAGE_TABLE_INDEX)
    readonly buffer VtPageTableBuffer {
    uint vt_page_table[];
};
layout(std430, set = TILE_PARAMS_SET, binding = TERRAIN_VT_META_INDEX)
    readonly buffer VtMetaBuffer {
    VirtualTextureMeta vt_meta[];
};
layout(std430, set = TILE_PARAMS_SET, binding = TERRAIN_VT_FEEDBACK_INDEX)
    buffer VtFeedbackBuffer {
    uint vt_feedback[];
};

#include "..\vt_sample.glsl.h"
layout(set = TILE_PARAMS_SET, binding = SRC_MAP_MASK_INDEX) uniform sampler2D src_map_mask;
layout(set = TILE_PARAMS_SET, binding = SRC_TEMP_TEX_INDEX) uniform sampler3D src_temp;
layout(set = TILE_PARAMS_SET, binding = DETAIL_NOISE_TEXTURE_INDEX) uniform sampler3D src_detail_noise_tex;
layout(set = TILE_PARAMS_SET, binding = ROUGH_NOISE_TEXTURE_INDEX) uniform sampler3D src_rough_noise_tex;

// ─────────────────────────────────────────────────────────────────────
//  TERRAIN MATERIAL SYSTEM
// ─────────────────────────────────────────────────────────────────────
// Splat-style layer blend. The generated map supplies COLOUR but no
// surface character, so close up the terrain read as a blurry stretched
// photo ("blocky"). Here each surface type contributes its own
// high-frequency detail, and the ML albedo is demoted to a low-frequency
// TINT — the world keeps its generated colour identity and gains texture.
//
//  weights : ML albedo colour (layout-aligned with the generator's
//            segmentation by construction) re-weighted by SLOPE and
//            ALTITUDE — steep ground sheds soil/snow to rock, etc.
//  detail  : SOLID 3D noise sampled at WORLD position, so there is no
//            projection to seam on cliffs (triplanar for free) and no
//            visible tiling grid — which is what made the old surface
//            look like blocks.
//  fade    : detail attenuates with view distance so it never aliases
//            into shimmer at range.

// fBm from the bound solid-noise volume; p is a world-space position.
float terrainSolidFbm(vec3 p, float scale) {
    float n  = 1.00f * texture(src_detail_noise_tex, p * scale).x;
    n       += 0.50f * texture(src_detail_noise_tex, p * scale * 2.13f).x;
    n       += 0.25f * texture(src_detail_noise_tex, p * scale * 4.79f).x;
    return n * (1.0f / 1.75f);
}

// Material weights packed as (grass, rock, sand, snow), normalised.
vec4 terrainMaterialWeights(vec3 macro, float slope01, float alt_m) {
    float luma = dot(macro, vec3(0.299f, 0.587f, 0.114f));
    float mx   = max(macro.r, max(macro.g, macro.b));
    float mn   = min(macro.r, min(macro.g, macro.b));
    float sat  = (mx - mn) / max(mx, 1e-4f);
    float green = macro.g - 0.5f * (macro.r + macro.b);

    // colour-driven base weights
    float w_grass = smoothstep(0.010f, 0.090f, green);
    float w_snow  = smoothstep(0.60f, 0.80f, luma)
                  * (1.0f - smoothstep(0.08f, 0.22f, sat));
    float w_sand  = smoothstep(0.04f, 0.16f, macro.r - macro.b)
                  * smoothstep(0.30f, 0.50f, luma)
                  * (1.0f - w_grass);
    // rock takes whatever the others don't claim
    float w_rock  = clamp(1.0f - (w_grass + w_snow + w_sand), 0.0f, 1.0f);
    // ...plus bare low-saturation mid-luma ground reads as rock outright
    w_rock += (1.0f - smoothstep(0.05f, 0.18f, sat)) * 0.35f
            * (1.0f - w_snow);

    // ── slope override: steep faces can't hold soil, sand or snow ──
    float steep = smoothstep(0.35f, 0.72f, slope01);
    w_grass *= (1.0f - steep);
    w_sand  *= (1.0f - steep * 1.15f);
    w_snow  *= (1.0f - steep * 0.85f);
    w_rock  += steep * 1.25f;

    // ── altitude: snow line only where it is already pale/cold ─────
    float high = smoothstep(120.0f, 190.0f, alt_m);
    w_snow += high * (1.0f - steep) * 0.5f * smoothstep(0.35f, 0.6f, luma);

    vec4 w = max(vec4(w_grass, w_rock, w_sand, w_snow), vec4(0.0f));
    return w / max(w.x + w.y + w.z + w.w, 1e-4f);
}

// Per-material high-frequency detail. Returns a multiplier around 1.0
// (albedo modulation) and writes the blended roughness.
vec3 terrainMaterialDetail(vec3 pos_ws, vec4 w, float fade,
                           out float roughness) {
    // Two bands per material: a MACRO band at the scale of the macro
    // albedo's 8 m texels (this is what actually dissolves the blocky
    // steps) plus a fine band for close-range grain.
    float m_break = terrainSolidFbm(pos_ws, 0.035f);   // ~28 m patches
    float n_grass = terrainSolidFbm(pos_ws, 0.55f);    // ~2 m clumping
    float n_rock  = terrainSolidFbm(pos_ws, 0.16f);    // ~6 m strata
    float n_rockf = texture(src_detail_noise_tex, pos_ws * 1.30f).x;
    float n_sand  = texture(src_detail_noise_tex, pos_ws * 2.20f).x;
    float n_snow  = terrainSolidFbm(pos_ws, 0.42f);

    // contrast per material: rock is the most varied, snow the least
    float d_grass = 1.0f + (n_grass - 0.5f) * 0.55f;
    float d_rock  = 1.0f + ((n_rock - 0.5f) * 0.70f
                          + (n_rockf - 0.5f) * 0.30f);
    float d_sand  = 1.0f + (n_sand  - 0.5f) * 0.22f;
    float d_snow  = 1.0f + (n_snow  - 0.5f) * 0.14f;

    float m = w.x * d_grass + w.y * d_rock + w.z * d_sand + w.w * d_snow;
    // macro breakup, applied at ALL distances the material system runs:
    // straight lerps between 8 m texels leave visible plateaus, so a
    // low-frequency multiplier is what stops the surface reading as
    // flat tiles at range.
    m *= 1.0f + (m_break - 0.5f) * 0.38f;
    m = mix(1.0f, m, fade);

    // slight per-material hue push so materials read apart even where
    // the macro map is flat (grass greener, rock cooler, sand warmer)
    vec3 tint = w.x * vec3(0.94f, 1.06f, 0.90f)
              + w.y * vec3(1.00f, 0.99f, 1.02f)
              + w.z * vec3(1.08f, 1.02f, 0.88f)
              + w.w * vec3(1.00f, 1.01f, 1.04f);
    tint = mix(vec3(1.0f), tint, fade * 0.75f);

    roughness = clamp(w.x * 0.93f + w.y * 0.76f
                    + w.z * 0.95f + w.w * 0.58f, 0.05f, 1.0f);
    // roughness breakup so specular doesn't sheet uniformly
    roughness = clamp(roughness
        + (texture(src_rough_noise_tex, pos_ws * 0.35f).x - 0.5f)
          * 0.10f * fade, 0.05f, 1.0f);

    return vec3(m) * tint;
}

// Bump the shading normal with the same solid noise field, so the
// detail is lit rather than being flat painted-on variation.
vec3 terrainMaterialNormal(vec3 pos_ws, vec3 n, vec4 w, float fade) {
    if (fade <= 0.001f) return n;
    // amplitude: rock bumps hardest, snow barely at all
    float amp = (w.x * 0.35f + w.y * 0.85f + w.z * 0.20f + w.w * 0.10f)
              * fade;
    if (amp <= 0.001f) return n;
    const float e = 0.35f;                       // metres
    float c  = terrainSolidFbm(pos_ws, 0.55f);
    float dx = terrainSolidFbm(pos_ws + vec3(e, 0.0f, 0.0f), 0.55f) - c;
    float dz = terrainSolidFbm(pos_ws + vec3(0.0f, 0.0f, e), 0.55f) - c;
    vec3 bumped = normalize(n + vec3(-dx, 0.0f, -dz) * amp * 6.0f);
    return normalize(mix(n, bumped, 0.85f));
}

void main() {
    vec3 pos = in_data.vertex_position;
    // Shading normal from the ACTUAL rendered heightfield: base rock
    // layer blended with the streamed 1 m detail tiles (the same field
    // the vertex shader displaces with), so shading matches geometry
    // exactly and is smooth across tile AND detail-tile borders.
    // Central differences; the step shrinks from one rock-layer texel
    // (4 m) to 1 m where detail is active.
    float detail_fade = terrainDetailFade(pos.xz, camera_info.position.xyz);
    vec2 rock_texel_ws = (1.0f / vec2(textureSize(rock_layer, 0)))
                         / tile_params.inv_world_range;             // meters
    float eps = mix(max(rock_texel_ws.x, rock_texel_ws.y), 1.0f, detail_fade);
    vec2 uv_eps = eps * tile_params.inv_world_range;
    vec2 huv = in_data.world_map_uv;
    float b_xn = texture(rock_layer, huv - vec2(uv_eps.x, 0.0f)).x;
    float b_xp = texture(rock_layer, huv + vec2(uv_eps.x, 0.0f)).x;
    float b_zn = texture(rock_layer, huv - vec2(0.0f, uv_eps.y)).x;
    float b_zp = texture(rock_layer, huv + vec2(0.0f, uv_eps.y)).x;
    float h_xn = terrainDetailHeight(pos.xz - vec2(eps, 0.0f), b_xn, detail_fade);
    float h_xp = terrainDetailHeight(pos.xz + vec2(eps, 0.0f), b_xp, detail_fade);
    float h_zn = terrainDetailHeight(pos.xz - vec2(0.0f, eps), b_zn, detail_fade);
    float h_zp = terrainDetailHeight(pos.xz + vec2(0.0f, eps), b_zp, detail_fade);
    // The shading normal is PURELY heightmap-derived: no procedural FBM
    // bump, no noise-texture perturbation — what the ML heightfield
    // says is what shades.
    vec3 normal = normalize(vec3((h_xn - h_xp) / (2.0f * eps),
                                 1.0f,
                                 (h_zn - h_zp) / (2.0f * eps)));

    // Surface colour comes ONLY from the ML-generated albedo (VT-backed
    // colour satellite map, or the plain map-mask fallback) — the old
    // procedural rock/soil tinting and temperature snow mix are gone.
    vec3 albedo;

    // Terrain surface colour: virtual-textured when a VT id is set
    // (streamed pages, 1 m albedo detail tiles later), otherwise the
    // plain full-world map-mask sample.
    albedo = texture(src_map_mask, in_data.world_map_uv).rgb;
    if (tile_params.vt_albedo_id != VT_INVALID_ID) {
        VirtualTextureMeta vmeta =
            vt_meta[vtIndexOf(tile_params.vt_albedo_id)];
        float lod_cont = vtComputeLod(vmeta, in_data.world_map_uv);
        uint  mip_max  = max(1u, vmeta.mip_count) - 1u;
        // Never use/request below mip 1: mip-0 requests for a screen-
        // filling 8k VT flood the page pool (LRU thrash against the
        // cluster materials).  True near-field sharpness comes from the
        // 1 m detail colour tiles; mip 1 (8 m/texel) carries the band
        // beyond the detail fade.
        uint  vt_mip = clamp(uint(lod_cont), 1u, mip_max);
        float vt_frac = (vt_mip == mip_max)
            ? 0.0f : clamp(lod_cont - float(vt_mip), 0.0f, 1.0f);

        // Streaming feedback: one tile-key per 8x8 screen block (same
        // contract as cluster_bindless.frag).
        ivec2 vt_pix = ivec2(gl_FragCoord.xy);
        if ((vt_pix.x & 7) == 0 && (vt_pix.y & 7) == 0) {
            uvec2 mip_pages = vtMipPagesXY(vmeta, vt_mip);
            vec2  wrapped   = fract(in_data.world_map_uv);
            uvec2 page      = uvec2(floor(wrapped * vec2(mip_pages)));
            page.x = min(page.x, mip_pages.x - 1u);
            page.y = min(page.y, mip_pages.y - 1u);
            ivec2 block  = vt_pix >> 3;
            uint  fb_idx = uint(block.y) * VT_FEEDBACK_PITCH + uint(block.x);
            vt_feedback[fb_idx] = vtMakeTileKey(
                vtIndexOf(tile_params.vt_albedo_id), vt_mip, page);
        }

        // Resolve with a mip-walk (like cluster_bindless.frag):
        // vtSampleAlbedo() alone returns its magenta diagnostic whenever
        // the exact picked mip isn't resident yet — walk coarser until a
        // resident page is found (the pinned smallest mip guarantees
        // termination); keep the map-mask sample if nothing resolves.
        vec2 phys_uv;
        uint walk_mip = vt_mip;
        for (uint i = 0u; i < VT_MAX_MIPS; ++i) {
            if (walk_mip > mip_max) break;
            if (vtResolve(tile_params.vt_albedo_id, in_data.world_map_uv,
                          vmeta, walk_mip, phys_uv)) {
                albedo = textureLod(vt_pool_albedo, phys_uv,
                                    walk_mip == vt_mip ? vt_frac : 0.0f).rgb;
                break;
            }
            ++walk_mip;
        }
    }
    // Near-field: the streamed 1 m albedo tile takes over from the
    // (4 m/texel) global map, fading with the same camera-distance band
    // as the height detail so colour and relief transition together.
    albedo = terrainDetailAlbedo(pos.xz, albedo, detail_fade);
    // Beyond the terrain map: neutral surround (matches the height fade
    // in tile.vert — no stretched border stripes in colour or shading).
    {
        vec2 ov = (in_data.world_map_uv
                   - clamp(in_data.world_map_uv, 0.0f, 1.0f))
                  / tile_params.inv_world_range;
        float sfade = smoothstep(0.0f, kTerrainSurroundFadeMeters,
                                 length(ov));
        albedo = mix(albedo, vec3(0.16f, 0.20f, 0.14f), sfade);
        normal = normalize(mix(normal, vec3(0.0f, 1.0f, 0.0f), sfade));
    }

    vec3 view_vec = camera_info.position.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

    // ── Terrain material layers ──────────────────────────────────────
    // Applied AFTER the macro colour is resolved (VT / map-mask / 1 m
    // tiles) so it modulates whatever the generator produced, and
    // attenuated with distance so high-frequency detail never aliases
    // into shimmer on far hillsides.
    float mat_fade = 1.0f - smoothstep(kTerrainMatDetailNear,
                                       kTerrainMatDetailFar, view_dist);
    float mat_rough = 0.85f;
    if (mat_fade > 0.001f) {
        float slope01 = clamp(1.0f - normal.y, 0.0f, 1.0f);
        vec4 mat_w = terrainMaterialWeights(albedo, slope01, pos.y);
        albedo *= terrainMaterialDetail(pos, mat_w, mat_fade, mat_rough);
        normal  = terrainMaterialNormal(pos, normal, mat_w, mat_fade);
    }

    MaterialInfo material_info;
    material_info.baseColor = albedo;

    vec3 f_diffuse = vec3(0);
    vec3 f_specular = vec3(0);

/*    float sha1 = 1.0f;
    float sha2 = 1.0f;

    float dif = clamp(dot(normal, kSunDir), 0.0f, 1.0f);
    dif *= sha1;
#ifndef LOWQUALITY
    dif *= sha2;
#endif

    float bac = clamp(dot(normalize(vec3(-kSunDir.x, 0.0, -kSunDir.z)), normal), 0.0f, 1.0f);
    float foc = clamp((pos.y + 100.0f) / 100.0f, 0.0f, 1.0f);
    float dom = clamp(0.5f + 0.5f*normal.y, 0.0f, 1.0f);
    vec3  lin = 1.0f*0.2f* mix(0.1f* vec3(0.1, 0.2, 0.1), vec3(0.7, 0.9, 1.5)*3.0f, dom)*foc;
    lin += 1.0f*8.5f* vec3(1.0, 0.9, 0.8)*dif;
    lin += 1.0f*0.27f* vec3(1.0)*bac*foc;

    color *= lin;*/

    float ior = 1.5;
    float f0_ior = 0.04;

    // Terrain is DIELECTRIC: soil, rock, sand and snow have no metallic
    // component.  The old 0.3 metallic drained the diffuse response and
    // tinted the specular with the base colour — that is a large part of
    // why the surface read as flat plastic sheeting rather than ground.
    material_info.metallic = 0.0f;
    // Roughness now comes from the material blend (rock polishes toward
    // 0.76, snow to 0.58, loose grass/sand stay ~0.95).
    material_info.perceptualRoughness = mat_rough;

    // Achromatic f0 based on IOR.
    vec3 f0 = vec3(f0_ior);

    material_info.albedoColor = mix(material_info.baseColor.rgb * (vec3(1.0) - f0),  vec3(0), material_info.metallic);
    material_info.f0 = mix(f0, material_info.baseColor.rgb, material_info.metallic);

    #ifdef USE_IBL
    float mip_count = 10;
    f_specular += getIBLRadianceGGX(normal, view, material_info.perceptualRoughness, material_info.f0, mip_count);
    f_diffuse += getIBLRadianceLambertian(normal, material_info.albedoColor);
    #endif

    //vec3 color = vec3(noise_value.w);
    vec3 color = f_diffuse + f_specular;

    float alpha = 1.0f;
    outColor = vec4(linearTosRGB(color), alpha);
//    outColor.xyz *= in_data.test_color;
}