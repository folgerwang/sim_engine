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

    MaterialInfo material_info;
    material_info.baseColor = albedo;

    vec3 view_vec = camera_info.position.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

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

    material_info.metallic = 0.3f;//material.metallic_factor;
    material_info.perceptualRoughness = 0.8f;//material.roughness_factor;

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