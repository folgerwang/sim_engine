#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "../global_definition.glsl.h"
#include "../functions.glsl.h"
#include "../brdf.glsl.h"
#include "../punctual.glsl.h"

#include "../ibl.glsl.h"
#include "tile_common.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) in VsPsData {
    vec3    vertex_position;
    vec2    world_map_uv;
    vec3    test_color;
    float   water_depth;
} in_data;

#if defined(WATER_ATTR) || defined(WATER_LBM)
// LBM river-surface sim output (lbm_water.comp): xyz = ripple normal,
// w = height deviation.  Bound on a DEDICATED set 3 so the existing
// tile descriptor layouts stay byte-identical for every other pass.
// Shared by BOTH permutations that see the sim: WATER_ATTR (glass
// attribute path) and WATER_LBM (the displaced-mesh forward path,
// which blends the ripple normal into its shading).
layout(set = 3, binding = 0) uniform sampler2D lbm_surface_tex;
layout(std430, set = 3, binding = 1) readonly buffer LbmRegionBuf {
    // xz = patch origin (world m), y = cell size, w = grid size
    vec4 lbm_region;
};
// The sim's GENERATED flowmap: per-cell velocity (m/s, world xz).
// The LBM authors the flow (from the static surface's gradient);
// here it advects the procedural surface detail along the current.
layout(set = 3, binding = 2) uniform sampler2D lbm_flow_tex;
#endif

#ifdef WATER_ATTR
// Water attribute permutation (tile_water_attr_frag.spv): river/pond
// surfaces re-rasterise into the same two GLASS ATTRIBUTE targets the
// window glass writes, tagged KIND = 1.0 (water).  The deferred
// resolve then runs REAL ray-traced reflection AND refraction for the
// surface — the screen-space refraction hack in the forward branch
// below can only bend what is already on screen; the traced version
// sees the river bed, the far bank, the bridge above.
//   attr0 = octEncode(N).xy, linear view depth (m), roughness
//   attr1 = water tint rgb, 1.0
layout(location = 0) out vec4 out_glass_nr;
layout(location = 1) out vec4 out_glass_tint;

vec2 octEncodeDir(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy
                            : (1.0 - abs(n.yx)) * vec2(
                                  n.x >= 0.0 ? 1.0 : -1.0,
                                  n.y >= 0.0 ? 1.0 : -1.0);
    return oct * 0.5 + 0.5;
}

vec4 outColor;   // satisfies the dead forward code past our early return
#else
layout(location = 0) out vec4 outColor;
#endif

// Deep-water tint.  ONE definition for both render paths: the traced
// path hands it to the resolve as the per-metre absorption tint, the
// forward path runs the same Beer-Lambert against it locally — so the
// water reads the same colour wherever the two paths hand off.
const vec3 kWaterTint = vec3(0.12, 0.32, 0.38);
// Per-metre extinction scale.  0.35 read as glass — the bed stayed
// visible through metres of water; 0.9 goes opaque by ~2-3 m of
// optical path while the first half-metre still shows the bottom.
// KEEP IN SYNC with glassWaterOverlay in deferred_resolve.comp.
const float kWaterExtinction = 0.9;

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

layout(set = TILE_PARAMS_SET, binding = SRC_COLOR_TEX_INDEX) uniform sampler2D src_tex;
layout(set = TILE_PARAMS_SET, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;
layout(set = TILE_PARAMS_SET, binding = WATER_NORMAL_BUFFER_INDEX) uniform sampler2D water_normal_tex;
layout(set = TILE_PARAMS_SET, binding = WATER_FLOW_BUFFER_INDEX) uniform sampler2D water_flow_tex;

// Standard noise warping. Call the noise function, then feed a variation of the result
// into itself. Rinse and repeat, etc. Completely made up on the spot, but keeping your 
// original concept in mind, which involved combining noise layers travelling in opposing
// directions.
float warpedNoise(vec2 p) {
    
    vec2 m = vec2(tile_params.time, -tile_params.time)*.25;
    float x = fractalNoise(p + m);
    float y = fractalNoise(p + m.yx + x);
    float z = fractalNoise(p - m - x + y);
    return fractalNoise(p + vec2(x, y) + vec2(y, z) + vec2(z, x) + length(vec3(x, y, z))*0.25);
    
}

void main() {
    if (in_data.water_depth < 0.03f) {
        discard;
    }

    float transparent_factor = clamp((in_data.water_depth - 0.03f) / 0.03f, 0.0f, 1.0f);

    vec3 pos = in_data.vertex_position;
    vec3 tnor = terrainNormal(vec2(pos.x, pos.z), 0.00025f, 2000.0f);

    float noise;
#if defined(WATER_ATTR) || defined(WATER_LBM)
    // ── LBM patch coverage + its GENERATED flow at this fragment ────
    // Resolved once here; the ripple-normal blend below reuses it.
    vec2  lbm_luv    = vec2(-1.0);
    float lbm_wgt    = 0.0;
    vec2  lbm_flow_v = vec2(0.0);
    {
        float lbm_span = lbm_region.w * lbm_region.y;
        if (lbm_span > 1.0) {
            vec2 luv = (pos.xz - lbm_region.xz) / lbm_span;
            if (all(greaterThan(luv, vec2(0.0))) &&
                all(lessThan(luv, vec2(1.0)))) {
                vec2 ef = smoothstep(0.0, 0.15, luv) *
                          (1.0 - smoothstep(0.85, 1.0, luv));
                lbm_luv    = luv;
                lbm_wgt    = ef.x * ef.y;
                lbm_flow_v = texture(lbm_flow_tex, luv).xy * lbm_wgt;
            }
        }
    }
    if (dot(lbm_flow_v, lbm_flow_v) > 1e-4) {
        // Two-phase flowmap scroll: the detail noise is sampled at two
        // positions advected backwards along the current, cross-faded
        // so neither phase is ever visibly reset — the standard trick,
        // but the flow field comes from the LBM itself.
        float fph = fract(tile_params.time * 0.5);
        float n1 = warpedNoise(
            (pos.xz - lbm_flow_v * (fph * 2.0)) * 0.04334f);
        float n2 = warpedNoise(
            (pos.xz - lbm_flow_v * ((fph - 1.0) * 2.0)) * 0.04334f);
        noise = mix(n1, n2, abs(fph * 2.0 - 1.0));
    } else
#endif
    {
        noise = warpedNoise(pos.xz * 0.04334f);
    }
    float water_noise = (noise * 2.0f - 1.0f);

    vec3 water_normal;
    water_normal.xz = texture(water_normal_tex, in_data.world_map_uv).xy;
    vec2 water_flow = texture(water_flow_tex, in_data.world_map_uv).xy;
    water_normal.y = sqrt(1.0f - dot(water_normal.xz, water_normal.xz));
    water_normal.xz += water_flow * 0.5;
    water_normal.y += water_noise * 0.35;
    water_normal = normalize(water_normal);

#if defined(WATER_ATTR) || defined(WATER_LBM)
    // Blend the LBM ripple normal in where the camera-following patch
    // covers this fragment: the D2Q9 sim carries travelling waves,
    // wakes and rain-rings the procedural noise can't, and it fades
    // back to the noise normal at the patch edge so the handoff is
    // invisible.  Coverage/weight were resolved above.
    if (lbm_wgt > 0.0) {
        vec3 lbm_n = texture(lbm_surface_tex, lbm_luv).xyz;
        water_normal = normalize(
            mix(water_normal, lbm_n, 0.8 * lbm_wgt));
    }
#endif // WATER_ATTR || WATER_LBM
#ifdef WATER_ATTR
    {
        float water_linz = camera_info.depth_params.y /
                           (camera_info.depth_params.x + gl_FragCoord.z);
        out_glass_nr = vec4(octEncodeDir(water_normal),
                            water_linz,
                            0.06);          // near-mirror water
        // Absorption tint the resolve applies per metre of refracted
        // travel — deep river water pulls toward blue-green.
        out_glass_tint = vec4(kWaterTint, 1.0);
        return;
    }
#endif // WATER_ATTR

    vec2 screen_uv = gl_FragCoord.xy * tile_params.inv_screen_size;
    float dist_scale = length(vec3((screen_uv * 2.0f - 1.0f) * camera_info.depth_params.zw, 1.0f));

    float depth_z = texture(src_depth, screen_uv).r;
    float bg_view_dist = camera_info.proj[3].z / (depth_z + camera_info.proj[2].z) * dist_scale;

    vec3 view_vec = camera_info.position.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

    float water_ray_dist = max(bg_view_dist - view_dist, 0.0f);
    float distorted_water_ray_dist = water_ray_dist + noise * 0.5f;
    vec3 refract_ray = refract(-view, water_normal, 1.0 / 1.33);
    vec3 refract_pos = in_data.vertex_position + refract_ray * water_ray_dist;

    vec4 refracted_screen_pos = camera_info.view_proj * vec4(refract_pos, 1.0f);
    refracted_screen_pos.xy /= refracted_screen_pos.w;

    float fade_dist_1 = max(water_ray_dist / 1.0f, 0);
    float fade_dist_2 = max(distorted_water_ray_dist / 5.0f, 0);

    float fade_rate = exp(-fade_dist_1 * fade_dist_1);
    float thickness_fade_rate = exp(-fade_dist_2 * fade_dist_2);

    vec2 refract_uv = refracted_screen_pos.xy * 0.5 + 0.5;
    refract_uv.x = refract_uv.x < 0 ? -refract_uv.x : refract_uv.x;
    refract_uv.y = refract_uv.y < 0 ? -refract_uv.y : refract_uv.y;
    refract_uv.x = refract_uv.x > 1.0 ? 2.0f - refract_uv.x : refract_uv.x;
    refract_uv.y = refract_uv.y > 1.0 ? 2.0f - refract_uv.y : refract_uv.y;

    vec3 bg_color = texture(src_tex, refract_uv).xyz;

    // bump map
    vec3 normal = water_normal;

    vec3 albedo = vec3(0.11, 0.115, 0.15)*.75f;
    //albedo = mix(albedo, bg_color, thickness_fade_rate);

    MaterialInfo material_info;
    material_info.baseColor = albedo;

    vec3 f_diffuse = vec3(0);
    vec3 f_specular = vec3(0);

    float ior = 1.5;
    float f0_ior = 0.04;

    material_info.metallic = 0.9f;//material.metallic_factor;
    material_info.perceptualRoughness = 0.2f;//material.roughness_factor;

    // Achromatic f0 based on IOR.
    vec3 f0 = vec3(f0_ior);

    material_info.albedoColor = mix(material_info.baseColor.rgb * (vec3(1.0) - f0),  vec3(0), material_info.metallic);
    material_info.f0 = mix(f0, material_info.baseColor.rgb, material_info.metallic);

    #ifdef USE_IBL
    float mip_count = 10;
    f_specular += getIBLRadianceGGX(normal, view, material_info.perceptualRoughness, material_info.f0, mip_count);
    f_diffuse += getIBLRadianceLambertian(normal, material_info.albedoColor);
    #endif

    // ── Depth-difference blend ───────────────────────────────────────
    // Two depths drive the look:
    //  * distorted_water_ray_dist — the refracted OPTICAL path to the
    //    bed: Beer-Lambert absorption toward the deep tint, with the
    //    SAME per-metre constants the traced water path in
    //    deferred_resolve.comp uses, so the forward and traced paths
    //    agree wherever they hand off.  Shallow bed shows through,
    //    a few metres of water goes opaque blue-green.
    //  * water_depth — the vertical column at this fragment: the
    //    SHORELINE blend.  The old hard 3 cm cutoff drew the water's
    //    edge as a polygon boundary; fading the surface in over the
    //    last ~45 cm of depth lands it on the bank as a soft wet line.
    vec3 absorb = exp(-min(distorted_water_ray_dist, 24.0) *
                      kWaterExtinction * (vec3(1.05) - kWaterTint));
    // sceneTonemap: same exposure+ACES curve as every other final-colour
    // writer (bg_color is already display-encoded scene colour).
    // The DEEP body colour is mostly what "dark water" is: damp the
    // diffuse IBL hard (open water swallows skylight) and keep only
    // the specular sky reflection on top.
    vec3 deep_col = sceneTonemap(f_diffuse * 0.25 + f_specular * 0.8);
    vec3 water_col = mix(deep_col, bg_color, absorb);
    float shore = smoothstep(0.03f, 0.45f, in_data.water_depth);
    vec3 color = mix(bg_color, water_col, shore);
    outColor = vec4(color, 1.0f);
/*
	vec2 uv = gl_FragCoord.xy / vec2(1920, 1080) * 12.0;
    vec2 i = floor(uv);
    vec2 n = fract(uv);
    vec4 min_d = vec4(9.0);
    
    for (float y = -1.0; y <= 1.0; ++y) {
        for(float x = -1.0; x <= 1.0; ++x) {
            vec2 point = sin(tile_params.time + 32.0 * hash2D(i + vec2(x, y))) * 0.5 + 0.5;
            float d = length(vec2(x, y) + point - n);
            
            min_d = (d < min_d.x) ? vec4(d, min_d.xyz) 
               	 : (d < min_d.y) ? vec4(min_d.x, d, min_d.yz) 
               	 : (d < min_d.z) ? vec4(min_d.xy, d, min_d.z) 
               	 : (d < min_d.w) ? vec4(min_d.xyz, d) 
                 : min_d;
        }
    }
    outColor = vec4(vec3(1.0 - min_d.x), 1.0);*/
}