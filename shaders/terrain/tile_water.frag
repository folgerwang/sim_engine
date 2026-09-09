#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"
#include "..\water_waves.glsl.h"
#include "..\functions.glsl.h"
#include "..\brdf.glsl.h"
#include "..\punctual.glsl.h"

#include "..\ibl.glsl.h"
#include "tile_common.glsl.h"
#include "..\weather\wind_field.glsl.h"
#include "..\underwater.glsl.h"   // the surface seen from below

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
const float kWaterExtinction = 0.55;
// The along-view path is capped at the water COLUMN divided by this
// cosine, i.e. at what the path would be looking in at ~70 degrees off
// vertical.  Without a cap, a low camera over a wide river sees a path
// of hundreds of metres through water a metre deep, absorption
// saturates, and every part of the surface past the first few metres
// goes uniformly opaque — no shallow-to-deep gradient anywhere.
const float kMinViewCos = 0.35;
// Ceiling on how clear the water may get.  Even an inch of it keeps
// this much body colour, so the shallows read as WATER over sand
// rather than as bare wet ground with a specular highlight.
const float kMaxClarity = 0.72;

// ── DEBUG SWITCHES — set back to 0 before shipping ──────────────────
// 1 = draw the water surface FULLY OPAQUE: no transmission, no
// shoreline fade, and the depth cutoff drops to a hair above zero, so
// every fragment the water layer covers is painted.  What the surface
// COVERS becomes legible — extent, silhouette, where it laps onto the
// bank, which channels exist at all — none of which you can judge
// through a transmission term that hides the thin ones.
#define WATER_DEBUG_OPAQUE                  0
// 1 = paint the water by COLUMN DEPTH instead of shading it, as a
// stepped ramp (implies the opaque mode above).  Reads directly as the
// bathymetry: black at the waterline, then blue / cyan / green /
// yellow / red at 0.25, 1, 2, 4, 8 m.  This is the view that answers
// "how flat is the bed" without leaving the engine.
#define WATER_DEBUG_DEPTH_RAMP              0

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

layout(set = TILE_PARAMS_SET, binding = SRC_COLOR_TEX_INDEX) uniform sampler2D src_tex;
layout(set = TILE_PARAMS_SET, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;
layout(set = TILE_PARAMS_SET, binding = WATER_NORMAL_BUFFER_INDEX) uniform sampler2D water_normal_tex;
layout(set = TILE_PARAMS_SET, binding = WATER_FLOW_BUFFER_INDEX) uniform sampler2D water_flow_tex;
// Local wind patch (WindField): per-cell wind m/s + its region.  The
// region is DEAD (w = 0) until the sim is live, and sampleWindFine
// returns weight 0 then — so kWindDir below stays the fallback and no
// frame ever reads an unstepped lattice.
layout(set = TILE_PARAMS_SET, binding = WIND_TEX_INDEX) uniform sampler2D wind_patch_tex;
layout(std430, set = TILE_PARAMS_SET, binding = WIND_REGION_BUFFER_INDEX) readonly buffer WindRegionBuf {
    vec4 wind_region;
};

// The water column, sampled PER PIXEL.  in_data.water_depth is the same
// quantity interpolated from the vertices, and that is what made the
// shoreline jagged: across a triangle the interpolation is LINEAR, so
// the depth==0 contour is a straight chord and the waterline came out
// as a chain of facets at tile-tessellation scale, not a curve.  Read
// from the (bilinear) layer texture instead and the contour follows the
// field at texel resolution.  The VERTEX shader still displaces the
// mesh by its own interpolated value; this only decides shading, so the
// two never need to agree to the millimetre.
layout(set = TILE_PARAMS_SET, binding = SOIL_WATER_LAYER_BUFFER_INDEX) uniform sampler2D soil_water_layer_ps;

// Depth at which the water's edge sits, and the minimum width of the
// fade across it.
const float kShoreEdgeM  = 0.02;
// Screen-space anti-aliasing for that contour.  A depth threshold is a
// step function, so on a shallow beach it lands entirely inside one
// pixel however smooth the underlying field is — the classic hard edge.
// Widening the fade by fwidth() of the depth spreads it over a fixed
// number of PIXELS regardless of how steeply the bed falls away, which
// is the same trick analytic alpha-test AA uses.
const float kShoreAaPx   = 1.75;
// Floor on the fade, in metres of column, so a dead-flat shelf (where
// fwidth is ~0 and the AA term vanishes) still fades rather than
// snapping on.
const float kShoreMinFade = 0.05;
// CEILING on it, which the first version of this did not have, and the
// omission drowned the world.  The cull was written as
// (kShoreEdgeM - shore_fade) so the fade could reach below the
// waterline; with a 0.05 m floor that expression is already NEGATIVE
// (0.02 - 0.05), depth is never negative, so nothing was ever culled
// and every terrain fragment on the map rendered as water.  A cap also
// keeps a grazing view — where fwidth runs to metres per pixel — from
// widening the blend until ankle-deep shallows disappear into it.
const float kShoreMaxFade = 0.20;
// Hard floor for the cull.  Ground the layer says is DRY must always be
// discarded no matter how wide the fade wants to be; the AA is only ever
// allowed to eat into water that genuinely exists.
const float kShoreCullMinM = 0.005;

// ── Runtime overrides ────────────────────────────────────────────────
// Everything above is the DEFAULT and the reasoning for it; these read
// the live values pushed from Settings > Water (glsl::WaterBlendParams,
// seeded from those same constants in TileObject).  A zero-initialised
// push constant would render the water invisible, so the guards below
// fall back to the constants if the block was never filled -- which is
// also what keeps any draw path that does not set them looking right.
// shore.w is the "this block was filled in" flag, deliberately NOT one
// of the tunables: using a tunable as its own sentinel would make the
// zero end of that slider silently mean "ignore me, use the default" --
// exactly the trap an opacity slider that cannot reach 0 would be.
bool waterParamsPushed() {
    return tile_params.water.shore.w > 0.5f;
}
// Runtime twin of WATER_DEBUG_OPAQUE (Rendering > Terrain > "Water
// surface only"): shore.w = 2 from packWaterBlend.
bool waterDebugOpaque() {
    return tile_params.water.shore.w > 1.5f && tile_params.water.shore.w < 2.5f;
}
float waterMaxClarity() {
    return waterParamsPushed()
        ? clamp(tile_params.water.blend.x, 0.0f, 1.0f) : kMaxClarity;
}
float waterExtinction() {
    return waterParamsPushed()
        ? max(tile_params.water.blend.y, 0.0f) : kWaterExtinction;
}
float waterDepthScale() {
    return waterParamsPushed()
        ? max(tile_params.water.blend.z, 1e-3f) : 1.0f;
}
float waterOpacity() {
    return waterParamsPushed()
        ? clamp(tile_params.water.blend.w, 0.0f, 1.0f) : 1.0f;
}
vec3 waterTint() {
    return waterParamsPushed()
        ? clamp(tile_params.water.tint.rgb, vec3(0.0f), vec3(1.0f))
        : kWaterTint;
}
float waterDeepDiffuse() {
    return waterParamsPushed()
        ? max(tile_params.water.tint.w, 0.0f) : 0.25f;
}
float waterShoreEdgeM() {
    return waterParamsPushed()
        ? max(tile_params.water.shore.x, 0.0f) : kShoreEdgeM;
}
float waterShoreFadeScale() {
    return waterParamsPushed()
        ? max(tile_params.water.shore.y, 0.0f) : kShoreAaPx;
}
float waterShoreFadeMaxM() {
    return waterParamsPushed()
        ? max(tile_params.water.shore.z, kShoreMinFade) : kShoreMaxFade;
}
// Ceiling on the reconstructed distance through water, in metres.  Both
// a sentinel for an unusable depth sample and a hard bound on the
// refraction offset.
const float kMaxWaterRayM = 512.0;

void main() {
    // Per-pixel column + the width of the shoreline fade around it.
    float ps_depth  = texture(soil_water_layer_ps, in_data.world_map_uv).y *
                      SOIL_WATER_LAYER_MAX_THICKNESS;
    float shore_fade = clamp(waterShoreFadeScale() * fwidth(ps_depth),
                             kShoreMinFade, waterShoreFadeMaxM());
    // Where the surface stops existing.  Clamped positive so dry ground
    // is always culled — see kShoreCullMinM.
    float shore_cull = max(waterShoreEdgeM() - shore_fade, kShoreCullMinM);
#if WATER_DEBUG_OPAQUE || WATER_DEBUG_DEPTH_RAMP
    const bool debug_opaque = true;
#else
    const bool debug_opaque = waterDebugOpaque();
#endif
    if (debug_opaque) {
        // Debug: keep every fragment the water layer actually covers, so
        // the extent is not trimmed by the same threshold being judged.
        if (ps_depth < 0.002f) {
            discard;
        }
    } else {
        // Cull a fade-width BELOW the waterline, not at it: those
        // fragments carry the outer half of the blend, and discarding
        // them would put a hard step back exactly where the fade was
        // supposed to be.  They cost almost nothing — shore lands at 0
        // there, so the shader writes the background colour it already
        // sampled.
        if (ps_depth < shore_cull) {
            discard;
        }
    }

    float transparent_factor = clamp((in_data.water_depth - 0.03f) / 0.03f, 0.0f, 1.0f);

    vec3 pos = in_data.vertex_position;
    // The authored level is the reference plane. Both geometry and normals
    // use the same bounded LBM height offsets, with no unrelated FBM tilt.
    vec3 water_normal = vec3(0.0, 1.0, 0.0);
    float wave_rough = 0.08;
    // World size of this pixel: the band limit for every wave term
    // below, sim or analytic.
    float footprint = max(length(fwidth(pos.xz)), 0.02);
    // The surface slope this fragment shades with.  Terms ADD into it —
    // see the wind-wave block below for why slopes and not normals.
    vec2 slope = vec2(0.0);
#if defined(WATER_ATTR) || defined(WATER_LBM)
    float cell = max(lbm_region.y, 0.05);
    float step_m = max(cell, footprint);
    vec2 dx = vec2(step_m, 0.0);
    vec2 dz = vec2(0.0, step_m);
    vec2 du = dx * tile_params.inv_world_range;
    vec2 dv = dz * tile_params.inv_world_range;
    float dl = texture(soil_water_layer_ps, in_data.world_map_uv - du).y * SOIL_WATER_LAYER_MAX_THICKNESS;
    float dr = texture(soil_water_layer_ps, in_data.world_map_uv + du).y * SOIL_WATER_LAYER_MAX_THICKNESS;
    float db = texture(soil_water_layer_ps, in_data.world_map_uv - dv).y * SOIL_WATER_LAYER_MAX_THICKNESS;
    float df = texture(soil_water_layer_ps, in_data.world_map_uv + dv).y * SOIL_WATER_LAYER_MAX_THICKNESS;
    float left = waterWaveOffset(lbm_surface_tex, lbm_region, pos.xz - dx, dl);
    float right = waterWaveOffset(lbm_surface_tex, lbm_region, pos.xz + dx, dr);
    float back = waterWaveOffset(lbm_surface_tex, lbm_region, pos.xz - dz, db);
    float front = waterWaveOffset(lbm_surface_tex, lbm_region, pos.xz + dz, df);
    // Unresolved lattice ripples become roughness rather than grazing-angle
    // stripes. The shoreline/patch feather is included in the derivative.
    float resolved = 1.0 - smoothstep(1.5 * cell, 5.0 * cell, footprint);
    // `resolved` belongs to the LATTICE slope only — it is the LBM
    // cell's own resolvability — so it is folded in here rather than
    // applied to the sum: the analytic waves below band-limit themselves
    // per octave, and multiplying them by this would delete the wind sea
    // exactly where the lattice runs out, which is the case this is for.
    slope += vec2(right - left, front - back) / (2.0 * step_m) * resolved;
    wave_rough += 0.04 * (1.0 - resolved);
#endif
    // ── WIND WAVES, ADDED TO WHATEVER THE SIM PRODUCED ──────────────
    // The LBM patch is ~500 m wide and only stirs where it has current;
    // everywhere else the slope above is exactly zero and the surface is
    // a mirror.  The analytic spectrum (water_waves.glsl.h) is the open-
    // water sea state and it exists at every pixel of water on the map,
    // so the two are SUMMED — slopes, not normals, which is the one
    // combination that is associative and cannot produce a seam where
    // the patch ends: outside it the sim contributes 0 and nothing about
    // the wind waves changes as the boundary crosses the screen.  (The
    // same additive-slope rule the near-camera seam fix settled on.)
    //
    // Outside the #if on purpose: the plain forward permutation has no
    // sim at all, and "no sim" is not a reason for water to be glass.
    //
    // Depth-gated: a wave cannot be taller than the water it is in, so
    // the sea state fades out through the shallows and the last few
    // centimetres at the shoreline are flat, as they are in life.
    {
        float wind_w = 0.0;
        vec2  wind_v = sampleWindFine(wind_patch_tex, wind_region,
                                      pos.xz, wind_w);
        // The patch is dead (or this pixel is outside it): fall back to
        // the same fixed breeze the vegetation sway uses.
        vec2  wind_dir = (wind_w > 0.01 && dot(wind_v, wind_v) > 1e-4)
                             ? normalize(wind_v) : kWaterWindFallback;
        float wind_ms = mix(4.5, length(wind_v), clamp(wind_w, 0.0, 1.0));
        // Sea state from wind speed: nothing below a light air, full
        // spectrum by ~9 m/s, and capped so it cannot run away.
        float sea = clamp((wind_ms - 0.8) / 8.0, 0.0, 1.25);
        float depth_gate = smoothstep(0.06, 0.55, ps_depth);
        WaterWaveField ww = waterWindWaves(pos.xz, tile_params.time,
                                           wind_dir, sea * depth_gate,
                                           footprint);
        slope += ww.slope;
        // Sub-footprint chop is roughness, not geometry (see the header).
        wave_rough += clamp(ww.unresolved * 0.28, 0.0, 0.13);
    }
    water_normal = normalize(vec3(-slope.x, 1.0, -slope.y));
#ifdef WATER_ATTR
    {
        float water_linz = camera_info.depth_params.y /
                           (camera_info.depth_params.x + gl_FragCoord.z);
        out_glass_nr = vec4(octEncodeDir(water_normal),
                            water_linz,
                            // near-mirror water, roughened by the
                            // sub-pixel wave detail
                            clamp(0.06f + wave_rough, 0.02f, 0.34f));
        // Absorption tint the resolve applies per metre of refracted
        // travel — deep river water pulls toward blue-green.
        out_glass_tint = vec4(waterTint(), 1.0);
        return;
    }
#endif // WATER_ATTR

    // ── The surface seen from BELOW (underwater.glsl.h) ─────────────
    // The eye is under water and this fragment is above it: the
    // underside of the surface.  Inside Snell's window (the refracted
    // ray, water -> air, exists) it shows what is up there -- the
    // resolved scene where the pixel's own column holds geometry (the
    // shore, already fogged by the resolve along the eye-to-surface
    // path), else the sky along the refracted ray -- blended by the
    // internal Fresnel with the reflection of the water's own gloom;
    // past the critical angle it is all reflection.  Then the water
    // between the eye and the surface fogs it, like every opaque pixel
    // the resolve fogged.  Written in display space like the rest of
    // this pass (src_tex is already tonemapped).
    if (camera_info.underwater_depth > 0.0f &&
        in_data.vertex_position.y > camera_info.position.y + 0.02f) {
        vec3  eye   = camera_info.position.xyz;
        vec3  to_s  = in_data.vertex_position - eye;
        float dist  = length(to_s);
        vec3  ray   = to_s / max(dist, 1e-4f);
        vec3  n_up  = water_normal;                // out of the water
        float expo  = sceneExposureScaleOf(camera_info.exposure_scale);
        vec3  insc  = uwInscatter(camera_info.underwater_depth);
        // The reflected side: the water below, seen mirrored -- its own
        // scattered light, a shade darker than the fog since it looks
        // down into deeper, dimmer water.
        vec3  below = insc * 0.55f;
        // refract() wants the normal on the incident side: below.
        vec3  t_dir = refract(ray, -n_up, kUwIor);
        float F     = 1.0f;                        // past the critical angle
        vec3  col;
        vec2  suv   = gl_FragCoord.xy * tile_params.inv_screen_size;
        bool  geom_above = texture(src_depth, suv).r < 1.0f;
        if (dot(t_dir, t_dir) > 1e-6f) {
            t_dir = normalize(t_dir);
            float cos_t = clamp(dot(t_dir, n_up), 0.0f, 1.0f);
            F = 0.02f + 0.98f * pow(1.0f - cos_t, 5.0f);
        }
        if (F < 1.0f && geom_above) {
            // The shore through the window: the resolve already fogged
            // it along this ray's underwater length, so only the
            // reflection is added here.
            vec3 shore = texture(src_tex, suv).rgb;
            col = mix(shore, sceneTonemapExposed(uwFog(below, dist, insc), expo), F);
        } else {
            vec3 above = (F < 1.0f) ? uwSkyRadiance(t_dir) : vec3(0.0f);
            vec3 lin   = mix(above, below, F);
            col = sceneTonemapExposed(uwFog(lin, dist, insc), expo);
        }
        outColor = vec4(col, 1.0f);
        return;
    }

    vec2 screen_uv = gl_FragCoord.xy * tile_params.inv_screen_size;
    float dist_scale = length(vec3((screen_uv * 2.0f - 1.0f) * camera_info.depth_params.zw, 1.0f));

    // ── Depth reconstruction, guarded ───────────────────────────────
    // A texel of the depth COPY that still holds the clear value (sky,
    // or anywhere the copy did not cover) drives this denominator to
    // zero, and the unguarded divide produced +Inf.  That Inf then ran
    // the whole refraction chain into a NaN — see the fold below — and
    // the NaN sampled src_tex as BLACK.  It stayed invisible while the
    // water was opaque and absorb was ~0; the moment the shallows got a
    // real transmission term it surfaced as hard-edged black blobs on
    // the beach.  Kill it at the source: no Inf, no NaN downstream.
    float depth_z = texture(src_depth, screen_uv).r;
    float depth_den = depth_z + camera_info.proj[2].z;
    float bg_view_dist =
        (abs(depth_den) > 1e-6f)
            ? camera_info.proj[3].z / depth_den * dist_scale
            : kMaxWaterRayM;

    vec3 view_vec = camera_info.position.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

    // Bounded, not just floored at 0: the far plane is kilometres out,
    // and pushing the refraction origin that far turns the reprojection
    // below into garbage even when it stays finite.
    float water_ray_dist =
        clamp(bg_view_dist - view_dist, 0.0f, kMaxWaterRayM);
    float distorted_water_ray_dist = water_ray_dist;
    vec3 refract_ray = refract(-view, water_normal, 1.0 / 1.33);
    vec3 refract_pos = in_data.vertex_position + refract_ray * water_ray_dist;

    vec4 refracted_screen_pos = camera_info.view_proj * vec4(refract_pos, 1.0f);
    // Behind the eye or on the plane: reprojection is meaningless, so
    // fall back to the unrefracted sample rather than dividing by ~0.
    bool refract_ok = refracted_screen_pos.w > 1e-4f;
    refracted_screen_pos.xy /= refract_ok ? refracted_screen_pos.w : 1.0f;

    float fade_dist_1 = max(water_ray_dist / 1.0f, 0);
    float fade_dist_2 = max(distorted_water_ray_dist / 5.0f, 0);

    float fade_rate = exp(-fade_dist_1 * fade_dist_1);
    float thickness_fade_rate = exp(-fade_dist_2 * fade_dist_2);

    // Mirror-fold back into [0,1].  The old version did this with four
    // ternaries on `<` and `>`, and a NaN FAILS EVERY COMPARISON — so a
    // poisoned coordinate took the else-branch of all four and passed
    // through untouched, straight into the sampler.  Folding with
    // abs/clamp instead means a bad value is bounded rather than
    // preserved, and refract_ok drops it entirely.
    vec2 refract_uv = refracted_screen_pos.xy * 0.5 + 0.5;
    // Off-screen refraction: fall back to the pixel's own column.  The
    // mirror-fold below is for coordinates a few pixels over the edge;
    // letting it wrap a WAY off-screen point re-samples the shoreline
    // and sky back onto open water, which smears them across the
    // surface in wave-shaped streaks.
    if (any(lessThan(refract_uv, vec2(-0.25f))) ||
        any(greaterThan(refract_uv, vec2(1.25f)))) {
        refract_uv = screen_uv;
    }
    refract_uv = abs(refract_uv);
    refract_uv = 1.0f - abs(1.0f - refract_uv);
    refract_uv = clamp(refract_uv, 0.0f, 1.0f);
    if (!refract_ok) {
        refract_uv = screen_uv;
    }

    // ── Bound the refraction offset by DISTANCE ─────────────────────
    // The reprojection above is honest geometry, and that is the
    // problem: at a grazing view the refracted ray runs nearly
    // horizontal, water_ray_dist is allowed up to kMaxWaterRayM, and
    // the reprojected point lands screens away from the pixel that
    // asked.  Sampling there drags whatever it hits — far bank, sky —
    // across the surface, stretched along the wave pattern; that is
    // exactly the "stretched reflection" streaking.  Refraction is a
    // NEAR-FIELD effect: what the eye actually sees at range is the
    // ripple distorting the pixel's own neighbourhood by at most a few
    // pixels.  So the offset budget shrinks with distance — ~12% of
    // the screen up close, under a pixel or two far away — and the
    // direction is kept while the magnitude is clamped.
    {
        vec2  r_ofs   = refract_uv - screen_uv;
        float max_ofs = 0.12f / (1.0f + view_dist * 0.08f);
        float r_len   = length(r_ofs);
        if (r_len > max_ofs) {
            refract_uv = screen_uv + r_ofs * (max_ofs / max(r_len, 1e-5f));
        }
    }

    // ── Reject samples from IN FRONT of the water ───────────────────
    // The offset sample can still land on geometry nearer the camera
    // than this fragment — a bank, a tree, a boat — and painting that
    // into the water reads as the shore bleeding into the surface.
    // Nothing between the camera and the water can be what a refracted
    // ray sees, so compare depths and fall back to the pixel's own
    // column when the sample is closer than the surface itself.
    {
        float r_z   = texture(src_depth, refract_uv).r;
        float r_den = r_z + camera_info.proj[2].z;
        if (abs(r_den) > 1e-6f) {
            float r_dist = camera_info.proj[3].z / r_den * dist_scale;
            if (r_dist < view_dist - 0.05f) {
                refract_uv = screen_uv;
            }
        }
    }

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
    // base gloss + whatever wave detail was too small to resolve
    material_info.perceptualRoughness = clamp(0.2f + wave_rough * 0.6f,
                                              0.0f, 0.5f);

    // Achromatic f0 based on IOR.
    vec3 f0 = vec3(f0_ior);

    material_info.albedoColor = mix(material_info.baseColor.rgb * (vec3(1.0) - f0),  vec3(0), material_info.metallic);
    material_info.f0 = mix(f0, material_info.baseColor.rgb, material_info.metallic);

    #ifdef USE_IBL
    float mip_count = 10;
    f_specular += getIBLRadianceGGX(normal, view, material_info.perceptualRoughness, material_info.f0, mip_count);
    f_diffuse += getIBLRadianceLambertian(normal, material_info.albedoColor);
    #endif

    // ── Depth blend ─────────────────────────────────────────────────
    // Three depths drive the look:
    //
    //  * distorted_water_ray_dist — the refracted OPTICAL path to the
    //    bed.  Correct in principle, useless on its own: at a grazing
    //    view it runs to hundreds of metres over water that is ankle
    //    deep, so Beer-Lambert saturates and the whole river renders
    //    as one flat opaque sheet with no shallow-to-deep gradient.
    //
    //  * water_depth / cos(view from vertical) — the CAP on that path.
    //    Clamping the view cosine at kMinViewCos ties the absorbed
    //    distance to the vertical column, which is the depth the eye
    //    actually reads, while still letting a steep look-down use the
    //    true (shorter) path.  This is the whole fix: absorption now
    //    ramps over roughly 0.2 m to 2.5 m of column instead of being
    //    saturated everywhere past the water's edge.
    //
    //  * water_depth — the SHORELINE fade.  A hard cutoff drew the
    //    water's edge as a polygon boundary, so the surface fades in
    //    over the last few CENTIMETRES of column.  It has to stay that
    //    tight: this ramp exists to soften the WATERLINE, and 55 cm of
    //    it is deeper than an entire stream — a 20 cm brook came out
    //    ~30% opacity along its whole width and read as a blue stain on
    //    the grass instead of as water.  Anything the eye should see as
    //    a water SURFACE must be fully faded in well before the
    //    shallowest channel on the map.
    float cos_v    = max(abs(view.y), kMinViewCos);
    // The depth scale multiplies the COLUMN, not the optical path: it is
    // there to say "this map's water is modelled shallower than it
    // should read", and the cap is the term the column drives.
    float path_cap = (ps_depth * waterDepthScale()) / cos_v;
    float path     = min(min(distorted_water_ray_dist, path_cap), 24.0);
    // Beer-Lambert toward the deep tint, floored by kMaxClarity so the
    // shallows never go fully see-through.
    vec3 absorb = min(exp(-path * waterExtinction() *
                          (vec3(1.05) - waterTint())),
                      vec3(waterMaxClarity()));
    // sceneTonemap: same exposure+ACES curve as every other final-colour
    // writer (bg_color is already display-encoded scene colour).
    // The DEEP body colour is mostly what "dark water" is: damp the
    // diffuse IBL hard (open water swallows skylight) and keep only
    // the specular sky reflection on top.
    vec3 deep_col = sceneTonemapExposed(
        f_diffuse * waterDeepDiffuse() + f_specular * 0.8,
        sceneExposureScaleOf(camera_info.exposure_scale));
    if (debug_opaque) {
        // Nothing of the bed comes through and the waterline does not
        // fade: the surface is drawn wherever it exists, at full
        // strength.
        absorb = vec3(0.0f);
    }
    vec3 water_col = mix(deep_col, bg_color, absorb);
    // Anchored to the cull, not to (edge - fade): when the clamp above
    // bites, those two differ, and a ramp that started below the cull
    // would jump straight to a non-zero value at the first surviving
    // fragment — the hard edge, back again.
    float shore = smoothstep(shore_cull,
                             shore_cull + 2.0f * shore_fade, ps_depth);
    if (debug_opaque) shore = 1.0f;
#if WATER_DEBUG_DEPTH_RAMP
    {
        // Stepped, not smooth: a continuous ramp makes a flat bed and a
        // gently graded one look alike, which is exactly the difference
        // being measured.  Bands are where the eye can count them.
        float dm = ps_depth;
        vec3 band = dm < 0.25f ? vec3(0.05f, 0.05f, 0.08f)
                  : dm < 1.0f  ? vec3(0.10f, 0.25f, 0.85f)
                  : dm < 2.0f  ? vec3(0.10f, 0.75f, 0.85f)
                  : dm < 4.0f  ? vec3(0.20f, 0.80f, 0.25f)
                  : dm < 8.0f  ? vec3(0.95f, 0.80f, 0.15f)
                               : vec3(0.90f, 0.20f, 0.15f);
        // a little of the surface shading kept on top, so waves and the
        // shoreline still read while the band carries the depth
        water_col = band * (0.82f + 0.18f * dot(deep_col, vec3(0.333f)));
    }
#endif
    // shore is the WATERLINE fade (centimetres of column); opacity is the
    // master "how present is this surface at all" control.  Multiplying
    // rather than replacing keeps the soft edge at every setting.
    vec3 color = mix(bg_color, water_col,
                     debug_opaque ? 1.0f : shore * waterOpacity());
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