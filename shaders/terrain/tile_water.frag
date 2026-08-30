#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"
#include "..\functions.glsl.h"
#include "..\brdf.glsl.h"
#include "..\punctual.glsl.h"

#include "..\ibl.glsl.h"
#include "tile_common.glsl.h"
#include "..\weather\wind_field.glsl.h"

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


// ── Wind waves — SHADING ONLY ───────────────────────────────────────
// A small sum of directional wave trains, evaluated for its SLOPE and
// nothing else.  No vertex moves, the water height field is untouched
// and the LBM sim is not consulted: collision, buoyancy, the depth test
// at the top of main() and every gameplay query still see exactly the
// flat surface they saw before.  Only the normal shading uses changes.
//
// Each train contributes an analytic d(height)/d(world xz), so the cost
// is one sin/cos per train rather than a finite difference of the very
// expensive warped noise.  Deep-water dispersion (w = sqrt(g*k)) ties a
// train's speed to its wavelength, which is what keeps the stack from
// sliding across the surface as one rigid sheet.
// FALLBACK wind only.  The wave trains below take their direction and
// strength from windAt() (weather/wind_field.glsl.h) once the wind
// field is bound; this literal is what they fall back to before the
// patch is live, and it is the reason the waves currently blow the
// same way over the whole map regardless of weather.  See the
// WIND_FIELD block in main().
const vec2  kWindDir     = vec2(0.8575f, 0.5145f);  // normalised
const float kWaveLen0_M  = 46.0f;   // longest train's wavelength
const float kWaveAmp0_M  = 0.42f;   // and its amplitude
const float kWaveLenFall = 0.44f;   // wavelength ratio between trains
const float kWaveAmpFall = 0.55f;   // amplitude ratio between trains
const float kWaveFan     = 1.07f;   // rad each train is fanned off the wind
const float kWaveGain    = 1.15f;   // overall steepness trim
const int   kWaveTrains  = 5;

// ── Randomisation ───────────────────────────────────────────────────
// Evenly fanned trains of harmonic wavelength interfere into a regular
// crosshatch: it moves, but it reads as a TEXTURE rather than as water,
// and the repeat is obvious the moment you look across a wide river.
// Three independent scrambles break it, none of them per-frame random
// (everything is a function of world position and time, so the surface
// is stable and identical on every machine):
//
//   * PER-TRAIN HASH — irregular fan angles, non-harmonic wavelengths
//     and random starting phases, so no two trains ever line up into
//     a moire.
//   * DOMAIN WARP — a slow low-frequency offset of the sample point, so
//     crests meander instead of running dead straight bank to bank.
//   * GUST FIELD — a coarse noise patch drifting downwind that scales
//     the whole stack, so the surface is choppy in places and nearly
//     glassy between them.  Real wind leaves exactly these cat's-paws,
//     and the varying amplitude is what stops the eye finding a period.
//
// The warp and gust use single-octave smoothNoise, not the 4-octave
// warpedNoise above: they are 200-600 m features, so the extra octaves
// would cost four times as much to say the same thing.
const float kWaveSpread  = 0.85f;   // rad of random slop on the fan
const float kWarpScale   = 0.0045f; // domain-warp frequency (1/m)
const float kWarpAmp_M   = 9.0f;    // how far crests meander
const float kGustScale   = 0.0016f; // gust-patch frequency (1/m) ~625 m
const float kGustDrift   = 0.9f;    // m/s the gust field slides downwind
const float kGustFloor   = 0.35f;   // amplitude in a lull
const float kGustRange   = 1.25f;   // extra amplitude in a gust
const float kWindTurn    = 0.35f;   // rad the LOCAL wind swings
// smoothNoise hashes on (x + 27*y), so its lattice repeats along one
// diagonal.  Sampling the three fields in ROTATED domains puts each
// one's repeat on a different axis, and they stop reinforcing into a
// visible tiling.
const mat2  kDecorr      = mat2(0.936f, 0.352f, -0.352f, 0.936f);

// Returns the surface slope (dh/dx, dh/dz).  `footprint` is the world
// span this fragment covers: a train whose wavelength approaches the
// pixel can no longer be drawn, so instead of aliasing into shimmer it
// retires into `out_micro` and comes back as roughness.
vec2 windWaveSlope(vec2 p, float t, float footprint, float jitter,
                   vec2 wind_dir, float wind_gain,
                   out float out_micro) {
    // ── the three scrambles, resolved once for this fragment ────────
    vec2  gp   = (p - wind_dir * (t * kGustDrift)) * kGustScale;
    float gust = kGustFloor + kGustRange * smoothNoise(gp);
    float turn = kWindTurn *
                 (smoothNoise(kDecorr * gp * 0.5f + 11.3f) * 2.0f - 1.0f);
    vec2  w0   = p * kWarpScale;
    vec2  wp   = p + kWarpAmp_M *
                 (vec2(smoothNoise(w0),
                       smoothNoise(kDecorr * w0 + 37.2f)) * 2.0f - 1.0f);
    float cw = cos(turn), sw = sin(turn);
    vec2  wind = vec2(wind_dir.x * cw - wind_dir.y * sw,
                      wind_dir.x * sw + wind_dir.y * cw);

    vec2  slope = vec2(0.0f);
    float micro = 0.0f;
    float len   = kWaveLen0_M;
    float amp   = kWaveAmp0_M;
    for (int i = 0; i < kWaveTrains; ++i) {
        vec2  h   = hash2D(vec2(float(i) * 17.13f + 4.7f, 91.7f));
        float ang = kWaveFan * float(i) + (h.x - 0.5f) * kWaveSpread;
        float ca = cos(ang), sa = sin(ang);
        vec2  dir = vec2(wind.x * ca - wind.y * sa,
                         wind.x * sa + wind.y * ca);
        float tl = len * (0.82f + 0.36f * h.y);     // off the harmonic
        float k  = 6.28318530718f / tl;
        float w  = sqrt(9.81f * k);                 // deep-water dispersion
        // random start phase + a wander from the warped noise, so even
        // two trains of similar length never beat against each other
        float ph = k * dot(dir, wp) - w * t
                 + h.x * 6.28318530718f
                 + jitter * float(i + 1) * 0.35f;
        // h = amp * sharp^2 with sharp = 0.5 + 0.5*sin: a flatter
        // trough and a tighter crest than a plain sine, which is the
        // shape wind actually pushes water into.
        float sharp = 0.5f + 0.5f * sin(ph);
        float dh    = amp * k * sharp * cos(ph);    // dh/d(dot(dir,wp))
        float vis   = smoothstep(1.5f, 4.0f, tl / footprint);
        slope += dir * (dh * vis);
        micro += amp * k * (1.0f - vis);
        len *= kWaveLenFall;
        amp *= kWaveAmpFall;
    }
    out_micro = micro * gust * wind_gain;
    return slope * (kWaveGain * gust * wind_gain);
}


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
    // Debug: keep every fragment the water layer actually covers, so
    // the extent is not trimmed by the same threshold being judged.
    if (ps_depth < 0.002f) {
        discard;
    }
#else
    // Cull a fade-width BELOW the waterline, not at it: those fragments
    // carry the outer half of the blend, and discarding them would put
    // a hard step back exactly where the fade was supposed to be.  They
    // cost almost nothing — shore lands at 0 there, so the shader
    // writes the background colour it already sampled.
    if (ps_depth < shore_cull) {
        discard;
    }
#endif

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
    // ── Wind waves into the normal ──────────────────────────────────
    // A height field's normal is (-dh/dx, 1, -dh/dz), so the wave
    // slopes have to STEER the normal in xz.  The line above only ever
    // scaled y, which changes how far the normal is tilted but never
    // which way it faces — no amount of it can draw a crest, which is
    // why the surface read as glass.  The trains ride on top of the
    // flow-map tilt, so a river's current still shapes them.
    // Waves shoal out at the bank: full height in open water, flat by
    // the time the column is a few tens of cm, so the surface still
    // meets the shore as the same soft wet line.
    float wave_micro;
    float footprint  = max(length(fwidth(pos.xz)), 0.02f);
    float shore_att  = smoothstep(0.05f, 0.90f, ps_depth);
    // ── Wind from the SIM, constants as fallback ────────────────────
    // Where the local patch covers this fragment, the wave direction
    // is the simulated wind's and the amplitude scales with its speed
    // (calm lake in a lull, whitecap chop in a blow — ~6 m/s is the
    // look the constants were tuned at).  The blend runs on the
    // patch's own feathered weight, so the handoff to the constant
    // wind outside the patch is the same invisible seam the LBM
    // ripples already use.
    float wf_w;
    vec2  wf_v = sampleWindFine(wind_patch_tex, wind_region,
                                pos.xz, wf_w);
    vec2  wdir; float wspd;
    windDirSpeed(vec3(wf_v.x, 0.0, wf_v.y), wdir, wspd);
    vec2  wind_dir  = normalize(mix(kWindDir, wdir, wf_w));
    float wind_gain = mix(1.0f, clamp(wspd * (1.0f / 6.0f), 0.15f, 1.8f),
                          wf_w);
    vec2  wave_slope = windWaveSlope(pos.xz, tile_params.time, footprint,
                                     water_noise * 2.2f,
                                     wind_dir, wind_gain, wave_micro);
    water_normal.xz -= wave_slope * shore_att;
    water_normal = normalize(water_normal);
    // Trains the footprint retired come back as roughness, so distance
    // dulls the specular into a haze instead of aliasing it into
    // sparkle — the same trade LEAN/Toksvig mapping makes.
    float wave_rough = clamp(wave_micro * shore_att * 0.6f, 0.0f, 0.28f);

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
                            // near-mirror water, roughened by the
                            // sub-pixel wave detail
                            clamp(0.06f + wave_rough, 0.02f, 0.34f));
        // Absorption tint the resolve applies per metre of refracted
        // travel — deep river water pulls toward blue-green.
        out_glass_tint = vec4(waterTint(), 1.0);
        return;
    }
#endif // WATER_ATTR

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
    float distorted_water_ray_dist = water_ray_dist + noise * 0.5f;
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
    vec3 deep_col = sceneTonemap(f_diffuse * waterDeepDiffuse() + f_specular * 0.8);
#if WATER_DEBUG_OPAQUE || WATER_DEBUG_DEPTH_RAMP
    // Nothing of the bed comes through and the waterline does not fade:
    // the surface is drawn wherever it exists, at full strength.
    absorb = vec3(0.0f);
#endif
    vec3 water_col = mix(deep_col, bg_color, absorb);
    // Anchored to the cull, not to (edge - fade): when the clamp above
    // bites, those two differ, and a ramp that started below the cull
    // would jump straight to a non-zero value at the first surviving
    // fragment — the hard edge, back again.
    float shore = smoothstep(shore_cull,
                             shore_cull + 2.0f * shore_fade, ps_depth);
#if WATER_DEBUG_OPAQUE || WATER_DEBUG_DEPTH_RAMP
    shore = 1.0f;
#endif
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
    vec3 color = mix(bg_color, water_col, shore * waterOpacity());
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