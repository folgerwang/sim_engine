#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "../global_definition.glsl.h"
#include "../weather/weather_common.glsl.h"
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

layout(location = 0) in TileVsPsData in_data;

#ifdef GBUFFER_OUTPUT
// Deferred path: material attributes go into the cluster G-buffer layout
// (same 4 RTs cluster_bindless.frag's GBUFFER_OUTPUT branch writes;
// consumed by deferred_resolve.comp).  No lighting happens in this
// shader in that mode.
layout(location = 0) out vec4 out_albedo_ao;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_emissive_metal;
layout(location = 3) out vec2 out_velocity;
#else
layout(location = 0) out vec4 outColor;
#endif

vec3  kSunDir = vec3(-0.624695f, 0.468521f, -0.624695f);

// Octahedral encode — identical to cluster_bindless.frag's copy, paired
// with octDecode in deferred_resolve.comp.
vec2 octEncodeDir(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 oct = (n.z >= 0.0) ? n.xy
                            : (1.0 - abs(n.yx)) * vec2(
                                  n.x >= 0.0 ? 1.0 : -1.0,
                                  n.y >= 0.0 ? 1.0 : -1.0);
    return oct * 0.5 + 0.5;
}

layout(set = TILE_PARAMS_SET, binding = SRC_COLOR_TEX_INDEX) uniform sampler2D src_tex;
layout(set = TILE_PARAMS_SET, binding = SRC_DEPTH_TEX_INDEX) uniform sampler2D src_depth;
layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;

#include "tile_detail.glsl.h"

// 1 m albedo detail tiles (fragment-only binding — streamed with the
// height tiles; RGBA8 2048^2 per slot).
layout(set = TILE_PARAMS_SET, binding = TERRAIN_DETAIL_COLOR_INDEX)
    uniform sampler2DArray detail_color_tiles;

// One detail-colour tap.  w = 0 when the tile is out of range or not
// resident.  textureLod(0) rather than texture(): the array is created
// with num_mips = 1 (terrain_detail_stream.cpp), so there is no LOD to
// select and the explicit form is identical — but it is also safe in the
// divergent control flow below, where implicit derivatives would not be.
vec4 terrainDetailTap(vec2 p) {
    vec2 rel = (p + vec2(kTerrainMapMeters * 0.5f)) / kDetailTileMeters;
    ivec2 t = ivec2(floor(rel));
    if (any(lessThan(t, ivec2(0))) ||
        any(greaterThanEqual(t, ivec2(kDetailTilesPerSide))))
        return vec4(0.0f);
    int slot = detail_color_slot[t.y * kDetailTilesPerSide + t.x];
    if (slot < 0) return vec4(0.0f);
    vec2 uv = rel - vec2(t);
    return vec4(textureLod(detail_color_tiles,
                           vec3(uv, float(slot)), 0.0f).rgb, 1.0f);
}

// Near-field surface colour from the 1 m tile if resident; `fallback`
// (global albedo) otherwise / beyond the fade band.
//
// blur_dir/blur_m implement ANISOTROPIC filtering along the slope.  The
// tile is indexed by world XZ, so on a steep face one XZ metre covers
// 1/n.y metres of surface ALONG THE GRADIENT ONLY — the along-contour
// direction is not stretched at all.  A mip bias cannot help here (the
// array has a single mip), and an isotropic blur would throw away the
// good direction, so we low-pass with a few taps along the gradient and
// leave the perpendicular axis untouched.  That is exactly what a correct
// anisotropic sample would do, and it is what stops the macro colour
// being dragged into 40 m vertical streaks down a cliff.
vec3 terrainDetailAlbedo(vec2 pos_xz_ws, vec3 fallback, float fade,
                         vec2 blur_dir, float blur_m) {
    if (fade <= 0.0f) return fallback;
    vec4 acc = terrainDetailTap(pos_xz_ws) * 3.0f;
    if (blur_m > 0.05f) {          // flat ground keeps the single tap
        vec2 d1 = blur_dir * (blur_m * 0.5f);
        vec2 d2 = blur_dir * blur_m;
        acc += (terrainDetailTap(pos_xz_ws - d1)
              + terrainDetailTap(pos_xz_ws + d1)) * 2.0f;
        acc +=  terrainDetailTap(pos_xz_ws - d2)
              + terrainDetailTap(pos_xz_ws + d2);
    }
    if (acc.w <= 0.0f) return fallback;
    return mix(fallback, acc.rgb / acc.w, fade);
}

// 1 m packed PBR surface detail tiles, streamed alongside the colour
// tiles: R,G = tangent normal XY (Z reconstructed), B = roughness,
// A = micro-occlusion.  One image rather than a normal array plus an ORM
// array because nine 2048^2 RGBA8 slots stay resident whether or not the
// camera is looking at any of them.
layout(set = TILE_PARAMS_SET, binding = TERRAIN_DETAIL_SURFACE_INDEX)
    uniform sampler2DArray detail_surf_tiles;

// One packed-surface tap.  w = 0 when the tile is out of range or has no
// resident SURFACE slot — which is NOT the same as having no colour
// slot, hence the separate table (see tile_detail.glsl.h).  textureLod(0)
// for the same reason as terrainDetailTap: single mip, divergent flow.
vec4 terrainDetailSurfTap(vec2 p, out float w) {
    w = 0.0f;
    vec2 rel = (p + vec2(kTerrainMapMeters * 0.5f)) / kDetailTileMeters;
    ivec2 t = ivec2(floor(rel));
    if (any(lessThan(t, ivec2(0))) ||
        any(greaterThanEqual(t, ivec2(kDetailTilesPerSide))))
        return vec4(0.0f);
    int slot = detail_surf_slot[t.y * kDetailTilesPerSide + t.x];
    if (slot < 0) return vec4(0.0f);
    w = 1.0f;
    return textureLod(detail_surf_tiles, vec3(rel - vec2(t), float(slot)),
                      0.0f);
}

// Near-field packed surface, filtered with the SAME five anisotropic taps
// as terrainDetailAlbedo.  Deliberately the same kernel: the normal and
// the colour describe one surface, and a normal left sharper than the
// albedo it belongs to would sparkle down exactly the cliff the colour
// was just blurred along.
//
// Returns the AUTHORITY of the result in [0,1] — how much of this pixel
// genuinely came from a resident 1 m tile, faded by camera distance.
// Zero means the caller keeps everything it already had.
float terrainDetailSurface(vec2 pos_xz_ws, float fade,
                           vec2 blur_dir, float blur_m,
                           out vec2 nrm_xy, out float rough, out float ao) {
    nrm_xy = vec2(0.0f);
    rough  = 0.0f;
    ao     = 1.0f;
    if (fade <= 0.0f) return 0.0f;
    float w;
    float acc_w = 0.0f;
    vec4  acc = terrainDetailSurfTap(pos_xz_ws, w) * 3.0f;
    acc_w += w * 3.0f;
    if (blur_m > 0.05f) {          // flat ground keeps the single tap
        vec2 d1 = blur_dir * (blur_m * 0.5f);
        vec2 d2 = blur_dir * blur_m;
        acc += terrainDetailSurfTap(pos_xz_ws - d1, w) * 2.0f; acc_w += w * 2.0f;
        acc += terrainDetailSurfTap(pos_xz_ws + d1, w) * 2.0f; acc_w += w * 2.0f;
        acc += terrainDetailSurfTap(pos_xz_ws - d2, w);        acc_w += w;
        acc += terrainDetailSurfTap(pos_xz_ws + d2, w);        acc_w += w;
    }
    if (acc_w <= 0.0f) return 0.0f;
    acc /= acc_w;
    // Decoding AFTER the average is identical to averaging the decoded
    // values — the .rg * 2 - 1 mapping is affine — and costs one madd
    // instead of five.
    nrm_xy = acc.rg * 2.0f - 1.0f;
    rough  = acc.b;
    ao     = acc.a;
    return fade;
}

// ── Virtual-textured terrain albedo ─────────────────────────────────
// Same RVT pools/page-table/feedback as the cluster bindless path (one
// shared VirtualTextureManager) — declaration order matters, see
// vt_sample.glsl.h.  tile_params.vt_albedo_id gates the whole path.
#include "../vt_types.glsl.h"

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

#include "../vt_sample.glsl.h"
layout(set = TILE_PARAMS_SET, binding = SRC_MAP_MASK_INDEX) uniform sampler2D src_map_mask;
layout(set = TILE_PARAMS_SET, binding = SRC_TEMP_TEX_INDEX) uniform sampler3D src_temp;
layout(set = TILE_PARAMS_SET, binding = DETAIL_NOISE_TEXTURE_INDEX) uniform sampler3D src_detail_noise_tex;
layout(set = TILE_PARAMS_SET, binding = ROUGH_NOISE_TEXTURE_INDEX) uniform sampler3D src_rough_noise_tex;

// ─────────────────────────────────────────────────────────────────────
//  TERRAIN MATERIAL SYSTEM
// ─────────────────────────────────────────────────────────────────────
// The generated map supplies COLOUR but no surface character, so on its
// own the terrain reads as a blurry stretched photo.  Detail is added
// here — but the way it is added matters more than how much:
//
// The previous version sampled a handful of ISOLATED noise bands (~28 m,
// ~6 m, ~2 m, ~0.45 m) out of the bound volume-cloud texture.  That
// produced the "low-res texture blended with high-res noise" look, for
// three separate reasons, all fixed below:
//
//  1. SPECTRAL GAP.  The macro albedo carries structure down to ~0.5 m
//     and then stops; the noise bands started at 28 m and 6 m with heavy
//     contrast.  Two disjoint bands with nothing between them can only
//     read as two surfaces stacked on top of each other.  Now ONE field
//     spans 64 m -> 0.12 m continuously with equal energy per octave
//     (1/f, pink), which is the spectrum real ground has.
//  2. CELLULAR SIGNATURE.  src_detail_noise_tex is the volume-cloud
//     noise: perlin fBm MULTIPLIED BY worley.  Its worley cells are what
//     showed up as distinct blobby patches.  Its mean is also ~0.37, not
//     the 0.5 the old code subtracted, so every band additionally
//     darkened the ground by ~13% of its contrast.  We now use the
//     analytic value noise from tile_common (zero DC bias, no cells) and
//     get exact derivatives from the same evaluation for free.
//  3. UNCORRELATED COLOUR AND LIGHT.  Colour noise and normal noise were
//     separate fields at separate scales, so the mottling was painted on
//     rather than lit.  One field now drives colour, relief and
//     roughness together.
//
//  weights : ML albedo colour (layout-aligned with the generator's
//            segmentation by construction) re-weighted by SLOPE and
//            ALTITUDE — steep ground sheds soil/snow to rock, etc.
//  fade    : per-OCTAVE, on its own view-distance ramp (an octave is
//            worth shading while its features still cover a couple of
//            pixels).  There is no single detail on/off band to see, and
//            fine grain survives right up to the camera.

// Normalisation constants measured from the field (fp32, 4 km extent):
// a SINGLE octave has value rms 0.467 and unit-relief slope rms 1.223.
//
// These divide by the RMS of the active weight set, sqrt(sum w^2), not by
// its sum.  The octaves are uncorrelated, so stacking N of them grows the
// total as sqrt(N), not N: dividing by the sum therefore over-normalises,
// and by an amount that grows with octave count.  Under the old
// sum-normalisation the value rms ran 0.33 at 8 active octaves (where the
// constant was fitted) but 0.47 at 4 and 0.66 at 2 — contrast climbing
// with view distance — and any octave ADDED at the fine end would have
// pulled the near field down to 0.28.  That is the trap: extending the
// spectrum downward while sum-normalised makes close ground FLATTER, not
// more detailed.  sqrt(sum w^2) holds the rms at 0.33 for every octave
// count, so the ladder below can be lengthened freely.
#define kGroundFieldNorm   0.707f    // -> value rms ~0.33 at ANY octave count
#define kGroundSlopeNorm   0.818f    // -> slope rms ~1.00 per unit relief

// Octave count.  The ladder is 64 m / 2.31^i, so 8 octaves bottomed out at
// 18 cm — coarser than anything the eye resolves on ground a metre away.
// Dry sand ripples (5-15 cm), grit and soil crumb all live BELOW that
// floor, so the near field carried no signal at its own scale and read as
// a smooth painted plane however the material constants were tuned.  11
// octaves reach 1.5 cm.
//
// Cost is bounded by each octave's own distance ramp and the coarse->fine
// break below: octave 8 (7.9 cm) is confined to 87 m, octave 9 to 38 m,
// octave 10 to 16 m.  Only pixels inside those radii pay for them.  Drop
// to 10 or 9 if the near field costs too much — the floor rises to 3.4 cm
// or 7.9 cm respectively.
#define kGroundOctaves     11

// hash1(vec2) is fract(a*b*(a+b)) with a,b = 50*fract(p/pi), so it
// collapses to EXACTLY 0 along the whole p.x == 0 and p.y == 0 lattice
// lines.  Sampled straight off world coordinates that paints a hard dark
// cross through the world origin, once per octave, all left-aligned on the
// same two lines - i.e. right through the town.  Shifting by more than the
// map's corner radius (2048 * sqrt2 = 2896 m, which rotation cannot grow)
// puts both lines permanently outside the map.
#define kGroundOrigin vec2(5119.0f, 6871.0f)

// One octave on one world plane.  Returns (value, d/du, d/dv) with the
// gradient rotated back out of the octave's frame into that plane's own
// axes.  rot_t is R^T and orthonormal, so the gradient rms - and therefore
// kGroundSlopeNorm - is unaffected by the rotation.
vec3 terrainGroundOctave(vec2 c, float inv_period, mat2 rot, mat2 rot_t) {
    vec3 n = noised((rot * c + kGroundOrigin) * inv_period);
    return vec3(n.x, rot_t * vec2(n.y, n.z));
}

// Pink (1/f) ground micro-surface field, world-anchored and TRIPLANAR.
//   returns : signed surface value, rms ~0.33, clamped to [-1, 1]
//   grad    : matching world-space gradient, rms ~1.0 — multiply by the
//             material's relief (in slope units) before use
// A single XZ projection smears into vertical streaks on cliffs, and the
// taller the face the longer the streak, so the three world-plane
// projections are blended by the normal instead.  The weights are
// sharpened to ^4 and thresholded, so flat ground lands ~1.0 on the Y
// plane and pays for ONE evaluation; only genuinely steep faces pay for
// two or three.
float terrainGroundField(vec3 pos_ws, vec3 n_ws, float dist, out vec3 grad) {
    const float kP0  = 64.0f;        // coarsest period, metres
    const float kLac = 2.31f;

    vec3 tw = abs(n_ws); tw *= tw; tw *= tw;          // ^4
    tw /= max(tw.x + tw.y + tw.z, 1e-4f);
    bool do_x = tw.x > 0.01f;      // face spans world ZY
    bool do_y = tw.y > 0.01f;      // ground, spans world XZ
    bool do_z = tw.z > 0.01f;      // face spans world XY

    // Rotating each octave also breaks up the axis-aligned grid ringing of
    // stacked value noise.
    mat2 rot   = mat2(1.0f, 0.0f, 0.0f, 1.0f);  // R_i
    mat2 rot_t = mat2(1.0f, 0.0f, 0.0f, 1.0f);  // (R_i)^T
    float period = kP0;
    float sum = 0.0f, w2_sum = 0.0f, sw2_sum = 0.0f;
    grad = vec3(0.0f);
    for (int i = 0; i < kGroundOctaves; ++i) {
        // ~2 px at 1080p / 60 deg is period/dist ~= 1/515, so start the
        // ramp before that and finish well after it.
        float w = 1.0f - smoothstep(period * 380.0f, period * 1100.0f,
                                    dist);
        if (w <= 0.004f) break;      // coarse -> fine, so nothing later
                                     // survives either
        float ip = 1.0f / period;
        float v  = 0.0f;
        vec3  g  = vec3(0.0f);
        if (do_y) {
            vec3 o = terrainGroundOctave(pos_ws.xz, ip, rot, rot_t);
            v += tw.y * o.x;
            g += tw.y * vec3(o.y, 0.0f, o.z);
        }
        if (do_x) {
            vec3 o = terrainGroundOctave(pos_ws.zy, ip, rot, rot_t);
            v += tw.x * o.x;
            g += tw.x * vec3(0.0f, o.z, o.y);
        }
        if (do_z) {
            vec3 o = terrainGroundOctave(pos_ws.xy, ip, rot, rot_t);
            v += tw.z * o.x;
            g += tw.z * vec3(o.y, o.z, 0.0f);
        }
        sum    += v * w;
        w2_sum += w * w;
        // The macro heightfield already owns the large-scale slope, so
        // bias the RELIEF toward the finer octaves (colour stays flat 1/f).
        // The ramp spans the whole ladder, so lengthening it moves the
        // relief emphasis onto the new fine octaves — which is the point:
        // close ground should be lit by centimetre relief, not decimetre.
        float sw = w * (0.28f + 0.72f * float(i)
                                * (1.0f / float(kGroundOctaves - 1)));
        grad    += g * sw;
        sw2_sum += sw * sw;
        rot    = m2  * rot;
        rot_t  = m2i * rot_t;
        period /= kLac;
    }
    grad *= kGroundSlopeNorm / max(sqrt(sw2_sum), 1e-3f);
    return clamp(sum * (kGroundFieldNorm / max(sqrt(w2_sum), 1e-3f)),
                 -1.0f, 1.0f);
}

// Material weights packed as (grass, rock, loose, snow), normalised.
// "loose" is bare ground — soil, dirt, gravel, sand.
vec4 terrainMaterialWeights(vec3 macro, float slope01, float alt_m) {
    float luma = dot(macro, vec3(0.299f, 0.587f, 0.114f));
    float mx   = max(macro.r, max(macro.g, macro.b));
    float mn   = min(macro.r, min(macro.g, macro.b));
    float sat  = (mx - mn) / max(mx, 1e-4f);
    float green = macro.g - 0.5f * (macro.r + macro.b);

    // colour-driven base weights
    float w_grass = smoothstep(0.010f, 0.090f, green);
    // SNOW needs NEAR-WHITE, not merely pale.  The macro sample is a
    // coarse mip average, and averaging green fields + tan paths + gray
    // hedgerows lands at luma ~0.55-0.62, sat < 0.1 — the old
    // (0.60..0.80, sat < 0.22) ramps classified that mix as snow and
    // painted whole green-countryside maps white.  Real painted snow in
    // these albedos sits at luma > 0.72 with sat < 0.10; start the ramp
    // just under that (same detector the plant scatter uses).
    float w_snow  = smoothstep(0.68f, 0.82f, luma)
                  * (1.0f - smoothstep(0.06f, 0.16f, sat));
    // Bare ground: warm (r > b) OR simply low-saturation and not pale.
    // The low-saturation half of this used to fall through to ROCK, which
    // is why ordinary dirt, graded roads and dry grass got rock's hard
    // high-contrast treatment and read as dark blotches on flat land.
    float w_loose = max(smoothstep(0.02f, 0.14f, macro.r - macro.b),
                        1.0f - smoothstep(0.06f, 0.20f, sat))
                  * (1.0f - w_grass) * (1.0f - w_snow);
    // rock takes whatever the others don't claim
    float w_rock  = clamp(1.0f - (w_grass + w_snow + w_loose), 0.0f, 1.0f);

    // ── slope override: steep faces can't hold soil, sand or snow ──
    float steep = smoothstep(0.35f, 0.72f, slope01);
    w_grass *= (1.0f - steep);
    w_loose *= (1.0f - steep * 1.15f);
    w_snow  *= (1.0f - steep * 0.85f);
    w_rock  += steep * 1.25f;

    // ── altitude: snow line only where it is already pale/cold ─────
    // 120-190 m was absurdly low for a kTerrainHeightAmpMeters = 2000
    // world — ordinary hills cleared it and the +0.5 bonus turned every
    // pale upland white ("why is everywhere snow?").  Alpine assist now
    // starts at 700 m and, per the comment above, actually REQUIRES the
    // albedo to already read pale-cold (0.55..0.72 luma ramp) instead
    // of granting it to anything brighter than mid-gray.
    float high = smoothstep(700.0f, 1000.0f, alt_m);
    w_snow += high * (1.0f - steep) * 0.5f * smoothstep(0.55f, 0.72f, luma);

    vec4 w = max(vec4(w_grass, w_rock, w_loose, w_snow), vec4(0.0f));
    return w / max(w.x + w.y + w.z + w.w, 1e-4f);
}

// Colour + roughness response to the shared field. Returns an albedo
// multiplier around 1.0.
vec3 terrainSurfaceShade(vec4 w, float f, out float roughness) {
    // Contrast per material: rock strata vary hardest, snow least.  These
    // are ~1/2 the old values because the field now has energy at every
    // scale instead of dumping it all into two bands — the same visible
    // richness with none of the blotching.
    float k = w.x * 0.34f + w.y * 0.46f + w.z * 0.28f + w.w * 0.10f;
    // Hollows darken more than crests brighten: crevices hold shadow and
    // damp.  This asymmetry is most of what makes the field read as
    // surface texture rather than painted-on patches.
    float m = 1.0f + k * (f < 0.0f ? f * 1.45f : f);

    // slight per-material hue push so materials read apart even where
    // the macro map is flat (grass greener, rock cooler, loose warmer)
    vec3 tint = w.x * vec3(0.95f, 1.05f, 0.92f)
              + w.y * vec3(1.00f, 0.99f, 1.02f)
              + w.z * vec3(1.05f, 1.01f, 0.93f)
              + w.w * vec3(1.00f, 1.01f, 1.04f);
    // crests read drier/warmer, hollows cooler and less saturated
    tint *= mix(vec3(0.985f, 1.000f, 1.020f),
                vec3(1.020f, 1.000f, 0.975f), 0.5f + 0.5f * f);

    // Raised across the board (grass .93→.97, rock .76→.88, loose
    // .95→.98, snow .58→.72): natural ground is matte — the old rock
    // and snow values gave hillsides and caps a plasticky sun sheen.
    roughness = clamp(w.x * 0.97f + w.y * 0.88f
                    + w.z * 0.98f + w.w * 0.72f, 0.05f, 1.0f);
    // damp hollows polish slightly, dry crests stay rough — correlated
    // with the same field, so the specular agrees with the colour
    roughness = clamp(roughness + f * 0.07f, 0.05f, 1.0f);

    return vec3(m) * tint;
}

// Light the same field instead of only painting it: relief is expressed
// as an rms SLOPE per material, so it stays plausible at every distance.
// ─────────────────────────────────────────────────────────────────────
//  FOREST-FLOOR LITTER + GROUND PARALLAX
// ─────────────────────────────────────────────────────────────────────
// The procedural field above gives the ground SHADING character, but at
// walking distance the surface still reads as a clean painted plane —
// nothing lies ON it, and there is no depth cue when the eye moves.  Two
// additions, both near-field only (kLitterFadeStartM..EndM) so they cost
// nothing at range:
//
//  1. LITTER: a micro-height field shaped like ground debris — leaf
//     clumps (~0.45 m blobs), fine grit (~0.13 m) and anisotropic twig
//     streaks — with an analytic gradient so the same field TINTS the
//     albedo, PERTURBS the normal and feeds the parallax, exactly the
//     "one field drives colour, relief and roughness" rule the material
//     system already follows.  Gated by the material weights: litter
//     accumulates on grass and bare soil, never on rock faces or snow.
//
//  2. PARALLAX (bump-offset): the combined ground+litter height at the
//     fragment offsets the SAMPLING position along the view ray's XZ
//     projection, so hollows slide under crests as the camera moves.
//     One field evaluation (not a stepped POM ray-march) — the relief is
//     only ~0.2 m and the offset is clamped by view.y, so a single
//     offset is visually indistinguishable from the march at a fraction
//     of the cost.  Applied to the NEAR-FIELD samplers only (1 m detail
//     tiles, procedural field, litter); the 4 m/texel macro maps would
//     not show a 0.2 m shift, and moving the VT feedback uv would only
//     perturb page streaming.
#define kLitterFadeStartM   14.0f
#define kLitterFadeEndM     26.0f
#define kGroundParallaxAmpM 0.15f

// Litter micro-height.  Returns signed height (~[-1,1]), its gradient
// d h / d(x,z) (chain rule through every shaping curve, so the lit
// relief matches the colour exactly), and the leaf / twig masks for
// tinting.  Offsets keep every octave's hash1 origin-cross far outside
// the map (see kGroundOrigin).
float terrainLitterField(vec2 p, out vec2 grad,
                         out float leaf, out float twig) {
    // Leaf clumps: ~0.45 m value-noise blobs, sharpened into patches.
    vec3 n1 = noised((p + kGroundOrigin + vec2( 337.7f,  911.3f))
                     * (1.0f / 0.45f));
    vec2 g1 = n1.yz * (1.0f / 0.45f);
    float t1 = clamp((n1.x + 0.15f) / 0.70f, 0.0f, 1.0f);
    leaf = t1 * t1 * (3.0f - 2.0f * t1);
    vec2 dleaf = (6.0f * t1 * (1.0f - t1) / 0.70f) * g1;

    // Fine grit / soil crumb: ~0.13 m, used raw.
    vec3 n2 = noised((p + kGroundOrigin + vec2(1191.2f, 1553.8f))
                     * (1.0f / 0.13f));
    vec2 g2 = n2.yz * (1.0f / 0.13f);

    // Twigs: strongly anisotropic ridges (2.1 m along x, 0.17 m across z)
    // thresholded to thin streaks.  abs() folds the noise into ridges.
    vec3 n3 = noised(vec2((p.x + kGroundOrigin.x + 217.9f) * (1.0f / 2.1f),
                          (p.y + kGroundOrigin.y + 471.1f) * (1.0f / 0.17f)));
    vec2 g3 = vec2(n3.y * (1.0f / 2.1f), n3.z * (1.0f / 0.17f));
    float r  = 1.0f - abs(n3.x);
    float t3 = clamp((r - 0.62f) / 0.30f, 0.0f, 1.0f);
    twig = t3 * t3 * (3.0f - 2.0f * t3);
    vec2 dtwig = (6.0f * t3 * (1.0f - t3) / 0.30f)
               * (-sign(n3.x)) * g3;

    grad = dleaf * 0.62f + g2 * 0.25f + dtwig * 0.13f;
    return leaf * 0.62f + n2.x * 0.25f + twig * 0.13f - 0.30f;
}

vec3 terrainSurfaceNormal(vec3 n, vec4 w, vec3 grad) {
    float relief = w.x * 0.10f + w.y * 0.26f
                 + w.z * 0.13f + w.w * 0.04f;
    // The field is triplanar now, so its gradient is meaningful on ANY
    // face — no more damping by n.y.  Perturb along the surface TANGENT
    // plane rather than assuming the XZ projection; on flat ground this
    // reduces exactly to the old n + (-g.x, 0, -g.z).
    vec3 t = grad - dot(grad, n) * n;
    return normalize(n - t * relief);
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

    // ── View ray (moved up: the parallax below needs it) ─────────────
    vec3 view_vec = camera_info.position.xyz - in_data.vertex_position;
    float view_dist = length(view_vec);
    vec3 view = normalize(view_vec);

    // ── Ground parallax (see the LITTER + PARALLAX block above) ──────
    // pos_p is the parallax-corrected position every NEAR-FIELD surface
    // sampler uses from here on; pos keeps feeding geometry-derived
    // quantities (heights, shadows, G-buffer position).
    float lit_fade = 1.0f - smoothstep(kLitterFadeStartM,
                                       kLitterFadeEndM, view_dist);
    // VEGETATION GATE.  The litter+parallax treatment belongs to living
    // ground; on the pale town/plaza albedo it warps a near-featureless
    // surface into wobble ("ground texture looks really bad" was pale
    // town ground under full-strength parallax).  Probe the macro
    // albedo at the UNSHIFTED uv — coarse is fine, this is a where-am-I
    // question, not a texture tap — and fade the whole near-field
    // treatment out where the ground doesn't read green.
    {
        vec3 vp = texture(src_map_mask, in_data.world_map_uv).rgb;
        float g_excess = 2.0f * vp.g - vp.r - vp.b;
        lit_fade *= smoothstep(0.02f, 0.09f, g_excess);
    }
    vec3 pos_p = pos;
    if (lit_fade > 0.0f && normal.y > 0.45f) {
        vec3  g_par;
        float f_par = terrainGroundField(pos, normal, view_dist, g_par);
        vec2  lg_par; float leaf_par, twig_par;
        float lh_par = terrainLitterField(pos.xz, lg_par,
                                          leaf_par, twig_par);
        float h01 = clamp(0.5f + 0.30f * f_par + 0.26f * lh_par,
                          0.0f, 1.0f);
        // Classic bump-offset: low points shift AWAY from the eye along
        // the view ray's ground projection.  view.y clamp bounds the
        // shift at grazing angles (max ~0.6 m) so silhouettes stay put.
        pos_p.xz -= (view.xz / max(view.y, 0.35f))
                    * ((1.0f - h01) * kGroundParallaxAmpM * lit_fade);
    }

    // ── Anisotropy guard for the top-down macro albedo ────────────────
    // Every macro colour source (map mask, VT pages, 1 m detail tiles) is
    // indexed by world XZ.  On a steep face a tiny XZ footprint covers a
    // tall strip of surface, but the samplers' derivatives only see XZ, so
    // they happily pick a SHARP mip and drag it down the whole face — that
    // is the vertical smearing on the cliffs.  One XZ metre spans 1/n.y
    // metres of surface, so bias the LOD by log2 of that stretch: the
    // bogus stretched detail blurs out and the (now triplanar) procedural
    // field supplies the surface character in its place.  Flat ground has
    // n.y = 1 and is completely unaffected.
    //
    // The VT pool and map mask are mipped, so an LOD bias is enough there
    // (and biasing the VT also stops cliffs requesting sharp pages they
    // cannot use).  The 1 m detail tiles have a single mip, so they get
    // real anisotropic taps instead — see terrainDetailAlbedo().
    float macro_lod_bias = clamp(-log2(max(normal.y, 0.02f)), 0.0f, 4.0f);
    float macro_blur_m   = clamp(0.5f * (1.0f / max(normal.y, 0.05f) - 1.0f),
                                 0.0f, 3.0f);
    vec2  macro_blur_dir = normalize(normal.xz
                                     + vec2(1e-5f, 0.0f));   // gradient dir

    // Macro (4 m/texel, whole-world) surface maps, filled from the VT
    // pools below when <heightmap>_nrm.png / _orm.png were registered.
    // Defaults are the "no map" identity: flat tangent normal, no
    // occlusion, and an authority of zero so the procedural estimate
    // stays in charge wherever a map was never authored.
    vec2  macro_nrm_xy = vec2(0.0f);
    float macro_nrm_w  = 0.0f;
    float macro_rough  = 0.0f;
    float macro_ao     = 1.0f;
    float macro_mr_w   = 0.0f;

    // Surface colour comes ONLY from the ML-generated albedo (VT-backed
    // colour satellite map, or the plain map-mask fallback) — the old
    // procedural rock/soil tinting and temperature snow mix are gone.
    vec3 albedo;

    // Terrain surface colour: virtual-textured when a VT id is set
    // (streamed pages, 1 m albedo detail tiles later), otherwise the
    // plain full-world map-mask sample.
    albedo = texture(src_map_mask, in_data.world_map_uv,
                     macro_lod_bias).rgb;
    if (tile_params.vt_albedo_id != VT_INVALID_ID) {
        VirtualTextureMeta vmeta =
            vt_meta[vtIndexOf(tile_params.vt_albedo_id)];
        // Bias here rather than at the sample so the FEEDBACK request goes
        // coarse too: cliffs stop pulling sharp pages they cannot use.
        float lod_cont = vtComputeLod(vmeta, in_data.world_map_uv)
                       + macro_lod_bias;
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
        bool  vt_hit      = false;
        float vt_pool_lod = 0.0f;
        for (uint i = 0u; i < VT_MAX_MIPS; ++i) {
            if (walk_mip > mip_max) break;
            if (vtResolve(tile_params.vt_albedo_id, in_data.world_map_uv,
                          vmeta, walk_mip, phys_uv)) {
                vt_pool_lod = (walk_mip == vt_mip) ? vt_frac : 0.0f;
                albedo = textureLod(vt_pool_albedo, phys_uv,
                                    vt_pool_lod).rgb;
                vt_hit = true;
                break;
            }
            ++walk_mip;
        }
        if (vt_hit) {
            // Same phys_uv, same pool lod: all four VT pools share one
            // slot allocator and one page-table entry, so the albedo's
            // resolve already located the normal and ORM pages too.  The
            // ids are per-layer PRESENCE flags — sampling a pool the
            // registration never filled would read whatever other
            // material happens to own those physical slots.
            if (tile_params.vt_normal_id != VT_INVALID_ID) {
                macro_nrm_xy = textureLod(vt_pool_normal, phys_uv,
                                          vt_pool_lod).rg * 2.0f - 1.0f;
                macro_nrm_w  = 1.0f;
            }
            if (tile_params.vt_mr_ao_id != VT_INVALID_ID) {
                // glTF ORM convention: R = occlusion, G = roughness,
                // B = metallic (terrain is dielectric, ignored here).
                vec4 mra = textureLod(vt_pool_mr_ao, phys_uv, vt_pool_lod);
                macro_ao    = mra.r;
                macro_rough = mra.g;
                macro_mr_w  = 1.0f;
            }
        }
    }
    // Near-field: the streamed 1 m albedo tile takes over from the
    // (4 m/texel) global map, fading with the same camera-distance band
    // as the height detail so colour and relief transition together.
    albedo = terrainDetailAlbedo(pos_p.xz, albedo, detail_fade,
                                 macro_blur_dir, macro_blur_m);
    // Beyond the terrain map: neutral surround (matches the height fade
    // in tile.vert — no stretched border stripes in colour or shading).
    // sfade lives in function scope, not inside the block below: the
    // authored surface maps have to be damped by it too.  vtResolve()
    // takes fract() of the uv, so a fragment past the map edge would
    // otherwise wrap the far side of the world onto the surround and
    // light it with somebody else's ridge occlusion.
    float sfade;
    {
        vec2 ov = (in_data.world_map_uv
                   - clamp(in_data.world_map_uv, 0.0f, 1.0f))
                  / tile_params.inv_world_range;
        sfade = smoothstep(0.0f, kTerrainSurroundFadeMeters, length(ov));
        albedo = mix(albedo, vec3(0.16f, 0.20f, 0.14f), sfade);
        normal = normalize(mix(normal, vec3(0.0f, 1.0f, 0.0f), sfade));
    }

    // Geometric normal: the pure heightfield surface before the authored
    // normal maps / procedural micro-relief perturb it.  The deferred
    // resolve uses this for shadow-ray biasing (see out_emissive_metal).
    vec3 geom_normal = normal;

    // ── Terrain material layers ──────────────────────────────────────
    // Applied AFTER the macro colour is resolved (VT / map-mask / 1 m
    // tiles) so it modulates whatever the generator produced.  There is
    // no global distance gate any more: terrainGroundField() retires one
    // octave at a time as each stops covering pixels, so the surface
    // never aliases at range and never loses its grain up close.
    float slope01 = clamp(1.0f - normal.y, 0.0f, 1.0f);
    vec4  mat_w   = terrainMaterialWeights(albedo, slope01, pos.y);
    vec3  g_grad;
    float g_field = terrainGroundField(pos_p, normal, view_dist, g_grad);
    float mat_rough;
    albedo *= terrainSurfaceShade(mat_w, g_field, mat_rough);
    normal  = terrainSurfaceNormal(normal, mat_w, g_grad);

    // ── Forest-floor litter (near field) ─────────────────────────────
    // Same field as the parallax, re-evaluated at the corrected position
    // so colour, relief and depth agree.  Litter holds on grass and bare
    // soil (mat_w.x + mat_w.z); rock and snow shed it.
    if (lit_fade > 0.0f) {
        vec2  lg; float leaf_m, twig_m;
        float lh = terrainLitterField(pos_p.xz, lg, leaf_m, twig_m);
        float ground_w = clamp(mat_w.x + mat_w.z, 0.0f, 1.0f);
        float lw = lit_fade * ground_w * (1.0f - sfade);
        if (lw > 0.003f) {
            float luma = dot(albedo, vec3(0.299f, 0.587f, 0.114f));
            // Dark macro ground (forest shade) decays toward wet leaves
            // and humus; bright open ground reads as dry thatch.
            vec3 leaf_col = mix(vec3(0.216f, 0.152f, 0.081f),
                                vec3(0.412f, 0.331f, 0.174f),
                                smoothstep(0.12f, 0.42f, luma));
            // Per-clump value jitter so the litter isn't one dye lot.
            leaf_col *= 0.85f + 0.42f * clamp(0.5f + 0.5f * lh, 0.0f, 1.0f);
            vec3 humus_col = albedo * vec3(0.52f, 0.47f, 0.40f);
            vec3 lit_col   = mix(humus_col, leaf_col, leaf_m);
            lit_col        = mix(lit_col, vec3(0.34f, 0.26f, 0.15f),
                                 twig_m * 0.7f);
            float blend = lw * (0.28f + 0.45f * leaf_m + 0.27f * twig_m);
            albedo = mix(albedo, lit_col, clamp(blend, 0.0f, 0.85f));
            // The litter has real height — light it.  Tangent-plane
            // projection like terrainSurfaceNormal, own relief scale.
            vec3 t_lit = vec3(lg.x, 0.0f, lg.y);
            t_lit -= dot(t_lit, normal) * normal;
            normal = normalize(normal - t_lit * (0.22f * lw));
            // Dry leaves and grit are matte.
            mat_rough = clamp(mat_rough + 0.10f * lw * leaf_m,
                              0.05f, 1.0f);
        }
    }

    // ── Authored surface maps over the procedural estimate ───────────
    // Three sources of relief, stacked by scale: the 1 m ML detail tile
    // wins where it is resident, the 4 m macro map carries the rest of
    // the 32 km, and the procedural field stays underneath BOTH — it is
    // the only one of the three with energy below a metre, and it is all
    // there is where neither map exists.  Nothing is switched off; the
    // authored data cross-fades in on top of what is already there.
    vec2  det_nrm_xy; float det_rough, det_ao;
    float det_w = terrainDetailSurface(pos_p.xz, detail_fade,
                                       macro_blur_dir, macro_blur_m,
                                       det_nrm_xy, det_rough, det_ao);
    vec2  surf_nrm_xy  = mix(macro_nrm_xy, det_nrm_xy, det_w);
    float surf_nrm_w   = mix(macro_nrm_w, 1.0f, det_w) * (1.0f - sfade);
    float surf_rough   = mix(macro_rough, det_rough, det_w);
    float surf_rough_w = mix(macro_mr_w, 1.0f, det_w) * (1.0f - sfade);
    // Occlusion MULTIPLIES rather than cross-fades: the macro ORM's red
    // channel is ridge occlusion integrated over the whole 32 km, and the
    // detail tile's alpha is micro-cavity only (the worker runs its AO
    // with sky=False for exactly this reason), so the two are different
    // stacking effects, not two estimates of one thing.
    float surf_ao = mix(macro_ao * mix(1.0f, det_ao, det_w), 1.0f, sfade);

    if (surf_nrm_w > 0.0f) {
        // Tangent frame of the terrain surface: the height field is
        // parameterised by world XZ, so T is world +X and B is world +Z,
        // re-orthogonalised against the shading normal.  cross(T, normal)
        // — NOT cross(normal, T) — is what keeps B on +Z: (T,B,N) here is
        // LEFT-handed (T x B = X x Z = -Y), and that is the frame the
        // python side encoded its tangent normals in.
        vec3 T = normalize(vec3(1.0f, 0.0f, 0.0f) - normal * normal.x);
        vec3 B = cross(T, normal);
        vec2 nxy = surf_nrm_xy * surf_nrm_w;
        float nz = sqrt(max(1e-4f, 1.0f - min(dot(nxy, nxy), 0.9999f)));
        normal = normalize(nxy.x * T + nxy.y * B + nz * normal);
    }
    // ── Authored roughness MODULATES the material guess ───────────────
    // It used to REPLACE it: surf_rough_w goes to 1.0 the moment a detail
    // surface tile becomes resident, so mix(mat_rough, surf_rough, 1.0)
    // discarded the material-derived value entirely and handed the whole
    // surface to the tile's B channel.
    //
    // That is what made the ground turn white as a scene finished
    // streaming, with nothing else changing — same camera, same shadow
    // technique, the tiles simply arrived.  Measured over all 90 cached
    // tile_*_surf.png of this map, the authored roughness runs
    //     min 0.505   p25 0.729   median 0.867   p75 0.925
    // against material values of grass 0.97 / loose 0.98 / rock 0.88 /
    // snow 0.72.  So residency drags walkable ground from ~0.97 toward
    // ~0.87, and a quarter of it below 0.73.  Terrain is DIELECTRIC
    // (metallic 0, achromatic f0 = 0.04), so everything that gain buys
    // is WHITE specular off the sky laid over the albedo — a sheen that
    // washes sand out completely while darker patches still read
    // through it.  The 0.50 floor below was supposed to catch this and
    // is simply too permissive to.
    //
    // The authored map still has real information — damp hollows, packed
    // tracks, polished rock — so it is not ignored.  It is bounded: the
    // detail tile may texture the roughness DOWN by a limited fraction,
    // never polish ground to semi-gloss.  kDetailRoughFloorFrac is that
    // bound; 1.0 restores the old replace-outright behaviour.
    const float kDetailRoughFloorFrac = 0.85f;
    float bounded_rough = max(surf_rough, mat_rough * kDetailRoughFloorFrac);
    // Absolute floor raised 0.50 -> 0.75 for the same reason the material
    // constants were raised: walkable GROUND is never glossier than
    // semi-matte, whatever any authored map says.
    mat_rough = clamp(mix(mat_rough, bounded_rough + g_field * 0.07f,
                          surf_rough_w), 0.75f, 1.0f);

#ifdef GBUFFER_OUTPUT
    // Deferred path: write material attributes and stop.  Lighting (sun +
    // CSM / raytraced shadows + RT-AO) runs once per pixel in
    // deferred_resolve.comp — the exact same path cluster geometry takes,
    // which is what keeps terrain lighting consistent with the rest of
    // the scene.
    //
    // albedo_ao.a doubles as the resolve's "G-buffer written" sentinel
    // (< 0.5 means forward pixel, leave its colour alone), so terrain AO
    // is compressed into [0.5, 1.0] instead of risking a skipped pixel.
    out_albedo_ao      = vec4(albedo, clamp(surf_ao, 0.5f, 1.0f));
    out_normal_rough   = vec4(octEncodeDir(normal), mat_rough, 0.0f);
    vec2 oct_geom      = octEncodeDir(geom_normal);
    out_emissive_metal = vec4(oct_geom, 0.0f, 0.0f);
    // Terrain is static: screen-space velocity is pure camera motion.
    vec4 cur_clip  = camera_info.view_proj      * vec4(pos, 1.0f);
    vec4 prev_clip = camera_info.prev_view_proj * vec4(pos, 1.0f);
    out_velocity   = cur_clip.xy / cur_clip.w - prev_clip.xy / prev_clip.w;
#else
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
    // Authored occlusion attenuates AMBIENT light only — that is what an
    // occlusion map means, and IBL is the ambient term here.  Applying it
    // to a direct sun term instead would darken lit slopes that the sky
    // simply cannot see, which is the classic AO-as-shadow mistake.
    f_specular += getIBLRadianceGGX(normal, view, material_info.perceptualRoughness, material_info.f0, mip_count) * surf_ao;
    f_diffuse += getIBLRadianceLambertian(normal, material_info.albedoColor) * surf_ao;
    #endif

    //vec3 color = vec3(noise_value.w);
    vec3 color = f_diffuse + f_specular;

    float alpha = 1.0f;
    // Shared scene tonemap — keeps the forward terrain branch on the same
    // curve as the deferred resolve (which lights this same terrain when
    // GBUFFER_OUTPUT is active).
    outColor = vec4(sceneTonemap(color), alpha);

    // ── Runtime render-debug override ────────────────────────────────
    // The forward terrain branch had NO debug-mode dispatch at all,
    // which made it the one surface in the frame you could not inspect:
    // "Render Debug > Albedo" answered for every drawable and for
    // deferred terrain (deferred_resolve.comp decodes the G-buffer) but
    // silently did nothing here, so the forward path could only ever be
    // debugged by staring at the shaded result.  Same modes, same
    // meanings, same packing as base.frag and cluster_bindless.frag.
    uint dbg_mode =
        (camera_info.input_features & FEATURE_INPUT_DEBUG_MODE_MASK)
            >> FEATURE_INPUT_DEBUG_MODE_SHIFT;
    if (dbg_mode == DEBUG_RENDER_MODE_ALBEDO) {
        outColor = vec4(albedo, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_NORMAL) {
        outColor = vec4(normal * 0.5f + 0.5f, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_GEOMETRIC_NORMAL) {
        outColor = vec4(geom_normal * 0.5f + 0.5f, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_DIFFUSE) {
        outColor = vec4(f_diffuse, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SPECULAR) {
        outColor = vec4(f_specular, 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_ROUGHNESS) {
        outColor = vec4(vec3(material_info.perceptualRoughness), 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_METALLIC) {
        outColor = vec4(vec3(material_info.metallic), 1.0f);
    } else if (dbg_mode == DEBUG_RENDER_MODE_SSAO) {
        // Parity with base.frag / cluster_bindless.frag: white here so
        // ssao_apply.comp's multiply leaves vec3(ao) on screen.  The
        // terrain's OWN occlusion is surf_ao and is already folded into
        // the ambient terms above.
        outColor = vec4(1.0f);
    }
//    outColor.xyz *= in_data.test_color;
#endif  // !GBUFFER_OUTPUT
}