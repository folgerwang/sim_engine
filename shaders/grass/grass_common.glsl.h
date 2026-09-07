#ifndef GRASS_COMMON_GLSL_H
#define GRASS_COMMON_GLSL_H

// ─────────────────────────────────────────────────────────────────────
// Procedural grass blade — shared by grass.mesh (live path) and
// grass.geom (the pre-mesh-shader fallback), so the two can never drift
// apart again.  No bindings are declared here: the caller samples the
// ground height and the wind and hands them in.
//
// WHY the blades changed shape.  The old blade was 1.0 m tall and up to
// 0.128 m WIDE — a 13 cm spear.  Real grass is 20-50 cm tall and 3-10 mm
// wide, so every blade in frame was rendering an order of magnitude too
// fat, which is most of why the field read as a bed of green plastic
// spikes rather than grass.  It was fat for a reason: the scatter is
// ~1 blade/m^2 (kMaxNumGrass = 8192 over a 128 m tile, x2), and one
// realistic blade per square metre covers nothing.
//
// The fix is CLUMPING, not width.  Blades are grouped into tufts of
// kGrassTuftBlades sharing one root position, so the same blade budget
// buys sparse tufts of believable grass instead of a uniform lattice of
// spears — which is also how grass actually grows.  Bare ground between
// tufts is fine: the terrain's own albedo underneath is already grass
// coloured, and grass.frag pulls each blade's hue from it.
// ─────────────────────────────────────────────────────────────────────

const int   kGrassRings      = 8;   // vertex rings from root to tip
const int   kGrassBladeVerts = 16;  // 2 per ring
const int   kGrassBladeTris  = 14;  // one strip

// 6 (was 5): the reference grassland grows in TUSSOCKS -- distinct
// clumps 30-50 cm across with their stems fanning out of one crown --
// and a clump of five thin blades reads as a sprig, not a tussock.
// 7 = six blades + one FLOWER SLOT (blade_in == 6, see grassMakeBlade);
// the slot is degenerate unless the tuft rolls a flower on green land,
// so the grass itself is unchanged and the field costs 1/7 more
// dispatch for the chance of a bloom.
const float kGrassTuftBlades = 7.0f;    // slots per root clump
const uint  kGrassFlowerSlot = 6u;      // ...the last of which is this

// ── View-distance fade ───────────────────────────────────────────────
// A blade is 1-2 cm wide: at 300 m that is a hundredth of a pixel, and
// a rasteriser given sub-pixel geometry does not render it faint — it
// snaps whole pixels, so every distant blade lands as one full dark
// speck.  Against pale ground (the snow mountains especially, which
// blow out to nearly the sky's white) those specks read as debris
// floating in the air.  Blades therefore SHRINK to nothing across this
// band; a zero-size blade rasterises no fragments, and the perceived
// cover handoff to the terrain's own grass-green albedo underneath is
// exactly what happens in reality at that range.  KEEP IN SYNC with
// TileObject::drawGrass, which skips the dispatch entirely for tiles
// wholly past the fade end.
// 300/500 rather than the 140/260 this shipped with: the first numbers
// were tuned at an eye-level camera and, combined with the C++ cull,
// deleted every blade in an AERIAL editor view (a camera a few hundred
// metres up is a few hundred metres from ALL grass).  The mid-band is
// kept renderable instead by WIDENING blades with distance (below), so
// they stay near pixel width instead of collapsing under it.
const float kGrassViewFadeStartM = 300.0f;
const float kGrassViewFadeEndM   = 500.0f;
// Distance-proportional width boost: from kGrassWidenStartM a blade's
// width grows up to kGrassWidenMax x so it keeps covering ~a pixel
// instead of aliasing into on/off specks — the same trade the terrain
// makes when it retires noise octaves into roughness.  Height is NOT
// boosted; far grass reads as a low sward, not a picket fence.
const float kGrassWidenStartM = 60.0f;
const float kGrassWidenMax    = 3.0f;
// ── Distance-driven density ──────────────────────────────────────────
// The dispatch is sized per TILE (TileObject::drawGrass): every tuft of
// a 128 m tile is emitted at the same density whether it stands 2 m
// from the eye or 120 m, so the field read as one uniform carpet at
// every range -- thin underfoot, where a square metre is hundreds of
// pixels and wants dozens of tufts, and needlessly thick at 200 m,
// where a tuft is a pixel and only its colour survives.  A real meadow
// is dense at your feet and a texture at range.
//
// So the KEEP fraction of tufts falls with camera distance: every tuft
// inside kGrassDenseNearM stands, and past kGrassDenseFarM only
// kGrassDenseFarKeep of them do.  The decision is a hashed threshold on
// the TUFT (one hash per clump, so clumps come and go whole, never a
// tuft losing two of its five blades), and it is a soft threshold:
// tufts whose hash sits just above the cut are shrunk instead of
// deleted, so walking toward a patch grows its grass in rather than
// popping it -- kGrassDenseSoft is the width of that band in hash
// units.  The per-tile near boost (kGrassNearDensityBoost in terrain.h)
// supplies the blades this thins; between the two the density follows
// the CAMERA, not the tile grid.
// ONE LAW OF DISTANCE, NOT A TILE BOOST.  The dispatch used to be
// sized per tile by the distance to the tile's CENTRE (the old
// kGrassNearDensityBoost fade), so the tile the camera stood on drew
// eight times denser than the one 40 m away across a border, and the
// field showed the 128 m grid.  Now the density is rho(d) = rho_near *
// keep(d) for every root, whichever tile it is in: TileObject::drawGrass
// dispatches a tile at keep(d_min) of full -- d_min the distance to the
// tile's NEAREST point (grassKeepAtDistance in terrain.h, the same
// curve, KEEP IN SYNC) -- and the lottery below keeps a tuft with
// probability keep(d) / keep(d_min), which is <= 1 by construction and
// lands the drawn density on rho(d) exactly.  Steeper than the old
// curve and much lower at range, because rho_near is now the density
// EVERYWHERE within 30 m, not just on one tile: 64 blades/m^2 at the
// default mul x boost, ~24 at 100 m, ~2 beyond 170 m (where the
// distance widening makes each blade cover a pixel anyway).
const float kGrassDenseNearM   = 30.0f;   // full density out to here
const float kGrassDenseFarM    = 170.0f;  // ...falling to the far keep here
const float kGrassDenseFarKeep = 0.03f;   // of full, at/after FarM
const float kGrassDenseSoft    = 0.12f;   // hash band a tuft grows in over
// ── Underfoot doubling ───────────────────────────────────────────────
// The mid and far field sat right at rho_near = 8 x mul; the ground
// within a few strides did not -- a square metre at 3 m is a quarter
// of the screen, and nine tufts in it still show the dirt between.
// So the law carries a SECOND, shorter ramp: rho_near is now twice
// what it was (kGrassNearDensityBoost 8 -> 16 in terrain.h) and
// keep(d) halves again between kGrassNearDoubleM and
// kGrassNearDoubleEndM, so beyond the ramp rho(d) is EXACTLY the curve
// the mid and far field were tuned at, and inside it the tufts double.
// Both ramps live in grassKeepAtDistance, which stays monotonic, so
// the per-tuft lottery keep(d) / keep(d_min) is still <= 1 everywhere.
// terrain.h carries the same three constants and the same product --
// KEEP IN SYNC.
const float kGrassNearDoubleM    = 20.0f;  // doubled out to here...
const float kGrassNearDoubleEndM = 60.0f;  // ...back on the old curve here
const float kGrassNearDoubleK    = 2.0f;   // the underfoot factor

// rho(d) / rho_near.
float grassKeepAtDistance(float d) {
    float base = mix(1.0f, kGrassDenseFarKeep,
                     smoothstep(kGrassDenseNearM, kGrassDenseFarM, d));
    float near = mix(1.0f, 1.0f / kGrassNearDoubleK,
                     smoothstep(kGrassNearDoubleM, kGrassNearDoubleEndM, d));
    return base * near;
}

// Distance from the camera (XZ) to the nearest point of this tile's
// footprint -- what the C++ sized the dispatch by.
float grassTileMinDist(vec2 cam_xz, vec2 tile_min, vec2 tile_range) {
    vec2 c = clamp(cam_xz, tile_min, tile_min + tile_range);
    return distance(cam_xz, c);
}

// Signed lottery margin: keep(d) / keep(d_min) - h.  >= 0 the tuft
// stands in full, <= -kGrassDenseSoft it is gone, in between it is
// shrinking.  `keep_tile` = grassKeepAtDistance(d_min) of the tuft's
// tile.
float grassDensityMargin(float h, float d, float keep_tile) {
    return grassKeepAtDistance(d) / max(keep_tile, 1e-4f) - h;
}

// Per-tuft keep weight: 1 = full blade, 0 = dropped, in between =
// shrunk.  `h` is the tuft's hash (identical for every blade of the
// tuft), `d` the camera distance to its root.
float grassDensityKeep(float h, float d, float keep_tile) {
    return clamp(1.0f + grassDensityMargin(h, d, keep_tile) / kGrassDenseSoft,
                 0.0f, 1.0f);
}

// ── Near-field re-seating ────────────────────────────────────────────
// Most of the near tile's blade budget (kGrassNearDensityBoost x) is
// thrown away: the lottery above drops the tufts that landed beyond
// ~40 m, and a dropped tuft was a degenerate strip that cost its
// dispatch for nothing.  Now a dropped tuft is offered up to
// kGrassNearSeats CANDIDATE roots -- extra hashes of the same tuft,
// fixed in the world like the first one -- and stands at a candidate
// that lies inside kGrassNearSeatM of the camera.  The two halves of
// the tuft take one candidate each, so nothing has to choose between
// them.  Nothing pops: a candidate grows in over kGrassNearSeatBandM
// at the disc's edge, and it fades OUT as the tuft's original root
// approaches its own lottery threshold (grassDensityMargin rising
// toward -kGrassDenseSoft), which is exactly when the original starts
// to grow -- the two never stand at once and the crossfade is spread
// over ~30 m of camera travel.  Net: roughly twice the tufts underfoot
// for the same dispatch, and the mid and far field untouched.
const float kGrassNearSeatM     = 30.0f;   // candidates stand inside this
const float kGrassNearSeatBandM = 8.0f;    // ...growing in over this band
const uint  kGrassNearSeats     = 2u;      // candidate roots per tuft

// ── Cover follows the vegetation map ─────────────────────────────────
// The blade scatter was blind to WHERE it was: a hashed root stood on
// pale sand as readily as on the green meadow, so a slope that carries
// three trees to the hectare grew the same sward as the river flat.
// The plant scatter is driven by the macro colour map's greenness (the
// same signal terrainMaterialWeights keys the grass material on), so
// the grass now reads that map too: a tuft's keep probability is
// mix(kGrassCoverFloor, 1, cover), cover the green excess of the map
// around the root, smoothed to ~30 m (mip 3 of the 4 m map) so it
// follows the broad vegetation zones the trees follow and not the
// texel edges, and roughened by a ~20 m noise so no zone edge is a
// line.  Its own hash lane and its own soft band, so it neither
// correlates with the distance lottery nor pops.
const float kGrassCoverLod   = 3.0f;    // map mip: 4 m texels -> ~32 m
const float kGrassCoverFloor = 0.15f;   // keep on bare ground
const float kGrassCoverSoft  = 0.15f;   // hash band a tuft shrinks over

float grassCoverField(vec3 macro_rgb, vec2 p) {
    float g_ex  = 2.0f * macro_rgb.g - macro_rgb.r - macro_rgb.b;
    float cover = smoothstep(-0.02f, 0.16f, g_ex);
    // ragged zone edges: a slow field nudges the threshold
    float rag = sin(p.x * 0.31f + 0.7f) * sin(p.y * 0.27f - 1.9f)
              + 0.5f * sin(p.x * 0.083f - 0.4f) * sin(p.y * 0.097f + 2.2f);
    cover = clamp(cover + 0.18f * rag, 0.0f, 1.0f);
    return mix(kGrassCoverFloor, 1.0f, cover);
}

// Per-tuft weight from the cover lottery: `h` the tuft's cover hash
// lane, `cover` grassCoverField at its root.
float grassCoverKeep(float h, float cover) {
    return clamp(1.0f + (cover - h) / kGrassCoverSoft, 0.0f, 1.0f);
}

// ── Stand height by zone ─────────────────────────────────────────────
// A meadow is not one height: it stands tall where the soil is deep
// and damp and short where it is thin or grazed, in patches of tens of
// metres.  Two sin octaves (~55 m and ~16 m), same family as the
// dryness field but at other frequencies and phases so the two do not
// line up.  0 = short sward, 1 = tall stand.
float grassZoneField(vec2 p) {
    float a = sin(p.x * 0.0117f + 0.9f) * sin(p.y * 0.0131f + 2.3f);
    float b = sin(p.x * 0.0410f - 1.4f) * sin(p.y * 0.0370f + 0.6f);
    return clamp(0.5f + 0.34f * a + 0.30f * b, 0.0f, 1.0f);
}

// ── Moisture: tall where it is damp, short where it is cured ─────────
// The height used to RISE with dryness (the cured seed stems of the
// reference stand are its tallest plants), so the straw drifts stood
// head-high and the green hollows were the short grass -- the opposite
// of how a field reads: grass grows where the water is.  Now the
// dryness field (terrainDryField, the one that tints the ground straw)
// is read as MOISTURE through the same curve that colours it,
// 1 - terrainDryCurve(dry), and the stand's height follows it: a green
// hollow stands kGrassMoistHeight x, fully cured ground kGrassDryHeight
// x.  The zone field above keeps its patchiness on top, at a narrower
// range than before since moisture now carries the big contrast, and
// the two blend into one STAND value (grassMakeBlade) that also picks
// the tuft kinds: turf on the dry ground, tussocks and the tall stems
// in the damp.  Site moisture, not the plant's: the kind's dry offset
// (a stem is drier than the turf beside it) is applied after.
const float kGrassDryHeight   = 0.62f;   // height x on fully cured ground
const float kGrassMoistHeight = 1.35f;   // height x in the wettest hollow
const float kGrassZoneLo      = 0.65f;   // zone height x, short sward
const float kGrassZoneHi      = 1.20f;   // zone height x, tall stand

// terrainDryCurve for the stages that include tile_common, a verbatim
// copy for the one that does not (grass.geom) -- keep the two identical.
float grassDryCurve(float d) {
#ifdef TERRAIN_TILE_COMMON_GLSL_H
    return terrainDryCurve(d);
#else
    return smoothstep(0.12f, 0.85f, d);
#endif
}

// 0 = cured ground, 1 = the wettest hollow.
float grassMoisture(float dry) {
    return 1.0f - grassDryCurve(dry);
}

// ── Integer hashes ───────────────────────────────────────────────────
// pcg4d (Jarzynski & Olano, "Hash Functions for GPU Rendering", 2020).
// The tuft and blade draws used hash43() on FLOAT indices, and a float
// hash of the form fract(idx * 0.1031) has ~9 bits of fraction left by
// idx ~ 1e5 -- and the boosted near tile runs to ~2e5 tufts.  Past
// that the draws collapse onto a few hundred distinct positions: the
// near field's tufts stood on a lattice, in curves, with many tufts
// sharing one root.  Integer mixing has no such cliff.
uvec4 grassPcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

// Four uniform [0, 1) draws for (tile, index, salt).  `tile` is the
// bit pattern of the tile's world min (floatBitsToUint), which is
// unique per tile and needs no integer conversion of the coordinate.
vec4 grassHash4(uvec2 tile, uint idx, uint salt) {
    uvec4 h = grassPcg4d(uvec4(tile.x ^ (salt * 0x9E3779B9u), tile.y,
                               idx, salt));
    return vec4(h) * (1.0f / 4294967296.0f);
}
// ── Waterline ────────────────────────────────────────────────────────
// The scatter is a pure XZ hash and never asked whether the root was
// under a river; blades sprouted from the bed and poked speckles
// through the surface.  A blade shrinks to nothing as the water column
// over its root passes through this band — the band (rather than a
// hard cut) leaves a fringe of shoreline grass standing in ankle-deep
// water, which is what a real bank looks like.
// Tightened 0.01/0.05 -> 0.005/0.02 after the flooded-flat screenshot:
// a broad shallow sheet carries only a few cm of column, which the
// first band read as "shoreline" and kept fully grassed.  2 cm of
// standing water is the ceiling now.
const float kGrassWaterFadeStartM = 0.005f;  // column where fade begins
const float kGrassWaterFadeEndM   = 0.02f;   // fully gone by here
// A wet tuft is not deleted, it is RE-SEATED: up to this many re-hashed
// tuft positions are tried before giving up and shrinking the blade,
// so the blade budget a river would have swallowed lands on dry ground
// instead — the density moves to where plants grow rather than
// thinning the whole tile.  Every blade of a tuft runs the identical
// retry sequence (the salt depends only on tuft index and attempt), so
// tufts relocate WHOLE instead of scattering into loose blades.
const int kGrassWaterRelocates = 3;

// ── Built ground ────────────────────────────────────────────────────
// The same rejection, keyed on the PCG's flatten mask instead of the
// water column: 1 where the terrain was GRADED for a house pad or a
// road slab, 0 on natural ground (TERRAIN_FLAT_MASK_INDEX).  Grass had
// no idea buildings existed -- the scatter is a pure XZ hash over the
// terrain and a house is a separate mesh standing on top of it -- so
// blades sprouted through floors and rose between the furniture.
//
// The mask already carries its own margins (2.5 m of skirt around a
// pad, 3.5 m of verge along a road).  These thresholds sit LOW on top
// of that: the mask is 4 m per texel, so a bilinear reading of 0.35 is
// about 1.4 m outside the graded edge, and stopping there keeps the
// last blade from clipping into a wall's base rather than ending
// exactly at it.  A tuft that lands on built ground RELOCATES first,
// on the same retry budget and the same whole-tuft salt as the
// waterline rejection above, so a village green keeps its blade budget
// instead of the town going bald.
const float kGrassBuiltRelocate  = 0.05f;  // any hint of it: try again
const float kGrassBuiltFadeStart = 0.05f;  // survivor starts shrinking
const float kGrassBuiltFadeEnd   = 0.35f;  // fully gone by here
// Proportions after the reference photograph (late-season veld): the
// cured flowering stems stand 0.4-0.9 m and are a few millimetres
// wide -- wispy, not leafy -- fanning from a crown ~0.4 m across.
// Height range 0.19-0.54 -> 0.36-0.95, width 5-10.5 mm -> 4-9 mm half
// width (the distance widening in grass.mesh keeps them pixel-safe at
// range), tussock radius 0.14 -> 0.20.
const float kGrassTuftRadius = 0.20f;   // m, spread of a clump
const float kGrassHeightMin  = 0.36f;   // m
const float kGrassHeightMax  = 0.95f;   // m
const float kGrassWidthMin   = 0.0040f; // m, HALF width at the widest ring
const float kGrassWidthMax   = 0.0090f;

// Half-width factor and height fraction per ring.  The widest point is
// a third of the way up and the tip closes to a point — a real blade
// tapers from a sheath, it is not a rectangle.
const vec2 kGrassProfile[8] = vec2[8](
    vec2(0.55f, 0.00f),
    vec2(0.95f, 0.17f),
    vec2(1.00f, 0.33f),
    vec2(0.96f, 0.49f),
    vec2(0.85f, 0.63f),
    vec2(0.66f, 0.77f),
    vec2(0.38f, 0.90f),
    vec2(0.04f, 1.00f));

struct GrassBlade {
    vec3  root_ws;
    vec3  side;    // unit, horizontal, across the ribbon at the root
    vec3  arc;     // total horizontal travel of the TIP (lean + wind), m
    float height;  // m
    float width;   // m, half width at the widest ring
    float twist;   // radians the ribbon rotates root -> tip
    float hash;    // per-blade [0,1)
    float dry;     // per-blade dryness [0,1]
    float flower;  // 0 = a blade; 1..4 = a flower stalk, palette index + 1
};

// ── Flowers ──────────────────────────────────────────────────────────
// A flower is drawn by the same 16-vertex / 14-triangle strip as a
// blade, so the mesh shader, the fallback geometry stage and the
// frustum cull take it unchanged.  Rings 0-5 are a thin stalk; rings
// 6 and 7 are NOT a continuation of the ribbon but a small HORIZONTAL
// head at the top of the stalk -- ring 6's two vertices lie along the
// stalk's side axis, ring 7's along the axis at right angles to it, so
// the strip's last four triangles form a diamond disc facing the sky
// (grassBladeVertex).  A disc reads as a bloom from every direction;
// the flat card of the first attempt was a sliver from most.  The
// frag paints it from a four-entry palette (attribs.w = index + 1).
const float kGrassFlowerShare  = 0.14f;   // of tufts on green land, lush
const float kGrassFlowerHeadM  = 0.024f;  // head half-width, m
const float kGrassFlowerStalkM = 0.0022f; // stalk half-width, m
const float kGrassFlowerHeadV  = 0.86f;   // head sits here along the stalk
const vec2 kGrassFlowerStalk[6] = vec2[6](
    vec2(0.70f, 0.00f), vec2(0.80f, 0.20f), vec2(0.80f, 0.40f),
    vec2(0.75f, 0.58f), vec2(0.65f, 0.74f), vec2(0.55f, 0.84f));

// Low-frequency dryness field.  Grass dries in PATCHES — a per-blade
// random would give salt-and-pepper, which reads as noise, not as a
// meadow that is scorched on the south slope and lush by the water.
// Two octaves at ~200 m and ~35 m, both far coarser than a blade.
// The ground is tinted by the SAME field (terrainDryField in
// terrain/tile_common.glsl.h), which is what keeps a cured tuft on
// straw-coloured turf and a lush one on green.  grass.vert / grass.mesh
// / grass.frag include tile_common first and take the shared body;
// grass.geom includes this header alone and gets the verbatim copy --
// keep the two identical.
float grassDryField(vec2 p) {
#ifdef TERRAIN_TILE_COMMON_GLSL_H
    return terrainDryField(p);
#else
    float a = sin(p.x * 0.0312f + 1.7f) * sin(p.y * 0.0271f - 0.4f);
    float b = sin(p.x * 0.0083f - 2.1f) * sin(p.y * 0.0091f + 1.1f);
    return clamp(0.5f + 0.30f * a + 0.42f * b, 0.0f, 1.0f);
#endif
}

vec2 grassRotY(vec2 v, float a) {
    float s = sin(a), c = cos(a);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// Tuft root + per-blade jitter.  h_tuft / h_blade are the caller's two
// hash43 draws (tuft-indexed and blade-indexed).
vec2 grassRootXZ(vec2 tile_min, vec2 tile_range,
                 vec4 h_tuft, vec4 h_blade) {
    vec2 tuft_xz = tile_min + h_tuft.xy * tile_range;
    // Blades of one tuft fan out from a shared root, denser at the
    // centre (sqrt keeps the disc uniform-ish but the bias is wanted).
    float r = kGrassTuftRadius * h_blade.x * h_blade.x;
    float a = h_blade.y * 6.2831853f;
    return tuft_xz + vec2(cos(a), sin(a)) * r;
}

// Fill everything except root_ws.y, which the caller supplies from the
// height field, and `arc`, to which the caller adds wind.
// ── Tuft kinds and tuft size ─────────────────────────────────────────
// Every tuft used to draw its blades from one height range, so the
// field was a carpet of like-sized clumps: variety per blade, none per
// plant.  A real sward is several plants: a low turf of fine short
// leaves, the mid tussocks, and here and there a tall flowering stem
// standing clear of the rest.  So each tuft rolls a KIND (h_kind.x) --
// the mix shifting with the STAND (zone blended with moisture), more
// stems in the tall damp stands and more turf on the short cured
// ground -- and a SIZE (h_kind.y, skewed so
// small is common and big is rare) that scales the whole clump.
//   turf     x0.45 height, wider, greener, all six blades
//   tussock  x1.0, the reference plant
//   stem     x1.6 height, thin, drier, only two or three blades stand
const float kGrassTurfShare  = 0.55f;   // share of turf in a SHORT zone
const float kGrassStemShare  = 0.28f;   // share of stems in a TALL zone

// (height x, width x, dry offset, blade keep threshold) for a kind roll.
// `stand` 0 = short cured sward, 1 = tall damp stand (grassMakeBlade).
vec4 grassTuftKind(float roll, float stand) {
    float p_turf = mix(kGrassTurfShare, 0.15f, stand);
    float p_stem = mix(0.04f, kGrassStemShare, stand);
    if (roll < p_turf)          return vec4(0.45f, 1.15f, -0.20f, 1.0f);
    if (roll > 1.0f - p_stem)   return vec4(1.60f, 0.70f,  0.25f, 0.45f);
    return vec4(1.0f, 1.0f, 0.0f, 1.0f);
}

// Whether a tuft's flower slot holds a flower, and which colour.
// `cover` is grassCoverField at the root (0.15 bare .. 1 green), `dry`
// the dryness field, `roll` the tuft's flower lane.  Flowers stand on
// GREEN land -- the share climbs from nothing at the cover floor to
// kGrassFlowerShare on full green -- and thin out as the meadow cures.
// Returns palette index + 1, or 0 for no flower.
float grassFlowerOf(float roll, float cover, float dry, float colour_roll) {
    float p = kGrassFlowerShare
            * smoothstep(0.55f, 0.95f, cover)
            * (1.0f - 0.65f * dry);
    return (roll < p) ? 1.0f + floor(colour_roll * 3.999f) : 0.0f;
}

// h_kind: the tuft's kind/size draw (grassHash4 salt 4), one per tuft
// -- except .w, which the CALLER sets per slot: 0 for a blade, or the
// flower code (grassFlowerOf) for the tuft's flower slot.
GrassBlade grassMakeBlade(vec2 root_xz, vec4 h_blade, vec4 h_kind,
                          float dry) {
    GrassBlade b;

    float zone  = grassZoneField(root_xz);
    // Site moisture (grassMoisture), read BEFORE the kind's dry offset.
    float moist = grassMoisture(dry);
    // One stand value for the kind mix and the widths: half zone, half
    // moisture, so a damp hollow in a short zone still grows up.
    float stand = clamp(mix(zone, moist, 0.5f), 0.0f, 1.0f);
    vec4  kind  = grassTuftKind(h_kind.x, stand);
    // Per-tuft size: 0.6x .. 1.5x, most clumps a little under 1.
    float size = exp2(mix(-0.75f, 0.60f, h_kind.y * h_kind.y));
    b.flower = h_kind.w;
    if (b.flower > 0.5f) {
        // ── The flower slot, holding a flower ────────────────────────
        // A single stalk in the middle of the clump, a little above
        // the blades around it, thin, live green; the head geometry is
        // grassBladeVertex's.  Leans less than a blade (a stalk is
        // stiffer) and does not cure.
        b.height = mix(0.34f, 0.62f, h_blade.z) * mix(0.75f, 1.15f, stand)
                 * mix(0.85f, 1.15f, size);
        b.width  = kGrassFlowerStalkM;
        float face = h_blade.w * 6.2831853f;
        b.side = vec3(cos(face), 0.0f, sin(face));
        float lean_dir = h_blade.x * 6.2831853f;
        b.arc = vec3(cos(lean_dir), 0.0f, sin(lean_dir))
              * (b.height * mix(0.06f, 0.20f, h_blade.y));
        b.twist = (h_blade.z * 2.0f - 1.0f) * 0.6f;
        b.hash  = h_blade.z;
        b.dry   = clamp(dry - 0.25f, 0.0f, 1.0f);
        b.root_ws = vec3(root_xz.x, 0.0f, root_xz.y);
        return b;
    }
    dry = clamp(dry + kind.z, 0.0f, 1.0f);
    float lush = 1.0f - dry;
    // Height: every blade in a tuft differs -- a tuft of identical
    // blades reads as a fan, not a plant.  MOISTURE sets the stand
    // (kGrassDryHeight .. kGrassMoistHeight: the damp hollows grow
    // tall, the cured ground stays low -- see grassMoisture), the ZONE
    // (grassZoneField) adds its patches of tens of metres on top, then
    // the tuft's kind and size.  Lush blades stay leafier and wider.
    b.height = mix(kGrassHeightMin, kGrassHeightMax, h_blade.z)
             * mix(kGrassDryHeight, kGrassMoistHeight, moist)
             * mix(kGrassZoneLo, kGrassZoneHi, zone)
             * kind.x * size;
    b.width  = mix(kGrassWidthMin, kGrassWidthMax, h_blade.w)
             * (0.85f + 0.30f * lush)
             * mix(0.90f, 1.08f, stand)
             * kind.y * mix(0.85f, 1.15f, h_kind.y);
    // A stem tuft is two or three stalks, not a fan: the other blades
    // of the clump stand down (degenerate, they cost nothing).  So does
    // an EMPTY flower slot (h_kind.w == -1: the slot, no flower).
    if (h_blade.x > kind.w || h_kind.w < -0.5f) {
        b.height = 0.0f;
        b.width  = 0.0f;
    }

    float face = h_blade.w * 6.2831853f;
    b.side = vec3(cos(face), 0.0f, sin(face));

    // Natural lean: taller blades fall over further under their own
    // weight, dry blades further still (they have lost turgor).  The
    // arc is horizontal travel of the tip, as a fraction of height.
    // 0.20-0.58 x height (was 0.14-0.42): a tall thin stem bows well
    // over under the weight of its own head -- the reference stand is
    // all arcs, hardly a vertical blade in it -- and the wind in
    // grass.mesh adds to this.
    float lean_dir = h_blade.x * 6.2831853f;
    float lean_amt = b.height * mix(0.20f, 0.58f, h_blade.y)
                   * (0.8f + 0.5f * dry);
    b.arc = vec3(cos(lean_dir), 0.0f, sin(lean_dir)) * lean_amt;

    // A blade is not a flat card: it rotates along its length, which is
    // what makes a field flicker as the light moves across it.
    b.twist = (h_blade.z * 2.0f - 1.0f) * 1.35f;

    b.hash = h_blade.z;
    b.dry  = dry;
    b.root_ws = vec3(root_xz.x, 0.0f, root_xz.y);
    return b;
}

// One vertex of the blade.  ring in [0, kGrassRings), side_sign is -1/+1.
void grassBladeVertex(GrassBlade b, int ring, float side_sign,
                      out vec3 pos_ws, out vec3 nrm_ws, out float v) {
    if (b.flower > 0.5f && ring >= 6) {
        // ── Flower head: a horizontal diamond disc on the stalk tip ──
        // Both head rings sit at kGrassFlowerHeadV; ring 6 spans the
        // stalk's side axis, ring 7 the axis at right angles, so the
        // strip's last triangles tile a disc that faces the sky (tilted
        // a little with the stalk's lean).  v = 0.9 / 1.0 so the frag's
        // head test (v >= kGrassFlowerHeadV) paints all of it.
        float vh = kGrassFlowerHeadV;
        v = (ring == 6) ? 0.90f : 1.00f;
        float arc_len = length(b.arc);
        float droop = 0.5f * arc_len * arc_len / max(b.height, 0.05f);
        vec3 c = b.root_ws;
        c.y += b.height * vh;
        c   += b.arc * (vh * vh);
        c.y -= droop * vh * vh * vh;
        c.y += (ring == 7) ? 0.006f : 0.0f;      // a shallow cone
        vec3 T = normalize(vec3(0.0f, b.height, 0.0f) + b.arc * (2.0f * vh));
        vec2 s2 = grassRotY(b.side.xz, b.twist * vh);
        vec3 side = normalize(vec3(s2.x, 0.0f, s2.y) - T * dot(T, vec3(s2.x, 0.0f, s2.y)));
        vec3 fwd  = normalize(cross(T, side));
        vec3 axis = (ring == 6) ? side : fwd;
        pos_ws = c + axis * (kGrassFlowerHeadM * side_sign);
        nrm_ws = normalize(T + axis * (side_sign * 0.35f));
        return;
    }
    // A flower stalk uses its own (thinner, shorter) profile.
    vec2 prof = (b.flower > 0.5f) ? kGrassFlowerStalk[ring]
                                  : kGrassProfile[ring];
    v = prof.y;
    float w = b.width * prof.x;

    // Ribbon cross-direction, rotated about the blade axis by the twist.
    vec2 s2 = grassRotY(b.side.xz, b.twist * v);
    vec3 side = vec3(s2.x, 0.0f, s2.y);

    float arc_len = length(b.arc);
    // Arc, not shear: the tip travels horizontally as v^2 and DIPS by
    // the amount that travel steals from its height, so the blade keeps
    // its length instead of stretching.
    float droop = 0.5f * arc_len * arc_len / max(b.height, 0.05f);

    vec3 c = b.root_ws;
    c.y += b.height * v;
    c   += b.arc * (v * v);
    c.y -= droop * v * v * v;

    // dc/dv -> along-blade tangent.
    vec3 T = vec3(0.0f, b.height, 0.0f) + b.arc * (2.0f * v);
    T.y   -= 3.0f * droop * v * v;
    T = normalize(T);

    // Face normal of the ribbon, then tilted outward at the two edges so
    // the blade has a rounded cross-section.  A perfectly flat card
    // gives every blade one uniform shade and is the other half of why
    // the field read as cut-out paper.
    vec3 n = cross(T, side);
    if (dot(n, vec3(0.0f, 1.0f, 0.0f)) < 0.0f) n = -n;
    nrm_ws = normalize(n + side * (side_sign * 0.45f));

    pos_ws = c + side * (w * side_sign);
}

#endif // GRASS_COMMON_GLSL_H
