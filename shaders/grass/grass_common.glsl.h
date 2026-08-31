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

const float kGrassTuftBlades = 5.0f;    // blades per root clump

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
const float kGrassTuftRadius = 0.14f;   // m, spread of a clump
const float kGrassHeightMin  = 0.19f;   // m
const float kGrassHeightMax  = 0.54f;   // m
const float kGrassWidthMin   = 0.0050f; // m, HALF width at the widest ring
const float kGrassWidthMax   = 0.0105f;

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
};

// Low-frequency dryness field.  Grass dries in PATCHES — a per-blade
// random would give salt-and-pepper, which reads as noise, not as a
// meadow that is scorched on the south slope and lush by the water.
// Two octaves at ~200 m and ~35 m, both far coarser than a blade.
float grassDryField(vec2 p) {
    float a = sin(p.x * 0.0312f + 1.7f) * sin(p.y * 0.0271f - 0.4f);
    float b = sin(p.x * 0.0083f - 2.1f) * sin(p.y * 0.0091f + 1.1f);
    return clamp(0.5f + 0.30f * a + 0.42f * b, 0.0f, 1.0f);
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
GrassBlade grassMakeBlade(vec2 root_xz, vec4 h_blade, float dry) {
    GrassBlade b;

    float lush = 1.0f - dry;
    // Height: dry patches are shorter and stubbier, and every blade in a
    // tuft differs — a tuft of identical blades reads as a fan, not a
    // plant.
    b.height = mix(kGrassHeightMin, kGrassHeightMax,
                   h_blade.z * (0.45f + 0.55f * lush));
    b.width  = mix(kGrassWidthMin, kGrassWidthMax, h_blade.w)
             * (0.85f + 0.30f * lush);

    float face = h_blade.w * 6.2831853f;
    b.side = vec3(cos(face), 0.0f, sin(face));

    // Natural lean: taller blades fall over further under their own
    // weight, dry blades further still (they have lost turgor).  The
    // arc is horizontal travel of the tip, as a fraction of height.
    float lean_dir = h_blade.x * 6.2831853f;
    float lean_amt = b.height * mix(0.14f, 0.42f, h_blade.y)
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
    v = kGrassProfile[ring].y;
    float w = b.width * kGrassProfile[ring].x;

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
