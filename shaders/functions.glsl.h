#ifndef FUNCTIONS_GLSL_H
#define FUNCTIONS_GLSL_H

const float M_PI = 3.141592653589793;

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;

// linear to sRGB approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec3 linearTosRGB(vec3 color)
{
    return pow(color, vec3(INV_GAMMA));
}

// sRGB to linear approximation
// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec3 sRGBToLinear(vec3 srgbIn)
{
    return vec3(pow(srgbIn, vec3(GAMMA)));
}

vec4 sRGBToLinear(vec4 srgbIn)
{
    return vec4(sRGBToLinear(srgbIn.xyz), srgbIn.w);
}

#include "tonemap.glsl.h"

struct NormalInfo {
    vec3 ng;   // Geometric normal
    vec3 n;    // Pertubed normal
    vec3 t;    // Pertubed tangent
    vec3 b;    // Pertubed bitangent
};

float clampedDot(vec3 x, vec3 y)
{
    return clamp(dot(x, y), 0.0, 1.0);
}

float sq(float t)
{
    return t * t;
}

vec2 sq(vec2 t)
{
    return t * t;
}

vec3 sq(vec3 t)
{
    return t * t;
}

vec4 sq(vec4 t)
{
    return t * t;
}

vec3 transmissionAbsorption(vec3 v, vec3 n, float ior, float thickness, vec3 absorptionColor)
{
    vec3 r = refract(-v, n, 1.0 / ior);
    return exp(-absorptionColor * thickness * dot(-n, r));
}

float rsi(vec3 r0, vec3 rd, float sr) {
    // Simplified ray-sphere intersection that assumes
    // the ray starts inside the sphere and that the
    // sphere is centered at the origin. Always intersects.
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    return (-b + sqrt((b * b) - 4.0 * a * c)) / (2.0 * a);
}

// ── Foliage translucency (thin-slab subsurface scattering) ──────────
// Barré-Brisebois & Bouchard, "Approximating Translucency for a Fast,
// Cheap and Convincing Subsurface Scattering Look" (GDC 2011).  A leaf
// is a thin slab: sunlight entering the BACK face exits the front
// tinted by the pigment, strongest when the camera looks toward the
// light through the leaf.  The distorted transmission direction
// (L + N*distortion) tilts the lobe so leaves angled across the light
// still glow along their surface, and the pow() sharpens the lobe so
// the effect reads as backlighting rather than uniform brightening.
//
// The per-term detail now lives on foliageTranslucency itself, below.
const float FOLIAGE_SSS_DISTORTION  = 0.35;
const float FOLIAGE_SSS_POWER       = 3.0;
const float FOLIAGE_SSS_SCALE       = 1.6;
const float FOLIAGE_SSS_AMBIENT     = 0.10;  // view-independent leak
const float FOLIAGE_SSS_SHADOW_MIN  = 0.30;  // in-shadow fraction kept
// Isotropic share of the transmitted light: what comes through the
// slab regardless of where the viewer stands, as opposed to the
// forward-scatter lobe above which only fires looking INTO the sun.
// Without it a backlit canopy went black the moment the view left the
// sun's line, because the whole transmission term was one sharp lobe.
const float FOLIAGE_T_DIFFUSE       = 0.55;

// ── THE TWO SIDES OF A LEAF ─────────────────────────────────────────
// A leaf is a thin slab with DIFFERENT surfaces front and back: the
// adaxial (upper) face carries the waxy cuticle and the palisade
// pigment layer — glossy and saturated; the abaxial (under) face is
// spongy mesophyll — matte and paler.  Shading both with one material
// is what makes a canopy read as flat green cards from below.
// `front` is 1.0 on a front-facing fragment, 0.0 on a back-facing one.
const float LEAF_BACK_DESAT = 0.35;   // pull toward luminance
const float LEAF_BACK_GAIN  = 1.15;   // ...and lift, undersides are paler
const float LEAF_SPEC_FRONT = 1.00;   // waxy cuticle: full sheen
const float LEAF_SPEC_BACK  = 0.25;   // underside: nearly matte

// Cheap stand-in for a second (abaxial) albedo texture.
vec3 leafBacksideAlbedo(vec3 albedo) {
    float g = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    return min(mix(albedo, vec3(g), LEAF_BACK_DESAT) * LEAF_BACK_GAIN,
               vec3(1.0));
}

// Specular weight for whichever side is being looked at.
float leafSpecularSide(float front) {
    return mix(LEAF_SPEC_BACK, LEAF_SPEC_FRONT, front);
}

// TRANSMITTED lighting — the lighting problem on the FAR side of the
// slab, which is a separate problem from the reflected one the caller
// already solved with the view-side normal N.  Light has to land on
// the face pointing AWAY from the viewer before any of it can come
// through, so the whole term is gated on dot(-N, L): the old version
// had no such gate and leaked a glow onto front-lit leaves at grazing
// angles, which read as leaves emitting light.
//
// What survives that gate splits in two:
//   • FOLIAGE_T_DIFFUSE — isotropic bleed through the slab, visible
//     from anywhere on this side.
//   • the forward-scatter lobe — sharp, peaks looking into the sun,
//     the "lit stained glass" of a backlit canopy.  DISTORTION bends
//     the transmitted direction back toward the surface so leaves
//     edge-on to the sun still glow along their length.
//
// The tint squares the albedo: transmitted light crosses the pigment
// layer, so its spectrum is filtered ~twice as hard as reflected light
// — this is what makes backlit leaves read as saturated green-gold.
//
// `shad` is sun visibility, and translucency is deliberately NOT
// multiplied by it directly: the leaves that should glow most are
// exactly the ones inside their own canopy's shadow map.  mix keeps a
// fraction in shadow (interior canopy glow) while still killing most
// of it behind a genuine occluder (a mountain, a wall).
//
// Consumers: cluster_bindless.frag forward branch (flag test on
// BINDLESS_MAT_FOLIAGE_SSS) and deferred_resolve.comp (flag decoded
// from gbuf_normal_rough.w).  Divide-by-pi matches the Lambertian
// direct term both paths use, so the scale constant is comparable.
vec3 foliageTranslucency(vec3 N, vec3 V, vec3 L, vec3 albedo,
                         vec3 light_col, float shad) {
    // Light landing on the side the viewer cannot see.
    float back_NdotL = clamp(dot(-N, L), 0.0, 1.0);
    vec3  Lt   = normalize(L + N * FOLIAGE_SSS_DISTORTION);
    float fwd  = pow(clamp(dot(V, -Lt), 0.0, 1.0), FOLIAGE_SSS_POWER)
                 * FOLIAGE_SSS_SCALE;
    vec3  tint = albedo * albedo;          // double-filtered transmission
    float vis  = mix(FOLIAGE_SSS_SHADOW_MIN, 1.0, shad);
    return tint * light_col *
           ((back_NdotL * (FOLIAGE_T_DIFFUSE + fwd) +
             FOLIAGE_SSS_AMBIENT) * vis / M_PI);
}

// ── GROUND-BOUNCE FILL ──────────────────────────────────────────────
// The single bounce a renderer with GI switched off has NO path for at
// all.  A shadowed pixel's entire light budget is `ambient`, and
// `ambient` is one lookup in a sky cube — so the light that hit the
// sunlit ground two metres away and scattered onto this wall, this
// trunk, the underside of this canopy, is not attenuated in the model,
// it is simply ABSENT.  Outdoors over bright ground that term is of the
// same order as the sky, which is why shaded verticals read as black
// while the sky above them is blown white.
//
// The estimate is the textbook infinite-Lambertian-plane one, and the
// pi cancels out of it exactly:
//
//   E_ground   = sun_col * max(L.y, 0)         irradiance on flat ground
//   L_ground   = albedo_g * E_ground / pi      its exit radiance
//   F(N)       = (1 - N.y) / 2                 view factor, plane below
//   E_bounce   = pi * L_ground * F(N)
//              = albedo_g * sun_col * max(L.y,0) * (1 - N.y)/2
//
// The view factor is what makes this well-behaved: an up-facing surface
// (N.y = 1) gets nothing, because flat ground does not bounce onto
// itself; a wall gets half; a downward-facing leaf underside gets all
// of it.  That is exactly the distribution missing from the frame.
//
// NOT multiplied by the receiver's shadow term.  The light arriving
// here came off the LIT ground nearby — a pixel in shadow is precisely
// the one that needs it, and multiplying by its own sun visibility
// would zero it exactly where it matters.  Callers still fold in their
// ambient occlusion: a crevice sees less of the ground too.
//
// WHAT IT IGNORES, and why the coefficient is under-set because of it:
// that the bouncing ground may itself be in shadow, that it is finite,
// and that it is not always below (a valley wall, a floor indoors).
// 0.22 sits deliberately below real dry sand (~0.4) or concrete (~0.3)
// to pay for those; raise it toward 0.35 for a desert map, drop it
// toward 0.12 for wet asphalt or deep forest floor.
//
// Tint it here if the ground has a strong colour — bounce off red rock
// or autumn litter is not neutral — but a per-map constant is a lie a
// single scene can afford and a whole world cannot, which is the real
// argument for turning RT GI back on rather than tuning this.
const float GROUND_BOUNCE_ALBEDO = 0.22;

vec3 groundBounceIrradiance(vec3 N, vec3 L, vec3 sun_col) {
    float sun_on_ground = max(L.y, 0.0);      // 0 at and below sunset
    float view_factor   = 0.5 - 0.5 * N.y;    // (1 - N.y) / 2
    return sun_col * (GROUND_BOUNCE_ALBEDO * sun_on_ground * view_factor);
}

vec3 getDirectionByYawAndPitch(float yaw, float pitch) {
    vec3 direction;
    direction.x = cos(radians(-yaw)) * cos(radians(pitch));
    direction.y = sin(radians(pitch));
    direction.z = sin(radians(-yaw)) * cos(radians(pitch));
    return normalize(direction);
}

#endif // FUNCTIONS_GLSL_H
