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
// The tint squares the albedo: transmitted light traverses the pigment
// layer, so its spectrum is filtered ~twice as hard as reflected light
// — this is what makes backlit leaves read as saturated green-gold.
//
// `shad` is the sun visibility.  Translucency is NOT multiplied by it
// directly: the leaves that should glow the most are exactly the ones
// inside their own canopy's shadow map.  mix keeps a fraction of the
// term in shadow (interior canopy glow) while still killing most of it
// behind genuine occluders (a mountain, a wall).
//
// Consumers: cluster_bindless.frag forward branch (flag test on
// BINDLESS_MAT_FOLIAGE_SSS) and deferred_resolve.comp (flag decoded
// from gbuf_normal_rough.w).  Divide-by-π matches the Lambertian
// direct term both paths use, so the scale constant is comparable.
const float FOLIAGE_SSS_DISTORTION  = 0.35;
const float FOLIAGE_SSS_POWER       = 3.0;
const float FOLIAGE_SSS_SCALE       = 1.6;
const float FOLIAGE_SSS_AMBIENT     = 0.10;  // view-independent leak
const float FOLIAGE_SSS_SHADOW_MIN  = 0.30;  // in-shadow fraction kept

vec3 foliageTranslucency(vec3 N, vec3 V, vec3 L, vec3 albedo,
                         vec3 light_col, float shad) {
    vec3  Lt   = normalize(L + N * FOLIAGE_SSS_DISTORTION);
    float back = pow(clamp(dot(V, -Lt), 0.0, 1.0), FOLIAGE_SSS_POWER)
                 * FOLIAGE_SSS_SCALE;
    vec3  tint = albedo * albedo;          // double-filtered transmission
    float vis  = mix(FOLIAGE_SSS_SHADOW_MIN, 1.0, shad);
    return tint * light_col *
           ((back + FOLIAGE_SSS_AMBIENT) * vis / M_PI);
}

vec3 getDirectionByYawAndPitch(float yaw, float pitch) {
    vec3 direction;
    direction.x = cos(radians(-yaw)) * cos(radians(pitch));
    direction.y = sin(radians(pitch));
    direction.z = sin(radians(-yaw)) * cos(radians(pitch));
    return normalize(direction);
}

#endif // FUNCTIONS_GLSL_H
