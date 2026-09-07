// Presence flag: lets a header that MAY be included with or without
// this one (grass_common.glsl.h) pick the shared definitions below.
#define TERRAIN_TILE_COMMON_GLSL_H 1

mat3 m3 = mat3(
    0.00f, 0.80f, 0.60f,
    -0.80f, 0.36f, -0.48f,
    -0.60f, -0.48f, 0.64f);
mat3 m3i = mat3(
    0.00f, -0.80f, -0.60f,
    0.80f, 0.36f, -0.48f,
    0.60f, -0.48f, 0.64f);
mat2 m2 = mat2(
    0.80f, 0.60f,
    -0.60f, 0.80f);
mat2 m2i = mat2(
    0.80f, -0.60f,
    0.60f, 0.80f);

float hash1(vec2 p)
{
    p = 50.0f * fract(p * 0.3183099f);
    return fract(p.x * p.y * (p.x + p.y));
}

float hash1(float n)
{
    return fract(n * 17.0f * fract(n * 0.3183099f));
}

float noise(vec2 x)
{
    vec2 p = floor(x);
    vec2 w = fract(x);
#if 1
    vec2 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);
#else
    vec2 u = w * w * (3.0f - 2.0f * w);
#endif

    float a = hash1(p + vec2(0, 0));
    float b = hash1(p + vec2(1, 0));
    float c = hash1(p + vec2(0, 1));
    float d = hash1(p + vec2(1, 1));

    return -1.0f + 2.0f * (a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y);
}

// value noise, and its analytical derivatives
vec4 noised(vec3 x)
{
    vec3 p = floor(x);
    vec3 w = fract(x);
#if 1
    vec3 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);
    vec3 du = 30.0f * w * w * (w * (w - 2.0f) + 1.0f);
#else
    vec3 u = w * w * (3.0f - 2.0f * w);
    vec3 du = 6.0f * w * (1.0f - w);
#endif

    float n = p.x + 317.0f * p.y + 157.0f * p.z;

    float a = hash1(n + 0.0f);
    float b = hash1(n + 1.0f);
    float c = hash1(n + 317.0f);
    float d = hash1(n + 318.0f);
    float e = hash1(n + 157.0f);
    float f = hash1(n + 158.0f);
    float g = hash1(n + 474.0f);
    float h = hash1(n + 475.0f);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return vec4(-1.0f + 2.0f * (k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z),
        2.0f * du * vec3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
            k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
            k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y));
}

vec3 noised(vec2 x)
{
    vec2 p = floor(x);
    vec2 w = fract(x);
#if 1
    vec2 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);
    vec2 du = 30.0f * w * w * (w * (w - 2.0f) + 1.0f);
#else
    vec2 u = w * w * (3.0f - 2.0f * w);
    vec2 du = 6.0f * w * (1.0f - w);
#endif

    float a = hash1(p + vec2(0, 0));
    float b = hash1(p + vec2(1, 0));
    float c = hash1(p + vec2(0, 1));
    float d = hash1(p + vec2(1, 1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3(-1.0f + 2.0f * (k0 + k1 * u.x + k2 * u.y + k4 * u.x * u.y),
        2.0f * du * vec2(k1 + k4 * u.y,
            k2 + k4 * u.x));
}

vec4 fbmd_8(vec3 x)
{
    float f = 1.92f;
    float s = 0.5f;
    float a = 0.0f;
    float b = 0.5f;
    vec3  d = vec3(0.0f);
    mat3  m = mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f);
    for (int i = 0; i < 7; i++)
    {
        vec4 n = noised(x);
        a += b * n.x;          // accumulate values		
        d += b * m * vec3(n.y, n.z, n.w);      // accumulate derivatives
        b *= s;
        x = f * m3 * x;
        m = f * m3i * m;
    }
    return vec4(a, d);
}

float fbm_9(vec2 x)
{
    float f = 1.9f;
    float s = 0.55f;
    float a = 0.0f;
    float b = 0.5f;
    for (int i = 0; i < 9; i++)
    {
        float n = noise(x);
        a += b * n;
        b *= s;
        x = f * m2 * x;
    }
    return a;
}

vec3 fbmd_9(vec2 x)
{
    float f = 1.97f;
    float s = 0.557f;
    float a = 0.0f;
    float b = 0.5f;
    vec2  d = vec2(0.0);
    mat2  m = mat2(1.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < 9; i++)
    {
        vec3 n = noised(x);
        a += b * n.x;          // accumulate values		
        d += b * m * vec2(n.y, n.z);       // accumulate derivatives
        b *= s;
        x = f * m2 * x;
        m = f * m2i * m;
    }
    return vec3(a, d);
}

// return smoothstep and its derivative
vec2 smoothstepd(float a, float b, float x)
{
    if (x < a) return vec2(0.0, 0.0);
    if (x > b) return vec2(1.0, 0.0);
    float ir = 1.0f / (b - a);
    x = (x - a) * ir;
    return vec2(x * x * (3.0f - 2.0f * x), 6.0f * x * (1.0f - x) * ir);
}

//------------------------------------------------------------------------------------------
// terrain
//------------------------------------------------------------------------------------------
vec2 terrainMap(vec2 p, float sca/*= 0.00025f*/, float amp/*= 2000.0f*/)
{
    p *= sca;
    float e = fbm_9(p + vec2(1.0f, -2.0f));
    float a = 1.0f - smoothstep(0.12f, 0.13f, abs(e + 0.12f)); // flag high-slope areas (-0.25, 0.0)
    e = e + 0.15f * smoothstep(-0.08f, -0.01f, e);
    e *= amp;
    e += amp * 0.5f;
    return vec2(e, a);
}

vec4 terrainMapD(vec2 p, float sca/*= 0.00025f*/, float amp/*= 2000.0f*/)
{
    p *= sca;
    vec3 e = fbmd_9(p + vec2(1.0f, -2.0f));
    vec2 c = smoothstepd(-0.08f, -0.01f, e.x);
    vec2 e_yz = vec2(e.y, e.z);
    e.x = e.x + 0.15f * c.x;
    e_yz = e_yz + 0.15f * c.y * e_yz;
    e.x *= amp;
    e_yz *= amp * sca;
    return vec4(e.x, normalize(vec3(-e_yz.x, 1.0f, -e_yz.y)));
}

vec3 terrainNormal(vec2 pos, float sca/*= 0.00025f*/, float amp/*= 2000.0f*/)
{
#if 1
    vec4 map_d = terrainMapD(pos, sca, amp);
    return vec3(map_d.y, map_d.z, map_d.w);
#else    
    vec2 e = vec2(0.03, 0.0);
    vec2 e_xy = vec2(e.x, e.y);
    vec2 e_yx = vec2(e.y, e.x);
    return normalize(vec3(terrainMap(pos - e_xy).x - terrainMap(pos + e_xy).x,
        2.0 * e.x,
        terrainMap(pos - e_yx).x - terrainMap(pos + e_yx).x));
#endif    
}

// Alexander Lemke, 2017

vec2 hash2D(in vec2 p)
{
    return fract(sin(p * mat2(12.98, 78.23, 127.99, 311.33)) * 43758.54);
}


// Virtually the same as your original function, just in more compact (and possibly less reliable) form.
float smoothNoise(vec2 p) {

    vec2 f = fract(p); p -= f; f *= f * (3. - f - f);

    return dot(mat2(fract(sin(vec4(0, 1, 27, 28) + p.x + p.y * 27.) * 1e5)) * vec2(1. - f.y, f.y), vec2(1. - f.x, f.x));

}

// ── Macro colour grade ───────────────────────────────────────────────
// The colour maps the generator paints -- and the 1 m detail tiles
// upscaled from them -- come out pale and washed: the current world's
// map measures a median luma of 0.55 at ~15% saturation, and its town
// centre is a near-white beige.  0.55 is chalk; grass, soil and leaf
// litter sit at 0.10-0.30, dry sand at ~0.40.  Lit by a full sky that
// is most of why the ground reads as a flat white sheet.
//
// Gamma > 1 pulls the mid-tones down while leaving white (the snow
// caps, which terrainMaterialWeights keys on being NEAR-WHITE) and
// black alone; the saturation lift is luma-preserving, so it colours
// the ground without brightening it.  Applied to the resolved macro
// colour in tile.frag AFTER the material layers (so the layer
// classification still sees the authored map) and to the ground sample
// grass.frag pulls its blade hue from, so the two agree.  1 / 1 = off.
#define kMacroGradeGamma   1.15f
#define kMacroGradeSat     1.30f
vec3 terrainGradeAlbedo(vec3 a) {
    a = pow(max(a, vec3(0.0f)), vec3(kMacroGradeGamma));
    float l = dot(a, vec3(0.2126f, 0.7152f, 0.0722f));
    return clamp(mix(vec3(l), a, kMacroGradeSat), vec3(0.0f), vec3(1.0f));
}

// ── Dryness field ────────────────────────────────────────────────────
// Where the meadow has cured.  Two sin octaves at ~200 m and ~35 m,
// both far coarser than a blade or a texel, so grass dries in DRIFTS
// -- the scorched south slope, the lush strip along the water -- and
// never as salt-and-pepper.  ONE field for the blades and the ground
// under them: grassDryField() (grass/grass_common.glsl.h) forwards to
// this wherever tile_common is in scope and carries a verbatim copy for
// the one stage that includes it alone, and tile.frag tints the turf
// with it, so a straw tuft never stands on green ground and a green one
// never on straw.  0 = lush, 1 = fully cured.
// Baseline 0.64 (was 0.50): the reference is late-season grassland --
// a golden sea with green only in the hollows and along water -- so
// the field sits cured by default and the lush drifts are the
// exception, not the other way round.
float terrainDryField(vec2 p) {
    float a = sin(p.x * 0.0312f + 1.7f) * sin(p.y * 0.0271f - 0.4f);
    float b = sin(p.x * 0.0083f - 2.1f) * sin(p.y * 0.0091f + 1.1f);
    return clamp(0.64f + 0.30f * a + 0.42f * b, 0.0f, 1.0f);
}

// Dryness -> "how cured does it LOOK", shared by the blades and the
// turf under them so the two brown together.  Was d*d in both places,
// which kept everything below d ~0.7 reading green; a blade is
// visibly straw well before it is fully dead, so the curve now rises
// through the middle of the range and saturates at the top.
float terrainDryCurve(float d) {
    return smoothstep(0.12f, 0.85f, d);
}

// Also the same as the original, but with one less layer.
float fractalNoise(vec2 p) {

    return smoothNoise(p) * 0.5333 + smoothNoise(p * 2.) * 0.2667 + smoothNoise(p * 4.) * 0.1333 + smoothNoise(p * 8.) * 0.0667;

    // Similar version with fewer layers. The highlighting sample distance would need to be tweaked.
    //return smoothNoise(p)*0.57 + smoothNoise(p*2.45)*0.28 + smoothNoise(p*6.)*0.15;

    // Even fewer layers, but the sample distance would need to be tweaked.
    //return smoothNoise(p)*0.65 + smoothNoise(p*4.)*0.35;

}
