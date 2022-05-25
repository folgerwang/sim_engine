/*
    FX implementation of Ken Perlin's "Improved Noise"
    sgg 6/26/04
    http://mrl.nyu.edu/~perlin/noise/
*/

layout(set = 0, binding = PERMUTATION_TEXTURE_INDEX) uniform sampler2D permutation_tex;
layout(set = 0, binding = PERMUTATION_2D_TEXTURE_INDEX) uniform sampler2D permutation_2d_tex;
layout(set = 0, binding = GRAD_TEXTURE_INDEX) uniform sampler2D grad_tex;
layout(set = 0, binding = PERM_GRAD_TEXTURE_INDEX) uniform sampler2D perm_grad_tex;
layout(set = 0, binding = PERM_GRAD_4D_TEXTURE_INDEX) uniform sampler2D perm_grad_4d_tex;
layout(set = 0, binding = GRAD_4D_TEXTURE_INDEX) uniform sampler2D grad_4d_tex;

vec3 fade(vec3 t)
{
    return t * t * t * (t * (t * 6 - 15) + 10); // new curve
}

vec4 fade(vec4 t)
{
    return t * t * t * (t * (t * 6 - 15) + 10); // new curve
}

float perm(float x)
{
    return texture(permutation_tex, vec2(x, 0.5)).x;
}

vec4 perm2d(vec2 p)
{
    return texture(permutation_2d_tex, p);
}

float grad(float x, vec3 p)
{
    return dot(texture(grad_tex, vec2(x * 16, 0.5)).xyz, p);
}

float gradperm(float x, vec3 p)
{
    return dot(texture(perm_grad_tex, vec2(x, 0.5)).xyz, p);
}

// 4d versions
float grad(float x, vec4 p)
{
    return dot(texture(grad_4d_tex, vec2(x, 0.5)), p);
}

float gradperm(float x, vec4 p)
{
    return dot(texture(perm_grad_4d_tex, vec2(x, 0.5)), p);
}

// 3D noise
#if 0

// original version
float inoise(vec3 p)
{
    vec3 P = fmod(floor(p), 256.0);	// FIND UNIT CUBE THAT CONTAINS POINT
    p -= floor(p);                      // FIND RELATIVE X,Y,Z OF POINT IN CUBE.
    vec3 f = fade(p);                 // COMPUTE FADE CURVES FOR EACH OF X,Y,Z.

    P = P / 256.0;
    const float one = 1.0 / 256.0;

    // HASH COORDINATES OF THE 8 CUBE CORNERS
    float A = perm(P.x) + P.y;
    vec4 AA;
    AA.x = perm(A) + P.z;
    AA.y = perm(A + one) + P.z;
    float B = perm(P.x + one) + P.y;
    AA.z = perm(B) + P.z;
    AA.w = perm(B + one) + P.z;

    // AND ADD BLENDED RESULTS FROM 8 CORNERS OF CUBE
    return mix(mix(mix(grad(perm(AA.x), p),
        grad(perm(AA.z), p + vec3(-1, 0, 0)), f.x),
        mix(grad(perm(AA.y), p + vec3(0, -1, 0)),
            grad(perm(AA.w), p + vec3(-1, -1, 0)), f.x), f.y),

        mix(mix(grad(perm(AA.x + one), p + vec3(0, 0, -1)),
            grad(perm(AA.z + one), p + vec3(-1, 0, -1)), f.x),
            mix(grad(perm(AA.y + one), p + vec3(0, -1, -1)),
                grad(perm(AA.w + one), p + vec3(-1, -1, -1)), f.x), f.y), f.z);
}

#else

// optimized version
float inoise(vec3 p)
{
    vec3 P = mod(floor(p), 256.0);	// FIND UNIT CUBE THAT CONTAINS POINT
    p -= floor(p);                      // FIND RELATIVE X,Y,Z OF POINT IN CUBE.
    vec3 f = fade(p);                 // COMPUTE FADE CURVES FOR EACH OF X,Y,Z.

    P = P / 256.0;
    const float one = 1.0 / 256.0;

    // HASH COORDINATES OF THE 8 CUBE CORNERS
    vec4 AA = perm2d(P.xy) + P.z;

    // AND ADD BLENDED RESULTS FROM 8 CORNERS OF CUBE
    return mix(mix(mix(gradperm(AA.x, p),
        gradperm(AA.z, p + vec3(-1, 0, 0)), f.x),
        mix(gradperm(AA.y, p + vec3(0, -1, 0)),
            gradperm(AA.w, p + vec3(-1, -1, 0)), f.x), f.y),

        mix(mix(gradperm(AA.x + one, p + vec3(0, 0, -1)),
            gradperm(AA.z + one, p + vec3(-1, 0, -1)), f.x),
            mix(gradperm(AA.y + one, p + vec3(0, -1, -1)),
                gradperm(AA.w + one, p + vec3(-1, -1, -1)), f.x), f.y), f.z);
}

#endif

// 4D noise
float inoise(vec4 p)
{
    vec4 P = mod(floor(p), 256.0);	// FIND UNIT HYPERCUBE THAT CONTAINS POINT
    p -= floor(p);                      // FIND RELATIVE X,Y,Z OF POINT IN CUBE.
    vec4 f = fade(p);                 // COMPUTE FADE CURVES FOR EACH OF X,Y,Z, W
    P = P / 256.0;
    const float one = 1.0 / 256.0;

    // HASH COORDINATES OF THE 16 CORNERS OF THE HYPERCUBE
    float A = perm(P.x) + P.y;
    float AA = perm(A) + P.z;
    float AB = perm(A + one) + P.z;
    float B = perm(P.x + one) + P.y;
    float BA = perm(B) + P.z;
    float BB = perm(B + one) + P.z;

    float AAA = perm(AA) + P.w, AAB = perm(AA + one) + P.w;
    float ABA = perm(AB) + P.w, ABB = perm(AB + one) + P.w;
    float BAA = perm(BA) + P.w, BAB = perm(BA + one) + P.w;
    float BBA = perm(BB) + P.w, BBB = perm(BB + one) + P.w;

    // INTERPOLATE DOWN
    return mix(
        mix(mix(mix(grad(perm(AAA), p),
            grad(perm(BAA), p + vec4(-1, 0, 0, 0)), f.x),
            mix(grad(perm(ABA), p + vec4(0, -1, 0, 0)),
                grad(perm(BBA), p + vec4(-1, -1, 0, 0)), f.x), f.y),

            mix(mix(grad(perm(AAB), p + vec4(0, 0, -1, 0)),
                grad(perm(BAB), p + vec4(-1, 0, -1, 0)), f.x),
                mix(grad(perm(ABB), p + vec4(0, -1, -1, 0)),
                    grad(perm(BBB), p + vec4(-1, -1, -1, 0)), f.x), f.y), f.z),

        mix(mix(mix(grad(perm(AAA + one), p + vec4(0, 0, 0, -1)),
            grad(perm(BAA + one), p + vec4(-1, 0, 0, -1)), f.x),
            mix(grad(perm(ABA + one), p + vec4(0, -1, 0, -1)),
                grad(perm(BBA + one), p + vec4(-1, -1, 0, -1)), f.x), f.y),

            mix(mix(grad(perm(AAB + one), p + vec4(0, 0, -1, -1)),
                grad(perm(BAB + one), p + vec4(-1, 0, -1, -1)), f.x),
                mix(grad(perm(ABB + one), p + vec4(0, -1, -1, -1)),
                    grad(perm(BBB + one), p + vec4(-1, -1, -1, -1)), f.x), f.y), f.z), f.w);
}

// utility functions

// calculate gradient of noise (expensive!)
vec3 inoiseGradient(vec3 p, float d)
{
    float f0 = inoise(p);
    float fx = inoise(p + vec3(d, 0, 0));
    float fy = inoise(p + vec3(0, d, 0));
    float fz = inoise(p + vec3(0, 0, d));
    return vec3(fx - f0, fy - f0, fz - f0) / d;
}

// fractal sum
float fBm(vec3 p, int octaves, float lacunarity/* = 2.0*/, float gain/* = 0.5*/)
{
    float freq = 1.0, amp = 0.5;
    float sum = 0;
    for (int i = 0; i < octaves; i++) {
        sum += inoise(p * freq) * amp;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum;
}

float turbulence(vec3 p, int octaves, float lacunarity/* = 2.0*/, float gain/* = 0.5*/)
{
    float sum = 0;
    float freq = 1.0, amp = 1.0;
    for (int i = 0; i < octaves; i++) {
        sum += abs(inoise(p * freq)) * amp;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum;
}

float turbulence(vec4 n0, vec4 n1)
{
    float sum = dot(n0, vec4(0.5f, 0.25f, 0.125f, 0.0625f));
    return sum;
}

// Ridged multifractal
// See "Texturing & Modeling, A Procedural Approach", Chapter 12
float ridge(float h, float offset)
{
    h = abs(h);
    h = offset - h;
    h = h * h;
    return h;
}

float ridgedmf(vec3 p, int octaves, float lacunarity/* = 2.0*/, float gain/* = 0.5*/, float offset/* = 1.0*/)
{
    float sum = 0;
    float freq = 1.0, amp = 0.5;
    float prev = 1.0;
    for (int i = 0; i < octaves; i++) {
        float n = ridge(inoise(p * freq), offset);
        sum += n * amp * prev;
        prev = n;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum;
}

vec3 random3(vec3 p) {
    return fract(sin(vec3(dot(p, vec3(127.1, 311.7, 187.3)), dot(p, vec3(269.5, 183.3, 173.4)), dot(p, vec3(169.5, 83.3, 265.4)))) * 43758.5453);
}

float getWorleyNoise(vec3 uvw, float cell_count) {
    vec3 st = uvw;
    // Scale
    st *= cell_count;

    // Tile the space
    vec3 i_st = floor(st);
    vec3 f_st = fract(st);

    float m_dist = 1.;  // minimum distance

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                // Neighbor place in the grid
                vec3 neighbor = vec3(float(x), float(y), float(z));

                // Random position from current + neighbor place in the grid
                vec3 point = random3(mod(i_st + neighbor, cell_count));

                // Animate the point
                //point = 0.5 + 0.5*sin(6.2831*point);

                // Vector between the pixel and the point
                vec3 diff = neighbor + point - f_st;

                // Distance to the point
                float dist = length(diff);

                // Keep the closer distance
                m_dist = min(m_dist, dist);
            }
        }
    }

    // Draw the min distance (distance field)
    return m_dist;
}


float random(vec3 p3, float mod_val) {
    p3 = mod(p3, vec3(mod_val));
    return fract(sin(dot(p3.xyz,
        vec3(12.9898, 78.233, 31.2134))) *
        43758.5453123);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise(vec3 st, float mod_val) {
    vec3 i = floor(st);
    vec3 f = fract(st);

    // Four corners in 2D of a tile
    float a000 = random(i + vec3(0.0, 0.0, 0.0), mod_val);
    float a001 = random(i + vec3(1.0, 0.0, 0.0), mod_val);
    float a010 = random(i + vec3(0.0, 1.0, 0.0), mod_val);
    float a011 = random(i + vec3(1.0, 1.0, 0.0), mod_val);
    float a100 = random(i + vec3(0.0, 0.0, 1.0), mod_val);
    float a101 = random(i + vec3(1.0, 0.0, 1.0), mod_val);
    float a110 = random(i + vec3(0.0, 1.0, 1.0), mod_val);
    float a111 = random(i + vec3(1.0, 1.0, 1.0), mod_val);

    vec3 f2 = f * f;
    vec3 u = f2 * f * (f2 * 6.0 - f * 15.0 + 10.0);

    return mix(mix(mix(a000, a001, u.x), mix(a010, a011, u.x), u.y), mix(mix(a100, a101, u.x), mix(a110, a111, u.x), u.y), u.z);
}

#define OCTAVES 6
float fbm(vec3 st, float mod_val) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st, mod_val);
        st *= 2.;
        mod_val *= 2;
        amplitude *= .5;
    }
    return value;
}