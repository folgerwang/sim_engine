#ifndef IBL_GLSL_H
#define IBL_GLSL_H

#include "functions.glsl.h"
#include "punctual.glsl.h"

layout(set = PBR_GLOBAL_PARAMS_SET, binding = GGX_LUT_INDEX) uniform sampler2D ggx_lut;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = GGX_ENV_TEX_INDEX) uniform samplerCube ggx_env_sampler;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = LAMBERTIAN_ENV_TEX_INDEX) uniform samplerCube lambertian_env_sampler;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = CHARLIE_LUT_INDEX) uniform sampler2D charlie_lut;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = CHARLIE_ENV_TEX_INDEX) uniform samplerCube charlie_env_sampler;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = DIRECT_SHADOW_INDEX) uniform sampler2DArray direct_shadow_sampler;
// Post-terrain scene depth.  Declared unconditionally so every consumer
// of this header agrees with the single set-0 layout; only base.frag's
// DECAL permutation actually fetches from it.
layout(set = PBR_GLOBAL_PARAMS_SET, binding = SCENE_DEPTH_TEX_INDEX) uniform sampler2D scene_depth_sampler;

// ── Sky-IBL radiance scale ───────────────────────────────────────────
// The environment cubes are convolved from the atmospheric scattering
// LUT, which is evaluated with a sun intensity of 22.0 (see
// cube_skybox.frag / cube_skybox_mini.comp).  That number belongs to the
// SKY SIMULATION; nothing ever reconciled it with the units the
// directional light and the shared ACES tonemapper use, so the ambient
// term arrived roughly an order of magnitude too hot and drowned the
// sun.
//
// Solved from a measurement rather than guessed.  With the sun already
// at 6.0 the frame STILL tonemapped to ~0.94 sRGB on terrain whose
// albedo is linear 0.515 (sampled from new-world_color.png at the
// camera; that neighbourhood's maximum luma is 208/255, so nothing in
// it is remotely white):
//
//   observed 0.94 sRGB          -> 2.52 linear
//   direct at sun 6.0           =  0.69 linear
//   => ambient                  =  1.83 linear
//   => lambertian_env(N)        =  3.55
//
// which puts sky : sun irradiance at 2.7 : 1.  A clear day with the hard
// tree shadows this scene renders is nearer 1 : 3 — the ambient was
// close to TEN TIMES the sun.  That is why every surface came out pale
// and desaturated instead of shadowed and saturated: a large uniform
// white term washes colour out of everything it is added to.
//
// Targeting shadowed ground at ~0.40 sRGB and sunlit at ~0.80 gives
// lambertian_env(N) = 0.349 and sun = 5.85 — the sun is already right,
// so the whole correction lands here: 0.349 / 3.55 = 0.098.
//
// Why this ALSO explains the load-time change: cube_ibl_mini.comp
// accumulates the irradiance cube across frames with a temporal EMA.
// At 8 fps during streaming it converges slowly, so early frames sample
// an UNDER-converged (dark) cube — which is exactly the tan sand, deep
// blue water and saturated greens that looked correct.  Nothing about
// loading changed the lighting; the cube simply had not finished
// climbing to its (far too bright) steady state yet.
// ── 2026-08 REVISION: 0.098 -> 0.28 ─────────────────────────────────
// The derivation above is still correct about the UNITS.  What it got
// wrong is what it solved FOR: it measured one sunlit sand pixel and
// picked the scale that put that pixel at ~0.80 sRGB.  Nothing in it
// looked at the SHADOW, and the shadow is the only thing this constant
// actually governs — a shadowed pixel has shad = 0, so the sky term is
// its ENTIRE light budget, while a sunlit pixel is dominated by the
// direct term and barely moves when this changes.
//
// Measured off a forest frame (grass albedo ~0.15, sun 6.0, hard tree
// shadows on flat ground), inverting the shared ACES curve back to
// radiance:
//
//   sunlit grass   0.470 sRGB  ->  0.2340 radiance
//   tree shadow    0.080 sRGB  ->  0.0175 radiance
//   ratio                                  0.075
//
// A clear-sky day puts diffuse sky at ~13-15% of total horizontal
// illuminance, so 7.5% is roughly half the light a real shadow gets,
// and the ACES toe then spends that deficit twice: at x = 0.011 the
// curve's local gain is ~0.39 where at x = 0.13 it is ~1.39, so a
// factor of two in radiance becomes far more than a factor of two on
// screen.  That is why it reads as black rather than as dim.
//
// Solving the same measurement for the shadow instead of the highlight:
//
//   kIbl    shadow   sunlit   shadow/sunlit
//   0.098    0.080    0.470       17%      <- was here
//   0.200    0.129    0.491       26%
//   0.250    0.151    0.500       30%
//   0.280    0.164    0.506       32%      <- here now
//   0.350    0.193    0.519       37%
//
// Note how little the sunlit column moves: the direct term is untouched
// and it already sits under the ACES shoulder, so this buys shadow
// detail almost entirely out of contrast rather than out of highlights.
// The pale washed-out frame the original note describes came from the
// ambient being ~10x the SUN; at 0.28 the sky:sun ratio is still about
// 1:3, which is the clear-day figure that note itself quotes as the
// target.
//
// WHAT THIS COSTS, and it is real: the lambertian cube delivers full
// open-sky irradiance to every pixel regardless of whether that pixel
// can see the sky, and cluster geometry writes ao = 1.0 unconditionally
// — so indoors there is nothing but screen-space AO holding the ambient
// down, and interiors WILL get brighter by the same 2.9x.  Walk into a
// house after changing this.  The durable answer is a real sky
// visibility factor (RT GI's traced_sky_vis, or a baked sky-occlusion
// channel in the G-buffer); until one exists this constant is a single
// compromise serving both the open air and the indoors, and it is now
// biased toward the open air, which is where the camera spends its time.
const float kIblIrradianceScale = 0.28;

vec3 getIBLRadianceGGX(vec3 n, vec3 v, float perceptualRoughness, vec3 specularColor, float mip_count)
{
    float n_dot_v = clampedDot(n, v);
    float lod = clamp(perceptualRoughness * float(mip_count), 0.0, float(mip_count));
    vec3 reflection = normalize(reflect(-v, n));

    vec2 brdfSamplePoint = clamp(vec2(n_dot_v, perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 brdf = texture(ggx_lut, brdfSamplePoint).rg;
    vec4 specularSample = textureLod(ggx_env_sampler, reflection, lod);

    vec3 specularLight = specularSample.rgb;

#ifndef USE_HDR
    specularLight = sRGBToLinear(specularLight);
#endif

    return specularLight * kIblIrradianceScale *
           (specularColor * brdf.x + brdf.y);
}

vec3 getIBLRadianceTransmission(vec3 n, vec3 v, float perceptualRoughness, float ior, vec3 baseColor, float mip_count)
{
    // Sample GGX LUT.
    float NdotV = clampedDot(n, v);
    vec2 brdfSamplePoint = clamp(vec2(NdotV, perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 brdf = texture(ggx_lut, brdfSamplePoint).rg;

    // Sample GGX environment map.
    float lod = clamp(perceptualRoughness * float(mip_count), 0.0, float(mip_count));

    // Approximate double refraction by assuming a solid sphere beneath the point.
    vec3 r = refract(-v, n, 1.0 / ior);
    vec3 m = 2.0 * dot(-n, r) * r + n;
    vec3 rr = -refract(-r, m, ior);

    vec4 specularSample = textureLod(ggx_env_sampler, rr, lod);
    vec3 specularLight = specularSample.rgb;

#ifndef USE_HDR
    specularLight = sRGBToLinear(specularLight);
#endif

   return specularLight * kIblIrradianceScale * (brdf.x + brdf.y);
}

vec3 getIBLRadianceLambertian(vec3 n, vec3 diffuseColor)
{
    vec3 diffuseLight = texture(lambertian_env_sampler, n).rgb;

    #ifndef USE_HDR
        diffuseLight = sRGBToLinear(diffuseLight);
    #endif

    // See kIblIrradianceScale: the cubes are convolved from a sky LUT
    // whose units were never reconciled with the sun or the tonemapper.
    return diffuseLight * kIblIrradianceScale * diffuseColor;
}

vec3 getIBLRadianceCharlie(vec3 n, vec3 v, float sheenRoughness, vec3 sheenColor, float sheenIntensity, float mip_count)
{
    float NdotV = clampedDot(n, v);
    float lod = clamp(sheenRoughness * float(mip_count), 0.0, float(mip_count));
    vec3 reflection = normalize(reflect(-v, n));

    vec2 brdfSamplePoint = clamp(vec2(NdotV, sheenRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    float brdf = texture(charlie_lut, brdfSamplePoint).b;
    vec4 sheenSample = textureLod(charlie_env_sampler, reflection, lod);

    vec3 sheenLight = sheenSample.rgb;

    #ifndef USE_HDR
    sheenLight = sRGBToLinear(sheenLight);
    #endif

    return sheenIntensity * sheenLight * kIblIrradianceScale *
           sheenColor * brdf;
}

vec3 getIBLRadianceSubsurface(vec3 n, vec3 v, float scale, float distortion, float power, vec3 color, float thickness)
{
    vec3 diffuseLight = texture(lambertian_env_sampler, n).rgb;

    #ifndef USE_HDR
        diffuseLight = sRGBToLinear(diffuseLight);
    #endif

    return diffuseLight * kIblIrradianceScale *
           getPunctualRadianceSubsurface(n, v, -v, scale, distortion, power, color, thickness);
}

#endif // IBL_GLSL_H
