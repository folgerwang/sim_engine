#ifndef UNDERWATER_GLSL_H
#define UNDERWATER_GLSL_H

// ─────────────────────────────────────────────────────────────────────
// Underwater -- the eye below the water surface.
//
// The application samples the hydrology water-level map at the camera
// every frame and writes ViewCameraInfo::underwater_depth (metres of
// water above the eye, <= 0 when the eye is in air) and
// ViewCameraInfo::water_level_y (the surface height there).  Two
// consumers, both including this header AFTER ibl.glsl.h (it reads the
// sky cubes):
//
//   deferred_resolve.comp  every opaque pixel: dimmed by the light that
//                          reaches its own depth, then fogged along the
//                          part of its view ray that lies in the water;
//   tile_water.frag        the surface seen from BELOW -- Snell's
//                          window of refracted sky/shore inside the
//                          critical angle, the reflected underwater
//                          gloom outside it -- then fogged like
//                          everything else.
//
// One medium model for both: Beer-Lambert extinction per metre of
// clear lake water, red first, so distance pulls everything toward
// blue-green; the in-scattered light is that same water lit by the
// sky, dimmed by how deep the eye is.  Linear radiance throughout, in
// the units the IBL cubes deliver (kIblIrradianceScale applied).
// ─────────────────────────────────────────────────────────────────────

const vec3  kUwExtinction  = vec3(0.30, 0.13, 0.085); // 1/m along the view
const vec3  kUwSunExtinct  = vec3(0.42, 0.18, 0.11);  // 1/m of depth, light going down
const vec3  kUwScatterTint = vec3(0.06, 0.24, 0.32);  // in-scatter albedo of the water
const float kUwIor         = 1.33;

// Sky radiance along `dir` (the lightly prefiltered radiance cube).
vec3 uwSkyRadiance(vec3 dir) {
    vec3 s = textureLod(ggx_env_sampler, dir, 1.0).rgb;
#ifndef USE_HDR
    s = sRGBToLinear(s);
#endif
    return s * kIblIrradianceScale;
}

// Sky irradiance on an upward-facing surface -- what lights the water.
vec3 uwSkyIrradianceUp() {
    vec3 s = texture(lambertian_env_sampler, vec3(0.0, 1.0, 0.0)).rgb;
#ifndef USE_HDR
    s = sRGBToLinear(s);
#endif
    return s * kIblIrradianceScale;
}

// The colour of the water itself as seen from `eye_depth` metres down:
// the sky's light, scattered by the water, having crossed that much of
// it on the way down.
vec3 uwInscatter(float eye_depth) {
    return uwSkyIrradianceUp() * kUwScatterTint *
           exp(-kUwSunExtinct * max(eye_depth, 0.0));
}

// Length of the view ray (eye -> eye + dir * hit_dist) that lies in the
// water, the eye being below the surface at height `level_y`.  A ray
// heading up leaves the water where it crosses the surface.
float uwPathLength(vec3 eye, vec3 dir, float hit_dist, float level_y) {
    if (dir.y > 1e-5) {
        float t_exit = (level_y - eye.y) / dir.y;
        return min(hit_dist, max(t_exit, 0.0));
    }
    return hit_dist;
}

// `color` seen through `path_m` metres of water.
vec3 uwFog(vec3 color, float path_m, vec3 inscatter) {
    vec3 T = exp(-kUwExtinction * max(path_m, 0.0));
    return color * T + inscatter * (1.0 - T);
}

// Light reaching a surface `depth_m` below the water surface.
vec3 uwDepthDimming(float depth_m) {
    return exp(-kUwSunExtinct * max(depth_m, 0.0));
}

#endif // UNDERWATER_GLSL_H
