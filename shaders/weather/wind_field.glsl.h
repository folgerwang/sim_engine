#ifndef WIND_FIELD_GLSL_H_
#define WIND_FIELD_GLSL_H_

// ── Two-tier wind sampling — ONE definition, every consumer ──────────
// Water waves, grass, clutter, trees and anything added later all read
// the wind through windAt() below, so they agree by construction: a
// gust that bends the grass is the same gust that chops the river, and
// nobody re-derives the LOD blend or the airflow packing on their own.
//
//   TIER 0  COARSE, whole map.  The airflow field weather_system
//     advances with airflow_update.comp: an rgba8 image3D spanning
//     world_min .. world_min + world_range, packed as
//     .xyz = normalize(dir) * 0.5 + 0.5, .w = length * kAirflowStrength-
//     NormalizeScale.  Kilometre-scale weather; far too coarse for a
//     blade of grass, and the ONLY thing that exists off in the
//     distance.
//
//   TIER 1  FINE, camera-following.  A 512-cell lattice at 1 m — a
//     512 m patch centred on the viewer, published as a region vec4
//     (origin.xz, cell_m, grid_size) exactly like LbmWater does.  This
//     is where gust fronts, terrain deflection and wakes live.
//
// The tiers are BLENDED, not switched: the fine patch fades out over
// its outer margin so a consumer crossing the boundary sees the wind
// vector rotate smoothly into the coarse field instead of snapping.
// Switching would put a visible seam 256 m from the camera that tracks
// the player around the world, which is the one artefact a patch-based
// LOD absolutely cannot have.

// Width of the fade at the fine patch's rim, as a fraction of the
// patch.  0.12 leaves ~450 m of full-strength fine wind inside a 512 m
// patch and spends the outer 30 m blending.
const float kWindPatchFeather = 0.12;

// The COARSE-tier helpers below need the airflow packing from
// weather_common.glsl.h (getPackedVectorLength).  Consumers that only
// ride the fine patch — the water waves, the grass sway — include this
// header WITHOUT weather_common and get just the fine-tier functions;
// anything wanting windAt()/sampleWindCoarse defines WIND_FIELD_COARSE
// after including weather_common.
#ifdef WIND_FIELD_COARSE
// Decode one airflow texel into a world-space velocity (m/s).
vec3 unpackAirflow(vec4 texel) {
    vec3 dir = texel.xyz * 2.0f - 1.0f;
    float len2 = dot(dir, dir);
    if (len2 < 1e-8f) {
        return vec3(0.0f);
    }
    return (dir * inversesqrt(len2)) * getPackedVectorLength(texel.w);
}

// TIER 0.  `p_ws` is world space; the field's z axis is altitude.
vec3 sampleWindCoarse(sampler3D airflow_tex,
                      vec3 world_min, vec3 world_range,
                      vec3 p_ws) {
    vec3 uvw = (p_ws - world_min) / max(world_range, vec3(1e-3f));
    // Altitude clamps rather than wraps: below ground and above the top
    // of the column both want the nearest real slice, not a fold.
    uvw = clamp(uvw, vec3(0.0f), vec3(1.0f));
    return unpackAirflow(texture(airflow_tex, uvw));
}

#endif  // WIND_FIELD_COARSE

// TIER 1.  `region` is (origin.x, origin.z, cell_m, grid_size) — the
// same layout LbmWater publishes.  Returns the patch velocity and, in
// out_weight, how much authority it has here (0 outside, 1 well
// inside).  The caller does not need to bounds-check.
vec2 sampleWindFine(sampler2D wind_tex, vec4 region, vec2 p_xz,
                    out float out_weight) {
    out_weight = 0.0f;
    float span = region.z * region.w;          // cell_m * grid_size
    if (span <= 1.0f) {
        return vec2(0.0f);                     // patch not live yet
    }
    vec2 uv = (p_xz - region.xy) / span;
    if (any(lessThanEqual(uv, vec2(0.0f))) ||
        any(greaterThanEqual(uv, vec2(1.0f)))) {
        return vec2(0.0f);
    }
    vec2 ef = smoothstep(0.0f, kWindPatchFeather, uv) *
              (1.0f - smoothstep(1.0f - kWindPatchFeather, 1.0f, uv));
    out_weight = ef.x * ef.y;
    return texture(wind_tex, uv).xy;
}

#ifdef WIND_FIELD_COARSE
// THE call every consumer should use.  World position in, wind
// velocity in m/s out (world xz in .xz, vertical in .y), fine patch
// blended over coarse field.
vec3 windAt(sampler3D airflow_tex, vec3 world_min, vec3 world_range,
            sampler2D wind_tex, vec4 region,
            vec3 p_ws) {
    vec3 coarse = sampleWindCoarse(airflow_tex, world_min, world_range, p_ws);
    float w = 0.0f;
    vec2 fine = sampleWindFine(wind_tex, region, p_ws.xz, w);
    // Only the horizontal components blend — the fine lattice is 2D, so
    // where it has authority the vertical term stays the coarse field's
    // rather than being zeroed, which would make updrafts vanish inside
    // the patch and reappear at its rim.
    return vec3(mix(coarse.x, fine.x, w),
                coarse.y,
                mix(coarse.z, fine.y, w));
}

#endif  // WIND_FIELD_COARSE

// Convenience for consumers that only want a direction + speed, which
// is most of them (a grass blade bends along the wind, it does not
// integrate it).
void windDirSpeed(vec3 wind, out vec2 out_dir, out float out_speed) {
    vec2 flat_w = wind.xz;
    out_speed = length(flat_w);
    out_dir = (out_speed > 1e-4f) ? flat_w / out_speed : vec2(1.0f, 0.0f);
}

#endif  // WIND_FIELD_GLSL_H_
