// ─── ambient_probes.glsl.h ─────────────────────────────────────────────
// Shader-side helpers for sampling the AmbientProbeSystem's SH probe
// grid and evaluating the diffuse irradiance at a world-space surface
// point.  Include from any fragment shader that wants probe-based
// ambient lighting.
//
// REQUIRES from the includer:
//   • A pre-defined macro `AMBIENT_PROBE_SET` giving the descriptor set
//     index where the probe SSBO + grid metadata are bound.
//   • A pre-defined macro `AMBIENT_PROBE_BINDING` giving the storage-
//     buffer binding within that set (default 0 if not specified).
//
// Usage:
//   #define AMBIENT_PROBE_SET 5
//   #include "ambient_probes.glsl.h"
//   ...
//   vec3 ambient = sampleAmbientProbeGrid(world_pos, normal);
// ─────────────────────────────────────────────────────────────────────

#ifndef AMBIENT_PROBES_GLSL_H_
#define AMBIENT_PROBES_GLSL_H_

#ifndef AMBIENT_PROBE_BINDING
#define AMBIENT_PROBE_BINDING 0
#endif

struct AmbientProbeData {
    vec4 position;       // .xyz = world pos, .w = "ready" flag (>0)
    vec4 sh[9];          // cosine-convolved 2nd-order SH RGB coefficients
};

layout(std430, set = AMBIENT_PROBE_SET, binding = AMBIENT_PROBE_BINDING)
    readonly buffer AmbientProbeBuffer {
    AmbientProbeData ambient_probes[];
};

// Grid metadata — match the layout the application uploads (we keep
// this as a uniform struct so the application can ship grid_dims and
// grid_min/max via a small UBO if desired; for now they're carried in
// fragment-shader push constants or supplied as compile-time
// fallbacks).  See base.frag / cluster_bindless.frag wiring.

#ifndef AMBIENT_PROBE_GRID_X
#define AMBIENT_PROBE_GRID_X 4
#endif
#ifndef AMBIENT_PROBE_GRID_Y
#define AMBIENT_PROBE_GRID_Y 4
#endif
#ifndef AMBIENT_PROBE_GRID_Z
#define AMBIENT_PROBE_GRID_Z 4
#endif

// ─── Evaluate 2nd-order SH at a unit direction ─────────────────────────
// Coefficients arrive cosine-convolved (sh_project.comp baked the
// per-band Â_l factors in), so this is just a direct dot product
// against the SH basis at `n`.  Standard formula from Ramamoorthi &
// Hanrahan 2001 "An Efficient Representation for Irradiance
// Environment Maps".
vec3 evaluateAmbientProbeSH(vec4 sh[9], vec3 n) {
    const float k0  = 0.282094791773878;
    const float k1  = 0.488602511902919;
    const float k2  = 1.092548430592079;
    const float k20 = 0.315391565252520;
    const float k22 = 0.546274215296039;

    float Y[9];
    Y[0] =  k0;
    Y[1] = -k1  * n.y;
    Y[2] =  k1  * n.z;
    Y[3] = -k1  * n.x;
    Y[4] =  k2  * n.x * n.y;
    Y[5] = -k2  * n.y * n.z;
    Y[6] =  k20 * (3.0 * n.z * n.z - 1.0);
    Y[7] = -k2  * n.x * n.z;
    Y[8] =  0.5 * k22 * (n.x * n.x - n.y * n.y);

    vec3 acc = vec3(0.0);
    for (int i = 0; i < 9; ++i) {
        acc += sh[i].rgb * Y[i];
    }
    return max(acc, vec3(0.0));
}

// ─── Trilinear-weighted 8-probe lookup ─────────────────────────────────
// Given a world-space surface point and normal, finds the surrounding
// 2×2×2 grid of probes and trilinearly interpolates their SH-evaluated
// irradiance.  Probes whose `ready` flag is zero are skipped (their
// weight is redistributed across the remaining ready neighbours), so
// the lighting is correct even before every probe has been baked.
vec3 sampleAmbientProbeGrid(
    vec3 world_pos, vec3 normal,
    vec3 grid_min, vec3 grid_max) {
    vec3 grid_size = grid_max - grid_min;
    if (any(lessThanEqual(grid_size, vec3(0.0)))) return vec3(0.0);

    // Convert world_pos → fractional grid coords ∈ [0, dims-1].
    // Inset of 1/(2N) on each side: probe at i sits at (i + 0.5)/N.
    // Inverting that: u = world_t * N - 0.5 where world_t ∈ [0, 1].
    vec3 dims = vec3(
        float(AMBIENT_PROBE_GRID_X),
        float(AMBIENT_PROBE_GRID_Y),
        float(AMBIENT_PROBE_GRID_Z));
    vec3 t = (world_pos - grid_min) / grid_size;
    vec3 u = clamp(t * dims - 0.5, vec3(0.0), dims - vec3(1.0001));

    ivec3 i0 = ivec3(floor(u));
    ivec3 i1 = min(i0 + ivec3(1),
        ivec3(AMBIENT_PROBE_GRID_X, AMBIENT_PROBE_GRID_Y,
              AMBIENT_PROBE_GRID_Z) - ivec3(1));
    vec3 f  = u - vec3(i0);

    vec3 acc          = vec3(0.0);
    float weight_sum  = 0.0;

    // Loop over 8 corners of the surrounding cube.
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                ivec3 ic = ivec3(
                    (dx == 0) ? i0.x : i1.x,
                    (dy == 0) ? i0.y : i1.y,
                    (dz == 0) ? i0.z : i1.z);
                int idx =
                    (ic.z * AMBIENT_PROBE_GRID_Y + ic.y) *
                        AMBIENT_PROBE_GRID_X + ic.x;
                if (ambient_probes[idx].position.w <= 0.0) continue;

                float wx = (dx == 0) ? (1.0 - f.x) : f.x;
                float wy = (dy == 0) ? (1.0 - f.y) : f.y;
                float wz = (dz == 0) ? (1.0 - f.z) : f.z;
                float w  = wx * wy * wz;

                acc        += w * evaluateAmbientProbeSH(
                    ambient_probes[idx].sh, normal);
                weight_sum += w;
            }
        }
    }

    if (weight_sum < 1e-4) return vec3(0.0);
    return acc / weight_sum;
}

#endif // AMBIENT_PROBES_GLSL_H_
