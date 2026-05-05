#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ─── probe_debug.frag ───────────────────────────────────────────────────
// Renders each probe as a small sphere whose surface colour reflects
// its SH-evaluated irradiance in the surface-normal direction.  Probes
// not yet baked (position.w <= 0) are drawn solid red so it's
// immediately obvious which probes are still pending their first SH
// projection.

#define AMBIENT_PROBE_SET     0
#define AMBIENT_PROBE_BINDING 0
#include "ambient_probes.glsl.h"

layout(location = 0) in       vec3 v_normal;
layout(location = 1) flat in  uint v_probe_idx;
layout(location = 0) out      vec4 out_color;

void main() {
    float ready = ambient_probes[v_probe_idx].position.w;
    if (ready <= 0.0) {
        // Not yet baked — bright red marker.
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    // SH coefficients store cosine-convolved irradiance (sh_project.comp
    // bakes the per-band Â_l constants in).  Evaluating SH at a unit
    // direction therefore returns the irradiance E(N) at that direction,
    // which for a bright daylit envmap is on the order of π·L ≈ 3 — way
    // above the [0, 1] LDR range, so it'd just clip to white if we wrote
    // it raw.  Apply a Reinhard-style tone map plus a 1/π exposure scale
    // so a uniform white sky maps to mid-grey (~0.5) on the sphere
    // surface, and dim corners stay legible.
    vec3 E = evaluateAmbientProbeSH(
        ambient_probes[v_probe_idx].sh,
        normalize(v_normal));

    const float kInvPi = 0.31830988618; // 1/π
    vec3 radiance = E * kInvPi;            // irradiance → radiance
    vec3 mapped   = radiance / (1.0 + radiance);  // Reinhard
    out_color = vec4(mapped, 1.0);
}
