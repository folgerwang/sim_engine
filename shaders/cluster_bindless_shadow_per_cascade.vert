#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ─── cluster_bindless_shadow_per_cascade.vert ───────────────────────────
// Depth-only vertex shader for the cluster CSM "Regular" (per-cascade)
// draw mode.  Mirrors the legacy drawable per-cascade path:
//   • Single-layer framebuffer per dispatch (csm_layer_views_[k]).
//   • One vkCmdDrawIndexed per cascade over the full merged VB/IB.
//   • Push constant carries the cascade index; the VS picks the matching
//     light_view_proj from the shared RuntimeLightsParams UBO.
//   • No GS, no mesh shader — the host loops cascades.
//
// Cluster vertices are baked to WORLD space at upload time
// (see uploadMeshClusters in cluster_renderer.cpp), so the VS only
// needs one matrix-vector multiply per vertex.
//
// Cost characteristic vs the GS / mesh-shader paths:
//   • Per-cascade: 6 separate vkCmdDrawIndexed calls + 6 render-pass
//     scopes — highest CPU recording overhead, no per-cluster cull.
//   • GS broadcast: 1 drawIndexed + GS amplifies each tri to 6 layers —
//     dedicated HW path on most desktop GPUs, no per-cluster cull.
//   • Mesh shader: 1 dispatch, task shader culls per (cluster, cascade)
//     pair and only emits surviving mesh workgroups — fewest primitives
//     reach the rasterizer.
// ─────────────────────────────────────────────────────────────────────────

layout(push_constant) uniform ClusterShadowPerCascadePC {
    uint cascade_idx;
} pc;

layout(std140, set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUBO {
    RuntimeLightsParams lights_params;
};

layout(location = 0) in vec3 in_position;   // world-space position

void main() {
    gl_Position =
        lights_params.light_view_proj[pc.cascade_idx] * vec4(in_position, 1.0);
}
