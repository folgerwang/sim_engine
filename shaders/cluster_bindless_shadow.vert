#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ─── cluster_bindless_shadow.vert ───────────────────────────────────
// Depth-only vertex shader for the cluster bindless CSM-shadow path.
//
// This is shape B (the current production setup).  Five shapes were
// measured on the test GPU (desktop NVIDIA, Bistro scene):
//
//   A. Per-cluster vkCmdDrawIndexedIndirectCount + GS broadcast.
//      One indirect cmd per cluster (~326k), GS amplifies each tri
//      to all CSM_CASCADE_COUNT layers.  Cost: ~28 ms.  Front-end
//      command-processor setup × 326k dominates wall time.
//
//   B. Single vkCmdDrawIndexed over the merged VB/IB + GS broadcast
//      (current).  Same triangle work as A, but ONE draw cmd through
//      the front end.  Cost: ~10 ms.
//
//   C. Single drawIndexed + hardware multiview (gl_ViewIndex).
//      On desktop NVIDIA the driver implements multiview as VS
//      replication, running the vertex shader once per view per
//      vertex.  Cost: ~56 ms (6× VS).  Reverted.
//
//   D. Per-cascade rendering (CSM_CASCADE_COUNT separate draws to
//      single-layer views) + per-cascade tight cull (compute writes
//      per-cascade indirect buffers).  Cost: ~26 ms.  Per-cascade
//      cull rejects a lot, but the per-cascade indirect command
//      processing still has front-end overhead (~50k cmds × 6 ≈
//      300k cmds).  Reverted.
//
//   D'. Same as D but with on-GPU index-compaction so each cascade
//       does one drawIndexed over a contiguous compacted index range.
//       Untested — projected target ~3-5 ms if cull rejects enough.
//       Adds a compute scatter pass per cascade and ~6× index-buffer
//       storage.  Significant additional complexity.
//
// Shape B is the current winner on this GPU.  This shader passes
// world-space position through to the geometry shader, which applies
// the per-cascade VP from RuntimeLightsParams.light_view_proj[cascade]
// and emits each input tri to all CSM_CASCADE_COUNT layers in a
// single GS invocation.
//
// Cluster vertices are baked to WORLD space at upload time (see
// uploadMeshClusters in cluster_renderer.cpp), so the VS only needs
// to pass position through to the geometry shader.
// ─────────────────────────────────────────────────────────────────────

layout(location = 0) in vec3 in_position;   // world-space position

// Pass world-space position to the geometry shader.  Per-cascade clip
// projection happens there, not here.
layout(location = 0) out vec3 v_world_pos;

void main() {
    v_world_pos = in_position;
    // gl_Position is overwritten in the GS; the value here only needs
    // to be well-defined (some drivers complain if a VS doesn't write
    // gl_Position even when a GS follows).
    gl_Position = vec4(in_position, 1.0);
}
