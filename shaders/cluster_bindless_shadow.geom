#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// ─── cluster_bindless_shadow.geom ───────────────────────────────────
// Depth-only geometry shader for the cluster bindless CSM-shadow path.
//
// Reads the CSM per-cascade light view-proj matrices from the runtime-
// lights UBO (same source the existing base_depthonly_csm.geom uses)
// and broadcasts each input triangle once per cascade, writing to the
// matching depth-array layer via gl_Layer.
//
// One triangle in → CSM_CASCADE_COUNT triangles out (one per array
// layer).  No fragment shader; the pipeline writes depth only.
// ─────────────────────────────────────────────────────────────────────

layout(std140, set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUBO {
    RuntimeLightsParams lights_params;
};

layout(triangles)                                            in;
layout(triangle_strip, max_vertices = CSM_CASCADE_COUNT * 3) out;

layout(location = 0) in vec3 v_world_pos[];

void main() {
    for (int cascade = 0; cascade < CSM_CASCADE_COUNT; ++cascade) {
        gl_Layer = cascade;
        for (int v = 0; v < 3; ++v) {
            gl_Position = lights_params.light_view_proj[cascade]
                        * vec4(v_world_pos[v], 1.0);
            EmitVertex();
        }
        EndPrimitive();
    }
}
