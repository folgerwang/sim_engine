#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// Reads all CSM VP matrices from the runtime lights UBO (already bound
// by the shadow pass).  No per-cascade descriptor set needed.
layout(std140, set = RUNTIME_LIGHTS_PARAMS_SET, binding = RUNTIME_LIGHTS_CONSTANT_INDEX)
    uniform RuntimeLightsUBO {
    RuntimeLightsParams lights_params;
};

// One triangle in → CSM_CASCADE_COUNT triangles out (one per array layer).
layout(triangles)                                      in;
layout(triangle_strip, max_vertices = CSM_CASCADE_COUNT * 3) out;

layout(location = 0) in  ObjectVsPsData in_data[];
layout(location = 0) out ObjectVsPsData out_data;

void main() {
    for (int cascade = 0; cascade < CSM_CASCADE_COUNT; ++cascade) {
        gl_Layer = cascade;
        for (int v = 0; v < 3; ++v) {
            // Use world-space position stored by the vertex shader and apply
            // the per-cascade light-space VP matrix here instead of in the VS.
            gl_Position = lights_params.light_view_proj[cascade]
                          * vec4(in_data[v].vertex_position, 1.0);
            out_data = in_data[v];
            EmitVertex();
        }
        EndPrimitive();
    }
}
