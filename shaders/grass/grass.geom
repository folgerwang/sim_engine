#version 450 core
#include "..\global_definition.glsl.h"
#include "grass_common.glsl.h"

// Fallback expander for USE_MESH_SHADER == 0.  Emits exactly what
// grass.mesh emits: one 16-vertex / 14-triangle strip per blade, same
// varying block, same geometry helper.
layout (points) in;
layout (triangle_strip, max_vertices = 16) out;

layout(location = 0) in GrassSeed {
    vec4 root_dry;
    vec4 h_blade;
    vec4 arc;
} in_seed[];

layout(location = 0) out GrassVsPsData {
    vec4 pos_ws_v;
    vec4 nrm_hash;
    vec4 attribs;
} out_data;

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

void main() {
    float dry = in_seed[0].root_dry.w;
    GrassBlade blade =
        grassMakeBlade(in_seed[0].root_dry.xz, in_seed[0].h_blade, dry);
    blade.root_ws = in_seed[0].root_dry.xyz;
    blade.arc     = in_seed[0].arc.xyz;
    // Waterline cull factor from the vertex stage (arc.w) — this stage
    // has no soil-water binding to compute it itself.
    blade.height *= in_seed[0].arc.w;
    blade.width  *= in_seed[0].arc.w;

    // View-distance fade + widen — same as grass.mesh (see
    // grass_common.glsl.h; 3D distance, because projected size is what
    // matters and altitude is most of it from an aerial camera).
    {
        float cam_dist = distance(camera_info.position, blade.root_ws);
        blade.width *= 1.0f + (kGrassWidenMax - 1.0f) *
            smoothstep(kGrassWidenStartM, kGrassViewFadeStartM, cam_dist);
        float k = 1.0f - smoothstep(kGrassViewFadeStartM,
                                    kGrassViewFadeEndM, cam_dist);
        blade.height *= k;
        blade.width  *= k;
    }

    for (int i_ring = 0; i_ring < kGrassRings; i_ring++) {
        for (int i_side = 0; i_side < 2; i_side++) {
            float side_sign = (i_side == 0) ? 1.0f : -1.0f;
            vec3  p_ws; vec3 n_ws; float v;
            grassBladeVertex(blade, i_ring, side_sign, p_ws, n_ws, v);

            gl_Position = camera_info.view_proj * vec4(p_ws, 1.0f);
            out_data.pos_ws_v = vec4(p_ws, v);
            out_data.nrm_hash = vec4(n_ws, blade.hash);
            out_data.attribs  =
                vec4(dry, side_sign * kGrassProfile[i_ring].x,
                     blade.height, 0.0f);
            EmitVertex();
        }
    }
    EndPrimitive();
}
