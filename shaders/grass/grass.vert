#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"
#include "..\terrain\tile_common.glsl.h"
#include "..\noise.glsl.h"
#include "..\weather\wind_field.glsl.h"
#include "grass_common.glsl.h"

// ── Pre-mesh-shader fallback path (USE_MESH_SHADER == 0) ─────────────
// Kept in lockstep with grass.mesh through grass_common.glsl.h: this
// stage resolves ONE blade's root, hashes, dryness and wind, and
// grass.geom expands it into the same 16-vertex ribbon the mesh shader
// emits, with the same varyings.  Nothing about the LOOK lives here.

layout(location = VINPUT_POSITION) in vec3 in_position;

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;
layout(set = TILE_PARAMS_SET, binding = SOIL_WATER_LAYER_BUFFER_INDEX) uniform sampler2D soil_water_layer;
layout(set = TILE_PARAMS_SET, binding = WIND_TEX_INDEX) uniform sampler2D wind_patch_tex;
layout(std430, set = TILE_PARAMS_SET, binding = WIND_REGION_BUFFER_INDEX) readonly buffer WindRegionBuf {
    vec4 wind_region;
};

layout(location = IINPUT_MAT_ROT_0) in vec3 in_loc_rot_mat_0;
layout(location = IINPUT_MAT_ROT_1) in vec3 in_loc_rot_mat_1;
layout(location = IINPUT_MAT_ROT_2) in vec3 in_loc_rot_mat_2;
layout(location = IINPUT_MAT_POS_SCALE) in vec4 in_loc_pos_scale;

layout(location = 0) out GrassSeed {
    vec4 root_dry;   // xyz = root world position, w = dryness
    vec4 h_blade;    // the per-blade hash draw
    vec4 arc;        // xyz = total horizontal tip travel (lean + wind)
} out_seed;

void main() {
    uint blade_idx = uint(gl_InstanceIndex);
    uint tuft_idx  = uint(float(blade_idx) * (1.0f / kGrassTuftBlades));

    vec4 h_tuft  = clamp(hash43(vec3(tile_params.min, float(tuft_idx))),
                         0.0f, 1.0f);
    vec4 h_blade = clamp(hash43(vec3(tile_params.min + vec2(17.13f, 41.77f),
                                     float(blade_idx))), 0.0f, 1.0f);

    vec2 root_xz = grassRootXZ(tile_params.min, tile_params.range,
                               h_tuft, h_blade);

    vec2 world_map_uv = (root_xz - tile_params.world_min) *
                        tile_params.inv_world_range;
    float ground_height = texture(rock_layer, world_map_uv).x;
    vec2  soil_water_thickness =
        texture(soil_water_layer, world_map_uv).xy * SOIL_WATER_LAYER_MAX_THICKNESS;
    ground_height += soil_water_thickness.x;

    float dry = grassDryField(root_xz);
    GrassBlade blade = grassMakeBlade(root_xz, h_blade, dry);

    {
        float wf_w;
        vec2  wv = sampleWindFine(wind_patch_tex, wind_region, root_xz, wf_w);
        float spd = length(wv);
        if (wf_w * spd > 1e-3f) {
            vec2  dir  = wv / max(spd, 1e-4f);
            float lean = min(spd * 0.055f, 0.40f) * blade.height;
            float flutter = 0.05f * min(spd, 8.0f) * blade.height *
                sin(tile_params.time * (3.0f + 2.0f * h_blade.w) +
                    h_blade.z * 6.2831853f);
            blade.arc += vec3(dir.x, 0.0f, dir.y) * ((lean + flutter) * wf_w);
        }
    }

    out_seed.root_dry = vec4(root_xz.x, ground_height - 0.01f, root_xz.y, dry);
    out_seed.h_blade  = h_blade;
    out_seed.arc      = vec4(blade.arc, 0.0f);
}
