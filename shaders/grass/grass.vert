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
layout(set = TILE_PARAMS_SET, binding = TERRAIN_FLAT_MASK_INDEX) uniform sampler2D terrain_flat_mask;
// The camera is what the 1 m detail fade is measured against.  The
// mesh path already had it (for the view fade); this stage did not,
// because grass.geom owns the view fade on this path -- but the ROOT
// HEIGHT needs it too, and the root is resolved here.
layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};
#include "..\terrain\tile_detail.glsl.h"
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


// ── The RENDERED surface height ─────────────────────────────────────
// Same as grass.mesh: heights are evaluated at the tile's LOD-grid
// vertices (the grid tile.vert actually draws; drawGrass pushes the
// tile's real per-LOD segment count) and bilinearly interpolated, so a
// blade roots on the drawn mesh instead of on height-field features
// the mesh grid cannot carry -- which is what left blades floating in
// mid-air over every crest the coarser grids cut under.
float grassVertexHeight(vec2 pos_ws) {
    vec2 uv = (pos_ws - tile_params.world_min) * tile_params.inv_world_range;
    float h = texture(rock_layer, uv).x;
    vec2 sw = texture(soil_water_layer, uv).xy * SOIL_WATER_LAYER_MAX_THICKNESS;
    float fade = terrainDetailFade(pos_ws, camera_info.position.xyz);
    fade *= 1.0f - clamp(sw.y * 2.0f, 0.0f, 1.0f);
    return terrainDetailHeight(pos_ws, h, fade) + sw.x;
}

// KEEP IN SYNC with grass.mesh, which carries the full rationale: the
// edge stitch must be reproduced (tile_params.offset) and the drawn
// surface is TWO TRIANGLES per quad, not a bilinear patch.
float grassSnapToGrid(float t, float nseg) {
    return floor(t * nseg + 0.5f) / nseg;
}

vec2 grassGridPos(vec2 g, float seg) {
    vec2 f = g * (1.0f / seg);
    uint elods = tile_params.offset;
    if (g.x == 0.0f) {
        float ns = float(elods & 0xFFu);
        if (ns > 0.0f && ns < seg) f.y = grassSnapToGrid(f.y, ns);
    } else if (g.x == seg) {
        float ns = float((elods >> 8) & 0xFFu);
        if (ns > 0.0f && ns < seg) f.y = grassSnapToGrid(f.y, ns);
    }
    if (g.y == 0.0f) {
        float ns = float((elods >> 16) & 0xFFu);
        if (ns > 0.0f && ns < seg) f.x = grassSnapToGrid(f.x, ns);
    } else if (g.y == seg) {
        float ns = float((elods >> 24) & 0xFFu);
        if (ns > 0.0f && ns < seg) f.x = grassSnapToGrid(f.x, ns);
    }
    return tile_params.min + f * tile_params.range;
}

float grassGroundHeight(vec2 root_xz) {
    float seg = float(tile_params.segment_count);
    vec2 grid = clamp((root_xz - tile_params.min) / tile_params.range,
                      0.0f, 1.0f) * seg;
    vec2 g0 = min(floor(grid), vec2(seg - 1.0f));
    vec2 gf = grid - g0;

    float h00 = grassVertexHeight(grassGridPos(g0,                    seg));
    float h10 = grassVertexHeight(grassGridPos(g0 + vec2(1.0f, 0.0f), seg));
    float h01 = grassVertexHeight(grassGridPos(g0 + vec2(0.0f, 1.0f), seg));
    float h11 = grassVertexHeight(grassGridPos(g0 + vec2(1.0f, 1.0f), seg));

    float h;
    if (gf.x + gf.y <= 1.0f) {
        h = h00 + (h10 - h00) * gf.x + (h01 - h00) * gf.y;
    } else {
        h = h11 + (h01 - h11) * (1.0f - gf.x)
                + (h10 - h11) * (1.0f - gf.y);
    }
    vec2 cell = tile_params.range * tile_params.inv_segment_count;
    return h - 0.02f * cell.x;
}

void main() {
    uint blade_idx = uint(gl_InstanceIndex);
    uint tuft_idx  = uint(float(blade_idx) * (1.0f / kGrassTuftBlades));

    vec4 h_tuft  = clamp(hash43(vec3(tile_params.min, float(tuft_idx))),
                         0.0f, 1.0f);
    vec4 h_blade = clamp(hash43(vec3(tile_params.min + vec2(17.13f, 41.77f),
                                     float(blade_idx))), 0.0f, 1.0f);

    // Root placement with waterline rejection — same retry loop as
    // grass.mesh (see there / grass_common.glsl.h for the reasoning).
    vec2  root_xz;
    vec2  world_map_uv;
    vec2  soil_water_thickness;
    float built;
    for (int attempt = 0; ; ++attempt) {
        root_xz = grassRootXZ(tile_params.min, tile_params.range,
                              h_tuft, h_blade);
        world_map_uv = (root_xz - tile_params.world_min) *
                       tile_params.inv_world_range;
        soil_water_thickness =
            texture(soil_water_layer, world_map_uv).xy *
            SOIL_WATER_LAYER_MAX_THICKNESS;
        built = texture(terrain_flat_mask, world_map_uv).x;
        bool rejected = soil_water_thickness.y > kGrassWaterFadeStartM ||
                        built > kGrassBuiltRelocate;
        if (!rejected || attempt >= kGrassWaterRelocates) {
            break;
        }
        h_tuft = clamp(hash43(vec3(
            tile_params.min + vec2(7.31f, -3.77f) * float(attempt + 1),
            float(tuft_idx))), 0.0f, 1.0f);
    }
    // Same rendered-surface height tile.vert's SOIL_PASS produces —
    // base rock mixed toward the streamed 1 m detail relief, with the
    // submerged-bed suppression.  See grass.mesh for why sampling the
    // base map alone left the blades hanging in the air.
    float ground_height = grassGroundHeight(root_xz);
    // Waterline cull factor — rides the seed's spare lane (arc.w)
    // because the geometry stage that expands this blade has no
    // soil-water binding of its own.  The built-ground cull rides the
    // same lane for the same reason.
    float water_k = 1.0f - smoothstep(kGrassWaterFadeStartM,
                                      kGrassWaterFadeEndM,
                                      soil_water_thickness.y);
    water_k *= 1.0f - smoothstep(kGrassBuiltFadeStart,
                                 kGrassBuiltFadeEnd, built);

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
    out_seed.arc      = vec4(blade.arc, water_k);
}
