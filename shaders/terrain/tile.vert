#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "..\global_definition.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) out TileVsPsData out_data;

layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;
layout(set = TILE_PARAMS_SET, binding = SOIL_WATER_LAYER_BUFFER_INDEX) uniform sampler2D soil_water_layer;
layout(set = TILE_PARAMS_SET, binding = ORTHER_INFO_LAYER_BUFFER_INDEX) uniform sampler2D other_info_layer;

#include "tile_detail.glsl.h"

// Snap an edge parameter onto a coarser neighbour's vertex grid so both
// tiles emit EXACTLY the same edge positions — no T-junction cracks
// between LODs.  nseg = the neighbour's segment count.
float snapToGrid(float t, float nseg) {
    return floor(t * nseg + 0.5f) / nseg;
}

void main() {
    uint tile_size = tile_params.segment_count + 1;

    uint col = gl_VertexIndex % tile_size;
    uint row = gl_VertexIndex / tile_size;

    // tile scale factor, range from (0 to 1)
    vec2 factor_xy = vec2(col, row) * tile_params.inv_segment_count;

    // ── LOD edge stitching ─────────────────────────────────────────────
    // tile_params.offset packs the 4 edge-neighbour SEGMENT COUNTS
    // (bytes: x- , x+ , y- , y+).  A vertex on an edge shared with a
    // COARSER neighbour snaps onto that neighbour's grid; equal or finer
    // neighbours need no action here (the finer side snaps to us).
    {
        uint elods = tile_params.offset;
        float seg = float(tile_params.segment_count);
        if (col == 0u) {
            float ns = float(elods & 0xFFu);
            if (ns > 0.0f && ns < seg) factor_xy.y = snapToGrid(factor_xy.y, ns);
        } else if (col == tile_params.segment_count) {
            float ns = float((elods >> 8) & 0xFFu);
            if (ns > 0.0f && ns < seg) factor_xy.y = snapToGrid(factor_xy.y, ns);
        }
        if (row == 0u) {
            float ns = float((elods >> 16) & 0xFFu);
            if (ns > 0.0f && ns < seg) factor_xy.x = snapToGrid(factor_xy.x, ns);
        } else if (row == tile_params.segment_count) {
            float ns = float((elods >> 24) & 0xFFu);
            if (ns > 0.0f && ns < seg) factor_xy.x = snapToGrid(factor_xy.x, ns);
        }
    }

    // tile world position.
    vec2 pos_xz_ws = tile_params.min + factor_xy * tile_params.range;

    // convert tile world position to uv coordinate.
    vec2 world_map_uv = (pos_xz_ws - tile_params.world_min) * tile_params.inv_world_range;

    float layer_height = texture(rock_layer, world_map_uv).x;
    // Runtime 1 m detail (streamed tiles), fading back to the base map
    // with camera distance.
    float detail_fade = terrainDetailFade(pos_xz_ws, camera_info.position.xyz);
    layer_height = terrainDetailHeight(pos_xz_ws, layer_height, detail_fade);
    // Beyond the terrain map: fade to the neutral surround plain instead
    // of stretching the border texels into stripes.
    {
        vec2 ov = (world_map_uv - clamp(world_map_uv, 0.0f, 1.0f))
                  / tile_params.inv_world_range;          // meters past edge
        float sfade = smoothstep(0.0f, kTerrainSurroundFadeMeters,
                                 length(ov));
        layer_height = mix(layer_height, kTerrainSurroundHeightMeters,
                           sfade);
    }
    // Outer-ring depth bias (cm, see TileObject::draw): coarse distant
    // tiles sit just below the fine terrain where they overlap.
    layer_height -= float(tile_params.pad_0) * 0.01f;
    out_data.water_depth = 0.0f;
#if defined(SOIL_PASS) || defined(WATER_PASS) || defined(SNOW_PASS)
    vec2 soil_water_thickness = texture(soil_water_layer, world_map_uv).xy * SOIL_WATER_LAYER_MAX_THICKNESS;
    layer_height += soil_water_thickness.x;
#endif
#if defined(WATER_PASS) || defined(SNOW_PASS)
    layer_height += soil_water_thickness.y;
    out_data.water_depth = soil_water_thickness.y;
#endif
#if defined(SNOW_PASS)
    layer_height += texture(other_info_layer, world_map_uv).y * SNOW_LAYER_MAX_THICKNESS;
#endif

    vec4 position_ws = vec4(pos_xz_ws.x, layer_height, pos_xz_ws.y, 1.0);
    gl_Position = camera_info.view_proj * position_ws;

    out_data.vertex_position = position_ws.xyz;
    out_data.world_map_uv = world_map_uv;

    uint idx_x = tile_params.tile_index % 0x03;
    uint idx_y = (tile_params.tile_index >> 2) % 0x03;
    uint idx_z = (tile_params.tile_index >> 4) % 0x03;
    out_data.test_color = vec3(idx_x / 3.0f, idx_y / 3.0f, idx_z / 3.0f);
}
