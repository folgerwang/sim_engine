#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(location = 0) out VsPsData {
    vec3    vertex_position;
    vec2    world_map_uv;
    vec3    test_color;
    float   water_depth;
} out_data;

layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;
layout(set = TILE_PARAMS_SET, binding = SOIL_WATER_LAYER_BUFFER_INDEX) uniform sampler2D soil_water_layer;
layout(set = TILE_PARAMS_SET, binding = ORTHER_INFO_LAYER_BUFFER_INDEX) uniform sampler2D orther_info_layer;

void main() {
    uint tile_size = tile_params.segment_count + 1;

    uint col = gl_VertexIndex % tile_size;
    uint row = gl_VertexIndex / tile_size;

    // tile scale factor, range from (0 to 1)
    vec2 factor_xy = vec2(col, row) * tile_params.inv_segment_count;

    // tile world position.
    vec2 pos_xz_ws = tile_params.min + factor_xy * tile_params.range;

    // convert tile world position to uv coordinate.
    vec2 world_map_uv = (pos_xz_ws - tile_params.world_min) * tile_params.inv_world_range;

    float layer_height = texture(rock_layer, world_map_uv).x;
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
    layer_height += texture(orther_info_layer, world_map_uv).y * SNOW_LAYER_MAX_THICKNESS;
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