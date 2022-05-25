#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "tile_common.glsl.h"
#include "noise.glsl.h"

layout(location = VINPUT_POSITION) in vec3 in_position;

layout(push_constant) uniform TileUniformBufferObject {
    TileParams tile_params;
};

layout(set = TILE_PARAMS_SET, binding = ROCK_LAYER_BUFFER_INDEX) uniform sampler2D rock_layer;
layout(set = TILE_PARAMS_SET, binding = SOIL_WATER_LAYER_BUFFER_INDEX) uniform sampler2D soil_water_layer;

layout(location = IINPUT_MAT_ROT_0) in vec3 in_loc_rot_mat_0;
layout(location = IINPUT_MAT_ROT_1) in vec3 in_loc_rot_mat_1;
layout(location = IINPUT_MAT_ROT_2) in vec3 in_loc_rot_mat_2;
layout(location = IINPUT_MAT_POS_SCALE) in vec4 in_loc_pos_scale;

layout(location = 0) out VsPsData {
    vec4 position_ws;
} out_data;

void main() {
    vec4 hash_values = clamp(hash43(vec3(tile_params.min, gl_InstanceIndex)), 0.0f, 1.0f);

    // tile world position.
    vec2 pos_xz_ws = tile_params.min + hash_values.xy * tile_params.range;

    // convert tile world position to uv coordinate.
    vec2 world_map_uv = (pos_xz_ws - tile_params.world_min) * tile_params.inv_world_range;

    float ground_height = texture(rock_layer, world_map_uv).x;
    vec2 soil_water_thickness = texture(soil_water_layer, world_map_uv).xy * SOIL_WATER_LAYER_MAX_THICKNESS;
    ground_height += soil_water_thickness.x;

    mat3 local_world_rot_mat =
        mat3x3(in_loc_rot_mat_0,
               in_loc_rot_mat_1,
               in_loc_rot_mat_2);

    out_data.position_ws.xyz =
        //local_world_rot_mat *
        in_position * 1.0f + vec3(pos_xz_ws.x, ground_height, pos_xz_ws.y); //in_loc_pos_scale.xyz;

    out_data.position_ws.w = hash_values.z;
}