#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "weather_common.glsl.h"

layout(push_constant) uniform DebugDrawUniformBufferObject {
    DebugDrawParams params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(location = 0) out VsPsData {
    vec4 debug_info;
} out_data;

layout(set = TILE_PARAMS_SET, binding = SRC_TEMP_TEX_INDEX) uniform sampler3D src_temp_volume;
layout(set = TILE_PARAMS_SET, binding = SRC_MOISTURE_TEX_INDEX) uniform sampler3D src_moisture_volume;
layout(set = TILE_PARAMS_SET, binding = SRC_AIRFLOW_INDEX) uniform sampler3D src_airflow;

void main() {
    uint vertex_idx = gl_VertexIndex % 3;
    uint triangle_idx = gl_VertexIndex / 3;

    uint xy_count = params.size.x * params.size.y;
    uint iz = triangle_idx / xy_count;
    uint nxy = triangle_idx % xy_count;
    uint iy = nxy / params.size.x;
    uint ix = nxy % params.size.x;

    vec3 f_xyz = (vec3(ix, iy, iz) + 0.5f) * params.inv_size;

    vec3 sample_pos = f_xyz * params.debug_range + params.debug_min;
    vec4 position_ss = camera_info.view_proj * vec4(sample_pos, 1.0);

    vec3 uvw;
    uvw.z = log2(max((sample_pos.y - kAirflowLowHeight), 0.0f) + 1.0f) /
            log2(kAirflowMaxHeight - kAirflowLowHeight + 1.0f);

    uvw.xy = (sample_pos.xz - params.world_min) * params.inv_world_range;
    float temp = texture(src_temp_volume, uvw).x;
    float moisture = texture(src_moisture_volume, uvw).x;
    out_data.debug_info.x = temp;
    out_data.debug_info.y = moisture;

    vec4 airflow_info = texture(src_airflow, uvw);
    vec3 arrow_dir = airflow_info.xyz * 2.0f - 1.0f;
    vec3 view_dir = sample_pos - camera_info.position.xyz;
    vec3 left_dir = normalize(cross(arrow_dir, view_dir));

    vec3 offset = vec3(0);
    if (vertex_idx == 0) {
        offset = -left_dir * 6.4f;
    }
    else if (vertex_idx == 1) {
        offset = left_dir * 6.4f;
    }
    else {
        offset = arrow_dir * 128.0f;
    }

    if (params.debug_type == DEBUG_DRAW_TEMPRETURE) {
        sample_pos += offset * position_ss.w * /*getPackedVectorLength(airflow_info.w) * */0.0005f;
    }
    else {
        sample_pos += offset * position_ss.w * /*out_data.debug_info.y * */0.0005f;
    }

    vec4 position_ws = vec4(sample_pos, 1.0);
    gl_Position = camera_info.view_proj * position_ws;
}