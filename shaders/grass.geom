#version 450 core
#include "global_definition.glsl.h"

layout (points) in;
layout (triangle_strip, max_vertices = 5) out;

layout(location = 0) in VsPsData {
    vec4 position_ws;
} in_data[];

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

void main() {
    float angle = in_data[0].position_ws.w * 2.0f * 3.1415926;
    vec2 sincos_xy = vec2(sin(angle), cos(angle));
    const float root_size = 0.05;
    const float leaf_size = 0.08;
    const float leaf_1_size = 0.03;

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(sincos_xy.x * root_size, 0.0, sincos_xy.y * root_size), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(-sincos_xy.x * root_size, 0.0, -sincos_xy.y * root_size), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(sincos_xy.x * leaf_size, 0.5, sincos_xy.y * leaf_size), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(-sincos_xy.x * leaf_size, 0.5, -sincos_xy.y * leaf_size), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(sincos_xy.x * leaf_1_size, 0.8, sincos_xy.y * leaf_1_size), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(-sincos_xy.x * leaf_1_size, 0.8, -sincos_xy.y * leaf_1_size), 1.0);
    EmitVertex();

    gl_Position = camera_info.view_proj * vec4(in_data[0].position_ws.xyz + vec3(0.0, 1.0, 0.0), 1.0);
    EmitVertex();
  
    EndPrimitive();
}  