#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

layout(set = PBR_MATERIAL_PARAMS_SET, binding = DIFFUSE_TEX_INDEX) uniform sampler2D diffuse_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = NORMAL_TEX_INDEX) uniform sampler2D normal_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = GLOSSNESS_TEX_INDEX) uniform sampler2D glossness_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = SPECULAR_TEX_INDEX) uniform sampler2D specular_tex;

layout(location = 0) in VsPsData {
    vec3 vertex_position;
    vec2 vertex_tex_coord;
    vec3 vertex_normal;
} in_data;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = in_data.vertex_tex_coord;
    vec3 uv_dx = dFdx(vec3(uv, 0.0));
    vec3 uv_dy = dFdy(vec3(uv, 0.0));

    vec3 v = normalize(camera_info.position.xyz - in_data.vertex_position);

    vec3 t_ = (uv_dy.t * dFdx(in_data.vertex_position) - uv_dx.t * dFdy(in_data.vertex_position)) /
        (uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t);

    vec3 n, t, b, ng;

    // Compute geometrical TBN:
    // Normals are either present as vertex attributes or approximated.
    ng = normalize(in_data.vertex_normal);
    t = normalize(t_ - ng * dot(ng, t_));
    b = cross(ng, t);

    // For a back-facing surface, the tangential basis vectors are negated.
/*    float facing = step(0.0, dot(v, ng)) * 2.0 - 1.0;
    t *= facing;
    b *= facing;
    ng *= facing;

    t = mat3(t, b, ng) * vec3(1.0, 0.0, 0.0);
    b = normalize(cross(ng, t));*/

    // Compute pertubed normals:
    n = texture(normal_tex, uv).rgb * 2.0 - vec3(1.0);
    n = -ng;//mat3(t, b, ng) * normalize(n);

    vec3 light = -camera_info.facing_dir;

    vec3 diffuse = max(dot(light, normalize(n)), 0.0) * texture(diffuse_tex, uv).rgb;

    outColor = vec4(diffuse, 1.0f);
}