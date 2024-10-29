#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"
#include "noise.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	ViewCameraInfo camera_info;
};

layout(set = PBR_MATERIAL_PARAMS_SET, binding = ALBEDO_TEX_INDEX) uniform sampler2D src_img;

layout(location = 0) out vec4 outColor;

layout(location = 0) in HairVsPsData ps_in_data;

void main() {
#if 0
    const float s_num_hairs = 200.0f;
    const uint  s_num_layers = 10;
    const float s_hair_oquacity = 0.8f;
    const float s_hair_width = 0.1f;
    const float s_inv_hair_width = 1.0f / s_hair_width;

    vec3 v = normalize(camera_info.position.xyz - ps_in_data.vertex_position);

    mat3 tangent_world_matrix =
        mat3(ps_in_data.vertex_tangent, ps_in_data.vertex_binormal, ps_in_data.vertex_normal);

    float accum_opacity = 1.0f;
    float accum_hair_color = 0.0f;
    float offset_dir = 1.0f;
    float hair_color = 0.8f;
    for (int i = 0; i < s_num_layers && accum_opacity > 0.01f; ++i) {
        vec2 noise_value = hash21(i * 1.34f);
        float uv_start = ps_in_data.vertex_tex_coord.x * 1.1f - 0.05f;
        float uv_end = uv_start * (1.0f + (noise_value.x - 0.5f) * 0.1f);
        float cur_uv = mix(uv_start, uv_end, pow(ps_in_data.vertex_tex_coord.y, 1.5f));
        float hair_idx = cur_uv * s_num_hairs;
        float i_hair_idx = floor(hair_idx);
        float hair_length = 0.95f + hash12(vec2(i * 1.34f, i_hair_idx)) * 0.05f;
        float sin_theta = clamp((fract(hair_idx) - 0.5f) * s_inv_hair_width, -1.0f, 1.0f);
        float cos_theta = sqrt(1.0f - sin_theta * sin_theta);
        float cur_opacity = min(cos_theta * s_hair_oquacity, 1.0f);
        vec3 hair_normal = tangent_world_matrix * vec3(sin_theta, cos_theta, 0);
        if (cur_uv > 1.0f || cur_uv < 0.0f || ps_in_data.vertex_tex_coord.y > hair_length)
			cur_opacity = 0.0f;
        accum_hair_color += accum_opacity * cur_opacity * hair_color * max(dot(-v, hair_normal), 0.0f);
        accum_opacity *= (1.0f - cur_opacity);

        offset_dir *= -1.0f;
	}

    outColor = vec4(vec3(accum_hair_color), accum_opacity);

    if (accum_opacity > 0.99f)
        discard;
#else
	outColor = texture(src_img, ps_in_data.vertex_tex_coord.xy);

    if (outColor.a > 0.99f)
        discard;
#endif
}