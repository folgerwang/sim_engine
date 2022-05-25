#version 460
#include "..\..\global_definition.glsl.h"

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable

layout(location = kPayLoadHitValueIdx) rayPayloadInEXT vec3 hit_value;
layout(location = kPayLoadShadowedIdx) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, set = 0) uniform UBO {
	mat4 view_inverse;
	mat4 proj_inverse;
	vec4 light_pos;
} ubo;

layout(binding = 3, set = 0) buffer Vertices { float v[]; } vertices;
layout(binding = 4, set = 0) buffer Indices { uint16_t i[]; } indices;
layout(binding = 5, set = 0) buffer Geometries { VertexBufferInfo info[]; } geometries;

struct Vertex
{
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 _pad0;
  vec4 _pad1;
 };

Vertex unpack(uint16_t index, in VertexBufferInfo geom_info)
{
	const uint normal_idx = geom_info.normal_base + index * 3;
	float x = vertices.v[normal_idx + 0];
	float y = vertices.v[normal_idx + 1];
	float z = vertices.v[normal_idx + 2];

	Vertex v;
	v.normal.x = dot(vec4(x, y, z, 0.0f), geom_info.matrix[0]);
	v.normal.y = dot(vec4(x, y, z, 0.0f), geom_info.matrix[1]);
	v.normal.z = dot(vec4(x, y, z, 0.0f), geom_info.matrix[2]);
	v.pos = vec3(0, 0, 0);
	float r = 1.0f, g = 1.0f, b = 1.0f;
	if (geom_info.color_base != 0xffffffff) {
		const uint color_idx = geom_info.color_base + index * 3;
		r = vertices.v[color_idx + 0];
		g = vertices.v[color_idx + 1];
		b = vertices.v[color_idx + 2];
	}
	v.color = vec4(r, g, b, 1.0);

	return v;
}

void main()
{
	VertexBufferInfo geom_info = geometries.info[gl_GeometryIndexEXT];
	uint base_idx = geom_info.index_base + 3 * gl_PrimitiveID;

	Vertex v0 = unpack(indices.i[base_idx], geom_info);
	Vertex v1 = unpack(indices.i[base_idx + 1], geom_info);
	Vertex v2 = unpack(indices.i[base_idx + 2], geom_info);

	// Interpolate normal
	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x +
							v1.normal * barycentricCoords.y +
							v2.normal * barycentricCoords.z);

	// Basic lighting
	vec3 light_vector = normalize(ubo.light_pos.xyz);
	float dot_product = max(dot(light_vector, normal), 0.2);
	hit_value = v0.color.rgb * dot_product;
 
	// Shadow casting
	float t_min = 0.001;
	float t_max = 10000.0;
	uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT |
					 gl_RayFlagsOpaqueEXT |
					 gl_RayFlagsSkipClosestHitShaderEXT;
	uint cull_mask = 0xFF;
	uint sbt_record_offset = 1;
	uint sbt_record_stride = 0;
	uint miss_index = 1;
	vec3 hit_point = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

	shadowed = true;  
	// Trace shadow ray and offset indices to match shadow hit/miss shader group indices
	traceRayEXT(
		topLevelAS,
		ray_flags,
		cull_mask,
		sbt_record_offset,
		sbt_record_stride,
		miss_index,
		hit_point,
		t_min,
		light_vector,
		t_max,
		kPayLoadShadowedIdx);

	if (shadowed) {
		hit_value *= 0.3;
	}

#ifdef DEBUG_GEOMETRY_IDX
	if (gl_GeometryIndexEXT < 2) {
		hit_value = vec3(1, 0, 0);
	}
	else if (gl_GeometryIndexEXT < 4) {
		hit_value = vec3(0, 1, 0);
	}
	else if (gl_GeometryIndexEXT < 6) {
		hit_value = vec3(0, 0, 1);
	}
	else if (gl_GeometryIndexEXT < 8) {
		hit_value = vec3(1, 1, 0);
	}
	else if (gl_GeometryIndexEXT < 10) {
		hit_value = vec3(1, 0, 1);
	}
	else if (gl_GeometryIndexEXT < 12) {
		hit_value = vec3(0, 1, 1);
	}
#endif
}
