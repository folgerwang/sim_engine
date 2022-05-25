#version 460
#include "..\..\global_definition.glsl.h"

#extension GL_EXT_ray_tracing : require

layout(location = kPayLoadHitValueIdx) rayPayloadInEXT vec3 hit_value;

void main()
{
    hit_value = vec3(0.0, 0.0, 0.2);
}