#version 460
#include "..\..\global_definition.glsl.h"

#extension GL_EXT_ray_tracing : require

layout(location = kPayLoadShadowedIdx) rayPayloadInEXT bool shadowed;

void main()
{
	shadowed = false;
}