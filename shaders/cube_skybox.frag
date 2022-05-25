#version 450
#extension GL_ARB_separate_shader_objects : enable

#define RENDER_SUN 1
#define RENDER_SUNLIGHT_SCATTERING  1
const int iSteps = 16;
const int jSteps = 8;

#include "global_definition.glsl.h"
#include "sun.glsl.h"
#include "sunlight_scattering.glsl.h"

#define UX3D_MATH_PI 3.1415926535897932384626433832795
#define UX3D_MATH_INV_PI (1.0 / UX3D_MATH_PI)

layout(push_constant) uniform SunSkyUniformBufferObject {
    SunSkyParams params;
};

layout (location = 0) in vec2 inUV;

// output cubemap faces
layout(location = 0) out vec4 outFace0;
layout(location = 1) out vec4 outFace1;
layout(location = 2) out vec4 outFace2;
layout(location = 3) out vec4 outFace3;
layout(location = 4) out vec4 outFace4;
layout(location = 5) out vec4 outFace5;

void writeFace(int face, vec3 colorIn)
{
	vec4 color = vec4(colorIn.rgb, 1.0f);
	
	if(face == 0)
		outFace0 = color;
	else if(face == 1)
		outFace1 = color;
	else if(face == 2)
		outFace2 = color;
	else if(face == 3)
		outFace3 = color;
	else if(face == 4)
		outFace4 = color;
	else //if(face == 5)
		outFace5 = color;
}

vec3 uvToXYZ(int face, vec2 uv)
{
	if(face == 0)
		return vec3(     1.f,   uv.y,    -uv.x);
		
	else if(face == 1)
		return vec3(    -1.f,   uv.y,     uv.x);
		
	else if(face == 2)
		return vec3(   +uv.x,   -1.f,    +uv.y);
	
	else if(face == 3)
		return vec3(   +uv.x,    1.f,    -uv.y);
		
	else if(face == 4)
		return vec3(   +uv.x,   uv.y,      1.f);
		
	else //if(face == 5)
		return vec3(    -uv.x,  +uv.y,     -1.f);
}

vec2 dirToUV(vec3 dir)
{
	return vec2(
		0.5f + 0.5f * atan(dir.z, dir.x) / UX3D_MATH_PI,
		1.f - acos(dir.y) / UX3D_MATH_PI);
}

float saturate(float v)
{
	return clamp(v, 0.0f, 1.0f);
}

// entry point
void main() 
{
	for(int face = 0; face < 6; ++face)
	{		
		vec3 scan = uvToXYZ(face, inUV*2.0-1.0);		
			
		vec3 view_dir = normalize(scan);		
		view_dir = vec3(view_dir.x, -view_dir.y, view_dir.z);
	
		vec3 color = vec3(0.0, 0.0, 0.0);
		vec3 sun_pos = normalize(params.sun_pos);

#if RENDER_SUNLIGHT_SCATTERING
    vec3 r = view_dir;
    vec3 r0 = vec3(0, kPlanetRadius, 0);// + view_params.camera_pos.xyz;
    float g = params.g;
    float cast_range = rsi(r0, r, kAtmosphereRadius);
#if ATMOSPHERE_USE_LUT
    color += atmosphereLut(
        r,                               // normalized ray direction
        r0,                              // ray origin
        cast_range,
        sun_pos,                        // position of the sun
        22.0,                           // intensity of the sun
        kPlanetRadius,                  // radius of the planet in meters
        kAtmosphereRadius,              // radius of the atmosphere in meters
        vec3(5.5e-6, 13.0e-6, 22.4e-6), // Rayleigh scattering coefficient
        21e-6,                          // Mie scattering coefficient
        params.inv_rayleigh_scale_height,           // Rayleigh scale height
        params.inv_mie_scale_height,                // Mie scale height
        g,                              // Mie preferred scattering direction
        vec2(1.0f, 1.0f),
        iSteps).xyz;
#else
    color += atmosphere(
        r,                              // normalized ray direction
        r0,                             // ray origin
        cast_range,
        sun_pos,                        // position of the sun
        22.0,                           // intensity of the sun
        kPlanetRadius,                  // radius of the planet in meters
        kAtmosphereRadius,              // radius of the atmosphere in meters
        vec3(5.5e-6, 13.0e-6, 22.4e-6), // Rayleigh scattering coefficient
        21e-6,                          // Mie scattering coefficient
        params.inv_rayleigh_scale_height,           // Rayleigh scale height
        params.inv_mie_scale_height,                // Mie scale height
        g                               // Mie preferred scattering direction
        );
#endif
#endif

#if RENDER_SUN
		  color += renderSun(view_dir, sun_pos);
#endif

		writeFace(face, max(color * 2.0f, 0.0f));
	}
}
