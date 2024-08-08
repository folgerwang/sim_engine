// Mountains. By David Hoskins - 2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// https://www.shadertoy.com/view/4slGD4
// A ray-marched version of my terrain renderer which uses
// streaming texture normals for speed:-
// http://www.youtube.com/watch?v=qzkBnCBpQAM

// It uses binary subdivision to accurately find the height map.
// Lots of thanks to Inigo and his noise functions!

// Video of my OpenGL version that 
// http://www.youtube.com/watch?v=qzkBnCBpQAM


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "terrain.h"
#include "shaders/global_definition.glsl.h"

#define USE_MESH_SHADER 1

#if 0
float tree_line = 0.0f;
float tree_col = 0.0f;
float i_time = 1000.0f;
const glm::ivec2 i_mouse = glm::ivec2(960, 540);

glm::vec3 sun_light = glm::normalize(glm::vec3(0.4f, 0.4f, 0.48f));
glm::vec3 sun_colour = glm::vec3(1.0f, .9f, .83f);
float specular = 0.0f;
glm::vec3 camera_pos;
float ambient;
glm::vec2 add = glm::vec2(1.0f, 0.0f);
const auto kHashScale1 = .1031f;
const auto kHashScale3 = glm::vec3(.1031f, .1030f, .0973f);
const auto kHashScale4 = glm::vec4(.1031f, .1030f, .0973f, .1099f);

// This peturbs the fractal positions for each iteration down...
// Helps make nice twisted landscapes...
const auto rotate_2d = glm::mat2(1.3623f, 1.7531f, -1.7131f, 1.4623f);

// Alternative rotation:-
//const auto rotate_2d = glm::mat2(1.2323f, 1.999231f, -1.999231f, 1.22f);


//  1 out, 2 in...
float hash12(const glm::vec2& p)
{
	glm::vec3 p3 = glm::fract(glm::vec3(p.x, p.y, p.x) * kHashScale1);
	p3 += glm::dot(p3, glm::vec3(p3.y, p3.z, p3.x) + 19.19f);
	return glm::fract((p3.x + p3.y) * p3.z);
}
glm::vec2 hash22(const glm::vec2& p)
{
	glm::vec3 p3 = glm::fract(glm::vec3(glm::vec3(p.x, p.y, p.x)) * kHashScale3);
	p3 += glm::dot(p3, glm::vec3(p3.y, p3.z, p3.x) + 19.19f);
	return glm::fract((p3.x + glm::vec2(p3.y, p3.z)) * glm::vec2(p3.z, p3.y));
}

float noise(const glm::vec2& x)
{
	glm::vec2 p = floor(x);
	glm::vec2 f = fract(x);
	f = f * f * (3.0f - 2.0f * f);

	float res = glm::mix(glm::mix(hash12(p), hash12(p + glm::vec2(add.x, add.y)), f.x),
		glm::mix(hash12(p + glm::vec2(add.y, add.x)), hash12(p + add.x), f.x), f.y);
	return res;
}

glm::vec2 noise2(const glm::vec2& x)
{
	glm::vec2 p = floor(x);
	glm::vec2 f = fract(x);
	f = f * f * (3.0f - 2.0f * f);
	float n = p.x + p.y * 57.0f;
	glm::vec2 res = glm::mix(glm::mix(hash22(p), hash22(p + glm::vec2(add.x, add.y)), f.x),
		glm::mix(hash22(p + glm::vec2(add.y, add.x)), hash22(p + add.x), f.x), f.y);
	return res;
}

//--------------------------------------------------------------------------
float Trees(const glm::vec2& p)
{

	//return (texture(iChannel1,0.04f*p).x * tree_line);
	return noise(p * 13.0f) * tree_line;
}


//--------------------------------------------------------------------------
// Low def version for ray-marching through the height field...
// Thanks to IQ for all the noise stuff...

float getTerrainHeight(const glm::vec2& p)
{
	glm::vec2 pos = p * 0.05f;
	float w = (noise(pos * .25f) * 0.75f + .15f);
	w = 66.0f * w * w;
	glm::vec2 dxy = glm::vec2(0.0f);
	float f = .0;
	for (int i = 0; i < 5; i++)
	{
		f += w * noise(pos);
		w = -w * 0.4f;	//...Flip negative and positive for variation
		pos = rotate_2d * pos;
	}
	float ff = noise(pos * .002f);

	f += glm::pow(glm::abs(ff), 5.0f) * 275.f - 5.0f;
	return f;
}

//--------------------------------------------------------------------------
// Map to lower resolution for height field mapping for Scene function...
float getHeightMap(const glm::vec3& p)
{
	auto p_xz = glm::vec2(p.x, p.z);
	float h = getTerrainHeight(p_xz);


	float ff = noise(p_xz * .3f) + noise(p_xz * 3.3f) * .5f;
	tree_line = glm::smoothstep(ff, .0f + ff * 2.0f, h) * glm::smoothstep(1.0f + ff * 3.0f, .4f + ff, h);
	tree_col = Trees(p_xz);
	h += tree_col;

	return p.y - h;
}

//--------------------------------------------------------------------------
// High def version only used for grabbing normal information.
float getTerrainHeightmap(const glm::vec2& p)
{
	// There's some real magic numbers in here! 
	// The noise calls add large mountain ranges for more variation over distances...
	glm::vec2 pos = p * 0.050f;
	float w = (noise(pos * .25f) * 0.75f + .15f);
	w = 66.0f * w * w;
	glm::vec2 dxy = glm::vec2(0.0f);
	float f = .0f;
	for (int i = 0; i < 5; i++)
	{
		f += w * noise(pos);
		w = -w * 0.4f;	//...Flip negative and positive for varition	   
		pos = rotate_2d * pos;
	}
	float ff = noise(pos * glm::vec2(.002f));
	f += glm::pow(glm::abs(ff), 5.0f) * 275.f - 5.0f;


	tree_col = Trees(p);
	f += tree_col;
	if (tree_col > 0.0f) return f;


	// That's the last of the low resolution, now go down further for the Normal data...
	for (int i = 0; i < 6; i++)
	{
		f += w * noise(pos);
		w = -w * 0.4f;
		pos = rotate_2d * pos;
	}

	return f;
}

//--------------------------------------------------------------------------
float fractalNoise(const glm::vec2& xy)
{
	float w = .7f;
	float f = 0.0f;

	glm::vec2 t_xy = xy;

	for (int i = 0; i < 4; i++)
	{
		f += noise(t_xy) * w;
		w *= 0.5f;
		t_xy *= glm::vec2(2.7f);
	}
	return f;
}

//--------------------------------------------------------------------------
// Simply Perlin clouds that fade to the horizon...
// 200 units above the ground...
glm::vec3 getClouds(const glm::vec3& sky, const glm::vec3& rd)
{
	if (rd.y < 0.01f) return sky;
	float v = (200.0f - camera_pos.y) / rd.y;
	glm::vec2 rd2 = (glm::vec2(rd.x, rd.z) * v + glm::vec2(camera_pos.x, camera_pos.z)) * glm::vec2(0.01f);
	float f = (fractalNoise(rd2) - .55f) * 5.0f;
	// Uses the ray's y component for horizon fade of fixed colour clouds...
	return glm::mix(sky, glm::vec3(.55f, .55f, .52f), glm::clamp(f * rd.y - .1f, 0.0f, 1.0f));
}

//--------------------------------------------------------------------------
// Grab all sky information for a given ray from camera
glm::vec3 getSky(const glm::vec3& rd)
{
	float sun_amount = glm::max(glm::dot(rd, sun_light), 0.0f);
	float v = pow(1.0f - glm::max(rd.y, 0.0f), 5.f) * .5f;
	glm::vec3 sky = glm::vec3(v * sun_colour.x * 0.4f + 0.18f,
							  v * sun_colour.y * 0.4f + 0.22f,
							  v * sun_colour.z * 0.4f + .4f);
	// Wide glare effect...
	sky = sky + sun_colour * glm::pow(sun_amount, 6.5f) * .32f;
	// Actual sun...
	sky = sky + sun_colour * glm::min(pow(sun_amount, 1150.0f), .3f) * .65f;
	return sky;
}

//--------------------------------------------------------------------------
// Merge mountains into the sky background for correct disappearance...
glm::vec3 applyFog(const glm::vec3& rgb, float dis, const glm::vec3& dir)
{
	float fog_amount = glm::exp(-dis * 0.00005f);
	return glm::mix(getSky(dir), rgb, fog_amount);
}

//--------------------------------------------------------------------------
// Calculate sun light...
glm::vec3 doLighting(const glm::vec3& in_color, const glm::vec3& pos, const glm::vec3& normal, const glm::vec3& eye_dir, float dis)
{
	float h = dot(sun_light, normal);
	float c = glm::max(h, 0.0f) + ambient;
	auto color = in_color * sun_colour * c;
	// Specular...
	if (h > 0.0f)
	{
		auto r = glm::reflect(sun_light, normal);
		float spec_amount = pow(glm::max(glm::dot(r, glm::normalize(eye_dir)), 0.0f), 3.0f) * specular;
		color = glm::mix(color, sun_colour, spec_amount);
	}

	return color;
}

//--------------------------------------------------------------------------
// Hack the height, position, and normal data to create the coloured landscape
glm::vec3 getTerrainColor(const glm::vec3& pos, const glm::vec3& normal, float dis)
{
	glm::vec3 mat;
	specular = .0f;
	ambient = .1f;
	glm::vec3 dir = normalize(pos - camera_pos);

	glm::vec3 mat_pos = pos * 2.0f;// ... I had change scale halfway though, this lazy multiply allow me to keep the graphic scales I had
	auto mat_pos_xz = glm::vec2(mat_pos.x, mat_pos.z);

	float dis_sqrd = dis * dis;// Squaring it gives better distance scales.

	float f = glm::clamp(noise(mat_pos_xz * .05f), 0.0f, 1.0f);//*10.8;
	f += noise(mat_pos_xz * .1f + glm::vec2(normal.y, normal.z) * 1.08f) * .85f;
	f *= .55f;
	glm::vec3 m = glm::mix(glm::vec3(.63f * f + .2f, .7f * f + .1f, .7f * f + .1f),
						   glm::vec3(f * .43f + .1f, f * .3f + .2f, f * .35f + .1f),
						   f * .65f);
	mat = m * glm::vec3(f * m.x + .36f, f * m.y + .30f, f * m.z + .28f);
	// Should have used smoothstep to add colours, but left it using 'if' for sanity...
	if (normal.y < .5f)
	{
		float v = normal.y;
		float c = (.5f - normal.y) * 4.0f;
		c = glm::clamp(c * c, 0.1f, 1.0f);
		f = noise(glm::vec2(mat_pos.x * .09f, mat_pos.z * .095f + mat_pos.y * 0.15f));
		f += noise(glm::vec2(mat_pos.x * 2.233f, mat_pos.z * 2.23f)) * 0.5f;
		mat = glm::mix(mat, glm::vec3(.4f * f), c);
		specular += .1f;
	}

	// Grass. Use the normal to decide when to plonk grass down...
	if (mat_pos.y < 45.35f && normal.y > .65f)
	{

		m = glm::vec3(noise(mat_pos_xz * .023f) * .5f + .15f,
					  noise(mat_pos_xz * .03f) * .6f + .25f, 0.0f);
		m *= (normal.y - 0.65f) * .6f;
		mat = glm::mix(mat, m, glm::clamp((normal.y - .65f) * 1.3f * (45.35f - mat_pos.y) * 0.1f, 0.0f, 1.0f));
	}

	auto t_normal = normal;
	if (tree_col > 0.0)
	{
		mat = glm::vec3(.02f + noise(mat_pos_xz * 5.0f) * .03f, .05f, .0f);
		t_normal = glm::normalize(normal + glm::vec3(noise(mat_pos_xz * 33.0f) * 1.0f - .5f, .0f, noise(mat_pos_xz * 33.0f) * 1.0f - .5f));
		specular = .0f;
	}

	// Snow topped mountains...
	if (mat_pos.y > 80.0 && t_normal.y > .42)
	{
		float snow = glm::clamp((mat_pos.y - 80.0f - noise(mat_pos_xz * .1f) * 28.0f) * 0.035f, 0.0f, 1.0f);
		mat = glm::mix(mat, glm::vec3(.7f, .7f, .8f), snow);
		specular += snow;
		ambient += snow * .3f;
	}
	// Beach effect...
	if (mat_pos.y < 1.45f)
	{
		if (t_normal.y > .4f)
		{
			f = noise(mat_pos_xz * .084f) * 1.5f;
			f = glm::clamp((1.45f - f - mat_pos.y) * 1.34f, 0.0f, .67f);
			float t = (t_normal.y - .4f);
			t = (t * t);
			mat = glm::mix(mat, glm::vec3(.09f + t, .07f + t, .03f + t), f);
		}
		// Cheap under water darkening...it's wet after all...
		if (mat_pos.y < 0.0)
		{
			mat *= .2f;
		}
	}

	mat = doLighting(mat, pos, t_normal, dir, dis_sqrd);

	// Do the water...
	if (mat_pos.y < 0.0)
	{
		// Pull back along the ray direction to get water surface point at y = 0.0 ...
		float time = (i_time) * .03f;
		auto wat_pos = mat_pos;
		wat_pos += -dir * (wat_pos.y / dir.y);
		// Make some dodgy waves...
		float tx = glm::cos(wat_pos.x * .052f) * 4.5f;
		float tz = glm::sin(wat_pos.z * .072f) * 4.5f;
		glm::vec2 co = noise2(glm::vec2(wat_pos.x * 4.7f + 1.3f + tz, wat_pos.z * 4.69f + time * 35.0f - tx));
		co += noise2(glm::vec2(wat_pos.z * 8.6f + time * 13.0f - tx, wat_pos.x * 8.712f + tz)) * .4f;
		glm::vec3 nor = glm::normalize(glm::vec3(co.x, 20.0f, co.y));
		nor = glm::normalize(glm::reflect(dir, nor));
		// Mix it in at depth transparancy to give beach cues..
		tx = wat_pos.y - mat_pos.y;
		mat = glm::mix(mat, getClouds(getSky(nor) * glm::vec3(.3f, .3f, .5f), nor) * .1f + glm::vec3(.0f, .02f, .03f), glm::clamp((tx) * .4f, .6f, 1.f));
		// Add some extra water glint...
		// todo.
		//mat += glm::vec3(.1f) * glm::clamp(1.f - pow(tx + .5f, 3.f) * texture(iChannel1, glm::vec2(wat_pos.x, wat_pos.z) * .1f, -2.f).x, 0.f, 1.0f);
		float sun_amount = glm::max(glm::dot(nor, sun_light), 0.0f);
		mat = mat + sun_colour * glm::pow(sun_amount, 228.5f) * .6f;
		auto temp = (wat_pos - camera_pos * 2.f) * .5f;
		dis_sqrd = glm::dot(temp, temp);
	}
	mat = applyFog(mat, dis_sqrd, dir);
	return mat;
}

//--------------------------------------------------------------------------
float binarySubdivision(const glm::vec3& rO, const glm::vec3& r_d, const glm::vec2& t)
{
	// Home in on the surface by dividing by two and split...
	float halfway_t;

	auto t_t = t;
	for (int i = 0; i < 5; i++)
	{
		halfway_t = glm::dot(t_t, glm::vec2(.5f));
		float d = getHeightMap(rO + halfway_t * r_d);
		t_t = glm::mix(glm::vec2(t_t.x, halfway_t), glm::vec2(halfway_t, t_t.y), glm::step(0.5f, d));

	}
	return halfway_t;
}

//--------------------------------------------------------------------------
bool getScene(const glm::vec3& rO, const glm::vec3& r_d, const glm::vec2& frag_coord, float& res_t)
{
	float t = 1.f + hash12(frag_coord) * 1.f;
	float old_t = 0.0;
	float delta = 0.0;
	bool fin = false;
	bool res = false;
	glm::vec2 distances;
	for (int j = 0; j < 150; j++)
	{
		if (fin || t > 240.0f) break;
		auto p = rO + t * r_d;
		//if (t > 240.0 || p.y > 195.0) break;
		float h = getHeightMap(p); // ...Get this positions height mapping.
		// Are we inside, and close enough to fudge a hit?...
		if (h < 0.5f)
		{
			fin = true;
			distances = glm::vec2(old_t, t);
			break;
		}
		// Delta ray advance - a fudge between the height returned
		// and the distance already travelled.
		// It's a really fiddly compromise between speed and accuracy
		// Too large a step and the tops of ridges get missed.
		delta = glm::max(0.01f, 0.3f * h) + (t * 0.0065f);
		old_t = t;
		t += delta;
	}
	if (fin) res_t = binarySubdivision(rO, r_d, distances);

	return fin;
}

//--------------------------------------------------------------------------
glm::vec3 getCameraPath(float t, const glm::ivec2& screen_size)
{
	float m = 1.0 + (i_mouse.x / screen_size.x) * 300.0f;
	t = (i_time * 1.5f + m + 657.0f) * .006f + t;
	glm::vec2 p = 476.0f * glm::vec2(glm::sin(3.5f * t), glm::cos(1.5f * t));
	return glm::vec3(35.0f - p.x, 0.6f, 4108.0f + p.y);
}

//--------------------------------------------------------------------------
// Some would say, most of the magic is done in post! :D
glm::vec3 postEffects(glm::vec3 rgb, glm::vec2 uv)
{
	//#define CONTRAST 1.1
	//#define SATURATION 1.12
	//#define BRIGHTNESS 1.3
	//rgb = pow(abs(rgb), vec3(0.45));
	//rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	rgb = (1.0f - glm::exp(-rgb * 6.0f)) * 1.0024f;
	//rgb = clamp(rgb+hash12(fragCoord.xy*rgb.r)*0.1, 0.0, 1.0);
	return rgb;
}

//--------------------------------------------------------------------------
glm::vec4 getMainImage(const glm::vec2& frag_coord, const glm::ivec2& screen_size)
{
	glm::vec2 xy = -1.0f + 2.0f * frag_coord / glm::vec2(screen_size);
	glm::vec2 uv = xy * glm::vec2((float)screen_size.x / screen_size.y, 1.0);
	glm::vec3 cam_tar;

	// Use several forward heights, of decreasing influence with distance from the camera.
	float h = 0.0;
	float f = 1.0;
	for (int i = 0; i < 7; i++)
	{
		auto cam_pos = getCameraPath((.6f - f) * .008f, screen_size);
		h += getTerrainHeight(glm::vec2(cam_pos.x, cam_pos.z)) * f;
		f -= .1f;
	}

	auto cam_pos = getCameraPath(0.0, screen_size);
	camera_pos.x = cam_pos.x;
	camera_pos.z = cam_pos.z;
	cam_tar = getCameraPath(.005f, screen_size);
	cam_tar.y = camera_pos.y = glm::max((h * .25f) + 3.5f, 1.5f + glm::sin(i_time * 5.f) * .5f);

	float roll = 0.15f * sin(i_time * .2f);
	glm::vec3 cw = glm::normalize(cam_tar - camera_pos);
	glm::vec3 cp = glm::vec3(glm::sin(roll), glm::cos(roll), 0.0);
	glm::vec3 cu = glm::normalize(cross(cw, cp));
	glm::vec3 cv = glm::normalize(cross(cu, cw));
	glm::vec3 rd = glm::normalize(uv.x * cu + uv.y * cv + 1.5f * cw);

	glm::vec3 col;
	float distance;
	if (!getScene(camera_pos, rd, frag_coord, distance))
	{
		// Missed scene, now just get the sky value...
		col = getSky(rd);
		col = getClouds(col, rd);
	}
	else
	{
		// Get world coordinate of landscape...
		glm::vec3 pos = camera_pos + distance * rd;
		// Get normal from sampling the high definition height map
		// Use the distance to sample larger gaps to help stop aliasing...
		float p = .02 + .00005 * distance * distance;
		auto pos_xz = glm::vec2(pos.x, pos.z);
		auto nor = glm::vec3(0.0f, getTerrainHeightmap(pos_xz), 0.0f);
		auto v2 = nor - glm::vec3(p, getTerrainHeightmap(pos_xz + glm::vec2(p, 0.0f)), 0.0f);
		auto v3 = nor - glm::vec3(0.0, getTerrainHeightmap(pos_xz + glm::vec2(0.0f, -p)), -p);
		nor = glm::cross(v2, v3);
		nor = glm::normalize(nor);

		// Get the colour using all available data...
		col = getTerrainColor(pos, nor, distance);
	}

	col = postEffects(col, uv);

#ifdef STEREO	
	col *= vec3(isCyan, 1.0 - isCyan, 1.0 - isCyan);
#endif

	auto frag_color = glm::vec4(col, 1.0f);

	return frag_color;
}

#else

//#define STATIC_CAMERA
#define LOWQUALITY


//==========================================================================================
// general utilities
//==========================================================================================
#define kZero (min(iFrame,0))

float sdEllipsoidY(const glm::vec3& p, const glm::vec2& r)
{
    glm::vec3 r_xyx = glm::vec3(r.x, r.y, r.x);
    float k0 = glm::length(p / r_xyx);
    float k1 = glm::length(p / (r_xyx * r_xyx));
    return k0 * (k0 - 1.0f) / k1;
}

// return smoothstep and its derivative
glm::vec2 smoothstepd(float a, float b, float x)
{
    if (x < a) return glm::vec2(0.0, 0.0);
    if (x > b) return glm::vec2(1.0, 0.0);
    float ir = 1.0f / (b - a);
    x = (x - a)*ir;
    return glm::vec2(x*x*(3.0f - 2.0f*x), 6.0f*x*(1.0f - x)*ir);
}

glm::mat3 setCamera(const glm::vec3& ro, const glm::vec3& ta, float cr)
{
    glm::vec3 cw = glm::normalize(ta - ro);
    glm::vec3 cp = glm::vec3(glm::sin(cr), glm::cos(cr), 0.0f);
    glm::vec3 cu = glm::normalize(glm::cross(cw, cp));
    glm::vec3 cv = glm::normalize(glm::cross(cu, cw));
    return glm::mat3(cu, cv, cw);
}

//==========================================================================================
// hashes
//==========================================================================================

float hash1(glm::vec2 p)
{
    p = 50.0f*glm::fract(p*0.3183099f);
    return glm::fract(p.x*p.y*(p.x + p.y));
}

float hash1(float n)
{
    return glm::fract(n*17.0f*glm::fract(n*0.3183099f));
}

glm::vec2 hash2(float n) { return glm::fract(glm::sin(glm::vec2(n, n + 1.0))*glm::vec2(43758.5453123, 22578.1459123)); }


glm::vec2 hash2(glm::vec2 p)
{
    const glm::vec2 k = glm::vec2(0.3183099f, 0.3678794f);
    p = p * k + glm::vec2(k.y, k.x);
    return glm::fract(16.0f * k * glm::fract(p.x*p.y*(p.x + p.y)));
}

//==========================================================================================
// noises
//==========================================================================================

// value noise, and its analytical derivatives
glm::vec4 noised(const glm::vec3& x)
{
    glm::vec3 p = glm::floor(x);
    glm::vec3 w = glm::fract(x);
#if 1
    glm::vec3 u = w * w*w*(w*(w*6.0f - 15.0f) + 10.0f);
    glm::vec3 du = 30.0f*w*w*(w*(w - 2.0f) + 1.0f);
#else
    glm::vec3 u = w * w*(3.0f - 2.0f*w);
    glm::vec3 du = 6.0f*w*(1.0f - w);
#endif

    float n = p.x + 317.0f*p.y + 157.0f*p.z;

    float a = hash1(n + 0.0f);
    float b = hash1(n + 1.0f);
    float c = hash1(n + 317.0f);
    float d = hash1(n + 318.0f);
    float e = hash1(n + 157.0f);
    float f = hash1(n + 158.0f);
    float g = hash1(n + 474.0f);
    float h = hash1(n + 475.0f);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return glm::vec4(-1.0f + 2.0f*(k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x*u.y + k5 * u.y*u.z + k6 * u.z*u.x + k7 * u.x*u.y*u.z),
        2.0f* du * glm::vec3(k1 + k4 * u.y + k6 * u.z + k7 * u.y*u.z,
            k2 + k5 * u.z + k4 * u.x + k7 * u.z*u.x,
            k3 + k6 * u.x + k5 * u.y + k7 * u.x*u.y));
}

float noise(const glm::vec3& x)
{
    glm::vec3 p = glm::floor(x);
    glm::vec3 w = glm::fract(x);

#if 1
    glm::vec3 u = w * w*w*(w*(w*6.0f - 15.0f) + 10.0f);
#else
    glm::vec3 u = w * w*(3.0f - 2.0f*w);
#endif



    float n = p.x + 317.0f*p.y + 157.0f*p.z;

    float a = hash1(n + 0.0f);
    float b = hash1(n + 1.0f);
    float c = hash1(n + 317.0f);
    float d = hash1(n + 318.0f);
    float e = hash1(n + 157.0f);
    float f = hash1(n + 158.0f);
    float g = hash1(n + 474.0f);
    float h = hash1(n + 475.0f);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return -1.0f + 2.0f*(k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x*u.y + k5 * u.y*u.z + k6 * u.z*u.x + k7 * u.x*u.y*u.z);
}

glm::vec3 noised(const glm::vec2& x)
{
    glm::vec2 p = glm::floor(x);
    glm::vec2 w = glm::fract(x);
#if 1
    glm::vec2 u = w * w*w*(w*(w*6.0f - 15.0f) + 10.0f);
    glm::vec2 du = 30.0f*w*w*(w*(w - 2.0f) + 1.0f);
#else
    glm::vec2 u = w * w*(3.0f - 2.0f*w);
    glm::vec2 du = 6.0f*w*(1.0f - w);
#endif

    float a = hash1(p + glm::vec2(0, 0));
    float b = hash1(p + glm::vec2(1, 0));
    float c = hash1(p + glm::vec2(0, 1));
    float d = hash1(p + glm::vec2(1, 1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return glm::vec3(-1.0f + 2.0f*(k0 + k1 * u.x + k2 * u.y + k4 * u.x*u.y),
        2.0f* du * glm::vec2(k1 + k4 * u.y,
            k2 + k4 * u.x));
}

float noise(const glm::vec2& x)
{
    glm::vec2 p = glm::floor(x);
    glm::vec2 w = glm::fract(x);
#if 1
    glm::vec2 u = w * w*w*(w*(w*6.0f - 15.0f) + 10.0f);
#else
    glm::vec2 u = w * w*(3.0f - 2.0f*w);
#endif

    float a = hash1(p + glm::vec2(0, 0));
    float b = hash1(p + glm::vec2(1, 0));
    float c = hash1(p + glm::vec2(0, 1));
    float d = hash1(p + glm::vec2(1, 1));

    return -1.0f + 2.0f*(a + (b - a)*u.x + (c - a)*u.y + (a - b - c + d)*u.x*u.y);
}

//==========================================================================================
// fbm constructions
//==========================================================================================

const glm::mat3 m3 = glm::mat3(
    0.00f, 0.80f, 0.60f,
    -0.80f, 0.36f, -0.48f,
    -0.60f, -0.48f, 0.64f);
const glm::mat3 m3i = glm::mat3(
    0.00f, -0.80f, -0.60f,
    0.80f, 0.36f, -0.48f,
    0.60f, -0.48f, 0.64f);
const glm::mat2 m2 = glm::mat2(
    0.80f, 0.60f,
    -0.60f, 0.80f);
const glm::mat2 m2i = glm::mat2(
    0.80f, -0.60f,
    0.60f, 0.80f);

//------------------------------------------------------------------------------------------

float fbm_4(const glm::vec2& x)
{
    float f = 1.9f;
    float s = 0.55f;
    float a = 0.0f;
    float b = 0.5f;
    auto t_x = x;
    for (int i = 0; i < 4; i++)
    {
        float n = noise(t_x);
        a += b * n;
        b *= s;
        t_x = f * m2* t_x;
    }
    return a;
}

float fbm_4(const glm::vec3& x)
{
    float f = 2.0f;
    float s = 0.5f;
    float a = 0.0f;
    float b = 0.5f;
    auto t_x = x;
    for (int i = 0; i < 4; i++)
    {
        float n = noise(t_x);
        a += b * n;
        b *= s;
        t_x = f * m3* t_x;
    }
    return a;
}

glm::vec4 fbmd_8(const glm::vec3& x)
{
    float f = 1.92f;
    float s = 0.5f;
    float a = 0.0f;
    float b = 0.5f;
    glm::vec3  d = glm::vec3(0.0f);
    glm::mat3  m = glm::mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f);
    auto t_x = x;
    for (int i = 0; i < 7; i++)
    {
        glm::vec4 n = noised(t_x);
        a += b * n.x;          // accumulate values		
        d += b * m*glm::vec3(n.y, n.z, n.w);      // accumulate derivatives
        b *= s;
        t_x = f * m3* t_x;
        m = f * m3i*m;
    }
    return glm::vec4(a, d);
}

float fbm_9(const glm::vec2& x)
{
    float f = 1.9f;
    float s = 0.55f;
    float a = 0.0f;
    float b = 0.5f;
    auto t_x = x;
    for (int i = 0; i < 9; i++)
    {
        float n = noise(t_x);
        a += b * n;
        b *= s;
        t_x = f * m2* t_x;
    }
    return a;
}

glm::vec3 fbmd_9(const glm::vec2& x)
{
    float f = 1.9f;
    float s = 0.55f;
    float a = 0.0f;
    float b = 0.5f;
    glm::vec2  d = glm::vec2(0.0);
    glm::mat2  m = glm::mat2(1.0, 0.0, 0.0, 1.0);
    auto t_x = x;
    for (int i = 0; i < 9; i++)
    {
        glm::vec3 n = noised(t_x);
        a += b * n.x;          // accumulate values		
        d += b * m*glm::vec2(n.y, n.z);       // accumulate derivatives
        b *= s;
        t_x = f * m2*t_x;
        m = f * m2i*m;
    }
    return glm::vec3(a, d);
}

//==========================================================================================
// specifics to the actual painting
//==========================================================================================


//------------------------------------------------------------------------------------------
// global
//------------------------------------------------------------------------------------------

const glm::vec3  kSunDir = glm::vec3(-0.624695f, 0.468521f, -0.624695f);
const float kMaxTreeHeight = 2.4f;
const float kMaxHeight = 120.0f;

glm::vec3 fog(const glm::vec3& col, float t)
{
    glm::vec3 ext = glm::exp2(-t * 0.001f* glm::vec3(0.5, 0.7, 1.7));
    return col * ext + 0.6f*(1.0f - ext);
}

//------------------------------------------------------------------------------------------
// clouds
//------------------------------------------------------------------------------------------

glm::vec4 cloudsMap(const glm::vec3& pos)
{
    glm::vec4 n = fbmd_8(pos*0.003f* glm::vec3(0.6f, 1.0f, 0.6f) - glm::vec3(0.1f, 1.9f, 2.8f));
    glm::vec2 h = smoothstepd(-60.0f, 10.0f, pos.y) - smoothstepd(10.0f, 500.0f, pos.y);
    h.x = 2.0f*n.x + h.x - 1.3f;
    return glm::vec4(h.x, 2.0f*glm::vec3(n.y, n.z, n.w)* glm::vec3(0.6, 1.0, 0.6)*0.003f + glm::vec3(0.0, h.y, 0.0));
}

float cloudsShadow(const glm::vec3& ro, const glm::vec3& rd, float tmin, float tmax)
{
    float sum = 0.0;

    // bounding volume!!
    float tl = (-10.0f - ro.y) / rd.y;
    float th = (300.0f - ro.y) / rd.y;
    if (tl > 0.0f) tmin = glm::max(tmin, tl);
    if (th > 0.0f) tmax = glm::min(tmax, th);

    // raymarch
    float t = tmin;
    for (int i = 0; i < 64; i++)
    {
        glm::vec3  pos = ro + t * rd;
        glm::vec4  denGra = cloudsMap(pos);
        float den = denGra.x;
        float dt = glm::max(0.2f, 0.02f*t);
        if (den > 0.001)
        {
            float alp = glm::clamp(den*0.3f* glm::min(dt, tmax - t - dt), 0.0f, 1.0f);
            sum = sum + alp * (1.0f - sum);
        }
        else
        {
            dt *= 1.0f + 4.0f* glm::abs(den);
        }
        t += dt;
        if (sum > 0.995f || t > tmax) break;
    }

    return glm::clamp(1.0f - sum, 0.0f, 1.0f);
}

float terrainShadow(const glm::vec3& ro, const glm::vec3& rd, float mint);

glm::vec4 renderClouds(const glm::vec3& ro, const glm::vec3& rd, float tmin, float tmax, float& resT)
{
    glm::vec4 sum = glm::vec4(0.0);

    // bounding volume!!
    float tl = (-10.0f - ro.y) / rd.y;
    float th = (300.0f - ro.y) / rd.y;
    if (tl > 0.0f)   tmin = glm::max(tmin, tl); else return sum;
    /*if( th>0.0 )*/ tmax = glm::min(tmax, th);

    float t = tmin;
    float lastT = t;
    float thickness = 0.0;
#ifdef LOWQUALITY
    for (int i = 0; i < 128; i++)
#else
    for (int i = 0; i < 300; i++)
#endif
    {
        glm::vec3  pos = ro + t * rd;
        glm::vec4  denGra = cloudsMap(pos);
        float den = denGra.x;
#ifdef LOWQUALITY
        float dt = glm::max(0.1f, 0.011f*t);
#else
        float dt = glm::max(0.05f, 0.005f*t);
#endif
        if (den > 0.001)
        {
            float sha = 1.0f;
#ifndef LOWQUALITY
            sha *= terrainShadow(pos, kSunDir, 50.0);
            sha *= cloudsShadow(pos, kSunDir, 0.0, 50.0);
#endif
            glm::vec3 nor = -glm::normalize(glm::vec3(denGra.y, denGra.z, denGra.w));
            float dif = glm::clamp(dot(nor, kSunDir), 0.0f, 1.0f)*sha;
            float fre = glm::clamp(1.0f + dot(nor, rd), 0.0f, 1.0f)*sha;
            // lighting
            glm::vec3 lin = glm::vec3(0.70, 0.80, 1.00)*0.9f*(0.6f + 0.4f*nor.y);
            lin += glm::vec3(0.20, 0.25, 0.20)*0.7f*(0.5f - 0.5f*nor.y);
            lin += glm::vec3(1.00, 0.70, 0.40)*2.5f*dif*(1.0f - den);
            lin += glm::vec3(0.80, 0.70, 0.50)*1.3f* glm::pow(fre, 32.0f)*(1.0f - den);
            // color
            glm::vec3 col = glm::vec3(0.8, 0.77, 0.72)* glm::clamp(1.0f - 4.0f*den, 0.0f, 1.0f);

            col *= lin;

            col = fog(col, t);

            // front to back blending    
            float alp = glm::clamp(den*0.25f* glm::min(dt, tmax - t - dt), 0.0f, 1.0f);
            col *= alp;
            sum = sum + glm::vec4(col, alp)*(1.0f - sum.a);

            thickness += dt * den;
            lastT = t;
        }
        else
        {
#ifdef LOWQUALITY
            dt *= 1.0f + 4.0f*abs(den);
#else
            dt *= 0.8f + 2.0f*abs(den);
#endif
        }
        t += dt;
        if (sum.a > 0.995f || t > tmax) break;
    }

    resT = glm::mix(resT, lastT, sum.w);

    glm::vec3 add_on = glm::vec3(0.0f);
    if (thickness > 0.0f)
        add_on = glm::vec3(1.00f, 0.60f, 0.40f) * 0.2f * 
            glm::pow(glm::clamp(dot(kSunDir, rd), 0.0f, 1.0f), 32.0f) * glm::exp(-0.3f*thickness) * glm::clamp(thickness*4.0f, 0.0f, 1.0f);
    sum.x += add_on.x;
    sum.y += add_on.y;
    sum.z += add_on.z;

    return glm::clamp(sum, 0.0f, 1.0f);
}


//------------------------------------------------------------------------------------------
// terrain
//------------------------------------------------------------------------------------------

glm::vec2 terrainMap(const glm::vec2& p)
{
    const float sca = 0.0010f;
    const float amp = 300.0f;
    auto t_p = p;
    t_p *= sca;
    float e = fbm_9(t_p + glm::vec2(1.0f, -2.0f));
    float a = 1.0f - glm::smoothstep(0.12f, 0.13f, glm::abs(e + 0.12f)); // flag high-slope areas (-0.25, 0.0)
    e = e + 0.15f*glm::smoothstep(-0.08f, -0.01f, e);
    e *= amp;
    return glm::vec2(e, a);
}

glm::vec4 terrainMapD(const glm::vec2& p)
{
    const float sca = 0.0010f;
    const float amp = 300.0f;
    auto t_p = p;
    t_p *= sca;
    glm::vec3 e = fbmd_9(t_p + glm::vec2(1.0f, -2.0f));
    glm::vec2 c = smoothstepd(-0.08f, -0.01f, e.x);
    auto e_yz = glm::vec2(e.y, e.z);
    e.x = e.x + 0.15f*c.x;
    e_yz = e_yz + 0.15f*c.y* e_yz;
    e.x *= amp;
    e_yz *= amp*sca;
    return glm::vec4(e.x, glm::normalize(glm::vec3(-e_yz.x, 1.0f, -e_yz.y)));
}

glm::vec3 terrainNormal(const glm::vec2& pos)
{
#if 1
    auto map_d = terrainMapD(pos);
    return glm::vec3(map_d.y, map_d.z, map_d.w);
#else    
    glm::vec2 e = glm::vec2(0.03, 0.0);
    auto e_xy = glm::vec2(e.x, e.y);
    auto e_yx = glm::vec2(e.y, e.x);
    return glm::normalize(glm::vec3(terrainMap(pos - e_xy).x - terrainMap(pos + e_xy).x,
        2.0*e.x,
        terrainMap(pos - e_yx).x - terrainMap(pos + e_yx).x));
#endif    
}

float terrainShadow(const glm::vec3& ro, const glm::vec3& rd, float mint)
{
    float res = 1.0;
    float t = mint;
#if def LOWQUALITY
    for (int i = 0; i < 32; i++)
    {
        glm::vec3  pos = ro + t * rd;
        glm::vec2  env = terrainMap(glm::vec2(pos.x, pos.z));
        float hei = pos.y - env.x;
        res = glm::min(res, 32.0f*hei / t);
        if (res<0.0001 || pos.y>kMaxHeight) break;
        t += glm::clamp(hei, 1.0f + t * 0.1f, 50.0f);
    }
#else
    for (int i = 0; i < 128; i++)
    {
        glm::vec3  pos = ro + t * rd;
        glm::vec2  env = terrainMap(glm::vec2(pos.x, pos.z));
        float hei = pos.y - env.x;
        res = glm::min(res, 32.0f*hei / t);
        if (res<0.0001 || pos.y>kMaxHeight) break;
        t += glm::clamp(hei, 0.5f + t * 0.05f, 25.0f);
    }
#endif
    return glm::clamp(res, 0.0f, 1.0f);
}

glm::vec2 raymarchTerrain(const glm::vec3& ro, const glm::vec3& rd, float tmin, float tmax)
{
    // bounding plane
    float tp = (kMaxHeight + kMaxTreeHeight - ro.y) / rd.y;
    if (tp > 0.0) tmax = glm::min(tmax, tp);

    // raymarch
    float dis, th;
    float t2 = -1.0f;
    float t = tmin;
    float ot = t;
    float odis = 0.0f;
    float odis2 = 0.0f;
    for (int i = 0; i < 400; i++)
    {
        th = 0.001f*t;

        glm::vec3  pos = ro + t * rd;
        glm::vec2  env = terrainMap(glm::vec2(pos.x, pos.z));
        float hei = env.x;

        // tree envelope
        float dis2 = pos.y - (hei + kMaxTreeHeight * 1.1f);
        if (dis2 < th)
        {
            if (t2 < 0.0f)
            {
                t2 = ot + (th - odis2)*(t - ot) / (dis2 - odis2); // linear interpolation for better accuracy
            }
        }
        odis2 = dis2;

        // terrain
        dis = pos.y - hei;
        if (dis < th) break;

        ot = t;
        odis = dis;
        t += dis * 0.8f*(1.0f - 0.75f*env.y); // slow down in step areas
        if (t > tmax) break;
    }

    if (t > tmax) t = -1.0;
    else t = ot + (th - odis)*(t - ot) / (dis - odis); // linear interpolation for better accuracy

    return glm::vec2(t, t2);
}

//------------------------------------------------------------------------------------------
// trees
//------------------------------------------------------------------------------------------

float treesMap(const glm::vec3& p, float rt, float& oHei, float& oMat, float& oDis)
{
    oHei = 1.0f;
    oDis = 0.1f;
    oMat = 0.0f;

    auto p_xz = glm::vec2(p.x, p.z);
    float base = terrainMap(p_xz).x;

    float bb = fbm_4(p_xz *0.15f);

    float d = 10.0f;
    glm::vec2 n = floor(p_xz);
    glm::vec2 f = fract(p_xz);
    for (int j = 0; j <= 1; j++)
        for (int i = 0; i <= 1; i++)
        {
            glm::vec2  g = glm::vec2(float(i), float(j)) - glm::step(f, glm::vec2(0.5f));
            glm::vec2  o = hash2(n + g);
            glm::vec2  v = hash2(n + g + glm::vec2(13.1f, 71.7f));
            glm::vec2  r = g - f + o;

            float height = kMaxTreeHeight * (0.4f + 0.8f*v.x);
            float width = 0.5f + 0.2f*v.x + 0.3f*v.y;

            if (bb < 0.0) width *= 0.5f; else height *= 0.7f;

            auto q = glm::vec3(r.x, p.y - base - height * 0.5f, r.y);

            float k = sdEllipsoidY(q, glm::vec2(width, 0.5f*height));

            if (k < d)
            {
                d = k;
                oMat = 0.5f*hash1(n + g + 111.0f);
                if (bb > 0.0f) oMat += 0.5f;
                oHei = (p.y - base) / height;
                oHei *= 0.5f + 0.5f* glm::length(q) / width;
            }
        }

    // distort ellipsoids to make them look like trees (works only in the distance really)
#ifdef LOWQUALITY
    if (rt < 350.0)
#else
    if (rt < 500.0)
#endif
    {
        float s = fbm_4(p*3.0f);
        s = s * s;
        oDis = s;
#ifdef LOWQUALITY
        float att = 1.0f - glm::smoothstep(150.0f, 350.0f, rt);
#else
        float att = 1.0f - glm::smoothstep(200.0f, 500.0f, rt);
#endif
        d += 2.0f*s*att*att;
    }

    return d;
}

float treesShadow(const glm::vec3& ro, const glm::vec3& rd)
{
    float res = 1.0f;
    float t = 0.02f;
#ifdef LOWQUALITY
    for (int i = 0; i < 64; i++)
    {
        float kk1, kk2, kk3;
        glm::vec3 pos = ro + rd * t;
        float h = treesMap(pos, t, kk1, kk2, kk3);
        res = glm::min(res, 32.0f*h / t);
        t += h;
        if (res<0.001f || t>50.0f || pos.y > kMaxHeight + kMaxTreeHeight) break;
    }
#else
    for (int i = 0; i < 150; i++)
    {
        float kk1, kk2, kk3;
        float h = treesMap(ro + rd * t, t, kk1, kk2, kk3);
        res = glm::min(res, 32.0f*h / t);
        t += h;
        if (res<0.001 || t>120.0) break;
    }
#endif
    return glm::clamp(res, 0.0f, 1.0f);
}

glm::vec3 treesNormal(const glm::vec3& pos, float t)
{
    float kk1, kk2, kk3;
#if 0
    const float eps = 0.005;
    glm::vec2 e = glm::vec2(1.0, -1.0)*0.5773f*eps;
    auto e_xyy = glm::vec3(e.x, e.y, e.y);
    auto e_yyx = glm::vec3(e.y, e.y, e.x);
    auto e_yxy = glm::vec3(e.y, e.x, e.y);
    return glm::normalize(e_xyy*treesMap(pos + e_xyy, t, kk1, kk2, kk3) +
        e_yyx*treesMap(pos + e_yyx, t, kk1, kk2, kk3) +
        e_yxy*treesMap(pos + e_yxy, t, kk1, kk2, kk3) +
        e.x*treesMap(pos + e.x, t, kk1, kk2, kk3));
#else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    glm::vec3 n = glm::vec3(0.0);
    for (int i = 0; i < 4; i++)
    {
        glm::vec3 e = 0.5773f*(2.0f* glm::vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0f);
        n += e * treesMap(pos + 0.005f*e, t, kk1, kk2, kk3);
    }
    return normalize(n);
#endif    
}

//------------------------------------------------------------------------------------------
// sky
//------------------------------------------------------------------------------------------

glm::vec3 renderSky(const glm::vec3& ro, const glm::vec3& rd)
{
    // background sky     
    glm::vec3 col = glm::vec3(0.45, 0.6, 0.85) / 0.85f - rd.y* glm::vec3(0.4, 0.36, 0.4);

    // clouds
    float t = (1000.0f - ro.y) / rd.y;
    if (t > 0.0f)
    {
        auto tt = ro + t * rd;
        glm::vec2 uv = glm::vec2(tt.x, tt.z);
        float cl = fbm_9(uv*0.002f);
        float dl = glm::smoothstep(-0.2f, 0.6f, cl);
        col = glm::mix(col, glm::vec3(1.0), 0.4*dl);
    }

    // sun glare    
    float sun = glm::clamp(dot(kSunDir, rd), 0.0f, 1.0f);
    col += 0.6f* glm::vec3(1.0f, 0.6f, 0.3f)* glm::pow(sun, 32.0f);

    return col;
}

//------------------------------------------------------------------------------------------
// main image making function
//------------------------------------------------------------------------------------------

glm::vec4 getMainImage(const glm::vec2& frag_coord, const glm::ivec2& screen_size)
{
    static auto i_frame = 1;
    static auto i_time = 1.0f;
    glm::vec2 o = hash2(float(i_frame)) - 0.5f;

    glm::vec2 p = (2.0f*(frag_coord + o) - glm::vec2(screen_size)) / glm::vec2(float(screen_size.x));

    //----------------------------------
    // setup
    //----------------------------------

    // camera
#ifdef  STATIC_CAMERA
    glm::vec3 ro = glm::vec3(0.0, -99.25, 5.0);
    glm::vec3 ta = glm::vec3(0.0, -99.0, 0.0);
#else
    float time = 1.0f* i_time;
    glm::vec3 ro = glm::vec3(0.0, -99.25, 5.0) + glm::vec3(10.0* glm::sin(0.02*time), 0.0, -10.0* glm::sin(0.2 + 0.031*time));
    glm::vec3 ta = glm::vec3(0.0, -98.25, -45.0 + ro.z);
#endif

    // ray
    glm::mat3 ca = setCamera(ro, ta, 0.0);
    glm::vec3 rd = ca * glm::normalize(glm::vec3(p, 1.5));

    float resT = 1000.0;

    //----------------------------------
    // sky
    //----------------------------------

    glm::vec3 col = renderSky(ro, rd);


    //----------------------------------
    // raycast terrain and tree envelope
    //----------------------------------
    {
        const float tmax = 1000.0;
        float hei, mid, displa;
        int   obj = 0;
        glm::vec2 t = raymarchTerrain(ro, rd, 15.0f, tmax);
        if (t.x > 0.0)
        {
            resT = t.x;
            obj = 1;
        }

        //----------------------------------
        // raycast trees, if needed
        //----------------------------------
        if (t.y > 0.0)
        {
            float tf = t.y;
            float tfMax = (t.x > 0.0) ? t.x : tmax;
            for (int i = 0; i < 64; i++)
            {
                glm::vec3  pos = ro + tf * rd;
                float dis = treesMap(pos, tf, hei, mid, displa);
                if (dis < (0.00025*tf)) break;
                tf += dis;
                if (tf > tfMax) break;
            }
            if (tf < tfMax)
            {
                resT = tf;
                obj = 2;
            }
        }

        //----------------------------------
        // shade
        //----------------------------------
        if (obj > 0)
        {
            glm::vec3 pos = ro + resT * rd;
            glm::vec3 epos = pos + glm::vec3(0.0f, 2.4f, 0.0f);

            float sha1 = terrainShadow(pos + glm::vec3(0, 0.01f, 0), kSunDir, 0.01f);
#ifndef LOWQUALITY
            if (sha1 > 0.001)
                sha1 *= cloudsShadow(epos, kSunDir, 0.1f, 1000.0f);
#endif
#ifndef LOWQUALITY
            float sha2 = treesShadow(pos + glm::vec3(0, 0.01f, 0), kSunDir);
#endif

            glm::vec3 tnor = terrainNormal(glm::vec2(pos.x, pos.z));

            //----------------------------------
            // terrain
            //----------------------------------
            if (obj == 1)
            {
                // bump map
                auto tt = fbmd_8(pos * 0.3f * glm::vec3(1.0f, 0.2f, 1.0f));
                glm::vec3 nor = glm::normalize(tnor + 0.8f*(1.0f - abs(tnor.y))*0.8f*glm::vec3(tt.y, tt.z, tt.w));

                col = glm::vec3(0.18, 0.11, 0.10)*.75f;
                col = 1.0f* glm::mix(col, glm::vec3(0.1, 0.1, 0.0)*0.2f, glm::smoothstep(0.7f, 0.9f, nor.y));

                float dif = glm::clamp(dot(nor, kSunDir), 0.0f, 1.0f);
                dif *= sha1;
#ifndef LOWQUALITY
                dif *= sha2;
#endif

                float bac = glm::clamp(glm::dot(glm::normalize(glm::vec3(-kSunDir.x, 0.0, -kSunDir.z)), nor), 0.0f, 1.0f);
                float foc = glm::clamp((pos.y + 100.0f) / 100.0f, 0.0f, 1.0f);
                float dom = glm::clamp(0.5f + 0.5f*nor.y, 0.0f, 1.0f);
                glm::vec3  lin = 1.0f*0.2f* glm::mix(0.1f* glm::vec3(0.1, 0.2, 0.1), glm::vec3(0.7, 0.9, 1.5)*3.0f, dom)*foc;
                lin += 1.0f*8.5f* glm::vec3(1.0, 0.9, 0.8)*dif;
                lin += 1.0f*0.27f* glm::vec3(1.0)*bac*foc;

                col *= lin;
            }
            //----------------------------------
            // trees
            //----------------------------------
            else //if( obj==2 )
            {
                glm::vec3 gnor = treesNormal(pos, resT);

                glm::vec3 nor = glm::normalize(gnor + 2.5f*tnor);

                // --- lighting ---
                glm::vec3  ref = glm::reflect(rd, nor);
                float occ = glm::clamp(hei, 0.0f, 1.0f) * pow(1.0f - 2.0f*displa, 3.0f);
                float dif = glm::clamp(0.1f + 0.9f*dot(nor, kSunDir), 0.0f, 1.0f);
                dif *= sha1;
                if (dif > 0.0001)
                {
                    float a = glm::clamp(0.5f + 0.5f*dot(tnor, kSunDir), 0.0f, 1.0f);
                    a = a * a;
                    a *= occ;
                    a *= 0.6f;
                    a *= glm::smoothstep(30.0f, 100.0f, resT);
                    // tree shadows with fake transmission
#ifdef LOWQUALITY
                    float sha2 = treesShadow(pos + glm::vec3(0, 0.01, 0), kSunDir);
#endif
                    dif *= a + (1.0f - a)*sha2;
                }
                float dom = glm::clamp(0.5f + 0.5f*nor.y, 0.0f, 1.0f);
                float fre = glm::clamp(1.0f + dot(nor, rd), 0.0f, 1.0f);
                float spe = pow(glm::clamp(dot(ref, kSunDir), 0.0f, 1.0f), 9.0f)*dif*(0.2f + 0.8f*pow(fre, 5.0f))*occ;

                // --- lights ---
                glm::vec3 lin = 1.1f*0.5f* glm::mix(0.1f* glm::vec3(0.1, 0.2, 0.0), glm::vec3(0.6, 1.0, 1.0), dom*occ);
#ifdef SOFTTREES
                lin += 1.3f*15.0f* glm::vec3(1.0, 0.9, 0.8)*dif*occ;
#else
                lin += 1.3f*10.0f* glm::vec3(1.0, 0.9, 0.8)*dif*occ;
#endif
                lin += 1.1f* glm::vec3(0.9, 1.0, 0.8)*pow(fre, 5.0f)*occ;
                lin += 0.06f* glm::vec3(0.15, 0.4, 0.1)*occ;

                // --- material ---
                float brownAreas = fbm_4(glm::vec2(pos.z, pos.x)*0.03f);
                col = glm::vec3(0.08, 0.08, 0.02)*0.45f;
                col = glm::mix(col, glm::vec3(0.13, 0.08, 0.02)*0.45f, glm::smoothstep(0.2f, 0.9f, glm::fract(2.0f*mid)));
                col *= (mid < 0.5) ? 0.55 : 1.0;
                col = glm::mix(col, glm::vec3(0.25, 0.16, 0.01)*0.15f, 0.7f* glm::smoothstep(0.1f, 0.3f, brownAreas)* glm::smoothstep(0.5f, 0.8f, tnor.y));
                col *= 2.0*1.64;

                // --- brdf * material ---
                col *= lin;
                col += spe * 1.2f* glm::vec3(1.0, 1.1, 2.5);
            }
            col = fog(col, resT);
        }
    }

    //----------------------------------
    // clouds
    //----------------------------------
    {
        glm::vec4 res = renderClouds(ro, rd, 0.0, resT, resT);
        col = col * (1.0f - res.w) + glm::vec3(res.x, res.y, res.z);
    }

    //----------------------------------
    // final
    //----------------------------------

    // sun glare    
    float sun = glm::clamp(dot(kSunDir, rd), 0.0f, 1.0f);
    col += 0.25f* glm::vec3(0.8, 0.4, 0.2)*pow(sun, 4.0f);


    // gamma
    col = sqrt(glm::clamp(col, 0.0f, 1.0f));

    // contrast
    col = col * col*(3.0f - 2.0f*col);
    // color grade    
    col = glm::pow(col, glm::vec3(1.0f, 0.92f, 1.0f));   // soft green
    col *= glm::vec3(1.02f, 0.99f, 0.99f);            // tint red
    col.z = (col.z + 0.1f) / 1.1f;                // bias blue
    //col = mix( col, col.yyy, 0.1 );       // desaturate

    return glm::vec4(col, 1.0f);
    //------------------------------------------
    // reproject from previous frame and average
    //------------------------------------------
#if 0
    mat4 oldCam = mat4(textureLod(iChannel0, vec2(0.5, 0.5) / iResolution.xy, 0.0),
        textureLod(iChannel0, vec2(1.5, 0.5) / iResolution.xy, 0.0),
        textureLod(iChannel0, vec2(2.5, 0.5) / iResolution.xy, 0.0),
        0.0, 0.0, 0.0, 1.0);

    // world space
    vec4 wpos = vec4(ro + rd * resT, 1.0);
    // camera space
    vec3 cpos = (wpos*oldCam).xyz; // note inverse multiply
    // ndc space
    vec2 npos = 1.5 * cpos.xy / cpos.z;
    // screen space
    vec2 spos = 0.5 + 0.5*npos*vec2(iResolution.y / iResolution.x, 1.0);
    // undo dither
    spos -= o / iResolution.xy;
    // raster space
    vec2 rpos = spos * iResolution.xy;

    if (rpos.y < 1.0 && rpos.x < 3.0)
    {
    }
    else
    {
        vec3 ocol = textureLod(iChannel0, spos, 0.0).xyz;
        if (iFrame == 0) ocol = col;
        col = mix(ocol, col, 0.1);
    }

    //----------------------------------

    if (fragCoord.y < 1.0 && fragCoord.x < 3.0)
    {
        if (abs(fragCoord.x - 2.5) < 0.5) fragColor = vec4(ca[2], -dot(ca[2], ro));
        if (abs(fragCoord.x - 1.5) < 0.5) fragColor = vec4(ca[1], -dot(ca[1], ro));
        if (abs(fragCoord.x - 0.5) < 0.5) fragColor = vec4(ca[0], -dot(ca[0], ro));
    }
    else
    {
        fragColor = vec4(col, 1.0);
    }
#endif
}

#endif

std::vector<uint32_t> generateTileMeshIndex(const uint32_t& segment_count) {
    std::vector<uint32_t> index_buffer;
    index_buffer.resize(segment_count * segment_count * 6);
    uint32_t* p_index_buffer = index_buffer.data();
    const uint32_t line_index_count = segment_count + 1;
    for (uint32_t y = 0; y < segment_count; y++) {
        for (uint32_t x = 0; x < segment_count; x++) {
            uint32_t i00 = y * line_index_count + x;
            uint32_t i01 = i00 + 1;
            uint32_t i10 = i00 + line_index_count;
            uint32_t i11 = i01 + line_index_count;
            *p_index_buffer++ = i00;
            *p_index_buffer++ = i10;
            *p_index_buffer++ = i01;
            *p_index_buffer++ = i10;
            *p_index_buffer++ = i11;
            *p_index_buffer++ = i01;
        }
    }

    uint32_t index_buffer_size = static_cast<uint32_t>(p_index_buffer - index_buffer.data());
    assert(index_buffer_size == segment_count * segment_count * 6);

    return index_buffer;
}

namespace engine {
namespace game_object {
namespace {

renderer::WriteDescriptorList addTileCreatorBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& heightmap_tex,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& grass_snow_layer) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_DEPTH_TEX_INDEX,
        texture_sampler,
        heightmap_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        ROCK_LAYER_BUFFER_INDEX,
        nullptr,
        rock_layer.view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        soil_water_layer.view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        nullptr,
        grass_snow_layer.view,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

renderer::WriteDescriptorList addTileUpdateBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& dst_water_normal) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        soil_water_layer.view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        DST_WATER_NORMAL_BUFFER_INDEX,
        nullptr,
        dst_water_normal.view,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

renderer::WriteDescriptorList addTileFlowUpdateBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& dst_soil_water_layer,
    const renderer::TextureInfo& dst_water_flow) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        soil_water_layer.view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        DST_SOIL_WATER_LAYER_BUFFER_INDEX,
        nullptr,
        dst_soil_water_layer.view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        DST_WATER_FLOW_BUFFER_INDEX,
        nullptr,
        dst_water_flow.view,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

renderer::WriteDescriptorList addTileResourceTextures(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& clamp_texture_sampler,
    const std::shared_ptr<renderer::Sampler>& repeat_texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_texture,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& rock_layer,
    const std::shared_ptr<renderer::ImageView>& soil_water_layer,
    const std::shared_ptr<renderer::ImageView>& grass_snow_layer,
    const std::shared_ptr<renderer::ImageView>& water_normal,
    const std::shared_ptr<renderer::ImageView>& water_flow,
    const std::shared_ptr<renderer::ImageView>& temp_tex,
    const std::shared_ptr<renderer::ImageView>& map_mask_tex,
    const std::shared_ptr<renderer::ImageView>& detail_noise_tex,
    const std::shared_ptr<renderer::ImageView>& rough_noise_tex) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(11);

    // src color.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_COLOR_TEX_INDEX,
        clamp_texture_sampler,
        src_texture,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // src depth.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_DEPTH_TEX_INDEX,
        clamp_texture_sampler,
        src_depth,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        clamp_texture_sampler,
        rock_layer,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        clamp_texture_sampler,
        soil_water_layer,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        clamp_texture_sampler,
        grass_snow_layer,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        WATER_NORMAL_BUFFER_INDEX,
        clamp_texture_sampler,
        water_normal,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        WATER_FLOW_BUFFER_INDEX,
        clamp_texture_sampler,
        water_flow,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEMP_TEX_INDEX,
        clamp_texture_sampler,
        temp_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_MAP_MASK_INDEX,
        clamp_texture_sampler,
        map_mask_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DETAIL_NOISE_TEXTURE_INDEX,
        repeat_texture_sampler,
        detail_noise_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROUGH_NOISE_TEXTURE_INDEX,
        repeat_texture_sampler,
        rough_noise_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static renderer::ShaderModuleList getTileShaderModules(
    std::shared_ptr<renderer::Device> device,
    const std::string& vs_name,
    const std::string& ps_name,
    const std::string& gs_name = "") {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            vs_name,
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            ps_name,
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

    if (gs_name.size() > 0) {
        shader_modules.push_back(
            renderer::helper::loadShaderModule(
                device,
                gs_name,
                renderer::ShaderStageFlagBits::GEOMETRY_BIT,
                std::source_location::current()));
    }
    return shader_modules;
}

static renderer::ShaderModuleList getTileMeshShaderModules(
    std::shared_ptr<renderer::Device> device,
    const std::string& ms_name,
    const std::string& ps_name,
    const std::string& ts_name = "") {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            ms_name,
            renderer::ShaderStageFlagBits::MESH_BIT_EXT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            ps_name,
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    if (ts_name.size() > 0) {
        shader_modules.push_back(
            renderer::helper::loadShaderModule(
                device,
                ts_name,
                renderer::ShaderStageFlagBits::TASK_BIT_EXT,
                std::source_location::current()));
    }
    return shader_modules;
}


static std::shared_ptr<renderer::DescriptorSetLayout> createTileCreateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(4);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SRC_DEPTH_TEX_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[3] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createTileUpdateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        DST_WATER_NORMAL_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> createTileFlowUpdateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(4);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        DST_SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[3] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        DST_WATER_FLOW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_IMAGE);  
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::DescriptorSetLayout> CreateTileResourceDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(11);
    bindings[0] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_COLOR_TEX_INDEX);
    bindings[1] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_DEPTH_TEX_INDEX);
    bindings[2] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT) |
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[3] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT) |
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[4] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ORTHER_INFO_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[5] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        WATER_NORMAL_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[6] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        WATER_FLOW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[7] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_TEMP_TEX_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[8] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_MAP_MASK_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[9] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        DETAIL_NOISE_TEXTURE_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[10] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROUGH_NOISE_TEXTURE_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createTileCreatorPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileCreateParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createTileUpdatePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileUpdateParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createTileFlowUpdatePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileUpdateParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createTilePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::PipelineLayout> createTileGrassPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
#if USE_MESH_SHADER
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) |
        SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT);
#else
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) |
        SET_FLAG_BIT(ShaderStage, GEOMETRY_BIT);
#endif
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::TileParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static std::shared_ptr<renderer::Pipeline> createTilePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size,
    const std::string& vs_name,
    const std::string& ps_name) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    auto shader_modules =
        getTileShaderModules(device, vs_name, ps_name);
    auto pipeline = device->createPipeline(
        render_pass,
        pipeline_layout,
        {},
        {},
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size,
        std::source_location::current());

    return pipeline;
}

size_t generateHash(
    const glm::vec2& min,
    const glm::vec2& max,
    const uint32_t& segment_count) {
    size_t hash;
    hash = std::hash<float>{}(min.x);
    hash_combine(hash, min.y);
    hash_combine(hash, max.x);
    hash_combine(hash, max.y);
    hash_combine(hash, segment_count);
    return hash;
}

static std::shared_ptr<renderer::Pipeline> createGrassPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {

#if    USE_MESH_SHADER
    auto shader_modules =
        getTileMeshShaderModules(
            device,
            "grass/grass_mesh.spv",
            "grass/grass_frag.spv");
#else
    auto shader_modules =
        getTileShaderModules(
            device,
            "grass/grass_vert.spv",
            "grass/grass_frag.spv",
            "grass/grass_geom.spv");
#endif

    renderer::PipelineInputAssemblyStateCreateInfo topology_info;
    topology_info.restart_enable = false;
    topology_info.topology = renderer::PrimitiveTopology::POINT_LIST;

    std::vector<renderer::VertexInputBindingDescription> binding_descs;
    std::vector<renderer::VertexInputAttributeDescription> attribute_descs;

    renderer::VertexInputBindingDescription vert_desc = {};
    vert_desc.binding = VINPUT_VERTEX_BINDING_POINT;
    vert_desc.stride = sizeof(glm::vec3);
    vert_desc.input_rate = renderer::VertexInputRate::VERTEX;
    binding_descs.push_back(vert_desc);

    renderer::VertexInputBindingDescription instance_desc;
    instance_desc.binding = VINPUT_INSTANCE_BINDING_POINT;
    instance_desc.input_rate = renderer::VertexInputRate::INSTANCE;
    instance_desc.stride = sizeof(glsl::GrassInstanceDataInfo);
    binding_descs.push_back(instance_desc);

    renderer::VertexInputAttributeDescription vert_attr;
    vert_attr.binding = VINPUT_VERTEX_BINDING_POINT;
    vert_attr.buffer_offset = 0;
    vert_attr.format = renderer::Format::R32G32B32_SFLOAT;
    vert_attr.buffer_view = 0;
    vert_attr.location = VINPUT_POSITION;
    vert_attr.offset = offsetof(glsl::GrassInstanceDataInfo, mat_rot_0);
    attribute_descs.push_back(vert_attr);

    renderer::VertexInputAttributeDescription attr;
    attr.binding = VINPUT_INSTANCE_BINDING_POINT;
    attr.buffer_offset = 0;
    attr.format = renderer::Format::R32G32B32_SFLOAT;
    attr.buffer_view = 0;
    attr.location = IINPUT_MAT_ROT_0;
    attr.offset = offsetof(glsl::GrassInstanceDataInfo, mat_rot_0);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_1;
    attr.offset = offsetof(glsl::GrassInstanceDataInfo, mat_rot_1);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_2;
    attr.offset = offsetof(glsl::GrassInstanceDataInfo, mat_rot_2);
    attribute_descs.push_back(attr);
    attr.format = renderer::Format::R32G32B32A32_SFLOAT;
    attr.location = IINPUT_MAT_POS_SCALE;
    attr.offset = offsetof(glsl::GrassInstanceDataInfo, mat_pos_scale);
    attribute_descs.push_back(attr);

    auto grass_pipeline = device->createPipeline(
        render_pass,
        pipeline_layout,
        binding_descs,
        attribute_descs,
        topology_info,
        graphic_pipeline_info,
        shader_modules,
        display_size,
        std::source_location::current());

    return grass_pipeline;
}


} // namespace

// static member definition.
std::unordered_map<size_t, std::shared_ptr<TileObject>> TileObject::tile_meshes_;
std::vector<std::shared_ptr<TileObject>> TileObject::visible_tiles_;
std::vector<uint32_t> TileObject::available_block_indexes_;
renderer::TextureInfo TileObject::rock_layer_;
renderer::TextureInfo TileObject::soil_water_layer_[2];
renderer::TextureInfo TileObject::grass_snow_layer_;
renderer::TextureInfo TileObject::water_normal_;
renderer::TextureInfo TileObject::water_flow_;
std::shared_ptr<renderer::DescriptorSet> TileObject::creator_buffer_desc_set_;
std::shared_ptr<renderer::DescriptorSet> TileObject::tile_update_buffer_desc_set_[2];
std::shared_ptr<renderer::DescriptorSet> TileObject::tile_flow_update_buffer_desc_set_[2];
std::shared_ptr<renderer::DescriptorSetLayout> TileObject::tile_creator_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> TileObject::tile_creator_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> TileObject::tile_creator_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> TileObject::tile_update_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> TileObject::tile_update_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> TileObject::tile_update_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> TileObject::tile_flow_update_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> TileObject::tile_flow_update_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> TileObject::tile_flow_update_pipeline_;
std::shared_ptr<renderer::PipelineLayout> TileObject::tile_pipeline_layout_;
std::shared_ptr<renderer::PipelineLayout> TileObject::tile_grass_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> TileObject::tile_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> TileObject::tile_res_desc_set_layout_;
std::shared_ptr<renderer::DescriptorSet> TileObject::tile_res_desc_set_[2];
std::shared_ptr<renderer::Pipeline> TileObject::tile_water_pipeline_;
std::shared_ptr<renderer::Pipeline> TileObject::tile_grass_pipeline_;

TileObject::TileObject(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    const glm::vec2& min,
    const glm::vec2& max,
    const size_t& hash_value,
    const uint32_t& block_idx) :
    min_(min),
    max_(max),
    hash_(hash_value),
    block_idx_(block_idx){
    createMeshBuffers(device);
    createGrassBuffers(device);
    assert(tile_creator_desc_set_layout_);
    assert(tile_update_desc_set_layout_);
    assert(tile_flow_update_desc_set_layout_);
    assert(tile_res_desc_set_layout_);
}

void TileObject::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    index_buffer_.destroy(device);
    grass_vertex_buffer_.destroy(device);
    grass_index_buffer_.destroy(device);
    grass_indirect_draw_cmd_.destroy(device);
    grass_instance_buffer_.destroy(device);
}

std::shared_ptr<TileObject> TileObject::addOneTile(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    const glm::vec2& min,
    const glm::vec2& max) {
    auto segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);
    auto hash_value = generateHash(min, max, segment_count);
    auto result = tile_meshes_.find(hash_value);
    if (result == tile_meshes_.end()) {
        if (available_block_indexes_.size() > 0) {
            auto block_index = available_block_indexes_.back();
            available_block_indexes_.pop_back();
            tile_meshes_[hash_value] =
                std::make_shared<TileObject>(
                    device,
                    descriptor_pool,
                    min,
                    max,
                    hash_value,
                    block_index);
        }
        else {
            assert(0);
        }
    }
    else {
        auto min_t = result->second->getMin();
        auto max_t = result->second->getMax();

        assert(min_t.x == min.x && min_t.y == min.y && max_t.x == max.x && max_t.y == max.y);
    }

    return tile_meshes_[hash_value];
}


const renderer::TextureInfo& TileObject::getRockLayer() {
    return rock_layer_;
}

const renderer::TextureInfo& TileObject::getSoilWaterLayer(int idx) {
    return soil_water_layer_[idx];
}

const renderer::TextureInfo& TileObject::getWaterFlow() {
    return water_flow_;
}

glm::vec2 TileObject::getWorldMin() {
    return glm::vec2(-kWorldMapSize / 2.0f);
}
glm::vec2 TileObject::getWorldRange() {
    return glm::vec2(kWorldMapSize);
}

float TileObject::getMinDistToCamera(const glm::vec2& camera_pos) {
    auto center = (getMin() + getMax()) * 0.5f;
    return glm::length(center - camera_pos);
}

void TileObject::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& water_render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::GraphicPipelineInfo& graphic_double_face_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    if (tile_creator_pipeline_layout_ == nullptr) {
        assert(tile_creator_desc_set_layout_);
        tile_creator_pipeline_layout_ =
            createTileCreatorPipelineLayout(
                device,
                { tile_creator_desc_set_layout_ });
    }

    if (tile_creator_pipeline_ == nullptr) {
        assert(tile_creator_pipeline_layout_);
        tile_creator_pipeline_ =
            renderer::helper::createComputePipeline(
                device,
                tile_creator_pipeline_layout_,
                "terrain/tile_creator_comp.spv",
                std::source_location::current());
    }

    if (tile_update_pipeline_layout_ == nullptr) {
        assert(tile_update_desc_set_layout_);
        tile_update_pipeline_layout_ =
            createTileUpdatePipelineLayout(
                device,
                { tile_update_desc_set_layout_ });
    }

    if (tile_update_pipeline_ == nullptr) {
        assert(tile_update_pipeline_layout_);
        tile_update_pipeline_ =
            renderer::helper::createComputePipeline(
                device,
                tile_update_pipeline_layout_,
                "terrain/tile_update_comp.spv",
                std::source_location::current());
    }

    if (tile_flow_update_pipeline_layout_ == nullptr) {
        assert(tile_flow_update_desc_set_layout_);
        tile_flow_update_pipeline_layout_ =
            createTileFlowUpdatePipelineLayout(
                device,
                { tile_flow_update_desc_set_layout_ });
    }

    if (tile_flow_update_pipeline_ == nullptr) {
        assert(tile_flow_update_pipeline_layout_);
        tile_flow_update_pipeline_ =
            renderer::helper::createComputePipeline(
                device,
                tile_flow_update_pipeline_layout_,
                "terrain/tile_flow_update_comp.spv",
                std::source_location::current());
    }

    auto desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(tile_res_desc_set_layout_);

    if (tile_pipeline_layout_ == nullptr) {
        tile_pipeline_layout_ =
            createTilePipelineLayout(
                device,
                desc_set_layouts);
    }

    if (tile_pipeline_ == nullptr) {
        assert(tile_pipeline_layout_);
        tile_pipeline_ =
            createTilePipeline(
                device,
                render_pass,
                tile_pipeline_layout_,
                graphic_pipeline_info,
                display_size,
                "terrain/tile_soil_vert.spv",
                "terrain/tile_frag.spv");
    }

    if (tile_water_pipeline_ == nullptr) {
        assert(tile_pipeline_layout_);
        tile_water_pipeline_ =
            createTilePipeline(
                device,
                water_render_pass,
                tile_pipeline_layout_,
                graphic_pipeline_info,
                display_size,
                "terrain/tile_water_vert.spv",
                "terrain/tile_water_frag.spv");
    }

    if (tile_grass_pipeline_layout_ == nullptr) {
        tile_grass_pipeline_layout_ =
            createTileGrassPipelineLayout(
                device,
                desc_set_layouts);
    }

    if (tile_grass_pipeline_ == nullptr) {
        assert(tile_grass_pipeline_layout_);
        tile_grass_pipeline_ =
            createGrassPipeline(
                device,
                render_pass,
                tile_grass_pipeline_layout_,
                graphic_double_face_pipeline_info,
                display_size);
    }
}

void TileObject::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& water_render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::GraphicPipelineInfo& graphic_double_face_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16_SFLOAT,
        glm::uvec2(static_cast<uint32_t>(TileConst::kRockLayerSize)),
        rock_layer_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    for (int i = 0; i < 2; i++) {
        renderer::Helper::create2DTextureImage(
            device,
            renderer::Format::R16G16_UNORM,
            glm::uvec2(static_cast<uint32_t>(TileConst::kSoilLayerSize)),
            soil_water_layer_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R8G8_UNORM,
        glm::uvec2(static_cast<uint32_t>(TileConst::kGrassSnowLayerSize)),
        grass_snow_layer_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R8G8_SNORM,
        glm::uvec2(static_cast<uint32_t>(TileConst::kWaterlayerSize)),
        water_normal_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16_SFLOAT,
        glm::uvec2(static_cast<uint32_t>(TileConst::kWaterlayerSize)),
        water_flow_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    auto num_cache_blocks = static_cast<uint32_t>(TileConst::kNumCachedBlocks);
    available_block_indexes_.resize(num_cache_blocks);
    for (uint32_t i = 0; i < num_cache_blocks; i++) {
        available_block_indexes_[i] = i;
    }

    if (tile_creator_desc_set_layout_ == nullptr) {
        tile_creator_desc_set_layout_ =
            createTileCreateDescriptorSetLayout(device);
    }

    if (tile_update_desc_set_layout_ == nullptr) {
        tile_update_desc_set_layout_ =
            createTileUpdateDescriptorSetLayout(device);
    }

    if (tile_flow_update_desc_set_layout_ == nullptr) {
        tile_flow_update_desc_set_layout_ =
            createTileFlowUpdateDescriptorSetLayout(device);
    }

    if (tile_res_desc_set_layout_ == nullptr) {
        tile_res_desc_set_layout_ =
            CreateTileResourceDescriptorSetLayout(device);
    }

    createStaticMembers(
        device,
        render_pass,
        water_render_pass,
        graphic_pipeline_info,
        graphic_double_face_pipeline_info,
        global_desc_set_layouts,
        display_size);
}

void TileObject::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& water_render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::GraphicPipelineInfo& graphic_double_face_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    if (tile_creator_pipeline_layout_) {
        device->destroyPipelineLayout(tile_creator_pipeline_layout_);
        tile_creator_pipeline_layout_ = nullptr;
    }

    if (tile_creator_pipeline_) {
        device->destroyPipeline(tile_creator_pipeline_);
        tile_creator_pipeline_ = nullptr;
    }

    if (tile_update_pipeline_layout_) {
        device->destroyPipelineLayout(tile_update_pipeline_layout_);
        tile_update_pipeline_layout_ = nullptr;
    }

    if (tile_update_pipeline_) {
        device->destroyPipeline(tile_update_pipeline_);
        tile_update_pipeline_ = nullptr;
    }

    if (tile_flow_update_pipeline_layout_) {
        device->destroyPipelineLayout(tile_flow_update_pipeline_layout_);
        tile_flow_update_pipeline_layout_ = nullptr;
    }

    if (tile_flow_update_pipeline_) {
        device->destroyPipeline(tile_flow_update_pipeline_);
        tile_flow_update_pipeline_ = nullptr;
    }

    if (tile_pipeline_layout_) {
        device->destroyPipelineLayout(tile_pipeline_layout_);
        tile_pipeline_layout_ = nullptr;
    }

    if (tile_grass_pipeline_layout_) {
        device->destroyPipelineLayout(tile_grass_pipeline_layout_);
        tile_grass_pipeline_layout_ = nullptr;
    }

    if (tile_pipeline_) {
        device->destroyPipeline(tile_pipeline_);
        tile_pipeline_ = nullptr;
    }

    if (tile_water_pipeline_) {
        device->destroyPipeline(tile_water_pipeline_);
        tile_water_pipeline_ = nullptr;
    }

    if (tile_grass_pipeline_) {
        device->destroyPipeline(tile_grass_pipeline_);
        tile_grass_pipeline_ = nullptr;
    }

    createStaticMembers(
        device,
        render_pass,
        water_render_pass,
        graphic_pipeline_info,
        graphic_double_face_pipeline_info,
        global_desc_set_layouts,
        display_size);
}

void TileObject::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(tile_creator_desc_set_layout_);
    device->destroyDescriptorSetLayout(tile_update_desc_set_layout_);
    device->destroyDescriptorSetLayout(tile_flow_update_desc_set_layout_);
    device->destroyDescriptorSetLayout(tile_res_desc_set_layout_);
    device->destroyPipelineLayout(tile_creator_pipeline_layout_);
    device->destroyPipelineLayout(tile_update_pipeline_layout_);
    device->destroyPipelineLayout(tile_flow_update_pipeline_layout_);
    device->destroyPipeline(tile_creator_pipeline_);
    device->destroyPipeline(tile_update_pipeline_);
    device->destroyPipeline(tile_flow_update_pipeline_);
    device->destroyPipelineLayout(tile_pipeline_layout_);
    device->destroyPipelineLayout(tile_grass_pipeline_layout_);
    device->destroyPipeline(tile_pipeline_);
    device->destroyPipeline(tile_water_pipeline_);
    device->destroyPipeline(tile_grass_pipeline_);
    rock_layer_.destroy(device);
    soil_water_layer_[0].destroy(device);
    soil_water_layer_[1].destroy(device);
    grass_snow_layer_.destroy(device);
    water_normal_.destroy(device);
    water_flow_.destroy(device);
}

void TileObject::createMeshBuffers(
    const std::shared_ptr<renderer::Device>& device) {
    auto segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);
    auto index_buffer = generateTileMeshIndex(segment_count);
    auto index_buffer_size = static_cast<uint32_t>(sizeof(index_buffer[0]) * index_buffer.size());
    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        index_buffer_.buffer,
        index_buffer_.memory,
        std::source_location::current(),
        index_buffer_size,
        index_buffer.data());
}

void addQuadIndex(
    std::vector<uint16_t>& index_list,
    uint32_t a,
    uint32_t b,
    uint32_t c,
    uint32_t d) {
    index_list.push_back(c);
    index_list.push_back(b);
    index_list.push_back(a);
    index_list.push_back(a);
    index_list.push_back(d);
    index_list.push_back(c);
}

void TileObject::createGrassBuffers(
    const std::shared_ptr<renderer::Device>& device) {

    glm::vec3 vertex_pos = glm::vec3(0);
    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        grass_vertex_buffer_.buffer,
        grass_vertex_buffer_.memory,
        std::source_location::current(),
        sizeof(glm::vec3),
        &vertex_pos);

    uint16_t index_list = 0;

    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        grass_index_buffer_.buffer,
        grass_index_buffer_.memory,
        std::source_location::current(),
        sizeof(uint16_t),
        &index_list);

    std::vector<renderer::DrawIndexedIndirectCommand> indirect_draw_cmd_buffer(1);
    indirect_draw_cmd_buffer[0].index_count = 1;
    indirect_draw_cmd_buffer[0].vertex_offset = 0;
    indirect_draw_cmd_buffer[0].first_index = 0;
    indirect_draw_cmd_buffer[0].first_instance = 0;
    indirect_draw_cmd_buffer[0].instance_count = static_cast<uint32_t>(TileConst::kMaxNumGrass);

    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, INDIRECT_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        grass_indirect_draw_cmd_.buffer,
        grass_indirect_draw_cmd_.memory,
        std::source_location::current(),
        indirect_draw_cmd_buffer.size() * sizeof(renderer::DrawIndexedIndirectCommand),
        indirect_draw_cmd_buffer.data());

    device->createBuffer(
        static_cast<uint32_t>(TileConst::kMaxNumGrass) * sizeof(glsl::GrassInstanceDataInfo),
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        grass_instance_buffer_.buffer,
        grass_instance_buffer_.memory,
        std::source_location::current());
}

void TileObject::generateStaticDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& heightmap_tex) {
    // tile creator buffer set.
    creator_buffer_desc_set_ = device->createDescriptorSets(
        descriptor_pool, tile_creator_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    // all world map buffer only create once, so always pick the first one.
    auto texture_descs = addTileCreatorBuffers(
        creator_buffer_desc_set_,
        texture_sampler,
        heightmap_tex,
        rock_layer_,
        soil_water_layer_[0], 
        grass_snow_layer_);
    device->updateDescriptorSets(texture_descs);

    // tile creator buffer set.
    for (int dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        tile_update_buffer_desc_set_[dbuf_idx] = device->createDescriptorSets(
            descriptor_pool, tile_update_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        texture_descs = addTileUpdateBuffers(
            tile_update_buffer_desc_set_[dbuf_idx],
            texture_sampler,
            rock_layer_,
            soil_water_layer_[dbuf_idx],
            water_normal_);
        device->updateDescriptorSets(texture_descs);

        tile_flow_update_buffer_desc_set_[dbuf_idx] = device->createDescriptorSets(
            descriptor_pool, tile_flow_update_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        texture_descs = addTileFlowUpdateBuffers(
            tile_flow_update_buffer_desc_set_[dbuf_idx],
            texture_sampler,
            rock_layer_,
            soil_water_layer_[1 - dbuf_idx],
            soil_water_layer_[dbuf_idx],
            water_flow_);
        device->updateDescriptorSets(texture_descs);

        // tile params set.
        tile_res_desc_set_[dbuf_idx] = device->createDescriptorSets(
            descriptor_pool, tile_res_desc_set_layout_, 1)[0];
    }
}

void TileObject::updateStaticDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& clamp_texture_sampler,
    const std::shared_ptr<renderer::Sampler>& repeat_texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_texture,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_tex,
    const std::shared_ptr<renderer::ImageView>& heightmap_tex,
    const std::shared_ptr<renderer::ImageView>& map_mask_tex,
    const std::shared_ptr<renderer::ImageView>& detail_volume_noise_tex,
    const std::shared_ptr<renderer::ImageView>& rough_volume_noise_tex) {

    if (tile_res_desc_set_[0] == nullptr) {
        generateStaticDescriptorSet(
            device,
            descriptor_pool,
            clamp_texture_sampler,
            heightmap_tex);
    }

    for (int dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        // create a global ibl texture descriptor set.
        auto tile_res_descs = addTileResourceTextures(
            tile_res_desc_set_[dbuf_idx],
            clamp_texture_sampler,
            repeat_texture_sampler,
            src_texture,
            src_depth,
            rock_layer_.view,
            soil_water_layer_[dbuf_idx].view,
            grass_snow_layer_.view,
            water_normal_.view,
            water_flow_.view,
            temp_tex[dbuf_idx],
            map_mask_tex,
            detail_volume_noise_tex,
            rough_volume_noise_tex);
        device->updateDescriptorSets(tile_res_descs);
    }
}

bool TileObject::validTileBySize(
    const glm::ivec2& min_tile_idx,
    const glm::ivec2& max_tile_idx,
    const float& tile_size) {

    glm::ivec2 tile_index =
        glm::ivec2((min_ + max_) * 0.5f * glm::vec2(1.0f / tile_size));

    return
        (tile_index.x >= min_tile_idx.x && tile_index.x <= max_tile_idx.x) &&
        (tile_index.y >= min_tile_idx.y && tile_index.y <= max_tile_idx.y);
}

void TileObject::generateTileBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        {rock_layer_.image,
         soil_water_layer_[0].image,
         grass_snow_layer_.image});

    auto dispatch_count = static_cast<uint32_t>(TileConst::kWaterlayerSize);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_creator_pipeline_);
    glsl::TileCreateParams tile_params = {};
    tile_params.world_min = getWorldMin();
    tile_params.world_range = getWorldRange();
    tile_params.width_pixel_count = dispatch_count;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_creator_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_creator_pipeline_layout_,
        { creator_buffer_desc_set_ });

    cmd_buf->dispatch(
        (dispatch_count + 7) / 8,
        (dispatch_count + 7) / 8, 1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { rock_layer_.image,
         soil_water_layer_[0].image,
         grass_snow_layer_.image });
}

void TileObject::updateTileBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    float current_time,
    int dbuf_idx) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        {soil_water_layer_[dbuf_idx].image,
         water_normal_.image});

    auto dispatch_count = static_cast<uint32_t>(TileConst::kWaterlayerSize);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_update_pipeline_);
    glsl::TileUpdateParams tile_params = {};
    tile_params.world_min = getWorldMin();
    tile_params.world_range = getWorldRange();
    tile_params.width_pixel_count = dispatch_count;
    tile_params.inv_width_pixel_count = 1.0f / dispatch_count;
    tile_params.range_per_pixel = tile_params.world_range * tile_params.inv_width_pixel_count;
    tile_params.current_time = current_time;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_update_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_update_pipeline_layout_,
        { tile_update_buffer_desc_set_[dbuf_idx] });

    cmd_buf->dispatch(
        (dispatch_count + 15) / 16,
        (dispatch_count + 15) / 16, 1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        {soil_water_layer_[dbuf_idx].image,
         water_normal_.image});
}

void TileObject::updateTileFlowBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    float current_time,
    int dbuf_idx) {

    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        {soil_water_layer_[dbuf_idx].image,
         soil_water_layer_[1 - dbuf_idx].image,
         water_flow_.image});

    auto dispatch_count = static_cast<uint32_t>(TileConst::kWaterlayerSize);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_flow_update_pipeline_);
    glsl::TileUpdateParams tile_params = {};
    tile_params.world_min = getWorldMin();
    tile_params.world_range = getWorldRange();
    tile_params.width_pixel_count = dispatch_count;
    tile_params.inv_width_pixel_count = 1.0f / dispatch_count;
    tile_params.range_per_pixel = tile_params.world_range * tile_params.inv_width_pixel_count;
    tile_params.flow_speed_factor = glm::vec2(1.0f);
    tile_params.current_time = current_time;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_flow_update_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_flow_update_pipeline_layout_,
        { tile_flow_update_buffer_desc_set_[dbuf_idx] });

    cmd_buf->dispatch(
        (dispatch_count + 15) / 16,
        (dispatch_count + 15) / 16, 1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        {soil_water_layer_[dbuf_idx].image,
         soil_water_layer_[1 - dbuf_idx].image,
         water_flow_.image });
}

void TileObject::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const glm::uvec2 display_size,
    int dbuf_idx,
    float delta_t,
    float cur_time,
    bool is_base_pass) {
    auto segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        is_base_pass ? tile_pipeline_ : tile_water_pipeline_);
    cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT32);

    glsl::TileParams tile_params = {};
    tile_params.world_min = getWorldMin();
    tile_params.inv_world_range = 1.0f / getWorldRange();
    tile_params.min = min_;
    tile_params.range = max_ - min_;
    tile_params.segment_count = segment_count;
    tile_params.inv_segment_count = 1.0f / segment_count;
    tile_params.offset = 0;
    tile_params.inv_screen_size = glm::vec2(1.0f / display_size.x, 1.0f / display_size.y);
    tile_params.delta_t = delta_t;
    tile_params.time = cur_time;
    tile_params.tile_index = block_idx_;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | 
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        tile_pipeline_layout_, 
        &tile_params, 
        sizeof(tile_params));

    auto new_desc_sets = desc_set_list;
    new_desc_sets.push_back(tile_res_desc_set_[dbuf_idx]);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        tile_pipeline_layout_, 
        new_desc_sets);

    cmd_buf->drawIndexed(segment_count * segment_count * 6);
}

void TileObject::drawGrass(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const glm::vec2& camera_pos,
    const glm::uvec2& display_size,
    int dbuf_idx,
    float delta_t,
    float cur_time) {
    auto segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);

    auto ratio = glm::clamp((getMinDistToCamera(camera_pos) - 256.0f) / 1024.0f, 0.0f, 1.0f);
    auto min_num_grass = static_cast<float>(TileConst::kMinNumGrass);
    auto max_num_grass = static_cast<float>(TileConst::kMaxNumGrass);
    auto num_grass = max_num_grass * (1.0f - ratio) + min_num_grass * ratio;

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, tile_grass_pipeline_);
    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = grass_instance_buffer_.buffer;
    offsets[0] = 0;
    cmd_buf->bindVertexBuffers(VINPUT_INSTANCE_BINDING_POINT, buffers, offsets);

    buffers[0] = grass_vertex_buffer_.buffer;
    cmd_buf->bindVertexBuffers(VINPUT_VERTEX_BINDING_POINT, buffers, offsets);

    cmd_buf->bindIndexBuffer(grass_index_buffer_.buffer, 0, renderer::IndexType::UINT16);

    glsl::TileParams tile_params = {};
    tile_params.world_min = getWorldMin();
    tile_params.inv_world_range = 1.0f / getWorldRange();
    tile_params.min = min_;
    tile_params.range = max_ - min_;
    tile_params.segment_count = segment_count;
    tile_params.inv_segment_count = 1.0f / segment_count;
    tile_params.offset = 0;
    tile_params.inv_screen_size = glm::vec2(1.0f / display_size.x, 1.0f / display_size.y);
    tile_params.delta_t = delta_t;
    tile_params.time = cur_time;
    tile_params.tile_index = block_idx_;
    cmd_buf->pushConstants(
#if USE_MESH_SHADER
        SET_FLAG_BIT(ShaderStage, MESH_BIT_EXT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
#else
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) |
        SET_FLAG_BIT(ShaderStage, GEOMETRY_BIT),
#endif
        tile_grass_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    auto new_desc_sets = desc_set_list;
    new_desc_sets.push_back(tile_res_desc_set_[dbuf_idx]);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        tile_grass_pipeline_layout_,
        new_desc_sets);

#if    USE_MESH_SHADER
    cmd_buf->drawMeshTasks(
        (static_cast<uint32_t>(num_grass * 2) + 15) / 16);
#else
    cmd_buf->drawIndexedIndirect(
        grass_indirect_draw_cmd_,
        0);
#endif
}

void TileObject::generateAllDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& heightmap_tex) {
    generateStaticDescriptorSet(
        device,
        descriptor_pool,
        texture_sampler,
        heightmap_tex);
}

void TileObject::drawAllVisibleTiles(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const glm::vec2& camera_pos,
    const glm::uvec2& display_size,
    int dbuf_idx,
    float delta_t,
    float cur_time,
    bool is_base_pass,
    bool render_grass) {

    for (auto& tile : visible_tiles_) {
        tile->draw(
            cmd_buf,
            desc_set_list,
            display_size,
            dbuf_idx,
            delta_t,
            cur_time,
            is_base_pass);

        if (is_base_pass && render_grass) {
            tile->drawGrass(
                cmd_buf,
                desc_set_list,
                camera_pos,
                display_size,
                dbuf_idx,
                delta_t,
                cur_time);
        }
    }
}

void TileObject::updateAllTiles(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    const float& tile_size,
    const glm::vec2& camera_pos) {

    uint32_t segment_count = static_cast<uint32_t>(TileConst::kSegmentCount);
    int32_t cache_tile_size = static_cast<int32_t>(TileConst::kCacheTileSize);
    int32_t visible_tile_size = static_cast<int32_t>(TileConst::kVisibleTileSize);
    int32_t num_cached_blocks = static_cast<int32_t>(TileConst::kNumCachedBlocks);

    glm::ivec2 center_tile_index = glm::ivec2(camera_pos / tile_size + glm::vec2(-0.5f, 0.5f));
    glm::ivec2 min_cache_tile_idx = center_tile_index - glm::ivec2(cache_tile_size);
    glm::ivec2 max_cache_tile_idx = center_tile_index + glm::ivec2(cache_tile_size);
    glm::ivec2 min_visi_tile_idx = center_tile_index - glm::ivec2(visible_tile_size);
    glm::ivec2 max_visi_tile_idx = center_tile_index + glm::ivec2(visible_tile_size);

    visible_tiles_.clear();

    std::vector<size_t> to_delete_tiles;
    // remove all the tiles outside of cache zone.
    for (auto& tile : tile_meshes_) {
        if (tile.second) {
            bool inside = tile.second->validTileBySize(
                min_cache_tile_idx,
                max_cache_tile_idx,
                tile_size);

            if (!inside) {
                to_delete_tiles.push_back(tile.second->getHash());
            }
        }
    }

    for (auto& hash_value : to_delete_tiles) {
        available_block_indexes_.push_back(tile_meshes_[hash_value]->block_idx_);
        auto search_result = tile_meshes_.find(hash_value);
        assert(search_result != tile_meshes_.end());
        search_result->second->destroy(device);
        tile_meshes_.erase(hash_value);
    }

    // add (kCacheTileSize * 2 + 1) x (kCacheTileSize * 2 + 1) tiles for caching.
    std::vector<std::shared_ptr<TileObject>> blocks(num_cached_blocks);
    int32_t tile_idx = 0;
    for (int y = min_cache_tile_idx.y; y <= max_cache_tile_idx.y; y++) {
        for (int x = min_cache_tile_idx.x; x <= max_cache_tile_idx.x; x++) {
            auto tile = addOneTile(
                device,
                descriptor_pool,
                glm::vec2(x, y) * tile_size - tile_size / 2,
                glm::vec2(x, y) * tile_size + tile_size / 2);
            blocks[tile_idx++] = tile;
        }
    }

    int32_t i = 0;
    auto row_size = cache_tile_size * 2 + 1;
    for (int y = min_cache_tile_idx.y; y <= max_cache_tile_idx.y; y++) {
        for (int x = min_cache_tile_idx.x; x <= max_cache_tile_idx.x; x++) {
            glm::ivec4 neighbors = glm::ivec4(
                x == min_cache_tile_idx.x ? -1 : (blocks[i - 1] ? blocks[i - 1]->block_idx_ : -1),
                x == max_cache_tile_idx.x ? -1 : (blocks[i + 1] ? blocks[i + 1]->block_idx_ : -1),
                y == min_cache_tile_idx.y ? -1 : (blocks[i - row_size] ? blocks[i - row_size]->block_idx_ : -1),
                y == max_cache_tile_idx.y ? -1 : (blocks[i + row_size] ? blocks[i + row_size]->block_idx_ : -1));
            if (blocks[i]) {
                blocks[i]->setNeighbors(neighbors);
            }
            i++;
        }
    }

    for (auto& tile : tile_meshes_) {
        if (tile.second) {
            bool inside = tile.second->validTileBySize(
                min_visi_tile_idx,
                max_visi_tile_idx,
                tile_size);

            if (inside) {
                visible_tiles_.push_back(tile.second);
            }
        }
    }

    for (auto& tile : visible_tiles_) {
        renderer::DrawIndexedIndirectCommand indirect_draw_cmd_buffer;
        auto ratio = glm::clamp((tile->getMinDistToCamera(camera_pos) - 256.0f) / 1024.0f, 0.0f, 1.0f);
        auto min_num_grass = static_cast<float>(TileConst::kMinNumGrass);
        auto max_num_grass = static_cast<float>(TileConst::kMaxNumGrass);
        auto num_grass = max_num_grass * (1.0f - ratio) + min_num_grass * ratio;

        indirect_draw_cmd_buffer.index_count = 1;
        indirect_draw_cmd_buffer.vertex_offset = 0;
        indirect_draw_cmd_buffer.first_index = 0;
        indirect_draw_cmd_buffer.first_instance = 0;
        indirect_draw_cmd_buffer.instance_count = int(num_grass);
        device->updateBufferMemory(
            tile->grass_indirect_draw_cmd_.memory,
            sizeof(renderer::DrawIndexedIndirectCommand),
            &indirect_draw_cmd_buffer);
    }
}

void TileObject::destroyAllTiles(
    const std::shared_ptr<renderer::Device>& device) {
    for (auto& tile_mesh : tile_meshes_) {
        tile_mesh.second->destroy(device);
    }
    tile_meshes_.clear();
    visible_tiles_.clear();
}

} // namespace game_object
} // namespace engine