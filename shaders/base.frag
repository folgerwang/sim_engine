#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"
#include "functions.glsl.h"
#include "brdf.glsl.h"
#include "punctual.glsl.h"

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
	GameCameraInfo camera_info;
};

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = PBR_CONSTANT_INDEX) uniform MaterialUniformBufferObject {
    PbrMaterialParams material;
};
#endif

#include "ibl.glsl.h"

layout(location = 0) in VsPsData {
    vec3 vertex_position;
    vec4 vertex_tex_coord;
#ifdef HAS_NORMALS
    vec3 vertex_normal;
#ifdef HAS_TANGENT
    vec3 vertex_tangent;
    vec3 vertex_binormal;
#endif
#endif
#ifdef HAS_VERTEX_COLOR_VEC3
    vec3 vertex_color;
#endif
#ifdef HAS_VERTEX_COLOR_VEC4
    vec4 vertex_color;
#endif
} in_data;

#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = BASE_COLOR_TEX_INDEX) uniform sampler2D basic_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = NORMAL_TEX_INDEX) uniform sampler2D normal_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = METAL_ROUGHNESS_TEX_INDEX) uniform sampler2D metallic_roughness_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = EMISSIVE_TEX_INDEX) uniform sampler2D emissive_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = OCCLUSION_TEX_INDEX) uniform sampler2D occlusion_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = THIN_FILM_LUT_INDEX) uniform sampler2D thin_film_lut;
#endif

layout(location = 0) out vec4 outColor;

struct MaterialInfo
{
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    vec3 f0;                        // full reflectance color (n incidence angle)

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])
    vec3 albedoColor;

    vec3 f90;                       // reflectance color at grazing angle
    float metallic;

    vec3 n;
    vec3 baseColor; // getBaseColor()

    float sheenIntensity;
    vec3 sheenColor;
    float sheenRoughness;

    float anisotropy;

    vec3 clearcoatF0;
    vec3 clearcoatF90;
    float clearcoatFactor;
    vec3 clearcoatNormal;
    float clearcoatRoughness;

    float subsurfaceScale;
    float subsurfaceDistortion;
    float subsurfacePower;
    vec3 subsurfaceColor;
    float subsurfaceThickness;

    float thinFilmFactor;
    float thinFilmThickness;

    float thickness;

    vec3 absorption;

    float transmission;
};


vec4 getVertexColor()
{
   vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

#ifdef HAS_VERTEX_COLOR_VEC3
    color.rgb = in_data.vertex_color;
#endif
#ifdef HAS_VERTEX_COLOR_VEC4
    color = in_data.vertex_color;
#endif

   return color;
}

vec2 getBaseColorUV()
{
#ifndef NO_MTL
    uint base_color_uv_set = material.uv_set_flags.x & 0x0f;
    vec3 uv = vec3(base_color_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_BASECOLOR_UV_TRANSFORM
    uv *= material.base_color_uv_transform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getNormalUV()
{
#ifndef NO_MTL
    uint normal_uv_set = (material.uv_set_flags.x >> 4) & 0x0f;
    vec3 uv = vec3(normal_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_NORMAL_UV_TRANSFORM
    uv *= material.normal_uv_transform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getMetallicRoughnessUV()
{
#ifndef NO_MTL
    uint metallic_roughness_uv_set = (material.uv_set_flags.x >> 8) & 0x0f;
    vec3 uv = vec3(metallic_roughness_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_METALLICROUGHNESS_UV_TRANSFORM
    uv *= material.metallic_roughness_uv_transform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getEmissiveUV()
{
#ifndef NO_MTL
    uint emissive_uv_set = (material.uv_set_flags.x >> 12) & 0x0f;
    vec3 uv = vec3(emissive_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_EMISSIVE_UV_TRANSFORM
    uv *= u_EmissiveUVTransform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getOcclusionUV()
{
#ifndef NO_MTL
    uint occlusion_uv_set = (material.uv_set_flags.x >> 16) & 0x0f;
    vec3 uv = vec3(occlusion_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_OCCLUSION_UV_TRANSFORM
    uv *= u_OcclusionUVTransform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

// Get normal, tangent and bitangent vectors.
NormalInfo getNormalInfo(vec3 v)
{
#ifndef NO_MTL
    vec2 uv = getNormalUV();
    vec3 uv_dx = dFdx(vec3(uv, 0.0));
    vec3 uv_dy = dFdy(vec3(uv, 0.0));

    vec3 t_ = (uv_dy.t * dFdx(in_data.vertex_position) - uv_dx.t * dFdy(in_data.vertex_position)) /
        (uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t);

    vec3 n, t, b, ng;

    bool has_normal_map = (material.material_features & FEATURE_HAS_NORMAL_MAP) != 0;

    // Compute geometrical TBN:
#ifdef HAS_NORMALS
#ifdef HAS_TANGENT
    // Trivial TBN computation, present as vertex attribute.
    // Normalize eigenvectors as matrix is linearly interpolated.
    t = normalize(in_data.vertex_tangent);
    b = normalize(in_data.vertex_binormal);
    ng = normalize(in_data.vertex_normal);
#else
    // Normals are either present as vertex attributes or approximated.
    ng = normalize(in_data.vertex_normal);
    t = normalize(t_ - ng * dot(ng, t_));
    b = cross(ng, t);
#endif
#else
    ng = normalize(cross(dFdx(in_data.vertex_position), dFdy(in_data.vertex_position)));
    t = normalize(t_ - ng * dot(ng, t_));
    b = cross(ng, t);
#endif

    // For a back-facing surface, the tangential basis vectors are negated.
    float facing = step(0.0, dot(v, ng)) * 2.0 - 1.0;
    t *= facing;
    b *= facing;
    ng *= facing;

    // Due to anisoptry, the tangent can be further rotated around the geometric normal.
    vec3 direction;
    #ifdef MATERIAL_ANISOTROPY
        #ifdef HAS_ANISOTROPY_DIRECTION_MAP
            direction = texture(u_AnisotropyDirectionSampler, getAnisotropyDirectionUV()).xyz * 2.0 - vec3(1.0);
        #else
            direction = u_AnisotropyDirection;
        #endif
    #else
        direction = vec3(1.0, 0.0, 0.0);
    #endif
    t = mat3(t, b, ng) * normalize(direction);
    b = normalize(cross(ng, t));

    // Compute pertubed normals:
    if (has_normal_map) {
        n = texture(normal_tex, uv).rgb * 2.0 - vec3(1.0);
        n *= vec3(material.normal_scale, material.normal_scale, 1.0);
        n = mat3(t, b, ng) * normalize(n);
    }
    else {
        n = ng;
    }

    NormalInfo info;
    info.ng = ng;
    info.t = t;
    info.b = b;
    info.n = n;
    return info;
#else
    NormalInfo info;
    info.ng = vec3(0);
    info.t = vec3(0);
    info.b = vec3(0);
    info.n = vec3(0);
    return info;
#endif
}

vec4 getBaseColor()
{
    vec4 baseColor = vec4(1, 1, 1, 1);
#ifndef NO_MTL
    bool enable_metallic_roughness = (material.material_features & FEATURE_MATERIAL_METALLICROUGHNESS) != 0;
    bool has_base_color_map = (material.material_features & FEATURE_HAS_BASE_COLOR_MAP) != 0;
    if (enable_metallic_roughness) {
        baseColor = material.base_color_factor;
        if (has_base_color_map) {
            baseColor *= sRGBToLinear(texture(basic_tex, getBaseColorUV()));
        }
    }
#endif
    return baseColor * getVertexColor();
}

MaterialInfo getSpecularGlossinessInfo(MaterialInfo info)
{
#ifndef NO_MTL
    info.f0 = material.specular_factor;
    info.perceptualRoughness = material.glossiness_factor;

#ifdef HAS_SPECULAR_GLOSSINESS_MAP
    vec4 sgSample = sRGBToLinear(texture(u_SpecularGlossinessSampler, getSpecularGlossinessUV()));
    info.perceptualRoughness *= sgSample.a ; // glossiness to roughness
    info.f0 *= sgSample.rgb; // specular
#endif // ! HAS_SPECULAR_GLOSSINESS_MAP

    info.perceptualRoughness = 1.0 - info.perceptualRoughness; // 1 - glossiness
    info.albedoColor = info.baseColor.rgb * (1.0 - max(max(info.f0.r, info.f0.g), info.f0.b));
#endif

    return info;
}

// KHR_extension_specular alters f0 on metallic materials based on the specular factor specified in the extention
float getMetallicRoughnessSpecularFactor()
{
#ifndef NO_MTL
    //F0 = 0.08 * specularFactor * specularTexture
    float metallic_roughness_specular_factor = 0.08 * material.metallic_roughness_specular_factor;
#ifdef HAS_METALLICROUGHNESS_SPECULAROVERRIDE_MAP
    vec4 specSampler =  texture(u_MetallicRoughnessSpecularSampler, getMetallicRoughnessSpecularUV());
    metallic_roughness_specular_factor *= specSampler.a;
#endif
    return metallic_roughness_specular_factor;
#else
    return 0;
#endif
}

MaterialInfo getMetallicRoughnessInfo(MaterialInfo info, float f0_ior)
{
#ifndef NO_MTL
    info.metallic = material.metallic_factor;
    info.perceptualRoughness = material.roughness_factor;

    bool has_metallic_roughness_map = (material.material_features & FEATURE_HAS_METALLIC_ROUGHNESS_MAP) != 0;
    if (has_metallic_roughness_map) {
        // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
        // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
        vec4 mrSample = texture(metallic_roughness_tex, getMetallicRoughnessUV());
        info.perceptualRoughness *= mrSample.g;
        info.metallic *= mrSample.b;
    }

#ifdef MATERIAL_METALLICROUGHNESS_SPECULAROVERRIDE
    // Overriding the f0 creates unrealistic materials if the IOR does not match up.
    vec3 f0 = vec3(getMetallicRoughnessSpecularFactor());
#else
    // Achromatic f0 based on IOR.
    vec3 f0 = vec3(f0_ior);
#endif

    info.albedoColor = mix(info.baseColor.rgb * (vec3(1.0) - f0),  vec3(0), info.metallic);
    info.f0 = mix(f0, info.baseColor.rgb, info.metallic);
#endif

    return info;
}

MaterialInfo getSheenInfo(MaterialInfo info)
{
#ifndef NO_MTL
    info.sheenColor = material.sheen_color_factor;
    info.sheenIntensity = material.sheen_intensity_factor;
    info.sheenRoughness = material.sheen_roughness;

    #ifdef HAS_SHEEN_COLOR_INTENSITY_MAP
        vec4 sheenSample = texture(u_SheenColorIntensitySampler, getSheenUV());
        info.sheenColor *= sheenSample.xyz;
        info.sheenIntensity *= sheenSample.w;
    #endif
#endif

    return info;
}

MaterialInfo getSubsurfaceInfo(MaterialInfo info)
{
#ifndef NO_MTL
    info.subsurfaceScale = material.subsurface_scale;
    info.subsurfaceDistortion = material.subsurface_distortion;
    info.subsurfacePower = material.subsurface_power;
    info.subsurfaceColor = material.subsurface_color_factor;
    info.subsurfaceThickness = material.subsurface_thickness_factor;

    #ifdef HAS_SUBSURFACE_COLOR_MAP
        info.subsurfaceColor *= texture(u_SubsurfaceColorSampler, getSubsurfaceColorUV()).rgb;
    #endif

    #ifdef HAS_SUBSURFACE_THICKNESS_MAP
        info.subsurfaceThickness *= texture(u_SubsurfaceThicknessSampler, getSubsurfaceThicknessUV()).r;
    #endif
#endif

    return info;
}

MaterialInfo getThinFilmInfo(MaterialInfo info)
{
#ifndef NO_MTL
    info.thinFilmFactor = material.thin_film_factor;
    info.thinFilmThickness = material.thin_film_thickness_maximum / 1200.0;

    #ifdef HAS_THIN_FILM_MAP
        info.thinFilmFactor *= texture(u_ThinFilmSampler, getThinFilmUV()).r;
    #endif

    #ifdef HAS_THIN_FILM_THICKNESS_MAP
        float thicknessSampled = texture(u_ThinFilmThicknessSampler, getThinFilmThicknessUV()).g;
        float thickness = mix(u_ThinFilmThicknessMinimum / 1200.0, u_ThinFilmThicknessMaximum / 1200.0, thicknessSampled);
        info.thinFilmThickness = thickness;
    #endif
#endif

    return info;
}

MaterialInfo getClearCoatInfo(MaterialInfo info, NormalInfo normal_info)
{
#ifndef NO_MTL
    info.clearcoatFactor = material.clearcoat_factor;
    info.clearcoatRoughness = material.clearcoat_roughness_factor;
    info.clearcoatF0 = vec3(0.04);
    info.clearcoatF90 = vec3(clamp(info.clearcoatF0 * 50.0, 0.0, 1.0));

    #ifdef HAS_CLEARCOAT_TEXTURE_MAP
        vec4 ccSample = texture(u_ClearcoatSampler, getClearcoatUV());
        info.clearcoatFactor *= ccSample.r;
    #endif

    #ifdef HAS_CLEARCOAT_ROUGHNESS_MAP
        vec4 ccSampleRough = texture(u_ClearcoatRoughnessSampler, getClearcoatRoughnessUV());
        info.clearcoatRoughness *= ccSampleRough.g;
    #endif

    #ifdef HAS_CLEARCOAT_NORMAL_MAP
        vec4 ccSampleNor = texture(u_ClearcoatNormalSampler, getClearcoatNormalUV());
        info.clearcoatNormal = normalize(ccSampleNor.xyz);
    #else
        info.clearcoatNormal = normal_info.ng;
    #endif

    info.clearcoatRoughness = clamp(info.clearcoatRoughness, 0.0, 1.0);
#endif

    return info;
}

MaterialInfo getTransmissionInfo(MaterialInfo info)
{
#ifndef NO_MTL
    info.transmission = material.transmission;
#endif
    return info;
}

MaterialInfo getAnisotropyInfo(MaterialInfo info)
{
#ifndef NO_MTL
    info.anisotropy = material.anisotropy;

#ifdef HAS_ANISOTROPY_MAP
    info.anisotropy *= texture(u_AnisotropySampler, getAnisotropyUV()).r * 2.0 - 1.0;
#endif
#endif

    return info;
}

MaterialInfo getThicknessInfo(MaterialInfo info)
{
    info.thickness = 1.0;

#ifndef NO_MTL
    if ((material.material_features & FEATURE_MATERIAL_THICKNESS) != 0) {
        info.thickness = material.thickness;

        #ifdef HAS_THICKNESS_MAP
        info.thickness *= texture(u_ThicknessSampler, getThicknessUV()).r;
        #endif
    }
#endif

    return info;
}

MaterialInfo getAbsorptionInfo(MaterialInfo info)
{
    info.absorption = vec3(0.0);

#ifndef NO_MTL
    if ((material.material_features & FEATURE_MATERIAL_ABSORPTION) != 0) {
        info.absorption = material.absorption_color;
    }
#endif

    return info;
}

vec3 getThinFilmF0(vec3 f0, vec3 f90, float NdotV, float thin_film_factor, float thin_film_thickness)
{
#ifndef NO_MTL
    if (thin_film_factor == 0.0)
    {
        // No thin film applied.
        return f0;
    }

    vec3 lut_sample = texture(thin_film_lut, vec2(thin_film_thickness, NdotV)).rgb - 0.5;
    vec3 intensity = thin_film_factor * 4.0 * f0 * (1.0 - f0);
    return clamp(intensity * lut_sample, 0.0, 1.0);
#else
    return vec3(0);
#endif
}

// Uncharted 2 tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 toneMapUncharted2Impl(vec3 color)
{
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec3 toneMapUncharted(vec3 color)
{
    const float W = 11.2;
    color = toneMapUncharted2Impl(color * 2.0);
    vec3 whiteScale = 1.0 / toneMapUncharted2Impl(vec3(W));
    return linearTosRGB(color * whiteScale);
}

// Hejl Richard tone map
// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 toneMapHejlRichard(vec3 color)
{
    color = max(vec3(0.0), color - vec3(0.004));
    return (color*(6.2*color+.5))/(color*(6.2*color+1.7)+0.06);
}

// ACES tone map
// see: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 toneMapACES(vec3 color)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return linearTosRGB(clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0));
}

vec3 toneMap(vec3 color)
{
#ifndef NO_MTL
    color *= material.exposure;

    if (material.tonemap_type == TONEMAP_UNCHARTED) {
        return toneMapUncharted(color);
    }

    if (material.tonemap_type == TONEMAP_HEJLRICHARD) {
        return toneMapHejlRichard(color);
    }

    if (material.tonemap_type == TONEMAP_ACES) {
        return toneMapACES(color);
    }
#endif

    return linearTosRGB(color);
}

void main() {
    vec4 baseColor = getBaseColor();

#ifndef NO_MTL
#ifdef ALPHAMODE_OPAQUE
    baseColor.a = 1.0;
#endif

#ifdef MATERIAL_UNLIT
    outColor = (vec4(linearTosRGB(baseColor.rgb), baseColor.a));
    return;
#endif
    vec3 v = normalize(camera_info.position.xyz - in_data.vertex_position);
    NormalInfo normal_info = getNormalInfo(v);

    vec3 n = normal_info.n;
    vec3 t = normal_info.t;
    vec3 b = normal_info.b;

    float n_dot_v = clampedDot(n, v);
    float t_dot_v = clampedDot(t, v);
    float b_dot_v = clampedDot(b, v);

    MaterialInfo material_info;
    material_info.baseColor = baseColor.rgb;

    uint material_features = material.material_features;
    bool enable_ior = (material_features & FEATURE_MATERIAL_IOR) != 0;
    bool enable_specular_glossing = (material_features & FEATURE_MATERIAL_SPECULARGLOSSINESS) != 0;
    bool enable_metallic_roughness = (material_features & FEATURE_MATERIAL_METALLICROUGHNESS) != 0;
    bool enable_sheen = (material_features & FEATURE_MATERIAL_SHEEN) != 0;
    bool enable_subsurface = (material_features & FEATURE_MATERIAL_SUBSURFACE) != 0;
    bool enable_thin_film = (material_features & FEATURE_MATERIAL_THIN_FILM) != 0;
    bool enable_clearcoat = (material_features & FEATURE_MATERIAL_CLEARCOAT) != 0;
    bool enable_transmission = (material_features & FEATURE_MATERIAL_TRANSMISSION) != 0;
    bool enable_anisotropy = (material_features & FEATURE_MATERIAL_ANISOTROPY) != 0;
    bool enable_absorption = (material_features & FEATURE_MATERIAL_ABSORPTION) != 0;
    bool has_emissive_map = (material_features & FEATURE_HAS_EMISSIVE_MAP) != 0;
    bool has_occlusion_map = (material_features & FEATURE_HAS_OCCLUSION_MAP) != 0;

    float ior = 1.5;
    float f0_ior = 0.04;
    if (enable_ior) {
        ior = material.ior_f0.x;
        f0_ior = material.ior_f0.y;
    } 

    if (enable_specular_glossing) {
        material_info = getSpecularGlossinessInfo(material_info);
    }

    if (enable_metallic_roughness) {
        material_info = getMetallicRoughnessInfo(material_info, f0_ior);
    }

    if (enable_sheen) {
        material_info = getSheenInfo(material_info);
    }

    if (enable_subsurface) {
        material_info = getSubsurfaceInfo(material_info);
    }

    if (enable_thin_film) {
        material_info = getThinFilmInfo(material_info);
    }

    if (enable_clearcoat) {
        material_info = getClearCoatInfo(material_info, normal_info);
    }

    if (enable_transmission) {
        material_info = getTransmissionInfo(material_info);
    }

    if (enable_anisotropy) {
        material_info = getAnisotropyInfo(material_info);
    }

    material_info = getThicknessInfo(material_info);
    material_info = getAbsorptionInfo(material_info);

    material_info.perceptualRoughness = clamp(material_info.perceptualRoughness, 0.0, 1.0);
    material_info.metallic = clamp(material_info.metallic, 0.0, 1.0);

    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness.
    material_info.alphaRoughness = material_info.perceptualRoughness * material_info.perceptualRoughness;

    // Compute reflectance.
    float reflectance = max(max(material_info.f0.r, material_info.f0.g), material_info.f0.b);

    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    material_info.f90 = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

    material_info.n = n;

    if (enable_thin_film) {
        material_info.f0 = getThinFilmF0(material_info.f0, material_info.f90, clampedDot(n, v),
            material_info.thinFilmFactor, material_info.thinFilmThickness);
    }

    // LIGHTING
    vec3 f_specular = vec3(0.0);
    vec3 f_diffuse = vec3(0.0);
    vec3 f_emissive = vec3(0.0);
    vec3 f_clearcoat = vec3(0.0);
    vec3 f_sheen = vec3(0.0);
    vec3 f_subsurface = vec3(0.0);
    vec3 f_transmission = vec3(0.0);

    // Calculate lighting contribution from image based lighting source (IBL)
#ifdef USE_IBL
    f_specular += getIBLRadianceGGX(n, v, material_info.perceptualRoughness, material_info.f0, material.mip_count);
    f_diffuse += getIBLRadianceLambertian(n, material_info.albedoColor);

    if (enable_clearcoat) {
        f_clearcoat += getIBLRadianceGGX(material_info.clearcoatNormal, v, material_info.clearcoatRoughness, material_info.clearcoatF0, material.mip_count);
    }

    if (enable_sheen) {
        f_sheen += getIBLRadianceCharlie(n, v, material_info.sheenRoughness, material_info.sheenColor, material_info.sheenIntensity, material.mip_count);
    }

    if (enable_subsurface) {
        f_subsurface += getIBLRadianceSubsurface(n, v, material_info.subsurfaceScale, material_info.subsurfaceDistortion, material_info.subsurfacePower, material_info.subsurfaceColor, material_info.subsurfaceThickness);
    }

    if (enable_transmission) {
        f_transmission += getIBLRadianceTransmission(n, v, material_info.perceptualRoughness, ior, material_info.baseColor, material.mip_count);
    }
#endif

#ifdef USE_PUNCTUAL
    for (int i = 0; i < LIGHT_COUNT; ++i)
    {
        Light light = material.lights[i];

        vec3 pointToLight = -light.direction;
        float rangeAttenuation = 1.0;
        float spotAttenuation = 1.0;

        if(light.type != LightType_Directional)
        {
            pointToLight = light.position - in_data.vertex_position;
        }

        // Compute range and spot light attenuation.
        if (light.type != LightType_Directional)
        {
            rangeAttenuation = getRangeAttenuation(light.range, length(pointToLight));
        }
        if (light.type == LightType_Spot)
        {
            spotAttenuation = getSpotAttenuation(pointToLight, light.direction, light.outerConeCos, light.innerConeCos);
        }

        vec3 intensity = rangeAttenuation * spotAttenuation * light.intensity * light.color;

        vec3 l = normalize(pointToLight);   // Direction from surface point to light
        vec3 h = normalize(l + v);          // Direction of the vector between l and v, called halfway vector
        float n_dot_l = clampedDot(n, l);
        float n_dot_v = clampedDot(n, v);
        float n_dot_h = clampedDot(n, h);
        float l_dot_h = clampedDot(l, h);
        float v_dot_h = clampedDot(v, h);

        if (n_dot_l > 0.0 || n_dot_v > 0.0)
        {
            // Calculation of analytical light
            //https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
            f_diffuse += intensity * n_dot_l *  BRDF_lambertian(material_info.f0, material_info.f90, material_info.albedoColor, v_dot_h);

            if (enable_anisotropy) {
                vec3 h = normalize(l + v);
                float t_dot_l = dot(t, l);
                float b_dot_l = dot(b, l);
                float t_dot_h = dot(t, h);
                float b_dot_h = dot(b, h);
                f_specular += intensity * n_dot_l * BRDF_specularAnisotropicGGX(material_info.f0, material_info.f90, material_info.alphaRoughness,
                    v_dot_h, n_dot_l, n_dot_v, n_dot_h,
                    b_dot_v, t_dot_v, t_dot_l, b_dot_l, t_dot_h, b_dot_h, material_info.anisotropy);
            }
            else {
                f_specular += intensity * n_dot_l * BRDF_specularGGX(material_info.f0, material_info.f90, material_info.alphaRoughness, v_dot_h, n_dot_l, n_dot_v, n_dot_h);
            }

            if (enable_sheen) {
                f_sheen += intensity * getPunctualRadianceSheen(material_info.sheenColor, material_info.sheenIntensity, material_info.sheenRoughness,
                    n_dot_l, n_dot_v, n_dot_h);
            }

            if (enable_clearcoat) {
                f_clearcoat += intensity * getPunctualRadianceClearCoat(material_info.clearcoatNormal, v, l,
                    h, v_dot_h,
                    material_info.clearcoatF0, material_info.clearcoatF90, material_info.clearcoatRoughness);
            }
        }

        if (enable_subsurface) {
            f_subsurface += intensity * getPunctualRadianceSubsurface(n, v, l,
                material_info.subsurfaceScale, material_info.subsurfaceDistortion, material_info.subsurfacePower,
                material_info.subsurfaceColor, material_info.subsurfaceThickness);
        }

        if (enable_transmission) {
            f_transmission += intensity * getPunctualRadianceTransmission(n, v, l, material_info.alphaRoughness, ior, material_info.f0);
        }
    }
#endif // !USE_PUNCTUAL

    f_emissive = material.emissive_factor;
    if (has_emissive_map) {
        f_emissive *= sRGBToLinear(texture(emissive_tex, getEmissiveUV())).rgb;
    }
    
    vec3 color = vec3(0);

///
/// Layer blending
///

    float clearcoatFactor = 0.0;
    vec3 clearcoatFresnel = vec3(0.0);

    if (enable_clearcoat) {
        clearcoatFactor = material_info.clearcoatFactor;
        clearcoatFresnel = F_Schlick(material_info.clearcoatF0, material_info.clearcoatF90, clampedDot(material_info.clearcoatNormal, v));
    }

    if (enable_absorption) {
        f_transmission *= transmissionAbsorption(v, n, ior, material_info.thickness, material_info.absorption);
    }

    if (enable_transmission) {
        f_diffuse = mix(f_diffuse, f_transmission, material_info.transmission);
    }

    color = (f_emissive + f_diffuse + f_specular + f_subsurface + (1.0 - reflectance) * f_sheen) * (1.0 - clearcoatFactor * clearcoatFresnel) + f_clearcoat * clearcoatFactor;

    float ao = 1.0;
    // Apply optional PBR terms for additional (optional) shading
    if (has_occlusion_map) {
        ao = texture(occlusion_tex,  getOcclusionUV()).r;
        color = mix(color, color * ao, material.occlusion_strength);
    }

#ifdef ALPHAMODE_MASK
    // Late discard to avaoid samplig artifacts. See https://github.com/KhronosGroup/glTF-Sample-Viewer/issues/267
    if(baseColor.a < material.alpha_cutoff)
    {
        discard;
    }
    baseColor.a = 1.0;
#endif

    // regular shading
    outColor = vec4(toneMap(color), baseColor.a);
#else
    outColor = baseColor;
#endif
}