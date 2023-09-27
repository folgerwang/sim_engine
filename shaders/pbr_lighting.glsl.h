#ifndef NO_MTL
layout(set = PBR_MATERIAL_PARAMS_SET, binding = ALBEDO_TEX_INDEX) uniform sampler2D albedo_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = NORMAL_TEX_INDEX) uniform sampler2D normal_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = METAL_ROUGHNESS_TEX_INDEX) uniform sampler2D metallic_roughness_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = EMISSIVE_TEX_INDEX) uniform sampler2D emissive_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = OCCLUSION_TEX_INDEX) uniform sampler2D occlusion_tex;
layout(set = PBR_MATERIAL_PARAMS_SET, binding = THIN_FILM_LUT_INDEX) uniform sampler2D thin_film_lut;
#endif

vec4 getVertexColor(
    in ObjectVsPsData in_data) {
   vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

#ifdef HAS_VERTEX_COLOR_VEC3
    color.rgb = in_data.vertex_color;
#endif
#ifdef HAS_VERTEX_COLOR_VEC4
    color = in_data.vertex_color;
#endif

   return color;
}

vec2 getBaseColorUV(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    uint base_color_uv_set = in_mat.uv_set_flags.x & 0x0f;
    vec3 uv = vec3(base_color_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_BASECOLOR_UV_TRANSFORM
    uv *= in_mat.base_color_uv_transform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getNormalUV(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    uint normal_uv_set = (in_mat.uv_set_flags.x >> 4) & 0x0f;
    vec3 uv = vec3(normal_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_NORMAL_UV_TRANSFORM
    uv *= in_mat.normal_uv_transform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getMetallicRoughnessUV(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    uint metallic_roughness_uv_set = (in_mat.uv_set_flags.x >> 8) & 0x0f;
    vec3 uv = vec3(metallic_roughness_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_METALLICROUGHNESS_UV_TRANSFORM
    uv *= in_mat.metallic_roughness_uv_transform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getEmissiveUV(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    uint emissive_uv_set = (in_mat.uv_set_flags.x >> 12) & 0x0f;
    vec3 uv = vec3(emissive_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

    #ifdef HAS_EMISSIVE_UV_TRANSFORM
    uv *= u_EmissiveUVTransform;
    #endif

    return uv.xy;
#else
    return vec2(0);
#endif
}

vec2 getOcclusionUV(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    uint occlusion_uv_set = (in_mat.uv_set_flags.x >> 16) & 0x0f;
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
NormalInfo getNormalInfo(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat,
    in vec3 v) {
#ifndef NO_MTL
    vec2 uv = getNormalUV(in_data, in_mat);
    vec3 uv_dx = dFdx(vec3(uv, 0.0));
    vec3 uv_dy = dFdy(vec3(uv, 0.0));

    vec3 t_ = (uv_dy.t * dFdx(in_data.vertex_position) - uv_dx.t * dFdy(in_data.vertex_position)) /
        (uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t);

    vec3 n, t, b, ng;

    bool has_normal_map = (in_mat.material_features & FEATURE_HAS_NORMAL_MAP) != 0;

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
        n *= vec3(in_mat.normal_scale, in_mat.normal_scale, 1.0);
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

vec4 getBaseColor(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat) {
    vec4 baseColor = vec4(1, 1, 1, 1);
#ifndef NO_MTL
    bool enable_metallic_roughness = (in_mat.material_features & FEATURE_MATERIAL_METALLICROUGHNESS) != 0;
    bool has_base_color_map = (in_mat.material_features & FEATURE_HAS_BASE_COLOR_MAP) != 0;
    if (enable_metallic_roughness) {
        baseColor = in_mat.base_color_factor;
        if (has_base_color_map) {
            baseColor *= sRGBToLinear(texture(albedo_tex, getBaseColorUV(in_data, in_mat)));
        }
    }
#endif
    return baseColor * getVertexColor(in_data);
}

void getSpecularGlossinessInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    info.f0 = in_mat.specular_factor;
    info.perceptualRoughness = in_mat.glossiness_factor;

#ifdef HAS_SPECULAR_GLOSSINESS_MAP
    vec4 sgSample = sRGBToLinear(texture(u_SpecularGlossinessSampler, getSpecularGlossinessUV()));
    info.perceptualRoughness *= sgSample.a ; // glossiness to roughness
    info.f0 *= sgSample.rgb; // specular
#endif // ! HAS_SPECULAR_GLOSSINESS_MAP

    info.perceptualRoughness = 1.0 - info.perceptualRoughness; // 1 - glossiness
    info.albedoColor = info.baseColor.rgb * (1.0 - max(max(info.f0.r, info.f0.g), info.f0.b));
#endif
}

// KHR_extension_specular alters f0 on metallic materials based on the specular factor specified in the extention
float getMetallicRoughnessSpecularFactor(
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    //F0 = 0.08 * specularFactor * specularTexture
    float metallic_roughness_specular_factor = 0.08 * in_mat.metallic_roughness_specular_factor;
#ifdef HAS_METALLICROUGHNESS_SPECULAROVERRIDE_MAP
    vec4 specSampler = texture(u_MetallicRoughnessSpecularSampler, getMetallicRoughnessSpecularUV());
    metallic_roughness_specular_factor *= specSampler.a;
#endif
    return metallic_roughness_specular_factor;
#else
    return 0;
#endif
}

void getMetallicRoughnessInfo(
    inout MaterialInfo info,
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_mat,
    float f0_ior)
{
#ifndef NO_MTL
    info.metallic = in_mat.metallic_factor;
    info.perceptualRoughness = in_mat.roughness_factor;

    bool has_metallic_roughness_map = (in_mat.material_features & FEATURE_HAS_METALLIC_ROUGHNESS_MAP) != 0;
    bool has_metallic_channel = (in_mat.material_features & FEATURE_HAS_METALLIC_CHANNEL) != 0;
    if (has_metallic_roughness_map) {
        // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
        // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
        vec4 mrSample = texture(metallic_roughness_tex, getMetallicRoughnessUV(in_data, in_mat));
        info.perceptualRoughness *= mrSample.g;
        if (has_metallic_channel) {
            info.metallic *= mrSample.b;
        }
        else {
            info.metallic *= (1.0f - mrSample.g);
        }
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
}

void getSheenInfo(
inout MaterialInfo info,
in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    info.sheenColor = in_mat.sheen_color_factor;
    info.sheenIntensity = in_mat.sheen_intensity_factor;
    info.sheenRoughness = in_mat.sheen_roughness;

    #ifdef HAS_SHEEN_COLOR_INTENSITY_MAP
        vec4 sheenSample = texture(u_SheenColorIntensitySampler, getSheenUV());
        info.sheenColor *= sheenSample.xyz;
        info.sheenIntensity *= sheenSample.w;
    #endif
#endif
}

void getSubsurfaceInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    info.subsurfaceScale = in_mat.subsurface_scale;
    info.subsurfaceDistortion = in_mat.subsurface_distortion;
    info.subsurfacePower = in_mat.subsurface_power;
    info.subsurfaceColor = in_mat.subsurface_color_factor;
    info.subsurfaceThickness = in_mat.subsurface_thickness_factor;

    #ifdef HAS_SUBSURFACE_COLOR_MAP
        info.subsurfaceColor *= texture(u_SubsurfaceColorSampler, getSubsurfaceColorUV()).rgb;
    #endif

    #ifdef HAS_SUBSURFACE_THICKNESS_MAP
        info.subsurfaceThickness *= texture(u_SubsurfaceThicknessSampler, getSubsurfaceThicknessUV()).r;
    #endif
#endif
}

void getThinFilmInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    info.thinFilmFactor = in_mat.thin_film_factor;
    info.thinFilmThickness = in_mat.thin_film_thickness_maximum / 1200.0;

    #ifdef HAS_THIN_FILM_MAP
        info.thinFilmFactor *= texture(u_ThinFilmSampler, getThinFilmUV()).r;
    #endif

    #ifdef HAS_THIN_FILM_THICKNESS_MAP
        float thicknessSampled = texture(u_ThinFilmThicknessSampler, getThinFilmThicknessUV()).g;
        float thickness = mix(u_ThinFilmThicknessMinimum / 1200.0, u_ThinFilmThicknessMaximum / 1200.0, thicknessSampled);
        info.thinFilmThickness = thickness;
    #endif
#endif
}

void getClearCoatInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat,
    in NormalInfo normal_info) {
#ifndef NO_MTL
    info.clearcoatFactor = in_mat.clearcoat_factor;
    info.clearcoatRoughness = in_mat.clearcoat_roughness_factor;
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
}

void getTransmissionInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    info.transmission = in_mat.transmission;
#endif
}

void getAnisotropyInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat) {
#ifndef NO_MTL
    info.anisotropy = in_mat.anisotropy;

#ifdef HAS_ANISOTROPY_MAP
    info.anisotropy *= texture(u_AnisotropySampler, getAnisotropyUV()).r * 2.0 - 1.0;
#endif
#endif
}

void getThicknessInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat) {
    info.thickness = 1.0;

#ifndef NO_MTL
    if ((in_mat.material_features & FEATURE_MATERIAL_THICKNESS) != 0) {
        info.thickness = in_mat.thickness;

        #ifdef HAS_THICKNESS_MAP
        info.thickness *= texture(u_ThicknessSampler, getThicknessUV()).r;
        #endif
    }
#endif
}

void getAbsorptionInfo(
    inout MaterialInfo info,
    in PbrMaterialParams in_mat)
{
    info.absorption = vec3(0.0);

#ifndef NO_MTL
    if ((in_mat.material_features & FEATURE_MATERIAL_ABSORPTION) != 0) {
        info.absorption = in_mat.absorption_color;
    }
#endif
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

vec3 toneMap(in PbrMaterialParams in_mat, vec3 color)
{
#ifndef NO_MTL
    color *= in_mat.exposure;

    if (in_mat.tonemap_type == TONEMAP_UNCHARTED) {
        return toneMapUncharted(color);
    }

    if (in_mat.tonemap_type == TONEMAP_HEJLRICHARD) {
        return toneMapHejlRichard(color);
    }

    if (in_mat.tonemap_type == TONEMAP_ACES) {
        return toneMapACES(color);
    }
#endif

    return linearTosRGB(color);
}

#ifndef NO_MTL
MaterialInfo setupMaterialInfo(
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_material,
    in NormalInfo normal_info,
    in vec3 view_dir,
    vec3 base_color) {
    MaterialInfo material_info;
    material_info.baseColor = base_color;

    uint material_features = in_material.material_features;
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
        ior = in_material.ior_f0.x;
        f0_ior = in_material.ior_f0.y;
    }

    material_info.ior = ior;

    if (enable_specular_glossing) {
        getSpecularGlossinessInfo(material_info, in_material);
    }

    if (enable_metallic_roughness) {
        getMetallicRoughnessInfo(material_info, in_data, in_material, f0_ior);
    }

    if (enable_sheen) {
        getSheenInfo(material_info, in_material);
    }

    if (enable_subsurface) {
        getSubsurfaceInfo(material_info, in_material);
    }

    if (enable_thin_film) {
        getThinFilmInfo(material_info, in_material);
    }

    if (enable_clearcoat) {
        getClearCoatInfo(material_info, in_material, normal_info);
    }

    if (enable_transmission) {
        getTransmissionInfo(material_info, in_material);
    }

    if (enable_anisotropy) {
        getAnisotropyInfo(material_info, in_material);
    }

    getThicknessInfo(material_info, in_material);
    getAbsorptionInfo(material_info, in_material);

    material_info.perceptualRoughness = clamp(material_info.perceptualRoughness, 0.0, 1.0);
    material_info.metallic = clamp(material_info.metallic, 0.0, 1.0);

    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness.
    material_info.alphaRoughness = material_info.perceptualRoughness * material_info.perceptualRoughness;

    // Compute reflectance.
    material_info.reflectance = max(max(material_info.f0.r, material_info.f0.g), material_info.f0.b);

    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
    material_info.f90 = vec3(clamp(material_info.reflectance * 50.0, 0.0, 1.0));

    material_info.n = normal_info.n;

    if (enable_thin_film) {
        material_info.f0 = getThinFilmF0(material_info.f0, material_info.f90, clampedDot(normal_info.n, view_dir),
            material_info.thinFilmFactor, material_info.thinFilmThickness);
    }

    return material_info;
}
#endif

PbrLightsColorInfo initColorInfo() {
    PbrLightsColorInfo color_info;
    color_info.f_specular = vec3(0.0);
    color_info.f_diffuse = vec3(0.0);
    color_info.f_emissive = vec3(0.0);
    color_info.f_clearcoat = vec3(0.0);
    color_info.f_sheen = vec3(0.0);
    color_info.f_subsurface = vec3(0.0);
    color_info.f_transmission = vec3(0.0);
    return color_info;
}

void iblLighting(
    inout PbrLightsColorInfo color_info,
    in PbrMaterialParams in_material,
    in MaterialInfo in_material_info,
    in NormalInfo normal_info,
    in vec3 view_dir) {

    uint material_features = in_material.material_features;
    bool enable_clearcoat = (material_features & FEATURE_MATERIAL_CLEARCOAT) != 0;
    bool enable_sheen = (material_features & FEATURE_MATERIAL_SHEEN) != 0;
    bool enable_subsurface = (material_features & FEATURE_MATERIAL_SUBSURFACE) != 0;
    bool enable_transmission = (material_features & FEATURE_MATERIAL_TRANSMISSION) != 0;

    color_info.f_specular +=
        getIBLRadianceGGX(
            normal_info.n,
            view_dir,
            in_material_info.perceptualRoughness,
            in_material_info.f0,
            in_material.mip_count);

    color_info.f_diffuse +=
        getIBLRadianceLambertian(
            normal_info.n,
            in_material_info.albedoColor);

    if (enable_clearcoat) {
        color_info.f_clearcoat +=
            getIBLRadianceGGX(
                in_material_info.clearcoatNormal,
                view_dir,
                in_material_info.clearcoatRoughness,
                in_material_info.clearcoatF0,
                in_material.mip_count);
    }

    if (enable_sheen) {
        color_info.f_sheen +=
            getIBLRadianceCharlie(
                normal_info.n,
                view_dir,
                in_material_info.sheenRoughness,
                in_material_info.sheenColor,
                in_material_info.sheenIntensity,
                in_material.mip_count);
    }

    if (enable_subsurface) {
        color_info.f_subsurface +=
            getIBLRadianceSubsurface(
                normal_info.n,
                view_dir,
                in_material_info.subsurfaceScale,
                in_material_info.subsurfaceDistortion,
                in_material_info.subsurfacePower,
                in_material_info.subsurfaceColor,
                in_material_info.subsurfaceThickness);
    }

    if (enable_transmission) {
        color_info.f_transmission +=
            getIBLRadianceTransmission(
                normal_info.n,
                view_dir,
                in_material_info.perceptualRoughness,
                in_material_info.ior,
                in_material_info.baseColor,
                in_material.mip_count);
    }
}

void punctualLighting(
    inout PbrLightsColorInfo color_info,
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_material,
    in MaterialInfo in_material_info,
    in Light light,
    in NormalInfo normal_info,
    in vec3 view_dir,
    in float light_intensity) {
    vec3 pointToLight = -light.direction;
    float rangeAttenuation = 1.0;
    float spotAttenuation = 1.0;

    if (light.type != LightType_Directional)
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

    vec3 intensity =
        rangeAttenuation *
        spotAttenuation *
        light.intensity *
        light_intensity *
        light.color;

    vec3 n = normal_info.n;
    vec3 t = normal_info.t;
    vec3 b = normal_info.b;
    vec3 v = view_dir;

    float t_dot_v = clampedDot(t, v);
    float b_dot_v = clampedDot(b, v);

    vec3 l = normalize(pointToLight);   // Direction from surface point to light
    vec3 h = normalize(l + v);          // Direction of the vector between l and v, called halfway vector
    float n_dot_l = clampedDot(in_material_info.n, l);
    float n_dot_v = clampedDot(in_material_info.n, v);
    float n_dot_h = clampedDot(in_material_info.n, h);
    float l_dot_h = clampedDot(l, h);
    float v_dot_h = clampedDot(v, h);

    uint material_features = in_material.material_features;
    bool enable_anisotropy = (material_features & FEATURE_MATERIAL_ANISOTROPY) != 0;
    bool enable_clearcoat = (material_features & FEATURE_MATERIAL_CLEARCOAT) != 0;
    bool enable_transmission = (material_features & FEATURE_MATERIAL_TRANSMISSION) != 0;
    bool enable_sheen = (material_features & FEATURE_MATERIAL_SHEEN) != 0;
    bool enable_subsurface = (material_features & FEATURE_MATERIAL_SUBSURFACE) != 0;

    if (n_dot_l > 0.0 || n_dot_v > 0.0)
    {
        // Calculation of analytical light
        //https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
        color_info.f_diffuse +=
            intensity *
            n_dot_l *
            BRDF_lambertian(
                in_material_info.f0,
                in_material_info.f90,
                in_material_info.albedoColor,
                v_dot_h);

        if (enable_anisotropy) {
            vec3 h = normalize(l + v);
            float t_dot_l = dot(t, l);
            float b_dot_l = dot(b, l);
            float t_dot_h = dot(t, h);
            float b_dot_h = dot(b, h);
            vec3 specular =
                BRDF_specularAnisotropicGGX(
                    in_material_info.f0,
                    in_material_info.f90,
                    in_material_info.alphaRoughness,
                    v_dot_h,
                    n_dot_l,
                    n_dot_v,
                    n_dot_h,
                    b_dot_v,
                    t_dot_v,
                    t_dot_l,
                    b_dot_l,
                    t_dot_h,
                    b_dot_h,
                    in_material_info.anisotropy);
            color_info.f_specular += intensity * n_dot_l * specular;
        }
        else {
            vec3 specular =
                BRDF_specularGGX(
                    in_material_info.f0,
                    in_material_info.f90,
                    in_material_info.alphaRoughness,
                    v_dot_h,
                    n_dot_l,
                    n_dot_v,
                    n_dot_h);
            color_info.f_specular += intensity * n_dot_l * specular;
        }

        if (enable_sheen) {
            vec3 sheen =
                getPunctualRadianceSheen(
                    in_material_info.sheenColor,
                    in_material_info.sheenIntensity,
                    in_material_info.sheenRoughness,
                    n_dot_l,
                    n_dot_v,
                    n_dot_h);
            color_info.f_sheen += intensity * sheen;
        }

        if (enable_clearcoat) {
            vec3 clearcoat =
                getPunctualRadianceClearCoat(
                    in_material_info.clearcoatNormal,
                    v,
                    l,
                    h,
                    v_dot_h,
                    in_material_info.clearcoatF0,
                    in_material_info.clearcoatF90,
                    in_material_info.clearcoatRoughness);
            color_info.f_clearcoat += intensity * clearcoat;
        }
    }

    if (enable_subsurface) {
        vec3 subsurface =
            getPunctualRadianceSubsurface(
                n,
                v,
                l,
                in_material_info.subsurfaceScale,
                in_material_info.subsurfaceDistortion,
                in_material_info.subsurfacePower,
                in_material_info.subsurfaceColor,
                in_material_info.subsurfaceThickness);
        color_info.f_subsurface += intensity * subsurface;
    }

    if (enable_transmission) {
        vec3 transmission =
            getPunctualRadianceTransmission(
                n,
                v,
                l,
                in_material_info.alphaRoughness,
                in_material_info.ior,
                in_material_info.f0);
        color_info.f_transmission += intensity * transmission;
    }
}

void layerBlending(
    inout PbrLightsColorInfo color_info,
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_material,
    in MaterialInfo in_material_info,
    in NormalInfo normal_info,
    in vec3 view_dir) {

    uint material_features = in_material.material_features;
    bool enable_transmission = (material_features & FEATURE_MATERIAL_TRANSMISSION) != 0;
    bool enable_absorption = (material_features & FEATURE_MATERIAL_ABSORPTION) != 0;
    bool has_emissive_map = (material_features & FEATURE_HAS_EMISSIVE_MAP) != 0;

    color_info.f_emissive = in_material.emissive_factor;
    if (has_emissive_map) {
#ifndef NO_MTL
        color_info.f_emissive *=
            sRGBToLinear(texture(emissive_tex, getEmissiveUV(in_data, in_material))).rgb;
#endif
    }

    ///
    /// Layer blending
    ///

    if (enable_absorption) {
        color_info.f_transmission *=
            transmissionAbsorption(
                view_dir,
                normal_info.n,
                in_material_info.ior,
                in_material_info.thickness,
                in_material_info.absorption);
    }

    if (enable_transmission) {
        color_info.f_diffuse =
            mix(
                color_info.f_diffuse,
                color_info.f_transmission,
                in_material_info.transmission);
    }
}

vec3 getFinalColor(
    in PbrLightsColorInfo in_color_info,
    in ObjectVsPsData in_data,
    in PbrMaterialParams in_material,
    in MaterialInfo in_material_info,
    in vec3 view_dir)
{
    uint material_features = in_material.material_features;
    bool enable_clearcoat = (material_features & FEATURE_MATERIAL_CLEARCOAT) != 0;
    bool has_occlusion_map = (material_features & FEATURE_HAS_OCCLUSION_MAP) != 0;

    float clearcoatFactor = 0.0;
    vec3 clearcoatFresnel = vec3(0.0);

    if (enable_clearcoat) {
        clearcoatFactor = in_material_info.clearcoatFactor;
        clearcoatFresnel =
            F_Schlick(
                in_material_info.clearcoatF0,
                in_material_info.clearcoatF90,
                clampedDot(in_material_info.clearcoatNormal, view_dir));
    }

    vec3 color = 
        (in_color_info.f_emissive +
         in_color_info.f_diffuse +
         in_color_info.f_specular +
         in_color_info.f_subsurface +
         (1.0 - in_material_info.reflectance) * in_color_info.f_sheen) * (1.0 - clearcoatFactor * clearcoatFresnel) +
        in_color_info.f_clearcoat * clearcoatFactor;

/*    float ao = 1.0;
    // Apply optional PBR terms for additional (optional) shading
    if (has_occlusion_map) {
#ifndef NO_MTL
        ao = texture(occlusion_tex, getOcclusionUV(in_data, in_material)).r;
        color = mix(color, color * ao, in_material.occlusion_strength);
#endif
    }*/

    return color;
}