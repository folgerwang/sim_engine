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
    uint base_color_uv_set = material.uv_set_flags.x & 0x0f;
    vec3 uv = vec3(base_color_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

#ifdef HAS_BASECOLOR_UV_TRANSFORM
    uv *= material.base_color_uv_transform;
#endif

    return uv.xy;
}

vec2 getNormalUV()
{
    uint normal_uv_set = (material.uv_set_flags.x >> 4) & 0x0f;
    vec3 uv = vec3(normal_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

#ifdef HAS_NORMAL_UV_TRANSFORM
    uv *= material.normal_uv_transform;
#endif

    return uv.xy;
}

vec2 getMetallicRoughnessUV()
{
    uint metallic_roughness_uv_set = (material.uv_set_flags.x >> 8) & 0x0f;
    vec3 uv = vec3(metallic_roughness_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

#ifdef HAS_METALLICROUGHNESS_UV_TRANSFORM
    uv *= material.metallic_roughness_uv_transform;
#endif

    return uv.xy;
}

vec2 getEmissiveUV()
{
    uint emissive_uv_set = (material.uv_set_flags.x >> 12) & 0x0f;
    vec3 uv = vec3(emissive_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

#ifdef HAS_EMISSIVE_UV_TRANSFORM
    uv *= u_EmissiveUVTransform;
#endif

    return uv.xy;
}

vec2 getOcclusionUV()
{
    uint occlusion_uv_set = (material.uv_set_flags.x >> 16) & 0x0f;
    vec3 uv = vec3(occlusion_uv_set < 1 ? in_data.vertex_tex_coord.xy : in_data.vertex_tex_coord.zw, 1.0);

#ifdef HAS_OCCLUSION_UV_TRANSFORM
    uv *= u_OcclusionUVTransform;
#endif

    return uv.xy;
}

// Get normal, tangent and bitangent vectors.
NormalInfo getNormalInfo(vec3 v)
{
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
}

vec4 getBaseColor()
{
    vec4 baseColor = vec4(1, 1, 1, 1);

    bool enable_metallic_roughness = (material.material_features & FEATURE_MATERIAL_METALLICROUGHNESS) != 0;
    bool has_base_color_map = (material.material_features & FEATURE_HAS_BASE_COLOR_MAP) != 0;
    if (enable_metallic_roughness) {
        baseColor = material.base_color_factor;
        if (has_base_color_map) {
            baseColor *= sRGBToLinear(texture(basic_tex, getBaseColorUV()));
        }
    }

    return baseColor * getVertexColor();
}

MaterialInfo getSpecularGlossinessInfo(MaterialInfo info)
{
    info.f0 = material.specular_factor;
    info.perceptualRoughness = material.glossiness_factor;

#ifdef HAS_SPECULAR_GLOSSINESS_MAP
    vec4 sgSample = sRGBToLinear(texture(u_SpecularGlossinessSampler, getSpecularGlossinessUV()));
    info.perceptualRoughness *= sgSample.a; // glossiness to roughness
    info.f0 *= sgSample.rgb; // specular
#endif // ! HAS_SPECULAR_GLOSSINESS_MAP

    info.perceptualRoughness = 1.0 - info.perceptualRoughness; // 1 - glossiness
    info.albedoColor = info.baseColor.rgb * (1.0 - max(max(info.f0.r, info.f0.g), info.f0.b));

    return info;
}

// KHR_extension_specular alters f0 on metallic materials based on the specular factor specified in the extention
float getMetallicRoughnessSpecularFactor()
{
    //F0 = 0.08 * specularFactor * specularTexture
    float metallic_roughness_specular_factor = 0.08 * material.metallic_roughness_specular_factor;
#ifdef HAS_METALLICROUGHNESS_SPECULAROVERRIDE_MAP
    vec4 specSampler = texture(u_MetallicRoughnessSpecularSampler, getMetallicRoughnessSpecularUV());
    metallic_roughness_specular_factor *= specSampler.a;
#endif
    return metallic_roughness_specular_factor;
}

MaterialInfo getMetallicRoughnessInfo(MaterialInfo info, float f0_ior)
{
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

    info.albedoColor = mix(info.baseColor.rgb * (vec3(1.0) - f0), vec3(0), info.metallic);
    info.f0 = mix(f0, info.baseColor.rgb, info.metallic);

    return info;
}

MaterialInfo getSheenInfo(MaterialInfo info)
{
    info.sheenColor = material.sheen_color_factor;
    info.sheenIntensity = material.sheen_intensity_factor;
    info.sheenRoughness = material.sheen_roughness;

#ifdef HAS_SHEEN_COLOR_INTENSITY_MAP
    vec4 sheenSample = texture(u_SheenColorIntensitySampler, getSheenUV());
    info.sheenColor *= sheenSample.xyz;
    info.sheenIntensity *= sheenSample.w;
#endif

    return info;
}

MaterialInfo getSubsurfaceInfo(MaterialInfo info)
{
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

    return info;
}

MaterialInfo getThinFilmInfo(MaterialInfo info)
{
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

    return info;
}

MaterialInfo getClearCoatInfo(MaterialInfo info, NormalInfo normal_info)
{
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

    return info;
}

MaterialInfo getTransmissionInfo(MaterialInfo info)
{
    info.transmission = material.transmission;
    return info;
}

MaterialInfo getAnisotropyInfo(MaterialInfo info)
{
    info.anisotropy = material.anisotropy;

#ifdef HAS_ANISOTROPY_MAP
    info.anisotropy *= texture(u_AnisotropySampler, getAnisotropyUV()).r * 2.0 - 1.0;
#endif

    return info;
}

MaterialInfo getThicknessInfo(MaterialInfo info)
{
    info.thickness = 1.0;

    if ((material.material_features & FEATURE_MATERIAL_THICKNESS) != 0) {
        info.thickness = material.thickness;

#ifdef HAS_THICKNESS_MAP
        info.thickness *= texture(u_ThicknessSampler, getThicknessUV()).r;
#endif
    }

    return info;
}

MaterialInfo getAbsorptionInfo(MaterialInfo info)
{
    info.absorption = vec3(0.0);

    if ((material.material_features & FEATURE_MATERIAL_ABSORPTION) != 0) {
        info.absorption = material.absorption_color;
    }
    return info;
}

vec3 getThinFilmF0(vec3 f0, vec3 f90, float NdotV, float thin_film_factor, float thin_film_thickness)
{
    if (thin_film_factor == 0.0)
    {
        // No thin film applied.
        return f0;
    }

    vec3 lut_sample = texture(thin_film_lut, vec2(thin_film_thickness, NdotV)).rgb - 0.5;
    vec3 intensity = thin_film_factor * 4.0 * f0 * (1.0 - f0);
    return clamp(intensity * lut_sample, 0.0, 1.0);
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
    return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
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
    return (color * (6.2 * color + .5)) / (color * (6.2 * color + 1.7) + 0.06);
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

    return linearTosRGB(color);
}
