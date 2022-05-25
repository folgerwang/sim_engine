layout(set = PBR_GLOBAL_PARAMS_SET, binding = GGX_LUT_INDEX) uniform sampler2D ggx_lut;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = GGX_ENV_TEX_INDEX) uniform samplerCube ggx_env_sampler;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = LAMBERTIAN_ENV_TEX_INDEX) uniform samplerCube lambertian_env_sampler;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = CHARLIE_LUT_INDEX) uniform sampler2D charlie_lut;
layout(set = PBR_GLOBAL_PARAMS_SET, binding = CHARLIE_ENV_TEX_INDEX) uniform samplerCube charlie_env_sampler;

vec3 getIBLRadianceGGX(vec3 n, vec3 v, float perceptualRoughness, vec3 specularColor, float mip_count)
{
    float n_dot_v = clampedDot(n, v);
    float lod = clamp(perceptualRoughness * float(mip_count), 0.0, float(mip_count));
    vec3 reflection = normalize(reflect(-v, n));

    vec2 brdfSamplePoint = clamp(vec2(n_dot_v, perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 brdf = texture(ggx_lut, brdfSamplePoint).rg;
    vec4 specularSample = textureLod(ggx_env_sampler, reflection, lod);

    vec3 specularLight = specularSample.rgb;

#ifndef USE_HDR
    specularLight = sRGBToLinear(specularLight);
#endif

    return specularLight * (specularColor * brdf.x + brdf.y);
}

vec3 getIBLRadianceTransmission(vec3 n, vec3 v, float perceptualRoughness, float ior, vec3 baseColor, float mip_count)
{
    // Sample GGX LUT.
    float NdotV = clampedDot(n, v);
    vec2 brdfSamplePoint = clamp(vec2(NdotV, perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 brdf = texture(ggx_lut, brdfSamplePoint).rg;

    // Sample GGX environment map.
    float lod = clamp(perceptualRoughness * float(mip_count), 0.0, float(mip_count));

    // Approximate double refraction by assuming a solid sphere beneath the point.
    vec3 r = refract(-v, n, 1.0 / ior);
    vec3 m = 2.0 * dot(-n, r) * r + n;
    vec3 rr = -refract(-r, m, ior);

    vec4 specularSample = textureLod(ggx_env_sampler, rr, lod);
    vec3 specularLight = specularSample.rgb;

#ifndef USE_HDR
    specularLight = sRGBToLinear(specularLight);
#endif

   return specularLight * (brdf.x + brdf.y);
}

vec3 getIBLRadianceLambertian(vec3 n, vec3 diffuseColor)
{
    vec3 diffuseLight = texture(lambertian_env_sampler, n).rgb;

    #ifndef USE_HDR
        diffuseLight = sRGBToLinear(diffuseLight);
    #endif

    return diffuseLight * diffuseColor;
}

vec3 getIBLRadianceCharlie(vec3 n, vec3 v, float sheenRoughness, vec3 sheenColor, float sheenIntensity, float mip_count)
{
    float NdotV = clampedDot(n, v);
    float lod = clamp(sheenRoughness * float(mip_count), 0.0, float(mip_count));
    vec3 reflection = normalize(reflect(-v, n));

    vec2 brdfSamplePoint = clamp(vec2(NdotV, sheenRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    float brdf = texture(charlie_lut, brdfSamplePoint).b;
    vec4 sheenSample = textureLod(charlie_env_sampler, reflection, lod);

    vec3 sheenLight = sheenSample.rgb;

    #ifndef USE_HDR
    sheenLight = sRGBToLinear(sheenLight);
    #endif

    return sheenIntensity * sheenLight * sheenColor * brdf;
}

vec3 getIBLRadianceSubsurface(vec3 n, vec3 v, float scale, float distortion, float power, vec3 color, float thickness)
{
    vec3 diffuseLight = texture(lambertian_env_sampler, n).rgb;

    #ifndef USE_HDR
        diffuseLight = sRGBToLinear(diffuseLight);
    #endif

    return diffuseLight * getPunctualRadianceSubsurface(n, v, -v, scale, distortion, power, color, thickness);
}
