//#define kRealWorldAirflowMaxHeight        12000.0f
//Reduce atmosphere height from 12000.0f to 8500.0f
#define kRealWorldAirflowMaxHeight        12000.0f
#define kDegreeDecreasePerKm              (6.5f / 1000.0f)

#define kAirflowMaxHeight                 (kRealWorldAirflowMaxHeight / kDegreeDecreasePerKm / 1000.0f * 6.5f)
#define kAirflowLowHeight                 -1.0f
#define kAirflowHeightRange               (kAirflowMaxHeight - kAirflowLowHeight)

#define kTemperaturePositiveOffset        78.0f
#define kTemperatureNormalizer            (1.0f / 128.0f)
#define kTemperatureDenormalizer          (1.0f / kTemperatureNormalizer)
#define kMoistureNormalizerExpMin         (-16.0f)
#define kMoistureNormalizerExpMax         (10.0f)
#define kMoistureNormalizerExpRange       (kMoistureNormalizerExpMax - kMoistureNormalizerExpMin)
#define kMoistureNormalizerMin            exp2(kMoistureNormalizerExpMin)

#define kMaxTemperatureAdjustRange        4.0f
#define kMaxAirflowStrength               1.0f
#define kAirflowStrengthNormalizeScale    (1.0f / kMaxAirflowStrength)

#define kMaxTempMoistDiff                 2.0f

#define kMinSampleHeight                  187.0f
#define kAirflowHeightMinMaxRatio         (kAirflowHeightRange / (kAirflowBufferHeight / 2 * kMinSampleHeight) - 1.0f)
#define kAirflowHeightFactorA             ((kAirflowHeightMinMaxRatio - 1.0f) * kAirflowBufferHeight * kMinSampleHeight / 2)
#define kAirflowHeightFactorB             (kMinSampleHeight * kAirflowBufferHeight)
#define kAirflowHeightFactorInvA          (1.0f / kAirflowHeightFactorA)
#define kAirflowHeightFactorBInv2A        (kAirflowHeightFactorB / 2.0f * kAirflowHeightFactorInvA)
#define kAirflowHeightFactorSqrBInv2A     (kAirflowHeightFactorBInv2A * kAirflowHeightFactorBInv2A)

#define kAirflowToHeightParamsX           log2(1.0f + kAirflowHeightRange)
#define kAirflowToHeightParamsY           (-1.0f + kAirflowLowHeight)
#define kAirflowFromHeightParamsX         (1.0f / log2(kAirflowMaxHeight - kAirflowLowHeight + 1.0f))

#define USE_LINEAR_HEIGHT_SAMPLE          1

float calculateAirflowSampleToHeight(float uvw_w) {
    float sample_h = (kAirflowHeightFactorB + kAirflowHeightFactorA * uvw_w) * uvw_w;
    return sample_h;
}

float calculateAirflowSampleToDeltaHeight(float uvw_w) {
    float delta_h = kAirflowHeightFactorB + 2.0f * kAirflowHeightFactorA * uvw_w;
    return delta_h / kAirflowBufferHeight;
}

float calculateAirflowSampleFromHeight(float sample_h) {
    float delta = max(kAirflowHeightFactorSqrBInv2A + sample_h * kAirflowHeightFactorInvA, 0.0f);
    float uvw_w = -kAirflowHeightFactorBInv2A + sqrt(delta);
    return uvw_w;
}

float getReferenceDegree(float sea_level_temp_c, float altitude) {

    float sea_level_temp = sea_level_temp_c;
    float temperature = clamp(sea_level_temp - altitude * kDegreeDecreasePerKm,
                              -kTemperaturePositiveOffset,
                              kTemperatureDenormalizer - kTemperaturePositiveOffset);
    return temperature;
}

float getSampleToHeight(float uvw_w) {
#if USE_LINEAR_HEIGHT_SAMPLE
    return calculateAirflowSampleToHeight(uvw_w);
#else
    return exp2(uvw_w *kAirflowToHeightParamsX) + kAirflowToHeightParamsY;
#endif
}

float getSampleToDeltaHeight(float uvw_w) {
#if USE_LINEAR_HEIGHT_SAMPLE
    return calculateAirflowSampleToDeltaHeight(uvw_w);
#else
    return exp2(uvw_w * kAirflowToHeightParamsX) * log(2.0f) * kAirflowToHeightParamsX / kAirflowBufferHeight;
#endif
}

float getHeightToSample(float sample_h) {
#if USE_LINEAR_HEIGHT_SAMPLE
    return calculateAirflowSampleFromHeight(sample_h);
#else
    return log2(max((sample_h - kAirflowLowHeight), 0.0f) + 1.0f) * kAirflowFromHeightParamsX;
#endif
}

vec2 getPositionWSXy(vec2 uvw_xy, vec2 world_min, vec2 world_range) {
    return uvw_xy * world_range + world_min;
}


float getNormalizedVectorLength(vec3 dir_vec) {
    return length(dir_vec) * kAirflowStrengthNormalizeScale;
}

float getPackedVectorLength(float packed_w) {
    return packed_w / kAirflowStrengthNormalizeScale;
}

float henyeyGreensteinPhaseFunc(float g, float cos_theta) {
    float g2 = g * g;
    const float kPi = 3.1415926535f;
    return (1.0f - g2) / (pow(1.0f + g2 - 2.0f * g * cos_theta, 1.5f) * 4.0f * kPi);
}

float rsi_n(vec3 r0, vec3 rd_n, float sr) {
    r0.y += kPlanetRadius;
    sr += kPlanetRadius;
    float b = dot(rd_n, r0);
    float c = dot(r0, r0) - (sr * sr);

    float result = 0.0f;
    float delta = b * b - c;
    if (delta >= 0) {
        float t0 = -b - sqrt(delta);
        float t1 = -b + sqrt(delta);

        if (t0 < 0.0f) {
            t0 = t1;
        }

        if (t1 < 0.0f) {
            t1 = t0;
        }

        float t = min(t0, t1);
        result = max(t, 0.0f);
    }

    return result;
}

float getBuckSaturatedVaporPressure(float temp) {
    float pressure_kpa =
        0.61121f * exp((18.678f - (temp / 234.5f)) * (temp / (257.14f + temp)));

    return pressure_kpa;
}

float getAirPressure(float temp, float height) {
    float pb = 101.325f;
    float g0 = 9.80665f;
    float m = 0.0289644f;
    float r = 8.3144598f;
    return pb * exp(-g0 * m * height / (r * (temp + 273.15f)));
}

#ifndef __cplusplus
vec3 worldPositionToUvw(vec3 pos_ws) {
    float pos_ws_y_org = pos_ws.y + kPlanetRadius;
    float scale = (kPlanetRadius + kAirflowLowHeight) / pos_ws_y_org;

    vec3 uvw;
    uvw.xy = (pos_ws.xz * scale + kCloudMapSize * 0.5f) / kCloudMapSize;

    float dist_to_center = length(vec3(pos_ws.x, pos_ws_y_org, pos_ws.z));
    uvw.z = getHeightToSample(dist_to_center - kPlanetRadius);
    return uvw;
}

vec3 uvwToWorldPosition(vec3 uvw) {
    float dist_to_center = getSampleToHeight(uvw.z) + kPlanetRadius;
    vec2 pos_ws_xz_scaled = uvw.xy * kCloudMapSize - kCloudMapSize * 0.5f;
    vec3 pos_ws_scaled = vec3(pos_ws_xz_scaled.x, kPlanetRadius + kAirflowLowHeight, pos_ws_xz_scaled.y);
    float scale = dist_to_center / length(pos_ws_scaled);
    return scale * pos_ws_scaled;
}
#endif
