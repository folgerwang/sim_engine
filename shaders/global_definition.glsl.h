#define HAS_BASE_COLOR_MAP          1
#define USE_IBL                     1
#define USE_HDR                     1
#ifdef  HAS_TANGENT 
#define HAS_NORMALS                 1
#endif
#define USE_PUNCTUAL                1

#define PI                          3.1415926535897f

//#define MATERIAL_UNLIT

#define VINPUT_VERTEX_BINDING_POINT         30
#define VINPUT_INSTANCE_BINDING_POINT       31

// Vertex input attribute location.
#define VINPUT_POSITION             0
#define VINPUT_TEXCOORD0            1
#define VINPUT_NORMAL               2
#define VINPUT_TANGENT              3
#define VINPUT_JOINTS_0             4
#define VINPUT_WEIGHTS_0            5
#define VINPUT_COLOR                6
#define VINPUT_JOINTS_1             7
#define VINPUT_WEIGHTS_1            8
#define VINPUT_TEXCOORD1            9

#define IINPUT_MAT_ROT_0            10
#define IINPUT_MAT_ROT_1            12
#define IINPUT_MAT_ROT_2            13
#define IINPUT_MAT_POS_SCALE        14

#define PBR_GLOBAL_PARAMS_SET       0
#define VIEW_PARAMS_SET             1
#define PBR_MATERIAL_PARAMS_SET     2
#define TILE_PARAMS_SET             2
#define MODEL_PARAMS_SET            3

// MODEL_PARAMS_SET.
#define JOINT_CONSTANT_INDEX        0
// VIEW_PARAMS_SET
#define VIEW_CAMERA_BUFFER_INDEX    6

// PBR_GLOBAL_PARAMS_SET ibl lighting textures.
#define GGX_LUT_INDEX               0
#define GGX_ENV_TEX_INDEX           1
#define LAMBERTIAN_ENV_TEX_INDEX    2
#define CHARLIE_LUT_INDEX           3
#define CHARLIE_ENV_TEX_INDEX       4
#define THIN_FILM_LUT_INDEX         5

// PBR_MATERIAL_PARAMS_SET
#define PBR_CONSTANT_INDEX          7
#define ALBEDO_TEX_INDEX            8
#define NORMAL_TEX_INDEX            (ALBEDO_TEX_INDEX + 1)
#define METAL_ROUGHNESS_TEX_INDEX   (NORMAL_TEX_INDEX + 1)
#define EMISSIVE_TEX_INDEX          (METAL_ROUGHNESS_TEX_INDEX + 1)
#define OCCLUSION_TEX_INDEX         (EMISSIVE_TEX_INDEX + 1)
#define DIFFUSE_TEX_INDEX           ALBEDO_TEX_INDEX
#define SPECULAR_TEX_INDEX          EMISSIVE_TEX_INDEX
#define GLOSSNESS_TEX_INDEX         METAL_ROUGHNESS_TEX_INDEX

#define CONEMAP_TEX_INDEX           (OCCLUSION_TEX_INDEX + 1)
#define PRT_PACK_TEX_INDEX          (CONEMAP_TEX_INDEX + 1)
#define PRT_PACK_INFO_TEX_INDEX     (PRT_PACK_TEX_INDEX + 1)
/*#define PRT_TEX_INDEX_0             (CONEMAP_TEX_INDEX + 1)
#define PRT_TEX_INDEX_1             (PRT_TEX_INDEX_0 + 1)
#define PRT_TEX_INDEX_2             (PRT_TEX_INDEX_1 + 1)
#define PRT_TEX_INDEX_3             (PRT_TEX_INDEX_2 + 1)
#define PRT_TEX_INDEX_4             (PRT_TEX_INDEX_3 + 1)
#define PRT_TEX_INDEX_5             (PRT_TEX_INDEX_4 + 1)
#define PRT_TEX_INDEX_6             (PRT_TEX_INDEX_5 + 1)*/

// TILE_TEXTURE_PARAMS_SET
#define TILE_BASE_PARAMS_INDEX              6
#define DST_SOIL_WATER_LAYER_BUFFER_INDEX   (TILE_BASE_PARAMS_INDEX + 0)
#define DST_WATER_NORMAL_BUFFER_INDEX       (TILE_BASE_PARAMS_INDEX + 1)
#define DST_WATER_FLOW_BUFFER_INDEX         (TILE_BASE_PARAMS_INDEX + 2)
#define SRC_COLOR_TEX_INDEX                 (TILE_BASE_PARAMS_INDEX + 3)
#define SRC_DEPTH_TEX_INDEX                 (TILE_BASE_PARAMS_INDEX + 4)
#define ROCK_LAYER_BUFFER_INDEX             (TILE_BASE_PARAMS_INDEX + 5)
#define SOIL_WATER_LAYER_BUFFER_INDEX       (TILE_BASE_PARAMS_INDEX + 6)
#define ORTHER_INFO_LAYER_BUFFER_INDEX      (TILE_BASE_PARAMS_INDEX + 7)
#define WATER_NORMAL_BUFFER_INDEX           (TILE_BASE_PARAMS_INDEX + 8)
#define WATER_FLOW_BUFFER_INDEX             (TILE_BASE_PARAMS_INDEX + 9)
#define SRC_AIRFLOW_INDEX                   (TILE_BASE_PARAMS_INDEX + 10)
#define SRC_MAP_MASK_INDEX                  (TILE_BASE_PARAMS_INDEX + 11)

// Airflow texture.
#define DST_TEMP_TEX_INDEX                  (TILE_BASE_PARAMS_INDEX + 12)
#define DST_MOISTURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 13)
#define DST_PRESSURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 14)
#define DST_AIRFLOW_TEX_INDEX               (TILE_BASE_PARAMS_INDEX + 15)
#define DST_CLOUD_LIGHTING_TEX_INDEX        (TILE_BASE_PARAMS_INDEX + 16)
#define DST_CLOUD_SHADOW_TEX_INDEX          (TILE_BASE_PARAMS_INDEX + 17)
#define SRC_TEMP_TEX_INDEX                  (TILE_BASE_PARAMS_INDEX + 18)
#define SRC_MOISTURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 19)
#define SRC_PRESSURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 20)
#define SRC_CLOUD_LIGHTING_TEX_INDEX        (TILE_BASE_PARAMS_INDEX + 21)
#define SRC_CLOUD_SHADOW_TEX_INDEX          (TILE_BASE_PARAMS_INDEX + 22)
#define DST_FOG_CLOUD_INDEX                 (TILE_BASE_PARAMS_INDEX + 23)
#define DST_SCATTERING_LUT_INDEX            (TILE_BASE_PARAMS_INDEX + 24)
#define DST_SCATTERING_LUT_SUM_INDEX        (TILE_BASE_PARAMS_INDEX + 24)
#define SRC_SCATTERING_LUT_INDEX            (TILE_BASE_PARAMS_INDEX + 25)
#define SRC_SCATTERING_LUT_SUM_INDEX        (TILE_BASE_PARAMS_INDEX + 25)

#define DETAIL_NOISE_TEXTURE_INDEX          (TILE_BASE_PARAMS_INDEX + 26)
#define ROUGH_NOISE_TEXTURE_INDEX           (TILE_BASE_PARAMS_INDEX + 27)
#define PERMUTATION_TEXTURE_INDEX           (TILE_BASE_PARAMS_INDEX + 28)
#define PERMUTATION_2D_TEXTURE_INDEX        (TILE_BASE_PARAMS_INDEX + 29)
#define GRAD_TEXTURE_INDEX                  (TILE_BASE_PARAMS_INDEX + 30)
#define PERM_GRAD_TEXTURE_INDEX             (TILE_BASE_PARAMS_INDEX + 31)
#define PERM_GRAD_4D_TEXTURE_INDEX          (TILE_BASE_PARAMS_INDEX + 32)
#define GRAD_4D_TEXTURE_INDEX               (TILE_BASE_PARAMS_INDEX + 33)

// Noise Texture.
#define DST_PERLIN_NOISE_TEX_INDEX          0

// IBL texure index
#define PANORAMA_TEX_INDEX                  0
#define ENVMAP_TEX_INDEX                    0
#define SRC_TEX_INDEX                       0
#define SRC_TEX_INDEX_1                     (SRC_TEX_INDEX + 1)
#define SRC_TEX_INDEX_2                     (SRC_TEX_INDEX_1 + 1)
#define SRC_INFO_TEX_INDEX                  (SRC_TEX_INDEX_2 + 1)
#define DST_TEX_INDEX                       (SRC_INFO_TEX_INDEX + 1)
#define DST_TEX_INDEX_1                     (DST_TEX_INDEX + 1)

#define VERTEX_BUFFER_INDEX                 0
#define INDEX_BUFFER_INDEX                  1

#define INDIRECT_DRAW_BUFFER_INDEX          0
#define GAME_OBJECTS_BUFFER_INDEX           1
#define INSTANCE_BUFFER_INDEX               2
#define CAMERA_OBJECT_BUFFER_INDEX          3

#define FEATURE_MATERIAL_SPECULARGLOSSINESS     0x00000001
#define FEATURE_MATERIAL_METALLICROUGHNESS      0x00000002
#define FEATURE_MATERIAL_SHEEN                  0x00000004
#define FEATURE_MATERIAL_SUBSURFACE             0x00000008
#define FEATURE_MATERIAL_THIN_FILM              0x00000010
#define FEATURE_MATERIAL_CLEARCOAT              0x00000020
#define FEATURE_MATERIAL_TRANSMISSION           0x00000040
#define FEATURE_MATERIAL_ANISOTROPY             0x00000080
#define FEATURE_MATERIAL_IOR                    0x00000100
#define FEATURE_MATERIAL_THICKNESS              0x00000200
#define FEATURE_MATERIAL_ABSORPTION             0x00000400

#define FEATURE_HAS_BASE_COLOR_MAP              0x00010000
#define FEATURE_HAS_NORMAL_MAP                  0x00020000
#define FEATURE_HAS_METALLIC_ROUGHNESS_MAP      0x00040000
#define FEATURE_HAS_EMISSIVE_MAP                0x00080000
#define FEATURE_HAS_OCCLUSION_MAP               0x00100000
#define FEATURE_HAS_METALLIC_CHANNEL            0x00200000

#define FEATURE_INPUT_HAS_TANGENT               0x00000001

#define LIGHT_COUNT                             1

#define TONEMAP_DEFAULT                         0
#define TONEMAP_UNCHARTED                       1
#define TONEMAP_HEJLRICHARD                     2
#define TONEMAP_ACES                            3

#define INDIRECT_DRAW_BUF_OFS                   4

#define SOIL_WATER_LAYER_MAX_THICKNESS          128.0f
#define SNOW_LAYER_MAX_THICKNESS                8.0f

#define NO_DEBUG_DRAW                           0
#define DEBUG_DRAW_TEMPRETURE                   1
#define DEBUG_DRAW_MOISTURE                     2

#define kAirflowBufferWidth                     256
#define kAirflowBufferBitCount                  6
#define kAirflowBufferHeight                    (1 << kAirflowBufferBitCount)

#define kAtmosphereScatteringLutGroupSize       64
#define kAtmosphereScatteringLutWidth           512
#define kAtmosphereScatteringLutHeight          512
#define kPlanetRadius                           6371e3
#define kAtmosphereRadius                       6471e3
#define kRayleighScaleHeight                    8e3
#define kMieScaleHeight                         1.2e3

#define kWorldMapSize                           16384.0f                  // meters
#define kCloudMapSize                           32768.0f

#define kDetailNoiseTextureSize                 256
#define kRoughNoiseTextureSize                  32

#define kNodeLeft                               0x00      // -x
#define kNodeRight                              0x01      // +x
#define kNodeBack                               0x02      // -y
#define kNodeFront                              0x03      // +y
#define kNodeBelow                              0x04      // -z
#define kNodeAbove                              0x05      // +z

#define kNodeWaterLeft                          0x00      // -x
#define kNodeWaterRight                         0x01      // +x
#define kNodeWaterBack                          0x02      // -y
#define kNodeWaterFront                         0x03      // +y

#define kNumGltfInstance                        4096

#define kPayLoadHitValueIdx                     0
#define kPayLoadShadowedIdx                     1

#define kConemapGenBlockCacheSizeX              128
#define kConemapGenBlockCacheSizeY              128
#define kConemapGenBlockSizeX                   (kConemapGenBlockCacheSizeX * 4)
#define kConemapGenBlockSizeY                   (kConemapGenBlockCacheSizeY * 4)
#define kConemapGenDispatchX                    32
#define kConemapGenDispatchY                    32
#define kConemapGenBlockRadius                  2
#define kPrtShadowGenBlockCacheSizeX            kConemapGenBlockCacheSizeX
#define kPrtShadowGenBlockCacheSizeY            kConemapGenBlockCacheSizeY
#define kPrtShadowGenDispatchX                  kConemapGenDispatchX
#define kPrtShadowGenDispatchY                  kConemapGenDispatchY
#define kPrtPhiSampleCount                      400
#define kPrtThetaSampleCount                    200
#define kPrtShadowInitBlockRadius               2

#define kPrtSampleAngleStep                     (2.0f * PI / float(kPrtPhiSampleCount))


#define GLFW_KEY_W                  87
#define GLFW_KEY_S                  83
#define GLFW_KEY_A                  65
#define GLFW_KEY_D                  68

#ifdef __cplusplus
#pragma once
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
using namespace glm;
namespace glsl {
#endif

// KHR_lights_punctual extension.
// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
struct Light
{
    vec3 direction;
    float range;

    vec3 color;
    float intensity;

    vec3 position;
    float innerConeCos;

    float outerConeCos;
    int type;

    vec2 padding;
};

const int LightType_Directional = 0;
const int LightType_Point = 1;
const int LightType_Spot = 2;

struct ViewParams {
    vec4 camera_pos;
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    mat4 inv_view_proj;
    mat4 inv_view_proj_relative;
    uvec4 input_features;
    vec4 depth_params;
};

struct ModelParams {
    mat4 model_mat;
};

struct PrtLightParams {
    mat4 model_mat;
    float coeffs[25];
    float height_scale;
    vec2 buffer_size;
    vec3 test_color;
};

struct IblParams {
    float roughness;
    uint currentMipLevel;
    uint width;
	float lodBias;
};

struct IblComputeParams {
    ivec4           size;
};

struct TileCreateParams {
    vec2            world_min;
    vec2            world_range;
    vec2            pad;
    uint            width_pixel_count;
    uint            pad1;
};

struct TileUpdateParams {
    vec2            world_min;
    vec2            world_range;
    vec2            range_per_pixel;
    vec2            flow_speed_factor;
    uint            width_pixel_count;
    float           inv_width_pixel_count;
    float           current_time;
    float           pad;
};

struct WeatherControl {
    float           sea_level_temperature;
    float           mix_rate;
    float           soil_temp_adj;
    float           water_temp_adj;
    float           moist_temp_convert;
    float           water_moist_adj;
    float           soil_moist_adj;
    float           transfer_ratio;
    float           transfer_noise_weight;
    float           cloud_forming_ratio;
    float           frozen_ext_factor;
    float           frozen_pow_curve;
};

struct AirflowUpdateParams {
    vec3            world_min;
    float           current_time;
    vec3            world_range;
    float           global_flow_angle;
    vec3            inv_size;
    float           global_flow_scale;
    ivec3           size;
    float           moist_to_pressure_ratio;

    WeatherControl  controls;
};

struct CloudLightingParams {
    vec3            world_min;
    float           current_time;
    vec3            world_range;
    float           g;
    vec3            inv_world_range;
    float           pad1;
    vec3            inv_size;
    float           pad2;
    ivec3           size;
    float           pad3;
    vec2            height_params;
    vec2            pad4;
    vec3            sun_dir;
    float           pad5;
};

struct CloudShadowParams {
    vec3            world_min;
    float           current_time;
    vec3            world_range;
    uint            layer_idx;
    vec3            inv_world_range;
    float           light_ext_factor;
    vec3            inv_size;
    float           pad1;
    ivec3           size;
    float           pad2;
    vec2            height_params;
    vec2            pad3;
    vec3            sun_dir;
    float           pad4;
};

struct TileParams {
    vec2            world_min;
    vec2            inv_world_range;
    vec2            min;
    vec2            range;
    vec2            inv_screen_size;
    uint            segment_count;
    uint            offset;
    float           inv_segment_count;
    float           delta_t;
    float           time;
    uint            tile_index;
};

struct DebugDrawParams {
    vec2            world_min;
    vec2            inv_world_range;
    vec3            debug_min;
    uint            debug_type;
    vec3            debug_range;
    float           pad1;
    uvec3           size;
    uint            pad2;
    vec3            inv_size;
    uint            pad3;
};

struct VolumeMoistrueParams {
    vec2            world_min;
    vec2            inv_world_range;
    uvec2           size;
    vec2            inv_screen_size;
    vec3            sun_pos;
    float           view_ext_factor;
    vec4            noise_weight_0;
    vec4            noise_weight_1;
    vec2            noise_scale;
    float           noise_thresold;
    float           noise_speed_scale;
    float           time;
    float           g;
    float           inv_rayleigh_scale_height;
    float           inv_mie_scale_height;
    float           view_ext_exponent;
    float           ambient_intensity;
    float           phase_intensity;
    float           pressure_to_moist_ratio;
};

struct BlurImageParams {
    uvec2           size;
    vec2            inv_size;
};

struct SkyScatteringParams {
    float           inv_rayleigh_scale_height;
    float           inv_mie_scale_height;
};

// 1 float base layer, rock.
// 1 half soil layer.
// 1 half grass layer.
// 1 half snow layer.
// 1 half water layer.
struct TileVertexInfo {
    uvec2   packed_land_layers;
};

struct SunSkyParams {
    vec3            sun_pos;
    float           g;
    float           inv_rayleigh_scale_height;
    float           inv_mie_scale_height;
    vec2            pad1;
};

struct CloudParams {
    float           pad;
};

struct ConemapGenParams {
    uvec2           size;
    vec2            inv_size;
    ivec2           block_index;
    ivec2           block_offset;
    ivec2           dst_block_offset;
    uint            is_height_map;
    uint            depth_channel;
};

struct PrtShadowCacheGenParams {
    uvec2           size;
    vec2            inv_size;
    ivec2           block_index;
    ivec2           block_offset;
    ivec2           dst_block_offset;
    uint            is_height_map;
    uint            depth_channel;
    float           shadow_noise_thread;
    float           shadow_intensity;
};

struct PrtPackParams {
    uvec2           size;
    ivec2           block_index;
    ivec2           block_offset;
    float           range_scale;
};

struct PrtGenParams {
    uvec2           size;
    vec2            inv_size;
    uvec2           block_offset;
    vec2            pixel_sample_size;
    float           shadow_intensity;
    float           shadow_noise_thread;
    uint            depth_channel;
    uint            is_height_map;
    float           sample_rate;
};

struct GameObjectsUpdateParams {
    vec2            world_min;
    vec2            inv_world_range;
    float           delta_t;
    int             frame_count;
    int             enble_airflow;
    float           water_flow_strength;
    float           air_flow_strength;
    uint            num_objects;
    vec2            pad;
};

struct InstanceBufferUpdateParams {
    uint            num_instances;
};

struct NoiseInitParams {
    float           inv_vol_size;
};

struct MaterialInfo
{
    float perceptualRoughness;      // roughness value, as authored by the model creator (input to shader)
    vec3 f0;                        // full reflectance color (n incidence angle)

    float alphaRoughness;           // roughness mapped to a more linear change in the roughness (proposed by [2])
    vec3 albedoColor;

    vec3 f90;                       // reflectance color at grazing angle
    float metallic;

    float ior;
    vec3 n;
    float reflectance;
    vec3 baseColor;                 // getBaseColor()

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

struct PbrMaterialParams {
    vec4            base_color_factor;

    float           glossiness_factor;
    float           metallic_roughness_specular_factor;
    float           metallic_factor;
    float           roughness_factor;

    vec3            specular_factor;
    float           sheen_intensity_factor;

    vec3            sheen_color_factor;
    float           sheen_roughness;

    float           subsurface_scale;
    float           subsurface_distortion;
    float           subsurface_power;
    float           subsurface_thickness_factor;

    vec3            subsurface_color_factor;
    float           thin_film_factor;

    float           thin_film_thickness_maximum;
    float           clearcoat_factor;
    float           clearcoat_roughness_factor;
    float           transmission;

    float           anisotropy;
    float           thickness;
    float           alpha_cutoff;
    float           exposure;

    vec3            absorption_color;
    float           mip_count;

    vec4            ior_f0;

    uvec4           uv_set_flags;

    uint            material_features;
    uint            pad_1;
    float           normal_scale;
    float           occlusion_strength;

    vec3            emissive_factor;
    uint            tonemap_type;

    Light           lights[LIGHT_COUNT];

    mat3            base_color_uv_transform;
    mat3            normal_uv_transform;
    mat3            metallic_roughness_uv_transform;
};

struct ObjectVsPsData {
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
};

struct TileVsPsData {
    vec3    vertex_position;
    vec2    world_map_uv;
    vec3    test_color;
    float   water_depth;
};

struct PbrLightsColorInfo {
    vec3 f_specular;
    vec3 f_diffuse;
    vec3 f_emissive;
    vec3 f_clearcoat;
    vec3 f_sheen;
    vec3 f_subsurface;
    vec3 f_transmission;
};

struct InstanceDataInfo {
    vec4            mat_rot_0;
    vec4            mat_rot_1;
    vec4            mat_rot_2;
    vec4            mat_pos_scale;
};

struct GrassInstanceDataInfo {
    vec4            mat_rot_0;
    vec4            mat_rot_1;
    vec4            mat_rot_2;
    vec4            mat_pos_scale;
};

struct PrtMinmaxInfo {
    vec2            prt_minmax[25];
};

// could be updated from frame to frame.
struct GameObjectInfo {
    vec3            position;                 // 32-bits float position.
    uint            packed_up_vector;         // 2 half x, y for up vector.

    uint            packed_facing_dir;        // 2 half x, y for facing vector.
    uint            packed_moving_dir_xy;     // 2 half x, y for moving vector.
    uint            packed_moving_dir_z_signs;// 2 half z and signs.
    uint            status;                   // 32 bits for status, todo.

    uint            packed_mass_scale;        // 2 half mass and scale.
    uint            packed_radius_angle;      // 2 half awareness radius and angle.
    uint            pad[2];
};

struct GameCameraParams {
    vec2            world_min;
    vec2            inv_world_range;
    vec3            init_camera_pos;
    int             key;
    vec3            init_camera_dir;
    int             frame_count;
    vec3            init_camera_up;
    float           delta_t;
    vec2            mouse_pos;
    float           camera_speed;
    float           fov;
    float           aspect;
    float           z_near;
    float           z_far;
    float           sensitivity;
    uint            num_game_objs;
    int             game_obj_idx;
    uint            camera_rot_update;
    float           mouse_wheel_offset;
    float           yaw;
    float           pitch;
    float           camera_follow_dist;
};

// could be updated from frame to frame.
struct GameCameraInfo {
    mat4            view;
    mat4            proj;
    mat4            view_proj;
    mat4            inv_view_proj;
    mat4            inv_view_proj_relative;
    vec4            depth_params;
    vec3            position;
    uint            status;                   // 32 bits for status, todo.
    vec3            up_vector;
    float           yaw;
    vec3            facing_dir;
    float           pitch;
    vec2            mouse_pos;
    float           camera_follow_dist;
    float           pad;
};

struct VertexBufferInfo {
    mat3x4          matrix;
    uint            position_base;
    uint            position_stride;
    uint            normal_base;
    uint            normal_stride;
    uint            uv_base;
    uint            uv_stride;
    uint            color_base;
    uint            color_stride;
    uint            tangent_base;
    uint            tangent_stride;
    uint            index_base;
    uint            index_bytes;
};

#ifdef __cplusplus
} //namespace glsl
#endif

#ifndef __cplusplus
// Partitions the bit pattern of 'x' so that we can interleave it with another number.
uint partBy2(uint x) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

// Interleave the bits of x and y to produce the Morton code / Z-order index
uint zOrder(uint x, uint y) {
    return (partBy2(x) | (partBy2(y) << 1));
}
#endif