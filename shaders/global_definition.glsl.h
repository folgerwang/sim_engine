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
#define VIEW_PARAMS_SET             (PBR_GLOBAL_PARAMS_SET + 1)
#define PBR_MATERIAL_PARAMS_SET     (VIEW_PARAMS_SET + 1)
#define TILE_PARAMS_SET             PBR_MATERIAL_PARAMS_SET
#define SKIN_PARAMS_SET             (PBR_MATERIAL_PARAMS_SET + 1)
#define RUNTIME_LIGHTS_PARAMS_SET   (SKIN_PARAMS_SET + 1)
#define MAX_NUM_PARAMS_SETS         (RUNTIME_LIGHTS_PARAMS_SET + 1)

#define RUNTIME_LIGHTS_CONSTANT_INDEX     0

// SKIN_PARAMS_SET.
#define JOINT_CONSTANT_INDEX        0
// VIEW_PARAMS_SET
#define VIEW_CAMERA_BUFFER_INDEX    7

// PBR_GLOBAL_PARAMS_SET ibl lighting textures.
#define GGX_LUT_INDEX               0
#define GGX_ENV_TEX_INDEX           1
#define LAMBERTIAN_ENV_TEX_INDEX    2
#define CHARLIE_LUT_INDEX           3
#define CHARLIE_ENV_TEX_INDEX       4
#define THIN_FILM_LUT_INDEX         5
#define DIRECT_SHADOW_INDEX         6

// PBR_MATERIAL_PARAMS_SET
#define PBR_CONSTANT_INDEX          8
#define ALBEDO_TEX_INDEX            (PBR_CONSTANT_INDEX + 1) // 9
#define NORMAL_TEX_INDEX            (ALBEDO_TEX_INDEX + 1)  // 10
#define METAL_ROUGHNESS_TEX_INDEX   (NORMAL_TEX_INDEX + 1) // 11
#define EMISSIVE_TEX_INDEX          (METAL_ROUGHNESS_TEX_INDEX + 1) // 12
#define OCCLUSION_TEX_INDEX         (EMISSIVE_TEX_INDEX + 1) // 13
#define SPECULAR_TEX_INDEX          (OCCLUSION_TEX_INDEX + 1) // 14
#define DIFFUSE_TEX_INDEX           ALBEDO_TEX_INDEX // 9    
#define GLOSSNESS_TEX_INDEX         METAL_ROUGHNESS_TEX_INDEX // 11
#define DST_COLOR_TEX_INDEX         (SPECULAR_TEX_INDEX + 1) // 15
#define DST_WEIGHT_TEX_INDEX        (DST_COLOR_TEX_INDEX + 1) // 16
#define SRC_WEIGHT_TEX_INDEX        (DST_WEIGHT_TEX_INDEX + 1) // 17

#define CONEMAP_TEX_INDEX           (SRC_WEIGHT_TEX_INDEX + 1) // 18
#define PRT_PACK_TEX_INDEX          (CONEMAP_TEX_INDEX + 1) // 19
#define PRT_PACK_INFO_TEX_INDEX     (PRT_PACK_TEX_INDEX + 1) // 20

// ── Alpha-only companion texture for shadow / depth-only pass ────────
// R8_UNORM sampler containing just the albedo's α channel for materials
// whose albedo has real cutout α (drawable_object.cpp:
// computeEffectiveOpaqueForMaterials extracts these at material-load).
// For materials without a real cutout (or no albedo, or scan unavailable),
// the binding is filled with a global 1×1 R8 white fallback so the shader
// can read it unconditionally without a discard never firing.  Only the
// depth-only fragment shader samples this; forward shaders ignore it.
#define ALPHA_ONLY_TEX_INDEX        (PRT_PACK_INFO_TEX_INDEX + 1) // 21
/*#define PRT_TEX_INDEX_0             (CONEMAP_TEX_INDEX + 1)
#define PRT_TEX_INDEX_1             (PRT_TEX_INDEX_0 + 1)
#define PRT_TEX_INDEX_2             (PRT_TEX_INDEX_1 + 1)
#define PRT_TEX_INDEX_3             (PRT_TEX_INDEX_2 + 1)
#define PRT_TEX_INDEX_4             (PRT_TEX_INDEX_3 + 1)
#define PRT_TEX_INDEX_5             (PRT_TEX_INDEX_4 + 1)
#define PRT_TEX_INDEX_6             (PRT_TEX_INDEX_5 + 1)*/

// TILE_TEXTURE_PARAMS_SET
#define TILE_BASE_PARAMS_INDEX              7
#define DST_SOIL_WATER_LAYER_BUFFER_INDEX   (TILE_BASE_PARAMS_INDEX + 0) // 7
#define DST_WATER_NORMAL_BUFFER_INDEX       (TILE_BASE_PARAMS_INDEX + 1) // 8
#define DST_WATER_FLOW_BUFFER_INDEX         (TILE_BASE_PARAMS_INDEX + 2) // 9
#define SRC_COLOR_TEX_INDEX                 (TILE_BASE_PARAMS_INDEX + 3) // 10
#define SRC_DEPTH_TEX_INDEX                 (TILE_BASE_PARAMS_INDEX + 4) // 11
#define ROCK_LAYER_BUFFER_INDEX             (TILE_BASE_PARAMS_INDEX + 5) // 12
#define SOIL_WATER_LAYER_BUFFER_INDEX       (TILE_BASE_PARAMS_INDEX + 6) // 13
#define ORTHER_INFO_LAYER_BUFFER_INDEX      (TILE_BASE_PARAMS_INDEX + 7) // 14
#define WATER_NORMAL_BUFFER_INDEX           (TILE_BASE_PARAMS_INDEX + 8) // 15
#define WATER_FLOW_BUFFER_INDEX             (TILE_BASE_PARAMS_INDEX + 9) // 16
#define SRC_AIRFLOW_INDEX                   (TILE_BASE_PARAMS_INDEX + 10) // 17    
#define SRC_MAP_MASK_INDEX                  (TILE_BASE_PARAMS_INDEX + 11) // 18

// Airflow texture.
#define DST_TEMP_TEX_INDEX                  (TILE_BASE_PARAMS_INDEX + 12) // 19
#define DST_MOISTURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 13) // 20
#define DST_PRESSURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 14) // 21
#define DST_AIRFLOW_TEX_INDEX               (TILE_BASE_PARAMS_INDEX + 15) // 22
#define DST_CLOUD_LIGHTING_TEX_INDEX        (TILE_BASE_PARAMS_INDEX + 16) // 23  
#define DST_CLOUD_SHADOW_TEX_INDEX          (TILE_BASE_PARAMS_INDEX + 17) // 24
#define SRC_TEMP_TEX_INDEX                  (TILE_BASE_PARAMS_INDEX + 18) // 25
#define SRC_MOISTURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 19) // 26
#define SRC_PRESSURE_TEX_INDEX              (TILE_BASE_PARAMS_INDEX + 20) // 27
#define SRC_CLOUD_LIGHTING_TEX_INDEX        (TILE_BASE_PARAMS_INDEX + 21) // 28
#define SRC_CLOUD_SHADOW_TEX_INDEX          (TILE_BASE_PARAMS_INDEX + 22) // 29
#define DST_FOG_CLOUD_INDEX                 (TILE_BASE_PARAMS_INDEX + 23) // 30
#define DST_SCATTERING_LUT_INDEX            (TILE_BASE_PARAMS_INDEX + 24) // 31   
#define DST_SCATTERING_LUT_SUM_INDEX        (TILE_BASE_PARAMS_INDEX + 24) // 31
#define SRC_SCATTERING_LUT_INDEX            (TILE_BASE_PARAMS_INDEX + 25) // 32
#define SRC_SCATTERING_LUT_SUM_INDEX        (TILE_BASE_PARAMS_INDEX + 25) // 32

#define DETAIL_NOISE_TEXTURE_INDEX          (TILE_BASE_PARAMS_INDEX + 26) // 33
#define ROUGH_NOISE_TEXTURE_INDEX           (TILE_BASE_PARAMS_INDEX + 27) // 34
#define PERMUTATION_TEXTURE_INDEX           (TILE_BASE_PARAMS_INDEX + 28) // 35     
#define PERMUTATION_2D_TEXTURE_INDEX        (TILE_BASE_PARAMS_INDEX + 29) // 36
#define GRAD_TEXTURE_INDEX                  (TILE_BASE_PARAMS_INDEX + 30) // 37
#define PERM_GRAD_TEXTURE_INDEX             (TILE_BASE_PARAMS_INDEX + 31) // 38
#define PERM_GRAD_4D_TEXTURE_INDEX          (TILE_BASE_PARAMS_INDEX + 32) // 39
#define GRAD_4D_TEXTURE_INDEX               (TILE_BASE_PARAMS_INDEX + 33) // 40

// Noise Texture.
#define DST_PERLIN_NOISE_TEX_INDEX          0

// IBL texure index
#define PANORAMA_TEX_INDEX                  0
#define ENVMAP_TEX_INDEX                    0
#define SRC_TEX_INDEX                       0
#define SRC_TEX_INDEX_1                     (SRC_TEX_INDEX + 1) // 1
#define SRC_TEX_INDEX_2                     (SRC_TEX_INDEX_1 + 1) // 2
#define SRC_INFO_TEX_INDEX                  (SRC_TEX_INDEX_2 + 1) // 3
#define DST_TEX_INDEX                       (SRC_INFO_TEX_INDEX + 1) // 4
#define DST_TEX_INDEX_1                     (DST_TEX_INDEX + 1) // 5

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
// Alpha-mode flags propagated from CPU AlphaMode::*.  Allows the fragment
// shader to dispatch on translucency at runtime — primarily so the
// "Translucent" render-debug mode can highlight blend / mask / opaque
// materials without inspecting per-mesh names.  Glass-by-name materials
// (matched in drawable_object loader) also set FEATURE_MATERIAL_BLEND so
// they show up in the visualisation alongside asset-authored blends.
#define FEATURE_MATERIAL_BLEND                  0x00000800
#define FEATURE_MATERIAL_ALPHA_MASK             0x00001000

#define FEATURE_HAS_BASE_COLOR_MAP              0x00010000
#define FEATURE_HAS_NORMAL_MAP                  0x00020000
#define FEATURE_HAS_METALLIC_ROUGHNESS_MAP      0x00040000
#define FEATURE_HAS_EMISSIVE_MAP                0x00080000
#define FEATURE_HAS_OCCLUSION_MAP               0x00100000
#define FEATURE_HAS_METALLIC_CHANNEL            0x00200000

#define FEATURE_INPUT_HAS_TANGENT               0x00000001
#define FEATURE_INPUT_SHADOW_DISABLED           0x00000002

// Debug render mode is packed into bits 16..23 of camera_info.input_features
// (8 bits, values 0..255).  base.frag and cluster_bindless.frag both unpack
// it as `(input_features >> SHIFT) & 0xFF` and override outColor when the
// value is non-zero.  Driven by the "Render Debug" combo in the menu bar.
#define FEATURE_INPUT_DEBUG_MODE_SHIFT          16
#define FEATURE_INPUT_DEBUG_MODE_MASK           0x00FF0000

#define DEBUG_RENDER_MODE_FINAL                 0   // shipping shaded path
#define DEBUG_RENDER_MODE_ALBEDO                1   // baseColor.rgb (linear)
#define DEBUG_RENDER_MODE_NORMAL                2   // perturbed N as RGB (×0.5+0.5)
#define DEBUG_RENDER_MODE_DIFFUSE               3   // IBL+punctual diffuse term
#define DEBUG_RENDER_MODE_SPECULAR              4   // IBL+punctual specular term
#define DEBUG_RENDER_MODE_SHADOW                5   // CSM shadow factor (grayscale)
#define DEBUG_RENDER_MODE_ROUGHNESS             6   // perceptual roughness (grayscale)
#define DEBUG_RENDER_MODE_METALLIC              7   // metallic (grayscale)
#define DEBUG_RENDER_MODE_GEOMETRIC_NORMAL      8   // ng (un-perturbed normal) as RGB
#define DEBUG_RENDER_MODE_TRANSLUCENT           9   // alpha-mode tint: opaque=dark grey,
                                                    // mask=yellow, blend/glass=magenta.
                                                    // Use to verify glass materials are
                                                    // correctly tagged AlphaMode::Blend
                                                    // in the cluster + standard paths.
#define DEBUG_RENDER_MODE_HIZ                   12  // Hi-Z pyramid mip:
                                                    //   selected mip rendered as
                                                    //   greyscale depth (close=black,
                                                    //   far=white).  Mip level packed
                                                    //   into bits 24..27 of
                                                    //   camera_info.input_features.
                                                    //   Only meaningful in deferred
                                                    //   mode — forward path renders
                                                    //   as final shaded.
#define FEATURE_INPUT_HIZ_MIP_SHIFT             24
#define FEATURE_INPUT_HIZ_MIP_MASK              0x0F000000
#define DEBUG_RENDER_MODE_VELOCITY              10  // screen-space NDC-delta velocity:
                                                    //   grey   = no motion (0,0)
                                                    //   red    = +X motion (camera panning left)
                                                    //   cyan   = -X
                                                    //   green  = +Y (camera tilting down)
                                                    //   magenta= -Y
                                                    // In deferred mode samples gbuf_velocity_;
                                                    // in forward mode computes inline from
                                                    // cur/prev clip varyings.  Cluster geometry
                                                    // only — terrain/grass/hair stay 0.
#define DEBUG_RENDER_MODE_SSAO                  11  // raw ambient-occlusion factor
                                                    // (1 = unoccluded white, 0 = fully
                                                    // occluded black).  Implemented as
                                                    // an early-out inside ssao_apply.comp
                                                    // — when this mode is active the apply
                                                    // pass overwrites color with vec3(ao)
                                                    // instead of multiplying it in.

#define LIGHT_COUNT                             1
// 6 cascades (was 4): smaller extent ratio between adjacent cascades
// (~2.5× vs ~4×), which makes transition boundaries less visible because
// the per-texel world size doesn't jump as much.  See cascade splits in
// application.cpp for the actual far-depth values.
#define CSM_CASCADE_COUNT                       6

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

#define kNumDrawableInstance                    1//4096

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

#define kHairPatchDispatchX                     8
#define kHairPatchDispatchY                     64

#define kLbmPatchDispatchX                      8
#define kLbmPatchDispatchY                      64

#define kPrtSampleAngleStep                     (2.0f * PI / float(kPrtPhiSampleCount))


#define GLFW_KEY_W                              87
#define GLFW_KEY_S                              83
#define GLFW_KEY_A                              65
#define GLFW_KEY_D                              68

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
    uint flip_uv_coord;
    // cascade_idx: only consumed by the CSM_PER_CASCADE permutation of
    // base_depthonly.vert (DrawMode::kCsmPerCascade — the "Regular"
    // option on the shadow draw-mode menu).  In every other pipeline
    // this field is unused (and either zero-initialised by drawNodes or
    // left at whatever drawMesh writes — readers ignore it).  Sits in
    // the old vec3 pad's first 4 bytes so the overall struct size and
    // alignment are unchanged (mat4 + uint + uint + uvec2 = 80 bytes,
    // matching the previous mat4 + uint + vec3 layout).
    uint cascade_idx;
    uvec2 pad;
};

struct PrtLightParams {
    mat4 model_mat;
    float coeffs[25];
    float height_scale;
    vec2 buffer_size;
    vec2 inv_buffer_size;
};

struct IblParams {
    float roughness;
    uint currentMipLevel;
    uint width;
	float lodBias;
};

// Push constants for the dithered + temporally-integrated "mini-buffer"
// IBL convolution update.
//
// Per dispatch the shader processes one mip of one IBL filter.  Two
// orthogonal cost reductions are applied on top of the reference
// fragment-shader path (cube_ibl.frag):
//
//   1. Spatial 8x8 dither.  Only 1/64 of the destination mip's texels are
//      touched per frame; dither_offset cycles through 64 unique (dx, dy)
//      positions over 64 frames so the whole mip is eventually refreshed.
//      For mips with face_size < 8 we fall back to a full update every
//      frame (dither_stride == 1, dither_offset == (0, 0)).
//
//   2. Temporal sample subdivision.  Each touched texel does only
//      NUM_SAMPLES_PER_FRAME = NUM_SAMPLES/64 importance samples per
//      frame.  The new partial estimate is blended into the existing
//      texel value with `temporal_alpha` (exponential moving average),
//      so the texel "integrates" sample contributions over many touches.
//      `frame_index` is used to stratify which sample subset is taken
//      this frame: across 64 touches a pixel sees all NUM_SAMPLES sample
//      indices.
//
// For specular / sheen, only mip 0 is convolved.  The lower-resolution
// mips are produced by a downsample (box-filter mipgen) pass after the
// dispatch - see updateIblSpecularMapMini / updateIblSheenMapMini.
struct IblMiniParams {
    float roughness;          // for THIS mip (matches old IblParams.roughness)
    int   currentMipLevel;    // mip index of the destination
    int   width;              // top-mip cube_size (used by solidAngleTexel)
    float lodBias;

    int   mip_face_size;      // size of THIS mip's face (cube_size >> mip)
    int   mini_size;          // dispatch threads per axis (mip_face_size/stride)
    int   dither_stride;      // 8 for sparse, 1 for full-update small mips
    // 1 on the very first dispatch into this IBL output (image is in
    // UNDEFINED layout / contains garbage).  In this mode the shader
    // clears the entire stride^2 block to vec4(0) (alpha=0 = "untouched"
    // sentinel) and writes the partial estimate only to the single
    // dither-position texel.  The EMA path uses effective_alpha=1.0
    // whenever alpha==0 so each texel seeds cleanly on its literal first
    // touch over the next 64 frames - no block-grid pop.
    int   is_first_touch;

    ivec2 dither_offset;      // (dx, dy) within the dither block
    int   frame_index;        // monotonic frame counter, used for sample stratification
    float temporal_alpha;     // EMA weight for the partial new estimate (0..1)
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

struct BaseShapeDrawParams {
    mat4            transform;
};

// Push constants for the "Nanite-lite" per-cluster flat-color debug draw.
// Populated on the C++ side by the ClusterDebugDraw helper, consumed by
// cluster_debug.vert / cluster_debug.frag. Only the model transform is
// needed — the hashed per-cluster color is derived entirely from the
// vertex-attribute `cluster_id` that the vertex shader passes through.
struct ClusterDebugParams {
    mat4            transform;
};

// ─── Nanite-like GPU cluster culling structures ──────────────────────────
// Shared between cluster_cull.comp and the C++ ClusterRenderer class.
// All data is packed for std430 SSBO access.

// Per-cluster culling data — one entry per cluster in the mesh.
struct ClusterCullInfo {
    vec4    bounds_sphere;       // xyz = center, w = radius
    vec4    cone_axis_cutoff;    // xyz = cone axis (unit), w = cutoff (cos angle)
    vec4    aabb_min_pad;        // xyz = AABB min, w = pad
    vec4    aabb_max_pad;        // xyz = AABB max, w = pad
};

// Per-cluster draw data — index buffer offset + count for indirect draw.
struct ClusterDrawInfo {
    uint    index_offset;        // byte offset into the global index buffer
    uint    index_count;         // number of indices (triangles * 3)
    uint    vertex_offset;       // base vertex (added to each index)
    uint    material_idx;        // material index for the cluster
};

// Maximum textures that can be bound in the cluster bindless pass.
// Must match the sampler2D array size in cluster_bindless.frag.
// Bistro has 600+ DDS textures; 256 covers the most common base-colour set
// without blowing the descriptor pool budget.
#define MAX_CLUSTER_TEXTURES 256

// Flags for BindlessMaterialParams.flags
#define BINDLESS_MAT_DOUBLE_SIDED   1   // bit 0: accept both face orientations (flip N on back face)
#define BINDLESS_MAT_ALPHA_MASK     2   // bit 1: discard if alpha < alpha_cutoff
#define BINDLESS_MAT_TRANSLUCENT    4   // bit 2: AlphaMode::Blend (glass / windows).
                                        // Currently the cluster pipeline draws this as
                                        // opaque too — the flag is set so the
                                        // "Translucent" render-debug mode can highlight
                                        // these materials, and so future work that adds
                                        // a real translucent pipeline pass has the data.

// Per-material colour data for the bindless cluster renderer.
//
// Struct size MUST stay aligned to 16 bytes (std430 needs no padding,
// but std140 binding contexts would).  Current size: 48 bytes.
// C++-side static_asserts in cluster_renderer.cpp guard the layout.
struct BindlessMaterialParams {
    vec4  base_color_factor;    // offset  0  RGBA base colour (linear)
    int   base_color_tex_idx;   // offset 16  index into base_color_textures[] (legacy bindless); -1 = no texture
    float alpha_cutoff;         // offset 20  ALPHA_MASK threshold
    int   flags;                // offset 24  BINDLESS_MAT_* bits
    int   normal_tex_idx;       // offset 28  index into normal_textures[] (legacy); -1 = no normal map
    // ── Runtime Virtual Texture IDs (one per layer) ──────────────
    // 0xFFFFFFFF = no VT registration → shader falls back to legacy
    // texture-array path.  When != INVALID, the VT path is used and
    // the bindless texture-array indices above are ignored for that
    // layer.  See vt_sample.glsl.h::vtSample* for the resolve.
    uint  albedo_vt_id;         // offset 32
    uint  normal_vt_id;         // offset 36
    uint  mr_ao_vt_id;          // offset 40
    uint  emissive_vt_id;       // offset 44
};

// Flattened BVH node for GPU traversal (iterative stack-based).
struct ClusterBVHNodeGPU {
    vec4    aabb_min_pad;        // xyz = min, w = left_child_idx (-1 if leaf)
    vec4    aabb_max_pad;        // xyz = max, w = right_child_idx (-1 if leaf)
    // For leaf: cluster_start/cluster_count encode the range into a
    // separate cluster_indices[] array. For inner: both are 0.
    uint    cluster_start;       // first index into cluster_indices[]
    uint    cluster_count;       // number of cluster indices in this leaf
    uint    pad0;
    uint    pad1;
};

// Push constants for the cluster culling compute shader.
struct ClusterCullPushConstants {
    mat4    view_proj;
    vec4    camera_pos_pad;      // xyz = camera world pos, w = pad
    uint    total_clusters;
    uint    total_bvh_nodes;
    float   lod_error_threshold; // screen-space error threshold for LOD
    uint    use_bvh;             // 0 = flat per-cluster, 1 = BVH traversal
    // 1 = sample the Hi-Z pyramid (built this frame from Phase A's depth)
    // and reject clusters whose bounding-sphere nearest point is behind
    // every visible surface in its footprint.  0 = skip the Hi-Z test
    // (plain frustum + cone cull only).  Toggled at runtime from the
    // cluster debug menu via ClusterRenderer::use_hiz_occlusion_cull_.
    uint    use_hiz_cull;
    // Two-pass occlusion phase selector:
    //   0 = single-pass legacy cull (forward path).
    //   1 = Phase A: gated on visibility bits, frustum+backface only,
    //                emits to indirect_draw_buffer_phase_a (binding 9/10).
    //   2 = Phase B: tests ALL clusters with frustum+backface+Hi-Z,
    //                emits to the standard opaque indirect (binding 2/3)
    //                AND atomicOr's the visibility bit for survivors so
    //                next frame's Phase A picks them up.
    // Phase A also atomicOr's its survivors' bits — that way the union
    // of A's emit set and B's emit set is the canonical "visible this
    // frame" signal, with no leak from the previous frame's bits.
    uint    cull_phase;
    // Two explicit pad uints before the next mat4.  std430 requires
    // mat4 to start on a 16-byte boundary; without these uints, GLSL
    // would insert 8 bytes of implicit padding (advancing
    // last_view_proj from offset 108 to 112) — but glm::mat4 in this
    // codebase has alignas 4 (no GLM_FORCE_DEFAULT_ALIGNED_GENTYPES),
    // so the C++ struct would put last_view_proj at 108.  Result: a
    // 4-byte size mismatch that pushes 188 bytes from CPU into a
    // 192-byte std430 layout — drivers handle this inconsistently and
    // on NV the entire push-constant view can corrupt, making
    // diagnostics like `cull_phase != 1u` evaluate the wrong way.
    // Two explicit pads bring last_view_proj to offset 112 in BOTH
    // layouts, eliminating the mismatch.
    uint    pad0;
    uint    pad1;
    // Reprojection matrix for the Hi-Z sample.  Named last_view_proj
    // for historical reasons — in the two-pass deferred path it's set
    // to the CURRENT frame's view_proj because the pyramid was built
    // from this frame's Phase A depth.  Legacy single-pass path passes
    // last frame's view_proj here to reproject against last-frame depth.
    mat4    last_view_proj;
    // x, y = Hi-Z mip 0 size in pixels (next-pow2 of swap-chain size).
    // z   = total mip count (so the shader can clamp the chosen mip).
    // w   = pad.
    vec4    hiz_size_mips_pad;
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

// Push constant for the sky envmap background fullscreen pass.
// Carries the rotation-only inverse view-projection so the fragment shader
// can reconstruct a world-space view direction per screen pixel, plus a
// runtime debug-mode selector exposed in the Skydome ImGui menu.
//
// debug_mode values (kept in sync with skybox_envmap.frag):
//   0 = normal (Reinhard tone-mapped)
//   1 = solid red          (fragment-shader smoke test)
//   3 = view_dir RGB       (camera-matrix smoke test)
//   4 = envmap raw         (HDR sample, no tone-map)
//   5 = envmap × 10000     (saturate sample, useful for low-radiance debug)
struct SkyboxEnvmapParams {
    mat4            inv_view_proj_relative;
    int             debug_mode;
    int             _pad0;
    int             _pad1;
    int             _pad2;
};

// Push constants for the dithered "mini-buffer" sky cubemap update.
// Each frame, only 1/64 of the full-res cubemap texels are recomputed at full
// quality: a (mini_size = cube_size/8)-resolution cubemap is fully refreshed
// and its texels are scattered into the corresponding 8x8 block in the
// full-res envmap, using a per-frame dither offset that cycles through all
// 64 unique positions over 64 frames.
struct SunSkyMiniParams {
    vec3            sun_pos;
    float           g;
    float           inv_rayleigh_scale_height;
    float           inv_mie_scale_height;
    int             cube_size;        // full-res cubemap face width (= mini_size * 8)
    int             mini_size;        // mini-buffer face width (= cube_size / 8)
    ivec2           dither_offset;    // (dx, dy) within the 8x8 block, 0..7
    int             frame_index;      // monotonic frame index (modulo 64 used in shader)
    // 1 on the very first dispatch for this buffer (== "first touch": the
    // image is in UNDEFINED layout / contains garbage memory).  In this
    // mode each compute thread evaluates one sample at its dither
    // position and broadcasts it to ALL 64 texels in the 8x8 block, so
    // the entire envmap is filled in a single cheap dispatch.  After that
    // the regular sparse 8x8 dither runs, refining individual texels.
    // Replaces the heavyweight `drawCubeSkyBox` bootstrap.
    int             is_first_touch;
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
    // When non-zero, the compute pass IGNORES game_objects_buffer_ and
    // writes an identity instance transform (zero translation, unit
    // rotation, scale 1) to every slot.  Used for drawables whose world
    // placement is driven entirely by the node-hierarchy root transform
    // (PlayerController-controlled player, hand-placed gltfs) — without
    // this, the shared game_objects_buffer_'s slot 0 (initialised to
    // camera_pos on frame 0 and drifting via gravity) gets re-applied
    // on top of the node translation, producing a double-transform that
    // throws the visible position 2× the camera offset away.
    uint            force_identity;
    uint            pad0;
    uint            pad1;
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
    vec3 baseColor;                 // getBaseColor()

    float reflectance;
    vec3 sheenColor;

    float sheenIntensity;
    vec3 clearcoatF0;

    float sheenRoughness;
    vec3 clearcoatF90;

    float anisotropy;
    vec3 clearcoatNormal;

    float clearcoatFactor;
    vec3 subsurfaceColor;

    float clearcoatRoughness;
    vec3 absorption;

    float subsurfaceScale;
    float subsurfaceDistortion;
    float subsurfacePower;
    float subsurfaceThickness;

    float thinFilmFactor;
    float thinFilmThickness;
    float thickness;
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

    vec3            specular_color;
    float           specular_exponent;

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
    uint            pad_2;
    float           normal_scale;
    float           occlusion_strength;

    vec3            emissive_factor;
    uint            tonemap_type;

    vec3            emissive_color;
    uint            pad_3;

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

struct BaseShapeVsPsData {
    vec3 vertex_position;
    vec2 vertex_tex_coord;
    vec3 vertex_normal;
    vec3 vertex_tangent;
    vec3 vertex_binormal;
};

struct HairVsPsData {
    vec3 vertex_position;
    vec4 vertex_tex_coord;
    vec3 vertex_normal;
    vec3 vertex_tangent;
    vec3 vertex_binormal;
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

struct ViewCameraParams {
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
struct ViewCameraInfo {
    mat4            view;
    mat4            proj;
    mat4            view_proj;
    mat4            inv_view_proj;
    mat4            inv_view_proj_relative;
    mat4            inv_view;
    mat4            inv_proj;
    // Previous frame's view_proj — captured BEFORE view_proj is
    // recomputed each frame (see ViewCamera::updateViewCameraInfo).
    // Consumed by any pass that needs screen-space motion vectors:
    // the cluster G-buffer's velocity attachment (RT3) computes
    // velocity = curNDC - prevNDC for each fragment, which downstream
    // passes (TAA, motion blur, temporal reprojection / upscaling)
    // can sample.  On the very first frame this equals view_proj,
    // so velocity reads zero everywhere — correct "no history yet".
    mat4            prev_view_proj;
    vec4            depth_params;
    vec3            position;
    uint            status;                   // 32 bits for status, todo.
    vec3            up_vector;
    float           yaw;
    vec3            facing_dir;
    float           pitch;
    vec2            mouse_pos;
    float           camera_follow_dist;
    uint            input_features;   // FEATURE_INPUT_* flags set per-frame by CPU
};

struct RuntimeLightsParams {
    mat4            light_view_proj[CSM_CASCADE_COUNT];
    // View-space far depths for each cascade.  Packed into vec4[2]
    // (8 floats, 32 bytes) so the std140-laid-out uniform buffer
    // doesn't bloat to one float per 16-byte slot.  CSM_CASCADE_COUNT
    // is 6 in practice; element 6 and 7 are unused.  Shaders index
    // as cascade_splits[i >> 2][i & 3].
    vec4            cascade_splits[2];
    // World-space corners of each cascade's main-camera frustum slab.
    // Indexed as cascade_slab_corners_ws[cascade * 8 + corner].  Order
    // matches computeCascadeMatrices's vs_corners layout:
    //   0 = near top-left      1 = near top-right
    //   2 = near bottom-left   3 = near bottom-right
    //   4 = far  top-left      5 = far  top-right
    //   6 = far  bottom-left   7 = far  bottom-right
    // Used by csm_silhouette_prepass.mesh to fill the in-frustum region
    // of each cascade's shadow map with depth=1 so that out-of-frustum
    // texels (still at the cleared 0.0) reject every shadow caster via
    // the LESS_OR_EQUAL depth test.  Hi-Z then propagates those rejects
    // to whole-tile primitive culling at the PD.
    vec4            cascade_slab_corners_ws[CSM_CASCADE_COUNT * 8];
    Light           lights[LIGHT_COUNT];
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