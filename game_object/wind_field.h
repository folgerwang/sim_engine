#pragma once
#include <vector>
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace game_object {

// ── Wind clipmap: the D2Q9 lattice at kWindClipLevels nested scales ──
// Level 0 is a 512-cell lattice at 1 m centred on the camera; each
// level above it is the same lattice at 4x the cell, so the top level
// spans the whole 32 km map.  Every level is one wind_patch.comp step
// per frame; a level's inflow is the level above it (feathered at that
// level's rim), the top level's inflow is the world airflow field the
// weather system advances.  Consumers read the published rg16f ARRAY
// (layer = level) + WindClipInfo through sampleWindClip() /
// vegWindAt(); the single-level windTexture()/regionBuffer() pair is
// level 0, kept for the water and grass paths that bind it directly.
//
// Moving actors (cars, people) stir the air: the citizen / vehicle
// systems push WindInjectors each frame through addInjector(); update()
// uploads the nearest kWindInjectorMax and every level applies them.
//
// The output textures and info buffers exist from construction (so
// descriptor sets can bind them before the sim has ever stepped: they
// read as zero wind / dead regions until then); the sim only steps
// once bindSources() has pointed it at the airflow field and the rock
// layer, and only if wind_patch_comp.spv was found (the constructor
// never throws for a missing shader — the field just stays calm).
class WindField {
public:
    static constexpr uint32_t kLevels   = kWindClipLevels;
    static constexpr uint32_t kGridSize = 512;
    static float cellMeters(uint32_t level) {   // 1, 4, 16, 64 m
        return 1.0f * float(1u << (2u * level));
    }

private:
    std::shared_ptr<renderer::DescriptorSetLayout> desc_set_layout_;
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;
    std::shared_ptr<renderer::DescriptorSet>  desc_sets_[kWindClipLevels][2];
    std::shared_ptr<renderer::PipelineLayout> pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>       pipeline_;

    // f0..f3 / f4..f7 / f8 distributions, per level, two parities.
    renderer::TextureInfo f_tex_[kWindClipLevels][2][3];
    // THE OUTPUT: rg16f array, layer L = level L's wind (m/s world xz).
    std::shared_ptr<renderer::Image>        wind_image_;
    std::shared_ptr<renderer::DeviceMemory> wind_memory_;
    std::shared_ptr<renderer::ImageView>    wind_array_view_;             // all layers, sampled
    std::shared_ptr<renderer::ImageView>    wind_layer_views_[kWindClipLevels];  // one layer each
    // Level 0 as a plain 2D texture (view of layer 0) for the legacy
    // single-patch consumers.  image/memory are NOT owned here.
    std::shared_ptr<renderer::TextureInfo> wind_tex_;
    renderer::BufferInfo region_buffer_;      // level 0: origin.x/origin.z/cell/grid
    renderer::BufferInfo clip_info_buffer_;   // WindClipInfo, every level
    renderer::BufferInfo injector_buffer_;    // WindInjector[kWindInjectorMax]
    std::shared_ptr<renderer::Sampler> sampler_;

    glm::vec2 origin_ws_[kWindClipLevels]{};
    glm::vec2 prev_origin_ws_[kWindClipLevels]{};
    uint32_t  parity_ = 0;
    bool      needs_reset_ = true;
    bool      sources_bound_ = false;
    bool      pipeline_ok_ = false;
    float     time_ = 0.0f;

    struct Injector { glm::vec2 pos, vel; float radius, coupling; };
    static std::vector<Injector> s_injectors_;

public:
    WindField(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Point the sim at the coarse airflow field (sampler3D) and the
    // terrain rock layer (obstacle test).  Rewrites live descriptor
    // sets — call when the device is idle (terrain apply is, via
    // waitIdle).  Until called, update() publishes dead regions and
    // dispatches nothing, and the wind textures stay zero.
    void bindSources(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::ImageView>& airflow_view,
        const std::shared_ptr<renderer::ImageView>& rock_layer_view);

    // Advance every level one step; recentres on the camera.
    // world_min/world_range are the airflow field's coverage (the
    // terrain footprint).
    void update(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const glm::vec3& camera_pos,
        const glm::vec3& world_min,
        const glm::vec3& world_range,
        float delta_t);

    // A moving actor's footprint on the air this frame: world xz
    // position and velocity (m/s), gaussian radius (m) and the
    // coupling — the fraction of the actor's velocity the air at the
    // centre takes on per second.  Cleared by update().
    static void addInjector(const glm::vec2& pos_xz, const glm::vec2& vel_xz,
                            float radius_m, float coupling);

    // Level 0 (legacy single-patch consumers).
    const std::shared_ptr<renderer::TextureInfo>& windTexture() const {
        return wind_tex_;
    }
    const renderer::BufferInfo& regionBuffer() const {
        return region_buffer_;
    }
    // The clipmap (set 0: WIND_CLIP_TEX_INDEX / WIND_CLIP_INFO_INDEX).
    const std::shared_ptr<renderer::ImageView>& clipArrayView() const {
        return wind_array_view_;
    }
    const renderer::BufferInfo& clipInfoBuffer() const {
        return clip_info_buffer_;
    }
    const std::shared_ptr<renderer::Sampler>& sampler() const {
        return sampler_;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}  // namespace game_object
}  // namespace engine
