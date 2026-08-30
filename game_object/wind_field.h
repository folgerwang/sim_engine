#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace game_object {

// ── Two-tier wind: the FINE tier host ────────────────────────────────
// A camera-following D2Q9 lattice (wind_patch.comp) solving the
// horizontal wind on one near-ground slice over a 512 m window; the
// COARSE tier is the world-wide airflow field the weather system
// already advances.  Deliberately shaped like LbmWater — same
// ping-ponged distribution textures, same recentre-and-shift update —
// because that machinery is proven in this engine.
//
// Consumers never touch this class: they sample the published wind
// texture + region through windAt() in weather/wind_field.glsl.h.  The
// output texture and region buffer exist from construction (so
// descriptor sets can bind them before the sim has ever stepped: the
// texture holds zero wind until then); the sim itself only steps once
// bindSources() has pointed it at the airflow field and the rock
// layer.
class WindField {
    static constexpr uint32_t kGridSize = 512;   // 512 x 1 m = 512 m
    static constexpr float    kCellM    = 1.0f;

    std::shared_ptr<renderer::DescriptorSetLayout> desc_set_layout_;
    // Retained for bindSources(): the compute sets are allocated THERE,
    // once the airflow + rock views exist, not at construction.
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool_;
    std::shared_ptr<renderer::DescriptorSet>  desc_sets_[2];
    std::shared_ptr<renderer::PipelineLayout> pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>       pipeline_;

    // f0..f3 / f4..f7 / f8 distributions, two parities (ping-pong).
    renderer::TextureInfo f_tex_[2][3];
    // THE OUTPUT: per-cell wind velocity, m/s world xz (rg16f).
    std::shared_ptr<renderer::TextureInfo> wind_tex_;
    renderer::BufferInfo region_buffer_;      // vec4: origin.x/cell/origin.z/grid
    std::shared_ptr<renderer::Sampler> sampler_;

    glm::vec2 origin_ws_{0.0f};
    glm::vec2 prev_origin_ws_{0.0f};
    uint32_t  parity_ = 0;
    bool      needs_reset_ = true;
    bool      sources_bound_ = false;
    float     time_ = 0.0f;

public:
    WindField(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Point the sim at the coarse airflow field (sampler3D) and the
    // terrain rock layer (obstacle test).  Rewrites live descriptor
    // sets — call when the device is idle (terrain apply is, via
    // waitIdle).  Until called, update() publishes the region but
    // dispatches nothing, and the wind texture stays zero.
    void bindSources(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::ImageView>& airflow_view,
        const std::shared_ptr<renderer::ImageView>& rock_layer_view);

    // Advance one step; recentres on the camera.  world_min/world_range
    // are the airflow field's coverage (the terrain footprint).
    void update(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const glm::vec3& camera_pos,
        const glm::vec3& world_min,
        const glm::vec3& world_range,
        float delta_t);

    const std::shared_ptr<renderer::TextureInfo>& windTexture() const {
        return wind_tex_;
    }
    const renderer::BufferInfo& regionBuffer() const {
        return region_buffer_;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}  // namespace game_object
}  // namespace engine
