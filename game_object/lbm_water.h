#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace game_object {

// ── D2Q9 shallow-water LBM river surface ─────────────────────────────
// A camera-following lattice patch (grid_size^2 cells at cell_m
// spacing) advanced one LBM step per frame by lbm_water.comp — see
// that shader for the physics.  Nine distribution functions ride three
// RGBA32F textures, ping-ponged A→B each step; the surface output
// (xyz = ripple normal, w = height deviation) is a sampleable RGBA16F
// the terrain WATER_ATTR pass blends into its water normal wherever
// the patch covers the fragment.  A tiny SSBO publishes the patch
// region (origin/cell/size) to that pass.
class LbmWater {
    static constexpr uint32_t kGridSize = 512;   // 512 x 0.25 m = 128 m
    static constexpr float    kCellM    = 0.25f;

    std::shared_ptr<renderer::DescriptorSetLayout> desc_set_layout_;
    // parity 0: A source, B dest.  parity 1: swapped.
    std::shared_ptr<renderer::DescriptorSet>  desc_sets_[2];
    std::shared_ptr<renderer::PipelineLayout> pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>       pipeline_;

    // f0..f3 / f4..f7 / f8 distribution textures, two parities.
    renderer::TextureInfo f_tex_[2][3];
    std::shared_ptr<renderer::TextureInfo> surface_tex_;
    renderer::BufferInfo region_buffer_;      // vec4: origin.xz/cell/size
    // Static water-surface map (binding 7): a 1x1 zero fallback is
    // bound at construction; setWaterLevelMap swaps in the real
    // <stem>_hydro_pondz.png once the terrain apply has loaded it.
    // The sim derives its body force from THIS surface's gradient —
    // the LBM generates the flow field, nothing paints one for it.
    std::shared_ptr<renderer::Sampler> flow_sampler_;
    renderer::TextureInfo flow_fallback_tex_;
    bool level_map_bound_ = false;
    // GENERATED FLOWMAP (binding 8): the sim's per-cell macroscopic
    // velocity (m/s, world xz), written every step.  Sampleable by the
    // water shading through flowTexture().
    std::shared_ptr<renderer::TextureInfo> flow_tex_;

    glm::vec2 origin_ws_{0.0f};
    glm::vec2 prev_origin_ws_{0.0f};
    uint32_t  parity_ = 0;
    bool      needs_reset_ = true;
    float     time_ = 0.0f;

public:
    LbmWater(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Advance one LBM step.  The patch recentres on the camera; a jump
    // larger than the patch resets the lattice to rest.
    void update(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const glm::vec3& camera_pos,
        const glm::vec2& flow_dir,
        float delta_t);

    // Bind the STATIC water-surface map (u16 height — the same
    // <stem>_hydro_pondz.png the tile creator fills the water layer
    // from; lbm_water.comp binding 7).  Pass null to fall back to the
    // zero 1x1 (no force — legacy flow_dir drift).  Rewrites live
    // descriptor sets: only call when the device is idle (the
    // terrain-apply path is, via waitIdle).
    void setWaterLevelMap(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::ImageView>& level_view);

    const std::shared_ptr<renderer::TextureInfo>& surfaceTexture() const {
        return surface_tex_;
    }
    // The sim's GENERATED flowmap: per-cell velocity (m/s, world xz)
    // over the camera-following patch, updated every step.
    const std::shared_ptr<renderer::TextureInfo>& flowTexture() const {
        return flow_tex_;
    }
    const renderer::BufferInfo& regionBuffer() const {
        return region_buffer_;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}  // namespace game_object
}  // namespace engine
