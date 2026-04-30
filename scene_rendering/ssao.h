#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class SSAO {
    // AO textures: raw output and blurred result.
    renderer::TextureInfo ao_raw_tex_;
    renderer::TextureInfo ao_blurred_tex_;

    // 4×4 random rotation noise texture (uploaded once).
    renderer::TextureInfo noise_tex_;

    // ── SSAO generation pass ──
    std::shared_ptr<renderer::DescriptorSetLayout> ssao_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       ssao_desc_set_;
    std::shared_ptr<renderer::PipelineLayout>      ssao_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            ssao_pipeline_;

    // ── Bilateral blur pass ──
    std::shared_ptr<renderer::DescriptorSetLayout> blur_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       blur_h_desc_set_;  // horizontal
    std::shared_ptr<renderer::DescriptorSet>       blur_v_desc_set_;  // vertical
    std::shared_ptr<renderer::PipelineLayout>      blur_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            blur_pipeline_;

    // ── Apply pass (multiply AO into HDR color) ──
    std::shared_ptr<renderer::DescriptorSetLayout> apply_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet>       apply_desc_set_;
    std::shared_ptr<renderer::PipelineLayout>      apply_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline>            apply_pipeline_;

    void createNoiseTex(const std::shared_ptr<renderer::Device>& device);

public:
    // Runtime-tunable parameters (exposed to ImGui via Menu).
    float radius    = 0.5f;
    float bias      = 0.025f;
    float power     = 1.5f;
    float intensity = 1.0f;
    float strength  = 1.0f;
    int   kernel_size = 32;
    bool  enabled   = true;

    SSAO(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const std::shared_ptr<renderer::ImageView>& hdr_color_view,
        const glm::uvec2& display_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& depth_view,
        const std::shared_ptr<renderer::ImageView>& hdr_color_view,
        const glm::uvec2& display_size);

    void render(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& view_desc_set,
        const std::shared_ptr<renderer::Image>& hdr_color_image,
        const glm::uvec2& display_size);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // namespace scene_rendering
} // namespace engine
