#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class VolumeNoise {
#if 0
    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;
#endif

    renderer::TextureInfo detail_noise_tex_;
    renderer::TextureInfo rough_noise_tex_;

#if 0
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> cloud_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> cloud_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_pipeline_;
#endif

    std::shared_ptr<renderer::DescriptorSetLayout> noise_init_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> detail_noise_init_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> rough_noise_init_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> noise_init_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> noise_init_pipeline_;

public:
    VolumeNoise(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::DescriptorSetLayout>& ibl_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::Sampler>& point_clamp_texture_sampler,
        const glm::uvec2& display_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::Sampler>& point_clamp_texture_sampler,
        const glm::uvec2& display_size);

    void initNoiseTexture(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    void update();

    void destroy(const std::shared_ptr<renderer::Device>& device);

    const renderer::TextureInfo& getDetailNoiseTexture() {
        return detail_noise_tex_;
    }

    const renderer::TextureInfo& getRoughNoiseTexture() {
        return rough_noise_tex_;
    }
};

}// namespace scene_rendering
}// namespace engine
