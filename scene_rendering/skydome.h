#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class Skydome {
    glm::vec3 sun_dir_ = glm::vec3(0);

    renderer::BufferInfo vertex_buffer_;
    renderer::BufferInfo index_buffer_;

    float g_ = 0.562f;
    float rayleigh_scale_height_ = kRayleighScaleHeight;
    float mie_scale_height_ = kMieScaleHeight;
    float lut_rayleigh_scale_height_ = 0.0f;
    float lut_mie_scale_height_ = 0.0f;

    renderer::TextureInfo sky_scattering_lut_tex_;
    renderer::TextureInfo sky_scattering_lut_sum_tex_;

    std::shared_ptr<renderer::DescriptorSetLayout> skybox_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> skybox_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> skybox_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> cube_skybox_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> skybox_pipeline_;
    std::shared_ptr<renderer::Pipeline> cube_skybox_pipeline_;

    std::shared_ptr<renderer::DescriptorSet> sky_scattering_lut_first_pass_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> sky_scattering_lut_sum_pass_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> sky_scattering_lut_final_pass_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> sky_scattering_lut_first_pass_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> sky_scattering_lut_sum_pass_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> sky_scattering_lut_final_pass_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> sky_scattering_lut_first_pass_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> sky_scattering_lut_sum_pass_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> sky_scattering_lut_final_pass_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> sky_scattering_lut_first_pass_pipeline_;
    std::shared_ptr<renderer::Pipeline> sky_scattering_lut_sum_pass_pipeline_;
    std::shared_ptr<renderer::Pipeline> sky_scattering_lut_final_pass_pipeline_;


public:
    Skydome(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size,
        const uint32_t& cube_size);

    inline glm::vec3 getSunDir() {
        return sun_dir_;
    }

    inline float& getRayleighScaleHeight() {
        return rayleigh_scale_height_;
    }

    inline float& getMieScaleHeight() {
        return mie_scale_height_;
    }

    inline float& getG() {
        return g_;
    }

    inline const std::shared_ptr<renderer::ImageView>& getScatteringLutTex() {
        return sky_scattering_lut_tex_.view;
    }

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& display_size);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    void drawCubeSkyBox(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::TextureInfo& rt_envmap_tex,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void update(float latitude, float longtitude, int d, int h, int m, int s);

    void updateSkyScatteringLut(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
