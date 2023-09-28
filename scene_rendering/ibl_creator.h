#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace scene_rendering {

class IblCreator {
    renderer::TextureInfo panorama_tex_;
    renderer::TextureInfo rt_envmap_tex_;
    renderer::TextureInfo tmp_ibl_diffuse_tex_;
    renderer::TextureInfo tmp_ibl_specular_tex_;
    renderer::TextureInfo tmp_ibl_sheen_tex_;
    renderer::TextureInfo rt_ibl_diffuse_tex_;
    renderer::TextureInfo rt_ibl_specular_tex_;
    renderer::TextureInfo rt_ibl_sheen_tex_;

    std::shared_ptr<renderer::DescriptorSet> envmap_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_diffuse_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_specular_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> ibl_sheen_tex_desc_set_;

    std::shared_ptr<renderer::DescriptorSetLayout> ibl_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> ibl_comp_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> ibl_comp_pipeline_layout_;

    std::shared_ptr<renderer::Pipeline> envmap_pipeline_;
    std::shared_ptr<renderer::Pipeline> lambertian_pipeline_;
    std::shared_ptr<renderer::Pipeline> ggx_pipeline_;
    std::shared_ptr<renderer::Pipeline> charlie_pipeline_;
    std::shared_ptr<renderer::Pipeline> blur_comp_pipeline_;

public:
    IblCreator(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const uint32_t& cube_size);

    inline const renderer::TextureInfo& getEnvmapTexture() const
    {
        return rt_envmap_tex_;
    }

    inline std::shared_ptr<renderer::DescriptorSetLayout> getIblDescSetLayout() const
    {
        return ibl_desc_set_layout_;
    }

    inline std::shared_ptr<renderer::DescriptorSet> getEnvmapTexDescSet() const
    {
        return envmap_tex_desc_set_;
    }

    void createCubeTextures(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const uint32_t& cube_size);

    void addToGlobalTextures(
        renderer::WriteDescriptorList& descriptor_writes,
        const std::shared_ptr<renderer::DescriptorSet>& description_set,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void createIblGraphicsPipelines(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
        const uint32_t& cube_size);

    void createDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void drawEnvmapFromPanoramaImage(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void createIblDiffuseMap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void createIblSpecularMap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void createIblSheenMap(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
        const std::vector<renderer::ClearValue>& clear_values,
        const uint32_t& cube_size);

    void blurIblMaps(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const uint32_t& cube_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
