#pragma once
#include "renderer/renderer.h"
#include "patch.h"

namespace engine {
namespace game_object {

class ObjectMesh {
    std::vector<std::shared_ptr<Patch>> patches_;
    std::shared_ptr<renderer::DescriptorSet>  object_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> object_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> object_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> object_pipeline_;
    std::shared_ptr<renderer::TextureInfo> diffuse_tex_;
    std::shared_ptr<renderer::TextureInfo> normal_tex_;
    std::shared_ptr<renderer::TextureInfo> glossiness_tex_;
    std::shared_ptr<renderer::TextureInfo> specular_tex_;

public:
    ObjectMesh() {}

    void loadObjectFile(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::string& object_name,
        const std::string& shader_name,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    void draw(std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        uint32_t draw_idx = -1);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine