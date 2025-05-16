#pragma once
#include "renderer/renderer.h"
#include "plane.h"
#include "box.h"

namespace engine {
namespace game_object {

class LbmTest {
    std::shared_ptr<renderer::DescriptorSet>  lbm_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> lbm_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> lbm_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> lbm_pipeline_;

public:
    LbmTest(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::TextureInfo>& lbm_patch_tex,
        const glm::uvec2& display_size,
        std::shared_ptr<Plane> unit_plane);

    void draw(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        std::shared_ptr<Plane> unit_plane,
        std::shared_ptr<Box> unit_box,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine