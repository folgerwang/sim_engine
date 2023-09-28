#pragma once
#include "renderer/renderer.h"
#include "plane.h"
#include "conemap_obj.h"

namespace engine {
namespace game_object {

class ConemapTest {
    std::shared_ptr<renderer::DescriptorSet>  prt_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> prt_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> prt_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> prt_pipeline_;
    std::shared_ptr<renderer::BufferInfo> uniform_buffer_;

public:
    ConemapTest(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& prt_base_tex,
        const renderer::TextureInfo& prt_normal_tex,
        const renderer::TextureInfo& prt_orh_tex,
        const std::shared_ptr<game_object::ConemapObj>& conemap_obj,
        const glm::uvec2& display_size,
        std::shared_ptr<Plane> unit_plane);

    void draw(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        std::shared_ptr<Plane> unit_plane,
        const std::shared_ptr<game_object::ConemapObj>& conemap_obj);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine