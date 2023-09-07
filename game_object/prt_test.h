#pragma once
#include "renderer/renderer.h"
#include "plane.h"

namespace engine {
namespace game_object {

class PrtTest {
    std::shared_ptr<renderer::DescriptorSet>  prt_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> prt_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> prt_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> prt_pipeline_;
    std::shared_ptr<renderer::TextureInfo> cone_map_tex_;
    std::shared_ptr<renderer::TextureInfo> prt_tex_;
public:
    PrtTest(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size,
        const glm::uvec2& buffer_size,
        std::shared_ptr<Plane> unit_plane);
    void draw(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        std::shared_ptr<Plane> unit_plane,
        const renderer::TextureInfo& prt_base_tex,
        const renderer::TextureInfo& prt_bump_tex,
        const std::shared_ptr<renderer::TextureInfo>& prt_packed_texture,
        const std::shared_ptr<renderer::BufferInfo>& prt_minmax_buffer);

    inline std::shared_ptr<renderer::TextureInfo> getConemapTex() const { return cone_map_tex_; }
    inline std::shared_ptr<renderer::TextureInfo> getPrtTex() const { return prt_tex_; }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine