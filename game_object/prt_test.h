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
    std::shared_ptr<renderer::TextureInfo> prt_texes_[6];
public:
    PrtTest(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size,
        std::shared_ptr<Plane> unit_plane,
        const engine::renderer::TextureInfo& prt_base_tex,
        const engine::renderer::TextureInfo& prt_bump_tex);
    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        std::shared_ptr<Plane> unit_plane);

    inline std::shared_ptr<renderer::TextureInfo> getConemapTex() const { return cone_map_tex_; }
    inline std::shared_ptr<renderer::TextureInfo> getPrtTex() const { return prt_tex_; }
    inline std::shared_ptr<renderer::TextureInfo> getPrtTex(int index) const { return prt_texes_[index]; }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine