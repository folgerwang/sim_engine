#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

#include "game_object/conemap_obj.h"

namespace engine {
namespace game_object {
    class ConemapObj;
}
namespace scene_rendering {

class Conemap {
    std::shared_ptr<renderer::DescriptorSetLayout> conemap_gen_init_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSetLayout> conemap_gen_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> conemap_gen_init_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> conemap_minmax_depth_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> conemap_gen_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> conemap_gen_init_pipeline_layout_;
    std::shared_ptr<renderer::PipelineLayout> conemap_gen_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> conemap_gen_init_pipeline_;
    std::shared_ptr<renderer::Pipeline> conemap_minmax_depth_pipeline_;
    std::shared_ptr<renderer::Pipeline> conemap_gen_pipeline_;
    std::shared_ptr<renderer::TextureInfo> conemap_minmax_depth_tex_;

public:
    Conemap(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& bump_tex,
        const renderer::TextureInfo& conemap_tex);

    void update(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<game_object::ConemapObj>& conemap_obj);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
