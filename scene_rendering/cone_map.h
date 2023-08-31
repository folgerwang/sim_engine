#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class ConeMap {
    std::shared_ptr<renderer::DescriptorSetLayout> conemap_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> conemap_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> conemap_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> conemap_pipeline_;

public:
    ConeMap(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& bump_tex,
        const renderer::TextureInfo& conemap_tex);

    void update();

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
