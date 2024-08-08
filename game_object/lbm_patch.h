#pragma once
#include "renderer/renderer.h"
#include "plane.h"
#include "box.h"

namespace engine {
namespace game_object {

class LbmPatch {
    std::shared_ptr<renderer::DescriptorSet>  lbm_patch_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> lbm_patch_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> lbm_patch_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> lbm_patch_pipeline_;

    std::shared_ptr<renderer::TextureInfo> lbm_patch_tex_;

public:
    LbmPatch(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const glm::uvec2& patch_size);

    void update(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list);

    inline const auto getLbmPatchTexture() {
        return lbm_patch_tex_;
    }

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine