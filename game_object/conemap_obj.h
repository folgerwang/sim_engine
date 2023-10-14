#pragma once
#include "renderer/renderer.h"
#include "plane.h"
#include "scene_rendering/prt_shadow.h"

namespace engine {
namespace scene_rendering {
    class PrtShadow;
}
namespace game_object {

class ConemapObj {
    std::shared_ptr<renderer::DescriptorSetLayout> gen_minmax_depth_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> gen_minmax_depth_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> gen_minmax_depth_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> gen_minmax_depth_pipeline_;

    std::shared_ptr<renderer::DescriptorSet> prt_shadow_cache_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> prt_shadow_cache_update_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> prt_shadow_gen_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> gen_prt_pack_info_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSet> pack_prt_tex_desc_set_;
    std::shared_ptr<renderer::TextureInfo> conemap_tex_;
    std::shared_ptr<renderer::TextureInfo> prt_pack_tex_;
    std::shared_ptr<renderer::TextureInfo> prt_pack_info_tex_;
    std::shared_ptr<renderer::TextureInfo> minmax_depth_tex_;

    uint32_t depth_channel_ = 0;
    bool is_height_map_ = false;
    float depth_scale_ = 0.0f;
    float shadow_intensity_ = 0.0f;
    float shadow_noise_thread_ = 0.0f;

public:
    ConemapObj(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& prt_bump_tex,
        const std::shared_ptr<scene_rendering::PrtShadow>& prt_shadow_gen,
        uint32_t depth_channel,
        bool is_height_depth,
        float depth_scale,
        float shadow_intensity,
        float shadow_noise_thread);

    void update(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::uvec2& src_buffer_size);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);

    inline float getDepthScale() {
        return depth_scale_;
    }

    inline bool isHeightMap() {
        return is_height_map_;
    }

    inline float getShadowNoiseThread() {
        return shadow_noise_thread_;
    }

    inline float getShadowIntensity() {
        return shadow_intensity_;
    }

    inline uint32_t getDepthChannel() {
        return depth_channel_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getPrtShadowGenTexDescSet() {
        return prt_shadow_gen_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getPrtShadowCacheTexDescSet() {
        return prt_shadow_cache_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getPrtShadowCacheUpdateTexDescSet() {
        return prt_shadow_cache_update_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getGenPrtPackInfoTexDescSet() {
        return gen_prt_pack_info_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getPackPrtTexDescSet() {
        return pack_prt_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::TextureInfo> getConemapTexture() {
        return conemap_tex_;
    }

    inline const std::shared_ptr<renderer::TextureInfo> getMinmaxDepthTexture() {
        return minmax_depth_tex_;
    }

    inline const std::shared_ptr<renderer::TextureInfo> getPackTexture() {
        return prt_pack_tex_;
    }

    inline const std::shared_ptr<renderer::TextureInfo> getPackInfoTexture() {
        return prt_pack_info_tex_;
    }
};

} // game_object
} // engine