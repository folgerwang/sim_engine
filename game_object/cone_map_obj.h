#pragma once
#include "renderer/renderer.h"
#include "plane.h"
#include "scene_rendering/prt.h"

namespace engine {
namespace scene_rendering {
    class Prt;
}
namespace game_object {

class ConeMapObj {
    std::shared_ptr<renderer::DescriptorSet>  prt_gen_tex_desc_set_;
    renderer::DescriptorSetList prt_ds_final_tex_desc_sets_;
    std::shared_ptr<renderer::DescriptorSet>  prt_pack_tex_desc_set_;
    std::shared_ptr<renderer::TextureInfo> cone_map_tex_;
    std::shared_ptr<renderer::TextureInfo> prt_pack_tex_;
    std::shared_ptr<renderer::BufferInfo> prt_minmax_buffer_;

    uint32_t depth_channel_ = 0;
    bool is_height_map_ = false;
    float depth_scale_ = 0.0f;
    float shadow_intensity_ = 0.0f;
    float shadow_noise_thread_ = 0.0f;

public:
    ConeMapObj(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& prt_bump_tex,
        const std::shared_ptr<scene_rendering::Prt>& prt_gen,
        uint32_t depth_channel,
        bool is_height_depth,
        float depth_scale,
        float shadow_intensity,
        float shadow_noise_thread);

    void destroy(const std::shared_ptr<renderer::Device>& device);

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

    inline const std::shared_ptr<renderer::DescriptorSet>& getPrtGenTexDescSet() {
        return prt_gen_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getPrtDsFinalTexDescSet(uint32_t index) {
        return prt_ds_final_tex_desc_sets_[index];
    }

    inline const std::shared_ptr<renderer::DescriptorSet>& getPrtPackTexDescSet() {
        return prt_pack_tex_desc_set_;
    }

    inline const std::shared_ptr<renderer::TextureInfo> getConemapTexture() {
        return cone_map_tex_;
    }

    inline const std::shared_ptr<renderer::TextureInfo> getPackTexture() {
        return prt_pack_tex_;
    }

    inline const std::shared_ptr<renderer::BufferInfo> getMinmaxBuffer() {
        return prt_minmax_buffer_;
    }
};

} // game_object
} // engine