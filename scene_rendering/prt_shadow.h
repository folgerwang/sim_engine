#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

#include "game_object/conemap_obj.h"

namespace engine {
    namespace game_object {
        class ConemapObj;
    }
    namespace scene_rendering {

        class PrtShadow {
            const uint32_t s_max_prt_buffer_size = 4096;

            std::shared_ptr<renderer::DescriptorSetLayout> prt_shadow_cache_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_shadow_cache_update_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_shadow_gen_with_cache_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_shadow_gen_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_ds_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> gen_prt_pack_info_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> pack_prt_desc_set_layout_;
            std::shared_ptr<renderer::PipelineLayout> prt_shadow_cache_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_shadow_cache_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_shadow_cache_update_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_shadow_cache_update_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_shadow_gen_with_cache_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_shadow_gen_with_cache_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_shadow_gen_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_shadow_gen_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_first_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_first_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> gen_prt_pack_info_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> gen_prt_pack_info_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> pack_prt_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> pack_prt_pipeline_;
            std::shared_ptr<renderer::DescriptorSet> prt_shadow_gen_with_cache_tex_desc_set_;
            std::shared_ptr<renderer::DescriptorSet> prt_ds_tex_desc_set_;

            std::shared_ptr<renderer::TextureInfo> prt_texes_;
            std::shared_ptr<renderer::TextureInfo> prt_ds_texes_;
            std::shared_ptr<renderer::TextureInfo> prt_shadow_cache_texes_;

        public:
            PrtShadow(
                const std::shared_ptr<renderer::Device>& device,
                const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
                const std::shared_ptr<renderer::Sampler>& texture_sampler);

            void update(
                const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
                const std::shared_ptr<game_object::ConemapObj>& conemap_obj);

            inline const std::shared_ptr<renderer::TextureInfo>& getPrtTextures() {
                return prt_texes_;
            }

            inline const std::shared_ptr<renderer::TextureInfo>& getPrtDsTextures() {
                return prt_ds_texes_;
            }

            inline const std::shared_ptr<renderer::TextureInfo>& getPrtShadowCacheTextures() {
                return prt_shadow_cache_texes_;
            }

            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPrtShadowGenDescSetLayout() {
                return prt_shadow_gen_desc_set_layout_;
            }

            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPrtShadowCacheDescSetLayout() {
                return prt_shadow_cache_desc_set_layout_;
            }

            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPrtShadowCacheUpdateDescSetLayout() {
                return prt_shadow_cache_update_desc_set_layout_;
            }
                
            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getGenPrtPackInfoDescSetLayout() {
                return gen_prt_pack_info_desc_set_layout_;
            }

            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPackPrtDescSetLayout() {
                return pack_prt_desc_set_layout_;
            }

            void destroy(const std::shared_ptr<renderer::Device>& device);
        };

    }// namespace scene_rendering
}// namespace engine
