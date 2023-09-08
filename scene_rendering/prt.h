#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

#include "game_object/cone_map_obj.h"

namespace engine {
    namespace game_object {
        class ConeMapObj;
    }
    namespace scene_rendering {

        class Prt {
            const uint32_t s_max_prt_buffer_size = 4096;

            std::shared_ptr<renderer::DescriptorSetLayout> prt_gen_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_ds_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_ds_final_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_pack_desc_set_layout_;
            std::shared_ptr<renderer::PipelineLayout> prt_gen_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_gen_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_first_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_first_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_final_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_final_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_pack_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_pack_pipeline_;
            renderer::DescriptorSetList prt_ds_tex_desc_sets_;

            std::array<std::shared_ptr<renderer::TextureInfo>, 7> prt_texes_;
            std::array<std::shared_ptr<renderer::TextureInfo>, 7> prt_ds_texes_[2];

        public:
            Prt(
                const renderer::DeviceInfo& device_info,
                const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
                const std::shared_ptr<renderer::Sampler>& texture_sampler);

            void update(
                const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
                const std::shared_ptr<game_object::ConeMapObj>& cone_map_obj);

            inline const std::array<std::shared_ptr<renderer::TextureInfo>, 7>& getPrtTextures() {
                return prt_texes_;
            }

            inline const std::array<std::shared_ptr<renderer::TextureInfo>, 7>& getPrtDsTextures(uint index) {
                return prt_ds_texes_[index];
            }

            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPrtGenDescSetLayout() {
                return prt_gen_desc_set_layout_;
            }
                
            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPrtDsFinalDescSetLayout() {
                return prt_ds_final_desc_set_layout_;
            }

            inline const std::shared_ptr<renderer::DescriptorSetLayout>& getPrtPackDescSetLayout() {
                return prt_pack_desc_set_layout_;
            }

            void destroy(const std::shared_ptr<renderer::Device>& device);
        };

    }// namespace scene_rendering
}// namespace engine
