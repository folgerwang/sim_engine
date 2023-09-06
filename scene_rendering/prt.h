#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
    namespace scene_rendering {

        class Prt {
            std::shared_ptr<renderer::DescriptorSetLayout> prt_desc_set_layout_;
            std::shared_ptr<renderer::PipelineLayout> prt_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_pipeline_;
            std::shared_ptr<renderer::DescriptorSet> prt_tex_desc_set_;

            std::shared_ptr<renderer::DescriptorSetLayout> prt_ds_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSetLayout> prt_ds_f_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSet> prt_ds_tex_desc_set_;
            std::shared_ptr<renderer::DescriptorSet> prt_ds_f_tex_desc_set_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_s_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_s_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_pipeline_;
            std::shared_ptr<renderer::PipelineLayout> prt_ds_f_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_ds_f_pipeline_;

            std::array<std::shared_ptr<renderer::TextureInfo>, 7> prt_texes_;
            std::array<std::shared_ptr<renderer::TextureInfo>, 7> prt_ds1_texes_;
            std::array<std::shared_ptr<renderer::TextureInfo>, 7> prt_ds2_texes_;
            std::shared_ptr<renderer::BufferInfo> prt_minmax_buffer_;

        public:
            Prt(
                const renderer::DeviceInfo& device_info,
                const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
                const glm::uvec2& buffer_size);

            void update(
                const std::shared_ptr<renderer::Device>& device,
                const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
                const std::shared_ptr<renderer::Sampler>& texture_sampler,
                const renderer::TextureInfo& bump_tex);

            inline std::array<std::shared_ptr<renderer::TextureInfo>, 7> getPrtTexes() const {
                return prt_texes_;
            }

            void destroy(const std::shared_ptr<renderer::Device>& device);
        };

    }// namespace scene_rendering
}// namespace engine
