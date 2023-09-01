#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
    namespace scene_rendering {

        class Prt {
            std::shared_ptr<renderer::DescriptorSetLayout> prt_desc_set_layout_;
            std::shared_ptr<renderer::DescriptorSet> prt_tex_desc_set_;
            std::shared_ptr<renderer::PipelineLayout> prt_pipeline_layout_;
            std::shared_ptr<renderer::Pipeline> prt_pipeline_;

        public:
            Prt(
                const renderer::DeviceInfo& device_info,
                const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
                const std::shared_ptr<renderer::Sampler>& texture_sampler,
                const renderer::TextureInfo& bump_tex,
                const renderer::TextureInfo& prt_tex,
                const renderer::TextureInfo& prt_tex_1,
                const renderer::TextureInfo& prt_tex_2,
                const renderer::TextureInfo& prt_tex_3,
                const renderer::TextureInfo& prt_tex_4,
                const renderer::TextureInfo& prt_tex_5,
                const renderer::TextureInfo& prt_tex_6);

            void update(
                const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
                const renderer::TextureInfo& prt_tex,
                const renderer::TextureInfo& prt_tex_1,
                const renderer::TextureInfo& prt_tex_2,
                const renderer::TextureInfo& prt_tex_3,
                const renderer::TextureInfo& prt_tex_4,
                const renderer::TextureInfo& prt_tex_5,
                const renderer::TextureInfo& prt_tex_6);

            void destroy(const std::shared_ptr<renderer::Device>& device);
        };

    }// namespace scene_rendering
}// namespace engine
