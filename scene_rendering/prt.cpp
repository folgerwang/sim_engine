#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "prt.h"

namespace {
    namespace er = engine::renderer;

    er::WriteDescriptorList addPrtTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set,
        const std::shared_ptr<er::Sampler>& texture_sampler,
        const std::shared_ptr<er::ImageView>& src_image,
        const std::shared_ptr<er::ImageView>& dst_image) {
        er::WriteDescriptorList descriptor_writes;
        descriptor_writes.reserve(2);

        // envmap texture.
        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            SRC_TEX_INDEX,
            texture_sampler,
            src_image,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // envmap texture.
        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX,
            nullptr,
            dst_image,
            er::ImageLayout::GENERAL);

        return descriptor_writes;
    }

    std::shared_ptr<er::PipelineLayout>
        createPrtPipelineLayout(
            const std::shared_ptr<er::Device>& device,
            const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
        er::PushConstantRange push_const_range{};
        push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
        push_const_range.offset = 0;
        push_const_range.size = sizeof(glsl::PrtParams);

        return device->createPipelineLayout(
            { desc_set_layout },
            { push_const_range });
    }

} // namespace

namespace engine {
namespace scene_rendering {

Prt::Prt(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& bump_tex,
    const renderer::TextureInfo& prt_tex) {

    const auto& device = device_info.device;

    prt_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
                renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    prt_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto prt_texture_descs = addPrtTextures(
        prt_tex_desc_set_,
        texture_sampler,
        bump_tex.view,
        prt_tex.view);
    device->updateDescriptorSets(prt_texture_descs);

    prt_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_desc_set_layout_);

    prt_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_pipeline_layout_,
            "prt_gen_comp.spv");

}

void Prt::update() {
}

void Prt::destroy(
    const std::shared_ptr<renderer::Device>& device) {
}

}//namespace scene_rendering
}//namespace engine
