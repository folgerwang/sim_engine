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
        const std::vector<std::shared_ptr<er::TextureInfo>>& prt_texes) {
        er::WriteDescriptorList descriptor_writes;
//        descriptor_writes.reserve(2);
        descriptor_writes.reserve(8);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            SRC_TEX_INDEX,
            texture_sampler,
            src_image,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        for (int i = 0; i < prt_texes.size(); ++i) {
            er::Helper::addOneTexture(
                descriptor_writes,
                description_set,
                er::DescriptorType::STORAGE_IMAGE,
                DST_TEX_INDEX_0 + i,
                nullptr,
                prt_texes[i]->view,
                er::ImageLayout::GENERAL);
        }

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
    const std::vector<std::shared_ptr<renderer::TextureInfo>> prt_texes) {

    const auto& device = device_info.device;

    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(8);
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER));

    for (int i = 0; i < prt_texes.size(); ++i) {
        bindings.push_back(
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_0 + i,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE));
    }

    prt_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    prt_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto prt_texture_descs = addPrtTextures(
        prt_tex_desc_set_,
        texture_sampler,
        bump_tex.view,
        /*prt_tex.view*/
        prt_texes);
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

void Prt::update(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::vector<std::shared_ptr<renderer::TextureInfo>> prt_texes) {

    std::vector<std::shared_ptr<renderer::Image>> images;
    images.reserve(prt_texes.size());
    for (const auto& tex : prt_texes) {
        images.push_back(tex->image);
    }

    renderer::helper::transitMapTextureToStoreImage(cmd_buf, images);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        prt_pipeline_);
    glsl::PrtParams params = {};
    params.size = glm::uvec2(prt_texes[0]->size);
    params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        prt_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        prt_pipeline_layout_,
        { prt_tex_desc_set_ });

    cmd_buf->dispatch(
        (params.size.x + 7) / 8,
        (params.size.y + 7) / 8,
        1);

    renderer::helper::transitMapTextureFromStoreImage(cmd_buf, images);
}

void Prt::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(prt_desc_set_layout_);
    device->destroyPipelineLayout(prt_pipeline_layout_);
    device->destroyPipeline(prt_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
