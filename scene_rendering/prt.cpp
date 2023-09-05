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
        const std::shared_ptr<er::ImageView>& dst_image,
        const std::shared_ptr<er::ImageView>& dst_image_1,
        const std::shared_ptr<er::ImageView>& dst_image_2,
        const std::shared_ptr<er::ImageView>& dst_image_3,
        const std::shared_ptr<er::ImageView>& dst_image_4,
        const std::shared_ptr<er::ImageView>& dst_image_5,
        const std::shared_ptr<er::ImageView>& dst_image_6) {
        er::WriteDescriptorList descriptor_writes;
//        descriptor_writes.reserve(2);
        descriptor_writes.reserve(8);

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
            DST_TEX_INDEX_0,
            nullptr,
            dst_image,
            er::ImageLayout::GENERAL);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX_1,
            nullptr,
            dst_image_1,
            er::ImageLayout::GENERAL);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX_2,
            nullptr,
            dst_image_2,
            er::ImageLayout::GENERAL);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX_3,
            nullptr,
            dst_image_3,
            er::ImageLayout::GENERAL);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX_4,
            nullptr,
            dst_image_4,
            er::ImageLayout::GENERAL);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX_5,
            nullptr,
            dst_image_5,
            er::ImageLayout::GENERAL);

        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            DST_TEX_INDEX_6,
            nullptr,
            dst_image_6,
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
    const renderer::TextureInfo& prt_tex,
    const renderer::TextureInfo& prt_tex_1,
    const renderer::TextureInfo& prt_tex_2,
    const renderer::TextureInfo& prt_tex_3,
    const renderer::TextureInfo& prt_tex_4,
    const renderer::TextureInfo& prt_tex_5,
    const renderer::TextureInfo& prt_tex_6) {

    const auto& device = device_info.device;

    prt_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_0,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_1,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_2,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_3,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_4,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_5,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_6,
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
        /*prt_tex.view*/
        prt_tex.view,
        prt_tex_1.view,
        prt_tex_2.view,
        prt_tex_3.view,
        prt_tex_4.view,
        prt_tex_5.view,
        prt_tex_6.view);
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
    const renderer::TextureInfo& prt_tex,
    const renderer::TextureInfo& prt_tex_1, 
    const renderer::TextureInfo& prt_tex_2, 
    const renderer::TextureInfo& prt_tex_3, 
    const renderer::TextureInfo& prt_tex_4, 
    const renderer::TextureInfo& prt_tex_5,
    const renderer::TextureInfo& prt_tex_6) {
    renderer::helper::transitMapTextureToStoreImage(
        cmd_buf,
        { prt_tex.image,
          prt_tex_1.image,
          prt_tex_2.image,
          prt_tex_3.image,
          prt_tex_4.image,
          prt_tex_5.image,
          prt_tex_6.image });

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        prt_pipeline_);
    glsl::PrtParams params = {};
    params.size = glm::uvec2(prt_tex.size);
    params.inv_size = glm::vec2(1.0f / prt_tex.size.x, 1.0f / prt_tex.size.y);

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
        (prt_tex.size.x + 7) / 8,
        (prt_tex.size.y + 7) / 8,
        1);

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { prt_tex.image,
          prt_tex_1.image,
          prt_tex_2.image,
          prt_tex_3.image,
          prt_tex_4.image,
          prt_tex_5.image,
          prt_tex_6.image });
}

void Prt::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(prt_desc_set_layout_);
    device->destroyPipelineLayout(prt_pipeline_layout_);
    device->destroyPipeline(prt_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
