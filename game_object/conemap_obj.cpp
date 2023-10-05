#include "conemap_obj.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace er = engine::renderer;
namespace {
er::WriteDescriptorList addPrtTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::array<std::shared_ptr<er::TextureInfo>, 7>& prt_texes) {
    er::WriteDescriptorList descriptor_writes;
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

er::WriteDescriptorList addDsFinalPrtTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::array<std::shared_ptr<er::TextureInfo>, 7>& src_prt_texes,
    const std::shared_ptr<er::BufferInfo>& dst_prt_minmax_buffer) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(8);

    for (int i = 0; i < src_prt_texes.size(); ++i) {
        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            SRC_TEX_INDEX_0 + i,
            texture_sampler,
            src_prt_texes[i]->view,
            er::ImageLayout::GENERAL);
    }

    er::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_BUFFER,
        DST_BUFFER_INDEX,
        dst_prt_minmax_buffer->buffer,
        dst_prt_minmax_buffer->buffer->getSize());

    return descriptor_writes;
}

er::WriteDescriptorList addPackedPrtTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::array<std::shared_ptr<er::TextureInfo>, 7>& src_prt_texes,
    const std::shared_ptr<er::BufferInfo>& src_prt_minmax_buffer,
    const std::shared_ptr<er::TextureInfo>& dst_prt_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(9);

    for (int i = 0; i < src_prt_texes.size(); ++i) {
        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            SRC_TEX_INDEX_0 + i,
            texture_sampler,
            src_prt_texes[i]->view,
            er::ImageLayout::GENERAL);
    }

    er::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_BUFFER,
        SRC_BUFFER_INDEX,
        src_prt_minmax_buffer->buffer,
        src_prt_minmax_buffer->buffer->getSize());

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_prt_tex->view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addGenMinmaxDepthTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& dst_image) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    // height/depth map texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEX_INDEX,
        texture_sampler,
        src_image,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // conemap/height map texture.
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
    generateMinmaxDepthPipelineLayout(
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::ConemapGenParams);

    return device->createPipelineLayout(
        { desc_set_layout },
        { push_const_range });
}
} // namespace

namespace game_object {

ConemapObj::ConemapObj(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& prt_bump_tex,
    const std::shared_ptr<scene_rendering::Prt>& prt_gen,
    uint32_t depth_channel,
    bool is_height_map,
    float depth_scale,
    float shadow_intensity,
    float shadow_noise_thread) {

    depth_channel_ = depth_channel;
    is_height_map_ = is_height_map;
    depth_scale_ = depth_scale;
    shadow_intensity_ = shadow_intensity;
    shadow_noise_thread_ = shadow_noise_thread;

    const glm::uvec2& buffer_size =
        glm::uvec2(prt_bump_tex.size);

    conemap_tex_ = std::make_shared<renderer::TextureInfo>();
    prt_pack_tex_ = std::make_shared<renderer::TextureInfo>();
    minmax_depth_tex_ = std::make_shared<renderer::TextureInfo>();

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16_SFLOAT,
        buffer_size / glm::uvec2(kConemapGenBlockCacheSizeX, kConemapGenBlockCacheSizeY),
        *minmax_depth_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R8G8B8A8_UNORM,
        buffer_size,
        *conemap_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32B32A32_UINT,
        buffer_size,
        *prt_pack_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    prt_minmax_buffer_ = std::make_shared<renderer::BufferInfo>();
    device->createBuffer(
        sizeof(glsl::PrtMinmaxInfo),
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_CACHED_BIT),
        0,
        prt_minmax_buffer_->buffer,
        prt_minmax_buffer_->memory);

    // create prt texture descriptor sets.
    prt_gen_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_gen->getPrtGenDescSetLayout(), 1)[0];

    auto prt_texture_descs = addPrtTextures(
        prt_gen_tex_desc_set_,
        texture_sampler,
        prt_bump_tex.view,
        prt_gen->getPrtTextures());
    device->updateDescriptorSets(prt_texture_descs);

    prt_ds_final_tex_desc_sets_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_gen->getPrtDsFinalDescSetLayout(), 2);

    for (uint32_t i = 0; i < 2; i++) {
        auto prt_ds_final_texture_descs = addDsFinalPrtTextures(
            prt_ds_final_tex_desc_sets_[i],
            texture_sampler,
            prt_gen->getPrtDsTextures(i),
            prt_minmax_buffer_);
        device->updateDescriptorSets(prt_ds_final_texture_descs);
    }

    prt_pack_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_gen->getPrtPackDescSetLayout(), 1)[0];

    auto prt_pack_texture_descs = addPackedPrtTextures(
        prt_pack_tex_desc_set_,
        texture_sampler,
        prt_gen->getPrtTextures(),
        prt_minmax_buffer_,
        prt_pack_tex_);

    device->updateDescriptorSets(prt_pack_texture_descs);

    gen_minmax_depth_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    gen_minmax_depth_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            gen_minmax_depth_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto gen_minmax_depth_texture_descs =
        addGenMinmaxDepthTextures(
            gen_minmax_depth_tex_desc_set_,
            texture_sampler,
            prt_bump_tex.view,
            minmax_depth_tex_->view);
    device->updateDescriptorSets(gen_minmax_depth_texture_descs);

    gen_minmax_depth_pipeline_layout_ =
        generateMinmaxDepthPipelineLayout(
            device,
            gen_minmax_depth_desc_set_layout_);

    gen_minmax_depth_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            gen_minmax_depth_pipeline_layout_,
            "gen_minmax_depth_comp.spv");
}

void ConemapObj::update(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::uvec2& src_buffer_size) {
    // generate minmax depth texture.
    {
        cmd_buf->bindPipeline(
            er::PipelineBindPoint::COMPUTE,
            gen_minmax_depth_pipeline_);

        glsl::ConemapGenParams params = {};
        params.size = src_buffer_size;
        params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
        params.depth_channel = getDepthChannel();
        params.is_height_map = isHeightMap() ? 1 : 0;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            gen_minmax_depth_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            gen_minmax_depth_pipeline_layout_,
            { gen_minmax_depth_tex_desc_set_ });

        cmd_buf->dispatch(
            (params.size.x + kConemapGenBlockCacheSizeX - 1) / kConemapGenBlockCacheSizeX,
            (params.size.y + kConemapGenBlockCacheSizeY - 1) / kConemapGenBlockCacheSizeY,
            1);
    }
}

void ConemapObj::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    if (conemap_tex_) {
        conemap_tex_->destroy(device);
    }

    if (prt_pack_tex_) {
        prt_pack_tex_->destroy(device);
    }

    if (prt_minmax_buffer_) {
        prt_minmax_buffer_->destroy(device);
    }

    if (minmax_depth_tex_) {
        minmax_depth_tex_->destroy(device);
    }

    device->destroyDescriptorSetLayout(gen_minmax_depth_desc_set_layout_);
    device->destroyPipelineLayout(gen_minmax_depth_pipeline_layout_);
    device->destroyPipeline(gen_minmax_depth_pipeline_);
}

} // game_object
} // engine
