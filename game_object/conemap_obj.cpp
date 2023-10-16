#include "conemap_obj.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace er = engine::renderer;
namespace {
er::WriteDescriptorList addPrtRelatedTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::TextureInfo>& dst_texes) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEX_INDEX,
        texture_sampler,
        src_image,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_texes->view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addPrtShadowCacheUpdateTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::TextureInfo>& minmax_depth,
    const std::shared_ptr<er::TextureInfo>& dst_texes) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEX_INDEX,
        texture_sampler,
        src_image,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // minmax depth texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_INFO_TEX_INDEX,
        nullptr,
        minmax_depth->view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_texes->view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addGenPrtPackInfoTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::TextureInfo>& src_prt_texes,
    const std::shared_ptr<er::TextureInfo>& dst_prt_info_texes) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_TEX_INDEX,
        nullptr,
        src_prt_texes->view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_prt_info_texes->view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addPackedPrtTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::TextureInfo>& src_prt_texes,
    const std::shared_ptr<er::TextureInfo>& src_prt_info_texes,
    const std::shared_ptr<er::TextureInfo>& dst_prt_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_TEX_INDEX,
        nullptr,
        src_prt_texes->view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_INFO_TEX_INDEX,
        nullptr,
        src_prt_info_texes->view,
        er::ImageLayout::GENERAL);

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
    const std::shared_ptr<scene_rendering::PrtShadow>& prt_shadowgen,
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
    prt_pack_info_tex_ = std::make_shared<renderer::TextureInfo>();

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

    glm::uvec2 pack_info_tex_size =
        buffer_size /
        glm::uvec2(kConemapGenBlockCacheSizeX, kConemapGenBlockCacheSizeY) *
        glm::uvec2(4);

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32B32A32_SFLOAT,
        pack_info_tex_size,
        *prt_pack_info_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    // create prt texture descriptor sets.
    prt_shadow_gen_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_shadowgen->getPrtShadowGenDescSetLayout(), 1)[0];

    auto prt_shadow_gen_texture_descs =
        addPrtRelatedTextures(
            prt_shadow_gen_tex_desc_set_,
            texture_sampler,
            prt_bump_tex.view,
            prt_shadowgen->getPrtTextures());
    device->updateDescriptorSets(prt_shadow_gen_texture_descs);

    prt_shadow_cache_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_shadowgen->getPrtShadowCacheDescSetLayout(), 1)[0];

    auto prt_shadow_cache_texture_descs =
        addPrtRelatedTextures(
            prt_shadow_cache_tex_desc_set_,
            texture_sampler,
            prt_bump_tex.view,
            prt_shadowgen->getPrtShadowCacheTextures());
    device->updateDescriptorSets(prt_shadow_cache_texture_descs);

    prt_shadow_cache_update_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_shadowgen->getPrtShadowCacheUpdateDescSetLayout(), 1)[0];

    auto prt_shadow_cache_update_texture_descs =
        addPrtShadowCacheUpdateTextures(
            prt_shadow_cache_update_tex_desc_set_,
            texture_sampler,
            prt_bump_tex.view,
            minmax_depth_tex_,
            prt_shadowgen->getPrtShadowCacheTextures());
    device->updateDescriptorSets(prt_shadow_cache_update_texture_descs);

    gen_prt_pack_info_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_shadowgen->getGenPrtPackInfoDescSetLayout(), 1)[0];

    auto gen_prt_pack_info_texture_descs =
        addGenPrtPackInfoTextures(
            gen_prt_pack_info_tex_desc_set_,
            prt_shadowgen->getPrtDsTextures(),
            prt_pack_info_tex_);
    device->updateDescriptorSets(gen_prt_pack_info_texture_descs);

    pack_prt_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_shadowgen->getPackPrtDescSetLayout(), 1)[0];

    auto pack_prt_texture_descs =
        addPackedPrtTextures(
            pack_prt_tex_desc_set_,
            prt_shadowgen->getPrtTextures(),
            prt_pack_info_tex_,
            prt_pack_tex_);

    device->updateDescriptorSets(pack_prt_texture_descs);

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

    if (prt_pack_info_tex_) {
        prt_pack_info_tex_->destroy(device);
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
