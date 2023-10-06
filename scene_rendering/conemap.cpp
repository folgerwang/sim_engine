#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "conemap.h"

namespace {
namespace er = engine::renderer;

er::WriteDescriptorList addConemapGenInitTextures(
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

er::WriteDescriptorList addConemapGenTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& minmax_depth_image,
    const std::shared_ptr<er::ImageView>& dst_image) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    // height/depth map texture.
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
        minmax_depth_image,
        er::ImageLayout::GENERAL);

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
createConemapPipelineLayout(
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

namespace engine {
namespace scene_rendering {

Conemap::Conemap(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& bump_tex,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj) {

    auto buffer_size =
        glm::ivec2(conemap_obj->getConemapTexture()->size);

    conemap_gen_init_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    conemap_gen_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_INFO_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    conemap_gen_init_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            conemap_gen_init_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto conemap_gen_init_texture_descs =
        addConemapGenInitTextures(
            conemap_gen_init_tex_desc_set_,
            texture_sampler,
            bump_tex.view,
            conemap_obj->getConemapTexture()->view);
    device->updateDescriptorSets(conemap_gen_init_texture_descs);

    conemap_gen_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            conemap_gen_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto conemap_gen_texture_descs = addConemapGenTextures(
        conemap_gen_tex_desc_set_,
        texture_sampler,
        bump_tex.view,
        conemap_obj->getMinmaxDepthTexture()->view,
        conemap_obj->getConemapTexture()->view);
    device->updateDescriptorSets(conemap_gen_texture_descs);

    conemap_gen_init_pipeline_layout_ =
        createConemapPipelineLayout(
            device,
            conemap_gen_init_desc_set_layout_);

    conemap_gen_pipeline_layout_ =
        createConemapPipelineLayout(
            device,
            conemap_gen_desc_set_layout_);

    conemap_gen_init_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            conemap_gen_init_pipeline_layout_,
            "conemap_gen_init_comp.spv");

    conemap_gen_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            conemap_gen_pipeline_layout_,
            "conemap_gen_comp.spv");
}

void Conemap::update(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj) {

    const auto& conemap_tex =
        conemap_obj->getConemapTexture();

    // generate first pass of conemap with closer blocks.
    {
        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            conemap_gen_init_pipeline_);

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            conemap_gen_init_pipeline_layout_,
            { conemap_gen_init_tex_desc_set_ });

        glsl::ConemapGenParams params = {};
        params.size = glm::uvec2(conemap_tex->size);
        params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
        params.depth_channel = conemap_obj->getDepthChannel();
        params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            conemap_gen_init_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->dispatch(
            (params.size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
            (params.size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
            1);

        // wait for writing to finish for first pass, after first pass, conemap buffer should be initialized.
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { conemap_tex->image });
    }

    {
        auto block_cache_num_x =
            (conemap_tex->size.x + kConemapGenBlockCacheSizeX - 1) / kConemapGenBlockCacheSizeX;
        auto block_cache_num_y =
            (conemap_tex->size.y + kConemapGenBlockCacheSizeY - 1) / kConemapGenBlockCacheSizeY;
        auto total_block_num = block_cache_num_x * block_cache_num_y;

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            conemap_gen_pipeline_);

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            conemap_gen_pipeline_layout_,
            { conemap_gen_tex_desc_set_ });

        glsl::ConemapGenParams params = {};
        params.size = glm::uvec2(conemap_tex->size);
        params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
        params.depth_channel = conemap_obj->getDepthChannel();
        params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;

        for (int i = 0; i < int(total_block_num); i++) {
            int y = i / block_cache_num_x;
            int x = i % block_cache_num_x;

            params.block_index =
                glm::uvec2(x, y);
            params.block_offset =
                glm::uvec2(x * kConemapGenBlockCacheSizeX, y * kConemapGenBlockCacheSizeY);

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                conemap_gen_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->dispatch(
                (params.size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
                (params.size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
                1);

            renderer::helper::transitMapTextureToStoreImage(
                cmd_buf,
                { conemap_tex->image });
        }
    }

    renderer::helper::transitMapTextureFromStoreImage(
        cmd_buf,
        { conemap_tex->image });
}

void Conemap::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(conemap_gen_init_desc_set_layout_);
    device->destroyDescriptorSetLayout(conemap_gen_desc_set_layout_);
    device->destroyPipelineLayout(conemap_gen_init_pipeline_layout_);
    device->destroyPipelineLayout(conemap_gen_pipeline_layout_);
    device->destroyPipeline(conemap_gen_init_pipeline_);
    device->destroyPipeline(conemap_gen_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
