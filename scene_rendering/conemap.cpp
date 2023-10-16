#include <algorithm>
#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "conemap.h"

namespace {
namespace er = engine::renderer;

const glm::uvec2 g_cache_block_size =
    glm::uvec2(
        kPrtShadowGenBlockCacheSizeX,
        kPrtShadowGenBlockCacheSizeY);

er::WriteDescriptorList addConemapGenInitTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& dst_image_0,
    const std::shared_ptr<er::ImageView>& dst_image_1) {
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

    // conemap/height map texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_image_0,
        er::ImageLayout::GENERAL);

    // conemap/height map texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX_1,
        nullptr,
        dst_image_1,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addConemapGenTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& minmax_depth_image,
    const std::shared_ptr<er::ImageView>& dst_image_0,
    const std::shared_ptr<er::ImageView>& dst_image_1) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

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
        dst_image_0,
        er::ImageLayout::GENERAL);

    // conemap/height map texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX_1,
        nullptr,
        dst_image_1,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addConemapPackTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& src_image_0,
    const std::shared_ptr<er::ImageView>& src_image_1,
    const std::shared_ptr<er::ImageView>& dst_image) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

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
        SRC_TEX_INDEX_1,
        nullptr,
        src_image_0,
        er::ImageLayout::GENERAL);

    // conemap/height map texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_TEX_INDEX_2,
        nullptr,
        src_image_1,
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

    conemap_temp_tex_[0] = std::make_shared<renderer::TextureInfo>();
    conemap_temp_tex_[1] = std::make_shared<renderer::TextureInfo>();

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32_SINT,
        glm::uvec2(kConemapGenBlockSizeX, kConemapGenBlockSizeY),
        *conemap_temp_tex_[0],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32_SINT,
        glm::uvec2(kConemapGenBlockSizeX, kConemapGenBlockSizeY),
        *conemap_temp_tex_[1],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    conemap_gen_init_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_1,
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
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_1,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    conemap_pack_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX_1,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX_2,
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
            conemap_temp_tex_[0]->view,
            conemap_temp_tex_[1]->view);
    device->updateDescriptorSets(conemap_gen_init_texture_descs);

    conemap_gen_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            conemap_gen_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto conemap_gen_texture_descs =
        addConemapGenTextures(
            conemap_gen_tex_desc_set_,
            texture_sampler,
            bump_tex.view,
            conemap_obj->getMinmaxDepthTexture()->view,
            conemap_temp_tex_[0]->view,
            conemap_temp_tex_[1]->view);
    device->updateDescriptorSets(conemap_gen_texture_descs);

    conemap_pack_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            conemap_pack_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto conemap_pack_texture_descs =
        addConemapPackTextures(
            conemap_pack_tex_desc_set_,
            texture_sampler,
            bump_tex.view,
            conemap_temp_tex_[0]->view,
            conemap_temp_tex_[1]->view,
            conemap_obj->getConemapTexture()->view);
    device->updateDescriptorSets(conemap_pack_texture_descs);

    conemap_gen_init_pipeline_layout_ =
        createConemapPipelineLayout(
            device,
            conemap_gen_init_desc_set_layout_);

    conemap_gen_pipeline_layout_ =
        createConemapPipelineLayout(
            device,
            conemap_gen_desc_set_layout_);

    conemap_pack_pipeline_layout_ =
        createConemapPipelineLayout(
            device,
            conemap_pack_desc_set_layout_);

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

    conemap_pack_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            conemap_pack_pipeline_layout_,
            "conemap_pack_comp.spv");
}

void Conemap::update(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj,
    uint32_t pass_start,
    uint32_t pass_end) {

    const auto& conemap_tex =
        conemap_obj->getConemapTexture();

    const auto full_buffer_size =
        glm::uvec2(conemap_tex->size);

    auto dispatch_block_size =
        glm::uvec2(kConemapGenBlockSizeX, kConemapGenBlockSizeY);

    auto dispatch_block_count =
        (full_buffer_size + dispatch_block_size - glm::uvec2(1)) / dispatch_block_size;

    auto num_passes =
        dispatch_block_count.x * dispatch_block_count.y;

    auto block_cache_num_x =
        (full_buffer_size.x + kConemapGenBlockCacheSizeX - 1) / kConemapGenBlockCacheSizeX;
    auto block_cache_num_y =
        (full_buffer_size.y + kConemapGenBlockCacheSizeY - 1) / kConemapGenBlockCacheSizeY;
    auto total_block_cache_count = block_cache_num_x * block_cache_num_y;

    renderer::BarrierList barrier_list;
    barrier_list.image_barriers.reserve(1);

    renderer::helper::addTexturesToBarrierList(
        barrier_list,
        { conemap_temp_tex_[0]->image,
          conemap_temp_tex_[1]->image },
        renderer::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT) |
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(Access, SHADER_READ_BIT) |
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT));

    // generate first pass of conemap with closer blocks.
    for (uint p = pass_start; p < pass_end; p++) {
        glm::uvec2 cur_block_index =
            glm::uvec2(p % dispatch_block_count.x, p / dispatch_block_count.x);
        glm::uvec2 cur_block_size =
            glm::min(full_buffer_size - cur_block_index * dispatch_block_size, dispatch_block_size);

        {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                conemap_gen_init_pipeline_);

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                conemap_gen_init_pipeline_layout_,
                { conemap_gen_init_tex_desc_set_ });

            glsl::ConemapGenParams params = {};
            params.size = full_buffer_size;
            params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
            params.dst_block_offset = cur_block_index * dispatch_block_size;
            params.depth_channel = conemap_obj->getDepthChannel();
            params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                conemap_gen_init_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->dispatch(
                (cur_block_size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
                (cur_block_size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
                1);

            // wait for writing to finish for first pass, after first pass, conemap buffer should be initialized.
            cmd_buf->addBarriers(
                barrier_list,
                SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
                SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
        }

        {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                conemap_gen_pipeline_);

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                conemap_gen_pipeline_layout_,
                { conemap_gen_tex_desc_set_ });

            glsl::ConemapGenParams params = {};
            params.size = full_buffer_size;
            params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
            params.depth_channel = conemap_obj->getDepthChannel();
            params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
            params.dst_block_offset = cur_block_index * dispatch_block_size;

            std::vector<uint64_t> block_indexes;
            block_indexes.reserve(total_block_cache_count);
            for (int i = 0; i < int(total_block_cache_count); i++) {
                int y = i / block_cache_num_x;
                int x = i % block_cache_num_x;

                glm::vec2 block_diff =
                    (glm::vec2(x, y) + 0.5f) * glm::vec2(g_cache_block_size) -
                    (glm::vec2(cur_block_index) + 0.5f) * glm::vec2(dispatch_block_size);
                float dist = glm::length(block_diff);
                uint64_t pack_value =
                    (uint64_t(*((uint32_t*)&dist)) << 32) | (uint64_t(y) << 16) | uint64_t(x);

                block_indexes.push_back(pack_value);
            }

            std::sort(block_indexes.begin(), block_indexes.end());

            for (auto& index : block_indexes) {
                int y = int((index & 0xffffffff) >> 16);
                int x = int(index & 0xffff);

                params.block_index =
                    glm::uvec2(x, y);
                params.block_offset =
                    params.block_index * glm::ivec2(g_cache_block_size);

                cmd_buf->pushConstants(
                    SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                    conemap_gen_pipeline_layout_,
                    &params,
                    sizeof(params));

                cmd_buf->dispatch(
                    (cur_block_size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
                    (cur_block_size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
                    1);

                cmd_buf->addBarriers(
                    barrier_list,
                    SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
                    SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
            }
        }

        {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                conemap_pack_pipeline_);

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                conemap_pack_pipeline_layout_,
                { conemap_pack_tex_desc_set_ });

            glsl::ConemapGenParams params = {};
            params.size = full_buffer_size;
            params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
            params.depth_channel = conemap_obj->getDepthChannel();
            params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
            params.dst_block_offset = cur_block_index * dispatch_block_size;

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                conemap_pack_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->dispatch(
                (cur_block_size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
                (cur_block_size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
                1);
        }
    }

    // only do it after last pass.
    if (pass_end == num_passes) {
        renderer::BarrierList barrier_list;
        barrier_list.image_barriers.clear();
        barrier_list.image_barriers.reserve(1);

        renderer::helper::addTexturesToBarrierList(
            barrier_list,
            { conemap_tex->image },
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT) |
            SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT));
    }
}

void Conemap::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    for (auto& tex : conemap_temp_tex_) {
        if (tex) {
            tex->destroy(device);
        }
    }

    device->destroyDescriptorSetLayout(conemap_gen_init_desc_set_layout_);
    device->destroyDescriptorSetLayout(conemap_gen_desc_set_layout_);
    device->destroyDescriptorSetLayout(conemap_pack_desc_set_layout_);
    device->destroyPipelineLayout(conemap_gen_init_pipeline_layout_);
    device->destroyPipelineLayout(conemap_gen_pipeline_layout_);
    device->destroyPipelineLayout(conemap_pack_pipeline_layout_);
    device->destroyPipeline(conemap_gen_init_pipeline_);
    device->destroyPipeline(conemap_gen_pipeline_);
    device->destroyPipeline(conemap_pack_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
