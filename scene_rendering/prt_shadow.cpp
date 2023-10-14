#include <algorithm>
#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "prt_shadow.h"

namespace {
    namespace er = engine::renderer;

    const glm::uvec2 g_block_size =
        glm::uvec2(kConemapGenBlockCacheSizeX, kConemapGenBlockCacheSizeY);

    er::WriteDescriptorList addPrtShadowGenTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set,
        const std::shared_ptr<er::TextureInfo>& shadow_cache_tex,
        const std::shared_ptr<er::TextureInfo>& dst_texes) {
        er::WriteDescriptorList descriptor_writes;
        descriptor_writes.reserve(2);

        // minmax depth texture.
        er::Helper::addOneTexture(
            descriptor_writes,
            description_set,
            er::DescriptorType::STORAGE_IMAGE,
            SRC_INFO_TEX_INDEX,
            nullptr,
            shadow_cache_tex->view,
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

    er::WriteDescriptorList addDsPrtTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set,
        const std::shared_ptr<er::TextureInfo>& src_prt_texes,
        const std::shared_ptr<er::TextureInfo>& dst_prt_texes) {
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
            dst_prt_texes->view,
            er::ImageLayout::GENERAL);

        return descriptor_writes;
    }

    std::shared_ptr<er::PipelineLayout>
        createPrtShadowGenPipelineLayout(
            const std::shared_ptr<er::Device>& device,
            const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
        er::PushConstantRange push_const_range{};
        push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
        push_const_range.offset = 0;
        push_const_range.size = sizeof(glsl::PrtGenParams);

        return device->createPipelineLayout(
            { desc_set_layout },
            { push_const_range });
    }

    std::shared_ptr<er::PipelineLayout>
        createPrtShadowCacheUpdatePipelineLayout(
            const std::shared_ptr<er::Device>& device,
            const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
        er::PushConstantRange push_const_range{};
        push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
        push_const_range.offset = 0;
        push_const_range.size = sizeof(glsl::PrtShadowCacheGenParams);

        return device->createPipelineLayout(
            { desc_set_layout },
            { push_const_range });
    }

    std::shared_ptr<er::PipelineLayout>
        createPrtDsPipelineLayout(
            const std::shared_ptr<er::Device>& device,
            const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
        return device->createPipelineLayout(
            { desc_set_layout },
            { });
    }

    std::shared_ptr<er::PipelineLayout>
        createPrtPackPipelineLayout(
            const std::shared_ptr<er::Device>& device,
            const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
        er::PushConstantRange push_const_range{};
        push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
        push_const_range.offset = 0;
        push_const_range.size = sizeof(glsl::PrtPackParams);

        return device->createPipelineLayout(
            { desc_set_layout },
            { push_const_range });
    }
} // namespace

namespace engine {
namespace scene_rendering {

PrtShadow::PrtShadow(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    const glm::uvec2 temp_buffer_size =
        g_block_size * glm::uvec2(8, 1);

    prt_texes_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32B32A32_SFLOAT,
        temp_buffer_size,
        *prt_texes_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    // 400 sample rays per pixel saved tangent angle for shadowing.
    const glm::uvec2 prt_shadow_cache_tex_size =
        g_block_size * glm::uvec2(uint(std::sqrt(kPrtPhiSampleCount / 4)));

    prt_shadow_cache_texes_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        prt_shadow_cache_tex_size,
        *prt_shadow_cache_texes_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    const glm::uvec2 temp_ds_buffer_size =
        g_block_size / glm::uvec2(16) * glm::uvec2(4);

    prt_ds_texes_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32B32A32_SFLOAT,
        temp_ds_buffer_size,
        *prt_ds_texes_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL);

    // create a prt shadow texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> prt_shadow_gen_with_cache_bindings;
    prt_shadow_gen_with_cache_bindings.reserve(2);
    prt_shadow_gen_with_cache_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_INFO_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_shadow_gen_with_cache_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_shadow_gen_with_cache_desc_set_layout_ =
        device->createDescriptorSetLayout(prt_shadow_gen_with_cache_bindings);

    // create prt texture descriptor sets.
    prt_shadow_gen_with_cache_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_shadow_gen_with_cache_desc_set_layout_, 1)[0];

    auto prt_shadow_gen_with_cache_texture_descs =
        addPrtShadowGenTextures(
            prt_shadow_gen_with_cache_tex_desc_set_,
            prt_shadow_cache_texes_,
            prt_texes_);
    device->updateDescriptorSets(prt_shadow_gen_with_cache_texture_descs);

    prt_shadow_gen_with_cache_pipeline_layout_ =
        createPrtShadowGenPipelineLayout(
            device,
            prt_shadow_gen_with_cache_desc_set_layout_);

    prt_shadow_gen_with_cache_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_shadow_gen_with_cache_pipeline_layout_,
            "prt_shadow_gen_with_cache_comp.spv");


    // create a prt shadow generating texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(2);
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER));

    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_shadow_gen_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    prt_shadow_gen_pipeline_layout_ =
        createPrtShadowGenPipelineLayout(
            device,
            prt_shadow_gen_desc_set_layout_);

    prt_shadow_gen_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_shadow_gen_pipeline_layout_,
            "prt_shadow_gen_comp.spv");

    prt_shadow_cache_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    prt_shadow_cache_pipeline_layout_ =
        createPrtShadowGenPipelineLayout(
            device,
            prt_shadow_cache_desc_set_layout_);

    prt_shadow_cache_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_shadow_cache_pipeline_layout_,
            "prt_shadow_cache_init_comp.spv");

    std::vector<renderer::DescriptorSetLayoutBinding> prt_shadow_update_bindings;
    prt_shadow_update_bindings.reserve(3);
    prt_shadow_update_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER));

    prt_shadow_update_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_INFO_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_shadow_update_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_shadow_cache_update_desc_set_layout_ =
        device->createDescriptorSetLayout(prt_shadow_update_bindings);

    prt_shadow_cache_update_pipeline_layout_ =
        createPrtShadowCacheUpdatePipelineLayout(
            device,
            prt_shadow_cache_update_desc_set_layout_);

    prt_shadow_cache_update_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_shadow_cache_update_pipeline_layout_,
            "prt_shadow_cache_update_comp.spv");

    // create a global ibl texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> ds_bindings;
    ds_bindings.reserve(2);
    ds_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    ds_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_ds_desc_set_layout_ =
        device->createDescriptorSetLayout(ds_bindings);

    prt_ds_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_ds_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto prt_ds_texture_descs =
        addDsPrtTextures(
            prt_ds_tex_desc_set_,
            prt_texes_,
            prt_ds_texes_);
    device->updateDescriptorSets(prt_ds_texture_descs);

    prt_ds_first_pipeline_layout_ =
        createPrtDsPipelineLayout(
            device,
            prt_ds_desc_set_layout_);

    prt_ds_first_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_ds_first_pipeline_layout_,
            "prt_minmax_ds_comp.spv");

    // create a global ibl texture descriptor set layout.
    gen_prt_pack_info_desc_set_layout_ =
        device->createDescriptorSetLayout(ds_bindings);

    gen_prt_pack_info_pipeline_layout_ =
        createPrtPackPipelineLayout(
            device,
            gen_prt_pack_info_desc_set_layout_);

    gen_prt_pack_info_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            gen_prt_pack_info_pipeline_layout_,
            "gen_prt_pack_info_comp.spv");

    // create a global ibl texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> pack_bindings;
    pack_bindings.reserve(3);
    pack_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    pack_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_INFO_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    pack_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    pack_prt_desc_set_layout_ =
        device->createDescriptorSetLayout(pack_bindings);

    pack_prt_pipeline_layout_ =
        createPrtPackPipelineLayout(
            device,
            pack_prt_desc_set_layout_);

    pack_prt_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            pack_prt_pipeline_layout_,
            "pack_prt_comp.spv");
}

void PrtShadow::update(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj) {

    auto src_size =
        glm::uvec2(conemap_obj->getPackTexture()->size);

    // create coarse prt textures.
    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { prt_texes_->image });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_shadow_gen_pipeline_);
        glsl::PrtGenParams params = {};
        params.size = src_size;
        params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
        params.block_offset = glm::uvec2(0);
        params.pixel_sample_size =
            glm::vec2(params.size) / glm::vec2(g_block_size);
        params.shadow_intensity = conemap_obj->getShadowIntensity();
        params.depth_channel = conemap_obj->getDepthChannel();
        params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
        params.shadow_noise_thread = conemap_obj->getShadowNoiseThread();
        params.sample_rate = 1.0f;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            prt_shadow_gen_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            prt_shadow_gen_pipeline_layout_,
            { conemap_obj->getPrtShadowGenTexDescSet() });

        cmd_buf->dispatch(
            (g_block_size.x + 31) / 32,
            (g_block_size.y + 31) / 32,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { prt_texes_->image },
            renderer::ImageLayout::GENERAL);
    }

    {
        er::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { prt_ds_texes_->image });

        cmd_buf->bindPipeline(
            er::PipelineBindPoint::COMPUTE,
            prt_ds_first_pipeline_);

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            prt_ds_first_pipeline_layout_,
            { prt_ds_tex_desc_set_ });

        cmd_buf->dispatch(
            (g_block_size.x + 15) / 16,
            (g_block_size.y + 15) / 16,
            1);

        er::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { prt_ds_texes_->image },
            er::ImageLayout::GENERAL);
    }

    {
        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            gen_prt_pack_info_pipeline_);

        glsl::PrtPackParams params = {};
        params.size = (g_block_size + uvec2(15)) / uvec2(16);
        params.block_index = glm::uvec2(0);
        params.range_scale = 1.2f;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            gen_prt_pack_info_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            gen_prt_pack_info_pipeline_layout_,
            { conemap_obj->getGenPrtPackInfoTexDescSet() });

        cmd_buf->dispatch(1, 1, 1);

        er::BarrierList barrier_list;
        er::helper::addTexturesToBarrierList(
            barrier_list,
            { conemap_obj->getPackInfoTexture()->image },
            renderer::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT) | SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
    }

    auto block_count =
        (src_size + g_block_size - glm::uvec2(1)) / g_block_size;

    auto num_passes = block_count.x * block_count.y;

    for (uint p = 0; p < num_passes; p++) {
        uint block_x = p % block_count.x;
        uint block_y = p / block_count.x;

        // cache shadow ray's tangent value.
        {
            renderer::helper::transitMapTextureToStoreImage(
                cmd_buf,
                { prt_shadow_cache_texes_->image });

            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                prt_shadow_cache_pipeline_);
            glsl::PrtGenParams params = {};
            params.size = src_size;
            params.inv_size =
                glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
            params.block_offset =
                glm::uvec2(block_x, block_y) * g_block_size;
            params.shadow_intensity = conemap_obj->getShadowIntensity();
            params.depth_channel = conemap_obj->getDepthChannel();
            params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
            params.shadow_noise_thread = conemap_obj->getShadowNoiseThread();

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                prt_shadow_cache_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                prt_shadow_cache_pipeline_layout_,
                { conemap_obj->getPrtShadowCacheTexDescSet() });

            cmd_buf->dispatch(
                (g_block_size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
                (g_block_size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
                1);

            renderer::helper::transitMapTextureFromStoreImage(
                cmd_buf,
                { prt_shadow_cache_texes_->image },
                renderer::ImageLayout::GENERAL);
        }

        // go through all the other cache blocks, update cache shadow ray's tangent value.
        {
            auto block_cache_num_x =
                (src_size.x + kConemapGenBlockCacheSizeX - 1) / kConemapGenBlockCacheSizeX;
            auto block_cache_num_y =
                (src_size.y + kConemapGenBlockCacheSizeY - 1) / kConemapGenBlockCacheSizeY;
            auto total_block_num = block_cache_num_x * block_cache_num_y;

            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                prt_shadow_cache_update_pipeline_);

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                prt_shadow_cache_update_pipeline_layout_,
                { conemap_obj->getPrtShadowCacheUpdateTexDescSet() });

            glsl::PrtShadowCacheGenParams params = {};
            params.size = src_size;
            params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
            params.depth_channel = conemap_obj->getDepthChannel();
            params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
            params.shadow_noise_thread = conemap_obj->getShadowNoiseThread();
            params.shadow_intensity = conemap_obj->getShadowIntensity();

            std::vector<uint32_t> block_indexes;
            block_indexes.reserve(total_block_num);
            for (int i = 0; i < int(total_block_num); i++) {
                int y = i / block_cache_num_x;
                int x = i % block_cache_num_x;

                if (x >= int(block_x - kPrtShadowInitBlockRadius) &&
                    x <= int(block_x + kPrtShadowInitBlockRadius) &&
                    y >= int(block_y - kPrtShadowInitBlockRadius) &&
                    y <= int(block_y + kPrtShadowInitBlockRadius)) {
                    continue;
                }

                glm::vec2 block_diff =
                    glm::vec2(x, y) - glm::vec2(block_x, block_y);
                float dist = glm::length(block_diff);
                uint32_t pack_value =
                    (uint32_t(dist * 256.0f) << 16) | (y << 8) | x;

                block_indexes.push_back(pack_value);
            }

            std::sort(block_indexes.begin(), block_indexes.end());

            for (auto& block_index : block_indexes) {
                int y = (block_index >> 8) & 0xff;
                int x = block_index & 0xff;

                params.block_index =
                    glm::uvec2(x, y);
                params.block_offset =
                    glm::uvec2(
                        x * kConemapGenBlockCacheSizeX,
                        y * kConemapGenBlockCacheSizeY);
                params.dst_block_offset =
                    glm::uvec2(
                        block_x * kConemapGenBlockCacheSizeX,
                        block_y * kConemapGenBlockCacheSizeY);

                cmd_buf->pushConstants(
                    SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                    prt_shadow_cache_update_pipeline_layout_,
                    &params,
                    sizeof(params));

                cmd_buf->dispatch(
                    (g_block_size.x + kConemapGenDispatchX - 1) / kConemapGenDispatchX,
                    (g_block_size.y + kConemapGenDispatchY - 1) / kConemapGenDispatchY,
                    1);

                renderer::helper::transitMapTextureToStoreImage(
                    cmd_buf,
                    { prt_shadow_cache_texes_->image });
            }
        }

        // create prt textures.
        {
            renderer::helper::transitMapTextureToStoreImage(
                cmd_buf,
                { prt_texes_->image });

            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                prt_shadow_gen_with_cache_pipeline_);
            glsl::PrtGenParams params = {};
            params.size = src_size;
            params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
            params.block_offset =
                glm::uvec2(block_x, block_y) * g_block_size;
            params.shadow_intensity = conemap_obj->getShadowIntensity();
            params.depth_channel = conemap_obj->getDepthChannel();
            params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
            params.shadow_noise_thread = conemap_obj->getShadowNoiseThread();

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                prt_shadow_gen_with_cache_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                prt_shadow_gen_with_cache_pipeline_layout_,
                { prt_shadow_gen_with_cache_tex_desc_set_ });

            cmd_buf->dispatch(
                (g_block_size.x + kPrtShadowGenDispatchX - 1) / kPrtShadowGenDispatchX,
                (g_block_size.y + kPrtShadowGenDispatchY - 1) / kPrtShadowGenDispatchY,
                1);

            renderer::helper::transitMapTextureFromStoreImage(
                cmd_buf,
                { prt_texes_->image },
                renderer::ImageLayout::GENERAL);
        }

        {
            renderer::helper::transitMapTextureToStoreImage(cmd_buf, { conemap_obj->getPackTexture()->image });

            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                pack_prt_pipeline_);

            glsl::PrtPackParams params = {};
            params.size = g_block_size;
            params.block_index =
                glm::uvec2(block_x, block_y);
            params.block_offset =
                glm::uvec2(block_x, block_y) * g_block_size;
            params.range_scale = 1.0f;

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                pack_prt_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                pack_prt_pipeline_layout_,
                { conemap_obj->getPackPrtTexDescSet() });

            cmd_buf->dispatch(
                (g_block_size.x + 7) / 8,
                (g_block_size.y + 7) / 8,
                1);

            cmd_buf->addImageBarrier(
                conemap_obj->getPackTexture()->image,
                er::Helper::getImageAsStore(),
                er::Helper::getImageAsStore());
        }
    }
}

void PrtShadow::destroy(
    const std::shared_ptr<renderer::Device>& device) {

    if (prt_texes_) {
        prt_texes_->destroy(device);
    }

    if (prt_ds_texes_) {
        prt_ds_texes_->destroy(device);
    }

    if (prt_shadow_cache_texes_) {
        prt_shadow_cache_texes_->destroy(device);
    }

    device->destroyDescriptorSetLayout(prt_shadow_gen_with_cache_desc_set_layout_);
    device->destroyPipelineLayout(prt_shadow_gen_with_cache_pipeline_layout_);
    device->destroyPipeline(prt_shadow_gen_with_cache_pipeline_);
    device->destroyDescriptorSetLayout(prt_shadow_cache_desc_set_layout_);
    device->destroyPipelineLayout(prt_shadow_cache_pipeline_layout_);
    device->destroyPipeline(prt_shadow_cache_pipeline_);
    device->destroyDescriptorSetLayout(prt_shadow_cache_update_desc_set_layout_);
    device->destroyPipelineLayout(prt_shadow_cache_update_pipeline_layout_);
    device->destroyPipeline(prt_shadow_cache_update_pipeline_);
    device->destroyDescriptorSetLayout(prt_shadow_gen_desc_set_layout_);
    device->destroyPipelineLayout(prt_shadow_gen_pipeline_layout_);
    device->destroyPipeline(prt_shadow_gen_pipeline_);

    device->destroyDescriptorSetLayout(prt_ds_desc_set_layout_);
    device->destroyDescriptorSetLayout(gen_prt_pack_info_desc_set_layout_);
    device->destroyDescriptorSetLayout(pack_prt_desc_set_layout_);
    device->destroyPipelineLayout(prt_ds_first_pipeline_layout_);
    device->destroyPipeline(prt_ds_first_pipeline_);
    device->destroyPipelineLayout(gen_prt_pack_info_pipeline_layout_);
    device->destroyPipeline(gen_prt_pack_info_pipeline_);
    device->destroyPipelineLayout(pack_prt_pipeline_layout_);
    device->destroyPipeline(pack_prt_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
