#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "prt.h"

namespace {
    namespace er = engine::renderer;

    er::WriteDescriptorList addDsPrtTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set,
        const std::shared_ptr<er::Sampler>& texture_sampler,
        const std::array<std::shared_ptr<er::TextureInfo>, 7>& src_prt_texes,
        const std::array<std::shared_ptr<er::TextureInfo>, 7>& dst_prt_texes) {
        er::WriteDescriptorList descriptor_writes;
        descriptor_writes.reserve(14);

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

        for (int i = 0; i < dst_prt_texes.size(); ++i) {
            er::Helper::addOneTexture(
                descriptor_writes,
                description_set,
                er::DescriptorType::STORAGE_IMAGE,
                DST_TEX_INDEX_0 + i,
                nullptr,
                dst_prt_texes[i]->view,
                er::ImageLayout::GENERAL);
        }

        return descriptor_writes;
    }

    std::shared_ptr<er::PipelineLayout>
        createPrtGenPipelineLayout(
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

    // create prt textures.
    void MinmaxDownsample(
        const std::shared_ptr<er::CommandBuffer>& cmd_buf,
        const std::shared_ptr<er::Pipeline>& pipeline,
        const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
        const std::shared_ptr<er::DescriptorSet>& desc_set,
        const std::array<std::shared_ptr<er::TextureInfo>, 7>& dst_prt_texes,
        const glm::uvec2& src_buffer_size)
    {
        std::vector<std::shared_ptr<er::Image>> images;
        images.reserve(dst_prt_texes.size());
        for (const auto& tex : dst_prt_texes) {
            images.push_back(tex->image);
        }

        er::helper::transitMapTextureToStoreImage(cmd_buf, images);

        cmd_buf->bindPipeline(
            er::PipelineBindPoint::COMPUTE,
            pipeline);
        glsl::PrtParams params = {};
        params.size = src_buffer_size;
        params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            pipeline_layout,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            { desc_set });

        cmd_buf->dispatch(
            (params.size.x + 7) / 8,
            (params.size.y + 7) / 8,
            1);

        er::helper::transitMapTextureFromStoreImage(cmd_buf, images/*, er::ImageLayout::GENERAL*/);
    }

} // namespace

namespace engine {
namespace scene_rendering {

Prt::Prt(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    const auto& device = device_info.device;

    const glm::uvec2 buffer_size =
        glm::uvec2(s_max_prt_buffer_size, s_max_prt_buffer_size);

    // create 7 temporary prt textures and downsamples too.
    for (int i = 0; i < 6; i++) {
        prt_texes_[i] = std::make_shared<renderer::TextureInfo>();
        renderer::Helper::create2DTextureImage(
            device_info,
            renderer::Format::R32G32B32A32_SFLOAT,
            buffer_size,
            *prt_texes_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        prt_ds_texes_[0][i] = std::make_shared<renderer::TextureInfo>();
        renderer::Helper::create2DTextureImage(
            device_info,
            renderer::Format::R32G32B32A32_SFLOAT,
            glm::uvec2((buffer_size.x + 7) / 8 * 2, (buffer_size.y + 7) / 8),
            *prt_ds_texes_[0][i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        prt_ds_texes_[1][i] = std::make_shared<renderer::TextureInfo>();
        renderer::Helper::create2DTextureImage(
            device_info,
            renderer::Format::R32G32B32A32_SFLOAT,
            glm::uvec2((buffer_size.x + 63) / 64 * 2, (buffer_size.y + 63) / 64),
            *prt_ds_texes_[1][i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    prt_texes_[6] = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32_SFLOAT,
        buffer_size,
        *prt_texes_[6],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    prt_ds_texes_[0][6] = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32_SFLOAT,
        glm::uvec2((buffer_size.x + 7) / 8 * 2, (buffer_size.y + 7) / 8),
        *prt_ds_texes_[0][6],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    prt_ds_texes_[1][6] = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32_SFLOAT,
        glm::uvec2((buffer_size.x + 63) / 64 * 2, (buffer_size.y + 63) / 64),
        *prt_ds_texes_[1][6],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // create a global ibl texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(8);
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER));

    for (int i = 0; i < prt_texes_.size(); ++i) {
        bindings.push_back(
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_0 + i,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE));
    }

    prt_gen_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    prt_gen_pipeline_layout_ =
        createPrtGenPipelineLayout(
            device,
            prt_gen_desc_set_layout_);

    prt_gen_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_gen_pipeline_layout_,
            "prt_gen_comp.spv");

    // create a global ibl texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> ds_bindings;
    ds_bindings.reserve(14);
    for (int i = 0; i < prt_texes_.size(); ++i) {
        ds_bindings.push_back(
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX_0 + i,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE));
    }

    for (int i = 0; i < prt_texes_.size(); ++i) {
        ds_bindings.push_back(
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX_0 + i,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE));
    }

    prt_ds_desc_set_layout_ =
        device->createDescriptorSetLayout(ds_bindings);

    prt_ds_tex_desc_sets_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_ds_desc_set_layout_, 3);

    // create a global ibl texture descriptor set.
    auto prt_ds_texture_descs = addDsPrtTextures(
        prt_ds_tex_desc_sets_[0],
        texture_sampler,
        prt_texes_,
        prt_ds_texes_[0]);
    device->updateDescriptorSets(prt_ds_texture_descs);

    prt_ds_texture_descs = addDsPrtTextures(
        prt_ds_tex_desc_sets_[1],
        texture_sampler,
        prt_ds_texes_[0],
        prt_ds_texes_[1]);
    device->updateDescriptorSets(prt_ds_texture_descs);

    prt_ds_texture_descs = addDsPrtTextures(
        prt_ds_tex_desc_sets_[2],
        texture_sampler,
        prt_ds_texes_[1],
        prt_ds_texes_[0]);
    device->updateDescriptorSets(prt_ds_texture_descs);

    prt_ds_first_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_ds_desc_set_layout_);

    prt_ds_first_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_ds_first_pipeline_layout_,
            "prt_minmax_ds_s_comp.spv");

    prt_ds_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_ds_desc_set_layout_);

    prt_ds_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_ds_pipeline_layout_,
            "prt_minmax_ds_comp.spv");

    // create a global ibl texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> ds_f_bindings;
    ds_f_bindings.reserve(8);
    for (int i = 0; i < prt_texes_.size(); ++i) {
        ds_f_bindings.push_back(
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX_0 + i,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE));
    }

    ds_f_bindings.push_back(
        renderer::helper::getBufferDescriptionSetLayoutBinding(
            DST_BUFFER_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_BUFFER));

    prt_ds_final_desc_set_layout_ =
        device->createDescriptorSetLayout(ds_f_bindings);

    prt_ds_final_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_ds_final_desc_set_layout_);

    prt_ds_final_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_ds_final_pipeline_layout_,
            "prt_minmax_ds_f_comp.spv");

    // create a global ibl texture descriptor set layout.
    std::vector<renderer::DescriptorSetLayoutBinding> pack_bindings;
    pack_bindings.reserve(9);
    for (int i = 0; i < prt_texes_.size(); ++i) {
        pack_bindings.push_back(
            renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX_0 + i,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE));
    }

    pack_bindings.push_back(
        renderer::helper::getBufferDescriptionSetLayoutBinding(
            SRC_BUFFER_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_BUFFER));

    pack_bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE));

    prt_pack_desc_set_layout_ =
        device->createDescriptorSetLayout(pack_bindings);

    prt_pack_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_pack_desc_set_layout_);

    prt_pack_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_pack_pipeline_layout_,
            "prt_pack_comp.spv");
}

void Prt::update(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj) {

    if (prt_texes_.empty()) {
        return;
    }

    auto src_size = glm::uvec2(conemap_obj->getPackTexture()->size);

    // create prt textures.
    {
        std::vector<std::shared_ptr<renderer::Image>> images;
        images.reserve(prt_texes_.size());
        for (const auto& tex : prt_texes_) {
            images.push_back(tex->image);
        }

        renderer::helper::transitMapTextureToStoreImage(cmd_buf, images);

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_gen_pipeline_);
        glsl::PrtGenParams params = {};
        params.size = src_size;
        params.inv_size = glm::vec2(1.0f / params.size.x, 1.0f / params.size.y);
        params.shadow_intensity = conemap_obj->getShadowIntensity();
        params.depth_channel = conemap_obj->getDepthChannel();
        params.is_height_map = conemap_obj->isHeightMap() ? 1 : 0;
        params.shadow_noise_thread = conemap_obj->getShadowNoiseThread();

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            prt_gen_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            prt_gen_pipeline_layout_,
            { conemap_obj->getPrtGenTexDescSet() });

        cmd_buf->dispatch(
            (src_size.x + 7) / 8,
            (src_size.y + 7) / 8,
            1);

        renderer::helper::transitMapTextureFromStoreImage(cmd_buf, images/*, renderer::ImageLayout::GENERAL*/);
    }

    MinmaxDownsample(
        cmd_buf,
        prt_ds_first_pipeline_,
        prt_ds_first_pipeline_layout_,
        prt_ds_tex_desc_sets_[0],
        prt_ds_texes_[0],
        src_size);

    src_size.x = (src_size.x + 7) / 8;
    src_size.y = (src_size.y + 7) / 8;

    auto src_width = std::max(src_size.x, src_size.y);

    uint32_t double_buffer_idx = 0;
    for (uint32_t w = src_width; w > 8; w = (w + 7) / 8) {
        MinmaxDownsample(
            cmd_buf,
            prt_ds_pipeline_,
            prt_ds_pipeline_layout_,
            prt_ds_tex_desc_sets_[1 + double_buffer_idx],
            prt_ds_texes_[1 - double_buffer_idx],
            src_size);

        src_size.x = (src_size.x + 7) / 8;
        src_size.y = (src_size.y + 7) / 8;

        double_buffer_idx = 1 - double_buffer_idx;
    }
    
    {
        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_ds_final_pipeline_);
        glsl::PrtParams params = {};
        params.size = src_size;
        params.inv_size = glm::vec2(1.0f / src_size.x, 1.0f / src_size.y);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            prt_ds_final_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            prt_ds_final_pipeline_layout_,
            { conemap_obj->getPrtDsFinalTexDescSet(double_buffer_idx)});

        cmd_buf->dispatch(1, 1, 1);

        er::BarrierList barrier_list;
        er::helper::addBuffersToBarrierList(
            barrier_list,
            { conemap_obj->getMinmaxBuffer()->buffer },
            SET_FLAG_BIT(Access, SHADER_READ_BIT) | SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
    }

    src_size = glm::uvec2(conemap_obj->getPackTexture()->size);
    {
        renderer::helper::transitMapTextureToStoreImage(cmd_buf, { conemap_obj->getPackTexture()->image });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_pack_pipeline_);

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            prt_pack_pipeline_layout_,
            { conemap_obj->getPrtPackTexDescSet()});

        cmd_buf->dispatch(
            (src_size.x + 7) / 8,
            (src_size.y + 7) / 8,
            1);

        renderer::helper::transitMapTextureFromStoreImage(cmd_buf, { conemap_obj->getPackTexture()->image});
    }
}

void Prt::destroy(
    const std::shared_ptr<renderer::Device>& device) {

    for (int i = 0; i < prt_texes_.size(); i++) {
        if (prt_texes_[i]) {
            prt_texes_[i]->destroy(device);
        }
        if (prt_ds_texes_[0][i]) {
            prt_ds_texes_[0][i]->destroy(device);
        }
        if (prt_ds_texes_[1][i]) {
            prt_ds_texes_[1][i]->destroy(device);
        }
    }

    device->destroyDescriptorSetLayout(prt_gen_desc_set_layout_);
    device->destroyPipelineLayout(prt_gen_pipeline_layout_);
    device->destroyPipeline(prt_gen_pipeline_);

    device->destroyDescriptorSetLayout(prt_ds_desc_set_layout_);
    device->destroyDescriptorSetLayout(prt_ds_final_desc_set_layout_);
    device->destroyDescriptorSetLayout(prt_pack_desc_set_layout_);
    device->destroyPipelineLayout(prt_ds_first_pipeline_layout_);
    device->destroyPipeline(prt_ds_first_pipeline_);
    device->destroyPipelineLayout(prt_ds_pipeline_layout_);
    device->destroyPipeline(prt_ds_pipeline_);
    device->destroyPipelineLayout(prt_ds_final_pipeline_layout_);
    device->destroyPipeline(prt_ds_final_pipeline_);
    device->destroyPipelineLayout(prt_pack_pipeline_layout_);
    device->destroyPipeline(prt_pack_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
