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
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::CommandBuffer>& cmd_buf,
        const std::shared_ptr<er::Pipeline>& pipeline,
        const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
        const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
        const std::shared_ptr<er::DescriptorSet>& desc_set,
        const std::shared_ptr<er::Sampler>& texture_sampler,
        const std::array<std::shared_ptr<er::TextureInfo>, 7>& src_prt_texes,
        const std::array<std::shared_ptr<er::TextureInfo>, 7>& dst_prt_texes,
        const glm::uvec2& src_buffer_size)
    {
        std::vector<std::shared_ptr<er::Image>> images;
        images.reserve(dst_prt_texes.size());
        for (const auto& tex : dst_prt_texes) {
            images.push_back(tex->image);
        }

        er::helper::transitMapTextureToStoreImage(cmd_buf, images);

        // create a global ibl texture descriptor set.
        auto prt_texture_descs = addDsPrtTextures(
            desc_set,
            texture_sampler,
            src_prt_texes,
            dst_prt_texes);
        device->updateDescriptorSets(prt_texture_descs);

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
    const glm::uvec2& buffer_size) {

    const auto& device = device_info.device;

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

        prt_ds1_texes_[i] = std::make_shared<renderer::TextureInfo>();
        renderer::Helper::create2DTextureImage(
            device_info,
            renderer::Format::R32G32B32A32_SFLOAT,
            glm::uvec2((buffer_size.x + 7) / 8 * 2, (buffer_size.y + 7) / 8),
            *prt_ds1_texes_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        prt_ds2_texes_[i] = std::make_shared<renderer::TextureInfo>();
        renderer::Helper::create2DTextureImage(
            device_info,
            renderer::Format::R32G32B32A32_SFLOAT,
            glm::uvec2((buffer_size.x + 63) / 64 * 2, (buffer_size.y + 63) / 64),
            *prt_ds2_texes_[i],
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

    prt_ds1_texes_[6] = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32_SFLOAT,
        glm::uvec2((buffer_size.x + 7) / 8 * 2, (buffer_size.y + 7) / 8),
        *prt_ds1_texes_[6],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    prt_ds2_texes_[6] = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32_SFLOAT,
        glm::uvec2((buffer_size.x + 63) / 64 * 2, (buffer_size.y + 63) / 64),
        *prt_ds2_texes_[6],
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    prt_pack_tex_ = std::make_shared<renderer::TextureInfo>();
    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32G32B32A32_UINT,
        buffer_size,
        *prt_pack_tex_,
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

    prt_desc_set_layout_ =
        device->createDescriptorSetLayout(bindings);

    prt_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_desc_set_layout_, 1)[0];

    prt_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_desc_set_layout_);

    prt_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_pipeline_layout_,
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

    prt_ds_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_ds_desc_set_layout_, 1)[0];

    prt_ds_s_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_ds_desc_set_layout_);

    prt_ds_s_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_ds_s_pipeline_layout_,
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

    prt_minmax_buffer_ = std::make_shared<renderer::BufferInfo>();
    device->createBuffer(
        sizeof(glsl::PrtMinmaxInfo),
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_CACHED_BIT),
        0,
        prt_minmax_buffer_->buffer,
        prt_minmax_buffer_->memory);

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

    prt_ds_f_desc_set_layout_ =
        device->createDescriptorSetLayout(ds_f_bindings);

    prt_ds_f_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_ds_f_desc_set_layout_, 1)[0];

    prt_ds_f_pipeline_layout_ =
        createPrtPipelineLayout(
            device,
            prt_ds_f_desc_set_layout_);

    prt_ds_f_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            prt_ds_f_pipeline_layout_,
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

    prt_pack_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            prt_pack_desc_set_layout_, 1)[0];

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
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& bump_tex) {

    if (prt_texes_.empty()) {
        return;
    }

    // create prt textures.
    glm::uvec2 src_size = glm::uvec2(prt_texes_[0]->size);
    {
        std::vector<std::shared_ptr<renderer::Image>> images;
        images.reserve(prt_texes_.size());
        for (const auto& tex : prt_texes_) {
            images.push_back(tex->image);
        }

        renderer::helper::transitMapTextureToStoreImage(cmd_buf, images);

        // create a global ibl texture descriptor set.
        auto prt_texture_descs = addPrtTextures(
            prt_tex_desc_set_,
            texture_sampler,
            bump_tex.view,
            prt_texes_);
        device->updateDescriptorSets(prt_texture_descs);

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_pipeline_);
        glsl::PrtParams params = {};
        params.size = src_size;
        params.inv_size = glm::vec2(1.0f / src_size.x, 1.0f / src_size.y);

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
            (src_size.x + 7) / 8,
            (src_size.y + 7) / 8,
            1);

        renderer::helper::transitMapTextureFromStoreImage(cmd_buf, images/*, renderer::ImageLayout::GENERAL*/);
    }

    MinmaxDownsample(
        device,
        cmd_buf,
        prt_ds_s_pipeline_,
        prt_ds_s_pipeline_layout_,
        prt_ds_desc_set_layout_,
        prt_ds_tex_desc_set_,
        texture_sampler,
        prt_texes_,
        prt_ds1_texes_,
        src_size);

    src_size.x = (src_size.x + 7) / 8;
    src_size.y = (src_size.y + 7) / 8;

    MinmaxDownsample(
        device,
        cmd_buf,
        prt_ds_pipeline_,
        prt_ds_pipeline_layout_,
        prt_ds_desc_set_layout_,
        prt_ds_tex_desc_set_,
        texture_sampler,
        prt_ds1_texes_,
        prt_ds2_texes_,
        src_size);

    src_size.x = (src_size.x + 7) / 8;
    src_size.y = (src_size.y + 7) / 8;

    {
        // create a global ibl texture descriptor set.
        auto prt_texture_descs = addDsFinalPrtTextures(
            prt_ds_f_tex_desc_set_,
            texture_sampler,
            prt_ds2_texes_,
            prt_minmax_buffer_);
        device->updateDescriptorSets(prt_texture_descs);

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_ds_f_pipeline_);
        glsl::PrtParams params = {};
        params.size = src_size;
        params.inv_size = glm::vec2(1.0f / src_size.x, 1.0f / src_size.y);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            prt_ds_f_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            prt_ds_f_pipeline_layout_,
            { prt_ds_f_tex_desc_set_ });

        cmd_buf->dispatch(
            1,
            1,
            1);

        er::BarrierList barrier_list;
        er::helper::addBuffersToBarrierList(
            barrier_list,
            { prt_minmax_buffer_->buffer },
            SET_FLAG_BIT(Access, SHADER_READ_BIT) | SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
    }

    src_size = glm::uvec2(prt_texes_[0]->size);
    {
        renderer::helper::transitMapTextureToStoreImage(cmd_buf, { prt_pack_tex_->image });

        // create a global ibl texture descriptor set.
        auto prt_texture_descs = addPackedPrtTextures(
            prt_pack_tex_desc_set_,
            texture_sampler,
            prt_texes_,
            prt_minmax_buffer_,
            prt_pack_tex_);
        device->updateDescriptorSets(prt_texture_descs);

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            prt_pack_pipeline_);

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            prt_pack_pipeline_layout_,
            { prt_pack_tex_desc_set_ });

        cmd_buf->dispatch(
            (src_size.x + 7) / 8,
            (src_size.y + 7) / 8,
            1);

        renderer::helper::transitMapTextureFromStoreImage(cmd_buf, { prt_pack_tex_->image });
    }
}

void Prt::destroy(
    const std::shared_ptr<renderer::Device>& device) {

    for (int i = 0; i < prt_texes_.size(); i++) {
        if (prt_texes_[i]) {
            prt_texes_[i]->destroy(device);
        }
        if (prt_ds1_texes_[i]) {
            prt_ds1_texes_[i]->destroy(device);
        }
        if (prt_ds2_texes_[i]) {
            prt_ds2_texes_[i]->destroy(device);
        }
    }

    prt_minmax_buffer_->destroy(device);
    prt_pack_tex_->destroy(device);

    device->destroyDescriptorSetLayout(prt_desc_set_layout_);
    device->destroyPipelineLayout(prt_pipeline_layout_);
    device->destroyPipeline(prt_pipeline_);

    device->destroyDescriptorSetLayout(prt_ds_desc_set_layout_);
    device->destroyDescriptorSetLayout(prt_ds_f_desc_set_layout_);
    device->destroyDescriptorSetLayout(prt_pack_desc_set_layout_);
    device->destroyPipelineLayout(prt_ds_s_pipeline_layout_);
    device->destroyPipeline(prt_ds_s_pipeline_);
    device->destroyPipelineLayout(prt_ds_pipeline_layout_);
    device->destroyPipeline(prt_ds_pipeline_);
    device->destroyPipelineLayout(prt_ds_f_pipeline_layout_);
    device->destroyPipeline(prt_ds_f_pipeline_);
    device->destroyPipelineLayout(prt_pack_pipeline_layout_);
    device->destroyPipeline(prt_pack_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
