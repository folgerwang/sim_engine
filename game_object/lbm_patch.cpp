#include "lbm_patch.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

static auto createLbmPatchDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(11);

    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DIFFUSE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));
#if 0
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(ALBEDO_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(METAL_ROUGHNESS_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(CONEMAP_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(EMISSIVE_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(OCCLUSION_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(THIN_FILM_LUT_INDEX));
/*    for (int i = 0; i < 7; ++i)
        bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(PRT_TEX_INDEX_0 + i));*/
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            PRT_PACK_TEX_INDEX,
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            PRT_PACK_INFO_TEX_INDEX,
            SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));

    renderer::DescriptorSetLayoutBinding ubo_pbr_layout_binding{};
    ubo_pbr_layout_binding.binding = PBR_CONSTANT_INDEX;
    ubo_pbr_layout_binding.descriptor_count = 1;
    ubo_pbr_layout_binding.descriptor_type = renderer::DescriptorType::UNIFORM_BUFFER;
    ubo_pbr_layout_binding.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    ubo_pbr_layout_binding.immutable_samplers = nullptr; // Optional
    bindings.push_back(ubo_pbr_layout_binding);
#endif
    return device->createDescriptorSetLayout(bindings);
}

static auto addLbmPatchTextures(
    const std::shared_ptr<renderer::DescriptorSet>& desc_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& dst_image) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        DIFFUSE_TEX_INDEX,
        nullptr,
        dst_image,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

static auto createLbmPatchPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& prt_desc_set_layout) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::PrtLightParams);

    auto desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(prt_desc_set_layout);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

} // namespace

namespace game_object {

LbmPatch::LbmPatch(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& patch_size) {

    lbm_patch_tex_ =
        std::make_shared<renderer::TextureInfo>();

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        patch_size,
        *lbm_patch_tex_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        renderer::ImageLayout::GENERAL,
        std::source_location::current(),
        renderer::ImageTiling::OPTIMAL,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        true);

    lbm_patch_desc_set_layout_ =
        createLbmPatchDescriptorSetLayout(
            device);

    lbm_patch_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool, lbm_patch_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto lbm_test_material_descs =
        addLbmPatchTextures(
            lbm_patch_desc_set_,
            texture_sampler,
            lbm_patch_tex_->view);

    device->updateDescriptorSets(lbm_test_material_descs);

    lbm_patch_pipeline_layout_ =
        createLbmPatchPipelineLayout(
            device,
            global_desc_set_layouts,
            lbm_patch_desc_set_layout_);

    lbm_patch_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            lbm_patch_pipeline_layout_,
            "lbm_patch_comp.spv",
            std::source_location::current());
}

void LbmPatch::update(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list) {

    auto buffer_size = lbm_patch_tex_->size;

    {
        renderer::BarrierList barrier_list;
        barrier_list.image_barriers.reserve(1);

        renderer::helper::addTexturesToBarrierList(
            barrier_list,
            { lbm_patch_tex_->image },
            renderer::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT));

        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
    }

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        lbm_patch_pipeline_);

    auto desc_sets = desc_set_list;
    desc_sets.push_back(lbm_patch_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        lbm_patch_pipeline_layout_,
        desc_sets);

    glsl::PrtLightParams params{};
    params.model_mat =
        glm::mat4(
            glm::vec4(1, 0, 0, 0),
            glm::vec4(0, 1, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, 0, 0, 1));

    params.buffer_size = buffer_size;
    params.inv_buffer_size = 1.0f / glm::vec2(buffer_size);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        lbm_patch_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->dispatch(
        (buffer_size.x + kLbmPatchDispatchX - 1) / kLbmPatchDispatchX,
        (buffer_size.y + kLbmPatchDispatchY - 1) / kLbmPatchDispatchY,
        1);

    {
        renderer::BarrierList barrier_list;
        barrier_list.image_barriers.reserve(1);

        renderer::helper::addTexturesToBarrierList(
            barrier_list,
            { lbm_patch_tex_->image },
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT));
    }
}

void LbmPatch::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(lbm_patch_desc_set_layout_);
    device->destroyPipelineLayout(lbm_patch_pipeline_layout_);
    device->destroyPipeline(lbm_patch_pipeline_);

    if (lbm_patch_tex_) {
        lbm_patch_tex_->destroy(device);
    }
}

} // game_object
} // engine
