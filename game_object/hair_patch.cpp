#include "hair_patch.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

static auto createHairPatchDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(2);

    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_COLOR_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));

    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_WEIGHT_TEX_INDEX,
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
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            PRT_PACK_INFO_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
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

static auto addHairPatchTextures(
    const std::shared_ptr<renderer::DescriptorSet>& desc_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& dst_color_image,
    const std::shared_ptr<renderer::ImageView>& dst_weight_image) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        DST_COLOR_TEX_INDEX,
        nullptr,
        dst_color_image,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        DST_WEIGHT_TEX_INDEX,
        nullptr,
        dst_weight_image,
        renderer::ImageLayout::GENERAL);

    return descriptor_writes;
}

static auto createHairPatchPipelineLayout(
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
        { push_const_range });
}

} // namespace

namespace game_object {

HairPatch::HairPatch(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& patch_size) {

    hair_patch_color_tex_ =
        std::make_shared<renderer::TextureInfo>();

    hair_patch_weight_tex_ =
        std::make_shared<renderer::TextureInfo>();

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        patch_size,
        *hair_patch_color_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL,
        renderer::ImageTiling::OPTIMAL,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        true);

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16_SFLOAT,
        patch_size,
        *hair_patch_weight_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::GENERAL,
        renderer::ImageTiling::OPTIMAL,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        true);

    hair_patch_desc_set_layout_ =
        createHairPatchDescriptorSetLayout(
            device);

    hair_patch_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool, hair_patch_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto hair_test_material_descs =
        addHairPatchTextures(
            hair_patch_desc_set_,
            texture_sampler,
            hair_patch_color_tex_->view,
            hair_patch_weight_tex_->view);

    device->updateDescriptorSets(hair_test_material_descs);

    hair_patch_pipeline_layout_ =
        createHairPatchPipelineLayout(
            device,
            global_desc_set_layouts,
            hair_patch_desc_set_layout_);

    hair_patch_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            hair_patch_pipeline_layout_,
            "hair_patch_comp.spv");
}

void HairPatch::update(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list) {

    cmd_buf->beginDebugUtilsLabel("hair patch update");

    auto buffer_size = hair_patch_color_tex_->size;

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        hair_patch_pipeline_);

    auto desc_sets = desc_set_list;
    desc_sets.push_back(hair_patch_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        hair_patch_pipeline_layout_,
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
        hair_patch_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->dispatch(
        (buffer_size.x + 15) / 16,
        (buffer_size.y + 15) / 16,
        1);

    renderer::BarrierList barrier_list;
    barrier_list.image_barriers.reserve(1);

    renderer::helper::addTexturesToBarrierList(
        barrier_list,
        { hair_patch_color_tex_->image,
          hair_patch_weight_tex_->image },
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT) |
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(Access, SHADER_READ_BIT));

    cmd_buf->addBarriers(
        barrier_list,
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT));

    cmd_buf->endDebugUtilsLabel();
}

void HairPatch::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(hair_patch_desc_set_layout_);
    device->destroyPipelineLayout(hair_patch_pipeline_layout_);
    device->destroyPipeline(hair_patch_pipeline_);

    if (hair_patch_color_tex_) {
        hair_patch_color_tex_->destroy(device);
    }

    if (hair_patch_weight_tex_) {
        hair_patch_weight_tex_->destroy(device);
    }
}

} // game_object
} // engine
