#include "hair_test.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

static auto createHairTestDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(1);

    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            ALBEDO_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER));

    return device->createDescriptorSetLayout(bindings);
}

static auto addHairTestTextures(
    const std::shared_ptr<renderer::DescriptorSet>& desc_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_image) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ALBEDO_TEX_INDEX,
        texture_sampler,
        src_image,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static auto createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& prt_desc_set_layout) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::PrtLightParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(prt_desc_set_layout);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

} // namespace

namespace game_object {

HairTest::HairTest(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::TextureInfo>& hair_patch_tex,
    const glm::uvec2& display_size,
    std::shared_ptr<Plane> unit_plane) {

    hair_desc_set_layout_ =
        createHairTestDescriptorSetLayout(
            device);

    hair_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool, hair_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto hair_test_material_descs =
        addHairTestTextures(
            hair_desc_set_,
            texture_sampler,
            hair_patch_tex->view);

    device->updateDescriptorSets(
        hair_test_material_descs);

    hair_pipeline_layout_ =
        createPipelineLayout(
            device,
            global_desc_set_layouts,
            hair_desc_set_layout_);

    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "hair_test_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT);
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "hair_test_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT);

    hair_pipeline_ =
        device->createPipeline(
            render_pass,
            hair_pipeline_layout_,
            unit_plane->getBindingDescs(),
            unit_plane->getAttribDescs(),
            input_assembly,
            graphic_pipeline_info,
            shader_modules,
            display_size);
}

void HairTest::draw(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    std::shared_ptr<Plane> unit_plane,
    std::shared_ptr<Box> unit_box) {

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        hair_pipeline_);

    renderer::DescriptorSetList desc_sets = desc_set_list;
    desc_sets.push_back(hair_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        hair_pipeline_layout_,
        desc_sets);

    glsl::PrtLightParams params{};
    params.model_mat =
        glm::mat4(
            glm::vec4(1, 0, 0, 0),
            glm::vec4(0, 1, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, 0, 0, 1));

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        hair_pipeline_layout_,
        &params,
        sizeof(params));

    if (unit_plane) {
        unit_plane->draw(cmd_buf);
    }

/*    if (unit_box) {
        unit_box->draw(cmd_buf);
    }*/
}

void HairTest::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(hair_desc_set_layout_);
    device->destroyPipelineLayout(hair_pipeline_layout_);
    device->destroyPipeline(hair_pipeline_);
}

} // game_object
} // engine
