#include "prt_test.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {
static std::shared_ptr<renderer::DescriptorSetLayout> createPrtDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(4);

    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(PRT_BASE_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(PRT_BUMP_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(PRT_CONEMAP_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(PRT_PRT_TEX_INDEX));

    return device->createDescriptorSetLayout(bindings);
}

static renderer::WriteDescriptorList addPrtTextures(
    const std::shared_ptr<renderer::DescriptorSet>& desc_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& base_tex,
    const renderer::TextureInfo& bump_tex,
    const renderer::TextureInfo& conemap_tex,
    const renderer::TextureInfo& prt_tex) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

    // diffuse.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PRT_BASE_TEX_INDEX,
        texture_sampler,
        base_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // normal.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PRT_BUMP_TEX_INDEX,
        texture_sampler,
        bump_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // specular.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PRT_CONEMAP_TEX_INDEX,
        texture_sampler,
        conemap_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // specular.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PRT_PRT_TEX_INDEX,
        texture_sampler,
        prt_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static std::shared_ptr<renderer::PipelineLayout> createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& prt_desc_set_layout) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::ModelParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(prt_desc_set_layout);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

} // namespace

namespace game_object {

PrtTest::PrtTest(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size,
    std::shared_ptr<Plane> unit_plane,
    const renderer::TextureInfo& prt_base_tex,
    const renderer::TextureInfo& prt_bump_tex) {

    cone_map_tex_ = std::make_shared<renderer::TextureInfo>();
    prt_tex_ = std::make_shared<renderer::TextureInfo>();

    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R8G8B8A8_UNORM,
        glm::uvec2(128, 128),
        *cone_map_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::create2DTextureImage(
        device_info,
        renderer::Format::R32G32B32A32_UINT,
        glm::uvec2(128, 128),
        *prt_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    prt_desc_set_layout_ = createPrtDescriptorSetLayout(
        device_info.device);

    prt_desc_set_ = device_info.device->createDescriptorSets(
        descriptor_pool, prt_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto material_descs = addPrtTextures(
        prt_desc_set_,
        texture_sampler,
        prt_base_tex,
        prt_bump_tex,
        *cone_map_tex_,
        *prt_tex_);

    device_info.device->updateDescriptorSets(material_descs);

    prt_pipeline_layout_ = createPipelineLayout(
        device_info.device,
        global_desc_set_layouts,
        prt_desc_set_layout_);

    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device_info.device,
            "prt_test_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT);
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device_info.device,
            "prt_test_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT);

    prt_pipeline_ = device_info.device->createPipeline(
        render_pass,
        prt_pipeline_layout_,
        unit_plane->getBindingDescs(),
        unit_plane->getAttribDescs(),
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size);

}

void PrtTest::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    std::shared_ptr<Plane> unit_plane) {

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, prt_pipeline_);

    renderer::DescriptorSetList desc_sets = desc_set_list;
    desc_sets.push_back(prt_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        prt_pipeline_layout_,
        desc_sets);

    glsl::ModelParams model_params{};
    model_params.model_mat =
        glm::mat4(
            glm::vec4(1, 0, 0, 0),
            glm::vec4(0, 1, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, 0, 0, 1));

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        prt_pipeline_layout_,
        &model_params,
        sizeof(model_params));

    if (unit_plane) {
        unit_plane->draw(cmd_buf);
    }
}

void PrtTest::destroy(const std::shared_ptr<renderer::Device>& device) {
    if (cone_map_tex_) {
        cone_map_tex_->destroy(device);
    }

    if (prt_tex_) {
        prt_tex_->destroy(device);
    }

    device->destroyDescriptorSetLayout(prt_desc_set_layout_);
    device->destroyPipelineLayout(prt_pipeline_layout_);
    device->destroyPipeline(prt_pipeline_);
}

} // game_object
} // engine