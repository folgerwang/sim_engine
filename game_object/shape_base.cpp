#include "shape_base.h"
#include "shaders/global_definition.glsl.h"
#include "renderer/renderer_helper.h"

namespace engine {
namespace {
static auto createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_2_FLAG_BITS(ShaderStage, VERTEX_BIT, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::BaseShapeDrawParams);

    renderer::DescriptorSetLayoutList desc_set_layouts =
        global_desc_set_layouts;

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

static renderer::ShaderModuleList getBaseShapeShaderModules(
    std::shared_ptr<renderer::Device> device) {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "base_shape_draw_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "base_shape_draw_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

    return shader_modules;
}

static auto createPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::vector<renderer::VertexInputBindingDescription>& binding_descs,
    const std::vector<renderer::VertexInputAttributeDescription>& attribute_descs,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    renderer::RasterizationStateOverride rasterization_state_info;

    auto shader_modules = getBaseShapeShaderModules(device);
    auto pipeline = device->createPipeline(
        pipeline_layout,
        binding_descs,
        attribute_descs,
        input_assembly,
        graphic_pipeline_info,
        getBaseShapeShaderModules(device),
        frame_buffer_format,
        rasterization_state_info,
        std::source_location::current());

    return pipeline;
}
} // namespace

namespace game_object {

std::vector<renderer::VertexInputBindingDescription> ShapeBase::s_binding_descs_;
std::vector<renderer::VertexInputAttributeDescription> ShapeBase::s_attrib_descs_;
std::shared_ptr<renderer::PipelineLayout> ShapeBase::s_base_shape_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> ShapeBase::s_base_shape_pipeline_;

void ShapeBase::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::PipelineRenderbufferFormats& frame_buffer_format) {
    renderer::VertexInputBindingDescription binding_desc;
    renderer::VertexInputAttributeDescription attribute_desc;

    uint32_t binding_idx = 0;
    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(glm::vec3);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_POSITION;
    attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
    attribute_desc.offset = 0;
    s_attrib_descs_.push_back(attribute_desc);
    binding_idx++;

    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(glm::vec3);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_NORMAL;
    attribute_desc.format = renderer::Format::R32G32B32_SFLOAT;
    attribute_desc.offset = 0;
    s_attrib_descs_.push_back(attribute_desc);
    binding_idx++;

    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(glm::vec4);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_TANGENT;
    attribute_desc.format = renderer::Format::R32G32B32A32_SFLOAT;
    attribute_desc.offset = 0;
    s_attrib_descs_.push_back(attribute_desc);
    binding_idx++;

    binding_desc.binding = binding_idx;
    binding_desc.stride = sizeof(glm::vec2);
    binding_desc.input_rate = renderer::VertexInputRate::VERTEX;
    s_binding_descs_.push_back(binding_desc);

    attribute_desc.binding = binding_idx;
    attribute_desc.location = VINPUT_TEXCOORD0;
    attribute_desc.format = renderer::Format::R32G32_SFLOAT;
    attribute_desc.offset = 0;
    s_attrib_descs_.push_back(attribute_desc);
    binding_idx++;

    s_base_shape_pipeline_layout_ =
        createPipelineLayout(device, global_desc_set_layouts);
    s_base_shape_pipeline_ = createPipeline(
        device,
        s_base_shape_pipeline_layout_,
        graphic_pipeline_info,
        s_binding_descs_,
        s_attrib_descs_,
        frame_buffer_format);
}

void ShapeBase::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyPipelineLayout(s_base_shape_pipeline_layout_);
    s_base_shape_pipeline_layout_ = nullptr;
    device->destroyPipeline(s_base_shape_pipeline_);
    s_base_shape_pipeline_ = nullptr;
}
    
void ShapeBase::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    if (position_buffer_) {
        position_buffer_->destroy(device);
    }
    if (uv_buffer_) {
        uv_buffer_->destroy(device);
    }
    if (normal_buffer_) {
        normal_buffer_->destroy(device);
    }
    if (tangent_buffer_) {
        tangent_buffer_->destroy(device);
    }
    if (index_buffer_) {
        index_buffer_->destroy(device);
    }
}

} // game_object
} // engine