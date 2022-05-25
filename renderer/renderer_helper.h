#pragma once
#include "renderer_structs.h"

namespace engine {
namespace renderer {
namespace helper {

DescriptorSetLayoutBinding getTextureSamplerDescriptionSetLayoutBinding(
    uint32_t binding, 
    ShaderStageFlags stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
    DescriptorType descript_type = DescriptorType::COMBINED_IMAGE_SAMPLER);

DescriptorSetLayoutBinding getBufferDescriptionSetLayoutBinding(
    uint32_t binding,
    ShaderStageFlags stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
    DescriptorType descript_type = DescriptorType::STORAGE_BUFFER);

PipelineColorBlendAttachmentState fillPipelineColorBlendAttachmentState(
    ColorComponentFlags color_write_mask = SET_FLAG_BIT(ColorComponent, ALL_BITS),
    bool blend_enable = false,
    BlendFactor src_color_blend_factor = BlendFactor::ONE,
    BlendFactor dst_color_blend_factor = BlendFactor::ZERO,
    BlendOp color_blend_op = BlendOp::ADD,
    BlendFactor src_alpha_blend_factor = BlendFactor::ONE,
    BlendFactor dst_alpha_blend_factor = BlendFactor::ZERO,
    BlendOp alpha_blend_op = BlendOp::ADD);

PipelineColorBlendStateCreateInfo fillPipelineColorBlendStateCreateInfo(
    const std::vector<PipelineColorBlendAttachmentState>& color_blend_attachments,
    bool logic_op_enable = false,
    LogicOp logic_op = LogicOp::NO_OP,
    glm::vec4 blend_constants = glm::vec4(0.0f));

PipelineRasterizationStateCreateInfo fillPipelineRasterizationStateCreateInfo(
    bool depth_clamp_enable = false,
    bool rasterizer_discard_enable = false,
    PolygonMode polygon_mode = PolygonMode::FILL,
    CullModeFlags cull_mode = SET_FLAG_BIT(CullMode, BACK_BIT),
    FrontFace front_face = FrontFace::COUNTER_CLOCKWISE,
    bool  depth_bias_enable = false,
    float depth_bias_constant_factor = 0.0f,
    float depth_bias_clamp = 0.0f,
    float depth_bias_slope_factor = 0.0f,
    float line_width = 1.0f);

PipelineMultisampleStateCreateInfo fillPipelineMultisampleStateCreateInfo(
    SampleCountFlagBits rasterization_samples = SampleCountFlagBits::SC_1_BIT,
    bool sample_shading_enable = false,
    float min_sample_shading = 1.0f,
    const SampleMask* sample_mask = nullptr,
    bool alpha_to_coverage_enable = false,
    bool alpha_to_one_enable = false);

StencilOpState fillStencilInfo(
    StencilOp fail_op = StencilOp::KEEP,
    StencilOp pass_op = StencilOp::KEEP,
    StencilOp depth_fail_op = StencilOp::KEEP,
    CompareOp compare_op = CompareOp::NEVER,
    uint32_t  compare_mask = 0xff,
    uint32_t  write_mask = 0xff,
    uint32_t  reference = 0x00);

PipelineDepthStencilStateCreateInfo fillPipelineDepthStencilStateCreateInfo(
    bool depth_test_enable = true,
    bool depth_write_enable = true,
    CompareOp depth_compare_op = CompareOp::LESS,
    bool depth_bounds_test_enable = false,
    float min_depth_bounds = 0.0f,
    float max_depth_bounds = 1.0f,
    bool stencil_test_enable = false,
    StencilOpState front = {},
    StencilOpState back = {});

void transitMapTextureToStoreImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::vector<std::shared_ptr<renderer::Image>>& images,
    const renderer::ImageLayout& old_layout = renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

void transitMapTextureFromStoreImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::vector<std::shared_ptr<renderer::Image>>& images,
    const renderer::ImageLayout& new_layout = renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

std::shared_ptr<ShaderModule> loadShaderModule(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& shader_name,
    const ShaderStageFlagBits& shader_stage);

void clearCachedShaderModules(const std::shared_ptr<renderer::Device>& device);

std::shared_ptr<renderer::PipelineLayout> createComputePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts,
    const uint32_t& push_const_range_size);

std::shared_ptr<renderer::Pipeline> createComputePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const std::string& compute_shader_name);

void releasePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::PipelineLayout>& pipeline_layout);

void releasePipeline(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::Pipeline>& pipeline);

} // namespace helper
} // namespace renderer
} // namespace engine
