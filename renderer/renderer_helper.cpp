#include <fstream>

#include "renderer.h"
#include "renderer_helper.h"
#include "../engine_helper.h"

namespace engine {
namespace renderer {
namespace helper {

AttachmentDescription FillAttachmentDescription(
    Format format,
    SampleCountFlagBits samples,
    ImageLayout initial_layout,
    ImageLayout final_layout,
    AttachmentLoadOp load_op/* = AttachmentLoadOp::CLEAR*/,
    AttachmentStoreOp store_op/* = AttachmentStoreOp::STORE*/,
    AttachmentLoadOp stencil_load_op/* = AttachmentLoadOp::DONT_CARE*/,
    AttachmentStoreOp stencil_store_op/* = AttachmentStoreOp::DONT_CARE*/) {

    AttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = samples;
    attachment.initial_layout = initial_layout;
    attachment.final_layout = final_layout;
    attachment.load_op = load_op;
    attachment.store_op = store_op;
    attachment.stencil_load_op = stencil_load_op;
    attachment.stencil_store_op = stencil_store_op;

    return attachment;
}

SubpassDescription FillSubpassDescription(
    PipelineBindPoint pipeline_bind_point,
    const std::vector<AttachmentReference>& color_attachments,
    const AttachmentReference* depth_stencil_attachment,
    SubpassDescriptionFlags flags/* = static_cast<SubpassDescriptionFlags>(0)*/,
    const std::vector<AttachmentReference>& input_attachments/* = {}*/,
    const std::vector<AttachmentReference>& resolve_attachments/* = {}*/) {
    SubpassDescription desc{};
    desc.flags = flags;
    desc.input_attachments = input_attachments;
    desc.color_attachments = color_attachments;
    desc.resolve_attachments = resolve_attachments;
    if (depth_stencil_attachment) {
        desc.depth_stencil_attachment.resize(1);
        desc.depth_stencil_attachment[0] = *depth_stencil_attachment;
    }
    desc.preserve_attachment_count = 0;
    desc.preserve_attachments = nullptr;

    return desc;
}

SubpassDependency FillSubpassDependency(
    uint32_t src_subpass,
    uint32_t dst_subpass,
    PipelineStageFlags src_stage_mask,
    PipelineStageFlags dst_stage_mask,
    AccessFlags src_access_mask,
    AccessFlags dst_access_mask,
    DependencyFlags dependency_flags/* = 0*/) {
    SubpassDependency dependency{};
    dependency.src_subpass = src_subpass;
    dependency.dst_subpass = dst_subpass;
    dependency.src_stage_mask = src_stage_mask;
    dependency.dst_stage_mask = dst_stage_mask;
    dependency.src_access_mask = src_access_mask;
    dependency.dst_access_mask = dst_access_mask;
    dependency.dependency_flags = dependency_flags;
    return dependency;
}

std::shared_ptr<RenderPass> createRenderPass(
    const std::shared_ptr<Device>& device,
    Format format,
    Format depth_format,
    bool clear/* = false */ ,
    SampleCountFlagBits sample_count/* = SampleCountFlagBits::SC_1_BIT*/,
    ImageLayout color_image_layout/* = ImageLayout::COLOR_ATTACHMENT_OPTIMAL*/) {
    auto color_attachment = FillAttachmentDescription(
        format,
        sample_count,
        clear ? ImageLayout::UNDEFINED : color_image_layout,
        color_image_layout,
        clear ? AttachmentLoadOp::CLEAR : AttachmentLoadOp::LOAD);

    AttachmentReference color_attachment_ref(
        0,
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    auto depth_attachment = FillAttachmentDescription(
        depth_format,
        sample_count,
        clear ? ImageLayout::UNDEFINED : ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        clear ? AttachmentLoadOp::CLEAR : AttachmentLoadOp::LOAD);

    AttachmentReference depth_attachment_ref(
        1,
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    auto subpass = FillSubpassDescription(
        PipelineBindPoint::GRAPHICS,
        { color_attachment_ref },
        &depth_attachment_ref);

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT) |
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_READ_BIT));

    std::vector<AttachmentDescription> attachments(2);
    attachments[0] = color_attachment;
    attachments[1] = depth_attachment;

    return device->createRenderPass(
        attachments,
        { subpass },
        { depency });
}

std::shared_ptr<RenderPass> createCubemapRenderPass(
    const std::shared_ptr<Device>& device,
    Format format/* = Format::R16G16B16A16_SFLOAT */) {
    auto color_attachment = FillAttachmentDescription(
        format,
        SampleCountFlagBits::SC_1_BIT,
        ImageLayout::UNDEFINED,
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    std::vector<AttachmentReference> color_attachment_refs(6);
    for (uint32_t i = 0; i < 6; i++) {
        color_attachment_refs[i].attachment_ = i;
        color_attachment_refs[i].layout_ = ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    auto subpass = FillSubpassDescription(
        PipelineBindPoint::GRAPHICS,
        color_attachment_refs,
        nullptr);

    auto depency = FillSubpassDependency(~0U, 0,
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT),
        0,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT));

    std::vector<AttachmentDescription> attachments =
        { 6,
          color_attachment };

    return device->createRenderPass(
        attachments,
        { subpass },
        { depency });
}

DescriptorSetLayoutBinding getTextureSamplerDescriptionSetLayoutBinding(
    uint32_t binding, 
    ShaderStageFlags stage_flags/* = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT)*/,
    DescriptorType descript_type/* = DescriptorType::COMBINED_IMAGE_SAMPLER*/) {
    DescriptorSetLayoutBinding texture_binding{};
    texture_binding.binding = binding;
    texture_binding.descriptor_count = 1;
    texture_binding.descriptor_type = descript_type;
    texture_binding.immutable_samplers = nullptr;
    texture_binding.stage_flags = stage_flags;

    return texture_binding;
}

DescriptorSetLayoutBinding getBufferDescriptionSetLayoutBinding(
    uint32_t binding,
    ShaderStageFlags stage_flags/* = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT)*/,
    DescriptorType descript_type/* = DescriptorType::STORAGE_BUFFER*/) {
    DescriptorSetLayoutBinding buffer_binding{};
    buffer_binding.binding = binding;
    buffer_binding.descriptor_count = 1;
    buffer_binding.descriptor_type = descript_type;
    buffer_binding.immutable_samplers = nullptr;
    buffer_binding.stage_flags = stage_flags;

    return buffer_binding;
}

PipelineColorBlendAttachmentState fillPipelineColorBlendAttachmentState(
    ColorComponentFlags color_write_mask/* = SET_FLAG_BIT(ColorComponent, ALL_BITS)*/,
    bool blend_enable/* = false*/,
    BlendFactor src_color_blend_factor/* = BlendFactor::ONE*/,
    BlendFactor dst_color_blend_factor/* = BlendFactor::ZERO*/,
    BlendOp color_blend_op/* = BlendOp::ADD*/,
    BlendFactor src_alpha_blend_factor/* = BlendFactor::ONE*/,
    BlendFactor dst_alpha_blend_factor/* = BlendFactor::ZERO*/,
    BlendOp alpha_blend_op/* = BlendOp::ADD*/) {
    PipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.color_write_mask = color_write_mask;
    color_blend_attachment.blend_enable = blend_enable;
    color_blend_attachment.src_color_blend_factor = src_color_blend_factor; // Optional
    color_blend_attachment.dst_color_blend_factor = dst_color_blend_factor; // Optional
    color_blend_attachment.color_blend_op = color_blend_op; // Optional
    color_blend_attachment.src_alpha_blend_factor = src_alpha_blend_factor; // Optional
    color_blend_attachment.dst_alpha_blend_factor = dst_alpha_blend_factor; // Optional
    color_blend_attachment.alpha_blend_op = alpha_blend_op; // Optional

    return color_blend_attachment;
}

PipelineColorBlendStateCreateInfo fillPipelineColorBlendStateCreateInfo(
    const std::vector<PipelineColorBlendAttachmentState>& color_blend_attachments,
    bool logic_op_enable/* = false*/,
    LogicOp logic_op/* = LogicOp::NO_OP*/,
    glm::vec4 blend_constants/* = glm::vec4(0.0f)*/) {
    PipelineColorBlendStateCreateInfo color_blending{};
    color_blending.logic_op_enable = logic_op_enable;
    color_blending.logic_op = logic_op; // Optional
    color_blending.attachment_count = static_cast<uint32_t>(color_blend_attachments.size());
    color_blending.attachments = color_blend_attachments.data();
    color_blending.blend_constants = blend_constants; // Optional

    return color_blending;
}

PipelineRasterizationStateCreateInfo fillPipelineRasterizationStateCreateInfo(
    bool depth_clamp_enable/* = false*/,
    bool rasterizer_discard_enable/* = false*/,
    PolygonMode polygon_mode/* = PolygonMode::FILL*/,
    CullModeFlags cull_mode/* = SET_FLAG_BIT(CullMode, BACK_BIT)*/,
    FrontFace front_face/* = FrontFace::COUNTER_CLOCKWISE*/,
    bool  depth_bias_enable/* = false*/,
    float depth_bias_constant_factor/* = 0.0f*/,
    float depth_bias_clamp/* = 0.0f*/,
    float depth_bias_slope_factor/* = 0.0f*/,
    float line_width/* = 1.0f*/) {
    PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depth_clamp_enable = depth_clamp_enable;
    rasterizer.rasterizer_discard_enable = rasterizer_discard_enable;
    rasterizer.polygon_mode = polygon_mode;
    rasterizer.line_width = line_width;
    rasterizer.cull_mode = cull_mode;
    rasterizer.front_face = front_face;
    rasterizer.depth_bias_enable = depth_bias_enable;
    rasterizer.depth_bias_constant_factor = depth_bias_constant_factor; // Optional
    rasterizer.depth_bias_clamp = depth_bias_clamp; // Optional
    rasterizer.depth_bias_slope_factor = depth_bias_slope_factor; // Optional

    return rasterizer;
}

PipelineMultisampleStateCreateInfo fillPipelineMultisampleStateCreateInfo(
    SampleCountFlagBits rasterization_samples/* = SampleCountFlagBits::SC_1_BIT*/,
    bool sample_shading_enable/* = false*/,
    float min_sample_shading/* = 1.0f*/,
    const SampleMask* sample_mask/* = nullptr*/,
    bool alpha_to_coverage_enable/* = false*/,
    bool alpha_to_one_enable/* = false*/) {
    PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sample_shading_enable = sample_shading_enable;
    multisampling.rasterization_samples = rasterization_samples;
    multisampling.min_sample_shading = min_sample_shading; // Optional
    multisampling.sample_mask = sample_mask; // Optional
    multisampling.alpha_to_coverage_enable = alpha_to_coverage_enable; // Optional
    multisampling.alpha_to_one_enable = alpha_to_one_enable; // Optional

    return multisampling;
}

StencilOpState fillStencilInfo(
    StencilOp fail_op/* = StencilOp::KEEP*/,
    StencilOp pass_op/* = StencilOp::KEEP*/,
    StencilOp depth_fail_op/* = StencilOp::KEEP*/,
    CompareOp compare_op/* = CompareOp::NEVER*/,
    uint32_t  compare_mask/* = 0xff*/,
    uint32_t  write_mask/* = 0xff*/,
    uint32_t  reference/* = 0x00*/) {
    StencilOpState stencil_op_state;
    stencil_op_state.fail_op = fail_op;
    stencil_op_state.pass_op = pass_op;
    stencil_op_state.depth_fail_op = depth_fail_op;
    stencil_op_state.compare_op = compare_op;
    stencil_op_state.compare_mask = compare_mask;
    stencil_op_state.write_mask = write_mask;
    stencil_op_state.reference = reference;

    return stencil_op_state;
}

PipelineDepthStencilStateCreateInfo fillPipelineDepthStencilStateCreateInfo(
    bool depth_test_enable/* = true*/,
    bool depth_write_enable/* = true*/,
    CompareOp depth_compare_op/* = CompareOp::LESS*/,
    bool depth_bounds_test_enable/* = false*/,
    float min_depth_bounds/* = 0.0f*/,
    float max_depth_bounds/* = 1.0f*/,
    bool stencil_test_enable/* = false*/,
    StencilOpState front/* = {}*/,
    StencilOpState back/* = {}*/) {
    PipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.depth_test_enable = depth_test_enable;
    depth_stencil.depth_write_enable = depth_write_enable;
    depth_stencil.depth_compare_op = depth_compare_op;
    depth_stencil.depth_bounds_test_enable = depth_bounds_test_enable;
    depth_stencil.min_depth_bounds = min_depth_bounds; // Optional
    depth_stencil.max_depth_bounds = max_depth_bounds; // Optional
    depth_stencil.stencil_test_enable = stencil_test_enable;
    depth_stencil.front = front; // Optional
    depth_stencil.back = back; // Optional

    return depth_stencil;
}

void addTexturesToBarrierList(
    renderer::BarrierList& barrier_list,
    const std::vector<std::shared_ptr<renderer::Image>>& images,
    const renderer::ImageLayout& new_layout,
    AccessFlags src_access_mask,
    AccessFlags dst_access_mask) {
    barrier_list.image_barriers.reserve(barrier_list.image_barriers.size() + images.size());

    for (auto& image : images) {
        renderer::ImageMemoryBarrier barrier;
        barrier.image = image;
        barrier.old_layout = image->getImageLayout();
        barrier.new_layout = new_layout;
        image->setImageLayout(barrier.new_layout);
        barrier.src_access_mask = src_access_mask;
        barrier.dst_access_mask = dst_access_mask;
        barrier.subresource_range.aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
        barrier_list.image_barriers.push_back(barrier);
    }
}

void addBuffersToBarrierList(
    renderer::BarrierList& barrier_list,
    const std::vector<std::shared_ptr<renderer::Buffer>>& buffers,
    AccessFlags src_access_mask,
    AccessFlags dst_access_mask) {
    barrier_list.buffer_barriers.reserve(barrier_list.buffer_barriers.size() + buffers.size());

    for (auto& buffer : buffers) {
        renderer::BufferMemoryBarrier barrier;

        barrier.buffer = buffer;
        barrier.offset = 0;
        barrier.size = buffer->getSize();
        barrier.src_access_mask = src_access_mask;
        barrier.dst_access_mask = dst_access_mask;
        barrier_list.buffer_barriers.push_back(barrier);
    }
}

void transitMapTextureToStoreImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::vector<std::shared_ptr<renderer::Image>>& images) {
    renderer::BarrierList barrier_list;
    barrier_list.image_barriers.reserve(images.size());

    addTexturesToBarrierList(
        barrier_list,
        images,
        renderer::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(Access, SHADER_READ_BIT) | SET_FLAG_BIT(Access, SHADER_WRITE_BIT));

    cmd_buf->addBarriers(
        barrier_list,
        SET_FLAG_BIT(PipelineStage, VERTEX_SHADER_BIT) |
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
}

void transitMapTextureFromStoreImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::vector<std::shared_ptr<renderer::Image>>& images,
    const renderer::ImageLayout& new_layout/* = renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL*/) {
    renderer::BarrierList barrier_list;
    barrier_list.image_barriers.reserve(images.size());

    addTexturesToBarrierList(
        barrier_list,
        images,
        new_layout,
        SET_FLAG_BIT(Access, SHADER_READ_BIT) | SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(Access, SHADER_READ_BIT));

    cmd_buf->addBarriers(
        barrier_list,
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT),
        SET_FLAG_BIT(PipelineStage, VERTEX_SHADER_BIT) |
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT));
}

static std::unordered_map<std::string, std::shared_ptr<ShaderModule>> s_shader_module_list;
std::shared_ptr<ShaderModule> loadShaderModule(
    const std::shared_ptr<renderer::Device>& device,
    const std::string& shader_name,
    const ShaderStageFlagBits& shader_stage) {
    auto path_file_name = std::string("lib/shaders/") + shader_name;
    auto search_result = s_shader_module_list.find(path_file_name);
    std::shared_ptr<ShaderModule> result;
    if (search_result == s_shader_module_list.end()) {
        uint64_t shader_code_size;
        auto shader_code = engine::helper::readFile(path_file_name.c_str(), shader_code_size);
        auto shader_module = device->createShaderModule(shader_code_size, shader_code.data(), shader_stage);
        s_shader_module_list[path_file_name] = shader_module;
        result = shader_module;
    }
    else {
        result = search_result->second;
    }

    return result;
}

void clearCachedShaderModules(const std::shared_ptr<renderer::Device>& device) {
    for (auto& shader_module : s_shader_module_list) {
        device->destroyShaderModule(shader_module.second);
    }
    s_shader_module_list.clear();
}

std::shared_ptr<renderer::PipelineLayout> createComputePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts,
    const uint32_t& push_const_range_size) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = push_const_range_size;

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

std::shared_ptr<renderer::Pipeline> createComputePipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const std::string& compute_shader_name) {
    auto shader_module =
        renderer::helper::loadShaderModule(
            device,
            compute_shader_name,
            renderer::ShaderStageFlagBits::COMPUTE_BIT);

    auto pipeline = device->createPipeline(
        pipeline_layout,
        shader_module);

    return pipeline;
}

void releasePipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::PipelineLayout>& pipeline_layout) {
    if (pipeline_layout != nullptr) {
        device->destroyPipelineLayout(pipeline_layout);
        pipeline_layout = nullptr;
    }
}

void releasePipeline(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::Pipeline>& pipeline) {
    if (pipeline != nullptr) {
        device->destroyPipeline(pipeline);
        pipeline = nullptr;
    }
}

} // namespace helper
} // namespace renderer
} // namespace engine
