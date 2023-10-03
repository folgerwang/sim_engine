#include <iostream>

#include "../renderer.h"
#include "vk_device.h"
#include "vk_command_buffer.h"
#include "vk_renderer_helper.h"

namespace engine {
namespace renderer {
namespace vk {
const char* VkResultToString(
    VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        // ... add other VkResult values as needed
    default: return "UNKNOWN_ERROR";
    }
}

VulkanDevice::VulkanDevice(
    const std::shared_ptr<PhysicalDevice>& physical_device,
    const QueueFamilyList& queue_list,
    const VkDevice& device,
    uint32_t queue_family_index)
    : physical_device_(physical_device), device_(device)
{
    transient_cmd_pool_ =
        createCommandPool(queue_family_index,
            static_cast<uint32_t>(CommandPoolCreateFlagBits::TRANSIENT_BIT) |
            static_cast<uint32_t>(CommandPoolCreateFlagBits::RESET_COMMAND_BUFFER_BIT));

    transient_cmd_buffer_ =
        allocateCommandBuffers(
            transient_cmd_pool_,
            1,
            true)[0];

    auto compute_queue_index =
        queue_list.getQueueFamilyIndex(
            QueueFlagBits::COMPUTE_BIT);

    auto transit_queue_index =
        queue_list.getQueueInfo(compute_queue_index).queue_count_ - 1;

    transient_compute_queue_ =
        getDeviceQueue(
            compute_queue_index,
            transit_queue_index);

    transient_fence_ = createFence();
}

VulkanDevice::~VulkanDevice()
{
    assert(buffer_list_.size() == 0);
    assert(image_list_.size() == 0);
    assert(image_view_list_.size() == 0);
    assert(shader_list_.size() == 0);
    assert(sampler_list_.size() == 0);
    assert(framebuffer_list_.size() == 0);
    assert(pipeline_layout_list_.size() == 0);
    assert(pipeline_list_.size() == 0);
    assert(render_pass_list_.size() == 0);
    assert(semaphore_list_.size() == 0);
    assert(fence_list_.size() == 0);
}

std::shared_ptr<CommandBuffer> VulkanDevice::setupTransientCommandBuffer() {
    transient_cmd_buffer_->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));
    return transient_cmd_buffer_;
}

void VulkanDevice::submitAndWaitTransientCommandBuffer() {
    transient_cmd_buffer_->endCommandBuffer();
    transient_compute_queue_->submit(
        { transient_cmd_buffer_ },
        transient_fence_);
    waitForFences({ transient_fence_ });
    resetFences({ transient_fence_ });
    transient_cmd_buffer_->reset(0);
}

std::shared_ptr<Buffer> VulkanDevice::createBuffer(
    uint64_t buf_size,
    BufferUsageFlags usage,
    bool sharing/* = false*/) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buf_size;
    buffer_info.usage = helper::toVkBufferUsageFlags(usage);
    buffer_info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    auto result =
        vkCreateBuffer(
            device_,
            &buffer_info,
            nullptr,
            &buffer);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create buffer! : ") +
            VkResultToString(result));
    }

    auto vk_buffer =
        std::make_shared<VulkanBuffer>(
            buffer,
            static_cast<uint32_t>(buf_size));
    buffer_list_.push_back(vk_buffer);

    return vk_buffer;
}

void VulkanDevice::createBuffer(
    const uint64_t& buffer_size,
    const BufferUsageFlags& usage,
    const MemoryPropertyFlags& properties,
    const MemoryAllocateFlags& allocate_flags,
    std::shared_ptr<Buffer>& buffer,
    std::shared_ptr<DeviceMemory>& buffer_memory) {
    buffer = createBuffer(buffer_size, usage);
    auto mem_requirements = getBufferMemoryRequirements(buffer);
    buffer_memory = allocateMemory(mem_requirements.size,
        mem_requirements.memory_type_bits,
        properties,
        allocate_flags);
    bindBufferMemory(buffer, buffer_memory);

    if ((usage & static_cast<uint32_t>(BufferUsageFlagBits::SHADER_DEVICE_ADDRESS_BIT)) != 0) {
        VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
        auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);
        bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAddressInfo.buffer = vk_buffer->get();
        vk_buffer->set_device_address(vkGetBufferDeviceAddressKHR(device_, &bufferDeviceAddressInfo));
    }
}

std::shared_ptr<Image> VulkanDevice::createImage(
    ImageType image_type,
    glm::uvec3 image_size,
    Format format,
    ImageUsageFlags usage,
    ImageTiling tiling,
    ImageLayout layout,
    ImageCreateFlags flags/* = 0*/,
    bool sharing/* = false*/,
    uint32_t num_samples/* = 1*/,
    uint32_t num_mips/* = 1*/,
    uint32_t num_layers/* = 1*/) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = helper::toVkImageType(image_type);
    image_info.extent.width = image_size.x;
    image_info.extent.height = image_size.y;
    image_info.extent.depth = image_size.z;
    image_info.mipLevels = num_mips;
    image_info.arrayLayers = num_layers;
    image_info.format = helper::toVkFormat(format);
    image_info.tiling = helper::toVkImageTiling(tiling);
    image_info.initialLayout = helper::toVkImageLayout(layout);
    image_info.usage = helper::toVkImageUsageFlags(usage);
    image_info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = static_cast<VkSampleCountFlagBits>(num_samples);
    image_info.flags = helper::toVkImageCreateFlags(flags);

    VkImage image;
    auto result =
        vkCreateImage(
            device_,
            &image_info,
            nullptr,
            &image);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create image! : ") +
            VkResultToString(result));
    }

    auto vk_image =
        std::make_shared<VulkanImage>(image);
    vk_image->setImageLayout(layout);
    image_list_.push_back(vk_image);

    return vk_image;
}

std::shared_ptr<ImageView> VulkanDevice::createImageView(
    std::shared_ptr<Image> image,
    ImageViewType view_type,
    Format format,
    ImageAspectFlags aspect_flags,
    uint32_t base_mip/* = 0*/,
    uint32_t mip_count/* = 1*/,
    uint32_t base_layer/* = 0*/,
    uint32_t layer_count/* = 1*/) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = vk_image->get();
    view_info.viewType = helper::toVkImageViewType(view_type);
    view_info.format = helper::toVkFormat(format);
    view_info.subresourceRange.aspectMask = helper::toVkImageAspectFlags(aspect_flags);
    view_info.subresourceRange.baseMipLevel = base_mip;
    view_info.subresourceRange.levelCount = mip_count;
    view_info.subresourceRange.baseArrayLayer = base_layer;
    view_info.subresourceRange.layerCount = layer_count;

    VkImageView image_view;
    auto result =
        vkCreateImageView(
            device_,
            &view_info,
            nullptr,
            &image_view);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create texture image view! : ") +
            VkResultToString(result));
    }

    auto vk_image_view =
        std::make_shared<VulkanImageView>(image_view);
    image_view_list_.push_back(vk_image_view);

    return vk_image_view;
}

std::shared_ptr<Sampler> VulkanDevice::createSampler(Filter filter, SamplerAddressMode address_mode, SamplerMipmapMode mipmap_mode, float anisotropy) {
    auto vk_filter = helper::toVkFilter(filter);
    auto vk_address_mode = helper::toVkSamplerAddressMode(address_mode);
    auto vk_mipmap_mode = helper::toVkSamplerMipmapMode(mipmap_mode);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = vk_filter;
    sampler_info.minFilter = vk_filter;
    sampler_info.addressModeU = vk_address_mode;
    sampler_info.addressModeV = vk_address_mode;
    sampler_info.addressModeW = vk_address_mode;
    sampler_info.anisotropyEnable = anisotropy > 0 ? VK_TRUE : VK_FALSE;
    sampler_info.maxAnisotropy = anisotropy;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = vk_mipmap_mode;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 1024.0f;

    VkSampler tex_sampler;
    auto result =
        vkCreateSampler(
            device_,
            &sampler_info,
            nullptr,
            &tex_sampler);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create texture sampler! : ") +
            VkResultToString(result));
    }

    auto vk_tex_sampler =
        std::make_shared<VulkanSampler>(tex_sampler);
    sampler_list_.push_back(vk_tex_sampler);

    return vk_tex_sampler;
}

std::shared_ptr<Semaphore> VulkanDevice::createSemaphore() {
    VkTimelineSemaphoreSubmitInfo timeline_info = {};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.pNext = nullptr;
    timeline_info.waitSemaphoreValueCount = 0;
    timeline_info.pWaitSemaphoreValues = nullptr;
    timeline_info.signalSemaphoreValueCount = 0;// 1;
    timeline_info.pSignalSemaphoreValues = nullptr;// &timeline_value_;

    VkSemaphoreTypeCreateInfo type_create_info = {};
    type_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_create_info.semaphoreType = VK_SEMAPHORE_TYPE_BINARY;// VK_SEMAPHORE_TYPE_TIMELINE;
    type_create_info.initialValue = 0; // Initial value for the semaphore
    type_create_info.pNext = nullptr;// &timeline_info;

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &type_create_info;

    VkSemaphore semaphore;
    auto result =
        vkCreateSemaphore(
            device_,
            &semaphore_info,
            nullptr,
            &semaphore);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create semaphore! : ") +
            VkResultToString(result));
    }

    auto vk_semaphore =
        std::make_shared<VulkanSemaphore>(semaphore);
    semaphore_list_.push_back(vk_semaphore);

    return vk_semaphore;
}

std::shared_ptr<Fence> VulkanDevice::createFence(bool signaled/* = false*/) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }

    VkFence fence;
    auto result =
        vkCreateFence(
            device_,
            &fence_info,
            nullptr,
            &fence);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create fence! : ") +
            VkResultToString(result));
    }

    auto vk_fence =
        std::make_shared<VulkanFence>(fence);
    fence_list_.push_back(vk_fence);
    
    return vk_fence;
}

std::shared_ptr<ShaderModule>
VulkanDevice::createShaderModule(
    uint64_t size,
    void* data,
    ShaderStageFlagBits shader_stage) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = size;
    create_info.pCode = reinterpret_cast<const uint32_t*>(data);

    VkShaderModule shader_module;
    auto result =
        vkCreateShaderModule(
            device_,
            &create_info,
            nullptr,
            &shader_module);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create shader module! : ") +
            VkResultToString(result));
    }

    auto vk_shader_module =
        std::make_shared<VulkanShaderModule>(
            shader_module,
            shader_stage);
    shader_list_.push_back(vk_shader_module);

    return vk_shader_module;
}

std::shared_ptr<CommandPool> VulkanDevice::createCommandPool(
    uint32_t queue_family_index,
    CommandPoolCreateFlags flags) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index;
    pool_info.flags = helper::toVkCommandPoolCreateFlags(flags);

    VkCommandPool cmd_pool;
    auto result =
        vkCreateCommandPool(
            device_,
            &pool_info,
            nullptr,
            &cmd_pool);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create command pool! : ") +
            VkResultToString(result));
    }

    auto vk_cmd_pool =
        std::make_shared<VulkanCommandPool>();
    vk_cmd_pool->set(cmd_pool);
    return vk_cmd_pool;
}

std::shared_ptr<Queue> VulkanDevice::getDeviceQueue(
    uint32_t queue_family_index,
    uint32_t queue_index/* = 0*/) {
    VkQueue queue;
    vkGetDeviceQueue(device_, queue_family_index, queue_index, &queue);

    auto vk_queue =
        std::make_shared<VulkanQueue>();
    vk_queue->set(queue);
    return vk_queue;
}

std::shared_ptr<DescriptorSetLayout> VulkanDevice::createDescriptorSetLayout(
    const std::vector<DescriptorSetLayoutBinding>& bindings) {
    std::vector<VkDescriptorSetLayoutBinding> vk_bindings(bindings.size());
    for (auto i = 0; i < bindings.size(); i++) {
        const auto& binding = bindings[i];
        auto& vk_binding = vk_bindings[i];
        vk_binding.binding = binding.binding;
        vk_binding.descriptorType = helper::toVkDescriptorType(binding.descriptor_type);
        vk_binding.descriptorCount = binding.descriptor_count;
        vk_binding.stageFlags = helper::toVkShaderStageFlags(binding.stage_flags);
        auto vk_samplers = RENDER_TYPE_CAST(Sampler, binding.immutable_samplers);
        vk_binding.pImmutableSamplers = vk_samplers ? vk_samplers->getPtr() : nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(vk_bindings.size());
    layout_info.pBindings = vk_bindings.data();

    VkDescriptorSetLayout descriptor_set_layout;
    auto result =
        vkCreateDescriptorSetLayout(
            device_,
            &layout_info,
            nullptr,
            &descriptor_set_layout);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create descriptor set layout! : ") +
            VkResultToString(result));
    }

    auto vk_set_layout =
        std::make_shared<VulkanDescriptorSetLayout>();
    vk_set_layout->set(descriptor_set_layout);
    return vk_set_layout;
}

std::shared_ptr<RenderPass> VulkanDevice::createRenderPass(
    const std::vector<AttachmentDescription>& attachments,
    const std::vector<SubpassDescription>& subpasses,
    const std::vector<SubpassDependency>& dependencies) {

    std::vector<VkAttachmentDescription> vk_attachments(attachments.size());
    for (int i = 0; i < attachments.size(); i++) {
        vk_attachments[i] = helper::FillVkAttachmentDescription(attachments[i]);
    }

    std::vector<helper::SubpassAttachments> subpass_attachments(subpasses.size());
    for (uint32_t i = 0; i < subpasses.size(); i++) {
        subpass_attachments[i].input_attachments = helper::FillVkAttachmentReference(subpasses[i].input_attachments);
        subpass_attachments[i].color_attachments = helper::FillVkAttachmentReference(subpasses[i].color_attachments);
        subpass_attachments[i].resolve_attachments = helper::FillVkAttachmentReference(subpasses[i].resolve_attachments);
        subpass_attachments[i].depth_stencil_attachment = helper::FillVkAttachmentReference(subpasses[i].depth_stencil_attachment);
    }

    std::vector<VkSubpassDescription> vk_subpasses(subpasses.size());
    for (int i = 0; i < subpasses.size(); i++) {
        vk_subpasses[i] = helper::FillVkSubpassDescription(subpasses[i], subpass_attachments[i]);
    }

    std::vector<VkSubpassDependency> vk_dependencies(dependencies.size());
    for (int i = 0; i < dependencies.size(); i++) {
        vk_dependencies[i] = helper::FillVkSubpassDependency(dependencies[i]);
    }

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(vk_attachments.size());
    render_pass_info.pAttachments = vk_attachments.data();
    render_pass_info.subpassCount = static_cast<uint32_t>(vk_subpasses.size());
    render_pass_info.pSubpasses = vk_subpasses.data();
    render_pass_info.dependencyCount = static_cast<uint32_t>(vk_dependencies.size());
    render_pass_info.pDependencies = vk_dependencies.data();

    VkRenderPass render_pass;
    auto result =
        vkCreateRenderPass(
            device_,
            &render_pass_info,
            nullptr,
            &render_pass);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create render pass! : ") +
            VkResultToString(result));
    }

    auto vk_render_pass =
        std::make_shared<VulkanRenderPass>(render_pass);
    render_pass_list_.push_back(vk_render_pass);

    return vk_render_pass;
}

DescriptorSetList VulkanDevice::createDescriptorSets(
    std::shared_ptr<DescriptorPool> descriptor_pool,
    std::shared_ptr<DescriptorSetLayout> descriptor_set_layout,
    uint64_t buffer_count) {
    auto vk_descriptor_pool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool);
    auto vk_descriptor_set_layout = RENDER_TYPE_CAST(DescriptorSetLayout, descriptor_set_layout);
    std::vector<VkDescriptorSetLayout> layouts(buffer_count, vk_descriptor_set_layout->get());
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = vk_descriptor_pool->get();
    alloc_info.descriptorSetCount = static_cast<uint32_t>(buffer_count);
    alloc_info.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> vk_desc_sets;
    vk_desc_sets.resize(buffer_count);

    auto result =
        vkAllocateDescriptorSets(
            device_,
            &alloc_info,
            vk_desc_sets.data());

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to allocate descriptor sets! : ") +
            VkResultToString(result));
    }

    DescriptorSetList desc_sets(vk_desc_sets.size());
    for (uint32_t i = 0; i < buffer_count; i++) {
        auto vk_desc_set = std::make_shared<VulkanDescriptorSet>();
        vk_desc_set->set(vk_desc_sets[i]);
        desc_sets[i] = vk_desc_set;
    }

    return std::move(desc_sets);
}

std::shared_ptr<PipelineLayout> VulkanDevice::createPipelineLayout(
    const DescriptorSetLayoutList& desc_set_layouts,
    const std::vector<PushConstantRange>& push_const_ranges) {

    std::vector<VkPushConstantRange> vk_push_const_ranges;
    vk_push_const_ranges.reserve(push_const_ranges.size());
    for (auto& push_const_range : push_const_ranges) {
        VkPushConstantRange vk_push_const_range{};
        vk_push_const_range.stageFlags = helper::toVkShaderStageFlags(push_const_range.stage_flags);
        vk_push_const_range.offset = push_const_range.offset;
        vk_push_const_range.size = push_const_range.size;
        vk_push_const_ranges.push_back(vk_push_const_range);
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // todo.
    std::vector<VkDescriptorSetLayout> vk_layouts;
    vk_layouts.reserve(desc_set_layouts.size());
    for (auto& desc_set_layout : desc_set_layouts) {
        vk_layouts.push_back(RENDER_TYPE_CAST(DescriptorSetLayout, desc_set_layout)->get());
    }

    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(vk_layouts.size());
    pipeline_layout_info.pSetLayouts = vk_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = static_cast<uint32_t>(vk_push_const_ranges.size());
    pipeline_layout_info.pPushConstantRanges = vk_push_const_ranges.data();

    VkPipelineLayout pipeline_layout;
    auto result =
        vkCreatePipelineLayout(
            device_,
            &pipeline_layout_info,
            nullptr,
            &pipeline_layout);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create pipeline layout! : ") +
            VkResultToString(result));
    }

    auto vk_pipeline_layout =
        std::make_shared<VulkanPipelineLayout>(pipeline_layout);
    pipeline_layout_list_.push_back(vk_pipeline_layout);

    return vk_pipeline_layout;
}

std::shared_ptr<Pipeline> VulkanDevice::createPipeline(
    const std::shared_ptr<RenderPass>& render_pass,
    const std::shared_ptr<PipelineLayout>& pipeline_layout,
    const std::vector<VertexInputBindingDescription>& binding_descs,
    const std::vector<VertexInputAttributeDescription>& attribute_descs,
    const PipelineInputAssemblyStateCreateInfo& topology_info,
    const GraphicPipelineInfo& graphic_pipeline_info,
    const ShaderModuleList& shader_modules,
    const glm::uvec2& extent) {

    VkGraphicsPipelineCreateInfo pipeline_info{};

    auto viewport = helper::fillViewport(extent);
    auto scissor = helper::fillScissor(extent);

    auto vk_blend_attachments = helper::fillVkPipelineColorBlendAttachments(*graphic_pipeline_info.blend_state_info);
    auto vk_color_blending = helper::fillVkPipelineColorBlendStateCreateInfo(*graphic_pipeline_info.blend_state_info, vk_blend_attachments);
    auto vk_rasterizer = helper::fillVkPipelineRasterizationStateCreateInfo(*graphic_pipeline_info.rasterization_info);
    auto vk_multisampling = helper::fillVkPipelineMultisampleStateCreateInfo(*graphic_pipeline_info.ms_info);
    auto vk_depth_stencil = helper::fillVkPipelineDepthStencilStateCreateInfo(*graphic_pipeline_info.depth_stencil_info);

    auto viewport_state = helper::fillVkPipelineViewportStateCreateInfo(&viewport, &scissor);

    auto vk_binding_descs = helper::toVkVertexInputBindingDescription(binding_descs);
    auto vk_attribute_descs = helper::toVkVertexInputAttributeDescription(attribute_descs);
    auto vk_vertex_input_info = helper::fillVkPipelineVertexInputStateCreateInfo(vk_binding_descs, vk_attribute_descs);
    auto vk_input_assembly = helper::fillVkPipelineInputAssemblyStateCreateInfo(topology_info);
    auto shader_stages = helper::getShaderStages(shader_modules);

    auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    assert(vk_pipeline_layout);

    auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
    assert(vk_render_pass);

    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vk_vertex_input_info;
    pipeline_info.pInputAssemblyState = &vk_input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &vk_rasterizer;
    pipeline_info.pMultisampleState = &vk_multisampling;
    pipeline_info.pDepthStencilState = &vk_depth_stencil;
    pipeline_info.pColorBlendState = &vk_color_blending;
    //    pipeline_info.pDynamicState = nullptr; // Optional
    pipeline_info.layout = vk_pipeline_layout->get();
    pipeline_info.renderPass = vk_render_pass->get();
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipeline_info.basePipelineIndex = -1; // Optional

    VkPipeline graphics_pipeline;
    auto result =
        vkCreateGraphicsPipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &pipeline_info,
            nullptr,
            &graphics_pipeline);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create graphics pipeline! : ") +
            VkResultToString(result));
    }

    auto vk_pipeline =
        std::make_shared<VulkanPipeline>(graphics_pipeline);
    pipeline_list_.push_back(vk_pipeline);

    return vk_pipeline;
}

std::shared_ptr<Pipeline> VulkanDevice::createPipeline(
    const std::shared_ptr<PipelineLayout>& pipeline_layout,
    const std::shared_ptr<ShaderModule>& shader_module) {
    auto shader_stages = helper::getShaderStages({ shader_module });

    auto vk_compute_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    assert(vk_compute_pipeline_layout);

    // flags = 0, - e.g. disable optimization
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stages[0];
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;
    pipeline_info.layout = vk_compute_pipeline_layout->get();

    VkPipeline compute_pipeline;
    auto result =
        vkCreateComputePipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &pipeline_info,
            nullptr,
            &compute_pipeline);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create compute pipeline! : ") +
            VkResultToString(result));
    }

    auto vk_pipeline =
        std::make_shared<VulkanPipeline>(compute_pipeline);
    pipeline_list_.push_back(vk_pipeline);

    return vk_pipeline;
}

std::shared_ptr<Pipeline> VulkanDevice::createPipeline(
    const std::shared_ptr<PipelineLayout>& pipeline_layout, 
    const ShaderModuleList& src_shader_modules,
    const RtShaderGroupCreateInfoList& src_shader_groups,
    const uint32_t ray_recursion_depth /*= 1*/) {

    auto vk_rt_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    assert(vk_rt_pipeline_layout);

    auto shader_stages = helper::getShaderStages(src_shader_modules);
    auto shader_groups = helper::getShaderGroups(src_shader_groups);

    VkRayTracingPipelineCreateInfoKHR pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipeline_info.stageCount = static_cast<uint32_t>(shader_stages.size());
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.groupCount = static_cast<uint32_t>(shader_groups.size());
    pipeline_info.pGroups = shader_groups.data();
    pipeline_info.maxPipelineRayRecursionDepth = ray_recursion_depth;
    pipeline_info.layout = vk_rt_pipeline_layout->get();

    VkPipeline rt_pipeline;
    auto result =
        vkCreateRayTracingPipelinesKHR(
            device_,
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            1,
            &pipeline_info,
            nullptr,
            &rt_pipeline);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create ray tracing pipeline! : ") +
            VkResultToString(result));
    }

    auto vk_pipeline =
        std::make_shared<VulkanPipeline>(rt_pipeline);
    pipeline_list_.push_back(vk_pipeline);

    return vk_pipeline;
}

std::shared_ptr<Swapchain> VulkanDevice::createSwapchain(
    const std::shared_ptr<Surface>& surface,
    const uint32_t& image_count,
    const Format& format,
    const glm::uvec2& buf_size,
    const ColorSpace& color_space,
    const SurfaceTransformFlagBits& transform,
    const PresentMode& present_mode,
    const ImageUsageFlags& usage,
    const std::vector<uint32_t>& queue_index) {

    auto vk_surface = RENDER_TYPE_CAST(Surface, surface);

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = vk_surface->get();
    create_info.minImageCount = image_count;
    create_info.imageFormat = helper::toVkFormat(format);
    create_info.imageColorSpace = helper::toVkColorSpace(color_space);
    create_info.imageExtent = { buf_size.x, buf_size.y };
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = helper::toVkImageUsageFlags(usage);

    create_info.imageSharingMode = queue_index.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_index.size() <= 1 ? 0 : queue_index.size());
    create_info.pQueueFamilyIndices = queue_index.size() <= 1 ? nullptr : queue_index.data();

    create_info.preTransform = helper::toVkSurfaceTransformFlags(transform);
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = helper::toVkPresentMode(present_mode);
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE; //need to be handled when resize.

    VkSwapchainKHR swap_chain;
    auto result =
        vkCreateSwapchainKHR(
            device_,
            &create_info,
            nullptr,
            &swap_chain);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create swap chain! : ") +
            VkResultToString(result));
    }

    auto vk_swap_chain =
        std::make_shared<VulkanSwapchain>();
    vk_swap_chain->set(swap_chain);
    return vk_swap_chain;
}

std::shared_ptr<Framebuffer> VulkanDevice::createFrameBuffer(
    const std::shared_ptr<RenderPass>& render_pass,
    const std::vector<std::shared_ptr<ImageView>>& attachments,
    const glm::uvec2& extent) {

    std::vector<VkImageView> image_views(attachments.size());
    for (int i = 0; i < attachments.size(); i++) {
        auto vk_image_view = RENDER_TYPE_CAST(ImageView, attachments[i]);
        image_views[i] = vk_image_view->get();
    }

    auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = vk_render_pass->get();
    framebuffer_info.attachmentCount = static_cast<uint32_t>(image_views.size());
    framebuffer_info.pAttachments = image_views.data();
    framebuffer_info.width = extent.x;
    framebuffer_info.height = extent.y;
    framebuffer_info.layers = 1;

    VkFramebuffer frame_buffer;
    auto result =
        vkCreateFramebuffer(
            device_,
            &framebuffer_info,
            nullptr,
            &frame_buffer);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create framebuffer! : ") +
            VkResultToString(result));
    }

    auto vk_frame_buffer =
        std::make_shared<VulkanFramebuffer>(frame_buffer);
    framebuffer_list_.push_back(vk_frame_buffer);

    return vk_frame_buffer;
}

std::shared_ptr<DescriptorPool> VulkanDevice::createDescriptorPool() {
    VkDescriptorPoolSize pool_sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 16 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 512 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 256 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 256 }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 256 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool descriptor_pool;
    auto result =
        vkCreateDescriptorPool(
            device_,
            &pool_info,
            nullptr,
            &descriptor_pool);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to create descriptor pool! : ") +
            VkResultToString(result));
    }

    auto vk_descriptor_pool =
        std::make_shared<VulkanDescriptorPool>();
    vk_descriptor_pool->set(descriptor_pool);
    return vk_descriptor_pool;
}

void VulkanDevice::updateDescriptorSets(
    const WriteDescriptorList& write_descriptors) {
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    std::vector<std::shared_ptr<VkDescriptorImageInfo>> desc_images;
    std::vector<std::shared_ptr<VkDescriptorBufferInfo>> desc_buffers;
    std::vector<std::unique_ptr<VkAccelerationStructureKHR[]>> desc_ases;

    for (auto i = 0; i < write_descriptors.size(); i++) {
        const auto src_write_desc = write_descriptors[i];
        bool is_texture =
            src_write_desc->desc_type == DescriptorType::SAMPLER ||
            src_write_desc->desc_type == DescriptorType::COMBINED_IMAGE_SAMPLER ||
            src_write_desc->desc_type == DescriptorType::SAMPLED_IMAGE ||
            src_write_desc->desc_type == DescriptorType::STORAGE_IMAGE;
        bool is_buffer =
            src_write_desc->desc_type == DescriptorType::UNIFORM_TEXEL_BUFFER ||
            src_write_desc->desc_type == DescriptorType::STORAGE_TEXEL_BUFFER ||
            src_write_desc->desc_type == DescriptorType::UNIFORM_BUFFER ||
            src_write_desc->desc_type == DescriptorType::STORAGE_BUFFER ||
            src_write_desc->desc_type == DescriptorType::UNIFORM_BUFFER_DYNAMIC ||
            src_write_desc->desc_type == DescriptorType::STORAGE_BUFFER_DYNAMIC;
         
        auto vk_desc_set = RENDER_TYPE_CAST(DescriptorSet, src_write_desc->desc_set);
        if (is_texture) {
            const auto src_tex_desc = static_cast<const TextureDescriptor*>(src_write_desc.get());
            auto vk_texture = RENDER_TYPE_CAST(ImageView, src_tex_desc->texture);
            
            auto desc_image = std::make_shared<VkDescriptorImageInfo>();
            desc_images.push_back(desc_image);
            desc_image->imageLayout = helper::toVkImageLayout(src_tex_desc->image_layout);
            desc_image->imageView = vk_texture->get();
            if (src_tex_desc->sampler) {
                desc_image->sampler = RENDER_TYPE_CAST(Sampler, src_tex_desc->sampler)->get();
            }
            else {
                desc_image->sampler = nullptr;
            }

            descriptor_writes.push_back(
                helper::addDescriptWrite(
                    vk_desc_set->get(),
                    desc_image.get(),
                    src_tex_desc->binding,
                    helper::toVkDescriptorType(src_tex_desc->desc_type)));
        }
        else if (is_buffer) {
            const auto src_buf_desc = static_cast<const BufferDescriptor*>(src_write_desc.get());
            auto vk_buffer = RENDER_TYPE_CAST(Buffer, src_buf_desc->buffer);
            auto desc_buffer = std::make_shared<VkDescriptorBufferInfo>();
            desc_buffers.push_back(desc_buffer);
            desc_buffer->buffer = vk_buffer->get();
            desc_buffer->offset = src_buf_desc->offset;
            desc_buffer->range = src_buf_desc->range;

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = vk_desc_set->get();
            descriptor_write.dstBinding = src_buf_desc->binding;
            descriptor_write.dstArrayElement = 0;

            descriptor_write.descriptorType = helper::toVkDescriptorType(src_buf_desc->desc_type);
            descriptor_write.descriptorCount = 1;
            descriptor_write.pBufferInfo = desc_buffer.get();
            descriptor_writes.push_back(descriptor_write);
        }
        else if (src_write_desc->desc_type == DescriptorType::ACCELERATION_STRUCTURE_KHR) {
            const auto src_as_desc = static_cast<const AccelerationStructDescriptor*>(src_write_desc.get());
            auto desc_buffer = std::make_shared<VkDescriptorBufferInfo>();

            auto num_as = static_cast<uint32_t>(src_as_desc->acc_structs.size());
            auto as_list = std::make_unique<VkAccelerationStructureKHR[]>(num_as);
            for (size_t i = 0; i < num_as; i++) {
                as_list[i] = reinterpret_cast<VkAccelerationStructureKHR>(src_as_desc->acc_structs[i]);
            }
            desc_ases.push_back(std::move(as_list));
            VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{};
            desc_as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            desc_as_info.accelerationStructureCount = num_as;
            desc_as_info.pAccelerationStructures = desc_ases[desc_ases.size()-1].get();

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = vk_desc_set->get();
            descriptor_write.dstBinding = src_write_desc->binding;
            descriptor_write.pNext = &desc_as_info;

            descriptor_write.descriptorType =
                helper::toVkDescriptorType(src_write_desc->desc_type);
            descriptor_write.descriptorCount = 1;
            descriptor_writes.push_back(descriptor_write);
        }
        else {
            assert(0);
        }
    }

    vkUpdateDescriptorSets(device_,
        static_cast<uint32_t>(descriptor_writes.size()),
        descriptor_writes.data(),
        0,
        nullptr);
}

void VulkanDevice::updateBufferMemory(
    const std::shared_ptr<DeviceMemory>& memory,
    uint64_t size,
    const void* src_data,
    uint64_t offset/* = 0*/) {
    if (memory) {
        void* dst_data = mapMemory(memory, size, offset);
        assert(dst_data);
        memcpy(dst_data, src_data, size);
        unmapMemory(memory);
    }
}

void VulkanDevice::dumpBufferMemory(
    const std::shared_ptr<DeviceMemory>& memory,
    uint64_t size,
    void* dst_data,
    uint64_t offset/* = 0*/) {
    if (memory) {
        void* src_data = mapMemory(memory, size, offset);
        assert(src_data);
        memcpy(dst_data, src_data, size);
        unmapMemory(memory);
    }
}

std::vector<std::shared_ptr<Image>> VulkanDevice::getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) {
    auto vk_swap_chain = RENDER_TYPE_CAST(Swapchain, swap_chain);
    uint32_t image_count;
    std::vector<VkImage> swap_chain_images;
    vkGetSwapchainImagesKHR(device_, vk_swap_chain->get(), &image_count, nullptr);
    swap_chain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device_, vk_swap_chain->get(), &image_count, swap_chain_images.data());

    std::vector<std::shared_ptr<Image>> vk_swap_chain_images(swap_chain_images.size());
    for (int i = 0; i < swap_chain_images.size(); i++) {
        auto vk_image = std::make_shared<VulkanImage>(swap_chain_images[i]);
        vk_swap_chain_images[i] = vk_image;
    }
    return vk_swap_chain_images;
}

std::shared_ptr<DeviceMemory>
VulkanDevice::allocateMemory(
    const uint64_t& buf_size,
    const uint32_t& memory_type_bits,
    const MemoryPropertyFlags& properties,
    const MemoryAllocateFlags& allocate_flags) {
    VkMemoryAllocateFlagsInfo alloc_flags_info{};
    alloc_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    alloc_flags_info.flags = helper::toVkMemoryAllocateFlags(allocate_flags);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = buf_size;
    if (allocate_flags != 0) {
        alloc_info.pNext = &alloc_flags_info;
    }
    alloc_info.memoryTypeIndex =
        helper::findMemoryType(
            getPhysicalDevice(),
            memory_type_bits,
            properties);

    VkDeviceMemory memory;
    auto result =
        vkAllocateMemory(
            device_,
            &alloc_info,
            nullptr,
            &memory);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to allocate buffer memory! : ") +
            VkResultToString(result));
    }

    auto vk_device_memory =
        std::make_shared<VulkanDeviceMemory>();
    vk_device_memory->set(memory);

    return vk_device_memory;
}

MemoryRequirements VulkanDevice::getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) {
    auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);

    MemoryRequirements mem_requirements = {};
    if (vk_buffer) {
        VkMemoryRequirements vk_mem_requirements;
        vkGetBufferMemoryRequirements(device_, vk_buffer->get(), &vk_mem_requirements);
        mem_requirements.size = vk_mem_requirements.size;
        mem_requirements.alignment = vk_mem_requirements.alignment;
        mem_requirements.memory_type_bits = vk_mem_requirements.memoryTypeBits;
    }

    return mem_requirements;
}

MemoryRequirements VulkanDevice::getImageMemoryRequirements(std::shared_ptr<Image> image) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);

    MemoryRequirements mem_requirements = {};
    if (vk_image) {
        VkMemoryRequirements vk_mem_requirements;
        vkGetImageMemoryRequirements(device_, vk_image->get(), &vk_mem_requirements);
        mem_requirements.size = vk_mem_requirements.size;
        mem_requirements.alignment = vk_mem_requirements.alignment;
        mem_requirements.memory_type_bits = vk_mem_requirements.memoryTypeBits;
    }

    return mem_requirements;
}

void VulkanDevice::bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset/* = 0*/) {
    auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);
    auto vk_buffer_memory = RENDER_TYPE_CAST(DeviceMemory, buffer_memory);

    if (vk_buffer && vk_buffer_memory) {
        vkBindBufferMemory(device_, vk_buffer->get(), vk_buffer_memory->get(), offset);
    }
}

void VulkanDevice::bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset/* = 0*/) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);
    auto vk_image_memory = RENDER_TYPE_CAST(DeviceMemory, image_memory);

    if (vk_image && vk_image_memory) {
        vkBindImageMemory(device_, vk_image->get(), vk_image_memory->get(), offset);
    }
}

std::vector<std::shared_ptr<CommandBuffer>> VulkanDevice::allocateCommandBuffers(
    std::shared_ptr<CommandPool> cmd_pool, 
    uint32_t num_buffers, 
    bool is_primary/* = true*/) {
    auto vk_cmd_pool = RENDER_TYPE_CAST(CommandPool, cmd_pool);
    std::vector<VkCommandBuffer> cmd_bufs(num_buffers);

    if (vk_cmd_pool) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = vk_cmd_pool->get();
        alloc_info.level = is_primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        alloc_info.commandBufferCount = num_buffers;

        auto result =
            vkAllocateCommandBuffers(
                device_,
                &alloc_info,
                cmd_bufs.data());

        if (result != VK_SUCCESS) {
            throw std::runtime_error(
                std::string("failed to allocate command buffers! : ") +
                VkResultToString(result));
        }
    }

    std::vector<std::shared_ptr<CommandBuffer>> result(num_buffers);
    for (uint32_t i = 0; i < num_buffers; i++) {
        auto cmd_buf = std::make_shared<VulkanCommandBuffer>();
        cmd_buf->set(cmd_bufs[i]);
        result[i] = cmd_buf;
    }

    return result;
}

void* VulkanDevice::mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset /*=0*/) {
    void* data = nullptr;
    auto vk_memory = RENDER_TYPE_CAST(DeviceMemory, memory);
    if (vk_memory) {
        vkMapMemory(device_, vk_memory->get(), offset, size, 0/*reserved*/, &data);
    }

    return data;
}

void VulkanDevice::unmapMemory(std::shared_ptr<DeviceMemory> memory) {
    auto vk_memory = RENDER_TYPE_CAST(DeviceMemory, memory);
    if (vk_memory) {
        vkUnmapMemory(device_, vk_memory->get());
    }
}

void VulkanDevice::destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) {
    auto vk_cmd_pool = RENDER_TYPE_CAST(CommandPool, cmd_pool);
    if (vk_cmd_pool) {
        vkDestroyCommandPool(device_, vk_cmd_pool->get(), nullptr);
    }
}

void VulkanDevice::destroySwapchain(std::shared_ptr<Swapchain> swapchain) {
    auto vk_swapchain = RENDER_TYPE_CAST(Swapchain, swapchain);
    if (vk_swapchain) {
        vkDestroySwapchainKHR(device_, vk_swapchain->get(), nullptr);
    }
}

void VulkanDevice::destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) {
    auto vk_descriptor_pool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool);
    if (vk_descriptor_pool) {
        vkDestroyDescriptorPool(device_, vk_descriptor_pool->get(), nullptr);
    }
}

void VulkanDevice::destroyPipeline(std::shared_ptr<Pipeline> pipeline) {
    auto result = std::find(pipeline_list_.begin(), pipeline_list_.end(), pipeline);
    if (result != pipeline_list_.end()) {
        auto vk_pipeline = RENDER_TYPE_CAST(Pipeline, pipeline);
        if (vk_pipeline) {
            vkDestroyPipeline(device_, vk_pipeline->get(), nullptr);
        }
        pipeline_list_.erase(result);
    }
}

void VulkanDevice::destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) {
    auto result = std::find(pipeline_layout_list_.begin(), pipeline_layout_list_.end(), pipeline_layout);
    if (result != pipeline_layout_list_.end()) {
        auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
        if (vk_pipeline_layout) {
            vkDestroyPipelineLayout(device_, vk_pipeline_layout->get(), nullptr);
        }
        pipeline_layout_list_.erase(result);
    }
}

void VulkanDevice::destroyRenderPass(std::shared_ptr<RenderPass> render_pass) {
    auto result = std::find(render_pass_list_.begin(), render_pass_list_.end(), render_pass);
    if (result != render_pass_list_.end()) {
        auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
        if (vk_render_pass) {
            vkDestroyRenderPass(device_, vk_render_pass->get(), nullptr);
        }
        render_pass_list_.erase(result);
    }
}

void VulkanDevice::destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) {
    auto result = std::find(framebuffer_list_.begin(), framebuffer_list_.end(), frame_buffer);
    if (result != framebuffer_list_.end()) {
        auto vk_framebuffer = RENDER_TYPE_CAST(Framebuffer, frame_buffer);
        if (vk_framebuffer) {
            vkDestroyFramebuffer(device_, vk_framebuffer->get(), nullptr);
        }
        framebuffer_list_.erase(result);
    }
}

void VulkanDevice::destroyImageView(std::shared_ptr<ImageView> image_view) {
    auto result = std::find(image_view_list_.begin(), image_view_list_.end(), image_view);
    if (result != image_view_list_.end()) {
        auto vk_image_view = RENDER_TYPE_CAST(ImageView, image_view);
        if (vk_image_view) {
            vkDestroyImageView(device_, vk_image_view->get(), nullptr);
        }
        image_view_list_.erase(result);
    }
}

void VulkanDevice::destroySampler(std::shared_ptr<Sampler> sampler) {
    auto result = std::find(sampler_list_.begin(), sampler_list_.end(), sampler);
    if (result != sampler_list_.end()) {
        auto vk_sampler = RENDER_TYPE_CAST(Sampler, sampler);
        if (vk_sampler) {
            vkDestroySampler(device_, vk_sampler->get(), nullptr);
        }
        sampler_list_.erase(result);
    }
}

void VulkanDevice::destroyImage(std::shared_ptr<Image> image) {
    auto result = std::find(image_list_.begin(), image_list_.end(), image);
    if (result != image_list_.end()) {
        auto vk_image = RENDER_TYPE_CAST(Image, image);
        if (vk_image) {
            vkDestroyImage(device_, vk_image->get(), nullptr);
        }
        image_list_.erase(result);
    }
}

void VulkanDevice::destroyBuffer(std::shared_ptr<Buffer> buffer) {
    auto result = std::find(buffer_list_.begin(), buffer_list_.end(), buffer);
    if (result != buffer_list_.end()) {
        auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);
        if (vk_buffer) {
            vkDestroyBuffer(device_, vk_buffer->get(), nullptr);
        }
        buffer_list_.erase(result);
    }
}

void VulkanDevice::destroySemaphore(std::shared_ptr<Semaphore> semaphore) {
    auto result = std::find(semaphore_list_.begin(), semaphore_list_.end(), semaphore);
    if (result != semaphore_list_.end()) {
        auto vk_semaphore = RENDER_TYPE_CAST(Semaphore, semaphore);
        if (vk_semaphore) {
            vkDestroySemaphore(device_, vk_semaphore->get(), nullptr);
        }
        semaphore_list_.erase(result);
    }
}

void VulkanDevice::destroyFence(std::shared_ptr<Fence> fence) {
    auto result = std::find(fence_list_.begin(), fence_list_.end(), fence);
    if (result != fence_list_.end()) {
        auto vk_fence = RENDER_TYPE_CAST(Fence, fence);
        if (vk_fence) {
            vkDestroyFence(device_, vk_fence->get(), nullptr);
        }
        fence_list_.erase(result);
    }
}

void VulkanDevice::destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) {
    auto vk_layout = RENDER_TYPE_CAST(DescriptorSetLayout, layout);
    if (vk_layout) {
        vkDestroyDescriptorSetLayout(device_, vk_layout->get(), nullptr);
    }
}

void VulkanDevice::destroyShaderModule(
    std::shared_ptr<ShaderModule> shader_module) {
    auto result = std::find(shader_list_.begin(), shader_list_.end(), shader_module);
    if (result != shader_list_.end()) {
        auto vk_shader_module = RENDER_TYPE_CAST(ShaderModule, shader_module);
        if (vk_shader_module) {
            vkDestroyShaderModule(device_, vk_shader_module->get(), nullptr);
        }
        shader_list_.erase(result);
    }
}

void VulkanDevice::destroy() {
    freeCommandBuffers(
        transient_cmd_pool_,
        { transient_cmd_buffer_ });

    destroyCommandPool(
        transient_cmd_pool_);

    destroyFence(transient_fence_);

    vkDestroyDevice(device_, nullptr);
}

void VulkanDevice::freeMemory(std::shared_ptr<DeviceMemory> memory) {
    auto vk_memory = RENDER_TYPE_CAST(DeviceMemory, memory);
    if (vk_memory) {
        vkFreeMemory(device_, vk_memory->get(), nullptr);
    }
}

void VulkanDevice::freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) {
    auto vk_cmd_pool = RENDER_TYPE_CAST(CommandPool, cmd_pool);
    if (vk_cmd_pool) {
        std::vector<VkCommandBuffer> vk_cmd_bufs(cmd_bufs.size());
        for (uint32_t i = 0; i < cmd_bufs.size(); i++) {
            auto vk_cmd_buf = RENDER_TYPE_CAST(CommandBuffer, cmd_bufs[i]);
            vk_cmd_bufs[i] = vk_cmd_buf->get();
        }

        vkFreeCommandBuffers(device_, vk_cmd_pool->get(), static_cast<uint32_t>(vk_cmd_bufs.size()), vk_cmd_bufs.data());
    }
}

void VulkanDevice::resetFences(const std::vector<std::shared_ptr<Fence>>& fences) {
    std::vector<VkFence> vk_fences(fences.size());
    for (int i = 0; i < fences.size(); i++) {
        auto vk_fence = RENDER_TYPE_CAST(Fence, fences[i]);
        vk_fences[i] = vk_fence->get();
    }
    auto result =
        vkResetFences(
            device_,
            static_cast<uint32_t>(vk_fences.size()),
            vk_fences.data());

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("reset fences error : ") +
            VkResultToString(result));
    }
}

void VulkanDevice::waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) {
    std::vector<VkFence> vk_fences(fences.size());
    for (int i = 0; i < fences.size(); i++) {
        auto vk_fence = RENDER_TYPE_CAST(Fence, fences[i]);
        vk_fences[i] = vk_fence->get();
    }

    auto result =
        vkWaitForFences(
            device_,
            static_cast<uint32_t>(vk_fences.size()),
            vk_fences.data(),
            VK_TRUE,
            UINT64_MAX);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("wait for fence error : ") +
                VkResultToString(result));
    }
}

void VulkanDevice::waitForSemaphores(
    const std::vector<std::shared_ptr<Semaphore>>& semaphores,
    uint64_t value) {

    std::vector<VkSemaphore> vk_semaphores(semaphores.size());
    std::vector<uint64_t> values(semaphores.size());
    for (int i = 0; i < semaphores.size(); i++) {
        auto vk_semaphore = RENDER_TYPE_CAST(Semaphore, semaphores[i]);
        vk_semaphores[i] = vk_semaphore->get();
        values[i] = value;
    }

    VkSemaphoreWaitInfo wait_info = {};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = static_cast<uint32_t>(vk_semaphores.size()); // Number of semaphores you're waiting on
    wait_info.pSemaphores = vk_semaphores.data(); // The semaphores you're waiting on
    wait_info.pValues = values.data(); // An array of values to wait for

    auto result =
        vkWaitSemaphores(
            device_,
            &wait_info,
            UINT64_MAX);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("wait for semaphores error : ") +
            VkResultToString(result));
    }
}

void VulkanDevice::waitIdle() {
    auto result =
        vkDeviceWaitIdle(device_);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("wait for device idle error : ") +
            VkResultToString(result));
    }
}

void VulkanDevice::getAccelerationStructureBuildSizes(
    AccelerationStructureBuildType         as_build_type,
    const AccelerationStructureBuildGeometryInfo& build_info,
    AccelerationStructureBuildSizesInfo& size_info) {

    std::unique_ptr<VkAccelerationStructureGeometryKHR[]> geoms;
    auto as_build_geo_info = helper::toVkAccelerationStructureBuildGeometryInfo(build_info, geoms);

    std::vector<uint32_t> max_primitive_counts(build_info.geometries.size());
    for (auto i = 0; i < build_info.geometries.size(); i++) {
        max_primitive_counts[i] = build_info.geometries[i]->max_primitive_count;
    }

    VkAccelerationStructureBuildSizesInfoKHR as_build_size_info{};
    as_build_size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        device_,
        helper::toVkAccelerationStructureBuildType(as_build_type),
        &as_build_geo_info,
        max_primitive_counts.data(),
        &as_build_size_info);

    size_info.struct_type = StructureType::ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    size_info.as_size = as_build_size_info.accelerationStructureSize;
    size_info.update_scratch_size = as_build_size_info.updateScratchSize;
    size_info.build_scratch_size = as_build_size_info.buildScratchSize;
}

AccelerationStructure VulkanDevice::createAccelerationStructure(
    const std::shared_ptr<Buffer>& buffer,
    const AccelerationStructureType& as_type) {
    VkAccelerationStructureCreateInfoKHR as_create_info{};
    as_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_create_info.buffer = RENDER_TYPE_CAST(Buffer, buffer)->get();
    as_create_info.size = buffer->getSize();
    as_create_info.type = helper::toVkAccelerationStructureType(as_type);
    VkAccelerationStructureKHR as_handle;
    vkCreateAccelerationStructureKHR(device_, &as_create_info, nullptr, &as_handle);
    return reinterpret_cast<AccelerationStructure>(as_handle);
}

void VulkanDevice::destroyAccelerationStructure(
    const AccelerationStructure& as) {
    auto vk_as = reinterpret_cast<VkAccelerationStructureKHR>(as);
    vkDestroyAccelerationStructureKHR(device_, vk_as, nullptr);
}

DeviceAddress VulkanDevice::getAccelerationStructureDeviceAddress(const AccelerationStructure& as) {
    VkAccelerationStructureDeviceAddressInfoKHR as_device_info{};
    as_device_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    as_device_info.accelerationStructure = reinterpret_cast<VkAccelerationStructureKHR>(as);
    auto device_address = vkGetAccelerationStructureDeviceAddressKHR(device_, &as_device_info);

    DeviceAddress result = device_address;
    return result;
}

void VulkanDevice::getRayTracingShaderGroupHandles(
    const std::shared_ptr<Pipeline>& pipeline,
    const uint32_t group_count,
    const uint32_t sbt_size,
    void* shader_handle_storage) {
    auto vk_pipeline = RENDER_TYPE_CAST(Pipeline, pipeline);
    auto result =
        vkGetRayTracingShaderGroupHandlesKHR(
            device_,
            vk_pipeline->get(),
            0,
            group_count,
            sbt_size,
            shader_handle_storage);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to call ray tracing shader group handles! : ") +
            VkResultToString(result));
    }
}

} // vk
} // renderer
} // engine