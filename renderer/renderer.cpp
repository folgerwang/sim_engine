#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <array>

#include "renderer.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_command_buffer.h"
#include "vulkan/vk_renderer_helper.h"

namespace engine {
namespace renderer {

namespace {
static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}
}

namespace vk {
void VulkanInstance::destroySurface(const std::shared_ptr<Surface>& surface) {
    vkDestroySurfaceKHR(instance_, RENDER_TYPE_CAST(Surface, surface)->get(), nullptr);
}

void VulkanInstance::destroy() {
    if (helper::hasEnabledValidationLayers()) {
        helper::DestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
    }
    vkDestroyInstance(instance_, nullptr);
}

void VulkanQueue::submit(
    const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
    const std::shared_ptr<Fence>& in_flight_fence) {
    std::vector<VkCommandBuffer> vk_cmd_bufs(command_buffers.size());
    for (int i = 0; i < command_buffers.size(); i++) {
        auto vk_cmd_buf = RENDER_TYPE_CAST(CommandBuffer, command_buffers[i]);
        vk_cmd_bufs[i] = vk_cmd_buf->get();
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());
    submit_info.pCommandBuffers = vk_cmd_bufs.data();

    auto vk_in_flight_fence = RENDER_TYPE_CAST(Fence, in_flight_fence);
    auto result =
        vkQueueSubmit(
            queue_,
            1,
            &submit_info,
            vk_in_flight_fence ? vk_in_flight_fence->get() : VK_NULL_HANDLE);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to submit command queue! : ") +
            VkResultToString(result));
    }
}

void VulkanQueue::waitIdle() {
    vkQueueWaitIdle(queue_);
}
} // namespace vk

ImageResourceInfo Helper::image_source_info_;
ImageResourceInfo Helper::image_as_color_attachement_;
ImageResourceInfo Helper::image_as_store_;
ImageResourceInfo Helper::image_as_shader_sampler_;
TextureInfo Helper::white_tex_;
TextureInfo Helper::black_tex_;
TextureInfo Helper::permutation_tex_;
TextureInfo Helper::permutation_2d_tex_;
TextureInfo Helper::grad_tex_;
TextureInfo Helper::perm_grad_tex_;
TextureInfo Helper::perm_grad_4d_tex_;
TextureInfo Helper::grad_4d_tex_;

void Helper::init(const std::shared_ptr<Device>& device) {
    vk::helper::create2x2Texture(device, 0xffffffff, white_tex_);
    vk::helper::create2x2Texture(device, 0xff000000, black_tex_);
    vk::helper::createPermutationTexture(device, permutation_tex_);
    vk::helper::createPermutation2DTexture(device, permutation_2d_tex_);
    vk::helper::createGradTexture(device, grad_tex_);
    vk::helper::createPermGradTexture(device, perm_grad_tex_);
    vk::helper::createPermGrad4DTexture(device, perm_grad_4d_tex_);
    vk::helper::createGrad4DTexture(device, grad_4d_tex_);

    image_source_info_ = {
        ImageLayout::UNDEFINED,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

    image_as_color_attachement_ = {
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

    image_as_store_ = {
        ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) |
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };

    image_as_shader_sampler_ = {
        ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) |
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
}

void Helper::destroy(const std::shared_ptr<Device>& device) {
    black_tex_.destroy(device);
    white_tex_.destroy(device);
    permutation_tex_.destroy(device);
    permutation_2d_tex_.destroy(device);
    grad_tex_.destroy(device);
    perm_grad_tex_.destroy(device);
    perm_grad_4d_tex_.destroy(device);
    grad_4d_tex_.destroy(device);
}

std::shared_ptr<Instance> Helper::createInstance() {
    return vk::helper::createInstance();
}

std::shared_ptr<Surface> Helper::createSurface(
    const std::shared_ptr<Instance>& instance,
    GLFWwindow* window) {
    return vk::helper::createSurface(instance, window);
}

PhysicalDeviceList Helper::collectPhysicalDevices(
    const std::shared_ptr<Instance>& instance) {
    return vk::helper::collectPhysicalDevices(instance);
}

std::shared_ptr<PhysicalDevice> Helper::pickPhysicalDevice(
    const PhysicalDeviceList& physical_devices,
    const std::shared_ptr<Surface>& surface) {
    return vk::helper::pickPhysicalDevice(physical_devices, surface);
}

QueueFamilyList Helper::findQueueFamilies(
    const std::shared_ptr<PhysicalDevice>& physical_device,
    const std::shared_ptr<Surface>& surface) {
    return vk::helper::findQueueFamilies(physical_device, surface);
}

std::shared_ptr<Device> Helper::createLogicalDevice(
    const std::shared_ptr<PhysicalDevice>& physical_device,
    const std::shared_ptr<Surface>& surface,
    const QueueFamilyList& list) {
    return vk::helper::createLogicalDevice(physical_device, surface, list);
}

void Helper::initRayTracingProperties(
    const std::shared_ptr<renderer::PhysicalDevice>& physical_device,
    const std::shared_ptr<renderer::Device>& device,
    PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
    PhysicalDeviceAccelerationStructureFeatures& as_features) {
    return vk::helper::initRayTracingProperties(physical_device, device, rt_pipeline_properties, as_features);
}

void Helper::createSwapChain(
    GLFWwindow* window,
    const std::shared_ptr<Device>& device,
    const std::shared_ptr<Surface>& surface,
    const QueueFamilyList& list,
    SwapChainInfo& swap_chain_info,
    const ImageUsageFlags& usage) {
    const auto& vk_device = RENDER_TYPE_CAST(Device, device);
    const auto& vk_physical_device = vk_device->getPhysicalDevice();
    vk::helper::SwapChainSupportDetails swap_chain_support = vk::helper::querySwapChainSupport(vk_physical_device, surface);

    VkSurfaceFormatKHR surface_format = vk::helper::chooseSwapSurfaceFormat(swap_chain_support.formats_);
    auto present_mode = vk::helper::chooseSwapPresentMode(swap_chain_support.present_modes_);
    VkExtent2D extent = vk::helper::chooseSwapExtent(window, swap_chain_support.capabilities_);

    uint32_t image_count = swap_chain_support.capabilities_.minImageCount + 1;
    if (swap_chain_support.capabilities_.maxImageCount > 0 && image_count > swap_chain_support.capabilities_.maxImageCount) {
        image_count = swap_chain_support.capabilities_.maxImageCount;
    }

    auto queue_family_index = list.getGraphicAndPresentFamilyIndex();

    swap_chain_info.format = vk::helper::fromVkFormat(surface_format.format);
    swap_chain_info.extent = glm::uvec2(extent.width, extent.height);

    swap_chain_info.swap_chain = device->createSwapchain(surface,
        image_count,
        swap_chain_info.format,
        swap_chain_info.extent,
        vk::helper::fromVkColorSpace(surface_format.colorSpace),
        vk::helper::fromVkSurfaceTransformFlags(swap_chain_support.capabilities_.currentTransform),
        present_mode,
        usage,
        queue_family_index);

    swap_chain_info.images = device->getSwapchainImages(swap_chain_info.swap_chain);
}

Format Helper::findDepthFormat(const std::shared_ptr<Device>& device) {
    return vk::helper::findDepthFormat(device);
}

void Helper::addOneTexture(
    WriteDescriptorList& descriptor_writes,
    const std::shared_ptr<DescriptorSet>& desc_set,
    const DescriptorType& desc_type,
    uint32_t binding,
    const std::shared_ptr<Sampler>& sampler,
    const std::shared_ptr<ImageView>& texture,
    ImageLayout image_layout) {

    auto tex_desc = std::make_shared<TextureDescriptor>();
    tex_desc->binding = binding;
    tex_desc->desc_set = desc_set;
    tex_desc->desc_type = desc_type;
    tex_desc->image_layout = image_layout;
    tex_desc->sampler = sampler;
    tex_desc->texture = texture;

    descriptor_writes.push_back(tex_desc);
}

void Helper::addOneBuffer(
    WriteDescriptorList& descriptor_writes,
    const std::shared_ptr<DescriptorSet>& desc_set,
    const DescriptorType& desc_type,
    uint32_t binding,
    const std::shared_ptr<Buffer>& buffer,
    uint32_t range,
    uint32_t offset/* = 0*/) {

    auto buffer_desc = std::make_shared<BufferDescriptor>();
    buffer_desc->binding = binding;
    buffer_desc->desc_set = desc_set;
    buffer_desc->desc_type = desc_type;
    buffer_desc->offset = offset;
    buffer_desc->range = range;
    buffer_desc->buffer = buffer;

    descriptor_writes.push_back(buffer_desc);
}

void Helper::addOneAccelerationStructure(
    WriteDescriptorList& descriptor_writes,
    const std::shared_ptr<DescriptorSet>& desc_set,
    const DescriptorType& desc_type,
    uint32_t binding,
    const std::vector<AccelerationStructure>& acceleration_structs) {

    auto as_desc = std::make_shared<AccelerationStructDescriptor>();
    as_desc->binding = binding;
    as_desc->desc_set = desc_set;
    as_desc->desc_type = desc_type;
    as_desc->acc_structs = acceleration_structs;

    descriptor_writes.push_back(as_desc);
}

void Helper::createBuffer(
    const std::shared_ptr<Device>& device,
    const BufferUsageFlags& usage,
    const MemoryPropertyFlags& memory_property,
    const MemoryAllocateFlags& allocate_flags,
    std::shared_ptr<Buffer>& buffer,
    std::shared_ptr<DeviceMemory>& buffer_memory,
    const uint64_t buffer_size /*= 0*/,
    const void* src_data/*= nullptr*/) {

    bool has_src_data = 
        buffer_size > 0 &&
        src_data;
    bool need_stage_buffer =
        has_src_data &&
        (memory_property == SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));

    device->createBuffer(
        buffer_size,
        (need_stage_buffer ? SET_FLAG_BIT(BufferUsage, TRANSFER_DST_BIT) : 0) | usage,
        memory_property,
        allocate_flags,
        buffer,
        buffer_memory);

    if (has_src_data) {
        if (need_stage_buffer) {
            std::shared_ptr<Buffer> staging_buffer;
            std::shared_ptr<DeviceMemory> staging_buffer_memory;
            device->createBuffer(
                buffer_size,
                SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
                SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
                SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
                0,
                staging_buffer,
                staging_buffer_memory);

            device->updateBufferMemory(staging_buffer_memory, buffer_size, src_data);

            vk::helper::copyBuffer(device, staging_buffer, buffer, buffer_size);

            device->destroyBuffer(staging_buffer);
            device->freeMemory(staging_buffer_memory);
        }
        else {
            device->updateBufferMemory(buffer_memory, buffer_size, src_data);
        }
    }
}

void Helper::updateBufferWithSrcData(
    const std::shared_ptr<Device>& device,
    const uint64_t& buffer_size,
    const void* src_data,
    const std::shared_ptr<Buffer>& buffer) {

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    device->createBuffer(
        buffer_size,
        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        staging_buffer,
        staging_buffer_memory);

    device->updateBufferMemory(staging_buffer_memory, buffer_size, src_data);

    vk::helper::copyBuffer(device, staging_buffer, buffer, buffer_size);

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void Helper::updateBufferWithSrcData(
    const std::shared_ptr<Device>& device,
    const uint64_t& buffer_size,
    const void* src_data,
    const std::shared_ptr<DeviceMemory>& buffer_memory) {
    device->updateBufferMemory(buffer_memory, buffer_size, src_data);
}

void Helper::generateMipmapLevels(
    const std::shared_ptr<CommandBuffer>& cmd_buf,
    const std::shared_ptr<Image>& image,
    uint32_t mip_count,
    uint32_t width,
    uint32_t height,
    const ImageLayout& cur_image_layout) {
    vk::helper::generateMipmapLevels(cmd_buf, image, mip_count, width, height, cur_image_layout);
}

void Helper::create2DTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    Format format,
    int tex_width,
    int tex_height,
    int tex_channels,
    const void* pixels,
    std::shared_ptr<Image>& texture_image,
    std::shared_ptr<DeviceMemory>& texture_image_memory) {

    VkDeviceSize image_size =
        static_cast<VkDeviceSize>(tex_width * tex_height * (format == Format::R16_UNORM ? 2 : 4));

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    device->createBuffer(
        image_size,
        SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        staging_buffer,
        staging_buffer_memory);

    device->updateBufferMemory(
        staging_buffer_memory,
        image_size,
        pixels);

    vk::helper::createTextureImage(
        device,
        glm::vec3(tex_width, tex_height, 1),
        format,
        ImageTiling::OPTIMAL,
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT) |
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        texture_image,
        texture_image_memory);

    auto cmd_buf = device->setupTransientCommandBuffer();
    vk::helper::transitionImageLayout(
        cmd_buf,
        texture_image,
        format,
        ImageLayout::UNDEFINED,
        ImageLayout::TRANSFER_DST_OPTIMAL);
    vk::helper::copyBufferToImage(
        cmd_buf,
        staging_buffer,
        texture_image,
        glm::uvec3(tex_width, tex_height, 1));
    vk::helper::transitionImageLayout(
        cmd_buf,
        texture_image,
        format,
        ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    device->submitAndWaitTransientCommandBuffer();

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void Helper::create2DTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    Format format,
    const glm::uvec2& size,
    TextureInfo& texture_2d,
    const renderer::ImageUsageFlags& usage,
    const renderer::ImageLayout& image_layout,
    const renderer::ImageTiling image_tiling,
    const uint32_t memory_property) {
    auto is_depth = vk::helper::isDepthFormat(format);
    vk::helper::createTextureImage(
        device,
        glm::uvec3(size, 1),
        format,
        image_tiling,
        usage,
        memory_property,
        texture_2d.image,
        texture_2d.memory);

    texture_2d.view =
        device->createImageView(
            texture_2d.image,
            ImageViewType::VIEW_2D,
            format,
            is_depth ?
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT) :
                SET_FLAG_BIT(ImageAspect, COLOR_BIT));

    vk::helper::transitionImageLayout(
        device,
        texture_2d.image,
        format,
        ImageLayout::UNDEFINED,
        image_layout);

    texture_2d.size = glm::uvec3(size, 1);
}

void Helper::dumpTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<Image>& src_texture_image,
    Format format,
    const glm::uvec3& image_size,
    const uint32_t& bytes_per_pixel,
    void* pixels) {

    VkDeviceSize buffer_size = static_cast<VkDeviceSize>(
        image_size.x * image_size.y * image_size.z * bytes_per_pixel);

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    device->createBuffer(
        buffer_size,
        SET_FLAG_BIT(BufferUsage, TRANSFER_DST_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        staging_buffer,
        staging_buffer_memory);

    auto cmd_buf = device->setupTransientCommandBuffer();
    vk::helper::transitionImageLayout(
        cmd_buf,
        src_texture_image,
        format,
        ImageLayout::UNDEFINED,
        ImageLayout::TRANSFER_SRC_OPTIMAL);
    vk::helper::copyImageToBuffer(
        cmd_buf,
        src_texture_image,
        staging_buffer,
        image_size);
    vk::helper::transitionImageLayout(
        cmd_buf,
        src_texture_image,
        format,
        ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    device->submitAndWaitTransientCommandBuffer();

    device->dumpBufferMemory(staging_buffer_memory, buffer_size, pixels);

    device->destroyBuffer(staging_buffer);
    device->freeMemory(staging_buffer_memory);
}

void Helper::create3DTextureImage(
    const std::shared_ptr<renderer::Device>& device,
    Format format,
    const glm::uvec3& size,
    TextureInfo& texture_3d,
    const renderer::ImageUsageFlags& usage,
    const renderer::ImageLayout& image_layout,
    const renderer::ImageTiling image_tiling,
    const uint32_t memory_property) {
    auto is_depth = vk::helper::isDepthFormat(format);
    vk::helper::createTextureImage(
        device,
        size,
        format,
        image_tiling,
        usage,
        memory_property,
        texture_3d.image,
        texture_3d.memory);

    texture_3d.view =
        device->createImageView(
            texture_3d.image,
            ImageViewType::VIEW_3D,
            format,
            is_depth ?
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT) :
            SET_FLAG_BIT(ImageAspect, COLOR_BIT));

    vk::helper::transitionImageLayout(
        device,
        texture_3d.image,
        format,
        ImageLayout::UNDEFINED,
        image_layout);
}

void Helper::createDepthResources(
    const std::shared_ptr<renderer::Device>& device,
    Format format,
    glm::uvec2 size,
    TextureInfo& texture_2d) {

    create2DTextureImage(
        device,
        format,
        size,
        texture_2d,
        SET_FLAG_BIT(ImageUsage, DEPTH_STENCIL_ATTACHMENT_BIT) |
        SET_FLAG_BIT(ImageUsage, TRANSFER_SRC_BIT),
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void Helper::createCubemapTexture(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<RenderPass>& render_pass,
    uint32_t width,
    uint32_t height,
    uint32_t mip_count,
    Format format,
    const std::vector<BufferImageCopyInfo>& copy_regions,
    TextureInfo& texture,
    uint64_t buffer_size /*= 0*/,
    void* data /*= nullptr*/) {
    bool use_as_framebuffer = data == nullptr;
    VkDeviceSize image_size = static_cast<VkDeviceSize>(buffer_size);

    std::shared_ptr<Buffer> staging_buffer;
    std::shared_ptr<DeviceMemory> staging_buffer_memory;
    if (data) {
        device->createBuffer(
            image_size,
            SET_FLAG_BIT(BufferUsage, TRANSFER_SRC_BIT),
            SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
            SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
            0,
            staging_buffer,
            staging_buffer_memory);

        device->updateBufferMemory(staging_buffer_memory, buffer_size, data);
    }

    auto image_usage_flags =
        SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT) |
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT);

    if (use_as_framebuffer) {
        image_usage_flags |=
            SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT) |
            SET_FLAG_BIT(ImageUsage, TRANSFER_SRC_BIT);
    }

    texture.image = device->createImage(
        ImageType::TYPE_2D,
        glm::uvec3(width, height, 1),
        format,
        image_usage_flags,
        ImageTiling::OPTIMAL,
        ImageLayout::UNDEFINED,
        SET_FLAG_BIT(ImageCreate, CUBE_COMPATIBLE_BIT),
        false,
        1,
        mip_count,
        6u);

    auto mem_requirements = device->getImageMemoryRequirements(texture.image);
    texture.memory = device->allocateMemory(
        mem_requirements.size,
        mem_requirements.memory_type_bits,
        vk::helper::toVkMemoryPropertyFlags(SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT)),
        0);
    device->bindImageMemory(texture.image, texture.memory);

    if (data) {
        auto cmd_buf = device->setupTransientCommandBuffer();
        vk::helper::transitionImageLayout(
            cmd_buf,
            texture.image,
            format,
            ImageLayout::UNDEFINED,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            0,
            mip_count,
            0,
            6);
        vk::helper::copyBufferToImageWithMips(
            cmd_buf,
            staging_buffer,
            texture.image,
            copy_regions);
        vk::helper::transitionImageLayout(
            cmd_buf,
            texture.image,
            format,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0,
            mip_count,
            0,
            6);
        device->submitAndWaitTransientCommandBuffer();
    }

    texture.view = device->createImageView(
        texture.image,
        ImageViewType::VIEW_CUBE,
        format,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        0,
        mip_count,
        0,
        6);

    assert(render_pass);

    if (use_as_framebuffer) {
        texture.surface_views.resize(mip_count);
        texture.framebuffers.resize(mip_count);
        auto w = width;
        auto h = height;

        for (uint32_t i = 0; i < mip_count; ++i)
        {
            texture.surface_views[i].resize(6, VK_NULL_HANDLE); //sides of the cube

            for (uint32_t j = 0; j < 6; j++)
            {
                texture.surface_views[i][j] =
                    device->createImageView(
                        texture.image,
                        ImageViewType::VIEW_2D,
                        format,
                        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                        i,
                        1,
                        j,
                        1);
            }

            texture.framebuffers[i] = device->createFrameBuffer(render_pass, texture.surface_views[i], glm::uvec2(w, h));
            w = std::max(w >> 1, 1u);
            h = std::max(h >> 1, 1u);
        }
    }

    texture.size = glm::uvec3(width, height, 1);

    if (data) {
        device->destroyBuffer(staging_buffer);
        device->freeMemory(staging_buffer_memory);
    }
}

void Helper::blitImage(
    const std::shared_ptr<CommandBuffer>& cmd_buf,
    const std::shared_ptr<Image>& src_image,
    const std::shared_ptr<Image>& dst_image,
    const ImageResourceInfo& src_old_info,
    const ImageResourceInfo& src_new_info,
    const ImageResourceInfo& dst_old_info,
    const ImageResourceInfo& dst_new_info,
    const ImageAspectFlags& src_aspect_flags,
    const ImageAspectFlags& dst_aspect_flags,
    const glm::ivec3& buffer_size) {
    BarrierList barrier_list;
    barrier_list.image_barriers.reserve(2);

    auto src_aspect_mask = vk::helper::toVkImageAspectFlags(src_aspect_flags);
    auto dst_aspect_mask = vk::helper::toVkImageAspectFlags(dst_aspect_flags);

    ImageMemoryBarrier barrier;
    barrier.image = src_image;
    barrier.old_layout = src_old_info.image_layout;
    barrier.new_layout = ImageLayout::TRANSFER_SRC_OPTIMAL;
    barrier.src_access_mask = src_old_info.access_flags;
    barrier.dst_access_mask = SET_FLAG_BIT(Access, TRANSFER_READ_BIT);
    barrier.subresource_range.aspect_mask = src_aspect_mask;
    barrier_list.image_barriers.push_back(barrier);
    if (src_old_info.stage_flags != dst_old_info.stage_flags) {
        cmd_buf->addBarriers(
            barrier_list,
            src_old_info.stage_flags,
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT));
        barrier_list.image_barriers.clear();
    }

    barrier.image = dst_image;
    barrier.old_layout = dst_old_info.image_layout;
    barrier.new_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.src_access_mask = dst_old_info.access_flags;
    barrier.dst_access_mask = SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT);
    barrier.subresource_range.aspect_mask = dst_aspect_mask;
    barrier_list.image_barriers.push_back(barrier);
    cmd_buf->addBarriers(
        barrier_list,
        dst_old_info.stage_flags,
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT));
    barrier_list.image_barriers.clear();

    ImageBlitInfo copy_region;
    copy_region.src_offsets[0] = glm::ivec3(0, 0, 0);
    copy_region.src_offsets[1] = buffer_size;
    copy_region.dst_offsets[0] = glm::ivec3(0, 0, 0);
    copy_region.dst_offsets[1] = buffer_size;
    copy_region.src_subresource.aspect_mask = src_aspect_mask;
    copy_region.dst_subresource.aspect_mask = dst_aspect_mask;

    cmd_buf->blitImage(
        src_image,
        ImageLayout::TRANSFER_SRC_OPTIMAL,
        dst_image,
        ImageLayout::TRANSFER_DST_OPTIMAL,
        { copy_region },
        Filter::NEAREST);

    barrier.image = src_image;
    barrier.old_layout = ImageLayout::TRANSFER_SRC_OPTIMAL;
    barrier.new_layout = src_new_info.image_layout;
    barrier.src_access_mask = SET_FLAG_BIT(Access, TRANSFER_READ_BIT);
    barrier.dst_access_mask = src_new_info.access_flags;
    barrier.subresource_range.aspect_mask = src_aspect_mask;
    barrier_list.image_barriers.push_back(barrier);
    if (src_new_info.stage_flags != dst_new_info.stage_flags) {
        cmd_buf->addBarriers(
            barrier_list,
            SET_FLAG_BIT(PipelineStage, TRANSFER_BIT),
            src_new_info.stage_flags);
        barrier_list.image_barriers.clear();
    }

    barrier.image = dst_image;
    barrier.old_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = dst_new_info.image_layout;
    barrier.src_access_mask = SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT);
    barrier.dst_access_mask = dst_new_info.access_flags;
    barrier.subresource_range.aspect_mask = dst_aspect_mask;
    barrier_list.image_barriers.push_back(barrier);
    cmd_buf->addBarriers(
        barrier_list,
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT),
        dst_new_info.stage_flags);
}

void Helper::transitionImageLayout(
    const std::shared_ptr<CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::Image>& image,
    const renderer::Format& format,
    const renderer::ImageLayout& old_layout,
    const renderer::ImageLayout& new_layout,
    uint32_t base_mip_idx/* = 0*/,
    uint32_t mip_count/* = 1*/,
    uint32_t base_layer/* = 0*/,
    uint32_t layer_count/* = 1*/) {
    vk::helper::transitionImageLayout(
        cmd_buf,
        image,
        format,
        old_layout,
        new_layout,
        base_mip_idx,
        mip_count,
        base_layer,
        layer_count);
}

void Helper::transitionImageLayout(
    const std::shared_ptr<Device>& device,
    const std::shared_ptr<renderer::Image>& image,
    const renderer::Format& format,
    const renderer::ImageLayout& old_layout,
    const renderer::ImageLayout& new_layout,
    uint32_t base_mip_idx/* = 0*/,
    uint32_t mip_count/* = 1*/,
    uint32_t base_layer/* = 0*/,
    uint32_t layer_count/* = 1*/) {
    vk::helper::transitionImageLayout(
        device,
        image,
        format,
        old_layout,
        new_layout,
        base_mip_idx,
        mip_count,
        base_layer,
        layer_count);
}

bool Helper::acquireNextImage(
    const std::shared_ptr<Device>& device,
    const std::shared_ptr<renderer::Swapchain>& swap_chain,
    const std::shared_ptr<engine::renderer::Semaphore>& semaphore,
    uint32_t& image_index) {
    auto vk_device = RENDER_TYPE_CAST(Device, device);
    auto vk_swap_chain = RENDER_TYPE_CAST(Swapchain, swap_chain);
    auto vk_img_available_semphores = RENDER_TYPE_CAST(Semaphore, semaphore);
    assert(vk_device);

    auto result = vkAcquireNextImageKHR(vk_device->get(), vk_swap_chain->get(), UINT64_MAX, vk_img_available_semphores->get(), VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return true;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }
    return false;
}

void Helper::addImGuiToCommandBuffer(
    const std::shared_ptr<CommandBuffer>& cmd_buf) {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, RENDER_TYPE_CAST(CommandBuffer, cmd_buf)->get());
}

ImTextureID Helper::addImTextureID(
    const std::shared_ptr<Sampler>& sampler,
    const std::shared_ptr<ImageView>& image_View) {
    auto vk_sampler = RENDER_TYPE_CAST(Sampler, sampler) -> get();
    auto vk_image_view = RENDER_TYPE_CAST(ImageView, image_View)->get();
    
    return ImGui_ImplVulkan_AddTexture(vk_sampler, vk_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void Helper::submitQueue(
    const std::shared_ptr<Queue>& command_queue,
    const std::shared_ptr<Fence>& in_flight_fence,
    const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
    const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
    const std::vector<std::shared_ptr<Semaphore>>& signal_semaphores,
    const std::vector<uint64_t>& signal_semaphore_values) {

    std::vector<VkSemaphore> vk_wait_semaphores(wait_semaphores.size());
    for (auto i = 0; i < wait_semaphores.size(); i++) {
        vk_wait_semaphores[i] = RENDER_TYPE_CAST(Semaphore, wait_semaphores[i])->get();
    }

    std::vector<VkCommandBuffer> vk_cmd_bufs(command_buffers.size());
    for (auto i = 0; i < command_buffers.size(); i++) {
        vk_cmd_bufs[i] = RENDER_TYPE_CAST(CommandBuffer, command_buffers[i])->get();
    }

    std::vector<VkSemaphore> vk_signal_semaphores(signal_semaphores.size());
    for (auto i = 0; i < signal_semaphores.size(); i++) {
        vk_signal_semaphores[i] = RENDER_TYPE_CAST(Semaphore, signal_semaphores[i])->get();
    }
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submit_info.waitSemaphoreCount = static_cast<uint32_t>(vk_wait_semaphores.size());
    submit_info.pWaitSemaphores = vk_wait_semaphores.data();
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = static_cast<uint32_t>(vk_cmd_bufs.size());
    submit_info.pCommandBuffers = vk_cmd_bufs.data();
    submit_info.signalSemaphoreCount = static_cast<uint32_t>(vk_signal_semaphores.size());
    submit_info.pSignalSemaphores = vk_signal_semaphores.data();
    if (signal_semaphore_values.size() > 0) {
        VkTimelineSemaphoreSubmitInfo timeline_info = {};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = static_cast<uint32_t>(signal_semaphore_values.size());
        timeline_info.pSignalSemaphoreValues = signal_semaphore_values.data();
        timeline_info.waitSemaphoreValueCount = 0;
        timeline_info.pWaitSemaphoreValues = nullptr;

        submit_info.pNext = &timeline_info;
    }

    auto vk_command_queue = RENDER_TYPE_CAST(Queue, command_queue);
    auto vk_in_flight_fence = RENDER_TYPE_CAST(Fence, in_flight_fence);
    auto result =
        vkQueueSubmit(
            vk_command_queue->get(),
            1,
            &submit_info,
            vk_in_flight_fence ? vk_in_flight_fence->get() : nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
}

bool Helper::presentQueue(
    const std::shared_ptr<Queue>& present_queue,
    const std::vector<std::shared_ptr<Swapchain>>& swap_chains,
    const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
    const uint32_t& image_index,
    bool& frame_buffer_resized) {
  
    std::vector<VkSemaphore> vk_wait_semaphores(wait_semaphores.size());
    for (auto i = 0; i < wait_semaphores.size(); i++) {
        vk_wait_semaphores[i] = RENDER_TYPE_CAST(Semaphore, wait_semaphores[i])->get();
    }

    std::vector<VkSwapchainKHR> vk_swap_chains(swap_chains.size());
    for (auto i = 0; i < swap_chains.size(); i++) {
        vk_swap_chains[i] = RENDER_TYPE_CAST(Swapchain, swap_chains[i])->get();
    }

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = static_cast<uint32_t>(vk_wait_semaphores.size());
    present_info.pWaitSemaphores = vk_wait_semaphores.data();
    present_info.swapchainCount = static_cast<uint32_t>(vk_swap_chains.size());
    present_info.pSwapchains = vk_swap_chains.data();
    present_info.pImageIndices = &image_index;
    present_info.pResults = nullptr; // Optional

    auto vk_present_queue = RENDER_TYPE_CAST(Queue, present_queue);
    auto result = vkQueuePresentKHR(vk_present_queue->get(), &present_info);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frame_buffer_resized) {
        frame_buffer_resized = false;
        return true;
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    return false;
}

void Helper::initImgui(
    GLFWwindow* window,
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<Instance>& instance,
    const QueueFamilyList& queue_family_list,
    const SwapChainInfo& swap_chain_info,
    const std::shared_ptr<Queue>& graphics_queue,
    const std::shared_ptr<DescriptorPool>& descriptor_pool,
    const std::shared_ptr<RenderPass>& render_pass) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    auto logic_device = RENDER_TYPE_CAST(Device, device);
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = RENDER_TYPE_CAST(Instance, instance)->get();
    init_info.PhysicalDevice = RENDER_TYPE_CAST(PhysicalDevice, logic_device->getPhysicalDevice())->get();
    init_info.Device = logic_device->get();
    init_info.QueueFamily = queue_family_list.getGraphicAndPresentFamilyIndex()[0]; // get graphic queue family index.
    init_info.Queue = RENDER_TYPE_CAST(Queue, graphics_queue)->get();
    init_info.PipelineCache = nullptr;// g_PipelineCache;
    init_info.DescriptorPool = RENDER_TYPE_CAST(DescriptorPool, descriptor_pool)->get();
    init_info.Allocator = nullptr; // g_Allocator;
    init_info.MinImageCount = static_cast<uint32_t>(swap_chain_info.framebuffers.size());
    init_info.ImageCount = static_cast<uint32_t>(swap_chain_info.framebuffers.size());
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info, RENDER_TYPE_CAST(RenderPass, render_pass)->get());

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    // Upload Fonts
    {
        auto cmd_buf = device->setupTransientCommandBuffer();
        ImGui_ImplVulkan_CreateFontsTexture(
            RENDER_TYPE_CAST(
                CommandBuffer,
                cmd_buf
            )->get()
        );
        device->submitAndWaitTransientCommandBuffer();
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }
}

void TextureInfo::destroy(const std::shared_ptr<Device>& device) {
    device->destroyImage(image);
    device->destroyImageView(view);
    device->freeMemory(memory);

    for (auto& s_views : surface_views) {
        for (auto& s_view : s_views) {
            device->destroyImageView(s_view);
        }
    }

    for (auto& framebuffer : framebuffers) {
        device->destroyFramebuffer(framebuffer);
    }
}

void BufferInfo::destroy(const std::shared_ptr<Device>& device) {
    device->destroyBuffer(buffer);
    device->freeMemory(memory);
}

} // namespace renderer
} // namespace engine