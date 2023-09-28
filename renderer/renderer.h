#pragma once
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <optional>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "device.h"
#include "command_buffer.h"

namespace engine {
namespace renderer {

class Queue {
public:
    virtual void submit(
        const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
        const std::shared_ptr<Fence>& in_flight_fence) = 0;
    virtual void waitIdle() = 0;
};

class CommandPool {
};

class Instance {
public:
    virtual void destroy() = 0;
    virtual void destroySurface(const std::shared_ptr<Surface>& surface) = 0;
};

class PhysicalDevice {
};

class Surface {
};

class Swapchain {
};

class DescriptorPool {
};

class Pipeline {
};

class PipelineLayout {
};

class RenderPass {
};

class Framebuffer {
};

class ImageView {
};

class Sampler {
};

class Image {
public:
    virtual ImageLayout getImageLayout() = 0;
    virtual void setImageLayout(ImageLayout layout) = 0;
};

class Buffer {
public:
    virtual uint32_t getSize() = 0;
    virtual uint64_t getDeviceAddress() = 0;
};

class Semaphore {
};

class Fence {
};

class DeviceMemory {
};

class DescriptorSetLayout {
};

class DescriptorSet {
};

class ShaderModule {
};

namespace vk {
class VulkanInstance : public Instance {
    VkInstance    instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;

public:
    VkInstance get() { return instance_; }
    VkDebugUtilsMessengerEXT getDebugMessenger() { return debug_messenger_; }
    void set(const VkInstance& instance) { instance_ = instance; }
    void setDebugMessenger(const VkDebugUtilsMessengerEXT debug_messenger) {
        debug_messenger_ = debug_messenger;
    }

    virtual void destroy() final;
    virtual void destroySurface(const std::shared_ptr<Surface>& surface) final;
};

class VulkanPhysicalDevice : public PhysicalDevice {
    VkPhysicalDevice physical_device_;

public:
    VkPhysicalDevice get() { return physical_device_; }
    void set(const VkPhysicalDevice& physical_device) { physical_device_ = physical_device; }
};

class VulkanSurface : public Surface {
    VkSurfaceKHR    surface_;
public:
    VkSurfaceKHR get() { return surface_; }
    void set(const VkSurfaceKHR& surface) { surface_ = surface; }
};

class VulkanQueue : public Queue {
    VkQueue         queue_;
public:
    VkQueue get() { return queue_; }
    void set(const VkQueue& queue) { queue_ = queue; }

    virtual void submit(
        const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
        const std::shared_ptr<Fence>& in_flight_fence) final;
    virtual void waitIdle() final;
};

class VulkanCommandPool : public CommandPool {
    VkCommandPool    cmd_pool_;
public:
    VkCommandPool get() { return cmd_pool_; }
    void set(const VkCommandPool& cmd_pool) { cmd_pool_ = cmd_pool; }
};

class VulkanSwapchain : public Swapchain {
    VkSwapchainKHR    swap_chain_;
public:
    VkSwapchainKHR get() { return swap_chain_; }
    void set(VkSwapchainKHR swap_chain) { swap_chain_ = swap_chain; }
};

class VulkanDescriptorPool : public DescriptorPool {
    VkDescriptorPool    descriptor_pool_;
public:
    VkDescriptorPool get() { return descriptor_pool_; }
    void set(const VkDescriptorPool& descriptor_pool) { descriptor_pool_ = descriptor_pool; }
};

class VulkanPipeline : public Pipeline {
    VkPipeline    pipeline_;
public:
    VulkanPipeline(const VkPipeline& pipeline) : pipeline_(pipeline) {}
    VkPipeline get() { return pipeline_; }
};

class VulkanPipelineLayout : public PipelineLayout {
    VkPipelineLayout    pipeline_layout_;
public:
    VulkanPipelineLayout(const VkPipelineLayout& layout) : pipeline_layout_(layout) {}
    VkPipelineLayout get() { return pipeline_layout_; }
};

class VulkanRenderPass : public RenderPass {
    VkRenderPass    render_pass_;
public:
    VulkanRenderPass(const VkRenderPass& render_pass) : render_pass_(render_pass) {}
    VkRenderPass get() { return render_pass_; }
};

class VulkanFramebuffer : public Framebuffer {
    VkFramebuffer    frame_buffer_;
public:
    VulkanFramebuffer(const VkFramebuffer frame_buffer) : frame_buffer_(frame_buffer) {}
    VkFramebuffer get() { return frame_buffer_; }
};

class VulkanImageView : public ImageView {
    VkImageView     image_view_;
public:
    VulkanImageView(const VkImageView& image_view) : image_view_(image_view) {}
    VkImageView& get() { return image_view_; }
};

class VulkanSampler : public Sampler {
    VkSampler       sampler_;
public:
    VulkanSampler(const VkSampler& sampler) : sampler_(sampler) {}
    VkSampler get() { return sampler_; }
    const VkSampler* getPtr()const { return &sampler_; }
};

class VulkanImage : public Image {
    VkImage         image_;
    ImageLayout     layout_ = ImageLayout::UNDEFINED;
public:
    VulkanImage(const VkImage& image) : image_(image) {}
    VkImage get() { return image_; }
    virtual ImageLayout getImageLayout() { return layout_; }
    virtual void setImageLayout(ImageLayout layout) { layout_ = layout; }
};

class VulkanBuffer : public Buffer {
    uint32_t         size_;
    VkBuffer         buffer_;
    uint64_t         device_address_ = 0;
public:
    VulkanBuffer(const VkBuffer& buffer, uint32_t size) :
        buffer_(buffer), size_(size) {}
    VkBuffer get() { return buffer_; }
    void set_device_address(uint64_t device_address) { device_address_ = device_address; }
    virtual uint32_t getSize() final {
        return size_;
    }
    virtual uint64_t getDeviceAddress() final {
        return device_address_;
    }
};

class VulkanSemaphore : public Semaphore {
    VkSemaphore      semaphore_;
public:
    VulkanSemaphore(const VkSemaphore& semaphore) : semaphore_(semaphore){}
    VkSemaphore get() { return semaphore_; }
};

class VulkanFence : public Fence {
    VkFence          fence_;
public:
    VulkanFence(const VkFence& fence) : fence_(fence) {}
    VkFence get() { return fence_; }
};

class VulkanDeviceMemory : public DeviceMemory {
    VkDeviceMemory  memory_;
public:
    VkDeviceMemory get() { return memory_; }
    void set(const VkDeviceMemory& memory) { memory_ = memory; }
};

class VulkanDescriptorSetLayout : public DescriptorSetLayout {
    VkDescriptorSetLayout  layout_;
public:
    VkDescriptorSetLayout get() { return layout_; }
    void set(const VkDescriptorSetLayout layout) { layout_ = layout; }
};

class VulkanDescriptorSet : public DescriptorSet {
    VkDescriptorSet  desc_set_;
public:
    VkDescriptorSet get() { return desc_set_; }
    void set(const VkDescriptorSet desc_set) { desc_set_ = desc_set; }
};
    

class VulkanShaderModule : public ShaderModule {
    VkShaderModule          shader_module_;
    ShaderStageFlagBits     shader_stage_;
public:
    VulkanShaderModule(
        const VkShaderModule& shader_module,
        const ShaderStageFlagBits& shader_stage) :
            shader_module_(shader_module),
            shader_stage_(shader_stage){}
    ShaderStageFlagBits getShaderStage() { return shader_stage_; }
    VkShaderModule get() { return shader_module_; }
};
} // namespace vk

struct SwapChainInfo {
    renderer::Format format;
    glm::uvec2 extent;
    std::shared_ptr<renderer::Swapchain> swap_chain;
    std::vector<std::shared_ptr<renderer::Image>> images;
    std::vector<std::shared_ptr<renderer::ImageView>> image_views;
    std::vector<std::shared_ptr<renderer::Framebuffer>> framebuffers;
};

class Helper {
public:
    static void init(const std::shared_ptr<Device>& device);

    static void destroy(const std::shared_ptr<Device>& device);

    static Format findDepthFormat(const std::shared_ptr<Device>& device);

    static ImageResourceInfo getImageAsSource() { return image_source_info_; }
    static ImageResourceInfo getImageAsColorAttachment() { return image_as_color_attachement_; }
    static ImageResourceInfo getImageAsStore() { return image_as_store_; }
    static ImageResourceInfo getImageAsShaderSampler() { return image_as_shader_sampler_; }

    static std::shared_ptr<Instance> createInstance();

    static std::shared_ptr<Surface> createSurface(
        const std::shared_ptr<Instance>& instance,
        GLFWwindow* window);

    static PhysicalDeviceList collectPhysicalDevices(
        const std::shared_ptr<Instance>& instance);

    static std::shared_ptr<PhysicalDevice> pickPhysicalDevice(
        const PhysicalDeviceList& physical_devices,
        const std::shared_ptr<Surface>& surface);

    static QueueFamilyList findQueueFamilies(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const std::shared_ptr<Surface>& surface);

    static std::shared_ptr<Device> createLogicalDevice(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const std::shared_ptr<Surface>& surface,
        const QueueFamilyList& list);

    static void initRayTracingProperties(
        const std::shared_ptr<renderer::PhysicalDevice>& physical_device,
        const std::shared_ptr<renderer::Device>& device,
        PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        PhysicalDeviceAccelerationStructureFeatures& as_features);

    static void createSwapChain(
        GLFWwindow* window,
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<Surface>& surface,
        const QueueFamilyList& indices,
        SwapChainInfo& swap_chain_info,
        const ImageUsageFlags& usage);

    static void addOneTexture(
        WriteDescriptorList& descriptor_writes,
        const std::shared_ptr<DescriptorSet>& desc_set,
        const DescriptorType& desc_type,
        uint32_t binding,
        const std::shared_ptr<Sampler>& sampler,
        const std::shared_ptr<ImageView>& texture,
        ImageLayout image_layout/* = ImageLayout::SHADER_READ_ONLY_OPTIMAL*/);

    static void addOneBuffer(
        WriteDescriptorList& descriptor_writes,
        const std::shared_ptr<DescriptorSet>& desc_set,
        const DescriptorType& desc_type,
        uint32_t binding,
        const std::shared_ptr<Buffer>& buffer,
        uint32_t range,
        uint32_t offset = 0);

    static void addOneAccelerationStructure(
        WriteDescriptorList& descriptor_writes,
        const std::shared_ptr<DescriptorSet>& desc_set,
        const DescriptorType& desc_type,
        uint32_t binding,
        const std::vector<AccelerationStructure>& acceleration_structs);

    static void generateMipmapLevels(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<Image>& image,
        uint32_t mip_count,
        uint32_t width,
        uint32_t height,
        const ImageLayout& cur_image_layout);

    static void create2DTextureImage(
        const std::shared_ptr<renderer::Device>& device,
        Format format,
        int tex_width,
        int tex_height,
        int tex_channels,
        const void* pixels,
        std::shared_ptr<Image>& texture_image,
        std::shared_ptr<DeviceMemory>& texture_image_memory);

    static void create2DTextureImage(
        const std::shared_ptr<renderer::Device>& device,
        Format depth_format,
        const glm::uvec2& size,
        TextureInfo& texture_info,
        const renderer::ImageUsageFlags& usage,
        const renderer::ImageLayout& image_layout,
        const renderer::ImageTiling image_tiling = renderer::ImageTiling::OPTIMAL,
        const uint32_t memory_property = SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));

    static void dumpTextureImage(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<Image>& src_texture_image,
        Format depth_format,
        const glm::uvec3& buffer_size,
        const uint32_t& bytes_per_pixel,
        void* pixels);

    static void create3DTextureImage(
        const std::shared_ptr<renderer::Device>& device,
        Format depth_format,
        const glm::uvec3& size,
        TextureInfo& texture_info,
        const renderer::ImageUsageFlags& usage,
        const renderer::ImageLayout& image_layout,
        const renderer::ImageTiling image_tiling = renderer::ImageTiling::OPTIMAL,
        const uint32_t memory_property = SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT));

    static void createDepthResources(
        const std::shared_ptr<renderer::Device>& device,
        Format format,
        glm::uvec2 size,
        TextureInfo& texture_2d);

    static void createCubemapTexture(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<RenderPass>& render_pass,
        uint32_t width,
        uint32_t height,
        uint32_t mip_count,
        Format format,
        const std::vector<BufferImageCopyInfo>& copy_regions,
        TextureInfo& texture,
        uint64_t buffer_size = 0,
        void* data = nullptr);

    static void createBuffer(
        const std::shared_ptr<Device>& device,
        const BufferUsageFlags& usage,
        const MemoryPropertyFlags& memory_property,
        const MemoryAllocateFlags& allocate_flags,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory,
        const uint64_t buffer_size = 0,
        const void* src_data = nullptr);

    static void updateBufferWithSrcData(
        const std::shared_ptr<Device>& device,
        const uint64_t& buffer_size,
        const void* src_data,
        const std::shared_ptr<Buffer>& buffer);

    static void updateBufferWithSrcData(
        const std::shared_ptr<Device>& device,
        const uint64_t& buffer_size,
        const void* src_data,
        const std::shared_ptr<DeviceMemory>& buffer_memory);

    static void transitionImageLayout(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::Image>& image,
        const renderer::Format& format,
        const renderer::ImageLayout& old_layout,
        const renderer::ImageLayout& new_layout,
        uint32_t base_mip_idx = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1);
    
    static void transitionImageLayout(
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<renderer::Image>& image,
        const renderer::Format& format,
        const renderer::ImageLayout& old_layout,
        const renderer::ImageLayout& new_layout,
        uint32_t base_mip_idx = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1);

    static void blitImage(
        const std::shared_ptr<CommandBuffer>& cmd_buf,
        const std::shared_ptr<Image>& src_image,
        const std::shared_ptr<Image>& dst_image,
        const ImageResourceInfo& src_old_info,
        const ImageResourceInfo& src_new_info,
        const ImageResourceInfo& dst_old_info,
        const ImageResourceInfo& dst_new_info,
        const ImageAspectFlags& src_aspect_flags,
        const ImageAspectFlags& dst_aspect_flags,
        const glm::ivec3& buffer_size);

    static bool acquireNextImage(
        const std::shared_ptr<Device>& device,
        const std::shared_ptr<renderer::Swapchain>& swap_chain,
        const std::shared_ptr<engine::renderer::Semaphore>& semaphore,
        uint32_t& image_index);

    static void addImGuiToCommandBuffer(
        const std::shared_ptr<CommandBuffer>& cmd_buf);

    static ImTextureID addImTextureID(
        const std::shared_ptr<Sampler>& sampler,
        const std::shared_ptr<ImageView>& image_View);

    static void submitQueue(
        const std::shared_ptr<Queue>& present_queue,
        const std::shared_ptr<Fence>& in_flight_fence,
        const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
        const std::vector<std::shared_ptr<CommandBuffer>>& command_buffers,
        const std::vector<std::shared_ptr<Semaphore>>& signal_semaphores,
        const std::vector<uint64_t>& signal_semaphore_values);

    static bool presentQueue(
        const std::shared_ptr<Queue>& present_queue,
        const std::vector<std::shared_ptr<Swapchain>>& swap_chains,
        const std::vector<std::shared_ptr<Semaphore>>& wait_semaphores,
        const uint32_t& image_index,
        bool& frame_buffer_resized);

    static void initImgui(
        GLFWwindow* window,
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<Instance>& instance,
        const QueueFamilyList& queue_family_indices,
        const SwapChainInfo& swap_chain_info,
        const std::shared_ptr<Queue>& graphics_queue,
        const std::shared_ptr<DescriptorPool>& descriptor_pool,
        const std::shared_ptr<RenderPass>& render_pass);

    static TextureInfo& getBlackTexture() { return black_tex_; }
    static TextureInfo& getWhiteTexture() { return white_tex_; }
    static TextureInfo& getPermutationTexture() {
        return permutation_tex_;
    }
    static TextureInfo& getPermutation2DTexture() {
        return permutation_2d_tex_;
    }
    static TextureInfo& getGradTexture() {
        return grad_tex_;
    }
    static TextureInfo& getPermGradTexture() {
        return perm_grad_tex_;
    }
    static TextureInfo& getPermGrad4DTexture() {
        return perm_grad_4d_tex_;
    }
    static TextureInfo& getGrad4DTexture() {
        return grad_4d_tex_;
    }

public:
    static TextureInfo black_tex_;
    static TextureInfo white_tex_;
    static TextureInfo permutation_tex_;
    static TextureInfo permutation_2d_tex_;
    static TextureInfo grad_tex_;
    static TextureInfo perm_grad_tex_;
    static TextureInfo perm_grad_4d_tex_;
    static TextureInfo grad_4d_tex_;
    static ImageResourceInfo image_source_info_;
    static ImageResourceInfo image_as_color_attachement_;
    static ImageResourceInfo image_as_store_;
    static ImageResourceInfo image_as_shader_sampler_;
};

} // namespace renderer
} // namespace engine