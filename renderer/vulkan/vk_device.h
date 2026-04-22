#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <vulkan/vulkan.h>
#include "../device.h"

namespace engine {
namespace renderer {
namespace vk {

const char* VkResultToString(VkResult result);

class VulkanDevice : public Device {
    VkDevice        device_;
    const std::shared_ptr<PhysicalDevice>& physical_device_;
    std::shared_ptr<CommandPool> transient_cmd_pool_;
    std::shared_ptr<CommandBuffer> transient_cmd_buffer_;
    std::shared_ptr<Queue> transient_compute_queue_;
    std::shared_ptr<Fence> transient_fence_;

    // ── Async loader queue (see base Device) ──
    // Second queue on the graphics family, used by the async mesh-load
    // worker thread to submit vertex/index/texture staging copies without
    // blocking the main frame tick. nullptr if the hardware exposes too
    // few queues (callers fall back to the synchronous transient path).
    std::shared_ptr<CommandPool> loader_cmd_pool_;
    std::shared_ptr<Queue>       loader_queue_;
    uint32_t                     loader_queue_family_index_ = (uint32_t)-1;

    // ── Worker-thread transient channel ──
    // setupTransientCommandBuffer() and submitAndWaitTransientCommandBuffer()
    // look at std::this_thread::get_id() and route to this channel when the
    // caller is the registered loader worker. This keeps existing helpers
    // (Helper::createBuffer etc.) usable unchanged from the worker thread
    // without serializing on the main thread's transient queue.
    std::shared_ptr<CommandBuffer> loader_transient_cmd_buffer_;
    std::shared_ptr<Fence>         loader_transient_fence_;
    std::atomic<std::thread::id>   loader_thread_id_{std::thread::id{}};

    // ── Resource tracking lists ──
    // Guarded by tracking_mutex_ because the async mesh-load worker thread
    // can create buffers/images concurrently with the main render thread.
    // Vulkan itself allows concurrent object creation on different command
    // pools, but our bookkeeping vectors are not thread-safe — hence the
    // lock. Held only around the std::vector push_back / erase / find.
    mutable std::mutex tracking_mutex_;
    std::vector<std::shared_ptr<Buffer>> buffer_list_;
    std::vector<std::shared_ptr<Image>> image_list_;
    std::vector<std::shared_ptr<ImageView>> image_view_list_;
    std::vector<std::shared_ptr<Sampler>> sampler_list_;
    std::vector<std::shared_ptr<ShaderModule>> shader_list_;
    std::vector<std::shared_ptr<Framebuffer>> framebuffer_list_;
    std::vector<std::shared_ptr<PipelineLayout>> pipeline_layout_list_;
    std::vector<std::shared_ptr<Pipeline>> pipeline_list_;
    std::vector<std::shared_ptr<RenderPass>> render_pass_list_;
    std::vector<std::shared_ptr<Semaphore>> semaphore_list_;
    std::vector<std::shared_ptr<Fence>> fence_list_;

public:
    VulkanDevice(
        const std::shared_ptr<PhysicalDevice>& physical_device,
        const QueueFamilyList& queue_list,
        const VkDevice& device,
        uint32_t queue_family_index);
    virtual ~VulkanDevice() final;

    VkDevice get() { return device_; }
    virtual std::shared_ptr<CommandBuffer> setupTransientCommandBuffer() final;
    virtual void submitAndWaitTransientCommandBuffer() final;

    virtual bool hasLoaderQueue() const final {
        return loader_queue_ != nullptr;
    }
    virtual std::shared_ptr<Queue> getLoaderQueue() final {
        return loader_queue_;
    }
    virtual uint32_t getLoaderQueueFamilyIndex() const final {
        return loader_queue_family_index_;
    }
    virtual std::shared_ptr<CommandPool> getLoaderCommandPool() final {
        return loader_cmd_pool_;
    }

    virtual void registerLoaderThread(std::thread::id id) final {
        loader_thread_id_.store(id, std::memory_order_release);
    }

    const std::shared_ptr<PhysicalDevice>& getPhysicalDevice() {
        return physical_device_; }
    virtual std::shared_ptr<DescriptorPool> createDescriptorPool() final;
    virtual void createBuffer(
        const uint64_t& buffer_size,
        const BufferUsageFlags& usage,
        const MemoryPropertyFlags& properties,
        const MemoryAllocateFlags& alloc_flags,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory,
        const std::source_location& src_location) final;
    virtual void updateDescriptorSets(
        const WriteDescriptorList& write_descriptors) final;
    virtual DescriptorSetList createDescriptorSets(
        std::shared_ptr<DescriptorPool> descriptor_pool,
        std::shared_ptr<DescriptorSetLayout> descriptor_set_layout,
        uint64_t buffer_count) final;
    virtual std::shared_ptr<PipelineLayout> createPipelineLayout(
        const DescriptorSetLayoutList& desc_set_layouts,
        const std::vector<PushConstantRange>& push_const_ranges,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<DescriptorSetLayout> createDescriptorSetLayout(
        const std::vector<DescriptorSetLayoutBinding>& bindings) final;
    virtual std::shared_ptr<RenderPass> createRenderPass(
        const std::vector<AttachmentDescription>& attachments,
        const std::vector<SubpassDescription>& subpasses,
        const std::vector<SubpassDependency>& dependencies,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::vector<VertexInputBindingDescription>& binding_descs,
        const std::vector<VertexInputAttributeDescription>& attribute_descs,
        const PipelineInputAssemblyStateCreateInfo& topology_info,
        const GraphicPipelineInfo& graphic_pipeline_info,
        const ShaderModuleList& shader_modules,
        const glm::uvec2& extent,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::vector<VertexInputBindingDescription>& binding_descs,
        const std::vector<VertexInputAttributeDescription>& attribute_descs,
        const PipelineInputAssemblyStateCreateInfo& topology_info,
        const GraphicPipelineInfo& graphic_pipeline_info,
        const ShaderModuleList& shader_modules,
        const PipelineRenderbufferFormats& frame_buffer_format,
        const RasterizationStateOverride& rasterization_state_override,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::shared_ptr<renderer::ShaderModule>& shader_module,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const ShaderModuleList& src_shader_modules,
        const RtShaderGroupCreateInfoList& src_shader_groups,
        const std::source_location& src_location,
        const uint32_t ray_recursion_depth = 1) final;
    virtual std::shared_ptr<Swapchain> createSwapchain(
        const std::shared_ptr<Surface>& surface,
        const uint32_t& image_count,
        const Format& format,
        const glm::uvec2& buf_size,
        const ColorSpace& color_space,
        const SurfaceTransformFlagBits& transform,
        const PresentMode& present_mode,
        const ImageUsageFlags& usage,
        const std::vector<uint32_t>& queue_index) final;
    virtual void updateBufferMemory(
        const std::shared_ptr<DeviceMemory>& memory,
        uint64_t size,
        const void* src_data,
        uint64_t offset = 0) final;
    virtual void dumpBufferMemory(
        const std::shared_ptr<DeviceMemory>& memory,
        uint64_t size,
        void* dst_data,
        uint64_t offset = 0) final;
    virtual std::vector<std::shared_ptr<Image>> getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) final;
    virtual std::shared_ptr<CommandPool> createCommandPool(uint32_t queue_family_index, CommandPoolCreateFlags flags) final;
    virtual std::shared_ptr<Queue> getDeviceQueue(uint32_t queue_family_index, uint32_t queue_index = 0) final;
    virtual std::shared_ptr<DeviceMemory> allocateMemory(
        const uint64_t& buf_size,
        const uint32_t& memory_type_bits,
        const MemoryPropertyFlags& properties,
        const MemoryAllocateFlags& allocate_flags) final;
    virtual MemoryRequirements getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) final;
    virtual MemoryRequirements getImageMemoryRequirements(std::shared_ptr<Image> image) final;
    virtual std::shared_ptr<Buffer> createBuffer(
        uint64_t buf_size,
        BufferUsageFlags usage,
        const std::source_location& src_location,
        bool sharing = false) final;
    virtual std::shared_ptr<Image> createImage(
        ImageType image_type,
        glm::uvec3 image_size,
        Format format,
        ImageUsageFlags usage,
        ImageTiling tiling,
        ImageLayout layout,
        const std::source_location& src_location,
        ImageCreateFlags flags = 0,
        bool sharing = false,
        uint32_t num_samples = 1,
        uint32_t num_mips = 1,
        uint32_t num_layers = 1) final;
    virtual std::shared_ptr<ShaderModule>
        createShaderModule(
            uint64_t size,
            void* data,
            ShaderStageFlagBits shader_stage,
            const std::source_location& src_location) final;
    virtual std::shared_ptr<ImageView>
        createImageView(
        std::shared_ptr<Image> image,
        ImageViewType view_type,
        Format format,
        ImageAspectFlags aspect_flags,
        const std::source_location& src_location,
        uint32_t base_mip = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1) final;
    virtual std::shared_ptr<Framebuffer>
        createFrameBuffer(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::vector<std::shared_ptr<ImageView>>& attachments,
        const glm::uvec2& extent,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Sampler> createSampler(
        Filter filter,
        SamplerAddressMode address_mode,
        SamplerMipmapMode mipmap_mode,
        float anisotropy,
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Semaphore> createSemaphore(
        const std::source_location& src_location) final;
    virtual std::shared_ptr<Fence> createFence(
        const std::source_location& src_location,
        bool signaled = false) final;
    virtual void bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset = 0) final;
    virtual void bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset = 0) final;
    virtual std::vector<std::shared_ptr<CommandBuffer>> allocateCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, uint32_t num_buffers, bool is_primary = true) final;
    virtual void* mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset = 0) final;
    virtual void unmapMemory(std::shared_ptr<DeviceMemory> memory) final;
    virtual void destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) final;
    virtual void destroySwapchain(std::shared_ptr<Swapchain> swapchain) final;
    virtual void destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) final;
    virtual void destroyPipeline(std::shared_ptr<Pipeline> pipeline) final;
    virtual void destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) final;
    virtual void destroyRenderPass(std::shared_ptr<RenderPass> render_pass) final;
    virtual void destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) final;
    virtual void destroyImageView(std::shared_ptr<ImageView> image_view) final;
    virtual void destroySampler(std::shared_ptr<Sampler> sampler) final;
    virtual void destroyImage(std::shared_ptr<Image> image) final;
    virtual void destroyBuffer(std::shared_ptr<Buffer> buffer) final;
    virtual void destroySemaphore(std::shared_ptr<Semaphore> semaphore) final;
    virtual void destroyFence(std::shared_ptr<Fence> fence) final;
    virtual void destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) final;
    virtual void destroyShaderModule(std::shared_ptr<ShaderModule> layout) final;
    virtual void destroy() final;
    virtual void freeMemory(std::shared_ptr<DeviceMemory> memory) final;
    virtual void freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) final;
    virtual void resetFences(const std::vector<std::shared_ptr<Fence>>& fences) final;
    virtual void waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) final;
    virtual bool isFenceSignaled(const std::shared_ptr<Fence>& fence) final;
    virtual void waitForSemaphores(const std::vector<std::shared_ptr<Semaphore>>& semaphores, uint64_t value) final;
    virtual void waitIdle() final;
    virtual void getAccelerationStructureBuildSizes(
        AccelerationStructureBuildType         as_build_type,
        const AccelerationStructureBuildGeometryInfo& build_info,
        AccelerationStructureBuildSizesInfo& size_info) final;
    virtual AccelerationStructure createAccelerationStructure(
        const std::shared_ptr<Buffer>& buffer,
        const AccelerationStructureType& as_type) final;
    virtual void destroyAccelerationStructure(const AccelerationStructure& as) final;
    virtual DeviceAddress getAccelerationStructureDeviceAddress(
        const AccelerationStructure& as) final;
    virtual void getRayTracingShaderGroupHandles(
        const std::shared_ptr<Pipeline>& pipeline,
        const uint32_t group_count,
        const uint32_t sbt_size,
        void* shader_handle_storage) final;

    virtual std::shared_ptr<QueryPool> createQueryPool(
        uint32_t query_count) final;
    virtual void destroyQueryPool(
        std::shared_ptr<QueryPool> query_pool) final;
    virtual bool getQueryPoolResults(
        const std::shared_ptr<QueryPool>& query_pool,
        uint32_t first_query,
        uint32_t query_count,
        std::vector<uint64_t>& results) final;
    virtual float getTimestampPeriod() final;
};

} // namespace vk
} // namespace renderer
} // namespace engine
