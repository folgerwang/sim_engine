#pragma once
#include <thread>
#include "renderer_structs.h"

namespace engine {
namespace renderer {

class Device {
public:
    virtual std::shared_ptr<DescriptorPool> createDescriptorPool() = 0;
    // Transient command buffer dispatch. When called from the loader
    // worker thread (registered via registerLoaderThread) the device
    // returns a worker-owned command buffer + fence + queue; otherwise
    // the main/compute channel is used. This keeps the 18 existing call
    // sites working on either thread without modification.
    virtual std::shared_ptr<CommandBuffer> setupTransientCommandBuffer() = 0;
    virtual void submitAndWaitTransientCommandBuffer() = 0;

    // Tell the device which thread should be treated as the loader
    // worker. Set at MeshLoadTaskManager worker startup. Re-setting
    // is safe; pass std::thread::id{} to clear.
    virtual void registerLoaderThread(std::thread::id id) = 0;

    // ── Async loader queue (Layer 1 of async mesh load) ──────────────────
    // A second queue intended for worker-thread uploads that must not block
    // the main frame tick. On hardware with a dedicated transfer family we
    // should eventually prefer that; for now we reuse the graphics family
    // and take a separate queue index so the loader doesn't contend with
    // the existing transient compute queue.
    //
    // Returns nullptr / (uint32_t)-1 if the hardware exposes too few queues
    // to spare one — callers should fall back to the synchronous path
    // (createBuffer + submitAndWaitTransientCommandBuffer) in that case.
    virtual bool hasLoaderQueue() const = 0;
    virtual std::shared_ptr<Queue> getLoaderQueue() = 0;
    virtual uint32_t getLoaderQueueFamilyIndex() const = 0;
    virtual std::shared_ptr<CommandPool> getLoaderCommandPool() = 0;

    virtual void createBuffer(
        const uint64_t& buffer_size,
        const BufferUsageFlags& usage,
        const MemoryPropertyFlags& properties,
        const MemoryAllocateFlags& alloc_flags,
        std::shared_ptr<Buffer>& buffer,
        std::shared_ptr<DeviceMemory>& buffer_memory,
        const std::source_location& src_location) = 0;
    virtual void updateDescriptorSets(
        const WriteDescriptorList& write_descriptors) = 0;
    virtual DescriptorSetList createDescriptorSets(
        std::shared_ptr<DescriptorPool> descriptor_pool,
        std::shared_ptr<DescriptorSetLayout> descriptor_set_layout,
        uint64_t buffer_count) = 0;
    virtual std::shared_ptr<PipelineLayout> createPipelineLayout(
        const DescriptorSetLayoutList& desc_set_layouts,
        const std::vector<PushConstantRange>& push_const_ranges,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<DescriptorSetLayout> createDescriptorSetLayout(
        const std::vector<DescriptorSetLayoutBinding>& bindings) = 0;
    virtual std::shared_ptr<RenderPass> createRenderPass(
        const std::vector<AttachmentDescription>& attachments,
        const std::vector<SubpassDescription>& subpasses,
        const std::vector<SubpassDependency>& dependencies,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::vector<VertexInputBindingDescription>& binding_descs,
        const std::vector<VertexInputAttributeDescription>& attribute_descs,
        const PipelineInputAssemblyStateCreateInfo& topology_info,
        const GraphicPipelineInfo& graphic_pipeline_info,
        const ShaderModuleList& shader_modules,
        const glm::uvec2& extent,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::vector<VertexInputBindingDescription>& binding_descs,
        const std::vector<VertexInputAttributeDescription>& attribute_descs,
        const PipelineInputAssemblyStateCreateInfo& topology_info,
        const GraphicPipelineInfo& graphic_pipeline_info,
        const ShaderModuleList& shader_modules,
        const PipelineRenderbufferFormats& frame_buffer_format,
        const RasterizationStateOverride& rasterization_state_override,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const std::shared_ptr<ShaderModule>& shader_module,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Pipeline> createPipeline(
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const ShaderModuleList& src_shader_modules,
        const RtShaderGroupCreateInfoList& src_shader_groups,
        const std::source_location& src_location,
        const uint32_t ray_recursion_depth = 1) = 0;
    virtual std::shared_ptr<Swapchain> createSwapchain(
        const std::shared_ptr<Surface>& surface,
        const uint32_t& image_count,
        const Format& format,
        const glm::uvec2& buf_size,
        const ColorSpace& color_space,
        const SurfaceTransformFlagBits& transform,
        const PresentMode& present_mode,
        const ImageUsageFlags& usage,
        const std::vector<uint32_t>& queue_index) = 0;
    virtual void updateBufferMemory(
        const std::shared_ptr<DeviceMemory>& memory,
        uint64_t size,
        const void* src_data,
        uint64_t offset = 0) = 0;
    virtual void dumpBufferMemory(
        const std::shared_ptr<DeviceMemory>& memory,
        uint64_t size,
        void* dst_data,
        uint64_t offset = 0) = 0;
    virtual std::vector<std::shared_ptr<renderer::Image>> getSwapchainImages(std::shared_ptr<Swapchain> swap_chain) = 0;
    virtual std::shared_ptr<CommandPool> createCommandPool(uint32_t queue_family_index, CommandPoolCreateFlags flags) = 0;
    virtual std::shared_ptr<Queue> getDeviceQueue(uint32_t queue_family_index, uint32_t queue_index = 0) = 0;
    virtual std::shared_ptr<DeviceMemory> allocateMemory(
        const uint64_t& buf_size,
        const uint32_t& memory_type_bits,
        const MemoryPropertyFlags& properties,
        const MemoryAllocateFlags& allocate_flags) = 0;
    virtual MemoryRequirements getBufferMemoryRequirements(std::shared_ptr<Buffer> buffer) = 0;
    virtual MemoryRequirements getImageMemoryRequirements(std::shared_ptr<Image> image) = 0;
    virtual std::shared_ptr<Buffer> createBuffer(
        uint64_t buf_size,
        BufferUsageFlags usage,
        const std::source_location& src_location,
        bool sharing = false) = 0;
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
        uint32_t num_layers = 1) = 0;
    virtual std::shared_ptr<ShaderModule>
        createShaderModule(
            uint64_t size,
            void* data,
            ShaderStageFlagBits shader_stage,
            const std::source_location& src_location) = 0;
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
        uint32_t layer_count = 1) = 0;
    virtual std::shared_ptr<Framebuffer>
        createFrameBuffer(
        const std::shared_ptr<RenderPass>& render_pass,
        const std::vector<std::shared_ptr<ImageView>>& attachments,
        const glm::uvec2& extent,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Sampler> createSampler(
        Filter filter,
        SamplerAddressMode address_mode,
        SamplerMipmapMode mipmap_mode,
        float anisotropy,
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Semaphore> createSemaphore(
        const std::source_location& src_location) = 0;
    virtual std::shared_ptr<Fence> createFence(
        const std::source_location& src_location,
        bool signaled = false) = 0;
    virtual void bindBufferMemory(std::shared_ptr<Buffer> buffer, std::shared_ptr<DeviceMemory> buffer_memory, uint64_t offset = 0) = 0;
    virtual void bindImageMemory(std::shared_ptr<Image> image, std::shared_ptr<DeviceMemory> image_memory, uint64_t offset = 0) = 0;
    virtual std::vector<std::shared_ptr<CommandBuffer>> allocateCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, uint32_t num_buffers, bool is_primary = true) = 0;
    virtual void* mapMemory(std::shared_ptr<DeviceMemory> memory, uint64_t size, uint64_t offset = 0) = 0;
    virtual void unmapMemory(std::shared_ptr<DeviceMemory> memory) = 0;
    virtual void destroyCommandPool(std::shared_ptr<CommandPool> cmd_pool) = 0;
    virtual void destroySwapchain(std::shared_ptr<Swapchain> swapchain) = 0;
    virtual void destroyDescriptorPool(std::shared_ptr<DescriptorPool> descriptor_pool) = 0;
    virtual void destroyPipeline(std::shared_ptr<Pipeline> pipeline) = 0;
    virtual void destroyPipelineLayout(std::shared_ptr<PipelineLayout> pipeline_layout) = 0;
    virtual void destroyRenderPass(std::shared_ptr<RenderPass> render_pass) = 0;
    virtual void destroyFramebuffer(std::shared_ptr<Framebuffer> frame_buffer) = 0;
    virtual void destroyImageView(std::shared_ptr<ImageView> image_view) = 0;
    virtual void destroySampler(std::shared_ptr<Sampler> sampler) = 0;
    virtual void destroyImage(std::shared_ptr<Image> image) = 0;
    virtual void destroyBuffer(std::shared_ptr<Buffer> buffer) = 0;
    virtual void destroySemaphore(std::shared_ptr<Semaphore> semaphore) = 0;
    virtual void destroyFence(std::shared_ptr<Fence> fence) = 0;
    virtual void destroyDescriptorSetLayout(std::shared_ptr<DescriptorSetLayout> layout) = 0;
    virtual void destroyShaderModule(std::shared_ptr<ShaderModule> layout) = 0;
    virtual void destroy() = 0;
    virtual void freeMemory(std::shared_ptr<DeviceMemory> memory) = 0;
    virtual void freeCommandBuffers(std::shared_ptr<CommandPool> cmd_pool, const std::vector<std::shared_ptr<CommandBuffer>>& cmd_bufs) = 0;
    virtual void resetFences(const std::vector<std::shared_ptr<Fence>>& fences) = 0;
    virtual void waitForFences(const std::vector<std::shared_ptr<Fence>>& fences) = 0;
    // Non-blocking fence status check. Returns true if the fence is signaled.
    // Used by the async loader poll loop (vkGetFenceStatus path).
    virtual bool isFenceSignaled(const std::shared_ptr<Fence>& fence) = 0;
    virtual void waitForSemaphores(const std::vector<std::shared_ptr<Semaphore>>& semaphores, uint64_t value) = 0;
    virtual void waitIdle() = 0;
    virtual void getAccelerationStructureBuildSizes(
        AccelerationStructureBuildType         as_build_type,
        const AccelerationStructureBuildGeometryInfo& build_info,
        AccelerationStructureBuildSizesInfo& size_info) = 0;
    virtual AccelerationStructure createAccelerationStructure(
        const std::shared_ptr<Buffer>& buffer,
        const AccelerationStructureType& as_type) = 0;
    virtual void destroyAccelerationStructure(const AccelerationStructure& as) = 0;
    virtual DeviceAddress getAccelerationStructureDeviceAddress(
        const AccelerationStructure& as) = 0;
    virtual void getRayTracingShaderGroupHandles(
        const std::shared_ptr<Pipeline>& pipeline,
        const uint32_t group_count,
        const uint32_t sbt_size,
        void* shader_handle_storage) = 0;

    // Timestamp query pool interface.
    virtual std::shared_ptr<QueryPool> createQueryPool(
        uint32_t query_count) = 0;
    virtual void destroyQueryPool(
        std::shared_ptr<QueryPool> query_pool) = 0;
    virtual bool getQueryPoolResults(
        const std::shared_ptr<QueryPool>& query_pool,
        uint32_t first_query,
        uint32_t query_count,
        std::vector<uint64_t>& results) = 0;
    virtual float getTimestampPeriod() = 0;
};

} // namespace renderer
} // namespace engine