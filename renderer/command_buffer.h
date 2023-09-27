#pragma once
#include "renderer_structs.h"

namespace engine {
namespace renderer {

class CommandBuffer {
public:
    virtual void beginCommandBuffer(CommandBufferUsageFlags flags) = 0;
    virtual void endCommandBuffer() = 0;
    virtual void copyBuffer(
        std::shared_ptr<Buffer> src_buf,
        std::shared_ptr<Buffer> dst_buf,
        std::vector<BufferCopyInfo> copy_regions) = 0;
    virtual void copyImage(
        std::shared_ptr<Image> src_img,
        ImageLayout src_img_layout,
        std::shared_ptr<Image> dst_img,
        ImageLayout dst_img_layout,
        std::vector<ImageCopyInfo> copy_regions) = 0;
    virtual void blitImage(
        std::shared_ptr<Image> src_img,
        ImageLayout src_img_layout,
        std::shared_ptr<Image> dst_img,
        ImageLayout dst_img_layout,
        std::vector<ImageBlitInfo> copy_regions,
        const Filter& filter) = 0;
    virtual void resolveImage(
        std::shared_ptr<Image> src_img,
        ImageLayout src_img_layout,
        std::shared_ptr<Image> dst_img,
        ImageLayout dst_img_layout,
        std::vector<ImageResolveInfo> copy_regions) = 0;
    virtual void copyBufferToImage(
        std::shared_ptr<Buffer> src_buf,
        std::shared_ptr<Image> dst_image,
        std::vector<BufferImageCopyInfo> copy_regions,
        ImageLayout layout) = 0;
    virtual void copyImageToBuffer(
        std::shared_ptr<Image> src_image,
        std::shared_ptr<Buffer> dst_buf,
        std::vector<BufferImageCopyInfo> copy_regions,
        ImageLayout layout) = 0;
    virtual void bindPipeline(PipelineBindPoint bind, std::shared_ptr<Pipeline> pipeline) = 0;
    virtual void bindVertexBuffers(uint32_t first_bind, const std::vector<std::shared_ptr<renderer::Buffer>>& vertex_buffers, std::vector<uint64_t> offsets) = 0;
    virtual void bindIndexBuffer(std::shared_ptr<Buffer> index_buffer, uint64_t offset, IndexType index_type) = 0;
    virtual void bindDescriptorSets(
        PipelineBindPoint bind_point,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const DescriptorSetList& desc_sets,
        const uint32_t first_set_idx = 0) = 0;
    virtual void pushConstants(
        ShaderStageFlags stages,
        const std::shared_ptr<PipelineLayout>& pipeline_layout,
        const void* data,
        uint32_t size,
        uint32_t offset = 0) = 0;
    virtual void draw(uint32_t vertex_count, 
        uint32_t instance_count = 1, 
        uint32_t first_vertex = 0, 
        uint32_t first_instance = 0) = 0;
    virtual void drawIndexed(
        uint32_t index_count, 
        uint32_t instance_count = 1, 
        uint32_t first_index = 0, 
        uint32_t vertex_offset = 0, 
        uint32_t first_instance = 0) = 0;
    virtual void drawIndexedIndirect(
        const renderer::BufferInfo& indirect_draw_cmd_buf,
        uint32_t buffer_offset = 0,
        uint32_t draw_count = 1,
        uint32_t stride = sizeof(DrawIndexedIndirectCommand)) = 0;
    virtual void drawIndirect(
        const renderer::BufferInfo& indirect_draw_cmd_buf,
        uint32_t buffer_offset = 0,
        uint32_t draw_count = 1,
        uint32_t stride = sizeof(DrawIndirectCommand)) = 0;
    typedef void (VKAPI_PTR* PFN_vkCmdDrawMeshTasksEXT)(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    typedef void (VKAPI_PTR* PFN_vkCmdDrawMeshTasksIndirectEXT)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride);
    typedef void (VKAPI_PTR* PFN_vkCmdDrawMeshTasksIndirectCountEXT)(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount, uint32_t stride);

    virtual void drawMeshTasks(
        uint32_t group_count_x = 1,
        uint32_t group_count_y = 1,
        uint32_t group_count_z = 1) = 0;
    virtual void drawMeshTasksIndirect() = 0;
    virtual void drawMeshTasksIndirectCount() = 0;
    virtual void dispatch(
        uint32_t group_count_x, 
        uint32_t group_count_y, 
        uint32_t group_count_z = 1) = 0;
    virtual void traceRays(
        const StridedDeviceAddressRegion& raygen_shader_entry,
        const StridedDeviceAddressRegion& miss_shader_entry,
        const StridedDeviceAddressRegion& hit_shader_entry,
        const StridedDeviceAddressRegion& callable_shader_entry,
        const glm::uvec3& size) = 0;
    virtual void beginRenderPass(
        std::shared_ptr<RenderPass> render_pass,
        std::shared_ptr<Framebuffer> frame_buffer,
        const glm::uvec2& extent,
        const std::vector<ClearValue>& clear_values) = 0;
    virtual void endRenderPass() = 0;
    virtual void reset(uint32_t flags) = 0;
    virtual void addBarriers(
        const BarrierList& barrier_list,
        PipelineStageFlags src_stage_flags,
        PipelineStageFlags dst_stage_flags) = 0;
    virtual void addImageBarrier(
        const std::shared_ptr<Image>& image,
        const ImageResourceInfo& src_info,
        const ImageResourceInfo& dst_info,
        uint32_t base_mip = 0,
        uint32_t mip_count = 1,
        uint32_t base_layer = 0,
        uint32_t layer_count = 1) = 0;
    virtual void addBufferBarrier(
        const std::shared_ptr<Buffer>& buffer,
        const BufferResourceInfo& src_info,
        const BufferResourceInfo& dst_info,
        uint32_t size = 0,
        uint32_t offset = 0) = 0;
    //virtual void pipelineBarrier() = 0;
    virtual void buildAccelerationStructures(
        const std::vector<AccelerationStructureBuildGeometryInfo>& as_build_geo_list,
        const std::vector<AccelerationStructureBuildRangeInfo>& as_build_range_list) = 0;
};

} // namespace renderer
} // namespace engine