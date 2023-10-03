#include <iostream>

#include "../renderer.h"
#include "vk_device.h"
#include "vk_command_buffer.h"
#include "vk_renderer_helper.h"

namespace engine {
namespace renderer {
namespace vk {

void VulkanCommandBuffer::beginCommandBuffer(
    CommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = helper::toCommandBufferUsageFlags(flags);
    begin_info.pInheritanceInfo = nullptr; // Optional

    auto result =
        vkBeginCommandBuffer(
            cmd_buf_,
            &begin_info);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to start recording command buffer! : ") +
            VkResultToString(result));
    }
};

void VulkanCommandBuffer::endCommandBuffer() {
    auto result =
        vkEndCommandBuffer(cmd_buf_);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("failed to finish recording command buffer! : ") +
            VkResultToString(result));
    }
}

void VulkanCommandBuffer::copyBuffer(
    std::shared_ptr<Buffer> src_buf,
    std::shared_ptr<Buffer> dst_buf,
    std::vector<BufferCopyInfo> copy_regions) {
    std::vector<VkBufferCopy> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i] = helper::toVkBufferCopy(copy_regions[i]);
    }
    auto vk_src_buf = RENDER_TYPE_CAST(Buffer, src_buf);
    auto vk_dst_buf = RENDER_TYPE_CAST(Buffer, dst_buf);

    vkCmdCopyBuffer(
        cmd_buf_,
        vk_src_buf->get(),
        vk_dst_buf->get(),
        static_cast<uint32_t>(vk_copy_regions.size()),
        vk_copy_regions.data());
}

void VulkanCommandBuffer::copyImage(
    std::shared_ptr<Image> src_img,
    ImageLayout src_img_layout,
    std::shared_ptr<Image> dst_img,
    ImageLayout dst_img_layout,
    std::vector<ImageCopyInfo> copy_regions) {
    std::vector<VkImageCopy> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i] = helper::toVkImageCopy(copy_regions[i]);
    }
    auto vk_src_img = RENDER_TYPE_CAST(Image, src_img);
    auto vk_dst_img = RENDER_TYPE_CAST(Image, dst_img);
    vkCmdCopyImage(
        cmd_buf_,
        vk_src_img->get(),
        helper::toVkImageLayout(src_img_layout),
        vk_dst_img->get(),
        helper::toVkImageLayout(dst_img_layout),
        static_cast<uint32_t>(vk_copy_regions.size()),
        vk_copy_regions.data());
}

void VulkanCommandBuffer::blitImage(
    std::shared_ptr<Image> src_img,
    ImageLayout src_img_layout,
    std::shared_ptr<Image> dst_img,
    ImageLayout dst_img_layout,
    std::vector<ImageBlitInfo> copy_regions,
    const Filter& filter) {
    std::vector<VkImageBlit> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i] = helper::toVkImageBlit(copy_regions[i]);
    }
    auto vk_src_img = RENDER_TYPE_CAST(Image, src_img);
    auto vk_dst_img = RENDER_TYPE_CAST(Image, dst_img);
    vkCmdBlitImage(
        cmd_buf_,
        vk_src_img->get(),
        helper::toVkImageLayout(src_img_layout),
        vk_dst_img->get(),
        helper::toVkImageLayout(dst_img_layout),
        static_cast<uint32_t>(vk_copy_regions.size()),
        vk_copy_regions.data(),
        helper::toVkFilter(filter));
}

void VulkanCommandBuffer::resolveImage(
    std::shared_ptr<Image> src_img,
    ImageLayout src_img_layout,
    std::shared_ptr<Image> dst_img,
    ImageLayout dst_img_layout,
    std::vector<ImageResolveInfo> copy_regions) {
    std::vector<VkImageResolve> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i] = helper::toVkImageResolve(copy_regions[i]);
    }
    auto vk_src_img = RENDER_TYPE_CAST(Image, src_img);
    auto vk_dst_img = RENDER_TYPE_CAST(Image, dst_img);
    vkCmdResolveImage(
        cmd_buf_,
        vk_src_img->get(),
        helper::toVkImageLayout(src_img_layout),
        vk_dst_img->get(),
        helper::toVkImageLayout(dst_img_layout),
        static_cast<uint32_t>(vk_copy_regions.size()),
        vk_copy_regions.data());
}

void VulkanCommandBuffer::copyBufferToImage(
    std::shared_ptr<Buffer> src_buf,
    std::shared_ptr<Image> dst_image,
    std::vector<BufferImageCopyInfo> copy_regions,
    ImageLayout layout) {
    std::vector<VkBufferImageCopy> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i] = helper::toVkBufferImageCopy(copy_regions[i]);
    }
    auto vk_src_buf = RENDER_TYPE_CAST(Buffer, src_buf);
    auto vk_dst_image = RENDER_TYPE_CAST(Image, dst_image);
    vkCmdCopyBufferToImage(cmd_buf_, vk_src_buf->get(), vk_dst_image->get(), helper::toVkImageLayout(layout), static_cast<uint32_t>(vk_copy_regions.size()), vk_copy_regions.data());
}

void VulkanCommandBuffer::copyImageToBuffer(
    std::shared_ptr<Image> src_image,
    std::shared_ptr<Buffer> dst_buf,
    std::vector<BufferImageCopyInfo> copy_regions,
    ImageLayout layout) {
    std::vector<VkBufferImageCopy> vk_copy_regions(copy_regions.size());
    for (uint32_t i = 0; i < copy_regions.size(); i++) {
        vk_copy_regions[i] = helper::toVkBufferImageCopy(copy_regions[i]);
    }
    auto vk_src_image = RENDER_TYPE_CAST(Image, src_image);
    auto vk_dst_buf = RENDER_TYPE_CAST(Buffer, dst_buf);
    vkCmdCopyImageToBuffer(cmd_buf_, vk_src_image->get(), helper::toVkImageLayout(layout), vk_dst_buf->get(), static_cast<uint32_t>(vk_copy_regions.size()), vk_copy_regions.data());
}

void VulkanCommandBuffer::bindPipeline(
    PipelineBindPoint bind,
    std::shared_ptr< Pipeline> pipeline) {
    auto vk_pipeline = RENDER_TYPE_CAST(Pipeline, pipeline);
    vkCmdBindPipeline(cmd_buf_, helper::toVkPipelineBindPoint(bind), vk_pipeline->get());
}

void VulkanCommandBuffer::bindVertexBuffers(
    uint32_t first_bind,
    const std::vector<std::shared_ptr<Buffer>>& vertex_buffers,
    std::vector<uint64_t> offsets) {
    std::vector<VkDeviceSize> vk_offsets(vertex_buffers.size());
    std::vector<VkBuffer> vk_vertex_buffers(vertex_buffers.size());

    for (int i = 0; i < vertex_buffers.size(); i++) {
        auto vk_vertex_buffer = RENDER_TYPE_CAST(Buffer, vertex_buffers[i]);
        vk_vertex_buffers[i] = vk_vertex_buffer->get();
        vk_offsets[i] = i < offsets.size() ? offsets[i] : 0;
    }
    vkCmdBindVertexBuffers(cmd_buf_, first_bind, static_cast<uint32_t>(vk_vertex_buffers.size()), vk_vertex_buffers.data(), vk_offsets.data());
}

void VulkanCommandBuffer::bindIndexBuffer(
    std::shared_ptr<Buffer> index_buffer,
    uint64_t offset,
    IndexType index_type) {
    auto vk_index_buffer = RENDER_TYPE_CAST(Buffer, index_buffer);
    vkCmdBindIndexBuffer(cmd_buf_, vk_index_buffer->get(), offset, helper::toVkIndexType(index_type));
}

void VulkanCommandBuffer::bindDescriptorSets(
    PipelineBindPoint bind_point,
    const std::shared_ptr<PipelineLayout>& pipeline_layout,
    const DescriptorSetList& desc_sets,
    const uint32_t first_set_idx/* = 0 */ ) {
    std::vector<VkDescriptorSet> vk_desc_sets;
    auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    for (auto i = 0; i < desc_sets.size(); i++) {
        vk_desc_sets.push_back(RENDER_TYPE_CAST(DescriptorSet, desc_sets[i])->get());
    }
    vkCmdBindDescriptorSets(
        cmd_buf_,
        helper::toVkPipelineBindPoint(bind_point),
        vk_pipeline_layout->get(),
        first_set_idx,
        static_cast<uint32_t>(vk_desc_sets.size()),
        vk_desc_sets.data(),
        0,
        nullptr);
}

void VulkanCommandBuffer::pushConstants(
    ShaderStageFlags stages,
    const std::shared_ptr<PipelineLayout>& pipeline_layout,
    const void* data,
    uint32_t size,
    uint32_t offset/* = 0*/) {
    auto vk_pipeline_layout = RENDER_TYPE_CAST(PipelineLayout, pipeline_layout);
    vkCmdPushConstants(
        cmd_buf_,
        vk_pipeline_layout->get(),
        helper::toVkShaderStageFlags(stages),
        offset,
        size,
        data);
}

void VulkanCommandBuffer::draw(
    uint32_t vertex_count,
    uint32_t instance_count/* = 1*/,
    uint32_t first_vertex/* = 0*/,
    uint32_t first_instance/* = 0*/) {
    vkCmdDraw(cmd_buf_, vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanCommandBuffer::drawIndexed(
    uint32_t index_count,
    uint32_t instance_count/* = 1*/,
    uint32_t first_index/* = 0*/,
    uint32_t vertex_offset/* = 0*/,
    uint32_t first_instance/* = 0*/) {
    vkCmdDrawIndexed(cmd_buf_, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void VulkanCommandBuffer::drawIndexedIndirect(
    const renderer::BufferInfo& indirect_draw_cmd_buf,
    uint32_t buffer_offset/* = 0*/,
    uint32_t draw_count/* = 1*/,
    uint32_t stride/* = sizeof(DrawIndexedIndirectCommand)*/) {
    auto vk_indirect_buffer = RENDER_TYPE_CAST(Buffer, indirect_draw_cmd_buf.buffer);
    vkCmdDrawIndexedIndirect(cmd_buf_, vk_indirect_buffer->get(), buffer_offset, draw_count, stride);
}

void VulkanCommandBuffer::drawIndirect(
    const renderer::BufferInfo& indirect_draw_cmd_buf,
    uint32_t buffer_offset/* = 0*/,
    uint32_t draw_count/* = 1*/,
    uint32_t stride/* = sizeof(DrawIndirectCommand)*/) {
    auto vk_indirect_buffer = RENDER_TYPE_CAST(Buffer, indirect_draw_cmd_buf.buffer);
    vkCmdDrawIndirect(cmd_buf_, vk_indirect_buffer->get(), buffer_offset, draw_count, stride);
}

void VulkanCommandBuffer::drawMeshTasks(
    uint32_t group_count_x/* = 1*/,
    uint32_t group_count_y/* = 1*/,
    uint32_t group_count_z/* = 1*/) {
    vkCmdDrawMeshTasksEXT(
        cmd_buf_,
        group_count_x,
        group_count_y,
        group_count_z);
}

void VulkanCommandBuffer::drawMeshTasksIndirect() {

}

void VulkanCommandBuffer::drawMeshTasksIndirectCount() {

}

void VulkanCommandBuffer::dispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z/* = 1*/) {
    vkCmdDispatch(cmd_buf_, group_count_x, group_count_y, group_count_z);
}

void VulkanCommandBuffer::traceRays(
    const StridedDeviceAddressRegion& raygen_shader_entry,
    const StridedDeviceAddressRegion& miss_shader_entry,
    const StridedDeviceAddressRegion& hit_shader_entry,
    const StridedDeviceAddressRegion& callable_shader_entry,
    const glm::uvec3& size) {
    auto vk_raygen_shader_entry = helper::toVkStridedDeviceAddressRegion(raygen_shader_entry);
    auto vk_miss_shader_entry = helper::toVkStridedDeviceAddressRegion(miss_shader_entry);
    auto vk_hit_shader_entry = helper::toVkStridedDeviceAddressRegion(hit_shader_entry);
    auto vk_callable_shader_entry = helper::toVkStridedDeviceAddressRegion(callable_shader_entry);
    
    vkCmdTraceRaysKHR(
        cmd_buf_,
        &vk_raygen_shader_entry,
        &vk_miss_shader_entry,
        &vk_hit_shader_entry,
        &vk_callable_shader_entry,
        size.x,
        size.y,
        size.z);
}

void VulkanCommandBuffer::beginRenderPass(
    std::shared_ptr<RenderPass> render_pass,
    std::shared_ptr<Framebuffer> frame_buffer,
    const glm::uvec2& extent,
    const std::vector<ClearValue>& clear_values) {
    std::vector<VkClearValue> vk_clear_values(clear_values.size());

    for (int i = 0; i < clear_values.size(); i++) {
        std::memcpy(&vk_clear_values[i].color, &clear_values[i].color, sizeof(VkClearValue));
    }

    auto vk_render_pass = RENDER_TYPE_CAST(RenderPass, render_pass);
    auto vk_frame_buffer = RENDER_TYPE_CAST(Framebuffer, frame_buffer);

    assert(vk_render_pass);
    assert(vk_frame_buffer);
    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = vk_render_pass->get();
    render_pass_info.framebuffer = vk_frame_buffer->get();
    render_pass_info.renderArea.offset = { 0, 0 };
    render_pass_info.renderArea.extent = { extent.x, extent.y };
    render_pass_info.clearValueCount = static_cast<uint32_t>(vk_clear_values.size());
    render_pass_info.pClearValues = vk_clear_values.data();

    vkCmdBeginRenderPass(cmd_buf_, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
}

void VulkanCommandBuffer::endRenderPass() {
    vkCmdEndRenderPass(cmd_buf_);
}

void VulkanCommandBuffer::reset(uint32_t flags) {
    auto result =
        vkResetCommandBuffer(
            cmd_buf_,
            flags);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("reset command buffer error : ") +
            VkResultToString(result));
    }
}

void VulkanCommandBuffer::addBarriers(
    const BarrierList& barrier_list,
    PipelineStageFlags src_stage_flags,
    PipelineStageFlags dst_stage_flags) {
    auto memory_barrier_count = barrier_list.memory_barriers.size();
    std::vector<VkMemoryBarrier> memory_barriers(memory_barrier_count);
    for (auto i = 0; i < memory_barrier_count; i++) {
        memory_barriers[i] =
            helper::toVkMemoryBarrier(barrier_list.memory_barriers[i]);
    }

    auto buffer_barrier_count = barrier_list.buffer_barriers.size();
    std::vector<VkBufferMemoryBarrier> buffer_barriers(buffer_barrier_count);
    for (auto i = 0; i < buffer_barrier_count; i++) {
        buffer_barriers[i] =
            helper::toVkBufferMemoryBarrier(barrier_list.buffer_barriers[i]);
    }

    auto image_barrier_count = barrier_list.image_barriers.size();
    std::vector<VkImageMemoryBarrier> image_barriers(image_barrier_count);
    for (auto i = 0; i < image_barrier_count; i++) {
        image_barriers[i] =
            helper::toVkImageMemoryBarrier(barrier_list.image_barriers[i]);
    }

    vkCmdPipelineBarrier(
        cmd_buf_,
        helper::toVkPipelineStageFlags(src_stage_flags),
        helper::toVkPipelineStageFlags(dst_stage_flags),
        0,
        static_cast<uint32_t>(memory_barrier_count), memory_barriers.data(),
        static_cast<uint32_t>(buffer_barrier_count), buffer_barriers.data(),
        static_cast<uint32_t>(image_barrier_count), image_barriers.data()
    );
}

void VulkanCommandBuffer::addImageBarrier(
    const std::shared_ptr<Image>& image,
    const ImageResourceInfo& src_info,
    const ImageResourceInfo& dst_info,
    uint32_t base_mip/* = 0*/,
    uint32_t mip_count/* = 1*/,
    uint32_t base_layer/* = 0*/,
    uint32_t layer_count/* = 1*/) {
    auto vk_image = RENDER_TYPE_CAST(Image, image);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = helper::toVkImageLayout(src_info.image_layout);
    barrier.newLayout = helper::toVkImageLayout(dst_info.image_layout);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = vk_image->get();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = base_mip;
    barrier.subresourceRange.levelCount = mip_count;
    barrier.subresourceRange.baseArrayLayer = base_layer;
    barrier.subresourceRange.layerCount = layer_count;
    barrier.srcAccessMask = helper::toVkAccessFlags(src_info.access_flags);
    barrier.dstAccessMask = helper::toVkAccessFlags(dst_info.access_flags);

    vkCmdPipelineBarrier(
        cmd_buf_,
        helper::toVkPipelineStageFlags(src_info.stage_flags),
        helper::toVkPipelineStageFlags(dst_info.stage_flags),
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void VulkanCommandBuffer::addBufferBarrier(
    const std::shared_ptr<Buffer>& buffer,
    const BufferResourceInfo& src_info,
    const BufferResourceInfo& dst_info,
    uint32_t size/* = 0*/,
    uint32_t offset/* = 0*/) {
    auto vk_buffer = RENDER_TYPE_CAST(Buffer, buffer);

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = helper::toVkAccessFlags(src_info.access_flags);
    barrier.dstAccessMask = helper::toVkAccessFlags(dst_info.access_flags);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = vk_buffer->get();
    barrier.offset = offset;
    barrier.size = size;

    vkCmdPipelineBarrier(
        cmd_buf_,
        helper::toVkPipelineStageFlags(src_info.stage_flags),
        helper::toVkPipelineStageFlags(dst_info.stage_flags),
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );
}

void VulkanCommandBuffer::buildAccelerationStructures(
    const std::vector<AccelerationStructureBuildGeometryInfo>& as_build_geo_list,
    const std::vector<AccelerationStructureBuildRangeInfo>& as_build_range_list) {

    std::vector<std::unique_ptr<VkAccelerationStructureGeometryKHR[]>> geoms(as_build_geo_list.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> vk_geoms(as_build_geo_list.size());
    for (auto i = 0; i < as_build_geo_list.size(); i++) {
        vk_geoms[i] = helper::toVkAccelerationStructureBuildGeometryInfo(as_build_geo_list[i], geoms[i]);
    }

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> vk_as_build_range_list(as_build_range_list.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> vk_as_build_range_ptr_list(as_build_range_list.size());
    for (auto i = 0; i < as_build_range_list.size(); i++) {
        vk_as_build_range_list[i] = helper::toVkAccelerationStructureBuildRangeInfo(as_build_range_list[i]);
        vk_as_build_range_ptr_list[i] = &vk_as_build_range_list[i];
    }
    
    vkCmdBuildAccelerationStructuresKHR(
        cmd_buf_,
        static_cast<uint32_t>(as_build_geo_list.size()),
        vk_geoms.data(),
        vk_as_build_range_ptr_list.data());
}

} // namespace vk
} // namespace renderer
} // namespace engine