#include <iostream>
#include <vector>
#include <chrono>

#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "view_capture.h"

namespace {
namespace er = engine::renderer;

} // namespace

namespace engine {
namespace scene_rendering {

ViewCapture::ViewCapture(
    const std::shared_ptr<renderer::Device>& device,
    const glm::uvec2& view_size) {

    // create color buffer.
    renderer::Helper::create2DTextureImage(
        device,
        color_format_,
        view_size,
        1,
        color_buffer_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT) |
        SET_FLAG_BIT(ImageUsage, COLOR_ATTACHMENT_BIT) |
        SET_FLAG_BIT(ImageUsage, TRANSFER_SRC_BIT),
        renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());

    if (has_color_copy_) {
        renderer::Helper::create2DTextureImage(
            device,
            color_format_,
            view_size,
            1,
            color_buffer_copy_,
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    // create depth buffer.
    renderer::Helper::createDepthResources(
        device,
        depth_format_,
        view_size,
        depth_buffer_,
        std::source_location::current());

    if (has_depth_copy_) {
        renderer::Helper::create2DTextureImage(
            device,
            renderer::Format::D32_SFLOAT,
            view_size,
            1,
            depth_buffer_copy_,
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, DEPTH_STENCIL_ATTACHMENT_BIT) |
            SET_FLAG_BIT(ImageUsage, TRANSFER_DST_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    // create render pass.
    render_pass_ =
        renderer::helper::createRenderPass(
            device,
            color_format_,
            depth_format_,
            std::source_location::current(),
            true);

    translucent_render_pass_ =
        renderer::helper::createRenderPass(
            device,
            color_format_,
            depth_format_,
            std::source_location::current(),
            false);

    // create frame buffer.
    assert(depth_buffer_.view);
    assert(render_pass_);
    std::vector<std::shared_ptr<er::ImageView>> attachments(2);
    attachments[0] = color_buffer_.view;
    attachments[1] = depth_buffer_.view;

    frame_buffer_ =
        device->createFrameBuffer(
            render_pass_,
            attachments,
            view_size,
            std::source_location::current());

    assert(translucent_render_pass_);
    translucent_frame_buffer_ =
        device->createFrameBuffer(
            translucent_render_pass_,
            attachments,
            view_size,
            std::source_location::current());
}

void ViewCapture::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set) {

}

void ViewCapture::update() {
}

void ViewCapture::destroy(const std::shared_ptr<renderer::Device>& device) {

}

}//namespace scene_rendering
}//namespace engine
