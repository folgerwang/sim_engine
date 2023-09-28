#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class ViewCapture {
    bool has_color_copy_ = false;
    bool has_depth_copy_ = false;

    renderer::Format color_format_ =
        renderer::Format::B10G11R11_UFLOAT_PACK32;
    renderer::Format depth_format_ =
        renderer::Format::D24_UNORM_S8_UINT;

    renderer::TextureInfo color_buffer_;
    renderer::TextureInfo color_buffer_copy_;
    renderer::TextureInfo depth_buffer_;
    renderer::TextureInfo depth_buffer_copy_;

    std::shared_ptr<renderer::Framebuffer> frame_buffer_;
    std::shared_ptr<renderer::Framebuffer> translucent_frame_buffer_;

    std::shared_ptr<renderer::RenderPass> render_pass_;
    std::shared_ptr<renderer::RenderPass> translucent_render_pass_;

public:
    ViewCapture(
        const std::shared_ptr<renderer::Device>& device,
        const glm::uvec2& view_size);

    void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set);

    void update();

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
