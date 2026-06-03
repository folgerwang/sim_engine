#pragma once
#include "renderer/renderer.h"
#include "scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace ui {

class ChatBox {

public:
    ChatBox() {}

    bool draw(
        const std::shared_ptr<renderer::CommandBuffer>& command_buffer,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::Framebuffer>& framebuffer,
        const glm::uvec2& screen_size,
        const std::shared_ptr<scene_rendering::Skydome>& skydome,
        bool& dump_volume_noise,
        const float& delta_t,
        // Screen-space rect the dialogue lives in: the editor Viewport when in
        // editor mode, else the full window.  (0,0)+0 means "use the main
        // viewport / DisplaySize" (default / non-editor).
        const glm::vec2& vp_origin = glm::vec2(0.0f),
        const glm::vec2& vp_size   = glm::vec2(0.0f));

    void destroy();
};

}// namespace ui
}// namespace engine
