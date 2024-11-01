#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/terrain.h"
#include "game_object/drawable_object.h"
#include "game_object/patch.h"
#include "game_object/view_object.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace scene_rendering {

class ObjectSceneView : public ego::ViewObject {
    std::vector<std::shared_ptr<ego::DrawableObject>> m_drawable_objects_;

    bool m_b_render_blend_ = false;

public:
    ObjectSceneView(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<er::TextureInfo>& color_buffer/* = nullptr*/,
        const std::shared_ptr<er::TextureInfo>& depth_buffer/* = nullptr*/);

    void duplicateColorAndDepthBuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const uint32_t& dbuf_idx,
        const int& input_key,
        const int& frame_count,
        const float& delta_t,
        const glm::vec2& last_mouse_pos,
        const float& mouse_wheel_offset,
        const bool& camera_rot_update);

    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        int dbuf_idx,
        float delta_t,
        float cur_time);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine