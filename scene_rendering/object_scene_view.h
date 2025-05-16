#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/terrain.h"
#include "game_object/drawable_object.h"
#include "game_object/patch.h"
#include "game_object/sphere.h"
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
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
        const std::shared_ptr<ego::CameraObject>& camera_object,
        const std::shared_ptr<er::TextureInfo>& color_buffer = nullptr,
        const std::shared_ptr<er::TextureInfo>& depth_buffer = nullptr,
        const glm::uvec2& buffer_size = glm::uvec2(2560, 1440),
        bool depth_only = false);

    void addDrawableObject(
        const std::shared_ptr<ego::DrawableObject>& drawable_object) {
        m_drawable_objects_.push_back(drawable_object);
    }

    void duplicateColorAndDepthBuffer(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    virtual void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_sets,
        std::shared_ptr<ego::Sphere> sphere,
        int dbuf_idx,
        float delta_t,
        float cur_time,
        bool depth_only = false);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine