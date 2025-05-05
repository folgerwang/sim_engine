#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/terrain.h"
#include "game_object/drawable_object.h"
#include "game_object/camera_object.h"
#include "patch.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace game_object {

class ViewObject {

protected:
    const std::shared_ptr<er::Device>& m_device_;
    const std::shared_ptr<er::DescriptorPool>& m_descriptor_pool_;
    std::shared_ptr<CameraObject> m_camera_object_;
    std::vector<std::shared_ptr<er::DescriptorSet>> m_tile_res_desc_sets_;

    glm::uvec2 m_buffer_size_ = glm::uvec2(2560, 1440);
    std::vector<er::ClearValue> m_clear_values_;

    bool m_depth_only_ = false;
    bool m_make_color_buffer_copy_ = true;
    bool m_make_depth_buffer_copy_ = true;

    std::shared_ptr<er::TextureInfo> m_color_buffer_;
    std::shared_ptr<er::TextureInfo> m_color_buffer_copy_;
    std::shared_ptr<er::TextureInfo> m_depth_buffer_;
    std::shared_ptr<er::TextureInfo> m_depth_buffer_copy_;

public:
    ViewObject(
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
        const std::shared_ptr<CameraObject>& camera_object,
        const std::shared_ptr<er::TextureInfo>& color_buffer,
        const std::shared_ptr<er::TextureInfo>& depth_buffer,
        const glm::uvec2& buffer_size = glm::uvec2(2560, 1440),
        bool depth_only = false);

    void AllocRenderBuffers(
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats);

    void resize(
        const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
        const glm::uvec2& new_buffer_size);

    virtual void draw(
        std::shared_ptr<er::CommandBuffer> cmd_buf,
        const std::shared_ptr<er::DescriptorSet>& prt_desc_set,
        int dbuf_idx,
        float delta_t,
        float cur_time,
        bool depth_only = false) {
    }

    const std::shared_ptr<CameraObject>& getCameraObject() {
        return m_camera_object_;
    }

    std::shared_ptr<er::TextureInfo> getColorBuffer() const {
        return m_color_buffer_;
    }

    void destroy(
        const std::shared_ptr<er::Device>& device);
};

} // game_object
} // engine