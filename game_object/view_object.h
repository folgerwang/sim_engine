#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/terrain.h"
#include "game_object/drawable_object.h"
#include "patch.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace game_object {

extern glm::vec3 getDirectionByYawAndPitch(float yaw, float pitch);

class ViewObject {
public:
    static std::shared_ptr<er::DescriptorSetLayout>
        s_view_camera_desc_set_layout_;

protected:
    const std::shared_ptr<renderer::Device>& m_device_;
    const std::shared_ptr<er::DescriptorPool>& m_descriptor_pool_;
    glsl::ViewCameraParams m_view_camera_params_;
    std::shared_ptr<ego::ViewCamera> m_view_camera_;
    std::shared_ptr<er::DescriptorSet> m_view_camera_desc_set_;
    std::vector<std::shared_ptr<renderer::DescriptorSet>> m_tile_res_desc_sets_;
    glsl::ViewCameraInfo m_gpu_game_camera_info_;

    er::Format m_color_format_ = er::Format::B10G11R11_UFLOAT_PACK32;
    er::Format m_depth_format_ = er::Format::D24_UNORM_S8_UINT;
    glm::uvec2 m_buffer_size_ = glm::uvec2(1280, 720);
    glm::vec3 m_camera_pos_ = glm::vec3(0.0f, 0.0f, 0.0f);
    std::vector<er::ClearValue> m_clear_values_;

    bool m_make_color_buffer_copy_ = true;
    bool m_make_depth_buffer_copy_ = true;

    std::shared_ptr<er::TextureInfo> m_color_buffer_;
    std::shared_ptr<er::TextureInfo> m_color_buffer_copy_;
    std::shared_ptr<er::TextureInfo> m_depth_buffer_;
    std::shared_ptr<er::TextureInfo> m_depth_buffer_copy_;
    std::shared_ptr<er::Framebuffer> m_frame_buffer_;
    std::shared_ptr<er::Framebuffer> m_blend_frame_buffer_;
    std::shared_ptr<er::RenderPass> m_render_pass_;
    std::shared_ptr<er::RenderPass> m_blend_render_pass_;

public:
    ViewObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<er::TextureInfo>& color_buffer,
        const std::shared_ptr<er::TextureInfo>& depth_buffer);

    static std::shared_ptr<er::DescriptorSetLayout>
        getViewCameraDescriptorSetLayout();

    static void createViewCameraDescriptorSetLayout(
        const std::shared_ptr<er::Device>& device);

    void AllocRenderBuffers();

    virtual void readCameraInfo();

    glm::vec3 getCameraPosition() const {
        return m_gpu_game_camera_info_.position;
    }

    void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    void resize(const glm::uvec2& new_buffer_size);

    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        int dbuf_idx,
        float delta_t,
        float cur_time);

    std::shared_ptr<er::DescriptorSet>
        getViewCameraDescriptorSet() {
        return m_view_camera_desc_set_;
    }

    std::shared_ptr<er::BufferInfo> getViewCameraBuffer() {
        return m_view_camera_->getViewCameraBuffer();
    }

    std::shared_ptr<er::TextureInfo> getColorBuffer() const {
        return m_color_buffer_;
    }

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine