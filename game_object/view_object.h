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
protected:
    glsl::ViewCameraParams view_camera_params_;
    std::shared_ptr<ego::ViewCamera> view_camera_;
    const std::shared_ptr<renderer::Device>& device_;
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool_;
    std::shared_ptr<er::DescriptorSet> view_camera_desc_set_;
    std::shared_ptr<er::DescriptorSetLayout> view_camera_desc_set_layout_;

    er::Format hdr_format_ = er::Format::B10G11R11_UFLOAT_PACK32;
    er::Format depth_format_ = er::Format::D24_UNORM_S8_UINT;
    glm::uvec2 buffer_size_ = glm::uvec2(1280, 720);
    glm::vec3 camera_pos_ = glm::vec3(0.0f, 0.0f, 0.0f);

    er::TextureInfo hdr_color_buffer_;
    er::TextureInfo hdr_color_buffer_copy_;
    er::TextureInfo depth_buffer_;
    er::TextureInfo depth_buffer_copy_;

public:
    ViewObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool);

    void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        int dbuf_idx,
        float delta_t,
        float cur_time);

    std::shared_ptr<er::DescriptorSetLayout>
        getViewCameraDescriptorSetLayout() {
        return view_camera_desc_set_layout_;
    }

    std::shared_ptr<er::DescriptorSet>
        getViewCameraDescriptorSet() {
        return view_camera_desc_set_;
    }

    std::shared_ptr<er::BufferInfo> getViewCameraBuffer() {
        return view_camera_->getViewCameraBuffer();
    }

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine