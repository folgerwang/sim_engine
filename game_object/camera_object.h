#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "game_object/drawable_object.h"
#include "patch.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace game_object {

extern glm::vec3 getDirectionByYawAndPitch(float yaw, float pitch);

class CameraObject {
public:
    static std::shared_ptr<er::DescriptorSetLayout>
        s_view_camera_desc_set_layout_;

protected:
    const std::shared_ptr<renderer::Device>& m_device_;
    const std::shared_ptr<er::DescriptorPool>& m_descriptor_pool_;

    glsl::ViewCameraParams m_view_camera_params_;
    std::shared_ptr<ego::ViewCamera> m_view_camera_;
    std::shared_ptr<er::DescriptorSet> m_view_camera_desc_set_;
    bool m_is_ortho_ = false;

public:
    CameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        bool is_ortho);

    virtual ~CameraObject() = default;

    static std::shared_ptr<er::DescriptorSetLayout>
        getViewCameraDescriptorSetLayout();

    static void createViewCameraDescriptorSetLayout(
        const std::shared_ptr<er::Device>& device);

    virtual void readGpuCameraInfo();

    virtual glm::vec3 getCameraPosition() const {
        return m_view_camera_->getCameraInfo().position;
    }

    virtual void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const uint32_t& dbuf_idx,
        const int& input_key,
        const int& frame_count,
        const float& delta_t,
        const glm::vec2& last_mouse_pos,
        const float& mouse_wheel_offset,
        const bool& camera_rot_update);

    virtual void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const glm::vec3& camera_pos);

    void createCameraDescSetWithTerrain(
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::BufferInfo& game_objects_buffer);

    virtual std::shared_ptr<er::DescriptorSet>
        getViewCameraDescriptorSet() const{
        return m_view_camera_desc_set_;
    }

    virtual std::shared_ptr<er::BufferInfo> getViewCameraBuffer() {
        return m_view_camera_->getViewCameraBuffer();
    }

    virtual glm::mat4 getViewProjMatrix() {
        return m_view_camera_->getCameraInfo().view_proj;
    }

    virtual glm::vec3 getCameraPos() {
        return m_view_camera_->getCameraInfo().position;
    }


    virtual void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

class ObjectViewCameraObject : public CameraObject{
public:
    ObjectViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        float fov,
        float aspect);
};

class TerrainViewCameraObject : public CameraObject{
public:
    TerrainViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        float fov,
        float aspect);
};

class ShadowViewCameraObject : public CameraObject {
public:
    ShadowViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const glm::vec3& light_dir);

    const glm::vec3& getLightDir() {
        return m_view_camera_params_.init_camera_dir;
    }
};

} // game_object
} // engine