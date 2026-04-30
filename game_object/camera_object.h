#pragma once
#include <array>
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

    // Re-allocate the view-camera descriptor set from the new pool.
    void recreateDescriptorSet();

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

    virtual const glsl::ViewCameraInfo& getCameraViewInfo() const {
        return m_view_camera_->getCameraInfo();
    }

    void setInputFeatureFlags(uint32_t flags) {
        m_view_camera_->setInputFeatureFlags(flags);
    }


    virtual void destroy(
        const std::shared_ptr<renderer::Device>& device);

    static void destroyStaticMembers(
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
private:
    // Per-cascade GPU storage buffers and their descriptor sets.
    // Each buffer holds one ViewCameraInfo with the cascade-specific VP matrix.
    std::array<std::shared_ptr<er::BufferInfo>, CSM_CASCADE_COUNT>   m_cascade_bufs_{};
    std::array<std::shared_ptr<er::DescriptorSet>, CSM_CASCADE_COUNT> m_cascade_desc_sets_{};
    int m_current_cascade_ = -1;

public:
    ShadowViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const glm::vec3& light_dir);

    const glm::vec3& getLightDir() {
        return m_view_camera_params_.init_camera_dir;
    }

    // Allocate the 4 per-cascade storage buffers and descriptor sets.
    // Must be called once after construction, before the first shadow pass.
    void initCascadeDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Re-allocate descriptor sets from a new pool after swap chain
    // recreation.  Keeps the existing cascade buffers.
    void recreateCascadeDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    // Write ViewCameraInfo for cascade k into its dedicated GPU buffer.
    void updateCascadeBuffer(int k, const glm::mat4& view_proj);

    // Select which cascade's descriptor set getViewCameraDescriptorSet() returns.
    void setCascadeIndex(int k) { m_current_cascade_ = k; }

    // Override: returns the per-cascade descriptor set when a cascade is active,
    // otherwise falls back to the base single-buffer descriptor set.
    virtual std::shared_ptr<er::DescriptorSet>
        getViewCameraDescriptorSet() const override;

    // Override: also destroys the per-cascade buffers.
    virtual void destroy(
        const std::shared_ptr<renderer::Device>& device) override;

    // Compute per-cascade light-space view-projection matrices.
    // main_view / main_proj: perspective matrices from the main camera.
    // cascade_far_vs: view-space (positive) far depths for each of the
    //   CSM_CASCADE_COUNT cascades.  z_near_vs is the main camera near plane.
    // out_vps: receives the CSM_CASCADE_COUNT orthographic view-proj matrices.
    void computeCascadeMatrices(
        const glm::mat4& main_view,
        const glm::mat4& main_proj,
        const glm::vec4& cascade_far_vs,
        float z_near_vs,
        std::array<glm::mat4, 4>& out_vps);

    // Push a specific cascade view-projection matrix into the shadow camera's
    // GPU-side ViewCameraInfo buffer so that base_depthonly.vert uses it.
    void updateCameraForCascade(const glm::mat4& cascade_view_proj);
};

} // game_object
} // engine