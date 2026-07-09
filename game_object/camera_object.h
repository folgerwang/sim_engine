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

    // Follow-camera overload: override BOTH the camera position and its view
    // (facing) direction.  Used by the third-person follow rig to place the
    // eye behind/below the character and aim the view up along her back.
    virtual void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const glm::vec3& camera_pos,
        const glm::vec3& facing_dir);

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

    // Override the projection aspect ratio (width/height).  Used by the editor
    // so the 3D framing matches the Viewport panel rather than the full window.
    void setAspect(float aspect) {
        if (aspect > 0.0f) m_view_camera_params_.aspect = aspect;
    }

    virtual glm::vec3 getCameraPos() {
        return m_view_camera_->getCameraInfo().position;
    }

    virtual const glsl::ViewCameraInfo& getCameraViewInfo() const {
        return m_view_camera_->getCameraInfo();
    }

    // Sync the mouse-look yaw/pitch to an explicit view direction — used
    // by the editor's "view through scene camera" snap so the per-frame
    // camera update (which derives facing from yaw/pitch) keeps the pose
    // instead of reverting to the previous mouse state.
    void setViewDirection(const glm::vec3& dir) {
        const glm::vec3 d = glm::normalize(dir);
        const float pitch = glm::degrees(glm::asin(
            glm::clamp(d.y, -1.0f, 1.0f)));
        // getDirectionByYawAndPitch: x=cos(-yaw)cosP, z=sin(-yaw)cosP.
        const float yaw = -glm::degrees(glm::atan(d.z, d.x));
        m_view_camera_->setYawPitch(yaw, pitch);
    }

    void setInputFeatureFlags(uint32_t flags) {
        m_view_camera_->setInputFeatureFlags(flags);
    }

    // Stage the FEATURE_INPUT_ISOLATE_MESH target material_idx.  Call
    // BEFORE setInputFeatureFlags (which re-uploads the camera struct).
    void setDebugIsolateMaterial(uint32_t material_idx) {
        m_view_camera_->setDebugIsolateMaterial(material_idx);
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
public:
    ShadowViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const glm::vec3& light_dir);

    const glm::vec3& getLightDir() const {
        return m_view_camera_params_.init_camera_dir;
    }

    // Update the light direction each frame (e.g. from Skydome::getSunDir()).
    // dir must be a unit vector pointing FROM the sun TOWARD the scene
    // (i.e. negate Skydome::getSunDir() which points ground→sun).
    void setLightDir(const glm::vec3& dir) {
        m_view_camera_params_.init_camera_dir = dir;
    }

    // Compute per-cascade light-space view-projection matrices.
    // main_view / main_proj: perspective matrices from the main camera.
    // cascade_far_vs: view-space (positive) far depths for each of the
    //   CSM_CASCADE_COUNT cascades.  z_near_vs is the main camera near plane.
    // out_vps: receives the CSM_CASCADE_COUNT orthographic view-proj matrices.
    //
    // The output matrices flow into RuntimeLightsParams::light_view_proj[] in
    // application.cpp and are consumed by the single-pass CSM geometry shader
    // (base_depthonly_csm.geom) via geometry-shader layer broadcast.  There is
    // no per-cascade GPU buffer / descriptor set on the production path.
    void computeCascadeMatrices(
        const glm::mat4& main_view,
        const glm::mat4& main_proj,
        const std::array<float, CSM_CASCADE_COUNT>& cascade_far_vs,
        float z_near_vs,
        std::array<glm::mat4, CSM_CASCADE_COUNT>& out_vps,
        // Optional out param: a single "union" light-space view-projection
        // matrix that covers all CSM cascade volumes — i.e. a light-space
        // orthographic projection fitted to the full main-camera frustum
        // from z_near_vs to cascade_far_vs.back(), with the same 200 m
        // pull-back along the light direction as the per-cascade VPs.
        // Used by ClusterRenderer::cullShadow to do a single shadow-frustum
        // cull dispatch whose survivor set covers every cascade.  Pass
        // nullptr if not needed (legacy callers).
        glm::mat4* out_union_vp = nullptr,
        // Optional out param: world-space corners of each cascade's
        // main-camera frustum slab.  Layout: [cascade * 8 + corner],
        // where corner indices match the vs_corners order inside
        // computeCascadeMatrices (near TL/TR/BL/BR, far TL/TR/BL/BR).
        // Used by csm_silhouette_prepass.mesh to fill the in-frustum
        // region of each cascade's shadow map with depth=1.  Pass
        // nullptr if not needed.
        std::array<glm::vec3, CSM_CASCADE_COUNT * 8>* out_slab_corners_ws = nullptr);
};

} // game_object
} // engine