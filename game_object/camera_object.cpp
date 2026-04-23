#include <string>
#include <sstream>
#include <unordered_map>
#include <array>
#include <cmath>
#include <algorithm>
#include "camera_object.h"
#include "glm/gtc/matrix_transform.hpp"
#include "terrain.h"
#include "helper/engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {
static std::shared_ptr<er::DescriptorSetLayout>
createViewCameraDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(1);
    bindings[0].binding = VIEW_CAMERA_BUFFER_INDEX;
    bindings[0].descriptor_count = 1;
    bindings[0].descriptor_type = er::DescriptorType::STORAGE_BUFFER;
    bindings[0].stage_flags =
        SET_5_FLAG_BITS(
            ShaderStage,
            VERTEX_BIT,
            MESH_BIT_EXT,
            FRAGMENT_BIT,
            GEOMETRY_BIT,
            COMPUTE_BIT);
    bindings[0].immutable_samplers = nullptr; // Optional

    return device->createDescriptorSetLayout(bindings);
}

} // 

namespace game_object {

std::shared_ptr<er::DescriptorSetLayout>
    CameraObject::s_view_camera_desc_set_layout_;

std::shared_ptr<er::DescriptorSetLayout>
CameraObject::getViewCameraDescriptorSetLayout() {
    assert(CameraObject::s_view_camera_desc_set_layout_ != nullptr);
    return CameraObject::s_view_camera_desc_set_layout_;
}

void CameraObject::createViewCameraDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    if (CameraObject::s_view_camera_desc_set_layout_ == nullptr) {
        CameraObject::s_view_camera_desc_set_layout_ =
            createViewCameraDescSetLayout(device);
    }
}

glm::vec3 getDirectionByYawAndPitch(float yaw, float pitch) {
    glm::vec3 direction;
    direction.x = cos(radians(-yaw)) * cos(radians(pitch));
    direction.y = sin(radians(pitch));
    direction.z = sin(radians(-yaw)) * cos(radians(pitch));
    return normalize(direction);
}

CameraObject::CameraObject(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    bool is_ortho) :
        m_device_(device),
        m_descriptor_pool_(descriptor_pool),
        m_is_ortho_(is_ortho){

    m_view_camera_params_.init_camera_pos = glm::vec3(0, 5.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(1, 0, 0);
    m_view_camera_params_.camera_speed = 0.01f;
    m_view_camera_params_.yaw = 0.0f;
    m_view_camera_params_.pitch = -90.0f;
    m_view_camera_params_.z_near = 0.1f;
    m_view_camera_params_.z_far = 4000.0f;

    m_view_camera_ =
        std::make_shared<ego::ViewCamera>(
            device,
            descriptor_pool,
            m_view_camera_params_,
            m_is_ortho_);

    m_view_camera_->initViewCameraBuffer(
        m_device_,
        m_view_camera_params_);

    assert(CameraObject::s_view_camera_desc_set_layout_ != nullptr);
    m_view_camera_desc_set_ =
        m_device_->createDescriptorSets(
            m_descriptor_pool_,
            CameraObject::s_view_camera_desc_set_layout_, 1)[0];
    er::WriteDescriptorList buffer_descs;
    buffer_descs.reserve(1);
    er::Helper::addOneBuffer(
        buffer_descs,
        m_view_camera_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        VIEW_CAMERA_BUFFER_INDEX,
        m_view_camera_->getViewCameraBuffer()->buffer,
        sizeof(glsl::ViewCameraInfo));
    m_device_->updateDescriptorSets(buffer_descs);
}

void CameraObject::recreateDescriptorSet() {
    assert(s_view_camera_desc_set_layout_ != nullptr);
    m_view_camera_desc_set_ =
        m_device_->createDescriptorSets(
            m_descriptor_pool_,
            s_view_camera_desc_set_layout_, 1)[0];
    er::WriteDescriptorList buffer_descs;
    buffer_descs.reserve(1);
    er::Helper::addOneBuffer(
        buffer_descs,
        m_view_camera_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        VIEW_CAMERA_BUFFER_INDEX,
        m_view_camera_->getViewCameraBuffer()->buffer,
        sizeof(glsl::ViewCameraInfo));
    m_device_->updateDescriptorSets(buffer_descs);
}

void CameraObject::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const uint32_t& dbuf_idx,
    const int& input_key,
    const int& frame_count,
    const float& delta_t,
    const glm::vec2& last_mouse_pos,
    const float& mouse_wheel_offset,
    const bool& camera_rot_update) {

    if (!m_is_ortho_) {
        m_view_camera_params_.camera_follow_dist = 5.0f;
        m_view_camera_params_.key = input_key;
        m_view_camera_params_.frame_count = frame_count;
        m_view_camera_params_.delta_t = delta_t;
        m_view_camera_params_.mouse_pos = last_mouse_pos;
        m_view_camera_params_.sensitivity = 0.2f;
        m_view_camera_params_.num_game_objs = 1;// static_cast<int32_t>(m_drawable_objects_.size());
        m_view_camera_params_.game_obj_idx = 0;
        m_view_camera_params_.camera_rot_update = camera_rot_update ? 1 : 0;
        m_view_camera_params_.mouse_wheel_offset = mouse_wheel_offset;
    }

    m_view_camera_->updateViewCameraInfo(
        m_view_camera_params_,
        nullptr);
}

void CameraObject::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const glm::vec3& camera_pos) {

    std::shared_ptr<glm::vec3> input_camera_pos =
        std::make_shared<glm::vec3>(camera_pos);
    m_view_camera_params_.init_camera_pos = camera_pos;
    m_view_camera_->updateViewCameraInfo(
        m_view_camera_params_,
        input_camera_pos);
}

void CameraObject::createCameraDescSetWithTerrain(
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    m_view_camera_->createViewCameraUpdateDescSet(
        m_device_,
        m_descriptor_pool_,
        texture_sampler,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        game_objects_buffer);
}

void CameraObject::readGpuCameraInfo() {
    m_view_camera_->readGpuCameraInfo(m_device_);
}

void CameraObject::destroy(const std::shared_ptr<renderer::Device>& device) {
    // NOTE: s_view_camera_desc_set_layout_ is a shared static — it must be
    // destroyed exactly once via destroyStaticMembers(), not per-instance
    // (otherwise double-free triggers "Couldn't find VkDescriptorSetLayout"
    // validation errors when both main and shadow cameras are torn down).
    m_view_camera_->destroy(m_device_);
};

void CameraObject::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    if (s_view_camera_desc_set_layout_) {
        device->destroyDescriptorSetLayout(s_view_camera_desc_set_layout_);
        s_view_camera_desc_set_layout_ = nullptr;
    }
}

ObjectViewCameraObject::ObjectViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        float fov,
        float aspect) :
    CameraObject(device, descriptor_pool, false){

    m_view_camera_params_.world_min = ego::TileObject::getWorldMin();
    m_view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    m_view_camera_params_.init_camera_pos = glm::vec3(0, 100.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(1, 0, 0);
    m_view_camera_params_.fov = fov;
    m_view_camera_params_.aspect = aspect;
    m_view_camera_params_.yaw = -125.800003f;
    m_view_camera_params_.pitch = 0.599997461f;
    m_view_camera_params_.init_camera_pos =
        glm::vec3(-1.70988739f, 2.48692441f, -13.6786499f);

    const float s_camera_speed = 0.1f;
    m_view_camera_params_.camera_speed = s_camera_speed;

    m_view_camera_params_.init_camera_dir =
        normalize(ego::getDirectionByYawAndPitch(
            m_view_camera_params_.yaw,
            m_view_camera_params_.pitch));
    m_view_camera_params_.init_camera_up =
        abs(m_view_camera_params_.init_camera_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);
}

TerrainViewCameraObject::TerrainViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        float fov,
        float aspect) :
    CameraObject(device, descriptor_pool, false) {

    m_view_camera_params_.world_min = ego::TileObject::getWorldMin();
    m_view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    m_view_camera_params_.init_camera_pos = glm::vec3(0, 500.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(1.0f, 0.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(0, 1, 0);
    m_view_camera_params_.fov = fov;
    m_view_camera_params_.aspect = aspect;

    m_view_camera_params_.init_camera_dir =
        normalize(ego::getDirectionByYawAndPitch(
            m_view_camera_params_.yaw,
            m_view_camera_params_.pitch));
    m_view_camera_params_.init_camera_up =
        abs(m_view_camera_params_.init_camera_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);
}

ShadowViewCameraObject::ShadowViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
        const glm::vec3& light_dir) :
    CameraObject(device, descriptor_pool, true) {

    m_view_camera_params_.world_min = ego::TileObject::getWorldMin();
    m_view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    m_view_camera_params_.init_camera_pos = glm::vec3(0, 0, 0);
    m_view_camera_params_.init_camera_dir = light_dir;
    m_view_camera_params_.init_camera_up = glm::vec3(1.0f, 0, 0);
    m_view_camera_params_.fov = 0.0f;
    m_view_camera_params_.aspect = 1.0f;
}

void ShadowViewCameraObject::computeCascadeMatrices(
    const glm::mat4& main_view,
    const glm::mat4& main_proj,
    const glm::vec4& cascade_far_vs,
    float z_near_vs,
    std::array<glm::mat4, 4>& out_vps) {

    // Derive tan(half-fov) and aspect ratio from the perspective matrix.
    // GLM stores proj[col][row]; proj[1][1] was negated for Vulkan y-flip.
    const float tan_half_fov_y = std::abs(1.0f / main_proj[1][1]);
    const float aspect = std::abs(main_proj[1][1]) / std::abs(main_proj[0][0]);

    // Stable light-space axes.
    const glm::vec3 light_dir = glm::normalize(m_view_camera_params_.init_camera_dir);
    glm::vec3 up = (std::abs(light_dir.y) < 0.99f)
        ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    const glm::vec3 light_right = glm::normalize(glm::cross(light_dir, up));
    up = glm::normalize(glm::cross(light_right, light_dir));

    const glm::mat4 inv_main_view = glm::inverse(main_view);

    for (int k = 0; k < 4; ++k) {
        const float cn = (k == 0) ? z_near_vs : cascade_far_vs[k - 1];
        const float cf = cascade_far_vs[k];

        // 8 view-space frustum corners of the cascade sub-frustum.
        // The main camera looks along -Z in view space.
        const float nh = cn * tan_half_fov_y, nw = nh * aspect;
        const float fh = cf * tan_half_fov_y, fw = fh * aspect;

        glm::vec3 vs_corners[8] = {
            { -nw,  nh, -cn }, {  nw,  nh, -cn },
            { -nw, -nh, -cn }, {  nw, -nh, -cn },
            { -fw,  fh, -cf }, {  fw,  fh, -cf },
            { -fw, -fh, -cf }, {  fw, -fh, -cf },
        };

        // World-space centre of the cascade frustum.
        glm::vec3 ws_centre(0.0f);
        for (int i = 0; i < 8; ++i)
            ws_centre += glm::vec3(inv_main_view * glm::vec4(vs_corners[i], 1.0f));
        ws_centre /= 8.0f;

        // Position the light eye far enough behind the centre so all scene
        // geometry (shadow casters) is in front of the camera.
        // We use a 200 m pull-back which is generous for typical scenes.
        constexpr float kPullBack = 200.0f;
        const glm::vec3 light_eye = ws_centre - light_dir * kPullBack;
        const glm::mat4 lv = glm::lookAt(light_eye, ws_centre, up);

        // Compute the tight AABB of the 8 corners in this light view space.
        glm::vec3 ls_min( 1e30f);
        glm::vec3 ls_max(-1e30f);
        for (int i = 0; i < 8; ++i) {
            glm::vec4 ws = inv_main_view * glm::vec4(vs_corners[i], 1.0f);
            glm::vec3 ls = glm::vec3(lv * ws);
            ls_min = glm::min(ls_min, ls);
            ls_max = glm::max(ls_max, ls);
        }

        // Small padding to avoid shadow acne at cascade boundaries.
        constexpr float kPad = 1.0f;
        ls_min.x -= kPad;  ls_max.x += kPad;
        ls_min.y -= kPad;  ls_max.y += kPad;

        // GLM ortho: near/far are positive distances from the camera along -Z.
        // ls.z for scene corners in front of the light eye is negative,
        // so near = -ls_max.z  (closest corners have z closest to 0),
        //    far  = -ls_min.z  (farthest corners have most-negative z).
        // Extend the far plane a bit to capture shadow casters just outside
        // the camera frustum (e.g., objects right behind the camera).
        const float ortho_near = std::max(-ls_max.z, 0.1f);
        const float ortho_far  = std::max(-ls_min.z + 10.0f, ortho_near + 1.0f);

        // Build orthographic projection.  No y-flip: the existing single-cascade
        // shadow code uses glm::ortho without flipping, and both the depth write
        // (base_depthonly.vert) and the lookup (base.frag) use the same matrix.
        const glm::mat4 lp = glm::ortho(
            ls_min.x, ls_max.x,
            ls_min.y, ls_max.y,
            ortho_near, ortho_far);

        out_vps[k] = lp * lv;
    }
}

void ShadowViewCameraObject::initCascadeDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) {

    assert(CameraObject::s_view_camera_desc_set_layout_ != nullptr);

    // Snapshot base camera info (position, direction, etc.) to pre-populate
    // each cascade buffer.  The view_proj field will be overwritten per-cascade.
    const glsl::ViewCameraInfo base_info = m_view_camera_->getCameraInfo();

    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        // Allocate a host-coherent storage buffer, one per cascade.
        m_cascade_bufs_[k] = std::make_shared<er::BufferInfo>();
        device->createBuffer(
            sizeof(glsl::ViewCameraInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_3_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_CACHED_BIT, HOST_COHERENT_BIT),
            0,
            m_cascade_bufs_[k]->buffer,
            m_cascade_bufs_[k]->memory,
            std::source_location::current());

        // Write initial data so the buffer is valid before the first draw.
        device->updateBufferMemory(
            m_cascade_bufs_[k]->memory,
            sizeof(glsl::ViewCameraInfo),
            &base_info);

        // Allocate a descriptor set pointing to this cascade's buffer.
        m_cascade_desc_sets_[k] =
            device->createDescriptorSets(
                descriptor_pool,
                CameraObject::s_view_camera_desc_set_layout_, 1)[0];

        er::WriteDescriptorList buffer_descs;
        buffer_descs.reserve(1);
        er::Helper::addOneBuffer(
            buffer_descs,
            m_cascade_desc_sets_[k],
            er::DescriptorType::STORAGE_BUFFER,
            VIEW_CAMERA_BUFFER_INDEX,
            m_cascade_bufs_[k]->buffer,
            sizeof(glsl::ViewCameraInfo));
        device->updateDescriptorSets(buffer_descs);
    }
}

void ShadowViewCameraObject::recreateCascadeDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) {

    assert(CameraObject::s_view_camera_desc_set_layout_ != nullptr);

    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (!m_cascade_bufs_[k]) continue;   // not initialised yet

        m_cascade_desc_sets_[k] =
            device->createDescriptorSets(
                descriptor_pool,
                CameraObject::s_view_camera_desc_set_layout_, 1)[0];

        er::WriteDescriptorList buffer_descs;
        buffer_descs.reserve(1);
        er::Helper::addOneBuffer(
            buffer_descs,
            m_cascade_desc_sets_[k],
            er::DescriptorType::STORAGE_BUFFER,
            VIEW_CAMERA_BUFFER_INDEX,
            m_cascade_bufs_[k]->buffer,
            sizeof(glsl::ViewCameraInfo));
        device->updateDescriptorSets(buffer_descs);
    }
}

void ShadowViewCameraObject::updateCascadeBuffer(
    int k, const glm::mat4& view_proj) {

    assert(k >= 0 && k < CSM_CASCADE_COUNT);
    assert(m_cascade_bufs_[k] != nullptr);

    // Copy current base camera info, override view_proj for this cascade,
    // and write to the cascade-specific host-coherent buffer.
    glsl::ViewCameraInfo cam_info = m_view_camera_->getCameraInfo();
    cam_info.view_proj = view_proj;

    m_device_->updateBufferMemory(
        m_cascade_bufs_[k]->memory,
        sizeof(glsl::ViewCameraInfo),
        &cam_info);
}

std::shared_ptr<er::DescriptorSet>
ShadowViewCameraObject::getViewCameraDescriptorSet() const {
    if (m_current_cascade_ >= 0 && m_current_cascade_ < CSM_CASCADE_COUNT
        && m_cascade_desc_sets_[m_current_cascade_]) {
        return m_cascade_desc_sets_[m_current_cascade_];
    }
    return m_view_camera_desc_set_;
}

void ShadowViewCameraObject::destroy(
    const std::shared_ptr<renderer::Device>& device) {

    // Destroy per-cascade buffers (descriptor sets are freed with the pool).
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (m_cascade_bufs_[k]) {
            m_cascade_bufs_[k]->destroy(device);
            m_cascade_bufs_[k] = nullptr;
        }
    }

    // Call base destroy.
    CameraObject::destroy(device);
}

void ShadowViewCameraObject::updateCameraForCascade(
    const glm::mat4& cascade_view_proj) {

    // Copy the current camera info, override view_proj with the cascade matrix,
    // and upload to the GPU buffer that base_depthonly.vert reads.
    glsl::ViewCameraInfo cam_info = m_view_camera_->getCameraInfo();
    cam_info.view_proj = cascade_view_proj;

    m_device_->updateBufferMemory(
        m_view_camera_->getViewCameraBuffer()->memory,
        sizeof(glsl::ViewCameraInfo),
        &cam_info);
}

} // game_object
} // engine