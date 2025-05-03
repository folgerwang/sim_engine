#include <string>
#include <sstream>
#include <unordered_map>
#include "camera_object.h"
#include "terrain.h"
#include "engine_helper.h"
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
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) :
        m_device_(device),
        m_descriptor_pool_(descriptor_pool) {

    m_view_camera_params_.init_camera_pos = glm::vec3(0, 5.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(1, 0, 0);
    m_view_camera_params_.camera_speed = 0.01f;

    m_view_camera_ =
        std::make_shared<ego::ViewCamera>(
            device,
            descriptor_pool,
            m_view_camera_params_);

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

void CameraObject::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const uint32_t& dbuf_idx,
    const int& input_key,
    const int& frame_count,
    const float& delta_t,
    const glm::vec2& last_mouse_pos,
    const float& mouse_wheel_offset,
    const bool& camera_rot_update) {

    const float s_camera_speed = 2.0f;

    m_view_camera_params_.camera_speed = s_camera_speed;

    m_view_camera_params_.yaw = 0.0f;
    m_view_camera_params_.pitch = -90.0f;
    m_view_camera_params_.init_camera_dir =
        normalize(ego::getDirectionByYawAndPitch(
            m_view_camera_params_.yaw,
            m_view_camera_params_.pitch));
    m_view_camera_params_.init_camera_up =
        abs(m_view_camera_params_.init_camera_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);

    m_view_camera_params_.z_near = 0.1f;
    m_view_camera_params_.z_far = 40000.0f;
    m_view_camera_params_.camera_follow_dist = 5.0f;
    m_view_camera_params_.key = input_key;
    m_view_camera_params_.frame_count = frame_count;
    m_view_camera_params_.delta_t = delta_t;
    m_view_camera_params_.mouse_pos = last_mouse_pos;
    m_view_camera_params_.fov = glm::radians(45.0f);
    m_view_camera_params_.aspect = 16.0f / 9.0f;// m_buffer_size_.x / (float)m_buffer_size_.y;
    m_view_camera_params_.sensitivity = 0.2f;
    m_view_camera_params_.num_game_objs = 1;// static_cast<int32_t>(m_drawable_objects_.size());
    m_view_camera_params_.game_obj_idx = 0;
    m_view_camera_params_.camera_rot_update = camera_rot_update ? 1 : 0;
    m_view_camera_params_.mouse_wheel_offset = mouse_wheel_offset;

    m_view_camera_->updateViewCameraInfo(
        m_view_camera_params_);
}

#if 0
void TerrainSceneView::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const uint32_t& dbuf_idx,
    const int& input_key,
    const int& frame_count,
    const float& delta_t,
    const glm::vec2& last_mouse_pos,
    const float& mouse_wheel_offset,
    const bool& camera_rot_update) {

    const float s_camera_speed = 10.0f;

    m_view_camera_params_.camera_speed = s_camera_speed;

    m_view_camera_params_.yaw = 0.0f;
    m_view_camera_params_.pitch = -90.0f;
    m_view_camera_params_.init_camera_dir =
        normalize(ego::getDirectionByYawAndPitch(
            m_view_camera_params_.yaw,
            m_view_camera_params_.pitch));
    m_view_camera_params_.init_camera_up =
        abs(m_view_camera_params_.init_camera_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);

    m_view_camera_params_.z_near = 0.1f;
    m_view_camera_params_.z_far = 40000.0f;
    m_view_camera_params_.camera_follow_dist = 5.0f;
    m_view_camera_params_.key = input_key;
    m_view_camera_params_.frame_count = frame_count;
    m_view_camera_params_.delta_t = delta_t;
    m_view_camera_params_.mouse_pos = last_mouse_pos;
    m_view_camera_params_.fov = glm::radians(45.0f);
    m_view_camera_params_.aspect = m_buffer_size_.x / (float)m_buffer_size_.y;
    m_view_camera_params_.sensitivity = 0.2f;
    m_view_camera_params_.num_game_objs = static_cast<int32_t>(m_drawable_objects_.size());
    m_view_camera_params_.game_obj_idx = 0;
    m_view_camera_params_.camera_rot_update = camera_rot_update ? 1 : 0;
    m_view_camera_params_.mouse_wheel_offset = mouse_wheel_offset;

    m_view_camera_->updateViewCameraBuffer(
        cmd_buf,
        m_view_camera_params_,
        dbuf_idx);
}
#endif

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
    m_device_->destroyDescriptorSetLayout(
        CameraObject::s_view_camera_desc_set_layout_);

    m_view_camera_->destroy(m_device_);
};

ObjectViewCameraObject::ObjectViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool) :
    CameraObject(device, descriptor_pool){

    m_view_camera_params_.world_min = ego::TileObject::getWorldMin();
    m_view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    m_view_camera_params_.init_camera_pos = glm::vec3(0, 100.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(1, 0, 0);
}

TerrainViewCameraObject::TerrainViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool) :
    CameraObject(device, descriptor_pool) {

    m_view_camera_params_.world_min = ego::TileObject::getWorldMin();
    m_view_camera_params_.inv_world_range = 1.0f / ego::TileObject::getWorldRange();
    m_view_camera_params_.init_camera_pos = glm::vec3(0, 500.0f, 0);
    m_view_camera_params_.init_camera_dir = glm::vec3(1.0f, 0.0f, 0.0f);
    m_view_camera_params_.init_camera_up = glm::vec3(0, 1, 0);
}

ShadowViewCameraObject::ShadowViewCameraObject(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool) :
    CameraObject(device, descriptor_pool) {
}

} // game_object
} // engine