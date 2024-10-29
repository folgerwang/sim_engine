#include <string>
#include <sstream>
#include <unordered_map>
#include "view_object.h"
#include "engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {
static std::shared_ptr<er::DescriptorSetLayout>
createViewCameraDescriptorSetLayout(
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

glm::vec3 getDirectionByYawAndPitch(float yaw, float pitch) {
    glm::vec3 direction;
    direction.x = cos(radians(-yaw)) * cos(radians(pitch));
    direction.y = sin(radians(pitch));
    direction.z = sin(radians(-yaw)) * cos(radians(pitch));
    return normalize(direction);
}

ViewObject::ViewObject(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) :
    device_(device), descriptor_pool_(descriptor_pool) {

    view_camera_params_.init_camera_pos = glm::vec3(0, 5.0f, 0);
    view_camera_params_.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
    view_camera_params_.init_camera_up = glm::vec3(1, 0, 0);
    view_camera_params_.camera_speed = 0.01f;

    view_camera_ =
        std::make_shared<ego::ViewCamera>(
            device,
            descriptor_pool);

    view_camera_->initViewCameraBuffer(device_);

    view_camera_desc_set_layout_ =
        createViewCameraDescriptorSetLayout(device_);

    view_camera_desc_set_ =
        device_->createDescriptorSets(
            descriptor_pool_,
            view_camera_desc_set_layout_, 1)[0];
    er::WriteDescriptorList buffer_descs;
    buffer_descs.reserve(1);
    er::Helper::addOneBuffer(
        buffer_descs,
        view_camera_desc_set_,
        er::DescriptorType::STORAGE_BUFFER,
        VIEW_CAMERA_BUFFER_INDEX,
        view_camera_->getViewCameraBuffer()->buffer,
        sizeof(glsl::ViewCameraInfo));
    device_->updateDescriptorSets(buffer_descs);
}

void ViewObject::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf) {

}

void ViewObject::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    int dbuf_idx,
    float delta_t,
    float cur_time) {

}

void ViewObject::destroy(const std::shared_ptr<renderer::Device>& device) {
    device_->destroyDescriptorSetLayout(
        view_camera_desc_set_layout_);

};

} // game_object
} // engine