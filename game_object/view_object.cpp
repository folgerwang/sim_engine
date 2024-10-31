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

std::shared_ptr<er::DescriptorSetLayout> ViewObject::view_camera_desc_set_layout_;

std::shared_ptr<er::DescriptorSetLayout>
ViewObject::getViewCameraDescriptorSetLayout() {
    assert(view_camera_desc_set_layout_ != nullptr);
    return view_camera_desc_set_layout_;
}

void ViewObject::createViewCameraDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    if (view_camera_desc_set_layout_ == nullptr) {
        view_camera_desc_set_layout_ =
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

ViewObject::ViewObject(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<er::TextureInfo>& color_buffer,
    const std::shared_ptr<er::TextureInfo>& depth_buffer) :
        device_(device),
        descriptor_pool_(descriptor_pool),
        color_buffer_(color_buffer),
        depth_buffer_(depth_buffer) {

    AllocRenderBuffers();

    clear_values_.resize(2);
    clear_values_[0].color =
    { 50.0f / 255.0f,
      50.0f / 255.0f,
      50.0f / 255.0f,
      1.0f };
    clear_values_[1].depth_stencil =
    { 1.0f,
      0 };

    render_pass_ =
        er::helper::createRenderPass(
            device_,
            hdr_format_,
            depth_format_,
            std::source_location::current(),
            true);
    blend_render_pass_ =
        er::helper::createRenderPass(
            device_,
            hdr_format_,
            depth_format_,
            std::source_location::current(),
            false);

    std::vector<std::shared_ptr<er::ImageView>> attachments(2);
    attachments[0] = color_buffer_->view;
    attachments[1] = depth_buffer_->view;

    frame_buffer_ =
        device_->createFrameBuffer(
            render_pass_,
            attachments,
            buffer_size_,
            std::source_location::current());

    assert(blend_render_pass_);
    blend_frame_buffer_ =
        device_->createFrameBuffer(
            blend_render_pass_,
            attachments,
            buffer_size_,
            std::source_location::current());

    view_camera_params_.init_camera_pos = glm::vec3(0, 5.0f, 0);
    view_camera_params_.init_camera_dir = glm::vec3(0.0f, -1.0f, 0.0f);
    view_camera_params_.init_camera_up = glm::vec3(1, 0, 0);
    view_camera_params_.camera_speed = 0.01f;

    view_camera_ =
        std::make_shared<ego::ViewCamera>(
            device,
            descriptor_pool);

    view_camera_->initViewCameraBuffer(device_);

    assert(view_camera_desc_set_layout_ != nullptr);
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

void ViewObject::AllocRenderBuffers() {
    if (color_buffer_ == nullptr) {
        color_buffer_ = std::make_shared<er::TextureInfo>();
        er::Helper::create2DTextureImage(
            device_,
            hdr_format_,
            buffer_size_,
            *color_buffer_,
            SET_4_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT, COLOR_ATTACHMENT_BIT, TRANSFER_SRC_BIT),
            er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            std::source_location::current());
    }

    if (make_color_buffer_copy_) {
        color_buffer_copy_ = std::make_shared<er::TextureInfo>();
        er::Helper::create2DTextureImage(
            device_,
            hdr_format_,
            buffer_size_,
            *color_buffer_copy_,
            SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT),
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    if (depth_buffer_ == nullptr) {
        depth_buffer_ = std::make_shared<er::TextureInfo>();
        er::Helper::createDepthResources(
            device_,
            depth_format_,
            buffer_size_,
            *depth_buffer_,
            std::source_location::current());
    }

    if (make_depth_buffer_copy_) {
        depth_buffer_copy_ = std::make_shared<er::TextureInfo>();
        er::Helper::create2DTextureImage(
            device_,
            er::Format::D32_SFLOAT,
            buffer_size_,
            *depth_buffer_copy_,
            SET_3_FLAG_BITS(ImageUsage, SAMPLED_BIT, DEPTH_STENCIL_ATTACHMENT_BIT, TRANSFER_DST_BIT),
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }
}

void ViewObject::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf) {

}

void ViewObject::resize(const glm::uvec2& new_buffer_size) {
    if (buffer_size_ == new_buffer_size) {
        return;
    }

    buffer_size_ = new_buffer_size;

    if (color_buffer_) {
        color_buffer_->destroy(device_);
        color_buffer_.reset();
    }

    if (color_buffer_copy_) {
        color_buffer_copy_->destroy(device_);
        color_buffer_copy_.reset();
    }

    if (depth_buffer_) {
        depth_buffer_->destroy(device_);
        depth_buffer_.reset();
    }

    if (depth_buffer_copy_) {
        depth_buffer_copy_->destroy(device_);
        depth_buffer_copy_.reset();
    }

    AllocRenderBuffers();
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