#include <string>
#include <sstream>
#include <unordered_map>
#include "view_object.h"
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

ViewObject::ViewObject(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const std::shared_ptr<CameraObject>& camera_object,
    const std::shared_ptr<er::TextureInfo>& color_buffer,
    const std::shared_ptr<er::TextureInfo>& depth_buffer,
    const glm::uvec2& buffer_size/* = glm::uvec2(2560, 1440)*/,
    bool depth_only/* = false*/) :
        m_device_(device),
        m_descriptor_pool_(descriptor_pool),
        m_camera_object_(camera_object),
        m_color_buffer_(color_buffer),
        m_depth_buffer_(depth_buffer),
        m_buffer_size_(buffer_size),
        m_depth_only_(depth_only){

    AllocRenderBuffers(renderbuffer_formats);

    m_clear_values_.resize(2);
    m_clear_values_[0].color =
    { 50.0f / 255.0f,
      50.0f / 255.0f,
      50.0f / 255.0f,
      1.0f };
    m_clear_values_[1].depth_stencil =
    { 1.0f,
      0 };
}

void ViewObject::AllocRenderBuffers(
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats) {
    if (m_depth_only_) {
        m_color_buffer_ = nullptr;
        m_color_buffer_copy_ = nullptr;
    }
    else {
        if (m_color_buffer_ == nullptr) {
            m_color_buffer_ = std::make_shared<er::TextureInfo>();
            er::Helper::create2DTextureImage(
                m_device_,
                renderbuffer_formats.color_formats[0],
                m_buffer_size_,
                1,
                *m_color_buffer_,
                SET_4_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT, COLOR_ATTACHMENT_BIT, TRANSFER_SRC_BIT),
                er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                std::source_location::current());
        }

        if (m_make_color_buffer_copy_) {
            m_color_buffer_copy_ = std::make_shared<er::TextureInfo>();
            er::Helper::create2DTextureImage(
                m_device_,
                renderbuffer_formats.color_formats[0],
                m_buffer_size_,
                1,
                *m_color_buffer_copy_,
                SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT),
                er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                std::source_location::current());
        }
    }

    if (m_depth_buffer_ == nullptr) {
        m_depth_buffer_ = std::make_shared<er::TextureInfo>();
        er::Helper::createDepthResources(
            m_device_,
            renderbuffer_formats.depth_format,
            m_buffer_size_,
            *m_depth_buffer_,
            std::source_location::current());
    }

    if (m_make_depth_buffer_copy_) {
        m_depth_buffer_copy_ = std::make_shared<er::TextureInfo>();
        er::Helper::create2DTextureImage(
            m_device_,
            renderbuffer_formats.depth_format,
            m_buffer_size_,
            1,
            *m_depth_buffer_copy_,
            SET_3_FLAG_BITS(ImageUsage, SAMPLED_BIT, DEPTH_STENCIL_ATTACHMENT_BIT, TRANSFER_DST_BIT),
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }
}

void ViewObject::resize(
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats,
    const glm::uvec2& new_buffer_size) {
    if (m_buffer_size_ == new_buffer_size) {
        return;
    }

    m_buffer_size_ = new_buffer_size;

    if (m_color_buffer_) {
        m_color_buffer_->destroy(m_device_);
        m_color_buffer_.reset();
    }

    if (m_color_buffer_copy_) {
        m_color_buffer_copy_->destroy(m_device_);
        m_color_buffer_copy_.reset();
    }

    if (m_depth_buffer_) {
        m_depth_buffer_->destroy(m_device_);
        m_depth_buffer_.reset();
    }

    if (m_depth_buffer_copy_) {
        m_depth_buffer_copy_->destroy(m_device_);
        m_depth_buffer_copy_.reset();
    }

    AllocRenderBuffers(renderbuffer_formats);
}

void ViewObject::destroy(const std::shared_ptr<er::Device>& device) {
    if (m_color_buffer_)
        m_color_buffer_->destroy(m_device_);

    if (m_color_buffer_copy_)
        m_color_buffer_copy_->destroy(m_device_);

    if (m_depth_buffer_)
        m_depth_buffer_->destroy(m_device_);

    if (m_depth_buffer_copy_)
        m_depth_buffer_copy_->destroy(m_device_);
};

} // game_object
} // engine