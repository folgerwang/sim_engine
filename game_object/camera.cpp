#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "engine_helper.h"
#include "game_object/camera.h"
#include "renderer/renderer_helper.h"

#include "tiny_gltf.h"
#include "tiny_mtx2.h"

constexpr float s_half_shadow_size = 256.0f;

namespace ego = engine::game_object;
namespace engine {

namespace {
renderer::WriteDescriptorList addViewCameraInfoBuffer(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::BufferInfo& game_objects_buffer,
    const renderer::BufferInfo& view_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(4);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        GAME_OBJECTS_BUFFER_INDEX,
        game_objects_buffer.buffer,
        game_objects_buffer.buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        CAMERA_OBJECT_BUFFER_INDEX,
        view_camera_buffer.buffer,
        view_camera_buffer.buffer->getSize());

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        texture_sampler,
        soil_water_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static std::shared_ptr<renderer::DescriptorSetLayout> createUpdateViewCameraDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(4);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        GAME_OBJECTS_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        CAMERA_OBJECT_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[2] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[3] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createViewCameraPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::ViewCameraParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

} // namespace

namespace game_object {

// static member definition.
std::shared_ptr<renderer::DescriptorSetLayout> ViewCamera::s_update_view_camera_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> ViewCamera::s_update_view_camera_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> ViewCamera::s_update_view_camera_pipeline_;

ViewCamera::ViewCamera(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const glsl::ViewCameraParams& view_camera_params) :
    m_device_(device),
    m_descriptor_pool_(descriptor_pool) {

    m_camera_info_.position = view_camera_params.init_camera_pos;
    m_camera_info_.mouse_pos = view_camera_params.mouse_pos;
    m_camera_info_.yaw = view_camera_params.yaw;
    m_camera_info_.pitch = view_camera_params.pitch;
    m_camera_info_.camera_follow_dist = view_camera_params.camera_follow_dist;
    m_camera_info_.status |= 0x00000001;

    m_shadow_camera_info_.position = view_camera_params.init_camera_pos;
    m_shadow_camera_info_.mouse_pos = view_camera_params.mouse_pos;
    m_shadow_camera_info_.yaw = view_camera_params.yaw;
    m_shadow_camera_info_.pitch = view_camera_params.pitch;
    m_shadow_camera_info_.camera_follow_dist = view_camera_params.camera_follow_dist;
    m_shadow_camera_info_.status |= 0x00000001;
}

void ViewCamera::createViewCameraUpdateDescSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    // create a global ibl texture descriptor set.
    for (int soil_water = 0; soil_water < 2; soil_water++) {
        // game objects buffer update set.
        m_update_view_camera_desc_set_[soil_water] =
            device->createDescriptorSets(
                descriptor_pool, s_update_view_camera_desc_set_layout_, 1)[0];

        assert(m_view_camera_buffer_);
        auto write_descs = addViewCameraInfoBuffer(
            m_update_view_camera_desc_set_[soil_water],
            texture_sampler,
            game_objects_buffer,
            *m_view_camera_buffer_,
            rock_layer,
            soil_water == 0 ? soil_water_layer_0 : soil_water_layer_1);

        device->updateDescriptorSets(write_descs);
    }
}

void ViewCamera::initViewCameraBuffer(
    const std::shared_ptr<renderer::Device>& device,
    const glsl::ViewCameraParams& view_camera_params) {
    if (!m_view_camera_buffer_) {
        m_view_camera_buffer_ = std::make_shared<renderer::BufferInfo>();
        device->createBuffer(
            sizeof(glsl::ViewCameraInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_3_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_CACHED_BIT, HOST_COHERENT_BIT),
            0,
            m_view_camera_buffer_->buffer,
            m_view_camera_buffer_->memory,
            std::source_location::current());

        m_camera_info_.position = view_camera_params.init_camera_pos;
        m_camera_info_.mouse_pos = view_camera_params.mouse_pos;
        m_camera_info_.yaw = view_camera_params.yaw;
        m_camera_info_.pitch = view_camera_params.pitch;
        m_camera_info_.camera_follow_dist = view_camera_params.camera_follow_dist;
        m_camera_info_.status |= 0x00000001;
        device->updateBufferMemory(
            m_view_camera_buffer_->memory,
            sizeof(m_camera_info_),
            &m_camera_info_);
    }

    if (!m_direct_shadow_camera_buffer_) {
        m_direct_shadow_camera_buffer_ = std::make_shared<renderer::BufferInfo>();
        device->createBuffer(
            sizeof(glsl::ViewCameraInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_3_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_CACHED_BIT, HOST_COHERENT_BIT),
            0,
            m_direct_shadow_camera_buffer_->buffer,
            m_direct_shadow_camera_buffer_->memory,
            std::source_location::current());

        m_shadow_camera_info_.position = view_camera_params.init_camera_pos;
        m_shadow_camera_info_.yaw = view_camera_params.yaw;
        m_shadow_camera_info_.pitch = view_camera_params.pitch;
        m_shadow_camera_info_.camera_follow_dist = view_camera_params.camera_follow_dist;
        m_shadow_camera_info_.status |= 0x00000001;
        device->updateBufferMemory(
            m_direct_shadow_camera_buffer_->memory,
            sizeof(m_shadow_camera_info_),
            &m_shadow_camera_info_);
    }
}

void ViewCamera::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {

    if (s_update_view_camera_desc_set_layout_ == nullptr) {
        s_update_view_camera_desc_set_layout_ =
            createUpdateViewCameraDescriptorSetLayout(device);
    }

    createStaticMembers(device);
}

void ViewCamera::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {

    {
        if (s_update_view_camera_pipeline_layout_) {
            device->destroyPipelineLayout(s_update_view_camera_pipeline_layout_);
            s_update_view_camera_pipeline_layout_ = nullptr;
        }

        if (s_update_view_camera_pipeline_layout_ == nullptr) {
            s_update_view_camera_pipeline_layout_ =
                createViewCameraPipelineLayout(
                    device,
                    { s_update_view_camera_desc_set_layout_ });
        }
    }

    {
        if (s_update_view_camera_pipeline_) {
            device->destroyPipeline(s_update_view_camera_pipeline_);
            s_update_view_camera_pipeline_ = nullptr;
        }

        if (s_update_view_camera_pipeline_ == nullptr) {
            assert(s_update_view_camera_pipeline_layout_);
            s_update_view_camera_pipeline_ =
                renderer::helper::createComputePipeline(
                    device,
                    s_update_view_camera_pipeline_layout_,
                    "update_camera_comp.spv",
                    std::source_location::current());
        }
    }
}

void ViewCamera::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {

    createStaticMembers(device);
}

void ViewCamera::generateDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {

    createViewCameraUpdateDescSet(
        device,
        descriptor_pool,
        texture_sampler,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        game_objects_buffer);
}

void ViewCamera::destroyStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(s_update_view_camera_desc_set_layout_);
    device->destroyPipelineLayout(s_update_view_camera_pipeline_layout_);
    device->destroyPipeline(s_update_view_camera_pipeline_);
}

void ViewCamera::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    m_view_camera_buffer_->destroy(device);
    m_direct_shadow_camera_buffer_->destroy(device);
}

void ViewCamera::updateViewCameraBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glsl::ViewCameraParams& view_camera_params,
    int soil_water) {

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        s_update_view_camera_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        s_update_view_camera_pipeline_layout_,
        &view_camera_params,
        sizeof(view_camera_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        s_update_view_camera_pipeline_layout_,
        { m_update_view_camera_desc_set_[soil_water] });

    cmd_buf->dispatch(1, 1);

    cmd_buf->addBufferBarrier(
        m_view_camera_buffer_->buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        m_view_camera_buffer_->buffer->getSize());
}

void ViewCamera::updateViewCameraInfo(
    const glsl::ViewCameraParams& view_camera_params) {

    auto& camera_info = m_camera_info_;

    if (camera_info.status == 0 || view_camera_params.frame_count == 0) {
        camera_info.position = view_camera_params.init_camera_pos;
        camera_info.mouse_pos = view_camera_params.mouse_pos;
        camera_info.yaw = view_camera_params.yaw;
        camera_info.pitch = view_camera_params.pitch;
        camera_info.camera_follow_dist = view_camera_params.camera_follow_dist;
        camera_info.status |= 0x00000001;
    }
    vec2 mouse_offset = view_camera_params.mouse_pos - camera_info.mouse_pos;
    mouse_offset *= view_camera_params.sensitivity;

    if (view_camera_params.camera_rot_update != 0) {
        camera_info.yaw += mouse_offset.x;
        camera_info.pitch = clamp(camera_info.pitch + mouse_offset.y, -89.0f, 89.0f);
    }

    camera_info.facing_dir =
        getDirectionByYawAndPitch(camera_info.yaw, camera_info.pitch);
    camera_info.up_vector =
        abs(camera_info.facing_dir.y) < 0.99f ?
        vec3(0, 1, 0) :
        vec3(1, 0, 0);
    vec3 camera_right =
        normalize(cross(camera_info.facing_dir, camera_info.up_vector));

    auto move_forward_vec =
        view_camera_params.camera_speed * camera_info.facing_dir;

    auto move_right_vec =
        view_camera_params.camera_speed * camera_right;
    
    if (view_camera_params.key == GLFW_KEY_W)
        camera_info.position += move_forward_vec;
    if (view_camera_params.key == GLFW_KEY_S)
        camera_info.position -= move_forward_vec;
    if (view_camera_params.key == GLFW_KEY_A)
        camera_info.position -= move_right_vec;
    if (view_camera_params.key == GLFW_KEY_D)
        camera_info.position += move_right_vec;

    auto eye_pos = camera_info.position;
    auto target_pos = eye_pos + camera_info.facing_dir;
    auto up_dir = camera_info.up_vector;

    camera_info.position = eye_pos;

    camera_info.view =
        lookAt(eye_pos, target_pos, up_dir);
    camera_info.proj =
        perspective(
            view_camera_params.fov,
            view_camera_params.aspect,
            view_camera_params.z_near,
            view_camera_params.z_far);
    camera_info.proj[1].y *= -1.0f;
    camera_info.view_proj = camera_info.proj * camera_info.view;
    camera_info.inv_view_proj = inverse(camera_info.view_proj);
    camera_info.inv_view = inverse(camera_info.view);
    camera_info.inv_proj = inverse(camera_info.proj);

    mat4 view_relative = lookAt(vec3(0), target_pos - eye_pos, up_dir);
    camera_info.inv_view_proj_relative = inverse(camera_info.proj * view_relative);
    camera_info.depth_params = vec4(
        camera_info.proj[2].z,
        camera_info.proj[3].z,
        1.0f / camera_info.proj[0].x,
        1.0f / camera_info.proj[1].y);
    camera_info.mouse_pos = view_camera_params.mouse_pos;

    m_device_->updateBufferMemory(
        m_view_camera_buffer_->memory,
        sizeof(camera_info),
        &camera_info);
}

void ViewCamera::updateShadowViewInfo(
    const glsl::ViewCameraParams& view_camera_params,
    const vec3& light_dir) {

    auto& camera_info = m_shadow_camera_info_;

    camera_info.facing_dir =
        light_dir;
    camera_info.up_vector =
        vec3(1, 0, 0);
    vec3 camera_right =
        normalize(cross(camera_info.facing_dir, camera_info.up_vector));
    camera_info.up_vector =
        normalize(cross(camera_right, camera_info.facing_dir));

    const auto& eye_pos = m_camera_info_.position;
    const auto target_pos = eye_pos + camera_info.facing_dir;
    const auto& up_dir = camera_info.up_vector;

    camera_info.position = eye_pos;

    camera_info.view =
        lookAt(eye_pos, target_pos, up_dir);
    camera_info.proj =
        ortho(
            -s_half_shadow_size,
            s_half_shadow_size,
            -s_half_shadow_size,
            s_half_shadow_size,
            view_camera_params.z_near,
            view_camera_params.z_far);

    camera_info.proj[1].y *= -1.0f;
    camera_info.view_proj = camera_info.proj * camera_info.view;
    camera_info.inv_view_proj = inverse(camera_info.view_proj);
    camera_info.inv_view = inverse(camera_info.view);
    camera_info.inv_proj = inverse(camera_info.proj);

    mat4 view_relative = lookAt(vec3(0), target_pos - eye_pos, up_dir);
    camera_info.inv_view_proj_relative = inverse(camera_info.proj * view_relative);
    camera_info.depth_params = vec4(
        camera_info.proj[2].z,
        camera_info.proj[3].z,
        1.0f / camera_info.proj[0].x,
        1.0f / camera_info.proj[1].y);
    camera_info.mouse_pos = view_camera_params.mouse_pos;

    m_device_->updateBufferMemory(
        m_direct_shadow_camera_buffer_->memory,
        sizeof(camera_info),
        &camera_info);
}

void ViewCamera::readGpuCameraInfo(
    const std::shared_ptr<renderer::Device>& device) {

    device->dumpBufferMemory(
        m_view_camera_buffer_->memory,
        sizeof(glsl::ViewCameraInfo),
        &m_camera_info_);
}

void ViewCamera::update(
    const std::shared_ptr<renderer::Device>& device,
    const float& time) {
}

} // game_object
} // engine