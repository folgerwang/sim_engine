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
std::shared_ptr<renderer::DescriptorSetLayout> ViewCamera::update_view_camera_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> ViewCamera::update_view_camera_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> ViewCamera::update_view_camera_pipeline_;

ViewCamera::ViewCamera(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) {
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
        update_view_camera_desc_set_[soil_water] =
            device->createDescriptorSets(
                descriptor_pool, update_view_camera_desc_set_layout_, 1)[0];

        assert(view_camera_buffer_);
        auto write_descs = addViewCameraInfoBuffer(
            update_view_camera_desc_set_[soil_water],
            texture_sampler,
            game_objects_buffer,
            *view_camera_buffer_,
            rock_layer,
            soil_water == 0 ? soil_water_layer_0 : soil_water_layer_1);

        device->updateDescriptorSets(write_descs);
    }
}

void ViewCamera::initViewCameraBuffer(
    const std::shared_ptr<renderer::Device>& device) {
    if (!view_camera_buffer_) {
        view_camera_buffer_ = std::make_shared<renderer::BufferInfo>();
        device->createBuffer(
            sizeof(glsl::ViewCameraInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_CACHED_BIT),
            0,
            view_camera_buffer_->buffer,
            view_camera_buffer_->memory,
            std::source_location::current());
    }
}

void ViewCamera::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {

    if (update_view_camera_desc_set_layout_ == nullptr) {
        update_view_camera_desc_set_layout_ =
            createUpdateViewCameraDescriptorSetLayout(device);
    }

    createStaticMembers(device);
}

void ViewCamera::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {

    {
        if (update_view_camera_pipeline_layout_) {
            device->destroyPipelineLayout(update_view_camera_pipeline_layout_);
            update_view_camera_pipeline_layout_ = nullptr;
        }

        if (update_view_camera_pipeline_layout_ == nullptr) {
            update_view_camera_pipeline_layout_ =
                createViewCameraPipelineLayout(
                    device,
                    { update_view_camera_desc_set_layout_ });
        }
    }

    {
        if (update_view_camera_pipeline_) {
            device->destroyPipeline(update_view_camera_pipeline_);
            update_view_camera_pipeline_ = nullptr;
        }

        if (update_view_camera_pipeline_ == nullptr) {
            assert(update_view_camera_pipeline_layout_);
            update_view_camera_pipeline_ =
                renderer::helper::createComputePipeline(
                    device,
                    update_view_camera_pipeline_layout_,
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
    device->destroyDescriptorSetLayout(update_view_camera_desc_set_layout_);
    device->destroyPipelineLayout(update_view_camera_pipeline_layout_);
    device->destroyPipeline(update_view_camera_pipeline_);
}

void ViewCamera::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    view_camera_buffer_->destroy(device);
}

void ViewCamera::updateViewCameraBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glsl::ViewCameraParams& view_camera_params,
    int soil_water) {

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, update_view_camera_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        update_view_camera_pipeline_layout_,
        &view_camera_params,
        sizeof(view_camera_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        update_view_camera_pipeline_layout_,
        { update_view_camera_desc_set_[soil_water] });

    cmd_buf->dispatch(1, 1);

    cmd_buf->addBufferBarrier(
        view_camera_buffer_->buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        view_camera_buffer_->buffer->getSize());
}

glsl::ViewCameraInfo ViewCamera::readCameraInfo(
    const std::shared_ptr<renderer::Device>& device,
    const uint32_t& idx) {

    glsl::ViewCameraInfo camera_info;
    device->dumpBufferMemory(view_camera_buffer_->memory, sizeof(glsl::ViewCameraInfo), &camera_info);

    return camera_info;
}

void ViewCamera::update(
    const std::shared_ptr<renderer::Device>& device,
    const float& time) {
}

std::shared_ptr<renderer::BufferInfo> ViewCamera::getViewCameraBuffer() {
    return view_camera_buffer_;
}

} // game_object
} // engine