#include "shape_base.h"

namespace engine {

namespace {
static std::shared_ptr<renderer::BufferInfo> createUnifiedMeshBuffer(
    const renderer::DeviceInfo& device_info,
    const renderer::BufferUsageFlags& usage,
    const uint64_t& size,
    const void* data) {
    auto v_buffer = std::make_shared<renderer::BufferInfo>();
    renderer::Helper::createBuffer(
        device_info,
        usage,
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        v_buffer->buffer,
        v_buffer->memory,
        size,
        data);

    return v_buffer;
}
} // namespace

namespace game_object {

void ShapeBase::destroy(const std::shared_ptr<renderer::Device>& device) {
    if (position_buffer_) {
        position_buffer_->destroy(device);
    }
    if (uv_buffer_) {
        uv_buffer_->destroy(device);
    }
    if (normal_buffer_) {
        normal_buffer_->destroy(device);
    }
    if (index_buffer_) {
        index_buffer_->destroy(device);
    }
}

} // game_object
} // engine