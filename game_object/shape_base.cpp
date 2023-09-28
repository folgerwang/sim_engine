#include "shape_base.h"

namespace engine {
namespace {

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
    if (tangent_buffer_) {
        tangent_buffer_->destroy(device);
    }
    if (index_buffer_) {
        index_buffer_->destroy(device);
    }
}

} // game_object
} // engine