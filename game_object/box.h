#pragma once
#include "shape_base.h"
#include "plane.h"

namespace engine {
namespace game_object {

class Box : public ShapeBase {
    std::array<std::shared_ptr<Plane>, 6> planes_;

public:
    Box(
        const std::shared_ptr<renderer::Device>& device,
        std::shared_ptr<std::array<glm::vec3, 8>> v,
        uint32_t split_num_x,
        uint32_t split_num_y,
        uint32_t split_num_z,
        const std::source_location& src_location);
    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);
    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine