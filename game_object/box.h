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
        uint32_t split_num_z);
    void draw(std::shared_ptr<renderer::CommandBuffer> cmd_buf);
};

} // game_object
} // engine