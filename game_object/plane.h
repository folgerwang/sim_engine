#pragma once
#include "renderer/renderer.h"
#include "shape_base.h"

namespace engine {
namespace game_object {

class Plane : public ShapeBase {
public:
    Plane(
        const std::shared_ptr<renderer::Device>& device,
        uint32_t split_num_x,
        uint32_t split_num_y);
    void draw(std::shared_ptr<renderer::CommandBuffer> cmd_buf);
};

} // game_object
} // engine