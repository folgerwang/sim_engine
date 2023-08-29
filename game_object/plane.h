#pragma once
#include "renderer/renderer.h"
#include "shape_base.h"

namespace engine {
namespace game_object {

class Plane : public ShapeBase {
public:
    Plane(const renderer::DeviceInfo& device_info);
    void draw(std::shared_ptr<renderer::CommandBuffer> cmd_buf);
};

} // game_object
} // engine