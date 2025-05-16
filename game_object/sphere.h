#pragma once
#include "shape_base.h"

namespace engine {
namespace game_object {

class Sphere : public ShapeBase {
public:
    Sphere(
        const std::shared_ptr<renderer::Device>& device,
        float radius,
        uint32_t subdivisions,
        const std::source_location& src_location);
    void draw(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const std::vector<renderer::Viewport>& viewports,
        const std::vector<renderer::Scissor>& scissors);
};

} // game_object
} // engine
