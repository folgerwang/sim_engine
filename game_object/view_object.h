#pragma once
#include "renderer/renderer.h"
#include "game_object/camera.h"
#include "patch.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace engine {
namespace game_object {

class ViewObject {
    std::shared_ptr<ego::ViewCamera> view_camera_;
    const std::shared_ptr<renderer::Device>& device_;
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool_;

public:
    ViewObject(const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<er::DescriptorPool>& descriptor_pool);

    void createCameraDescSetWithTerrain(
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::BufferInfo& game_objects_buffer);

    void updateCamera(
        std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    void draw(std::shared_ptr<renderer::CommandBuffer> cmd_buf);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

} // game_object
} // engine