#include <string>
#include <sstream>
#include <unordered_map>
#include "view_object.h"
#include "engine_helper.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {

namespace game_object {

ViewObject::ViewObject(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool) :
    device_(device), descriptor_pool_(descriptor_pool) {

    view_camera_ =
        std::make_shared<ego::ViewCamera>(
            device,
            descriptor_pool);

    view_camera_->initViewCameraBuffer(device_);
}

void ViewObject::createCameraDescSetWithTerrain(
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::BufferInfo& game_objects_buffer) {
    view_camera_->createViewCameraUpdateDescSet(
        device_,
        descriptor_pool_,
        texture_sampler,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        game_objects_buffer);
}

void ViewObject::updateCamera(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf) {

}

void ViewObject::draw(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    int dbuf_idx,
    float delta_t,
    float cur_time) {

    // Draw all visible tiles
    for (auto& tile : visible_tiles_) {
        tile->draw(
            cmd_buf,
            desc_set_list,
            buffer_size_,
            dbuf_idx,
            delta_t,
            cur_time,
            base_pass_);

        if (base_pass_ && render_grass_) {
            tile->drawGrass(
                cmd_buf,
                desc_set_list,
                camera_pos_,
                buffer_size_,
                dbuf_idx,
                delta_t,
                cur_time);
        }
    }
}

void ViewObject::destroy(const std::shared_ptr<renderer::Device>& device) {
};

} // game_object
} // engine